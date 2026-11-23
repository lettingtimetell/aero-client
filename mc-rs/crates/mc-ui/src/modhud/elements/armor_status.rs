//! Armor Status (`armor_status`): helmet, chestplate, leggings, boots and the held item, each
//! as its vanilla GUI icon with the vanilla durability bar and stack count (the hotbar's own
//! `renderItemAndEffectIntoGUI` + `renderItemOverlayIntoGUI` port), plus a durability number
//! for damageable items.
//!
//! * Empty slots are skipped, so the element shrinks; with everything empty it is not drawn.
//! * A held **bow** shows the arrows you carry (`HudData::arrow_count`) where the stack count
//!   goes, which is what SkyWars players want to know.
//! * The text column has a fixed width per durability mode, so the element does not change size
//!   as numbers change.
//! * Anchored on the right third of the screen it mirrors: icons at the screen edge, text inside.
//! * Text is rebuilt only when a slot's item, damage or (bow) arrow count changes.

use std::fmt::Write as _;

use mc_mods::{HudAnchor, ModSettings};

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement};
use crate::screens::container::render::{
    draw_count_text, draw_count_text_with_override, draw_durability_bar, draw_item_icon_tinted,
};
use crate::screens::container::stack_ext::{self, ItemStack};
use crate::widget::{DrawList, UiTexture};

/// Vanilla item ids this element special-cases.
const BOW: i16 = 261;

/// Icon size plus one pixel between rows.
const ROW_PITCH: f32 = 17.0;
/// Icon size.
const ICON: f32 = 16.0;
/// Gap between an icon and its text.
const TEXT_GAP: f32 = 2.0;
/// Gap between entries in the horizontal layout.
const ENTRY_GAP: f32 = 4.0;

/// The `durability` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Durability {
    /// `363`: uses left.
    #[default]
    Value,
    /// `87%`.
    Percent,
    /// `363/363`.
    ValueMax,
    /// Icons only.
    Off,
}

impl Durability {
    /// The widest text this mode can produce, for the fixed text column.
    fn widest_sample(self) -> &'static str {
        match self {
            Durability::Value => "0000",
            Durability::Percent => "100%",
            Durability::ValueMax => "0000/0000",
            Durability::Off => "",
        }
    }
}

/// The text colour for `remaining` of `max` uses: white when new, then green, yellow, orange,
/// red, and dark red below 10 %.
#[must_use]
fn durability_color(remaining: i32, max: i32) -> u32 {
    let (remaining, max) = (i64::from(remaining), i64::from(max.max(1)));
    let at_least = |percent: i64| remaining * 100 >= percent * max;
    if at_least(80) {
        0xFF_FFFF
    } else if at_least(60) {
        0x55_FF55
    } else if at_least(40) {
        0xFF_FF55
    } else if at_least(25) {
        0xFF_AA00
    } else if at_least(10) {
        0xFF_5555
    } else {
        0xAA_0000
    }
}

/// What a row's text was built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowKey {
    item: i16,
    damage: i16,
    /// The arrow count, for a held bow only.
    arrows: Option<u32>,
}

/// One slot's cached text and this frame's placement.
#[derive(Debug, Default)]
struct Row {
    key: Option<RowKey>,
    /// The durability text, empty for items that do not wear out.
    text: String,
    text_width: f32,
    /// `0xRRGGBB` of the text (alpha comes from the text colour option).
    rgb: u32,
    /// The held bow's arrow count, drawn where vanilla draws a stack count; empty otherwise.
    count: String,
    /// Drawn this frame.
    present: bool,
    /// Icon position.
    x: f32,
    y: f32,
    /// Text position.
    text_x: f32,
}

/// The `armor_status` element.
#[derive(Debug)]
pub struct ArmorStatusElement {
    text_color: u32,
    shadow: bool,
    mode: Durability,
    show_held: bool,
    horizontal: bool,
    color_by_durability: bool,
    /// Anchored on the right: icons on the right, text to their left.
    mirrored: bool,
    /// The stitched item atlas key the hotbar uses too.
    atlas: UiTexture,
    /// Helmet, chestplate, leggings, boots, held.
    rows: [Row; 5],
    /// Width of the text column for `mode`.
    text_column: f32,
    font_generation: u64,
    /// The editor preview (a diamond set and a sword), built the first time it is needed.
    samples: Option<Box<[ItemStack; 5]>>,
    use_samples: bool,
    size: (f32, f32),
}

impl Default for ArmorStatusElement {
    fn default() -> Self {
        ArmorStatusElement {
            text_color: 0xFFFF_FFFF,
            shadow: true,
            mode: Durability::Value,
            show_held: true,
            horizontal: false,
            color_by_durability: true,
            mirrored: false,
            atlas: crate::screens::container::default_atlas_texture(),
            rows: Default::default(),
            text_column: 0.0,
            font_generation: u64::MAX,
            samples: None,
            use_samples: false,
            size: (0.0, 0.0),
        }
    }
}

fn sample_stacks() -> Box<[ItemStack; 5]> {
    // Diamond helmet, chestplate, leggings, boots and sword, a little worn.
    let stack = |item_id: i16, damage: i16| ItemStack {
        item_id,
        count: 1,
        damage,
        nbt: None,
    };
    Box::new([
        stack(310, 20),
        stack(311, 100),
        stack(312, 300),
        stack(313, 400),
        stack(276, 61),
    ])
}

impl ArmorStatusElement {
    /// Slot `i`'s stack this frame: 0..4 are helmet..boots, 4 is the held item.
    fn source<'a>(&'a self, ctx: &'a HudCtx<'_>, i: usize) -> Option<&'a ItemStack> {
        if self.use_samples {
            return self.samples.as_ref().map(|samples| &samples[i]);
        }
        match i {
            0..=3 => ctx.data.armor[i].as_ref(),
            _ if self.show_held => ctx.data.held.as_ref(),
            _ => None,
        }
    }

    /// Rebuild row `i`'s text for `stack`.
    fn rebuild_row(&mut self, i: usize, stack: &ItemStack, key: RowKey, font: &FontRenderer) {
        let mode = self.mode;
        let color_by_durability = self.color_by_durability;
        let text_color = self.text_color;
        let row = &mut self.rows[i];
        row.key = Some(key);
        row.text.clear();
        row.count.clear();
        row.rgb = text_color & 0x00FF_FFFF;
        if let Some(arrows) = key.arrows {
            // Vanilla's own count text marks an empty stack red (`§c0`).
            let _ = if arrows == 0 {
                write!(row.count, "\u{a7}c0")
            } else {
                write!(row.count, "{arrows}")
            };
        }
        let max = i32::from(stack_ext::props(stack).max_damage);
        if mode != Durability::Off && stack_ext::is_stack_damageable(stack) && max > 0 {
            let remaining = (max - i32::from(stack.damage)).clamp(0, max);
            let _ = match mode {
                Durability::Value => write!(row.text, "{remaining}"),
                Durability::Percent => {
                    let mut percent = remaining * 100 / max;
                    if remaining > 0 {
                        percent = percent.max(1);
                    }
                    write!(row.text, "{percent}%")
                }
                Durability::ValueMax => write!(row.text, "{remaining}/{max}"),
                Durability::Off => Ok(()),
            };
            if color_by_durability {
                row.rgb = durability_color(remaining, max);
            }
        }
        row.text_width = font.string_width(&row.text).max(0) as f32;
    }
}

impl HudElement for ArmorStatusElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("armor_status", "armor_status")
    }

    fn sync(&mut self, settings: &ModSettings) {
        const ID: &str = "armor_status";
        self.text_color = settings.color(ID, "text_color");
        self.shadow = settings.bool(ID, "text_shadow");
        self.mode = match settings.choice(ID, "durability") {
            1 => Durability::Percent,
            2 => Durability::ValueMax,
            3 => Durability::Off,
            _ => Durability::Value,
        };
        self.show_held = settings.bool(ID, "show_held");
        self.horizontal = settings.choice(ID, "layout") == 1;
        self.color_by_durability = settings.bool(ID, "color_by_durability");
        self.mirrored = matches!(
            settings.placement(ID, ID).anchor,
            HudAnchor::TopRight | HudAnchor::CenterRight | HudAnchor::BottomRight
        );
        for row in &mut self.rows {
            row.key = None;
        }
        self.font_generation = u64::MAX;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        if ctx.data.spectator && !ctx.editing {
            return None;
        }
        let font_changed = font.layout_generation() != self.font_generation;
        if font_changed {
            self.font_generation = font.layout_generation();
            self.text_column = font.string_width(self.mode.widest_sample()).max(0) as f32;
        }
        self.use_samples = false;
        if ctx.editing && (0..5).all(|i| self.source(ctx, i).is_none()) {
            self.samples.get_or_insert_with(sample_stacks);
            self.use_samples = true;
        }
        let mut count = 0;
        for i in 0..5 {
            let Some(stack) = self.source(ctx, i) else {
                self.rows[i].present = false;
                continue;
            };
            let key = RowKey {
                item: stack.item_id,
                damage: stack.damage,
                arrows: (i == 4 && stack.item_id == BOW).then_some(ctx.data.arrow_count),
            };
            if self.rows[i].key != Some(key) || font_changed {
                // The source borrows `self` (the samples) or `ctx`; clone the small stack out
                // only on this rare rebuild path.
                let stack = stack.clone();
                self.rebuild_row(i, &stack, key, font);
            }
            self.rows[i].present = true;
            count += 1;
        }
        if count == 0 {
            return None;
        }
        let column = if self.mode == Durability::Off {
            0.0
        } else {
            TEXT_GAP + self.text_column
        };
        self.size = if self.horizontal {
            let mut x = 0.0;
            for row in self.rows.iter_mut().filter(|row| row.present) {
                if x > 0.0 {
                    x += ENTRY_GAP;
                }
                row.x = x;
                row.y = 0.0;
                row.text_x = x + ICON + TEXT_GAP;
                x += ICON;
                if !row.text.is_empty() {
                    x += column;
                }
            }
            (x, ICON)
        } else {
            let width = ICON + column;
            let mirrored = self.mirrored;
            for (n, row) in self.rows.iter_mut().filter(|row| row.present).enumerate() {
                row.y = n as f32 * ROW_PITCH;
                if mirrored {
                    row.x = width - ICON;
                    row.text_x = row.x - TEXT_GAP - row.text_width;
                } else {
                    row.x = 0.0;
                    row.text_x = ICON + TEXT_GAP;
                }
            }
            (width, count as f32 * ROW_PITCH - 1.0)
        };
        Some(self.size)
    }

    fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        mc_core::profile_scope!("hud.mod.armor_status");
        let alpha = self.text_color & 0xFF00_0000;
        for (i, row) in self.rows.iter().enumerate() {
            if !row.present {
                continue;
            }
            let Some(stack) = self.source(ctx, i) else {
                continue;
            };
            let (x, y) = (row.x as i32, row.y as i32);
            draw_item_icon_tinted(d, &self.atlas, x, y, stack, 0xFFFF_FFFF);
            draw_durability_bar(d, x, y, stack);
            if row.count.is_empty() {
                draw_count_text(d, font, x, y, stack);
            } else {
                draw_count_text_with_override(d, font, x, y, stack, Some(&row.count));
            }
            if !row.text.is_empty() {
                font.draw_string(
                    d,
                    &row.text,
                    row.text_x,
                    row.y + 4.0,
                    (alpha | row.rgb) as i32,
                    self.shadow,
                );
            }
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

    fn stack(item_id: i16, damage: i16, count: i32) -> Option<ItemStack> {
        Some(ItemStack {
            item_id,
            count,
            damage,
            nbt: None,
        })
    }

    #[test]
    fn durability_colours_follow_the_thresholds() {
        assert_eq!(durability_color(363, 363), 0xFF_FFFF);
        assert_eq!(durability_color(80, 100), 0xFF_FFFF);
        assert_eq!(durability_color(79, 100), 0x55_FF55);
        assert_eq!(durability_color(40, 100), 0xFF_FF55);
        assert_eq!(durability_color(25, 100), 0xFF_AA00);
        assert_eq!(durability_color(10, 100), 0xFF_5555);
        assert_eq!(durability_color(9, 100), 0xAA_0000);
    }

    #[test]
    fn empty_slots_are_skipped_and_text_is_per_mode() {
        let mut settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ArmorStatusElement::default();
        element.sync(&settings);
        let mut data = HudData::default();
        assert!(element
            .layout(&ctx(&data, &settings, false), &font)
            .is_none());
        // A diamond helmet (363 uses) at 63 damage and diamond boots, nothing between.
        data.armor[0] = stack(310, 63, 1);
        data.armor[3] = stack(313, 0, 1);
        // A stack of blocks in hand: no durability text.
        data.held = stack(1, 0, 64);
        let (w, h) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(h, 3.0 * ROW_PITCH - 1.0, "three rows: helmet, boots, held");
        let column = font.string_width("0000") as f32;
        assert_eq!(w, ICON + TEXT_GAP + column);
        assert_eq!(element.rows[0].text, "300");
        assert_eq!(element.rows[0].rgb, 0xFF_FFFF, "82 % is still white");
        assert_eq!(element.rows[3].text, "429");
        assert!(element.rows[4].text.is_empty());
        assert!(!element.rows[1].present && !element.rows[2].present);
        assert_eq!(
            (element.rows[3].y, element.rows[4].y),
            (ROW_PITCH, 2.0 * ROW_PITCH)
        );

        settings.set_choice("armor_status", "durability", 1);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.rows[0].text, "82%");
        settings.set_choice("armor_status", "durability", 2);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.rows[0].text, "300/363");
        // A nearly broken piece still reads at least 1 % and is dark red.
        data.armor[0] = stack(310, 362, 1);
        settings.set_choice("armor_status", "durability", 1);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.rows[0].text, "1%");
        assert_eq!(element.rows[0].rgb, 0xAA_0000);
    }

    #[test]
    fn a_held_bow_counts_arrows_and_text_is_cached() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ArmorStatusElement::default();
        element.sync(&settings);
        let mut data = HudData {
            held: stack(BOW, 4, 1),
            arrow_count: 23,
            ..HudData::default()
        };
        element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(element.rows[4].count, "23");
        assert_eq!(element.rows[4].text, "380");
        let ptr = element.rows[4].text.as_ptr();
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(
            element.rows[4].text.as_ptr(),
            ptr,
            "nothing changed: no rebuild"
        );
        data.arrow_count = 0;
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.rows[4].count, "\u{a7}c0");
    }

    #[test]
    fn spectators_see_nothing_and_the_editor_previews_a_set() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ArmorStatusElement::default();
        element.sync(&settings);
        let data = HudData {
            spectator: true,
            held: stack(276, 0, 1),
            ..HudData::default()
        };
        assert!(element
            .layout(&ctx(&data, &settings, false), &font)
            .is_none());
        let empty = HudData::default();
        let (_, h) = element
            .layout(&ctx(&empty, &settings, true), &font)
            .unwrap();
        assert_eq!(h, 5.0 * ROW_PITCH - 1.0, "four armour pieces and a sword");
    }

    #[test]
    fn right_anchors_mirror_and_horizontal_lays_out_in_a_row() {
        let mut settings = ModSettings::new();
        settings.set_placement(
            "armor_status",
            "armor_status",
            mc_mods::HudPlacement::new(HudAnchor::BottomRight, -2.0, -2.0),
        );
        let font = uniform_font();
        let mut element = ArmorStatusElement::default();
        element.sync(&settings);
        let data = HudData {
            armor: [stack(310, 0, 1), None, None, None],
            ..HudData::default()
        };
        let (w, _) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(element.rows[0].x, w - ICON, "icon at the screen edge");
        assert!(element.rows[0].text_x < element.rows[0].x);

        settings.set_choice("armor_status", "layout", 1);
        element.sync(&settings);
        let both = HudData {
            armor: [stack(310, 0, 1), None, None, stack(313, 0, 1)],
            ..HudData::default()
        };
        let (w, h) = element
            .layout(&ctx(&both, &settings, false), &font)
            .unwrap();
        let column = font.string_width("0000") as f32;
        assert_eq!(h, ICON);
        assert_eq!(w, 2.0 * (ICON + TEXT_GAP + column) + ENTRY_GAP);
        assert_eq!(element.rows[3].x, ICON + TEXT_GAP + column + ENTRY_GAP);
    }
}
