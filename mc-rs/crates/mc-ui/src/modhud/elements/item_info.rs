//! Held Item Info (`item_info`): the look, a compact line over the hotbar
//! with the held item's enchantments, and a stats line under it.
//!
//! ```text
//! Sh 5 FA 2 Un 3             bold; a held potion lists its effects instead
//! +13.25 dmg  23 arrows
//! ```
//!
//! * Enchantments are short codes plus the level, in NBT order (`StoredEnchantments` for
//!   an enchanted book). A code is two letters made from the enchantment's English name
//!   ([`write_enchant_code`]). An id 1.8.9 does not register shows as `?<id>` rather than
//!   failing.
//! * A potion lists `Speed II (1:30)` per effect, as its tooltip does (`ItemPotion.getEffects`;
//!   instant effects have no time).
//! * Damage is the tooltip's own "+X Attack Damage": the item's attack modifier plus 1.25 per
//!   Sharpness level (Smite and Bane count for nothing against the `UNDEFINED` creature type),
//!   formatted like `ItemStack.DECIMALFORMAT`, `#.###`.
//! * The count is how many of the held item (same id and damage) you carry in the 36 main
//!   slots, or your arrows while you hold a bow (`HudData::held_total` / `arrow_count`).
//! * Half-size text by default, lines centred, just under vanilla's item name.
//! * The lines are rebuilt only when the held stack, a count it shows or an option changes.

use std::fmt::Write as _;

use mc_mods::ModSettings;

use super::potion_status::{scale_alpha, write_duration, write_level, ROMAN};
use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement, HudTextStyle};
use crate::screens::container::stack_ext::{self, ItemStack};
use crate::screens::container::tooltip;
use crate::widget::{DrawList, Transform};

/// Vanilla item ids this element special-cases.
const BOW: i16 = 261;
const POTION: i16 = 373;
const ENCHANTED_BOOK: i16 = 403;
/// Sharpness, the one enchantment that adds to the tooltip's attack damage.
const SHARPNESS: i16 = 16;

/// Line pitch at full size, in font pixels (a 9-pixel line plus one).
const LINE_PITCH: f32 = 10.0;

/// Small words a short code skips ("Bane of Arthropods", "Luck of the Sea").
const SKIPPED_WORDS: [&str; 2] = ["of", "the"];

/// Append the short code of a 1.8.9 enchantment id, made by one rule from vanilla's own
/// English name (`mc_data::enchantment`, the `en_US` name): a name of several words gives the
/// initials of its words ("Fire Aspect" is `FA`, "Luck of the Sea" is `LS`), a one-word name its
/// first two letters ("Sharpness" is `Sh`). Every 1.8.9 enchantment gets its own code. `false`
/// for an id 1.8.9 does not register.
fn write_enchant_code(out: &mut String, id: i16) -> bool {
    let Some(info) = u8::try_from(id).ok().and_then(mc_data::enchantment) else {
        return false;
    };
    let words = || {
        info.display_name
            .split(' ')
            .filter(|word| !word.is_empty() && !SKIPPED_WORDS.contains(word))
    };
    if words().nth(1).is_some() {
        let initials = words().filter_map(|word| word.chars().next());
        out.extend(initials.map(|c| c.to_ascii_uppercase()));
    } else {
        out.extend(info.display_name.chars().take(2));
    }
    true
}

/// Append `CODE level`, space-separated from what is already on the line.
fn write_enchant(out: &mut String, id: i16, level: i16, roman: bool) {
    if !out.is_empty() {
        out.push(' ');
    }
    if !write_enchant_code(out, id) {
        let _ = write!(out, "?{id}");
    }
    out.push(' ');
    match usize::try_from(level) {
        Ok(n @ 1..=10) if roman => out.push_str(ROMAN[n - 1]),
        _ => {
            let _ = write!(out, "{level}");
        }
    }
}

/// Vanilla's `ItemStack.DECIMALFORMAT` (`#.###`): at most three decimals, trailing zeros
/// trimmed. Rust's `{:.3}` rounds ties to even, as `DecimalFormat`'s default `HALF_EVEN`.
fn write_amount(out: &mut String, value: f64) {
    let start = out.len();
    let _ = write!(out, "{value:.3}");
    if out[start..].contains('.') {
        while out.ends_with('0') {
            out.pop();
        }
        if out.ends_with('.') {
            out.pop();
        }
    }
}

/// The tooltip's attack damage for `stack`: the modifier `ItemSword`/`ItemTool` put on the
/// attribute map (`ItemProps::attack_damage` already holds that total: a diamond sword is 7)
/// plus `EnchantmentHelper.getModifierForCreature(stack, UNDEFINED)`, which is 1.25 per
/// Sharpness level, every `ench` entry counted. `None` for items without one.
#[must_use]
fn attack_damage(stack: &ItemStack) -> Option<f64> {
    let props = stack_ext::props(stack);
    if !(props.is_tool() && props.attack_damage > 0.0 && props.tool_material.is_some()) {
        return None;
    }
    let sharpness: f64 = stack
        .nbt
        .as_ref()
        .and_then(|nbt| nbt.get_list("ench"))
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.as_compound())
        .filter(|enchantment| enchantment.get_i16("id") == Some(SHARPNESS))
        .map(|enchantment| f64::from(enchantment.get_i16("lvl").unwrap_or(0)) * 1.25)
        .sum();
    Some(f64::from(props.attack_damage) + sharpness)
}

/// The vanilla held-item name's alpha for `ticks` of highlight left
/// (`GuiIngame.renderSelectedItem`: `min(255, ticks * 256 / 10)`).
#[must_use]
fn name_alpha(ticks: i32) -> u32 {
    (ticks.max(0) as u32 * 256 / 10).min(255)
}

/// What the lines were built from.
#[derive(Debug, Clone, PartialEq)]
struct Built {
    held: Option<ItemStack>,
    /// The arrow count, while holding a bow with counts on (else 0).
    arrows: u32,
    /// The held item's total, while holding anything else with counts on (else 0).
    total: u32,
}

/// The `item_info` element.
#[derive(Debug)]
pub struct ItemInfoElement {
    style: HudTextStyle,
    show_enchants: bool,
    show_damage: bool,
    show_count: bool,
    roman: bool,
    fade_with_name: bool,
    small_text: bool,
    /// What `lines` were built from; `None` forces a rebuild.
    built: Option<Built>,
    font_generation: u64,
    /// The enchantment (or potion) line and the stats line; an empty line is not drawn.
    lines: [String; 2],
    widths: [f32; 2],
    /// The HUD editor's sample (`Sh 5 FA 2` / `+13.25 dmg`), built the first time it is shown.
    preview: Option<([String; 2], [f32; 2])>,
    /// This frame draws the preview.
    use_preview: bool,
    /// This frame's text alpha (the fade with the item name), 0..=255.
    alpha: u32,
    size: (f32, f32),
}

impl Default for ItemInfoElement {
    fn default() -> Self {
        ItemInfoElement {
            style: HudTextStyle::default(),
            show_enchants: true,
            show_damage: true,
            show_count: true,
            roman: false,
            fade_with_name: false,
            small_text: true,
            built: None,
            font_generation: u64::MAX,
            lines: Default::default(),
            widths: [0.0; 2],
            preview: None,
            use_preview: false,
            alpha: 255,
            size: (0.0, 0.0),
        }
    }
}

impl ItemInfoElement {
    /// Font pixels to element pixels.
    fn text_scale(&self) -> f32 {
        if self.small_text {
            0.5
        } else {
            1.0
        }
    }

    /// Rebuild both lines for `built`.
    fn rebuild(&mut self, built: Built, font: &FontRenderer) {
        for line in &mut self.lines {
            line.clear();
        }
        if let Some(stack) = built.held.as_ref() {
            if self.show_enchants {
                self.write_first_line(stack);
            }
            let stats = &mut self.lines[1];
            if self.show_damage {
                if let Some(damage) = attack_damage(stack) {
                    stats.push('+');
                    write_amount(stats, damage);
                    stats.push_str(" dmg");
                }
            }
            let count = if stack.item_id == BOW {
                built.arrows
            } else if stack_ext::max_stack_size(stack) > 1 && built.total > 1 {
                built.total
            } else {
                0
            };
            if count > 0 {
                if !stats.is_empty() {
                    stats.push_str("  ");
                }
                let _ = match (stack.item_id == BOW, count) {
                    (true, 1) => write!(stats, "1 arrow"),
                    (true, n) => write!(stats, "{n} arrows"),
                    (false, n) => write!(stats, "x{n}"),
                };
            }
        }
        for (width, line) in self.widths.iter_mut().zip(&self.lines) {
            *width = font.string_width(line).max(0) as f32;
        }
        self.built = Some(built);
    }

    /// The enchantment line (bold), or a potion's effects.
    fn write_first_line(&mut self, stack: &ItemStack) {
        let roman = self.roman;
        let line = &mut self.lines[0];
        if stack.item_id == POTION {
            for (i, effect) in tooltip::potion_effects(stack)
                .unwrap_or_default()
                .iter()
                .enumerate()
            {
                if i > 0 {
                    line.push_str("  ");
                }
                match mc_data::potion(effect.id) {
                    Some(potion) => line.push_str(potion.display_name),
                    None => {
                        let _ = write!(line, "Effect {}", effect.id);
                    }
                }
                write_level(line, effect.amplifier, roman);
                // java: `ItemPotion.addInformation` adds the time only when `duration > 20`.
                if effect.duration > 20 {
                    line.push_str(" (");
                    write_duration(line, effect.duration, false);
                    line.push(')');
                }
            }
            return;
        }
        let key = if stack.item_id == ENCHANTED_BOOK {
            "StoredEnchantments"
        } else {
            "ench"
        };
        let Some(list) = stack.nbt.as_ref().and_then(|nbt| nbt.get_list(key)) else {
            return;
        };
        let mut codes = String::new();
        for enchantment in list.iter().filter_map(|entry| entry.as_compound()) {
            let Some(id) = enchantment.get_i16("id") else {
                continue;
            };
            write_enchant(
                &mut codes,
                id,
                enchantment.get_i16("lvl").unwrap_or(0),
                roman,
            );
        }
        if !codes.is_empty() {
            line.push_str("\u{a7}l");
            line.push_str(&codes);
        }
    }

    /// The editor sample: Sharpness V and Fire Aspect II on a diamond sword.
    fn build_preview(&self, font: &FontRenderer) -> ([String; 2], [f32; 2]) {
        let mut codes = String::new();
        write_enchant(&mut codes, SHARPNESS, 5, self.roman);
        write_enchant(&mut codes, 20, 2, self.roman);
        let lines = [format!("\u{a7}l{codes}"), String::from("+13.25 dmg")];
        let widths = lines
            .each_ref()
            .map(|line| font.string_width(line).max(0) as f32);
        (lines, widths)
    }
}

impl HudElement for ItemInfoElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("item_info", "item_info")
    }

    fn sync(&mut self, settings: &ModSettings) {
        const ID: &str = "item_info";
        self.style = HudTextStyle::read(settings, ID);
        self.show_enchants = settings.bool(ID, "show_enchants");
        self.show_damage = settings.bool(ID, "show_damage");
        self.show_count = settings.bool(ID, "show_count");
        self.roman = settings.choice(ID, "numerals") == 1;
        self.fade_with_name = settings.bool(ID, "fade_with_name");
        self.small_text = settings.bool(ID, "small_text");
        self.built = None;
        self.preview = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        let data = ctx.data;
        if data.spectator && !ctx.editing {
            return None;
        }
        self.alpha = if self.fade_with_name && !ctx.editing {
            name_alpha(data.held_highlight_ticks)
        } else {
            255
        };
        if self.alpha == 0 {
            return None;
        }
        if font.layout_generation() != self.font_generation {
            self.font_generation = font.layout_generation();
            self.built = None;
            self.preview = None;
        }
        let held = data.held.as_ref();
        let bow = held.is_some_and(|stack| stack.item_id == BOW);
        let arrows = if self.show_count && bow {
            data.arrow_count
        } else {
            0
        };
        let total = if self.show_count && !bow {
            data.held_total
        } else {
            0
        };
        let fresh = self.built.as_ref().is_some_and(|built| {
            built.arrows == arrows && built.total == total && built.held.as_ref() == held
        });
        if !fresh {
            let built = Built {
                held: held.cloned(),
                arrows,
                total,
            };
            self.rebuild(built, font);
        }
        self.use_preview = ctx.editing && self.lines.iter().all(String::is_empty);
        if self.use_preview && self.preview.is_none() {
            self.preview = Some(self.build_preview(font));
        }
        let (lines, widths) = match (&self.preview, self.use_preview) {
            (Some((lines, widths)), true) => (lines, widths),
            _ => (&self.lines, &self.widths),
        };
        let shown = lines.iter().filter(|line| !line.is_empty()).count();
        if shown == 0 {
            return None;
        }
        let widest = widths.iter().copied().fold(0.0, f32::max);
        let scale = self.text_scale();
        self.size = (widest * scale, shown as f32 * LINE_PITCH * scale);
        Some(self.size)
    }

    fn draw(&mut self, _ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        let color = if self.alpha >= 255 {
            self.style.color
        } else {
            scale_alpha(self.style.color, self.alpha as f32 / 255.0)
        };
        let (lines, widths) = match (&self.preview, self.use_preview) {
            (Some((lines, widths)), true) => (lines, widths),
            _ => (&self.lines, &self.widths),
        };
        let scale = self.text_scale();
        let inner_width = self.size.0 / scale;
        let shadow = self.style.shadow;
        d.with_transform(Transform::scale(scale), |d| {
            let mut y = 0.0;
            for (line, width) in lines.iter().zip(widths) {
                if line.is_empty() {
                    continue;
                }
                // Whole font pixels, so half-size text stays on the physical pixel grid.
                let x = ((inner_width - width) * 0.5).floor();
                font.draw_string(d, line, x, y, color as i32, shadow);
                y += LINE_PITCH;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::HudData;
    use crate::widget::test_support::uniform_font;
    use mc_nbt::{Nbt, NbtCompound};

    fn ctx<'a>(data: &'a HudData, settings: &'a ModSettings, editing: bool) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen: (960.0, 540.0),
            pixel_scale: 2.0,
            editing,
        }
    }

    fn stack(item_id: i16, damage: i16, count: i32) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage,
            nbt: None,
        }
    }

    fn enchanted(item_id: i16, key: &str, enchantments: &[(i16, i16)]) -> ItemStack {
        let list = enchantments
            .iter()
            .map(|&(id, lvl)| {
                let mut entry = NbtCompound::new();
                entry.insert("id", Nbt::Short(id));
                entry.insert("lvl", Nbt::Short(lvl));
                Nbt::Compound(entry)
            })
            .collect();
        let mut root = NbtCompound::new();
        root.insert(key, Nbt::list(list));
        ItemStack {
            nbt: Some(root),
            ..stack(item_id, 0, 1)
        }
    }

    fn holding(held: ItemStack) -> HudData {
        HudData {
            held: Some(held),
            ..HudData::default()
        }
    }

    fn lines(element: &mut ItemInfoElement, data: &HudData, settings: &ModSettings) -> [String; 2] {
        let font = uniform_font();
        element.layout(&ctx(data, settings, false), &font);
        element.lines.clone()
    }

    #[test]
    fn enchantments_use_short_codes_in_nbt_order() {
        let mut settings = ModSettings::new();
        let mut element = ItemInfoElement::default();
        element.sync(&settings);
        let sword = enchanted(276, "ench", &[(16, 5), (20, 2), (34, 3)]);
        let [first, second] = lines(&mut element, &holding(sword.clone()), &settings);
        assert_eq!(first, "\u{a7}lSh 5 FA 2 Un 3");
        assert_eq!(second, "+13.25 dmg", "diamond sword 7 + 5 x 1.25");
        settings.set_choice("item_info", "numerals", 1);
        element.sync(&settings);
        let [first, _] = lines(&mut element, &holding(sword), &settings);
        assert_eq!(first, "\u{a7}lSh V FA II Un III");
        // A custom server's id and level: shown, never a panic.
        let odd = enchanted(1, "ench", &[(70, 1), (16, 40), (-3, 1)]);
        let [first, second] = lines(&mut element, &holding(odd), &settings);
        assert_eq!(first, "\u{a7}l?70 I Sh 40 ?-3 I");
        assert!(second.is_empty(), "a block has no attack damage");
        // An enchanted book lists its stored enchantments.
        let book = enchanted(ENCHANTED_BOOK, "StoredEnchantments", &[(48, 4)]);
        let [first, _] = lines(&mut element, &holding(book), &settings);
        assert_eq!(first, "\u{a7}lPo IV");
    }

    /// The codes come from one rule over vanilla's English names: initials for a name of
    /// several words, the first two letters of a one-word name. All of 1.8.9's differ.
    #[test]
    fn short_codes_follow_the_name_rule_and_never_collide() {
        let code = |id: i16| {
            let mut out = String::new();
            assert!(write_enchant_code(&mut out, id), "id {id}");
            out
        };
        let expected: [(i16, &str); 25] = [
            (0, "Pr"),
            (1, "FP"),
            (2, "FF"),
            (3, "BP"),
            (4, "PP"),
            (5, "Re"),
            (6, "AA"),
            (7, "Th"),
            (8, "DS"),
            (16, "Sh"),
            (17, "Sm"),
            (18, "BA"),
            (19, "Kn"),
            (20, "FA"),
            (21, "Lo"),
            (32, "Ef"),
            (33, "ST"),
            (34, "Un"),
            (35, "Fo"),
            (48, "Po"),
            (49, "Pu"),
            (50, "Fl"),
            (51, "In"),
            (61, "LS"),
            (62, "Lu"),
        ];
        for (id, want) in expected {
            assert_eq!(code(id), want, "id {id}");
        }
        // Every registered enchantment has a code, and no two share one.
        let all: Vec<String> = mc_data::enchantments()
            .map(|info| code(i16::from(info.id)))
            .collect();
        assert_eq!(all.len(), expected.len());
        let distinct: std::collections::HashSet<&String> = all.iter().collect();
        assert_eq!(distinct.len(), all.len(), "{all:?}");
        let mut out = String::new();
        assert!(!write_enchant_code(&mut out, 9), "an unregistered id");
        assert!(!write_enchant_code(&mut out, 300), "an id past u8");
        assert!(out.is_empty());
    }

    #[test]
    fn damage_counts_and_arrows() {
        let settings = ModSettings::new();
        let mut element = ItemInfoElement::default();
        element.sync(&settings);
        let [first, second] = lines(&mut element, &holding(stack(267, 0, 1)), &settings);
        assert!(first.is_empty());
        assert_eq!(second, "+6 dmg", "iron sword");
        let mut bow = holding(stack(BOW, 0, 1));
        bow.arrow_count = 23;
        assert_eq!(lines(&mut element, &bow, &settings)[1], "23 arrows");
        bow.arrow_count = 1;
        assert_eq!(lines(&mut element, &bow, &settings)[1], "1 arrow");
        bow.arrow_count = 0;
        assert!(lines(&mut element, &bow, &settings)[1].is_empty());
        // Stackables show the total you carry when it is more than one.
        let mut pearls = holding(stack(368, 0, 3));
        pearls.held_total = 5;
        assert_eq!(lines(&mut element, &pearls, &settings)[1], "x5");
        pearls.held_total = 1;
        assert!(lines(&mut element, &pearls, &settings)[1].is_empty());
        // A sword never stacks, whatever the total.
        let mut swords = holding(stack(267, 0, 1));
        swords.held_total = 2;
        assert_eq!(lines(&mut element, &swords, &settings)[1], "+6 dmg");
    }

    #[test]
    fn a_held_potion_lists_its_effects() {
        let settings = ModSettings::new();
        let mut element = ItemInfoElement::default();
        element.sync(&settings);
        // 8226: Potion of Swiftness II.
        let [first, second] = lines(&mut element, &holding(stack(POTION, 8226, 1)), &settings);
        assert_eq!(first, "Speed 2 (1:30)");
        assert!(second.is_empty());
        // 8197: Potion of Healing, instant: no time.
        let [first, _] = lines(&mut element, &holding(stack(POTION, 8197, 1)), &settings);
        assert_eq!(first, "Instant Health");
    }

    #[test]
    fn rebuilt_only_on_change_and_sized_at_half_scale() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ItemInfoElement::default();
        element.sync(&settings);
        let data = holding(enchanted(276, "ench", &[(16, 5)]));
        let (w, h) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        let bold = font.string_width("\u{a7}lSh 5") as f32;
        assert_eq!(
            (w, h),
            (bold.max(font.string_width("+13.25 dmg") as f32) * 0.5, 10.0)
        );
        element.lines[0].push('!');
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.lines[0], "\u{a7}lSh 5!", "same stack: not rebuilt");
        // The arrow count of a non-bow does not matter.
        let more_arrows = HudData {
            arrow_count: 9,
            ..data.clone()
        };
        element.layout(&ctx(&more_arrows, &settings, false), &font);
        assert_eq!(element.lines[0], "\u{a7}lSh 5!");
        let other = holding(enchanted(276, "ench", &[(16, 4)]));
        element.layout(&ctx(&other, &settings, false), &font);
        assert_eq!(element.lines[0], "\u{a7}lSh 4");
    }

    #[test]
    fn fade_spectator_and_the_editor_preview() {
        let mut settings = ModSettings::new();
        settings.set_bool("item_info", "fade_with_name", true);
        let mut font = uniform_font();
        let mut element = ItemInfoElement::default();
        element.sync(&settings);
        let mut data = holding(stack(267, 0, 1));
        assert!(element
            .layout(&ctx(&data, &settings, false), &font)
            .is_none());
        data.held_highlight_ticks = 40;
        assert!(element
            .layout(&ctx(&data, &settings, false), &font)
            .is_some());
        assert_eq!(element.alpha, 255);
        data.held_highlight_ticks = 5;
        element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(element.alpha, 128);
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings, false), &mut font, &mut d);
        assert!(d
            .vertices()
            .iter()
            .any(|v| (v.color[3] * 255.0).round() as u32 == 128));
        // Spectators have no hotbar.
        data.spectator = true;
        assert!(element
            .layout(&ctx(&data, &settings, false), &font)
            .is_none());
        // The editor shows a sample when the real item has nothing to say.
        let empty = HudData::default();
        assert!(element
            .layout(&ctx(&empty, &settings, true), &font)
            .is_some());
        assert!(element.use_preview);
        let (lines, _) = element.preview.as_ref().unwrap();
        assert_eq!(lines[0], "\u{a7}lSh 5 FA 2");
        assert_eq!(lines[1], "+13.25 dmg");
    }

    #[test]
    fn amounts_follow_decimalformat() {
        let amount = |value: f64| {
            let mut out = String::new();
            write_amount(&mut out, value);
            out
        };
        assert_eq!(amount(7.0), "7");
        assert_eq!(amount(13.25), "13.25");
        assert_eq!(amount(8.25), "8.25");
        assert_eq!(amount(0.0625), "0.062", "ties to even");
        assert_eq!(name_alpha(0), 0);
        assert_eq!(name_alpha(1), 25);
        assert_eq!(name_alpha(10), 255);
        assert_eq!(name_alpha(40), 255);
    }
}
