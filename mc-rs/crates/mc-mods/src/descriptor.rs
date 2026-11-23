//! Data-only descriptions of mods and their options. The menu renders a generic widget per
//! [`OptionKind`], so a mod never draws its own settings UI.

use std::borrow::Cow;

use crate::category::Category;
use crate::hud::HudPlacement;

/// One mod in the catalog.
///
/// Every field is `'static` data so descriptors can be `static` items in `src/mods/<id>.rs`.
#[derive(Debug)]
pub struct ModDescriptor {
    /// Stable id: the JSON key and the lookup key for every [`crate::ModSettings`] getter.
    /// Never rename one, or users lose their settings for it.
    pub id: &'static str,
    /// The row label in the menu.
    pub name: &'static str,
    /// Which window the mod is listed in.
    pub category: Category,
    /// One line, shown under the row when its options are expanded.
    pub description: &'static str,
    /// Whether a fresh install has the mod on. `false` for every mod (PLAN.md: off by default)
    /// except `raw_input` and `borderless_fullscreen`, where on is today's behaviour.
    pub default_enabled: bool,
    /// The options the menu shows when the row is expanded, in display order.
    pub options: &'static [OptionDescriptor],
    /// Draggable HUD elements this mod draws. Empty for mods with no HUD presence.
    pub hud_elements: &'static [HudElementDescriptor],
}

impl ModDescriptor {
    /// The option with this id.
    #[must_use]
    pub fn option(&self, id: &str) -> Option<&'static OptionDescriptor> {
        self.options.iter().find(|option| option.id == id)
    }

    /// Index of the option with this id in [`ModDescriptor::options`].
    #[must_use]
    pub fn option_index(&self, id: &str) -> Option<usize> {
        self.options.iter().position(|option| option.id == id)
    }

    /// Index of the HUD element with this id in [`ModDescriptor::hud_elements`].
    #[must_use]
    pub fn hud_element_index(&self, id: &str) -> Option<usize> {
        self.hud_elements
            .iter()
            .position(|element| element.id == id)
    }
}

/// One user-facing setting of a mod.
#[derive(Debug)]
pub struct OptionDescriptor {
    /// Stable id, unique within the mod. The JSON key.
    pub id: &'static str,
    /// The label shown in the menu.
    pub label: &'static str,
    /// Which widget the menu draws, and the value's valid range.
    pub kind: OptionKind,
    /// The value a fresh install starts with. Must match `kind` (a unit test checks the whole
    /// catalog): `Toggle` → `Bool`, `Slider` → `Float`, `Choice` → `Choice`, `Color` → `Color`,
    /// `Keybind` → `Key`, `Text` → `Text`.
    pub default: OptionValue,
    /// Optional one-line help text (may be empty).
    pub description: &'static str,
}

/// What kind of value an option holds, and therefore which widget the menu draws for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionKind {
    /// On/off. Drawn as an Aero switch.
    Toggle,
    /// A number in `min..=max`, snapped to `step`. Drawn as a slider whose value shows as a
    /// badge: `format!("{:.decimals$}{unit}", value)`.
    Slider {
        /// Smallest value.
        min: f32,
        /// Largest value.
        max: f32,
        /// Snap increment; `0.0` means continuous.
        step: f32,
        /// Suffix shown after the number in the badge, e.g. `"%"`, `"x"`, `" ms"`.
        unit: &'static str,
        /// Digits after the decimal point in the badge.
        decimals: u8,
    },
    /// One of a fixed list. Stored as the index; persisted as the label text. Drawn as a badge
    /// that cycles on click (right-click cycles backwards).
    Choice {
        /// Display labels, in order. Index 0 is the first.
        choices: &'static [&'static str],
    },
    /// A packed `0xAARRGGBB` colour. Drawn as a swatch that opens presets plus HSV (and alpha
    /// when `alpha` is true) sliders.
    Color {
        /// Whether the user may change alpha. When `false` the stored alpha is always `0xFF`.
        alpha: bool,
    },
    /// A key or mouse button, in vanilla's LWJGL `options.txt` encoding: a positive LWJGL key
    /// code, a mouse button as `button - 100` (left = `-100`), or `0` for unbound. Convert with
    /// `mc_render::input::Input::from_lwjgl`. Drawn as a capture button: click, then press a key
    /// or mouse button; Escape cancels, Backspace unbinds.
    Keybind,
    /// Free text up to `max_len` characters. Drawn as an inline text field.
    Text {
        /// Maximum length in `char`s.
        max_len: usize,
    },
    /// A push button. Stores nothing (`default: OptionValue::Action`, never written to the
    /// file); a click sends `ModEvent::Button { mod_id, option_id }` to the client.
    Button {
        /// The text on the button, e.g. `"Test mic"`.
        label: &'static str,
    },
    /// Text that is persisted like [`OptionKind::Text`] but **not drawn** by the menu: state a
    /// mod shows its own way, e.g. the chosen audio device behind a dynamic cycle row.
    HiddenText {
        /// Maximum length in `char`s.
        max_len: usize,
    },
    /// Exactly [`OptionKind::Slider`] (value `Float`, clamped and snapped the same way), but the
    /// badge text comes from `label`, e.g. `"Full"` at the top stop or `"12:00"` for a time.
    SliderLabeled {
        /// Smallest value.
        min: f32,
        /// Largest value.
        max: f32,
        /// Snap increment; `0.0` means continuous.
        step: f32,
        /// Writes the badge text for a value into the (cleared) string.
        label: SliderLabel,
    },
}

/// The badge formatter of [`OptionKind::SliderLabeled`]: `SliderLabel(my_fn)` with
/// `fn my_fn(value: f32, out: &mut String)`. (A newtype so the kind keeps `PartialEq` without
/// comparing function pointers.)
#[derive(Clone, Copy)]
pub struct SliderLabel(pub fn(f32, &mut String));

impl std::fmt::Debug for SliderLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SliderLabel(..)")
    }
}

impl PartialEq for SliderLabel {
    fn eq(&self, other: &Self) -> bool {
        // Formatters are compared by identity; the address is all a fn pointer has.
        self.0 as usize == other.0 as usize
    }
}

/// A stored option value. Also the type of [`OptionDescriptor::default`].
#[derive(Debug, Clone, PartialEq)]
pub enum OptionValue {
    /// [`OptionKind::Toggle`].
    Bool(bool),
    /// [`OptionKind::Slider`].
    Float(f32),
    /// [`OptionKind::Choice`]: an index into `choices`.
    Choice(usize),
    /// [`OptionKind::Color`]: `0xAARRGGBB`.
    Color(u32),
    /// [`OptionKind::Keybind`]: LWJGL code (see [`OptionKind::Keybind`]).
    Key(i32),
    /// [`OptionKind::Text`] and [`OptionKind::HiddenText`]. `Cow::Borrowed` makes a `const`
    /// default possible.
    Text(Cow<'static, str>),
    /// [`OptionKind::Button`]: no value.
    Action,
}

/// The LWJGL codes a key option may hold: mouse buttons down to `-100 - 15`, keys up to 255.
pub const KEY_CODE_RANGE: std::ops::RangeInclusive<i32> = -115..=255;

impl OptionKind {
    /// Whether `value` has the variant this kind stores (range is not checked).
    #[must_use]
    pub fn accepts(&self, value: &OptionValue) -> bool {
        matches!(
            (self, value),
            (OptionKind::Toggle, OptionValue::Bool(_))
                | (OptionKind::Slider { .. }, OptionValue::Float(_))
                | (OptionKind::SliderLabeled { .. }, OptionValue::Float(_))
                | (OptionKind::Choice { .. }, OptionValue::Choice(_))
                | (OptionKind::Color { .. }, OptionValue::Color(_))
                | (OptionKind::Keybind, OptionValue::Key(_))
                | (OptionKind::Text { .. }, OptionValue::Text(_))
                | (OptionKind::HiddenText { .. }, OptionValue::Text(_))
                | (OptionKind::Button { .. }, OptionValue::Action)
        )
    }

    /// Whether the value is written to the settings file (everything except buttons).
    #[must_use]
    pub fn is_persisted(&self) -> bool {
        !matches!(self, OptionKind::Button { .. })
    }

    /// Whether the menu draws a widget for this option (everything except hidden text).
    #[must_use]
    pub fn is_visible(&self) -> bool {
        !matches!(self, OptionKind::HiddenText { .. })
    }

    /// Coerce `value` into this kind's valid range, falling back to `default` when it has the
    /// wrong variant or is unusable (NaN, out-of-range index).
    ///
    /// * slider: clamped to `min..=max`, snapped to `step` from `min`;
    /// * choice: an out-of-range index becomes the default;
    /// * colour: alpha forced to `0xFF` unless the option allows alpha;
    /// * key: codes outside [`KEY_CODE_RANGE`] become unbound (`0`);
    /// * text: truncated to `max_len` characters, control characters removed.
    #[must_use]
    pub fn sanitize(&self, value: OptionValue, default: &OptionValue) -> OptionValue {
        if !self.accepts(&value) {
            return default.clone();
        }
        match (self, value) {
            (
                OptionKind::Slider { min, max, step, .. }
                | OptionKind::SliderLabeled { min, max, step, .. },
                OptionValue::Float(v),
            ) => {
                if !v.is_finite() {
                    return default.clone();
                }
                OptionValue::Float(snap_slider(v, *min, *max, *step))
            }
            (OptionKind::Choice { choices }, OptionValue::Choice(index)) => {
                if index < choices.len() {
                    OptionValue::Choice(index)
                } else {
                    default.clone()
                }
            }
            (OptionKind::Color { alpha }, OptionValue::Color(argb)) => {
                OptionValue::Color(if *alpha { argb } else { argb | 0xFF00_0000 })
            }
            (OptionKind::Keybind, OptionValue::Key(code)) => {
                OptionValue::Key(if KEY_CODE_RANGE.contains(&code) {
                    code
                } else {
                    0
                })
            }
            (
                OptionKind::Text { max_len } | OptionKind::HiddenText { max_len },
                OptionValue::Text(text),
            ) => {
                let clean = text.chars().any(char::is_control) || text.chars().count() > *max_len;
                if clean {
                    OptionValue::Text(Cow::Owned(
                        text.chars()
                            .filter(|c| !c.is_control())
                            .take(*max_len)
                            .collect(),
                    ))
                } else {
                    OptionValue::Text(text)
                }
            }
            (_, value) => value,
        }
    }

    /// The text the menu shows in a slider or choice badge. Other kinds return an empty string
    /// (their widgets have no text value).
    #[must_use]
    pub fn format_value(&self, value: &OptionValue) -> String {
        match (self, value) {
            (OptionKind::Slider { unit, decimals, .. }, OptionValue::Float(v)) => {
                format!("{:.*}{unit}", usize::from(*decimals), v)
            }
            (OptionKind::SliderLabeled { label, .. }, OptionValue::Float(v)) => {
                let mut out = String::new();
                (label.0)(*v, &mut out);
                out
            }
            (OptionKind::Choice { choices }, OptionValue::Choice(index)) => {
                choices.get(*index).copied().unwrap_or("?").to_owned()
            }
            _ => String::new(),
        }
    }
}

/// Clamp `v` into `min..=max` and snap it to the `step` grid anchored at `min`.
#[must_use]
pub fn snap_slider(v: f32, min: f32, max: f32, step: f32) -> f32 {
    let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
    let mut v = v.clamp(lo, hi);
    if step > 0.0 && step.is_finite() {
        v = lo + ((v - lo) / step).round() * step;
        v = v.clamp(lo, hi);
        // Kill float noise such as 0.30000001 so the badge and the JSON read cleanly.
        v = (v * 10_000.0).round() / 10_000.0;
    }
    // A symmetric range snaps 0 to -0.0 (`-0.2 + 40 * 0.005` rounds to -0); `-0.0 + 0.0` is
    // +0.0, so the badge never reads "-0.000" and the file never says `-0`.
    v + 0.0
}

/// One draggable HUD element a mod draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudElementDescriptor {
    /// Stable id, unique within the mod (often the mod id itself). The JSON key.
    pub id: &'static str,
    /// The label the HUD editor shows on the element's outline.
    pub name: &'static str,
    /// Where the element sits until the user drags it.
    pub default_placement: HudPlacement,
}
