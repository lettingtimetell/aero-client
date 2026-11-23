//! Held Item Info: Shows the held item's enchantments above the hotbar.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The look: a half-size
//! bold line of short enchantment codes (`Sh 5 FA 2`) or a held potion's
//! effects, and a stats line (`+13.25 dmg`, `23 arrows`), just under vanilla's item name.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

static OPTIONS: [OptionDescriptor; 8] = [
    OptionDescriptor {
        id: "show_enchants",
        label: "Enchantments",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Short codes, e.g. Sh 5 = Sharpness V, FA 2 = Fire Aspect II; a held \
                      potion lists its effects.",
    },
    OptionDescriptor {
        id: "show_damage",
        label: "Attack damage",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The tooltip's attack damage, Sharpness included.",
    },
    OptionDescriptor {
        id: "show_count",
        label: "Item / arrow count",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "How many of the held item you carry; arrows while holding a bow.",
    },
    OptionDescriptor {
        id: "numerals",
        label: "Level numerals",
        kind: OptionKind::Choice {
            choices: &["Arabic", "Roman"],
        },
        default: OptionValue::Choice(0),
        description: "",
    },
    OptionDescriptor {
        id: "fade_with_name",
        label: "Fade with item name",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Show it only for the two seconds the vanilla item name shows.",
    },
    OptionDescriptor {
        id: "small_text",
        label: "Small text",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Half-size text, to keep the line compact.",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
];

/// Just under vanilla's held-item name (which sits at the bottom - 59).
static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "item_info",
    name: "Held Item Info",
    default_placement: HudPlacement::new(HudAnchor::BottomCenter, 0.0, -42.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "item_info",
    name: "Held Item Info",
    category: Category::Hud,
    description: "Shows the held item's enchantments, attack damage and count above the \
                  hotbar.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
