//! Armor Status: Shows your armour and held item with durability.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Vanilla item icons with their
//! durability bar and count, plus a durability number; a held bow shows the arrows you carry.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

/// Durability text modes, in index order.
pub const DURABILITY_CHOICES: [&str; 4] = ["Value", "Percent", "Value/Max", "Off"];

static OPTIONS: [OptionDescriptor; 6] = [
    OptionDescriptor {
        id: "durability",
        label: "Durability",
        kind: OptionKind::Choice {
            choices: &DURABILITY_CHOICES,
        },
        default: OptionValue::Choice(0),
        description: "Uses left (363), percent left (87%), uses left of the maximum, or none.",
    },
    OptionDescriptor {
        id: "show_held",
        label: "Held item",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Also show the item in your hand (a bow shows your arrow count).",
    },
    OptionDescriptor {
        id: "layout",
        label: "Layout",
        kind: OptionKind::Choice {
            choices: &["Vertical", "Horizontal"],
        },
        default: OptionValue::Choice(0),
        description: "",
    },
    OptionDescriptor {
        id: "color_by_durability",
        label: "Color by durability",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "White when new, through green, yellow and orange to red when nearly broken.",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
];

/// Just right of the hotbar (its right edge is the screen centre + 91).
static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "armor_status",
    name: "Armor Status",
    default_placement: HudPlacement::new(HudAnchor::BottomCenter, 116.0, -2.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "armor_status",
    name: "Armor Status",
    category: Category::Hud,
    description: "Shows your armor and held item with their durability.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
