//! Status Effects: Lists your active potion effects and the time left on each.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The inventory's effect list on the HUD:
//! vanilla icons from `textures/gui/container/inventory.png`, name with level, time left.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

static OPTIONS: [OptionDescriptor; 12] = [
    OptionDescriptor {
        id: "show_icons",
        label: "Icons",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "numerals",
        label: "Level numerals",
        kind: OptionKind::Choice {
            choices: &["Roman", "Arabic"],
        },
        default: OptionValue::Choice(0),
        description: "\"Speed II\" or \"Speed 2\".",
    },
    OptionDescriptor {
        id: "blink",
        label: "Blink when ending",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "An effect's text pulses when it is about to run out.",
    },
    OptionDescriptor {
        id: "blink_time_only",
        label: "Blink remaining time only",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Only the time left pulses; effect names stay steady.",
    },
    OptionDescriptor {
        id: "effect_name_color",
        label: "Color names by effect",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Each effect's name takes its potion's color, at your text color's opacity.",
    },
    OptionDescriptor {
        id: "blink_seconds",
        label: "Blink below",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 30.0,
            step: 1.0,
            unit: " s",
            decimals: 0,
        },
        default: OptionValue::Float(10.0),
        description: "",
    },
    OptionDescriptor {
        id: "duration_color",
        label: "Duration color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0xFFAA_AAAA),
        description: "",
    },
    OptionDescriptor {
        id: "hide_permanent",
        label: "Hide endless effects",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Hides effects shown as **:**, such as a lobby's Speed or Night Vision.",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    presets::BACKGROUND,
    presets::BACKGROUND_COLOR,
];

/// The free middle of the left edge: below the top-left readouts, above the chat.
static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "potion_status",
    name: "Status Effects",
    default_placement: HudPlacement::new(HudAnchor::CenterLeft, 2.0, 0.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "potion_status",
    name: "Status Effects",
    category: Category::Hud,
    description: "Lists your active potion effects and the time left on each.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
