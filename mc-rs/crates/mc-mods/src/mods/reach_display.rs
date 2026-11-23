//! Reach Display: Shows how far away your last hit landed.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Measured when your own melee attack is
//! sent: from your eyes to the point where the look ray entered the target's (0.1-expanded)
//! box, the pick vanilla itself made. Some servers' rules, large ones included, list distance
//! HUDs as not allowed, so it is off by default like every mod, and its description says so.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

static OPTIONS: [OptionDescriptor; 8] = [
    OptionDescriptor {
        id: "decimals",
        label: "Decimals",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 3.0,
            step: 1.0,
            unit: "",
            decimals: 0,
        },
        default: OptionValue::Float(2.0),
        description: "",
    },
    OptionDescriptor {
        id: "reset_seconds",
        label: "Reset after",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 10.0,
            step: 1.0,
            unit: " s",
            decimals: 0,
        },
        default: OptionValue::Float(3.0),
        description: "Back to 0 when you have not hit anything for this long.",
    },
    OptionDescriptor {
        id: "hide_when_idle",
        label: "Hide when idle",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Draw nothing instead of 0 between fights.",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    presets::BACKGROUND,
    presets::BACKGROUND_COLOR,
    presets::BRACKETS,
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "reach_display",
    name: "Reach Display",
    default_placement: HudPlacement::new(HudAnchor::TopLeft, 2.0, 56.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "reach_display",
    name: "Reach Display",
    category: Category::Hud,
    description: "Shows how far away your last hit landed. Distance HUDs may not be allowed \
                  on some servers, including large ones.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
