//! CPS: Shows your clicks per second.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Presses of the attack (and use) binding that
//! reach the game, in a rolling one-second window; `keystrokes` reads the same counter.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

static OPTIONS: [OptionDescriptor; 6] = [
    OptionDescriptor {
        id: "show_right",
        label: "Right-click CPS",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Also count right clicks: \"12 | 4 CPS\".",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    presets::BACKGROUND,
    presets::BACKGROUND_COLOR,
    presets::BRACKETS,
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "cps",
    name: "CPS",
    default_placement: HudPlacement::new(HudAnchor::TopLeft, 2.0, 20.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "cps",
    name: "CPS",
    category: Category::Hud,
    description: "Shows how many times you clicked in the last second.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
