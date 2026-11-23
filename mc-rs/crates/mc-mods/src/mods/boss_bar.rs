//! Boss Bar: hide the boss bar, strip it to the name or the bar, or drag and scale it anywhere.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Position and scale come from the HUD editor
//! (like every HUD element); the default placement (top centre, 2 px down) reproduces vanilla's
//! pixels exactly, so turning the mod on changes nothing until an option or the editor does.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};

/// Hide, then the two halves, then the name's shadow.
static OPTIONS: [OptionDescriptor; 4] = [
    OptionDescriptor {
        id: "hide",
        label: "Hide boss bar",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Never draws the boss bar. Servers often use it for ads.",
    },
    OptionDescriptor {
        id: "show_text",
        label: "Show name",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "show_bar",
        label: "Show health bar",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "text_shadow",
        label: "Text shadow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "boss_bar",
    name: "Boss Bar",
    // Vanilla: name at y = 2, bar at y = 12, centred on `width / 2`.
    default_placement: HudPlacement::new(HudAnchor::TopCenter, 0.0, 2.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "boss_bar",
    name: "Boss Bar",
    category: Category::Hud,
    description: "Hide the boss bar, or drag and scale it in Edit HUD.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
