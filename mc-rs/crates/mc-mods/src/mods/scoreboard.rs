//! Scoreboard: restyle the sidebar scoreboard (red numbers, background, shadow), hide it, or
//! drag and scale it anywhere.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Position and scale come from the HUD
//! editor. The default placement reproduces vanilla's box exactly: right edge at `width - 1`,
//! and (through the element's own `resolve`) vanilla's vertical pivot two thirds down the box
//! at `height / 2`, so the board grows up by 6 and down by 3 pixels per row as vanilla's does.
//! "Hide red numbers" defaults to **on**.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};

static OPTIONS: [OptionDescriptor; 4] = [
    OptionDescriptor {
        id: "hide",
        label: "Hide scoreboard",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "",
    },
    OptionDescriptor {
        id: "hide_numbers",
        label: "Hide red numbers",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The sort keys on the right; most servers' are meaningless.",
    },
    OptionDescriptor {
        id: "background_opacity",
        label: "Background opacity",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "100% is vanilla's plates.",
    },
    OptionDescriptor {
        id: "text_shadow",
        label: "Text shadow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "",
    },
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "scoreboard",
    name: "Scoreboard",
    // Vanilla: right edge at `width - 1`, vertical position from the element's `resolve`.
    default_placement: HudPlacement::new(HudAnchor::CenterRight, -1.0, 0.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "scoreboard",
    name: "Scoreboard",
    category: Category::Hud,
    description: "Restyle or hide the sidebar, or drag and scale it in Edit HUD.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
