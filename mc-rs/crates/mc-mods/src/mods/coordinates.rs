//! Coordinates: Shows your position and the direction you face.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Block coordinates by default (F3's
//! `Block:` line), an 8-way facing with the axes you move along, and hidden while the server
//! has set reduced debug info (F3 hides them then too).

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

static OPTIONS: [OptionDescriptor; 7] = [
    OptionDescriptor {
        id: "layout",
        label: "Layout",
        kind: OptionKind::Choice {
            choices: &["Vertical", "Horizontal"],
        },
        default: OptionValue::Choice(0),
        description: "One line per axis, or everything on one line.",
    },
    OptionDescriptor {
        id: "decimals",
        label: "Decimals",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 2.0,
            step: 1.0,
            unit: "",
            decimals: 0,
        },
        default: OptionValue::Float(0.0),
        description: "0 shows the block you stand in; 1 or 2 show your exact position.",
    },
    OptionDescriptor {
        id: "show_direction",
        label: "Direction",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The way you face, e.g. \"NE (+X -Z)\".",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    presets::BACKGROUND,
    presets::BACKGROUND_COLOR,
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "coordinates",
    name: "Coordinates",
    default_placement: HudPlacement::new(HudAnchor::TopLeft, 2.0, 74.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "coordinates",
    name: "Coordinates",
    category: Category::Hud,
    description: "Shows your position and the direction you face. Hidden while the server \
                  hides F3 coordinates (reduced debug info).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
