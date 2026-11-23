//! Fire Height: Lowers the fire overlay while you are burning.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 3] = [
    OptionDescriptor {
        id: "height",
        label: "Height",
        // Added to the two flame quads' Y translation (vanilla's -0.3). Each 0.1 moves the
        // flames by about 14% of the screen height.
        kind: OptionKind::Slider {
            min: -0.5,
            max: 0.3,
            step: 0.01,
            unit: "",
            decimals: 2,
        },
        default: OptionValue::Float(-0.3),
        description: "Negative lowers the flames. 0 is vanilla.",
    },
    OptionDescriptor {
        id: "opacity",
        label: "Opacity",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "100% is vanilla. 0% hides the overlay.",
    },
    OptionDescriptor {
        id: "hide_fire_resistance",
        label: "Hide with Fire Resistance",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Hidden while Fire Resistance has more than 5 seconds left.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "fire_height",
    name: "Fire Height",
    category: Category::Pvp,
    description: "Lowers the fire overlay while you are burning.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
