//! Block Overlay: Restyles the outline of the block you are looking at.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 4] = [
    OptionDescriptor {
        id: "outline_color",
        label: "Outline color",
        kind: OptionKind::Color { alpha: true },
        // Black at 60% (vanilla's line is black at 40%).
        default: OptionValue::Color(0x9900_0000),
        description: "Alpha 0 hides the outline.",
    },
    OptionDescriptor {
        id: "outline_width",
        label: "Outline width",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 8.0,
            step: 0.5,
            unit: " px",
            decimals: 1,
        },
        default: OptionValue::Float(3.0),
        description: "In screen pixels, like vanilla's line width.",
    },
    OptionDescriptor {
        id: "fill_color",
        label: "Fill color",
        kind: OptionKind::Color { alpha: true },
        // Black at 15%.
        default: OptionValue::Color(0x2600_0000),
        description: "Alpha 0 means no fill.",
    },
    // Read by `mc-client/src/app/mods/x4_input.rs`.
    OptionDescriptor {
        id: "adventure_outline",
        label: "Show in adventure mode",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Outline every block in adventure mode. Vanilla outlines only blocks the \
                      held item may break or be placed on (CanDestroy / CanPlaceOn).",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "block_overlay",
    name: "Block Overlay",
    category: Category::World,
    description: "Restyles the outline of the block you are looking at.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
