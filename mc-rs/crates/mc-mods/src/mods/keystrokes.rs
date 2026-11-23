//! Keystrokes: Shows your movement keys and mouse buttons as you press them.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! W/A/S/D, LMB/RMB (with the `cps` counter
//! inside) and the space bar, lit from the key **bindings**: a press shows at once, a release
//! fades back over `fade_ms`.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

static OPTIONS: [OptionDescriptor; 9] = [
    OptionDescriptor {
        id: "show_mouse",
        label: "Mouse buttons",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "show_cps",
        label: "CPS in mouse buttons",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Clicks per second under LMB and RMB (the CPS mod's counter).",
    },
    OptionDescriptor {
        id: "show_space",
        label: "Space bar",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "fade_ms",
        label: "Release fade",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 250.0,
            step: 5.0,
            unit: " ms",
            decimals: 0,
        },
        default: OptionValue::Float(75.0),
        description: "How long a released key takes to fade back. Presses always show at once.",
    },
    OptionDescriptor {
        id: "pressed_color",
        label: "Pressed key color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0x6FFF_FFFF),
        description: "",
    },
    OptionDescriptor {
        id: "pressed_text_color",
        label: "Pressed text color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0xFF00_0000),
        description: "",
    },
    // The shared style ids, with this mod's own labels and defaults.
    OptionDescriptor {
        id: presets::TEXT_COLOR_ID,
        label: "Text color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0xFFFF_FFFF),
        description: "",
    },
    OptionDescriptor {
        id: presets::TEXT_SHADOW_ID,
        label: "Text shadow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "",
    },
    OptionDescriptor {
        id: presets::BACKGROUND_COLOR_ID,
        label: "Key color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0x6F00_0000),
        description: "The color of a key that is not pressed.",
    },
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "keystrokes",
    name: "Keystrokes",
    default_placement: HudPlacement::new(HudAnchor::TopRight, -2.0, 2.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "keystrokes",
    name: "Keystrokes",
    category: Category::Hud,
    description: "Shows your movement keys and mouse buttons as you press them.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
