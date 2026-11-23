//! Zoom: Hold a key to zoom in.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Client half: `m3_controls.rs` (the FOV divisor,
//! its easing, the scroll wheel, the zoomed sensitivity and the smooth camera).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 6] = [
    OptionDescriptor {
        id: "key",
        label: "Zoom key",
        kind: OptionKind::Keybind,
        // LWJGL 46 = C.
        default: OptionValue::Key(46),
        description: "Hold to zoom.",
    },
    OptionDescriptor {
        id: "factor",
        label: "Zoom amount",
        kind: OptionKind::Slider {
            min: 2.0,
            max: 10.0,
            step: 0.5,
            unit: "x",
            decimals: 1,
        },
        default: OptionValue::Float(4.0),
        description: "How far the view zooms in. The default is 4x.",
    },
    OptionDescriptor {
        id: "smooth_zoom",
        label: "Smooth zoom",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Eases in and out over about a tenth of a second.",
    },
    OptionDescriptor {
        id: "scroll",
        label: "Scroll to adjust",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "While zoomed, the wheel zooms further in or out instead of changing slot.",
    },
    OptionDescriptor {
        id: "smooth_camera",
        label: "Smooth camera while zoomed",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Vanilla's cinematic camera while the key is held.",
    },
    OptionDescriptor {
        id: "sensitivity",
        label: "Sensitivity while zoomed",
        kind: OptionKind::Slider {
            min: 10.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "Mouse speed while zoomed, as a share of your normal sensitivity.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "zoom",
    name: "Zoom",
    category: Category::Controls,
    description: "Hold a key to zoom in.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
