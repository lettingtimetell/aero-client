//! Cape Physics: waving, flexible capes with adjustable motion.
//!
//! The wave styles replaced the earlier motion modes. Keep the
//! module and `motion` option IDs stable so saved enable states and numeric choices
//! survive the replacement. The module remains off by default; off uses vanilla capes.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// Wave styles in the existing `motion` option's numeric index order.
pub const MOTIONS: [&str; 4] = ["Wavy", "Flowing", "Windy", "Draped"];

static OPTIONS: [OptionDescriptor; 4] = [
    OptionDescriptor {
        id: "motion",
        label: "Style",
        kind: OptionKind::Choice { choices: &MOTIONS },
        default: OptionValue::Choice(0),
        description: "Choose a cape style. Waves follow movement and settle when standing still.",
    },
    OptionDescriptor {
        id: "strength",
        label: "Wave strength",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(60.0),
        description: "How far the cape bends and ripples.",
    },
    OptionDescriptor {
        id: "speed",
        label: "Wave speed",
        kind: OptionKind::Slider {
            min: 25.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How quickly waves travel along the cape.",
    },
    OptionDescriptor {
        id: "stiffness",
        label: "Stiffness",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(50.0),
        description: "Higher values keep the cape firmer and reduce bending.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "cape_physics",
    name: "Cape Physics",
    category: Category::Players,
    description: "Capes that flow with movement and settle at rest. Off uses vanilla capes.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
