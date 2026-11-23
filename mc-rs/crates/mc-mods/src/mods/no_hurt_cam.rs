//! No Hurt Cam: Stops the camera shaking when you take damage.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Scales the roll of
//! `EntityRenderer.hurtCameraEffect` on the world camera and the held item alike. The death
//! roll stays vanilla. Client half: `m3_controls.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "intensity",
    label: "Shake intensity",
    kind: OptionKind::Slider {
        min: 0.0,
        max: 100.0,
        step: 5.0,
        unit: "%",
        decimals: 0,
    },
    default: OptionValue::Float(0.0),
    description: "0% removes the shake; 100% is vanilla.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "no_hurt_cam",
    name: "No Hurt Cam",
    category: Category::Pvp,
    description: "Stops the camera shaking when you take damage.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
