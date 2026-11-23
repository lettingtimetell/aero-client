//! Volumetric Clouds: 3D clouds with real lighting, in place of vanilla's clouds.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The renderer is `mc_render::sky::volumetric`; the client side is
//! `mc-client/src/app/mods/x11_clouds.rs`. The Aero title screen always draws these clouds,
//! whether or not this mod is on.
//!
//! [`HEIGHT`] "Above you" (the new default) keeps the cloud bases
//! [`DISTANCE_ABOVE`] blocks over the player, eased over several seconds so that jumps, bridging
//! and falls never move them while a new map or a big climb does; the layer can never reach the
//! eye. "World height" is the vanilla cloud height the mod used before. The title screen's clouds
//! are not affected.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The options, in menu order.
static OPTIONS: [OptionDescriptor; 5] = [
    OptionDescriptor {
        id: "quality",
        label: "Quality",
        // Must match `mc_render::sky::volumetric::CloudQuality::LABELS` (index order).
        kind: OptionKind::Choice {
            choices: &["Low", "Medium", "High"],
        },
        default: OptionValue::Choice(1),
        description: "High traces at twice the resolution: sharper, about three times the cost.",
    },
    OptionDescriptor {
        id: "coverage",
        label: "Coverage",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(50.0),
        description: "How much of the sky is cloudy. Rain adds more.",
    },
    OptionDescriptor {
        id: "wind",
        label: "Wind",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 300.0,
            step: 10.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How fast the clouds drift and change shape.",
    },
    OptionDescriptor {
        id: HEIGHT,
        label: "Height",
        kind: OptionKind::Choice { choices: HEIGHTS },
        default: OptionValue::Choice(0),
        description:
            "Above you: the clouds keep a set distance over your head, so jumps, bridging \
                      and falls never move them; a new map or a big climb does, slowly. World \
                      height: vanilla's fixed height (y 128).",
    },
    OptionDescriptor {
        id: DISTANCE_ABOVE,
        label: "Distance above you",
        kind: OptionKind::Slider {
            min: 20.0,
            max: 160.0,
            step: 5.0,
            unit: " blocks",
            decimals: 0,
        },
        default: OptionValue::Float(60.0),
        description: "How far over your head the cloud bases sit with Height at Above you.",
    },
];

/// "Height": the option id.
pub const HEIGHT: &str = "height";
/// The "Height" choices, in index order: 0 follows the player, 1 is vanilla's cloud height.
pub const HEIGHTS: &[&str] = &["Above you", "World height"];
/// "Distance above you": the option id.
pub const DISTANCE_ABOVE: &str = "distance_above";

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "volumetric_clouds",
    name: "Volumetric Clouds",
    category: Category::World,
    description: "3D clouds with real light and shadow. They replace vanilla's clouds, by \
                  default a set distance above you, and follow the Clouds video setting (Off \
                  hides them).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
