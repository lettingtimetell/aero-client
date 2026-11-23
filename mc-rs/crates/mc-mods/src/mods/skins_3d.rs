//! 3D Skins: Draws the outer skin layer as 3D pixels.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The user's request (item 29): "3d skins". The voxel mesh is
//! `mc-render/src/entity/skin3d.rs`; the client half is `mc-client/src/app/mods/x5_cosmetics.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 6] = [
    OptionDescriptor {
        id: "distance",
        label: "Distance",
        kind: OptionKind::Slider {
            min: 4.0,
            max: 32.0,
            step: 1.0,
            unit: " blocks",
            decimals: 0,
        },
        default: OptionValue::Float(16.0),
        description: "Players closer than this get 3D layers; farther ones keep the flat layer.",
    },
    OptionDescriptor {
        id: "hat",
        label: "Hat",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Flat while the player wears a skull or a pumpkin.",
    },
    OptionDescriptor {
        id: "jacket",
        label: "Jacket",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "sleeves",
        label: "Sleeves",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "pants",
        label: "Pants",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "first_person",
        label: "First-person sleeve",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Your own arm's sleeve in first person.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "skins_3d",
    name: "3D Skins",
    category: Category::Players,
    description: "Draws the outer skin layer (hat, jacket, sleeves, pants) as 3D pixels. Armor \
                  still draws over it.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
