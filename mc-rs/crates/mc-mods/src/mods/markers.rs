//! Markers: press a key to mark the spot you look at, for a few minutes (item 28 of the
//! 2026-11-09 list).
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The markers themselves are runtime state, not settings: [`set::MarkerSet`] holds them and
//! implements the removal rules. The client half is `app/mods/x13_markers.rs` (aiming, sound,
//! drawing); the label geometry is `mc_ui::marker_label`. Purely local: nothing is sent.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

pub mod set;

/// LWJGL `KEY_Z`: unbound in vanilla 1.8.9 and by every other mod here, and next to WASD.
/// (Middle mouse, an obvious choice for a ping, is vanilla's pick block.)
pub const DEFAULT_PLACE_KEY: i32 = 44;

/// The default colour: vanilla's yellow, `§e`.
pub const DEFAULT_COLOR: u32 = 0xFFFF_FF55;

static OPTIONS: [OptionDescriptor; 8] = [
    OptionDescriptor {
        id: "key",
        label: "Place marker",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(DEFAULT_PLACE_KEY),
        description: "Marks the block, player or mob you look at, up to 256 blocks away. \
                      Press it on a marker to remove that marker.",
    },
    OptionDescriptor {
        id: "clear_key",
        label: "Clear markers",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "Removes every marker.",
    },
    OptionDescriptor {
        id: "duration",
        label: "Remove after",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 60.0,
            step: 1.0,
            unit: " min",
            decimals: 0,
        },
        default: OptionValue::Float(5.0),
        description: "Each marker disappears this long after you place it.",
    },
    OptionDescriptor {
        id: "teleport_distance",
        label: "Teleport distance",
        kind: OptionKind::Slider {
            min: 25.0,
            max: 1000.0,
            step: 25.0,
            unit: " blocks",
            decimals: 0,
        },
        default: OptionValue::Float(200.0),
        description: "A teleport further than this clears every marker. A new world or \
                      dimension always does.",
    },
    OptionDescriptor {
        id: "max_markers",
        label: "Max markers",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 10.0,
            step: 1.0,
            unit: "",
            decimals: 0,
        },
        default: OptionValue::Float(3.0),
        description: "Placing one more replaces the oldest.",
    },
    OptionDescriptor {
        id: "color",
        label: "Color",
        kind: OptionKind::Color { alpha: false },
        default: OptionValue::Color(DEFAULT_COLOR),
        description: "",
    },
    OptionDescriptor {
        id: "sound",
        label: "Sound",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "A short ping that only you hear when you place or remove a marker.",
    },
    OptionDescriptor {
        id: "size",
        label: "Marker size",
        kind: OptionKind::Slider {
            min: 50.0,
            max: 200.0,
            step: 10.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "It stays this size on screen at any distance.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "markers",
    name: "Markers",
    category: Category::Players,
    description: "Press a key to mark the spot you look at. Only you see it: through walls, \
                  with its distance, for a few minutes.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
