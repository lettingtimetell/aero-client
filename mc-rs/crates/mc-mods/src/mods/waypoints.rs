//! Waypoints: Mark places in the world and see them from anywhere.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The waypoints themselves are data, not settings: they live in [`store`] and in their own
//! file, `<gameDir>/mc-rs-waypoints.json`, so `mc-rs-mods.json` stays small and a waypoint list
//! never rides on the settings writer. The menu lists them through dynamic rows
//! (`mc_mods::dynamic`), filled by the client.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

pub mod store;

/// LWJGL `KEY_B` creates a waypoint: vanilla 1.8.9 leaves B unbound.
pub const DEFAULT_CREATE_KEY: i32 = 48;

static OPTIONS: [OptionDescriptor; 5] = [
    OptionDescriptor {
        id: "key_create",
        label: "Create waypoint",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(DEFAULT_CREATE_KEY),
        description: "Marks the block at your feet. Ignored while F3 is held (F3+B).",
    },
    OptionDescriptor {
        id: "death_waypoint",
        label: "Death waypoint",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Marks where you died. One per server: a new death replaces it.",
    },
    OptionDescriptor {
        id: "show_beam",
        label: "Beacon beam",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "A beam in the waypoint's color, hidden within 4 blocks.",
    },
    OptionDescriptor {
        id: "show_distance",
        label: "Show distance",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: "label_scale",
        label: "Label size",
        kind: OptionKind::Slider {
            min: 50.0,
            max: 200.0,
            step: 10.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "waypoints",
    name: "Waypoints",
    category: Category::Players,
    description: "Mark places in the world and see them from anywhere.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
