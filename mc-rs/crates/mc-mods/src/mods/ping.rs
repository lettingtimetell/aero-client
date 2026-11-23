//! Ping: Shows your latency to the server.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The number is your own tab-list latency (what
//! the server measured and broadcast); no extra connection, no extra packets.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

static OPTIONS: [OptionDescriptor; 6] = [
    OptionDescriptor {
        id: "color_by_ping",
        label: "Color by ping",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Green below 80 ms, yellow below 150, gold below 250, red above.",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    presets::BACKGROUND,
    presets::BACKGROUND_COLOR,
    presets::BRACKETS,
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "ping",
    name: "Ping",
    default_placement: HudPlacement::new(HudAnchor::TopLeft, 2.0, 38.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "ping",
    name: "Ping",
    category: Category::Hud,
    description: "Shows your latency to the server, as the tab list reports it.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
