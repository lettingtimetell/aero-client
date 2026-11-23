//! Tab List: Edit the player list: header, footer and ping.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Everything is applied inside `mc_ui::screens::tab_list::TabListOverlay::draw` through its
//! `TabListStyle`; with the mod off the style is vanilla's and the draw is unchanged.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The `ping` choices, in order. Index 0 is vanilla's five-bar icon.
pub const PING_CHOICES: &[&str] = &["Bars", "Numbers", "Hidden"];

static OPTIONS: [OptionDescriptor; 6] = [
    OptionDescriptor {
        id: "header",
        label: "Show header",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The server's text above the names.",
    },
    OptionDescriptor {
        id: "footer",
        label: "Show footer",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The server's text below the names.",
    },
    OptionDescriptor {
        id: "ping",
        label: "Ping",
        kind: OptionKind::Choice {
            choices: PING_CHOICES,
        },
        default: OptionValue::Choice(1),
        description: "Bars (vanilla), the latency in ms colored by quality, or nothing.",
    },
    OptionDescriptor {
        id: "background_color",
        label: "Background color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0x8000_0000),
        description: "Color and opacity of the header, footer and outer panel.",
    },
    OptionDescriptor {
        id: "row_color",
        label: "Row color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0x20FF_FFFF),
        description: "Color and opacity behind each player row.",
    },
    OptionDescriptor {
        id: "background_opacity",
        label: "Overall opacity",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "Opacity of the dark panel and row plates. 100% is vanilla.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "tab_list",
    name: "Tab List",
    category: Category::Interface,
    description: "Hide the header or footer, show ping as numbers or hide it.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
