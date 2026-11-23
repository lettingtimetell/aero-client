//! Titles: move, resize or hide the big title and subtitle text.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Vanilla draws the S45 title at 4x and the subtitle at 2x the chat font, centred on the screen,
//! both with a shadow and one shared fade. While the mod is on, the vanilla draw is skipped and the
//! `titles` HUD element paints both lines with the same fade. Each line has its own size (100 % is
//! vanilla); the HUD editor moves the pair and scales it as a whole. The default placement
//! reproduces vanilla's pixels.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};

static OPTIONS: [OptionDescriptor; 4] = [
    OptionDescriptor {
        id: "hide",
        label: "Hide titles",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Never draw titles or subtitles.",
    },
    OptionDescriptor {
        id: "title_scale",
        label: "Title size",
        kind: OptionKind::Slider {
            min: 25.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How big the title is. 100% is vanilla (4x the chat font).",
    },
    OptionDescriptor {
        id: "subtitle_scale",
        label: "Subtitle size",
        kind: OptionKind::Slider {
            min: 25.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How big the subtitle is. 100% is vanilla (2x the chat font).",
    },
    OptionDescriptor {
        id: "text_shadow",
        label: "Text shadow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "titles",
    name: "Titles",
    // Vanilla: the title's top is 40 pixels above the screen centre and the subtitle's shadow
    // ends 28 below it, so the 68-pixel box is centred 6 pixels above the screen centre.
    default_placement: HudPlacement::new(HudAnchor::Center, 0.0, -6.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "titles",
    name: "Titles",
    category: Category::Hud,
    description: "Resize or hide the big title and subtitle, or drag them in Edit HUD.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
