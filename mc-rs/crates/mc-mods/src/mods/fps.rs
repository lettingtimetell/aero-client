//! FPS: your frame rate on the HUD, in frames per second or the frame time in ms.
//!
//! The reference HUD mod. Keep the `id` stable (it is the settings-file key); the `category` line
//! must name its home in `crate::menu_layout`, which decides where the menu lists it (a test
//! keeps the two in step).
//!
//! `display` picks `240 FPS`, `0.80 ms` (the mean
//! frame time over the same one-second window, `1000 / fps`) or `240 FPS (4.17 ms)`.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

/// What to show, then the shared HUD text style.
static OPTIONS: [OptionDescriptor; 6] = [
    OptionDescriptor {
        id: "display",
        label: "Show",
        kind: OptionKind::Choice {
            choices: &["FPS", "ms/frame", "Both"],
        },
        default: OptionValue::Choice(0),
        description: "Frames per second, the mean frame time over the same second, or both.",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    presets::BACKGROUND,
    presets::BACKGROUND_COLOR,
    presets::BRACKETS,
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "fps",
    name: "FPS",
    default_placement: HudPlacement::new(HudAnchor::TopLeft, 2.0, 2.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "fps",
    name: "FPS",
    category: Category::Hud,
    description: "Your frame rate on the HUD, in frames per second or milliseconds per frame.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
