//! Action Bar: move, resize or hide the action bar text above the hotbar.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The action bar is vanilla's "overlay message" (`GuiIngame.recordPlaying`): chat packets with
//! position 2 and the jukebox's "Now playing" line, with vanilla's 60-tick fade. While the mod is
//! on, the vanilla draw is skipped and the `action_bar` HUD element paints the same text with the
//! same colour and fade. Position and scale come from the HUD editor; the default placement
//! reproduces vanilla's pixels, so turning the mod on changes nothing until an option or the
//! editor does.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};

static OPTIONS: [OptionDescriptor; 2] = [
    OptionDescriptor {
        id: "hide",
        label: "Hide action bar",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Never draw the text above the hotbar.",
    },
    OptionDescriptor {
        id: "text_shadow",
        label: "Text shadow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Vanilla draws it without a shadow.",
    },
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "action_bar",
    name: "Action Bar",
    // Vanilla: the text's top is at `height - 72`, centred on `width / 2`. The element is 9
    // pixels tall (8 glyph rows plus the shadow row), so its bottom sits 63 pixels up.
    default_placement: HudPlacement::new(HudAnchor::BottomCenter, 0.0, -63.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "action_bar",
    name: "Action Bar",
    category: Category::Hud,
    description: "Hide the action bar text, add a shadow, or drag and scale it in Edit HUD.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
