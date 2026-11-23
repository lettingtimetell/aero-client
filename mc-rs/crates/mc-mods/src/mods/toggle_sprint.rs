//! Toggle Sprint: Keeps sprinting (and optionally sneaking) without holding the key.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! A tap of the vanilla Sprint key flips
//! the toggle; while it is on, the movement input reads the key as held, so the packets are
//! exactly a player's who holds it. The client half is `mc-client/src/app/mods/m3_controls.rs`.

use std::borrow::Cow;

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};
use crate::presets;

/// Id of the hidden option that persists the toggle (`"on"` / `"off"`).
pub const SPRINT_TOGGLED_ID: &str = "sprint_toggled";

/// Behaviour, then the status text's style (the shared preset ids, so `HudTextStyle::read`
/// builds the style), then the persisted toggle state.
static OPTIONS: [OptionDescriptor; 8] = [
    OptionDescriptor {
        id: "toggle_sneak",
        label: "Toggle sneak too",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "A tap of the sneak key toggles sneaking. Opening any screen stops it.",
    },
    OptionDescriptor {
        id: "show_text",
        label: "Show status text",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Shows [Sprinting (Toggled)] and similar on the HUD.",
    },
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    // The preset ids with this mod's own defaults: plain text in brackets.
    OptionDescriptor {
        id: presets::BACKGROUND_ID,
        label: "Background",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "",
    },
    presets::BACKGROUND_COLOR,
    OptionDescriptor {
        id: presets::BRACKETS_ID,
        label: "Brackets",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    // Not drawn: whether sprint was toggled on, so it survives a restart.
    OptionDescriptor {
        id: SPRINT_TOGGLED_ID,
        label: "Sprint toggled",
        kind: OptionKind::HiddenText { max_len: 3 },
        default: OptionValue::Text(Cow::Borrowed("off")),
        description: "",
    },
];

static ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "toggle_sprint",
    name: "Toggle Sprint",
    default_placement: HudPlacement::new(HudAnchor::TopLeft, 2.0, 130.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "toggle_sprint",
    name: "Toggle Sprint",
    category: Category::Controls,
    description: "Tap Sprint once and you keep sprinting, as if the key were held. Tap it again \
                  to stop.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &ELEMENTS,
};
