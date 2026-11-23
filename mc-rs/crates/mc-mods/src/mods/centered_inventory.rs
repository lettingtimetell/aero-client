//! Centered Inventory: keeps the survival and creative inventories centred when you have
//! potion effects (vanilla shifts them right to make room for the effect list).
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The effect list stays where
//! vanilla draws it relative to the inventory (`guiLeft - 124`), i.e. immediately left of the
//! centred inventory; `show_effects` off hides it.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "show_effects",
    label: "Show effects beside inventory",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(true),
    description: "Off hides the effect panels (e.g. with the Status Effects HUD on).",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "centered_inventory",
    name: "Centered Inventory",
    category: Category::Interface,
    description: "Keeps the inventory centered when you have potion effects.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
