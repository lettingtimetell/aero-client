//! Separate inventory/container and hotbar sizing without changing the global GUI scale.
use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// Default follows Video Settings; the other choices use vanilla's resolution limits.
pub const SCALES: &[&str] = &["Default", "Auto", "Small", "Normal", "Large"];

static OPTIONS: [OptionDescriptor; 2] = [
    OptionDescriptor {
        id: "inventory_scale",
        label: "Inventory / crafting scale",
        kind: OptionKind::Choice { choices: SCALES },
        default: OptionValue::Choice(0),
        description: "Size of inventory, crafting, creative and other container screens.",
    },
    OptionDescriptor {
        id: "hotbar_scale",
        label: "Hotbar scale",
        kind: OptionKind::Choice { choices: SCALES },
        default: OptionValue::Choice(0),
        description: "Size of the hotbar, health/food, XP and their captions. Other HUDs keep their own size.",
    },
];

/// The opt-in catalog entry. Disabling it restores the global GUI scale for both surfaces.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "independent_gui",
    name: "Independent GUI Scale",
    category: Category::Interface,
    description: "Set inventory and hotbar sizes independently of each other and the other menus.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
