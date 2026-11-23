//! Shiny Pots: potions in the inventory glint over a whole square, as in 1.7.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// `area` choice index: the 16x16 item icon.
pub const AREA_ITEM_BOX: usize = 0;
/// `area` choice index: 1.7's 20x20 square, two pixels past the icon on every side.
pub const AREA_WHOLE_SLOT: usize = 1;

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "area",
    label: "Glow area",
    kind: OptionKind::Choice {
        choices: &["Item box", "Whole slot"],
    },
    default: OptionValue::Choice(AREA_ITEM_BOX),
    description: "Item box covers the 16x16 icon; Whole slot is 1.7's 20x20 square.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "shiny_pots",
    name: "Shiny Pots",
    category: Category::Pvp,
    description: "Potions in your inventory glint over a whole square, the way 1.7 drew them.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
