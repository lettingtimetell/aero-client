//! Stuck Arrows: arrows stuck in players.
//!
//! Keep
//! the `id` stable (it is the settings-file key); the `category` line must name the mod's home in
//! `crate::menu_layout`, which decides where the menu lists it (a test keeps the two in step).
//!
//! The switch stands for a vanilla feature, so it is named after it and defaults **on**:
//! `LayerArrow` draws `EntityLivingBase.getArrowCountInEntity()` arrows in every player model
//! (watched index 9). Off hides every stuck arrow.
//!
//! "Only invisible players" selects the invisible-player subset. With that option off, arrows
//! appear only on visible players; on, arrows appear only on invisible players. Turning the
//! module off hides both. The client half is `crates/mc-client/src/app/mods/x6_entities.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "on_invisible",
    label: "Only invisible players",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(false),
    description:
        "Show stuck arrows only on invisible players. Off shows them only on visible players.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "stuck_arrows",
    name: "Stuck Arrows",
    category: Category::Pvp,
    description: "Arrows stuck in players. Turn off to hide every stuck arrow.",
    default_enabled: true,
    options: &OPTIONS,
    hud_elements: &[],
};
