//! Connected Textures: Joins the textures of glass and other connecting blocks.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Connected textures for the
//! resource pack's `mcpatcher/ctm` rules (`method=ctm`): glass reads as one framed sheet.
//! Turning the mod on re-stitches the block atlas once (a resource reload); turning it off or
//! changing the mode only rebuilds the chunk meshes.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// Mode choices, in index order (the format's Fancy and Fast).
pub const MODE_CHOICES: [&str; 2] = ["Fancy", "Fast"];

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "mode",
    label: "Mode",
    kind: OptionKind::Choice {
        choices: &MODE_CHOICES,
    },
    default: OptionValue::Choice(0),
    description: "Fancy also checks corners. Fast skips them.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "connected_textures",
    name: "Connected Textures",
    category: Category::World,
    description: "Seamless glass from your resource pack's connected textures. \
                  Turning it on reloads resources once.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
