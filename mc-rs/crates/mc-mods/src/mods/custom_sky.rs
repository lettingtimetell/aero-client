//! Custom Sky (id `custom_sky`): the resource pack's custom sky layers.
//!
//! Keep the
//! `id` stable (it is the settings-file key); the `category` line must name the mod's home in
//! `crate::menu_layout`, which decides where the menu lists it (a test keeps the two in step).
//!
//! The pack's `mcpatcher/sky/world0/skyN.properties` (or `optifine/sky/...`) layers are drawn
//! over the vanilla sky with the format's fade times, blend modes and rotation; the parser is
//! `mc_assets::custom_sky`, the renderer `mc_render::sky::custom`, the client glue
//! `mc-client/src/app/mods/w3_sky.rs`. Turning the mod on loads the sky's textures once (a short
//! pause with a big sky pack); turning it off frees them.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "smooth",
    label: "Smooth textures",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(false),
    description: "Blends the sky's pixels instead of drawing them sharp.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "custom_sky",
    name: "Custom Sky",
    category: Category::World,
    description: "Your resource pack's custom sky: its stars, sunsets and skyboxes. \
                  Turning it on loads the sky once.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
