//! Anisotropic Filtering: keeps block textures sharp at grazing angles.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The effect lives in `mc_render::terrain::aniso` (the block atlas only); the client glue is
//! `mc-client/src/app/mods/x8_ao.rs`. Off draws terrain with today's pipelines and sampler.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The `level` choices.
pub const LEVEL_CHOICES: [&str; 4] = ["2x", "4x", "8x", "16x"];

/// The maximum anisotropy for each [`LEVEL_CHOICES`] index.
pub const LEVELS: [u32; 4] = [2, 4, 8, 16];

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "level",
    label: "Level",
    kind: OptionKind::Choice {
        choices: &LEVEL_CHOICES,
    },
    default: OptionValue::Choice(3),
    description: "The most texture samples taken along a stretched pixel. 16x is the sharpest.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "anisotropic_filtering",
    name: "Anisotropic Filtering",
    category: Category::Graphics,
    description: "Sharper block textures on floors and walls seen at a shallow angle. Close-up \
                  pixels stay crisp. Needs Mipmap Levels above 0 (Video Settings).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
