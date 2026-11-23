//! Anti-Aliasing: TAA or FXAA over the world.
//!
//! Keep the `id` stable (it is the settings-file key); the `category`
//! line must name the mod's home in `crate::menu_layout`, which decides where the menu lists it
//! (a test keeps the two in step).
//!
//! The effect is `mc_render::post::taa::AntiAliasPhase` (with the motion target
//! `mc_render::post::velocity::VelocityPhase` for TAA), recorded after the world and its fog and
//! before the waypoint labels, the hand and the HUD; the client glue is
//! `mc-client/src/app/mods/x9_aa.rs`. Off (the mod disabled) costs nothing: no pass, no texture,
//! no camera jitter.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The modes, in menu order. Index 0 is the default the mod starts in when switched on.
pub const MODES: [&str; 2] = ["TAA", "FXAA"];

/// Index of TAA in [`MODES`].
pub const MODE_TAA: usize = 0;

/// Index of FXAA in [`MODES`].
pub const MODE_FXAA: usize = 1;

static OPTIONS: [OptionDescriptor; 3] = [
    OptionDescriptor {
        id: "mode",
        label: "Mode",
        kind: OptionKind::Choice { choices: &MODES },
        default: OptionValue::Choice(MODE_TAA),
        description: "TAA: the smoothest, steadiest edges, a little softer in motion. \
                      FXAA: the cheapest, no trails, textures stay crisp.",
    },
    OptionDescriptor {
        id: "sharpness",
        label: "TAA sharpness",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(25.0),
        description: "Wins back the pixel-art detail TAA softens. Never adds halos.",
    },
    OptionDescriptor {
        id: "texture_edges",
        label: "FXAA texture edges",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Also smooth edges inside textures (classic FXAA). Off smooths only real \
                      block and entity edges and keeps textures pixel-sharp.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "anti_aliasing",
    name: "Anti-Aliasing",
    category: Category::Graphics,
    description: "Smooths jagged edges in the world. The HUD, hand and labels stay sharp. \
                  Does nothing while MSAA is on (msaaSamples above 1 in options.txt).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mode_indices_match_the_choices() {
        assert_eq!(MODES[MODE_TAA], "TAA");
        assert_eq!(MODES[MODE_FXAA], "FXAA");
        assert_eq!(
            DESCRIPTOR.id, "anti_aliasing",
            "the settings key never changes"
        );
        assert!(!DESCRIPTOR.default_enabled);
    }
}
