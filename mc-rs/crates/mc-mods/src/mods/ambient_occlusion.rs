//! Ambient Occlusion: screen-space shading in corners and crevices (GTAO).
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The effect is `mc_render::post::ao::AoPhase` (ground-truth-style horizon AO over the opaque
//! world, before particles, translucent terrain and the hand); the client glue is
//! `mc-client/src/app/mods/x8_ao.rs`. With the mod off the phase reports itself disabled and
//! owns no GPU memory, so the frame is exactly today's.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The `quality` choices, cheapest first. Index = `mc_render::post::ao::AoQuality` order.
pub const QUALITY_CHOICES: [&str; 4] = ["Low", "Medium", "High", "Ultra"];

static OPTIONS: [OptionDescriptor; 4] = [
    OptionDescriptor {
        id: "quality",
        label: "Quality",
        kind: OptionKind::Choice {
            choices: &QUALITY_CHOICES,
        },
        default: OptionValue::Choice(1),
        description: "How many directions and steps each pixel searches. Higher is smoother and \
                      costs more GPU time.",
    },
    OptionDescriptor {
        id: "radius",
        label: "Radius",
        kind: OptionKind::Slider {
            min: 0.5,
            max: 4.0,
            step: 0.25,
            unit: " blocks",
            decimals: 2,
        },
        default: OptionValue::Float(1.5),
        description: "How far, in blocks, a surface can shade its neighbors.",
    },
    OptionDescriptor {
        id: "strength",
        label: "Strength",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How dark the shading gets.",
    },
    OptionDescriptor {
        id: "half_resolution",
        label: "Half resolution",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Computes the shading at half the screen size and upscales it along edges. \
                      Much cheaper and looks almost the same.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "ambient_occlusion",
    name: "Ambient Occlusion",
    category: Category::Graphics,
    description: "Soft shading in corners, under ledges and around objects (screen-space GTAO). \
                  Shades the world only: never the sky, the HUD or your hand.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
