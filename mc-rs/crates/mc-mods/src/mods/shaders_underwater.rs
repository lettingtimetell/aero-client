//! Underwater Shaders (id `shaders_underwater`): the world as if under the sea (caustics, light
//! shafts, deep-water color, a shimmering surface overhead), plus a better look when the eyes are
//! really under water.
//!
//! Keep the
//! `id` and every option id stable (they are settings-file keys). The three shader packs are
//! exclusive: turning one on turns the other two off (client glue:
//! `mc-client/src/app/mods/tw_shaders.rs`). The effect is
//! `mc_render::shaderpack::underwater`; the client half is
//! `mc-client/src/app/mods/tw_underwater.rs`.
//!
//! The `preset` choice is a starting point for the water's look: picking one writes its
//! [`WaterPreset`] values into the four settings it covers ([`PRESET_OPTIONS`]), and changing any
//! of those four afterwards switches the choice to `Custom` (the client glue does both).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The mod's id.
pub const ID: &str = "shaders_underwater";

/// `mode` choices. Index 0 is the default.
pub const MODE_CHOICES: [&str; 2] = ["Everywhere", "Only under water"];
/// `mode` index: the whole world looks under the sea.
pub const MODE_EVERYWHERE: usize = 0;
/// `mode` index: nothing changes until the eyes are in water.
pub const MODE_UNDER_WATER: usize = 1;

/// `preset` choices, in [`PRESETS`] order, then `Custom`.
pub const PRESET_CHOICES: [&str; 5] = ["Lagoon", "Ocean", "Deep sea", "Murky", "Custom"];
/// `preset` index of `Custom`: the four preset settings are the user's own.
pub const PRESET_CUSTOM: usize = 4;

/// `quality` choices, cheapest first.
pub const QUALITY_CHOICES: [&str; 3] = ["Low", "Medium", "High"];

/// One water look: the values a `preset` choice writes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterPreset {
    /// `water_color`, `0xFFRRGGBB`.
    pub water_color: u32,
    /// `clarity`, blocks.
    pub clarity: f32,
    /// `red_fade`, percent.
    pub red_fade: f32,
    /// `depth_darkening`, percent.
    pub depth_darkening: f32,
}

/// The settings a preset writes, in [`WaterPreset`] field order.
pub const PRESET_OPTIONS: [&str; 4] = ["water_color", "clarity", "red_fade", "depth_darkening"];

/// The presets, in [`PRESET_CHOICES`] order. Every preset except Murky keeps players on the next
/// island (30 to 60 blocks) clearly visible.
pub const PRESETS: [WaterPreset; 4] = [
    // Lagoon: bright turquoise, very clear. The default.
    WaterPreset {
        water_color: 0xFF2E_A9B4,
        clarity: 128.0,
        red_fade: 35.0,
        depth_darkening: 25.0,
    },
    // Ocean: open-sea blue.
    WaterPreset {
        water_color: 0xFF22_6FB0,
        clarity: 104.0,
        red_fade: 45.0,
        depth_darkening: 35.0,
    },
    // Deep sea: dark navy, heavy depth darkening.
    WaterPreset {
        water_color: 0xFF12_3B70,
        clarity: 80.0,
        red_fade: 55.0,
        depth_darkening: 60.0,
    },
    // Murky: green lake water, short sight lines (not for PvP).
    WaterPreset {
        water_color: 0xFF4A_7650,
        clarity: 44.0,
        red_fade: 50.0,
        depth_darkening: 45.0,
    },
];

/// The default preset index (Lagoon).
pub const DEFAULT_PRESET: usize = 0;

static OPTIONS: [OptionDescriptor; 21] = [
    // ---- General
    OptionDescriptor {
        id: "mode",
        label: "Mode",
        kind: OptionKind::Choice {
            choices: &MODE_CHOICES,
        },
        default: OptionValue::Choice(MODE_EVERYWHERE),
        description: "Everywhere makes the whole world look like it is under the sea. Only under \
                      water changes nothing until your eyes are in water.",
    },
    OptionDescriptor {
        id: "preset",
        label: "Water preset",
        kind: OptionKind::Choice {
            choices: &PRESET_CHOICES,
        },
        default: OptionValue::Choice(DEFAULT_PRESET),
        description: "Sets the water color, clarity, red fade and depth darkening together. \
                      Lagoon keeps players of every color easiest to see far away; Deep sea and \
                      Murky are moodier but hide dark players at range. Changing one of the four \
                      switches this to Custom.",
    },
    // ---- Water
    OptionDescriptor {
        id: "water_color",
        label: "Water color",
        kind: OptionKind::Color { alpha: false },
        default: OptionValue::Color(PRESETS[DEFAULT_PRESET].water_color),
        description: "The color far-away things fade into.",
    },
    OptionDescriptor {
        id: "clarity",
        label: "Clarity",
        kind: OptionKind::Slider {
            min: 32.0,
            max: 256.0,
            step: 4.0,
            unit: " blocks",
            decimals: 0,
        },
        default: OptionValue::Float(PRESETS[DEFAULT_PRESET].clarity),
        description: "How far you see before the water color takes over. At 80 blocks and up, \
                      players on the next island stay easy to see.",
    },
    OptionDescriptor {
        id: "clear_range",
        label: "Clear range",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 24.0,
            step: 1.0,
            unit: " blocks",
            decimals: 0,
        },
        default: OptionValue::Float(8.0),
        description: "No water haze at all this close to you, so fights stay crisp.",
    },
    OptionDescriptor {
        id: "red_fade",
        label: "Red fades first",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(PRESETS[DEFAULT_PRESET].red_fade),
        description: "How much faster red light fades with distance than green and blue, as in \
                      real water. Lower keeps red teams redder far away.",
    },
    OptionDescriptor {
        id: "depth_darkening",
        label: "Depth darkening",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(PRESETS[DEFAULT_PRESET].depth_darkening),
        description: "Darkens what is far below you, like deep water. Never turns anything black.",
    },
    OptionDescriptor {
        id: "surface_sky",
        label: "Water surface sky",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Looking up shows the shimmering underside of the sea, with the real sky \
                      through a bright window overhead.",
    },
    // ---- Light
    OptionDescriptor {
        id: "caustics",
        label: "Caustics",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Dancing patterns of light on the ground. Kept faint close to you and on \
                      walls, so they never crawl over players.",
    },
    OptionDescriptor {
        id: "caustic_strength",
        label: "Caustic strength",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(50.0),
        description: "How bright the caustic lines get.",
    },
    OptionDescriptor {
        id: "caustic_size",
        label: "Caustic size",
        kind: OptionKind::Slider {
            min: 0.5,
            max: 2.0,
            step: 0.1,
            unit: "x",
            decimals: 1,
        },
        default: OptionValue::Float(1.0),
        description: "How large the caustic pattern is.",
    },
    OptionDescriptor {
        id: "wave_speed",
        label: "Wave speed",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 200.0,
            step: 10.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How fast the caustics, light shafts and surface move. 0% freezes them.",
    },
    OptionDescriptor {
        id: "light_shafts",
        label: "Light shafts",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Rays of sunlight slanting down through the water.",
    },
    OptionDescriptor {
        id: "shaft_strength",
        label: "Shaft strength",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(40.0),
        description: "How bright the light shafts are.",
    },
    // ---- Effects
    OptionDescriptor {
        id: "marine_snow",
        label: "Marine snow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Tiny specks drifting slowly through the water. Sparse and faint by default.",
    },
    OptionDescriptor {
        id: "snow_amount",
        label: "Snow amount",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(30.0),
        description: "How many specks drift around you.",
    },
    OptionDescriptor {
        id: "refraction",
        label: "Refraction wobble",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description:
            "Makes the world ripple gently, as seen through water. Off by default because \
                      it moves what you aim at.",
    },
    OptionDescriptor {
        id: "refraction_strength",
        label: "Wobble strength",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(30.0),
        description: "How far the picture ripples.",
    },
    // ---- Under water (eyes really in water)
    OptionDescriptor {
        id: "underwater_clarity",
        label: "Clarity under water",
        kind: OptionKind::Slider {
            min: 16.0,
            max: 128.0,
            step: 4.0,
            unit: " blocks",
            decimals: 0,
        },
        default: OptionValue::Float(48.0),
        description: "How far you see when your eyes are really in water. Never less than \
                      vanilla, whatever the setting.",
    },
    OptionDescriptor {
        id: "vanilla_overlay",
        label: "Vanilla water overlay",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Also draws vanilla's blue texture over the screen when your eyes are in \
                      water. The pack's own look replaces it when off.",
    },
    // ---- Performance
    OptionDescriptor {
        id: "quality",
        label: "Quality",
        kind: OptionKind::Choice {
            choices: &QUALITY_CHOICES,
        },
        default: OptionValue::Choice(1),
        description: "How finely the light shafts are traced. Higher is smoother and costs more \
                      GPU time.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: ID,
    name: "Underwater Shaders",
    category: Category::Graphics,
    description: "Makes the world look like it is under the sea: dancing caustics, light shafts, \
                  deep-sea color and a shimmering surface overhead. Everywhere, or only when you \
                  are in water. Tuned for PvP: players, nametags and hits stay clear. One shader \
                  pack at a time; switches itself off while MSAA is on.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

/// The preset whose four values these are, if any.
#[must_use]
pub fn matching_preset(
    water_color: u32,
    clarity: f32,
    red_fade: f32,
    depth_darkening: f32,
) -> Option<usize> {
    PRESETS.iter().position(|preset| {
        preset.water_color | 0xFF00_0000 == water_color | 0xFF00_0000
            && (preset.clarity - clarity).abs() < 1e-3
            && (preset.red_fade - red_fade).abs() < 1e-3
            && (preset.depth_darkening - depth_darkening).abs() < 1e-3
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_option_id_is_unique_and_every_default_fits_its_kind() {
        for (i, option) in OPTIONS.iter().enumerate() {
            assert!(option.kind.accepts(&option.default), "{}", option.id);
            assert_eq!(
                option
                    .kind
                    .sanitize(option.default.clone(), &option.default),
                option.default,
                "{} default must survive sanitize",
                option.id
            );
            assert!(
                OPTIONS[..i].iter().all(|other| other.id != option.id),
                "duplicate id {}",
                option.id
            );
        }
    }

    #[test]
    fn the_defaults_are_the_default_preset() {
        let d = &DESCRIPTOR;
        let color = match d.option("water_color").unwrap().default {
            OptionValue::Color(c) => c,
            _ => unreachable!(),
        };
        let float = |id: &str| match d.option(id).unwrap().default {
            OptionValue::Float(v) => v,
            _ => unreachable!(),
        };
        assert_eq!(
            matching_preset(
                color,
                float("clarity"),
                float("red_fade"),
                float("depth_darkening")
            ),
            Some(DEFAULT_PRESET)
        );
        assert_eq!(PRESET_CHOICES.len(), PRESETS.len() + 1);
        assert_eq!(PRESET_CHOICES[PRESET_CUSTOM], "Custom");
    }

    #[test]
    fn every_preset_value_is_on_its_slider_grid() {
        for preset in PRESETS {
            for (id, value) in [
                ("clarity", preset.clarity),
                ("red_fade", preset.red_fade),
                ("depth_darkening", preset.depth_darkening),
            ] {
                let option = DESCRIPTOR.option(id).unwrap();
                let snapped = option
                    .kind
                    .sanitize(OptionValue::Float(value), &option.default);
                assert_eq!(snapped, OptionValue::Float(value), "{id} {value}");
            }
        }
    }

    #[test]
    fn the_pvp_defaults_hold() {
        let default_bool = |id: &str| match DESCRIPTOR.option(id).unwrap().default {
            OptionValue::Bool(b) => b,
            _ => unreachable!(),
        };
        assert!(!DESCRIPTOR.default_enabled, "a shader pack is opt-in");
        assert!(
            !default_bool("refraction"),
            "the wobble moves what you aim at"
        );
        assert!(!default_bool("vanilla_overlay"));
        assert!(
            PRESETS[DEFAULT_PRESET].clarity >= 80.0,
            "the next island stays visible"
        );
    }
}
