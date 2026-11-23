//! Classic Shaders (id `shaders_classic`). Classic: the traditional shader-pack look (sun shadows, warm light, water reflections, bloom, light rays), tuned so players stay easy to see.
//!
//! Keep the `id` stable (it is the settings-file key). The three shader packs are exclusive:
//! turning one on turns the other two off (client glue: `mc-client/src/app/mods/tw_shaders.rs`).
//!
//! The render half is `mc_render::shaderpack` (the core: shadows, sun shading, water, waving
//! plants, bloom, tone mapping) plus the light rays. Every option id below is a settings-file
//! key: never rename one. The defaults are the PvP tuning: a strong look (shadows, warm sun,
//! reflective water) that never makes a player hard to see (shade stays bright, no fog, no
//! vignette, bloom too low to blind, subtle waving).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The `shadow_quality` choices: shadow-map resolution.
pub const SHADOW_QUALITY_CHOICES: [&str; 3] = ["Low (1024)", "Medium (2048)", "High (4096)"];
/// The shadow-map edge for each [`SHADOW_QUALITY_CHOICES`] index.
pub const SHADOW_RESOLUTIONS: [u32; 3] = [1024, 2048, 4096];
/// The `tone_curve` choices. Index = `mc_render::shaderpack::ToneCurve` order (Filmic, ACES,
/// None = linear).
pub const TONE_CURVE_CHOICES: [&str; 3] = ["Filmic", "ACES", "None"];

const fn percent(min: f32, max: f32, step: f32) -> OptionKind {
    OptionKind::Slider {
        min,
        max,
        step,
        unit: "%",
        decimals: 0,
    }
}

static OPTIONS: [OptionDescriptor; 30] = [
    // ---- Lighting ----
    OptionDescriptor {
        id: "sun_shading",
        label: "Sun shading",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Lights each side of a block by where the sun is, warm in sunlight and cool \
                      in shade, instead of vanilla's fixed shading.",
    },
    OptionDescriptor {
        id: "sun_strength",
        label: "Sun strength",
        kind: percent(0.0, 100.0, 5.0),
        default: OptionValue::Float(70.0),
        description:
            "How much of daylight comes straight from the sun. Higher means brighter sunlit \
                      sides and deeper shade.",
    },
    OptionDescriptor {
        id: "shade_brightness",
        label: "Shade brightness",
        kind: percent(20.0, 100.0, 5.0),
        default: OptionValue::Float(40.0),
        description:
            "The darkest shaded walls and shadows get, compared with sunlight. Players are \
                      never darkened by shade.",
    },
    OptionDescriptor {
        id: "sun_path_tilt",
        label: "Sun path tilt",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 40.0,
            step: 5.0,
            unit: "°",
            decimals: 0,
        },
        default: OptionValue::Float(25.0),
        description: "Tilts the sun's path toward the south, so the midday sun casts shadows with \
                      a direction instead of straight down. 0° is vanilla's path.",
    },
    OptionDescriptor {
        id: "sunset_warmth",
        label: "Sunset warmth",
        kind: percent(0.0, 200.0, 10.0),
        default: OptionValue::Float(100.0),
        description: "How golden the light turns at sunrise and sunset.",
    },
    OptionDescriptor {
        id: "moonlight_tint",
        label: "Moonlight tint",
        kind: percent(0.0, 200.0, 10.0),
        default: OptionValue::Float(100.0),
        description: "How blue moonlight looks at night.",
    },
    // ---- Shadows ----
    OptionDescriptor {
        id: "shadows",
        label: "Shadows",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Blocks cast shadows from the sun and moon. Shadows dim only sunlight, so \
                      torches and lava still light shaded places.",
    },
    OptionDescriptor {
        id: "shadow_distance",
        label: "Shadow distance",
        kind: OptionKind::Slider {
            min: 32.0,
            max: 192.0,
            step: 16.0,
            unit: " blocks",
            decimals: 0,
        },
        default: OptionValue::Float(96.0),
        description: "How far around you shadows are drawn. Shorter is sharper and cheaper.",
    },
    OptionDescriptor {
        id: "shadow_quality",
        label: "Shadow quality",
        kind: OptionKind::Choice {
            choices: &SHADOW_QUALITY_CHOICES,
        },
        default: OptionValue::Choice(1),
        description: "The shadow map's resolution. Higher gives crisper edges and uses more GPU \
                      memory.",
    },
    OptionDescriptor {
        id: "shadow_softness",
        label: "Shadow softness",
        kind: percent(0.0, 100.0, 5.0),
        default: OptionValue::Float(25.0),
        description: "How soft shadow edges are. 0% is hard-edged.",
    },
    OptionDescriptor {
        id: "shadow_strength",
        label: "Shadow strength",
        kind: percent(0.0, 100.0, 5.0),
        default: OptionValue::Float(85.0),
        description: "How dark cast shadows are, up to the shade brightness.",
    },
    OptionDescriptor {
        id: "plant_shadows",
        label: "Leaf and plant shadows",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Leaves, grass, flowers and glass panes cast shadows too.",
    },
    OptionDescriptor {
        id: "player_shadows",
        label: "Player shadows",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Players, mobs and dropped items cast shadows on the ground. They are never \
                      darkened by shadows themselves.",
    },
    // ---- Water ----
    OptionDescriptor {
        id: "water",
        label: "Shader water",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Water gets moving waves, sky and sun reflections, and is clearer near the \
                      surface.",
    },
    OptionDescriptor {
        id: "water_reflections",
        label: "Reflections",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Water reflects the world around it, not just the sky.",
    },
    OptionDescriptor {
        id: "water_waves",
        label: "Waves",
        kind: percent(0.0, 200.0, 10.0),
        default: OptionValue::Float(100.0),
        description: "How choppy the water surface looks. 0% is a flat mirror.",
    },
    OptionDescriptor {
        id: "water_clarity",
        label: "Clarity",
        kind: percent(0.0, 100.0, 5.0),
        default: OptionValue::Float(45.0),
        description: "How see-through shallow water is.",
    },
    // ---- Plants ----
    OptionDescriptor {
        id: "waving_plants",
        label: "Waving plants",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Grass, flowers and crops sway gently in the wind.",
    },
    OptionDescriptor {
        id: "waving_leaves",
        label: "Waving leaves",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Leaves and vines sway gently in the wind.",
    },
    OptionDescriptor {
        id: "waving_strength",
        label: "Wind strength",
        kind: percent(0.0, 200.0, 10.0),
        default: OptionValue::Float(100.0),
        description: "How far plants and leaves sway. The default is subtle on purpose.",
    },
    // ---- Light ----
    OptionDescriptor {
        id: "light_rays",
        label: "Light rays",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Soft rays of light around the sun when it is on screen.",
    },
    OptionDescriptor {
        id: "light_rays_strength",
        label: "Light ray strength",
        kind: percent(0.0, 100.0, 5.0),
        default: OptionValue::Float(45.0),
        description: "How bright the light rays are.",
    },
    OptionDescriptor {
        id: "sun_glow",
        label: "Sun glow",
        kind: percent(0.0, 100.0, 5.0),
        default: OptionValue::Float(50.0),
        description: "A soft halo around the sun and moon, and a warm haze on the horizon at \
                      sunrise and sunset. 0% turns it off.",
    },
    OptionDescriptor {
        id: "bloom",
        label: "Bloom",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Bright things like the sun, lava and glowstone glow softly.",
    },
    OptionDescriptor {
        id: "bloom_strength",
        label: "Bloom strength",
        kind: percent(0.0, 100.0, 5.0),
        default: OptionValue::Float(20.0),
        description: "How strong the glow is. Kept low by default so it never blinds you.",
    },
    // ---- Color ----
    OptionDescriptor {
        id: "tone_curve",
        label: "Tone curve",
        kind: OptionKind::Choice {
            choices: &TONE_CURVE_CHOICES,
        },
        default: OptionValue::Choice(0),
        description: "The film-style curve applied to the final image. Filmic is soft and \
                      natural, ACES punchier.",
    },
    OptionDescriptor {
        id: "exposure",
        label: "Exposure",
        kind: OptionKind::Slider {
            min: -2.0,
            max: 2.0,
            step: 0.1,
            unit: " EV",
            decimals: 1,
        },
        default: OptionValue::Float(0.0),
        description: "Brightens or darkens the whole image, in camera stops.",
    },
    OptionDescriptor {
        id: "contrast",
        label: "Contrast",
        kind: percent(50.0, 150.0, 5.0),
        default: OptionValue::Float(115.0),
        description: "The difference between dark and bright areas. 100% is unchanged.",
    },
    OptionDescriptor {
        id: "saturation",
        label: "Saturation",
        kind: percent(0.0, 200.0, 5.0),
        default: OptionValue::Float(115.0),
        description: "How colorful the image is. 100% is unchanged.",
    },
    OptionDescriptor {
        id: "vibrance",
        label: "Vibrance",
        kind: percent(-100.0, 100.0, 5.0),
        default: OptionValue::Float(25.0),
        description: "Makes dull colors more colorful while leaving already vivid ones alone, \
                      which keeps team colors clear.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "shaders_classic",
    name: "Classic Shaders",
    category: Category::Graphics,
    description: "A traditional shader pack: sun shadows, warm sunlight, reflective water, soft bloom and light rays, tuned for PvP so players stay clear. Only one shader pack is on at a time. Turns itself off while MSAA (Multisample) is above Off.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_default_matches_its_kind_and_ids_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for option in DESCRIPTOR.options {
            assert!(option.kind.accepts(&option.default), "{}", option.id);
            assert!(seen.insert(option.id), "duplicate id {}", option.id);
        }
        assert_eq!(SHADOW_QUALITY_CHOICES.len(), SHADOW_RESOLUTIONS.len());
    }
}
