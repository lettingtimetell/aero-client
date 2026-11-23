//! Dreamy Shaders (id `shaders_dreamy`): an atmospheric, moody look with deep color controls and
//! an optional ground fog.
//!
//! Keep the
//! `id` and every option id stable (they are settings-file keys). The three shader packs are
//! exclusive: turning one on turns the other two off (client glue: `app/mods/tw_shaders.rs`).
//! The render half is `mc_render::shaderpack::dreamy`, the glue `app/mods/tw_dreamy.rs`.
//!
//! # Presets
//!
//! [`LOOKS`] are the presets of the `preset` choice; [`LOOK_IDS`] are the options a preset sets.
//! Changing the preset writes its look into those options; moving any of them by hand switches
//! the preset to Custom ([`CUSTOM`]). The defaults below are the first look (Dreamy), so a fresh
//! install shows exactly that preset. Haze, ground fog, light rays and the lens effects are not
//! part of a look: they decide what you can see, so a preset never changes them.
//!
//! # Units
//!
//! Percent sliders store percent (35 = 35 %); the glue divides by 100. Hues are degrees, distances
//! blocks, exposure stops (EV).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The mod id.
pub const ID: &str = "shaders_dreamy";

/// Option ids, for the glue.
pub mod ids {
    /// The preset choice.
    pub const PRESET: &str = "preset";
    /// Exposure, EV.
    pub const EXPOSURE: &str = "exposure";
    /// Contrast, percent.
    pub const CONTRAST: &str = "contrast";
    /// Saturation, percent.
    pub const SATURATION: &str = "saturation";
    /// Vibrance, percent (-100..100).
    pub const VIBRANCE: &str = "vibrance";
    /// White-balance temperature, -100..100.
    pub const TEMPERATURE: &str = "temperature";
    /// White-balance tint, -100..100.
    pub const TINT: &str = "tint";
    /// Hue shift, degrees.
    pub const HUE_SHIFT: &str = "hue_shift";
    /// Shadows wheel hue, degrees.
    pub const SHADOWS_HUE: &str = "shadows_hue";
    /// Shadows wheel strength, percent.
    pub const SHADOWS_STRENGTH: &str = "shadows_strength";
    /// Midtones wheel hue, degrees.
    pub const MIDTONES_HUE: &str = "midtones_hue";
    /// Midtones wheel strength, percent.
    pub const MIDTONES_STRENGTH: &str = "midtones_strength";
    /// Highlights wheel hue, degrees.
    pub const HIGHLIGHTS_HUE: &str = "highlights_hue";
    /// Highlights wheel strength, percent.
    pub const HIGHLIGHTS_STRENGTH: &str = "highlights_strength";
    /// Split-toning shadow color.
    pub const SPLIT_SHADOWS: &str = "split_shadows";
    /// Split-toning highlight color.
    pub const SPLIT_HIGHLIGHTS: &str = "split_highlights";
    /// Split-toning balance, -100..100.
    pub const SPLIT_BALANCE: &str = "split_balance";
    /// Split-toning strength, percent.
    pub const SPLIT_STRENGTH: &str = "split_strength";
    /// Fade (lifted blacks), percent.
    pub const FADE: &str = "fade";
    /// Gradient map mix, percent.
    pub const GRADIENT_MIX: &str = "gradient_mix";
    /// Gradient map dark stop.
    pub const GRADIENT_DARK: &str = "gradient_dark";
    /// Gradient map middle stop.
    pub const GRADIENT_MID: &str = "gradient_mid";
    /// Gradient map light stop.
    pub const GRADIENT_LIGHT: &str = "gradient_light";
    /// Soft glow, percent.
    pub const GLOW: &str = "glow";
    /// Glow radius, percent.
    pub const GLOW_RADIUS: &str = "glow_radius";
    /// Halation, percent.
    pub const HALATION: &str = "halation";
    /// Sky tint color.
    pub const SKY_TINT: &str = "sky_tint";
    /// Sky tint strength, percent.
    pub const SKY_TINT_STRENGTH: &str = "sky_tint_strength";
    /// Distance haze, percent.
    pub const HAZE: &str = "haze";
    /// Haze start distance, blocks.
    pub const HAZE_START: &str = "haze_start";
    /// Ground fog switch.
    pub const GROUND_FOG: &str = "ground_fog";
    /// Ground fog height mode (Below you / World height).
    pub const FOG_HEIGHT_MODE: &str = "fog_height_mode";
    /// Blocks between your feet and the fog's top (Below you).
    pub const FOG_DEPTH: &str = "fog_depth";
    /// The fog's top as a world height (World height).
    pub const FOG_LEVEL: &str = "fog_level";
    /// Fog density, percent.
    pub const FOG_DENSITY: &str = "fog_density";
    /// Fog softness (falloff above the top), blocks.
    pub const FOG_SOFTNESS: &str = "fog_softness";
    /// Fog color mode (Follow the sky / Custom).
    pub const FOG_COLOR_MODE: &str = "fog_color_mode";
    /// Fog color (Custom).
    pub const FOG_COLOR: &str = "fog_color";
    /// How much the sun lights the fog, percent.
    pub const FOG_SUN_GLOW: &str = "fog_sun_glow";
    /// Fog drift, percent.
    pub const FOG_WIND: &str = "fog_wind";
    /// No fog nearer than this, blocks.
    pub const FOG_START: &str = "fog_start";
    /// Fog resolution (Half / Quarter).
    pub const FOG_QUALITY: &str = "fog_quality";
    /// Direction-aware sunlight (the shared core's sun shading).
    pub const SUNLIGHT: &str = "sunlight";
    /// Soft sun and moon shadows (the shared core's shadow map).
    pub const SOFT_SHADOWS: &str = "soft_shadows";
    /// Shadow strength, percent.
    pub const SHADOW_STRENGTH: &str = "shadow_strength";
    /// Light rays switch.
    pub const LIGHT_RAYS: &str = "light_rays";
    /// Light-ray strength, percent.
    pub const RAYS_STRENGTH: &str = "rays_strength";
    /// Light-ray length, percent.
    pub const RAYS_LENGTH: &str = "rays_length";
    /// Vignette, percent.
    pub const VIGNETTE: &str = "vignette";
    /// Film grain, percent.
    pub const GRAIN: &str = "grain";
    /// Chromatic aberration, percent.
    pub const CHROMATIC: &str = "chromatic_aberration";
}

/// The preset choice's labels, in order. The last is Custom.
pub const PRESETS: &[&str] = &[
    "Dreamy",
    "Dusk",
    "Pastel",
    "Golden hour",
    "Moonlight",
    "Neon noir",
    "Vaporwave",
    "Custom",
];

/// The index of Custom in [`PRESETS`].
pub const CUSTOM: usize = 7;

/// `fog_height_mode`: the fog's top follows your height, a fixed distance below your feet.
pub const FOG_BELOW_YOU: usize = 0;
/// `fog_height_mode`: the fog's top is a fixed world height.
pub const FOG_WORLD_HEIGHT: usize = 1;

/// `fog_color_mode`: the fog takes the sky's (vanilla fog) color.
pub const FOG_FOLLOW_SKY: usize = 0;

/// One preset: the value of every option in [`LOOK_IDS`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    /// Exposure, EV.
    pub exposure: f32,
    /// Contrast, percent.
    pub contrast: f32,
    /// Saturation, percent.
    pub saturation: f32,
    /// Vibrance, percent.
    pub vibrance: f32,
    /// Temperature, -100..100.
    pub temperature: f32,
    /// Tint, -100..100.
    pub tint: f32,
    /// Hue shift, degrees.
    pub hue_shift: f32,
    /// Shadows wheel: hue (degrees), strength (percent).
    pub shadows: (f32, f32),
    /// Midtones wheel.
    pub midtones: (f32, f32),
    /// Highlights wheel.
    pub highlights: (f32, f32),
    /// Split toning: shadow color (`0xAARRGGBB`).
    pub split_shadows: u32,
    /// Split toning: highlight color.
    pub split_highlights: u32,
    /// Split toning balance, -100..100.
    pub split_balance: f32,
    /// Split toning strength, percent.
    pub split_strength: f32,
    /// Fade, percent.
    pub fade: f32,
    /// Gradient map mix, percent.
    pub gradient_mix: f32,
    /// Gradient map stops: dark, mid, light.
    pub gradient: [u32; 3],
    /// Soft glow, percent.
    pub glow: f32,
    /// Glow radius, percent.
    pub glow_radius: f32,
    /// Halation, percent.
    pub halation: f32,
    /// Sky tint color.
    pub sky_tint: u32,
    /// Sky tint strength, percent.
    pub sky_tint_strength: f32,
}

/// The options a preset sets, in [`Look::values`] order.
pub const LOOK_IDS: [&str; 27] = [
    ids::EXPOSURE,
    ids::CONTRAST,
    ids::SATURATION,
    ids::VIBRANCE,
    ids::TEMPERATURE,
    ids::TINT,
    ids::HUE_SHIFT,
    ids::SHADOWS_HUE,
    ids::SHADOWS_STRENGTH,
    ids::MIDTONES_HUE,
    ids::MIDTONES_STRENGTH,
    ids::HIGHLIGHTS_HUE,
    ids::HIGHLIGHTS_STRENGTH,
    ids::SPLIT_SHADOWS,
    ids::SPLIT_HIGHLIGHTS,
    ids::SPLIT_BALANCE,
    ids::SPLIT_STRENGTH,
    ids::FADE,
    ids::GRADIENT_MIX,
    ids::GRADIENT_DARK,
    ids::GRADIENT_MID,
    ids::GRADIENT_LIGHT,
    ids::GLOW,
    ids::GLOW_RADIUS,
    ids::HALATION,
    ids::SKY_TINT,
    ids::SKY_TINT_STRENGTH,
];

impl Look {
    /// The look as option values, in [`LOOK_IDS`] order.
    #[must_use]
    pub fn values(&self) -> [OptionValue; 27] {
        use OptionValue::{Color, Float};
        [
            Float(self.exposure),
            Float(self.contrast),
            Float(self.saturation),
            Float(self.vibrance),
            Float(self.temperature),
            Float(self.tint),
            Float(self.hue_shift),
            Float(self.shadows.0),
            Float(self.shadows.1),
            Float(self.midtones.0),
            Float(self.midtones.1),
            Float(self.highlights.0),
            Float(self.highlights.1),
            Color(self.split_shadows),
            Color(self.split_highlights),
            Float(self.split_balance),
            Float(self.split_strength),
            Float(self.fade),
            Float(self.gradient_mix),
            Color(self.gradient[0]),
            Color(self.gradient[1]),
            Color(self.gradient[2]),
            Float(self.glow),
            Float(self.glow_radius),
            Float(self.halation),
            Color(self.sky_tint),
            Float(self.sky_tint_strength),
        ]
    }
}

/// The presets, in [`PRESETS`] order (Custom has no look).
pub static LOOKS: [Look; 7] = [
    DREAMY,
    // Dusk: warm orange highlights over indigo shadows, a little darker, a coral sky.
    Look {
        exposure: -0.15,
        contrast: 104.0,
        saturation: 102.0,
        vibrance: 12.0,
        temperature: 20.0,
        tint: 8.0,
        hue_shift: 0.0,
        shadows: (255.0, 18.0),
        midtones: (20.0, 6.0),
        highlights: (30.0, 15.0),
        split_shadows: 0xFF4A_3A7A,
        split_highlights: 0xFFFF_A060,
        split_balance: -5.0,
        split_strength: 26.0,
        fade: 5.0,
        gradient_mix: 0.0,
        gradient: [0xFF1B_1035, 0xFF9A_6A9E, 0xFFFF_E6C8],
        glow: 32.0,
        glow_radius: 70.0,
        halation: 25.0,
        sky_tint: 0xFFFF_8A6A,
        sky_tint_strength: 40.0,
    },
    // Pastel: light and soft, low contrast, muted but airy colors, powder blue and pink.
    Look {
        exposure: 0.15,
        contrast: 90.0,
        saturation: 88.0,
        vibrance: 30.0,
        temperature: 4.0,
        tint: 6.0,
        hue_shift: 0.0,
        shadows: (280.0, 10.0),
        midtones: (30.0, 0.0),
        highlights: (330.0, 8.0),
        split_shadows: 0xFF8F_A8D8,
        split_highlights: 0xFFFF_C0D8,
        split_balance: 10.0,
        split_strength: 20.0,
        fade: 14.0,
        gradient_mix: 0.0,
        gradient: [0xFF1B_1035, 0xFF9A_6A9E, 0xFFFF_E6C8],
        glow: 36.0,
        glow_radius: 70.0,
        halation: 8.0,
        sky_tint: 0xFFFF_C8E8,
        sky_tint_strength: 35.0,
    },
    // Golden hour: warm sunlight everywhere, cool shadows keep the contrast, a honey sky.
    Look {
        exposure: 0.1,
        contrast: 102.0,
        saturation: 110.0,
        vibrance: 20.0,
        temperature: 26.0,
        tint: 2.0,
        hue_shift: 0.0,
        shadows: (220.0, 8.0),
        midtones: (35.0, 8.0),
        highlights: (40.0, 18.0),
        split_shadows: 0xFF4A_5A8A,
        split_highlights: 0xFFFF_C070,
        split_balance: 15.0,
        split_strength: 24.0,
        fade: 3.0,
        gradient_mix: 0.0,
        gradient: [0xFF1B_1035, 0xFF9A_6A9E, 0xFFFF_E6C8],
        glow: 36.0,
        glow_radius: 60.0,
        halation: 25.0,
        sky_tint: 0xFFFF_C080,
        sky_tint_strength: 35.0,
    },
    // Moonlight: cool, quiet and a little darker; blue shadows, silver highlights.
    Look {
        exposure: -0.25,
        contrast: 104.0,
        saturation: 82.0,
        vibrance: 10.0,
        temperature: -24.0,
        tint: -3.0,
        hue_shift: 0.0,
        shadows: (225.0, 22.0),
        midtones: (215.0, 8.0),
        highlights: (200.0, 6.0),
        split_shadows: 0xFF1A_2A5A,
        split_highlights: 0xFFB8_D0FF,
        split_balance: -10.0,
        split_strength: 24.0,
        fade: 8.0,
        gradient_mix: 0.0,
        gradient: [0xFF1B_1035, 0xFF9A_6A9E, 0xFFFF_E6C8],
        glow: 26.0,
        glow_radius: 80.0,
        halation: 0.0,
        sky_tint: 0xFF7A_9AE0,
        sky_tint_strength: 40.0,
    },
    // Neon noir: punchy contrast, teal shadows and magenta highlights, deep blacks.
    Look {
        exposure: -0.1,
        contrast: 114.0,
        saturation: 110.0,
        vibrance: 25.0,
        temperature: -6.0,
        tint: 10.0,
        hue_shift: 0.0,
        shadows: (200.0, 20.0),
        midtones: (30.0, 0.0),
        highlights: (320.0, 14.0),
        split_shadows: 0xFF0A_6A8A,
        split_highlights: 0xFFFF_4AB0,
        split_balance: 0.0,
        split_strength: 24.0,
        fade: 0.0,
        gradient_mix: 0.0,
        gradient: [0xFF1B_1035, 0xFF9A_6A9E, 0xFFFF_E6C8],
        glow: 40.0,
        glow_radius: 50.0,
        halation: 15.0,
        sky_tint: 0xFFB0_4AFF,
        sky_tint_strength: 35.0,
    },
    // Vaporwave: purple shadows, cyan highlights, a pink-to-cyan gradient map and a pink sky.
    Look {
        exposure: 0.05,
        contrast: 96.0,
        saturation: 106.0,
        vibrance: 20.0,
        temperature: -4.0,
        tint: 14.0,
        hue_shift: 0.0,
        shadows: (270.0, 20.0),
        midtones: (300.0, 6.0),
        highlights: (185.0, 12.0),
        split_shadows: 0xFF6A_3AA0,
        split_highlights: 0xFF7A_E8F0,
        split_balance: 0.0,
        split_strength: 24.0,
        fade: 10.0,
        gradient_mix: 12.0,
        gradient: [0xFF2A_1050, 0xFFFF_5AB0, 0xFF8A_F0FF],
        glow: 38.0,
        glow_radius: 70.0,
        halation: 12.0,
        sky_tint: 0xFFFF_8AD8,
        sky_tint_strength: 45.0,
    },
];

/// Dreamy, the first preset and the defaults: warm peach highlights, soft indigo shadows, a
/// gentle glow and a pink-lilac sky. Tuned on the SkyWars scene: between a first version that
/// washed everything purple-grey and a second that was barely there.
pub const DREAMY: Look = Look {
    exposure: 0.05,
    contrast: 102.0,
    saturation: 108.0,
    vibrance: 18.0,
    temperature: 8.0,
    tint: 3.0,
    hue_shift: 0.0,
    shadows: (235.0, 14.0),
    midtones: (30.0, 0.0),
    highlights: (35.0, 12.0),
    split_shadows: 0xFF5F_5FB8,
    split_highlights: 0xFFFF_C2A0,
    split_balance: 5.0,
    split_strength: 18.0,
    fade: 5.0,
    gradient_mix: 0.0,
    gradient: [0xFF1B_1035, 0xFF9A_6A9E, 0xFFFF_E6C8],
    glow: 28.0,
    glow_radius: 60.0,
    halation: 15.0,
    sky_tint: 0xFFFF_B3D9,
    sky_tint_strength: 30.0,
};

/// The look a fresh install starts with.
const D: Look = DREAMY;

const fn slider(min: f32, max: f32, step: f32, unit: &'static str, decimals: u8) -> OptionKind {
    OptionKind::Slider {
        min,
        max,
        step,
        unit,
        decimals,
    }
}

const PERCENT: OptionKind = slider(0.0, 100.0, 1.0, "%", 0);
const SIGNED_PERCENT: OptionKind = slider(-100.0, 100.0, 1.0, "%", 0);
const BALANCE: OptionKind = slider(-100.0, 100.0, 1.0, "", 0);
const HUE: OptionKind = slider(0.0, 360.0, 1.0, "°", 0);
const COLOR: OptionKind = OptionKind::Color { alpha: false };

const fn float(
    id: &'static str,
    label: &'static str,
    kind: OptionKind,
    default: f32,
    description: &'static str,
) -> OptionDescriptor {
    OptionDescriptor {
        id,
        label,
        kind,
        default: OptionValue::Float(default),
        description,
    }
}

const fn color(
    id: &'static str,
    label: &'static str,
    default: u32,
    description: &'static str,
) -> OptionDescriptor {
    OptionDescriptor {
        id,
        label,
        kind: COLOR,
        default: OptionValue::Color(default),
        description,
    }
}

const fn toggle(
    id: &'static str,
    label: &'static str,
    default: bool,
    description: &'static str,
) -> OptionDescriptor {
    OptionDescriptor {
        id,
        label,
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(default),
        description,
    }
}

const fn choice(
    id: &'static str,
    label: &'static str,
    choices: &'static [&'static str],
    default: usize,
    description: &'static str,
) -> OptionDescriptor {
    OptionDescriptor {
        id,
        label,
        kind: OptionKind::Choice { choices },
        default: OptionValue::Choice(default),
        description,
    }
}

/// The options, in menu order.
static OPTIONS: [OptionDescriptor; 51] = [
    choice(
        ids::PRESET,
        "Preset",
        PRESETS,
        0,
        "A starting look. Picking one sets the color, glow and sky settings below; changing \
         any of them switches to Custom.",
    ),
    // Color.
    float(
        ids::EXPOSURE,
        "Exposure",
        slider(-2.0, 2.0, 0.05, " EV", 2),
        D.exposure,
        "Brightens or darkens the world, in camera stops.",
    ),
    float(
        ids::CONTRAST,
        "Contrast",
        slider(50.0, 150.0, 1.0, "%", 0),
        D.contrast,
        "Spreads light and dark apart (above 100%) or brings them together. Black stays black.",
    ),
    float(
        ids::SATURATION,
        "Saturation",
        slider(0.0, 200.0, 1.0, "%", 0),
        D.saturation,
        "How colorful everything is. 0% is black and white.",
    ),
    float(
        ids::VIBRANCE,
        "Vibrance",
        SIGNED_PERCENT,
        D.vibrance,
        "Boosts muted colors more than vivid ones, so skin and team colors don't oversaturate.",
    ),
    float(
        ids::TEMPERATURE,
        "Temperature",
        BALANCE,
        D.temperature,
        "Warms the white balance toward orange (+) or cools it toward blue (-).",
    ),
    float(
        ids::TINT,
        "Tint",
        BALANCE,
        D.tint,
        "Shifts the white balance toward magenta (+) or green (-).",
    ),
    float(
        ids::HUE_SHIFT,
        "Hue shift",
        slider(-180.0, 180.0, 1.0, "°", 0),
        D.hue_shift,
        "Rotates every color around the color wheel. Keep it near 0 so team colors stay true.",
    ),
    // Color wheels.
    float(
        ids::SHADOWS_HUE,
        "Shadows color",
        HUE,
        D.shadows.0,
        "The hue the dark parts of the image lean toward (lift).",
    ),
    float(
        ids::SHADOWS_STRENGTH,
        "Shadows strength",
        PERCENT,
        D.shadows.1,
        "How far the shadows lean toward their color.",
    ),
    float(
        ids::MIDTONES_HUE,
        "Midtones color",
        HUE,
        D.midtones.0,
        "The hue the middle tones lean toward (gamma).",
    ),
    float(
        ids::MIDTONES_STRENGTH,
        "Midtones strength",
        PERCENT,
        D.midtones.1,
        "How far the midtones lean toward their color.",
    ),
    float(
        ids::HIGHLIGHTS_HUE,
        "Highlights color",
        HUE,
        D.highlights.0,
        "The hue the bright parts of the image lean toward (gain).",
    ),
    float(
        ids::HIGHLIGHTS_STRENGTH,
        "Highlights strength",
        PERCENT,
        D.highlights.1,
        "How far the highlights lean toward their color.",
    ),
    // Split toning.
    color(
        ids::SPLIT_SHADOWS,
        "Shadow tone",
        D.split_shadows,
        "The color blended softly into the shadows.",
    ),
    color(
        ids::SPLIT_HIGHLIGHTS,
        "Highlight tone",
        D.split_highlights,
        "The color blended softly into the highlights.",
    ),
    float(
        ids::SPLIT_BALANCE,
        "Tone balance",
        BALANCE,
        D.split_balance,
        "Favors the shadow tone (-) or the highlight tone (+).",
    ),
    float(
        ids::SPLIT_STRENGTH,
        "Split toning",
        PERCENT,
        D.split_strength,
        "How strongly the two tones color the image. 0% is off.",
    ),
    // Fade and gradient map.
    float(
        ids::FADE,
        "Fade",
        PERCENT,
        D.fade,
        "Lifts the blacks to a soft, matte gray, like faded film.",
    ),
    float(
        ids::GRADIENT_MIX,
        "Gradient map",
        PERCENT,
        D.gradient_mix,
        "Recolors the image by brightness with the three colors below. 0% is off; high \
         values hide team colors.",
    ),
    color(
        ids::GRADIENT_DARK,
        "Gradient dark",
        D.gradient[0],
        "The gradient map's color for dark areas.",
    ),
    color(
        ids::GRADIENT_MID,
        "Gradient middle",
        D.gradient[1],
        "The gradient map's color for middle tones.",
    ),
    color(
        ids::GRADIENT_LIGHT,
        "Gradient light",
        D.gradient[2],
        "The gradient map's color for bright areas.",
    ),
    // Glow and sky.
    float(
        ids::GLOW,
        "Soft glow",
        PERCENT,
        D.glow,
        "A dreamy glow around bright areas, laid over the sharp image so edges stay crisp.",
    ),
    float(
        ids::GLOW_RADIUS,
        "Glow radius",
        PERCENT,
        D.glow_radius,
        "How far the glow spreads.",
    ),
    float(
        ids::HALATION,
        "Halation",
        PERCENT,
        D.halation,
        "A warm film-like fringe around the brightest highlights.",
    ),
    color(
        ids::SKY_TINT,
        "Sky tint",
        D.sky_tint,
        "The color the sky is tinted toward, strongest near the horizon.",
    ),
    float(
        ids::SKY_TINT_STRENGTH,
        "Sky tint strength",
        PERCENT,
        D.sky_tint_strength,
        "How strongly the sky takes the tint. 0% is off.",
    ),
    float(
        ids::HAZE,
        "Haze",
        PERCENT,
        30.0,
        "Soft atmospheric haze that grows with distance and glows toward the sun.",
    ),
    float(
        ids::HAZE_START,
        "Haze start",
        slider(0.0, 96.0, 1.0, " blocks", 0),
        32.0,
        "No haze closer than this, so fights stay clear.",
    ),
    // Ground fog.
    toggle(
        ids::GROUND_FOG,
        "Ground fog",
        true,
        "A layer of fog that pools in low places: below the islands in SkyWars, in valleys \
         elsewhere.",
    ),
    choice(
        ids::FOG_HEIGHT_MODE,
        "Fog height",
        &["Below you", "World height"],
        FOG_BELOW_YOU,
        "Below you keeps the fog's top a set distance under your feet (it follows you slowly), \
         so players around you stay clear. World height puts it at a fixed height.",
    ),
    float(
        ids::FOG_DEPTH,
        "Distance below you",
        slider(2.0, 40.0, 1.0, " blocks", 0),
        14.0,
        "How far under your feet the fog's top sits.",
    ),
    float(
        ids::FOG_LEVEL,
        "Fog top height",
        slider(0.0, 256.0, 1.0, "", 0),
        63.0,
        "The world height (y) of the fog's top.",
    ),
    float(
        ids::FOG_DENSITY,
        "Fog density",
        PERCENT,
        40.0,
        "How thick the fog is inside the layer.",
    ),
    float(
        ids::FOG_SOFTNESS,
        "Fog softness",
        slider(1.0, 16.0, 0.5, " blocks", 1),
        6.0,
        "How many blocks the fog's top fades over. Nothing above that is ever fogged.",
    ),
    choice(
        ids::FOG_COLOR_MODE,
        "Fog color",
        &["Follow the sky", "Custom"],
        FOG_FOLLOW_SKY,
        "Follow the sky matches the fog to the horizon at every time of day.",
    ),
    color(
        ids::FOG_COLOR,
        "Custom fog color",
        0xFFC8_C0E8,
        "The fog's color when Fog color is Custom.",
    ),
    float(
        ids::FOG_SUN_GLOW,
        "Fog sun glow",
        PERCENT,
        50.0,
        "How brightly the sun or moon lights the fog toward it.",
    ),
    float(
        ids::FOG_WIND,
        "Fog drift",
        PERCENT,
        20.0,
        "How fast the fog drifts and swirls. 0% is still.",
    ),
    float(
        ids::FOG_START,
        "Fog start",
        slider(0.0, 64.0, 1.0, " blocks", 0),
        16.0,
        "No fog closer than this.",
    ),
    choice(
        ids::FOG_QUALITY,
        "Fog quality",
        &["Sharp", "Fast"],
        0,
        "Sharp traces the fog at half resolution, Fast at quarter resolution.",
    ),
    // Lighting (the shared shader core's sun shading and shadow map, tuned lighter than Classic).
    toggle(
        ids::SUNLIGHT,
        "Sunlight",
        true,
        "Warm, direction-aware sunlight and cool skylight in place of the flat vanilla shading, \
         golden at sunrise and sunset. Players keep their vanilla lighting.",
    ),
    toggle(
        ids::SOFT_SHADOWS,
        "Soft shadows",
        true,
        "Soft shadows from the sun and moon, lighter than the Classic pack's so players in \
         shade stay easy to see. Torches and lava still light shaded places.",
    ),
    float(
        ids::SHADOW_STRENGTH,
        "Shadow strength",
        PERCENT,
        35.0,
        "How dark the shadows are.",
    ),
    // Light rays.
    toggle(
        ids::LIGHT_RAYS,
        "Light rays",
        true,
        "Shafts of sunlight (or moonlight) streaming past blocks and trees.",
    ),
    float(
        ids::RAYS_STRENGTH,
        "Light ray strength",
        PERCENT,
        25.0,
        "How bright the light rays are.",
    ),
    float(
        ids::RAYS_LENGTH,
        "Light ray length",
        PERCENT,
        60.0,
        "How far the rays reach across the screen.",
    ),
    // Lens.
    float(
        ids::VIGNETTE,
        "Vignette",
        PERCENT,
        15.0,
        "Darkens the corners of the screen.",
    ),
    float(
        ids::GRAIN,
        "Film grain",
        PERCENT,
        10.0,
        "A fine moving film grain.",
    ),
    float(
        ids::CHROMATIC,
        "Chromatic aberration",
        PERCENT,
        0.0,
        "Splits red and blue toward the screen's edges, like a cheap lens. Off by default: it \
         blurs distant players.",
    ),
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: ID,
    name: "Dreamy Shaders",
    category: Category::Graphics,
    description: "An atmospheric, dreamy shader pack: rich color grading with presets, a soft \
                  glow, haze, light rays and an optional ground fog, tuned for PvP so nearby \
                  players stay clear. Only one shader pack is on at a time. Does nothing while \
                  MSAA is on (msaaSamples above 1 in options.txt).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh install shows exactly the first preset: every look option's default is the
    /// first look's value.
    #[test]
    fn the_defaults_are_the_first_preset() {
        assert_eq!(LOOKS[0], DREAMY);
        let values = LOOKS[0].values();
        for (id, value) in LOOK_IDS.iter().zip(values) {
            let option = DESCRIPTOR
                .option(id)
                .unwrap_or_else(|| panic!("{id} missing"));
            assert_eq!(option.default, value, "{id}");
        }
        assert_eq!(
            DESCRIPTOR.option(ids::PRESET).map(|o| &o.default),
            Some(&OptionValue::Choice(0))
        );
    }

    /// Every preset value is exactly what the settings store keeps (on its slider's step and in
    /// range), so applying a preset never reads back as a hand edit.
    #[test]
    fn every_preset_value_survives_the_store() {
        assert_eq!(PRESETS.len(), LOOKS.len() + 1);
        assert_eq!(PRESETS[CUSTOM], "Custom");
        for (index, look) in LOOKS.iter().enumerate() {
            for (id, value) in LOOK_IDS.iter().zip(look.values()) {
                let option = DESCRIPTOR.option(id).expect("look option");
                assert!(option.kind.accepts(&value), "{} {id}", PRESETS[index]);
                let kept = option.kind.sanitize(value.clone(), &option.default);
                assert_eq!(kept, value, "{} {id}", PRESETS[index]);
            }
        }
    }

    #[test]
    fn option_ids_are_unique_and_every_default_matches_its_kind() {
        let mut seen = std::collections::HashSet::new();
        for option in DESCRIPTOR.options {
            assert!(seen.insert(option.id), "duplicate {}", option.id);
            assert!(option.kind.accepts(&option.default), "{}", option.id);
            let kept = option
                .kind
                .sanitize(option.default.clone(), &option.default);
            assert_eq!(
                kept, option.default,
                "{} default is off its step",
                option.id
            );
        }
    }

    /// PvP defaults: no chromatic aberration, a light vignette, haze and fog that start past
    /// fighting range, no gradient map, no hue shift.
    #[test]
    fn the_defaults_are_pvp_safe() {
        let float = |id| match DESCRIPTOR.option(id).map(|o| &o.default) {
            Some(OptionValue::Float(v)) => *v,
            other => panic!("{id}: {other:?}"),
        };
        assert_eq!(float(ids::CHROMATIC), 0.0);
        assert!(float(ids::VIGNETTE) <= 20.0);
        assert!(float(ids::HAZE_START) >= 24.0);
        assert!(float(ids::FOG_START) >= 12.0);
        assert!(float(ids::FOG_DEPTH) >= 8.0);
        for look in &LOOKS {
            assert_eq!(look.hue_shift, 0.0);
            assert!(look.gradient_mix <= 20.0);
        }
    }

    /// US spelling and sentence case in every label (the menu's style rule).
    #[test]
    fn labels_use_us_spelling() {
        for option in DESCRIPTOR.options {
            for text in [option.label, option.description] {
                assert!(!text.contains("colour"), "{}: {text}", option.id);
            }
        }
        assert!(!DESCRIPTOR.description.contains("colour"));
    }
}
