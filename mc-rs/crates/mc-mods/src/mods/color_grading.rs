//! Color Grading: slope, offset, power and saturation, applied as the ASC CDL (v1.2) defines
//! them.
//!
//! The menu row is plain "Color Grading", without "(ASC CDL)"; the maths below is still ASC CDL
//! v1.2.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The maths is ASC CDL v1.2 (`mc_render::post::motion_grade::CdlParams`), run on the
//! display-encoded world frame before the HUD, in the pass it shares with `motion_blur`. The 10
//! CDL numbers are the per-channel `slope_*`, `offset_*`, `power_*` and `saturation`; the three
//! `master_*` sliders scale (slope, power) or shift (offset) all three channels at once:
//! `slope = slope_c * master_slope`, `offset = offset_c + master_offset`,
//! `power = power_c * master_power`. Every default is the identity, and at the identity the
//! pass is skipped entirely, so enabling the mod costs nothing until a slider moves.
//!
//! The menu has no section labels, so each label names its group ("Master Slope",
//! "Red Slope", ..., the names color graders use). Ranges are narrower than the
//! standard's on purpose: a grade outside them is unplayable (`research/color_grading.md`).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// Slope and saturation: a multiplier, 1 = unchanged.
const GAIN: OptionKind = OptionKind::Slider {
    min: 0.0,
    max: 2.0,
    step: 0.01,
    unit: "",
    decimals: 2,
};
/// Offset: added after slope, in full-scale units (0.2 = 51/255).
const OFFSET: OptionKind = OptionKind::Slider {
    min: -0.2,
    max: 0.2,
    step: 0.005,
    unit: "",
    decimals: 3,
};
/// Power: the exponent, 1 = unchanged, above 1 darkens the mids.
const POWER: OptionKind = OptionKind::Slider {
    min: 0.5,
    max: 2.0,
    step: 0.01,
    unit: "",
    decimals: 2,
};

const fn slider(
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

/// Masters first (what most people move), then the 10 CDL numbers grouped by parameter.
static OPTIONS: [OptionDescriptor; 13] = [
    slider(
        "master_slope",
        "Master slope",
        GAIN,
        1.0,
        "Scales R, G and B (contrast).",
    ),
    slider(
        "master_offset",
        "Master offset",
        OFFSET,
        0.0,
        "Lifts or lowers R, G and B.",
    ),
    slider(
        "master_power",
        "Master power",
        POWER,
        1.0,
        "Above 1 darkens the mids.",
    ),
    slider(
        "saturation",
        "Saturation",
        GAIN,
        1.0,
        "0 is gray, 1 unchanged, 2 double.",
    ),
    slider("slope_r", "Red slope", GAIN, 1.0, ""),
    slider("slope_g", "Green slope", GAIN, 1.0, ""),
    slider("slope_b", "Blue slope", GAIN, 1.0, ""),
    slider("offset_r", "Red offset", OFFSET, 0.0, ""),
    slider("offset_g", "Green offset", OFFSET, 0.0, ""),
    slider("offset_b", "Blue offset", OFFSET, 0.0, ""),
    slider("power_r", "Red power", POWER, 1.0, ""),
    slider("power_g", "Green power", POWER, 1.0, ""),
    slider("power_b", "Blue power", POWER, 1.0, ""),
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "color_grading",
    name: "Color Grading",
    category: Category::Graphics,
    description: "Slope, offset, power and saturation on the world. The HUD is not graded. \
                  Does nothing while MSAA is on (msaaSamples above 1 in options.txt).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    /// "(ASC CDL)" is gone from the Color Grading tab: the row name and the description it shows
    /// when expanded.
    #[test]
    fn the_menu_row_is_plain_color_grading() {
        assert_eq!(DESCRIPTOR.name, "Color Grading");
        for text in [DESCRIPTOR.name, DESCRIPTOR.description] {
            assert!(!text.contains("ASC") && !text.contains("CDL"), "{text}");
        }
        assert_eq!(
            DESCRIPTOR.id, "color_grading",
            "the settings key never changes"
        );
    }

    /// Every slider rests on the identity, so turning the mod on changes nothing (and costs
    /// nothing: the client skips the pass at the identity).
    #[test]
    fn every_default_is_the_identity() {
        for option in DESCRIPTOR.options {
            let OptionValue::Float(value) = option.default else {
                panic!("{} is not a slider", option.id);
            };
            let identity = if option.id.contains("offset") {
                0.0
            } else {
                1.0
            };
            assert_eq!(value, identity, "{}", option.id);
            // The settings store snaps every value; the identity must survive that exactly.
            let snapped = option
                .kind
                .sanitize(option.default.clone(), &option.default);
            assert_eq!(
                snapped,
                OptionValue::Float(identity),
                "{} does not snap onto the identity",
                option.id
            );
        }
    }
}
