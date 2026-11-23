//! Motion Blur: frame feedback whose trail length is set in time, not in frames.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The **Frame blend** mode (the default, and the first mode)
//! is `mc_render::post::motion_grade::MotionGradePhase`, one fullscreen pass shared with
//! `color_grading`; the client glue is `mc-client/src/app/mods/m7_postfx.rs`. Every frame is
//! mixed with the previous blend using the weight `a = s^(120·dt)`, so a strength looks the same
//! at 60, 240 or 3000 FPS (`research/motion_blur.md`).
//!
//! The **Camera** and **Camera + objects** modes blur each pixel along its own motion
//! instead: `mc_render::post::velocity_blur::VelocityBlurPhase` over the motion target
//! (`mc_render::post::velocity`), glue in `mc-client/src/app/mods/x9_aa.rs`. The same Strength
//! sets the exposure `ln 2 / (120 · ln(1/s))` seconds, the half-life of Frame blend's trail at
//! that strength (8.3 ms at 50 %), and the trail is tent-weighted so it fades like Frame
//! blend's: a strength looks about as strong in every mode. Camera + objects adds each moving
//! entity's own motion.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The modes, in menu order. Index 0 (Frame blend) is today's behaviour and the default.
pub const MODES: [&str; 3] = ["Frame blend", "Camera", "Camera + objects"];

/// Index of Frame blend in [`MODES`].
pub const MODE_FRAME_BLEND: usize = 0;

/// Index of Camera in [`MODES`].
pub const MODE_CAMERA: usize = 1;

/// Index of Camera + objects in [`MODES`].
pub const MODE_CAMERA_OBJECTS: usize = 2;

/// Mode, then strength. 95 % is the cap: past it the trail is a smear (half-life 113 ms).
static OPTIONS: [OptionDescriptor; 2] = [
    OptionDescriptor {
        id: "mode",
        label: "Mode",
        kind: OptionKind::Choice { choices: &MODES },
        default: OptionValue::Choice(MODE_FRAME_BLEND),
        description: "Frame blend: a fading trail of whole frames. Camera: each pixel blurs \
                      along how it moves as you look and walk; the hand stays sharp. \
                      Camera + objects: moving players and mobs blur along their own motion.",
    },
    OptionDescriptor {
        id: "strength",
        label: "Strength",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 95.0,
            step: 1.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(50.0),
        description: "Trail length. At 120 FPS, 50% keeps half of the previous frame; the trail \
                      looks the same at any frame rate.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "motion_blur",
    name: "Motion Blur",
    category: Category::Graphics,
    description: "Blurs motion: whole frames or per pixel (Mode). The HUD stays sharp. \
                  Does nothing while MSAA is on (msaaSamples above 1 in options.txt).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

/// The per-pixel modes' exposure in seconds for a strength fraction `s` (0.5 = 50 %): the
/// half-life of Frame blend's exponential trail at the same strength, `ln 2 / (120 · ln(1/s))`
/// (exactly one 120 Hz frame, 8.3 ms, at 50 %). The per-pixel trail is drawn with a tent
/// weighting, so it fades at its ends like Frame blend's does. 0 for a strength of 0 (no blur).
#[must_use]
pub fn exposure_seconds(strength: f32) -> f32 {
    if !(strength.is_finite() && strength > 0.0) {
        return 0.0;
    }
    let s = strength.min(0.95);
    std::f32::consts::LN_2 / (120.0 * (1.0 / s).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mode_indices_match_the_choices_and_frame_blend_stays_the_default() {
        assert_eq!(MODES[MODE_FRAME_BLEND], "Frame blend");
        assert_eq!(MODES[MODE_CAMERA], "Camera");
        assert_eq!(MODES[MODE_CAMERA_OBJECTS], "Camera + objects");
        assert_eq!(OPTIONS[0].default, OptionValue::Choice(MODE_FRAME_BLEND));
        assert_eq!(
            DESCRIPTOR.id, "motion_blur",
            "the settings key never changes"
        );
    }

    /// A settings file written before the Mode option existed (a real one has exactly this
    /// entry) loads unchanged and lands on Frame blend, today's behaviour.
    #[test]
    fn a_file_without_the_mode_keeps_its_strength_and_frame_blend() {
        let text = r#"{"format": 1, "mods": {"motion_blur": {"enabled": true,
                      "options": {"strength": 12.0}}}}"#;
        let (settings, _report) = crate::settings::ModSettings::from_json_str(text);
        assert!(settings.enabled("motion_blur"));
        assert_eq!(settings.f32("motion_blur", "strength"), 12.0);
        assert_eq!(settings.choice("motion_blur", "mode"), MODE_FRAME_BLEND);
    }

    #[test]
    fn the_exposure_is_frame_blends_half_life() {
        assert_eq!(exposure_seconds(0.0), 0.0);
        assert!(
            (exposure_seconds(0.5) - 1.0 / 120.0).abs() < 1e-6,
            "one 120 Hz frame"
        );
        assert!(exposure_seconds(0.95) > exposure_seconds(0.5));
        assert!(exposure_seconds(0.95) < 0.12);
    }
}
