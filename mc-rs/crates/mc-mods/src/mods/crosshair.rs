//! Crosshair: Changes the crosshair's shape, size and colour.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// `color_mode` choice index: white rectangles drawn with vanilla's inverting blend.
pub const COLOR_MODE_INVERTED: usize = 0;
/// `color_mode` choice index: rectangles in the `color` option, alpha blended.
pub const COLOR_MODE_CUSTOM: usize = 1;

/// Sizes are GUI pixels, so the crosshair follows the GUI scale like the vanilla sprite.
static OPTIONS: [OptionDescriptor; 7] = [
    OptionDescriptor {
        id: "length",
        label: "Arm length",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 10.0,
            step: 1.0,
            unit: " px",
            decimals: 0,
        },
        default: OptionValue::Float(4.0),
        description: "How long each arm is. 0 draws no arms (only the dot, with Center dot on).",
    },
    OptionDescriptor {
        id: "gap",
        label: "Gap",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 8.0,
            step: 1.0,
            unit: " px",
            decimals: 0,
        },
        default: OptionValue::Float(2.0),
        description: "Space between the center and each arm.",
    },
    OptionDescriptor {
        id: "thickness",
        label: "Thickness",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 4.0,
            step: 1.0,
            unit: " px",
            decimals: 0,
        },
        default: OptionValue::Float(1.0),
        description: "",
    },
    OptionDescriptor {
        id: "dot",
        label: "Center dot",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "",
    },
    OptionDescriptor {
        id: "color_mode",
        label: "Color",
        // Persisted by label: files saved as "Custom colour" still load (settings.rs reads a
        // choice label in either spelling).
        kind: OptionKind::Choice {
            choices: &["Inverted", "Custom color"],
        },
        default: OptionValue::Choice(COLOR_MODE_INVERTED),
        description: "Inverted flips the colors behind it, like vanilla. Custom color uses the \
                      color below.",
    },
    OptionDescriptor {
        id: "color",
        label: "Custom color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0xFFFF_FFFF),
        description: "The crosshair's color and opacity.",
    },
    OptionDescriptor {
        id: "show_in_f5",
        label: "Show in third person",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "crosshair",
    name: "Crosshair",
    category: Category::Hud,
    description: "Changes the crosshair's shape, size and color.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The choice is persisted by its label, and files saved
    /// before the wave say "Custom colour". They must still load as Custom color.
    #[test]
    fn a_file_saved_with_the_uk_label_keeps_custom_color() {
        let text = r##"{"format": 1, "mods": {"crosshair": {"enabled": true,
                      "options": {"color_mode": "Custom colour", "color": "#FFFF5555"}}}}"##;
        let (settings, report) = crate::settings::ModSettings::from_json_str(text);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(
            settings.choice("crosshair", "color_mode"),
            COLOR_MODE_CUSTOM
        );
        assert_eq!(settings.color("crosshair", "color"), 0xFFFF_5555);
        // Saving writes today's label.
        let saved = settings.to_json_string();
        assert!(saved.contains("\"Custom color\""), "{saved}");
        let (again, _) = crate::settings::ModSettings::from_json_str(&saved);
        assert_eq!(again.choice("crosshair", "color_mode"), COLOR_MODE_CUSTOM);
    }
}
