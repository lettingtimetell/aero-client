//! Fullbright: Lights the world as if every block were at full brightness.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).

use std::fmt::Write as _;

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue, SliderLabel};

/// The slider's top stop. At this value the mod uses the renderer's full-white lightmap
/// (`LightmapParams::fullbright`); below it the value is a gamma override (`value / 100`).
pub const BRIGHTNESS_FULL: f32 = 1500.0;

/// The `brightness` badge: `Full` at the top stop, otherwise the gamma as a percentage.
pub fn brightness_label(value: f32, out: &mut String) {
    if value >= BRIGHTNESS_FULL {
        out.push_str("Full");
    } else {
        let _ = write!(out, "{value:.0}%");
    }
}

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "brightness",
    label: "Brightness",
    kind: OptionKind::SliderLabeled {
        min: 100.0,
        max: BRIGHTNESS_FULL,
        step: 50.0,
        label: SliderLabel(brightness_label),
    },
    default: OptionValue::Float(BRIGHTNESS_FULL),
    description: "Full lights every block fully. Lower values raise the gamma instead.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "fullbright",
    name: "Fullbright",
    category: Category::World,
    description: "Lights the world as if every block were at full brightness.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_badge_reads_full_at_the_top_stop_and_a_percentage_below_it() {
        let kind = OPTIONS[0].kind;
        assert_eq!(kind.format_value(&OptionValue::Float(1500.0)), "Full");
        assert_eq!(kind.format_value(&OptionValue::Float(1450.0)), "1450%");
        assert_eq!(kind.format_value(&OptionValue::Float(100.0)), "100%");
    }
}
