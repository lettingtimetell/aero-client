//! Time Changer: Sets the time of day you see without changing the server's.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).

use std::fmt::Write as _;

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue, SliderLabel};

/// Minecraft ticks in one day.
const TICKS_PER_DAY: i64 = 24_000;

/// The `time` option's world ticks: a clock time in hours (`12.0` = noon, `13.5` = 13:30) as
/// vanilla's time of day, where tick 0 is 06:00 and 1000 ticks are one hour.
#[must_use]
pub fn time_of_day_ticks(hours: f32) -> i64 {
    let ticks = ((f64::from(hours) - 6.0) * 1000.0).round();
    if ticks.is_finite() {
        (ticks as i64).rem_euclid(TICKS_PER_DAY)
    } else {
        6_000
    }
}

/// The `time` badge: the clock time as `HH:MM`.
pub fn clock_label(value: f32, out: &mut String) {
    let minutes = (f64::from(value) * 60.0).round() as i64;
    let (hours, minutes) = (
        minutes.div_euclid(60).rem_euclid(24),
        minutes.rem_euclid(60),
    );
    let _ = write!(out, "{hours:02}:{minutes:02}");
}

/// The `time` option is a clock time in hours, in 15-minute steps (`12.0` = noon).
static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "time",
    label: "Time of day",
    kind: OptionKind::SliderLabeled {
        min: 0.0,
        max: 23.75,
        step: 0.25,
        label: SliderLabel(clock_label),
    },
    default: OptionValue::Float(12.0),
    description: "The clock time you see. The server's time, the clock item and F3 are unchanged.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "time_changer",
    name: "Time Changer",
    category: Category::World,
    description: "Sets the time of day you see without changing the server's.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_hours_map_to_vanilla_ticks() {
        // Vanilla's table: 0 = 06:00, 6000 = noon, 12000 = 18:00, 18000 = midnight.
        assert_eq!(time_of_day_ticks(6.0), 0);
        assert_eq!(time_of_day_ticks(12.0), 6_000);
        assert_eq!(time_of_day_ticks(18.0), 12_000);
        assert_eq!(time_of_day_ticks(0.0), 18_000);
        assert_eq!(time_of_day_ticks(5.0), 23_000);
        assert_eq!(time_of_day_ticks(13.5), 7_500);
        assert_eq!(time_of_day_ticks(23.75), 17_750);
        assert_eq!(time_of_day_ticks(f32::NAN), 6_000);
    }

    #[test]
    fn the_badge_is_a_24_hour_clock() {
        let kind = OPTIONS[0].kind;
        assert_eq!(kind.format_value(&OptionValue::Float(12.0)), "12:00");
        assert_eq!(kind.format_value(&OptionValue::Float(0.0)), "00:00");
        assert_eq!(kind.format_value(&OptionValue::Float(13.5)), "13:30");
        assert_eq!(kind.format_value(&OptionValue::Float(23.75)), "23:45");
        assert_eq!(kind.format_value(&OptionValue::Float(6.25)), "06:15");
    }
}
