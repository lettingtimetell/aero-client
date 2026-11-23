//! How the replay screens write times, lengths, sizes and dates. Pure functions, tested without
//! a clock or a timezone of their own (the caller passes "now" and the zone).

use std::fmt::Write as _;

use chrono::{DateTime, Datelike, TimeZone, Timelike};

/// Microseconds per second.
pub const US_PER_S: u64 = 1_000_000;

/// A length as a clock: `m:ss` under an hour, `h:mm:ss` from an hour on. Rounds down, so a
/// timeline never shows a time past its end.
#[must_use]
pub fn clock(us: u64) -> String {
    let mut out = String::with_capacity(8);
    write_clock(&mut out, us);
    out
}

/// [`clock`] into `out` (no allocation once `out` has room).
pub fn write_clock(out: &mut String, us: u64) {
    let total = us / US_PER_S;
    let (h, m, s) = (total / 3600, (total / 60) % 60, total % 60);
    if h > 0 {
        let _ = write!(out, "{h}:{m:02}:{s:02}");
    } else {
        let _ = write!(out, "{m}:{s:02}");
    }
}

/// A playback position with tenths: `m:ss.t` (or `h:mm:ss.t`).
#[must_use]
pub fn clock_tenths(us: u64) -> String {
    let mut out = String::with_capacity(10);
    write_clock_tenths(&mut out, us);
    out
}

/// [`clock_tenths`] into `out`.
pub fn write_clock_tenths(out: &mut String, us: u64) {
    write_clock(out, us);
    let tenths = (us % US_PER_S) / 100_000;
    let _ = write!(out, ".{tenths}");
}

/// A short length for a selection or a range: `12.7 s` under a minute, else a [`clock`].
#[must_use]
pub fn span(us: u64) -> String {
    if us < 60 * US_PER_S {
        let tenths = us / 100_000;
        format!("{}.{} s", tenths / 10, tenths % 10)
    } else {
        clock(us)
    }
}

/// A file size in decimal units (1 MB = 1,000,000 bytes), the units the `replay` mod's storage
/// limit and common upload limits use, so "9.4 MB" is under "10 MB": `940 KB`,
/// `8.6 MB`, `46 MB`, `1.3 GB`, `20 GB`.
#[must_use]
pub fn size(bytes: u64) -> String {
    /// `12.0` → `12`, `1.30` → `1.3`.
    fn trimmed(value: f64, decimals: usize) -> String {
        let text = format!("{value:.decimals$}");
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text
        }
    }
    let b = bytes as f64;
    if b < 1e3 {
        format!("{bytes} B")
    } else if b < 1e6 {
        format!("{} KB", trimmed(b / 1e3, 0))
    } else if b < 1e9 {
        let mb = b / 1e6;
        format!("{} MB", trimmed(mb, usize::from(mb < 10.0)))
    } else {
        let gb = b / 1e9;
        format!("{} GB", trimmed(gb, if gb < 10.0 { 2 } else { 1 }))
    }
}

/// A frame count with thousands separators: `12,345`.
#[must_use]
pub fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A speed as players say it: `1x`, `0.5x`, `0.25x`, `8x`.
#[must_use]
pub fn speed(speed: f32) -> String {
    let rounded = (speed * 100.0).round() / 100.0;
    if (rounded - rounded.round()).abs() < 1e-4 {
        format!("{}x", rounded.round() as i64)
    } else if ((rounded * 10.0) - (rounded * 10.0).round()).abs() < 1e-4 {
        format!("{rounded:.1}x")
    } else {
        format!("{rounded:.2}x")
    }
}

fn hour12(hour: u32) -> (u32, &'static str) {
    match hour {
        0 => (12, "AM"),
        1..=11 => (hour, "AM"),
        12 => (12, "PM"),
        _ => (hour - 12, "PM"),
    }
}

fn month_name(month: u32) -> &'static str {
    [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][(month.clamp(1, 12) - 1) as usize]
}

fn weekday_name(day: chrono::Weekday) -> &'static str {
    match day {
        chrono::Weekday::Mon => "Mon",
        chrono::Weekday::Tue => "Tue",
        chrono::Weekday::Wed => "Wed",
        chrono::Weekday::Thu => "Thu",
        chrono::Weekday::Fri => "Fri",
        chrono::Weekday::Sat => "Sat",
        chrono::Weekday::Sun => "Sun",
    }
}

/// When a replay was recorded, relative to `now_ms` (both milliseconds since the Unix epoch) in
/// the zone `tz`: `Today, 9:41 PM`, `Yesterday, 11:02 AM`, `Mon, 9:41 PM` within the week,
/// `Nov 2, 9:41 PM` within the year, `Nov 2, 2025` before that. US style (the brief's spelling).
#[must_use]
pub fn when<Tz: TimeZone>(started_ms: u64, now_ms: u64, tz: &Tz) -> String {
    let Some(at) = to_local(started_ms, tz) else {
        return "Unknown date".to_owned();
    };
    let Some(now) = to_local(now_ms, tz) else {
        return "Unknown date".to_owned();
    };
    let (hour, ampm) = hour12(at.hour());
    let time = format!("{hour}:{:02} {ampm}", at.minute());
    let days = now
        .date_naive()
        .signed_duration_since(at.date_naive())
        .num_days();
    match days {
        0 => format!("Today, {time}"),
        1 => "Yesterday, ".to_owned() + &time,
        2..=6 => format!("{}, {time}", weekday_name(at.weekday())),
        _ if at.year() == now.year() => {
            format!("{} {}, {time}", month_name(at.month()), at.day())
        }
        // In the future (a wrong clock) or more than a year ago: the full date.
        _ => format!("{} {}, {}", month_name(at.month()), at.day(), at.year()),
    }
}

fn to_local<Tz: TimeZone>(ms: u64, tz: &Tz) -> Option<DateTime<Tz>> {
    let ms = i64::try_from(ms).ok()?;
    tz.timestamp_millis_opt(ms).single()
}

/// A default export file name from a replay title and the range: letters, digits, spaces,
/// dashes and underscores survive; anything Windows forbids in a file name goes; at most 60
/// characters; never empty.
#[must_use]
pub fn file_stem(title: &str) -> String {
    let mut out = String::with_capacity(title.len().min(60));
    let mut last_space = true;
    for c in title.chars() {
        let keep = c.is_alphanumeric() || matches!(c, '-' | '_' | '(' | ')' | '.' | ',');
        if keep {
            out.push(c);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
        if out.chars().count() >= 60 {
            break;
        }
    }
    let trimmed = out.trim().trim_end_matches('.').trim();
    if trimmed.is_empty() {
        "Replay".to_owned()
    } else {
        trimmed.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    #[test]
    fn clocks_round_down_and_grow_an_hour_field() {
        assert_eq!(clock(0), "0:00");
        assert_eq!(clock(59_999_999), "0:59");
        assert_eq!(clock(754 * US_PER_S), "12:34");
        assert_eq!(clock(3_725 * US_PER_S), "1:02:05");
        assert_eq!(clock_tenths(83_450_000), "1:23.4");
        assert_eq!(clock_tenths(0), "0:00.0");
        assert_eq!(span(12_700_000), "12.7 s");
        assert_eq!(span(75 * US_PER_S), "1:15");
    }

    #[test]
    fn sizes_use_the_storage_limits_decimal_units() {
        assert_eq!(size(512), "512 B");
        assert_eq!(size(940_000), "940 KB");
        assert_eq!(size(8_600_000), "8.6 MB");
        assert_eq!(size(9_400_000), "9.4 MB", "under a 10 MB limit");
        assert_eq!(size(12_000_000), "12 MB");
        assert_eq!(size(340_000_000), "340 MB");
        assert_eq!(size(1_300_000_000), "1.3 GB");
        assert_eq!(size(20_000_000_000), "20 GB", "the storage limit as set");
        assert_eq!(size(10_000_000), "10 MB");
    }

    #[test]
    fn counts_and_speeds_read_naturally() {
        assert_eq!(count(7), "7");
        assert_eq!(count(1234), "1,234");
        assert_eq!(count(7_620_000), "7,620,000");
        assert_eq!(speed(1.0), "1x");
        assert_eq!(speed(0.5), "0.5x");
        assert_eq!(speed(0.25), "0.25x");
        assert_eq!(speed(0.1), "0.1x");
        assert_eq!(speed(8.0), "8x");
    }

    #[test]
    fn dates_are_relative_to_now_in_the_given_zone() {
        // 2026-11-12 21:41 at UTC-5 (EST).
        let est = FixedOffset::west_opt(5 * 3600).unwrap();
        let now = est
            .with_ymd_and_hms(2026, 11, 12, 21, 41, 0)
            .unwrap()
            .timestamp_millis() as u64;
        let at = |y, mo, d, h, mi| {
            est.with_ymd_and_hms(y, mo, d, h, mi, 0)
                .unwrap()
                .timestamp_millis() as u64
        };
        assert_eq!(when(at(2026, 11, 12, 21, 5), now, &est), "Today, 9:05 PM");
        assert_eq!(when(at(2026, 11, 12, 0, 30), now, &est), "Today, 12:30 AM");
        assert_eq!(
            when(at(2026, 11, 11, 11, 2), now, &est),
            "Yesterday, 11:02 AM"
        );
        // 2026-11-09 is a Monday.
        assert_eq!(when(at(2026, 11, 9, 12, 0), now, &est), "Mon, 12:00 PM");
        assert_eq!(when(at(2026, 11, 2, 9, 41), now, &est), "Nov 2, 9:41 AM");
        assert_eq!(when(at(2025, 6, 1, 9, 41), now, &est), "Jun 1, 2025");
    }

    #[test]
    fn file_stems_are_safe_on_windows() {
        assert_eq!(
            file_stem("SkyWars: Dwarf Fortress"),
            "SkyWars Dwarf Fortress"
        );
        assert_eq!(file_stem("a/b\\c*d?e\"f<g>h|i"), "a b c d e f g h i");
        assert_eq!(file_stem("   "), "Replay");
        assert_eq!(file_stem("..."), "Replay");
        assert!(file_stem(&"x".repeat(200)).chars().count() <= 60);
    }
}
