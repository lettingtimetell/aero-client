//! The replay browser's list grouped by day ("Today",
//! "Yesterday", "Friday", "November 13"), like the date folders on disk, and short recordings
//! folded away.
//!
//! **Short recordings.** "Record every game" used to leave a file for every few seconds spent
//! in a lobby (the user's library had ten of 1-14 s among 23). The recorder now keeps those
//! lobby hops in the next game's file, but a connection that lasts only a few seconds still
//! makes one. A full recording shorter than [`SHORT_US`] with no bookmark is "short": the browser
//! hides those until the player shows them (the S key, or the footer's "Show N short" button).
//! Nothing is deleted: they stay on disk, in the storage total, and one press shows them.
//! Clips are never short (the player asked for each).

use chrono::{Datelike, NaiveDate, TimeZone};

use super::{format, ReplayEntry};

/// Full recordings shorter than this, with no bookmark, are "short".
pub const SHORT_US: u64 = 20_000_000;

/// Whether `entry` is a short recording (see the module docs).
#[must_use]
pub fn is_short(entry: &ReplayEntry) -> bool {
    !entry.clip && entry.duration_us < SHORT_US && entry.bookmarks == 0
}

/// One row of the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Item {
    /// A day's header: its index in [`Groups::days`].
    Day(usize),
    /// An entry: its index in the list's entries.
    Entry(usize),
}

/// A day's header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Day {
    /// "Today", "Yesterday", "Friday", "November 13", "November 13, 2025".
    pub label: String,
    /// "3 recordings · 48 MB".
    pub summary: String,
}

/// The list laid out top to bottom: each item with its top (design pixels) and height.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Groups {
    /// The day headers, in list order.
    pub days: Vec<Day>,
    /// The rows: `(item, top, height)`.
    pub items: Vec<(Item, f32, f32)>,
    /// The content's height (no gap after the last row).
    pub height: f32,
}

impl Groups {
    /// Lay out `entries` (newest first) under a header per local day of `tz`: headers
    /// `header_h` tall, entries `row_h` with `gap` after each row.
    #[must_use]
    pub fn layout<Tz: TimeZone>(
        entries: &[ReplayEntry],
        clips: bool,
        now_ms: u64,
        tz: &Tz,
        dims: (f32, f32, f32),
    ) -> Groups {
        Groups::layout_by(
            entries,
            clips,
            day_of(now_ms, tz),
            |e| day_of(e.started_unix_ms, tz),
            dims,
        )
    }

    /// [`Groups::layout`] with each entry's day from `day_of_entry` (the browser's cache) and
    /// `today` for the labels.
    #[must_use]
    pub fn layout_by(
        entries: &[ReplayEntry],
        clips: bool,
        today: Option<NaiveDate>,
        day_of_entry: impl Fn(&ReplayEntry) -> Option<NaiveDate>,
        (header_h, row_h, gap): (f32, f32, f32),
    ) -> Groups {
        let mut groups = Groups::default();
        let mut y = 0.0;
        let mut current: Option<Option<NaiveDate>> = None;
        for (index, entry) in entries.iter().enumerate() {
            let day = day_of_entry(entry);
            if current != Some(day) {
                current = Some(day);
                let (count, bytes) = entries[index..]
                    .iter()
                    .take_while(|e| day_of_entry(e) == day)
                    .fold((0usize, 0u64), |(n, b), e| (n + 1, b + e.size_bytes));
                let noun = match (clips, count) {
                    (true, 1) => "clip",
                    (true, _) => "clips",
                    (false, 1) => "recording",
                    (false, _) => "recordings",
                };
                groups.days.push(Day {
                    label: day.map_or_else(|| "Unknown date".to_owned(), |d| day_label(d, today)),
                    summary: format!("{count} {noun}  \u{b7}  {}", format::size(bytes)),
                });
                groups
                    .items
                    .push((Item::Day(groups.days.len() - 1), y, header_h));
                y += header_h;
            }
            groups.items.push((Item::Entry(index), y, row_h));
            y += row_h + gap;
        }
        groups.height = (y - gap).max(0.0);
        groups
    }

    /// The top of entry `index`'s row, if it is in the list.
    #[must_use]
    pub fn entry_top(&self, index: usize) -> Option<f32> {
        self.items
            .iter()
            .find(|(item, _, _)| *item == Item::Entry(index))
            .map(|(_, top, _)| *top)
    }
}

/// The local time of day of `ms` in `tz`: `9:41 PM` (the row's detail line; its day is in the
/// header above it).
#[must_use]
pub fn time_of_day<Tz: TimeZone>(ms: u64, tz: &Tz) -> String {
    let Some(at) = i64::try_from(ms)
        .ok()
        .and_then(|ms| tz.timestamp_millis_opt(ms).single())
    else {
        return "Unknown time".to_owned();
    };
    at.naive_local().format("%-I:%M %p").to_string()
}

/// The local date of `ms` (milliseconds since the Unix epoch) in `tz`.
#[must_use]
pub fn day_of<Tz: TimeZone>(ms: u64, tz: &Tz) -> Option<NaiveDate> {
    let ms = i64::try_from(ms).ok()?;
    tz.timestamp_millis_opt(ms)
        .single()
        .map(|at| at.naive_local().date())
}

/// A day's header: "Today", "Yesterday", the weekday within the week, then "November 13"
/// (with the year when it is not this year's).
#[must_use]
pub fn day_label(day: NaiveDate, today: Option<NaiveDate>) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let month = MONTHS[day.month0() as usize % 12];
    let Some(today) = today else {
        return format!("{month} {}, {}", day.day(), day.year());
    };
    match today.signed_duration_since(day).num_days() {
        0 => "Today".to_owned(),
        1 => "Yesterday".to_owned(),
        2..=6 => match day.weekday() {
            chrono::Weekday::Mon => "Monday",
            chrono::Weekday::Tue => "Tuesday",
            chrono::Weekday::Wed => "Wednesday",
            chrono::Weekday::Thu => "Thursday",
            chrono::Weekday::Fri => "Friday",
            chrono::Weekday::Sat => "Saturday",
            chrono::Weekday::Sun => "Sunday",
        }
        .to_owned(),
        _ if day.year() == today.year() => format!("{month} {}", day.day()),
        _ => format!("{month} {}, {}", day.day(), day.year()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    fn est() -> FixedOffset {
        FixedOffset::west_opt(5 * 3600).unwrap()
    }

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> u64 {
        est()
            .with_ymd_and_hms(y, mo, d, h, mi, 0)
            .unwrap()
            .timestamp_millis() as u64
    }

    fn entry(id: u64, started: u64, seconds: u64) -> ReplayEntry {
        ReplayEntry {
            id,
            title: format!("r{id}"),
            server: "mc.example.net".to_owned(),
            started_unix_ms: started,
            duration_us: seconds * 1_000_000,
            size_bytes: 2_000_000,
            clip: false,
            has_mic: false,
            problem: None,
            bookmarks: 0,
            folder: String::new(),
            player: String::new(),
        }
    }

    #[test]
    fn days_get_a_header_each_in_the_local_zone() {
        let now = at(2026, 11, 14, 1, 30);
        let entries = vec![
            entry(1, at(2026, 11, 14, 0, 10), 300),
            // 21:05 EST on the 13th is 02:05 UTC on the 14th: still "Yesterday" here.
            entry(2, 1_794_621_923_186, 208),
            entry(3, at(2026, 11, 13, 9, 33), 600),
            entry(4, at(2026, 11, 9, 12, 0), 600),
            entry(5, at(2026, 10, 2, 12, 0), 600),
            entry(6, at(2025, 6, 1, 12, 0), 600),
        ];
        let groups = Groups::layout(&entries, false, now, &est(), (30.0, 80.0, 6.0));
        let labels: Vec<&str> = groups.days.iter().map(|d| d.label.as_str()).collect();
        assert_eq!(
            labels,
            ["Today", "Yesterday", "Monday", "October 2", "June 1, 2025"]
        );
        assert_eq!(groups.days[1].summary, "2 recordings  \u{b7}  4 MB");
        assert_eq!(groups.days[0].summary, "1 recording  \u{b7}  2 MB");
        // Header, row; header, row, row; ...
        assert_eq!(groups.items[0], (Item::Day(0), 0.0, 30.0));
        assert_eq!(groups.items[1], (Item::Entry(0), 30.0, 80.0));
        assert_eq!(groups.items[2], (Item::Day(1), 116.0, 30.0));
        assert_eq!(groups.entry_top(2), Some(146.0 + 86.0));
        let rows = 6.0 * 86.0 - 6.0;
        assert!((groups.height - (rows + 5.0 * 30.0)).abs() < 1e-3);
        // In UTC the recording from another time zone is on the 14th: "Today".
        let utc = FixedOffset::east_opt(0).unwrap();
        let groups = Groups::layout(&entries[..2], false, now, &utc, (30.0, 80.0, 6.0));
        assert_eq!(groups.days.len(), 1);
    }

    #[test]
    fn short_means_a_full_recording_under_20_s_without_bookmarks() {
        let mut e = entry(1, 0, 9);
        assert!(is_short(&e));
        e.bookmarks = 1;
        assert!(!is_short(&e), "a bookmark says it matters");
        e.bookmarks = 0;
        e.clip = true;
        assert!(!is_short(&e), "clips are never folded");
        assert!(!is_short(&entry(2, 0, 20)));
    }
}
