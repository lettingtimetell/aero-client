//! The playback timeline as Windows reports it, turned into a position sampled "now".
//!
//! Windows gives the start, end and position of the track as 100-ns ticks, plus the wall-clock
//! time (`LastUpdatedTime`, FILETIME ticks since 1601) at which the player last set the position.
//! Players do not update the position continuously (Spotify sets it on play, pause, seek and track
//! change), so the position "now" is the reported one plus the time since that update while
//! playing. [`crate::NowPlaying::position_at`] then carries it forward from an `Instant`.

use std::time::{SystemTime, UNIX_EPOCH};

/// 100-ns ticks per millisecond.
const TICKS_PER_MS: i64 = 10_000;

/// FILETIME ticks from 1601-01-01 to 1970-01-01.
const UNIX_EPOCH_TICKS: i64 = 116_444_736_000_000_000;

/// A "last updated" time further back than this is not trusted for extrapolation.
const MAX_EXTRAPOLATION_TICKS: i64 = 24 * 60 * 60 * 1_000 * TICKS_PER_MS;

/// The raw timeline values of a media session, in 100-ns ticks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Timeline {
    /// `StartTime`: usually 0.
    pub(crate) start: i64,
    /// `EndTime`: the track's length when `start` is 0.
    pub(crate) end: i64,
    /// `Position` at `last_updated`.
    pub(crate) position: i64,
    /// `LastUpdatedTime` as FILETIME ticks (UTC); 0 when the player never set it.
    pub(crate) last_updated: i64,
}

impl Timeline {
    /// `(position_ms, duration_ms)` at `now` (FILETIME ticks): the reported position, advanced by
    /// the time since the player set it if `playing`, and clamped to the track. Both are `None`
    /// when the player reports no timeline at all (all zeros); the duration alone is `None` for
    /// a stream without an end.
    #[must_use]
    pub(crate) fn sample(&self, now: i64, playing: bool) -> (Option<u64>, Option<u64>) {
        let span = self.end.saturating_sub(self.start);
        let offset = self.position.saturating_sub(self.start);
        if span <= 0 && offset <= 0 {
            return (None, None);
        }
        let mut position = offset.max(0);
        if playing && self.last_updated > 0 {
            let elapsed = now.saturating_sub(self.last_updated);
            // A negative gap (the clock moved back) or a very old one is not extrapolated.
            if (0..=MAX_EXTRAPOLATION_TICKS).contains(&elapsed) {
                position = position.saturating_add(elapsed);
            }
        }
        if span > 0 {
            position = position.min(span);
        }
        (
            Some(ticks_to_ms(position)),
            (span > 0).then(|| ticks_to_ms(span)),
        )
    }
}

/// The current UTC time as FILETIME ticks, the clock `LastUpdatedTime` uses.
#[must_use]
pub(crate) fn filetime_now() -> i64 {
    let since_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let ticks = i64::try_from(since_unix.as_nanos() / 100).unwrap_or(i64::MAX);
    ticks.saturating_add(UNIX_EPOCH_TICKS)
}

fn ticks_to_ms(ticks: i64) -> u64 {
    u64::try_from(ticks / TICKS_PER_MS).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: i64 = 1_000 * TICKS_PER_MS;

    /// A paused state read from a real player (3:02.978 long, at 1:27.133).
    fn spotify_paused(last_updated: i64) -> Timeline {
        Timeline {
            start: 0,
            end: 1_829_780_000,
            position: 871_330_000,
            last_updated,
        }
    }

    #[test]
    fn a_paused_track_reports_its_position_as_is() {
        let now = filetime_now();
        let timeline = spotify_paused(now - 4 * SECOND);
        assert_eq!(timeline.sample(now, false), (Some(87_133), Some(182_978)));
    }

    #[test]
    fn a_playing_track_advances_by_the_time_since_the_last_update() {
        let now = filetime_now();
        let timeline = spotify_paused(now - 4 * SECOND);
        assert_eq!(timeline.sample(now, true), (Some(91_133), Some(182_978)));
    }

    #[test]
    fn the_position_never_passes_the_end() {
        let now = filetime_now();
        let timeline = spotify_paused(now - 600 * SECOND);
        assert_eq!(timeline.sample(now, true), (Some(182_978), Some(182_978)));
    }

    #[test]
    fn a_missing_stale_or_future_update_time_is_not_extrapolated() {
        let now = filetime_now();
        assert_eq!(
            spotify_paused(0).sample(now, true).0,
            Some(87_133),
            "never set"
        );
        assert_eq!(
            spotify_paused(now + 10 * SECOND).sample(now, true).0,
            Some(87_133),
            "clock moved back"
        );
        assert_eq!(
            spotify_paused(now - 2 * MAX_EXTRAPOLATION_TICKS)
                .sample(now, true)
                .0,
            Some(87_133),
            "a day-old update"
        );
    }

    #[test]
    fn no_timeline_streams_and_start_offsets() {
        let now = filetime_now();
        assert_eq!(
            Timeline::default().sample(now, true),
            (None, None),
            "no timeline"
        );
        let stream = Timeline {
            start: 0,
            end: 0,
            position: 30 * SECOND,
            last_updated: now - SECOND,
        };
        assert_eq!(
            stream.sample(now, true),
            (Some(31_000), None),
            "no end: live stream"
        );
        let offset = Timeline {
            start: 10 * SECOND,
            end: 70 * SECOND,
            position: 25 * SECOND,
            last_updated: 0,
        };
        assert_eq!(offset.sample(now, false), (Some(15_000), Some(60_000)));
        let before_start = Timeline {
            start: 10 * SECOND,
            end: 70 * SECOND,
            position: 5 * SECOND,
            last_updated: 0,
        };
        assert_eq!(before_start.sample(now, false), (Some(0), Some(60_000)));
    }

    #[test]
    fn filetime_now_is_on_the_windows_epoch() {
        // 2026-01-01T00:00:00Z as FILETIME ticks: 11_644_473_600 s (1601 to 1970) plus
        // 1_767_225_600 s (Unix time) = 13_411_699_200 s after 1601-01-01.
        let start_of_2026 = 134_116_992_000_000_000;
        let now = filetime_now();
        assert!(now > start_of_2026, "{now}");
        assert!(now < start_of_2026 + 5 * 365 * 24 * 3_600 * SECOND, "{now}");
    }
}
