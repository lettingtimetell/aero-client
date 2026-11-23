//! The open/close animation: pure functions of real elapsed time, so the motion is the same at
//! 60 FPS and at 3000.
//!
//! Opening: each window fades in and slides up [`SLIDE_DIP`] with an ease-out cubic over
//! [`OPEN_MS`], window `i` starting [`STAGGER_MS`]·`i` later; the dim and the glass strength
//! ramp in over the same [`OPEN_MS`]. Closing is the time-reverse, faster ([`CLOSE_MS`]), with a
//! small reversed stagger (the rightmost window leaves first).

/// Duration of one window's opening motion.
pub const OPEN_MS: f32 = 200.0;
/// Delay between consecutive windows while opening.
pub const STAGGER_MS: f32 = 35.0;
/// Duration of one window's closing motion.
pub const CLOSE_MS: f32 = 120.0;
/// Delay between consecutive windows while closing (reversed order).
pub const CLOSE_STAGGER_MS: f32 = 12.0;
/// How far a window travels while it fades, in design pixels.
pub const SLIDE_DIP: f32 = 14.0;

/// `1 - (1 - t)^3`, clamped.
#[must_use]
pub fn ease_out_cubic(t: f32) -> f32 {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let u = 1.0 - t;
    1.0 - u * u * u
}

/// Window `index`'s opening progress (`0` hidden, `1` in place) `elapsed_ms` after the open.
#[must_use]
pub fn window_open_progress(elapsed_ms: f32, index: usize) -> f32 {
    ease_out_cubic((elapsed_ms - index as f32 * STAGGER_MS) / OPEN_MS)
}

/// Window `index`'s progress while closing, `1` (in place) down to `0`.
#[must_use]
pub fn window_close_progress(elapsed_ms: f32, index: usize, count: usize) -> f32 {
    let reversed = count.saturating_sub(1).saturating_sub(index) as f32;
    let t = (elapsed_ms - reversed * CLOSE_STAGGER_MS) / CLOSE_MS;
    // The opening curve played backwards: slow start, quick finish.
    ease_out_cubic(1.0 - t.clamp(0.0, 1.0))
}

/// The dim and glass-strength ramp while opening.
#[must_use]
pub fn dim_open_progress(elapsed_ms: f32) -> f32 {
    ease_out_cubic(elapsed_ms / OPEN_MS)
}

/// The dim and glass-strength ramp while closing.
#[must_use]
pub fn dim_close_progress(elapsed_ms: f32) -> f32 {
    ease_out_cubic(1.0 - (elapsed_ms / CLOSE_MS).clamp(0.0, 1.0))
}

/// When the closing animation of `count` windows is over.
#[must_use]
pub fn close_finished(elapsed_ms: f32, count: usize) -> bool {
    elapsed_ms >= CLOSE_MS + count.saturating_sub(1) as f32 * CLOSE_STAGGER_MS
}

/// Frame-rate independent exponential approach: the fraction of the remaining distance to
/// cover in `dt_s` seconds at `rate` per second.
#[must_use]
pub fn approach(dt_s: f32, rate: f32) -> f32 {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return 0.0;
    }
    1.0 - (-dt_s * rate).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_is_staggered_eased_and_complete_after_the_last_window() {
        assert_eq!(window_open_progress(0.0, 0), 0.0);
        // Ease-out: well past halfway at half time.
        let half = window_open_progress(OPEN_MS * 0.5, 0);
        assert!(half > 0.8 && half < 0.9, "{half}");
        // Window 2 starts 70 ms later.
        assert_eq!(window_open_progress(2.0 * STAGGER_MS, 2), 0.0);
        assert!(window_open_progress(2.0 * STAGGER_MS + 1.0, 2) > 0.0);
        let last = 5;
        let done_at = OPEN_MS + last as f32 * STAGGER_MS;
        assert_eq!(window_open_progress(done_at, last), 1.0);
        assert_eq!(window_open_progress(done_at + 1000.0, 0), 1.0);
        assert_eq!(dim_open_progress(OPEN_MS), 1.0);
        // Monotone.
        let mut previous = 0.0;
        for ms in 0..400 {
            let p = window_open_progress(ms as f32, 3);
            assert!(p >= previous);
            previous = p;
        }
    }

    #[test]
    fn closing_reverses_faster_with_the_last_window_first() {
        assert_eq!(window_close_progress(0.0, 0, 6), 1.0);
        // The rightmost window starts leaving first.
        let at = 10.0;
        assert!(window_close_progress(at, 5, 6) < window_close_progress(at, 0, 6));
        assert!(!close_finished(CLOSE_MS, 6));
        let end = CLOSE_MS + 5.0 * CLOSE_STAGGER_MS;
        assert!(close_finished(end, 6));
        for index in 0..6 {
            assert_eq!(window_close_progress(end, index, 6), 0.0);
        }
        assert_eq!(dim_close_progress(CLOSE_MS), 0.0);
    }

    // Closing is faster than opening (checked at compile time).
    const _: () = assert!(CLOSE_MS < OPEN_MS, "closing is faster");

    #[test]
    fn progress_depends_on_time_not_on_frame_count() {
        // Sampling the same instant through many tiny frames or one big one is identical,
        // because nothing is integrated per frame.
        let coarse = window_open_progress(123.0, 1);
        let mut fine = 0.0;
        for step in 0..=1230 {
            fine = window_open_progress(step as f32 * 0.1, 1);
        }
        assert_eq!(coarse, fine);
        // The exponential smoother converges the same total amount for split time steps.
        let one = approach(0.02, 18.0);
        let two = 1.0 - (1.0 - approach(0.01, 18.0)).powi(2);
        assert!((one - two).abs() < 1e-5);
    }
}
