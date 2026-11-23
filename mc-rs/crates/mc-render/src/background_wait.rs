//! The frame pacing of a window in the background: another window has the focus, or the window is
//! minimized (see [`crate::ShellApp::background_frame_cap`]).
//!
//! A focused window is paced by [`crate::frame::FrameLimiter`], which sleeps on the main thread.
//! Nothing can wake that sleep, so a window held at 10 frames a second could answer the player's
//! return up to 100 ms late. A window in the background is paced here instead: the shell tells the
//! event loop to wait until its next frame is due (`ControlFlow::WaitUntil`), and a change of the
//! window's state (its focus coming back, a restore) ends the wait at once. The wait is the event
//! loop's own (on Windows a high-resolution waitable timer with `MsgWaitForMultipleObjectsEx`), so
//! it does not spin at all.
//!
//! Plain data and arithmetic, so the whole state machine is tested without a window.

use std::time::{Duration, Instant};

/// The cap a frame of a window in the background is paced at, or `None` when the frame is paced
/// as a focused one is (by the limiter): the window has the focus and is not minimized (one
/// `bool` test, and neither closure runs), or the app has no background cap (`app_cap`, asked
/// with whether the window is minimized). The background cap never raises the usual one
/// (`normal`, `None` when uncapped).
pub(crate) fn background_cap(
    focused: bool,
    minimized: bool,
    app_cap: impl FnOnce(bool) -> Option<u32>,
    normal: impl FnOnce() -> Option<u32>,
) -> Option<u32> {
    if focused && !minimized {
        return None;
    }
    let background = app_cap(minimized)?;
    Some(normal().map_or(background, |cap| cap.min(background)))
}

/// What the event loop does next ([`BackgroundWait::step`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopStep {
    /// The next frame is not due yet: wait until then, or until a window event.
    WaitUntil(Instant),
    /// The wait is over: the event loop goes back to polling, and the next frame starts now.
    Resume,
    /// No frame waits: draw as usual.
    Run,
}

/// The schedule of a window in the background.
#[derive(Debug, Default)]
pub(crate) struct BackgroundWait {
    /// The window is minimized (it reports a zero size while it is).
    minimized: bool,
    /// When the next frame is due, while one waits.
    due: Option<Instant>,
    /// The event loop was told to wait and has not been put back to polling yet.
    waiting: bool,
    /// The due time of the last scheduled frame and the cap it was scheduled at: the next one is a
    /// period later (an absolute schedule, as [`crate::frame::FrameLimiter`] keeps).
    last: Option<(Instant, u32)>,
}

impl BackgroundWait {
    /// Whether the window is minimized.
    pub(crate) fn minimized(&self) -> bool {
        self.minimized
    }

    /// The window's size changed: a zero size means it is minimized. Becoming minimized or being
    /// restored ends a wait, so the next frame starts at once with its new cap. Returns whether a
    /// frame was waiting.
    pub(crate) fn set_minimized(&mut self, minimized: bool) -> bool {
        if self.minimized == minimized {
            return false;
        }
        self.minimized = minimized;
        self.wake()
    }

    /// End a wait now (the window's focus changed): the next frame starts at once, and the
    /// schedule starts over from it. Returns whether a frame was waiting.
    pub(crate) fn wake(&mut self) -> bool {
        self.last = None;
        self.due.take().is_some()
    }

    /// Schedule the frame after the one that has just ended (at `now`), at `cap` frames a second:
    /// one period after the last one was due, but never later than a period from now nor earlier
    /// than now. A frame that ran long starts the next at once and costs its own time and no more.
    pub(crate) fn schedule(&mut self, cap: u32, now: Instant) {
        let period = Duration::from_secs_f64(1.0 / f64::from(cap.max(1)));
        let due = match self.last {
            Some((last, last_cap)) if last_cap == cap => (last + period).min(now + period).max(now),
            _ => now + period,
        };
        self.last = Some((due, cap));
        self.due = (due > now).then_some(due);
    }

    /// Whether the event loop has to look at this wait at all: a frame is waiting, or the loop was
    /// told to wait and must be put back to polling. One test per iteration otherwise.
    pub(crate) fn active(&self) -> bool {
        self.due.is_some() || self.waiting
    }

    /// What the event loop does next, at `now`.
    pub(crate) fn step(&mut self, now: Instant) -> LoopStep {
        match self.due {
            Some(due) if now < due => {
                self.waiting = true;
                LoopStep::WaitUntil(due)
            }
            _ => {
                self.due = None;
                if std::mem::take(&mut self.waiting) {
                    LoopStep::Resume
                } else {
                    LoopStep::Run
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// The decision at the end of every frame: focused windows never ask the app, a window in
    /// the background gets the lower of the two caps, and no background cap means the old pacing.
    #[test]
    fn the_cap_is_the_lower_of_the_two_and_only_in_the_background() {
        let app = |minimized: bool| Some(if minimized { 10 } else { 30 });
        let never = |_: bool| -> Option<u32> { panic!("a focused window never asks the app") };
        assert_eq!(background_cap(true, false, never, || Some(360)), None);
        // Unfocused: 30 under an uncapped or a 360 FPS game; a lower usual cap stays.
        assert_eq!(background_cap(false, false, app, || None), Some(30));
        assert_eq!(background_cap(false, false, app, || Some(360)), Some(30));
        assert_eq!(background_cap(false, false, app, || Some(20)), Some(20));
        // Minimized, focused or not (the focus can stay on a minimized window).
        assert_eq!(background_cap(false, true, app, || Some(360)), Some(10));
        assert_eq!(background_cap(true, true, app, || None), Some(10));
        // No background cap (the option off, an export, a load): paced as before.
        assert_eq!(background_cap(false, true, |_| None, || Some(360)), None);
        assert_eq!(background_cap(false, false, |_| None, || None), None);
    }

    /// Nothing scheduled: the loop never looks (the focused window's one test).
    #[test]
    fn an_idle_wait_is_inactive() {
        let mut wait = BackgroundWait::default();
        assert!(!wait.active());
        assert_eq!(wait.step(Instant::now()), LoopStep::Run);
        assert!(!wait.active());
        assert!(!wait.minimized());
    }

    /// A frame at 10 FPS waits 100 ms in the event loop, then the loop polls again once.
    #[test]
    fn a_scheduled_frame_waits_until_due_then_resumes_polling_once() {
        let mut wait = BackgroundWait::default();
        let t0 = Instant::now();
        wait.schedule(10, t0);
        assert!(wait.active());
        assert_eq!(wait.step(t0 + ms(1)), LoopStep::WaitUntil(t0 + ms(100)));
        // A window event in between (a mouse move) wakes the loop: it waits again.
        assert_eq!(wait.step(t0 + ms(40)), LoopStep::WaitUntil(t0 + ms(100)));
        assert_eq!(wait.step(t0 + ms(100)), LoopStep::Resume);
        assert!(!wait.active(), "back to the usual frame");
        assert_eq!(wait.step(t0 + ms(101)), LoopStep::Run);
    }

    /// The focus coming back ends the wait at once, and the loop is put back to polling.
    #[test]
    fn a_wake_ends_the_wait_at_once() {
        let mut wait = BackgroundWait::default();
        let t0 = Instant::now();
        assert!(!wait.wake(), "nothing waits yet");
        wait.schedule(10, t0);
        assert!(matches!(wait.step(t0 + ms(5)), LoopStep::WaitUntil(_)));
        assert!(wait.wake(), "a frame was waiting");
        assert!(
            wait.active(),
            "the loop still has to be put back to polling"
        );
        assert_eq!(wait.step(t0 + ms(6)), LoopStep::Resume);
        assert!(!wait.active());
        assert!(!wait.wake());
    }

    /// Being minimized or restored ends a wait (the cap changes); the same state again does not.
    #[test]
    fn minimizing_and_restoring_wake_and_repeats_do_not() {
        let mut wait = BackgroundWait::default();
        let t0 = Instant::now();
        wait.schedule(30, t0);
        assert!(!wait.set_minimized(false));
        assert!(matches!(wait.step(t0), LoopStep::WaitUntil(_)), "no change");
        assert!(wait.set_minimized(true), "a waiting frame was woken");
        assert!(wait.minimized());
        assert_eq!(wait.step(t0 + ms(1)), LoopStep::Resume, "minimized: now");
        wait.schedule(10, t0 + ms(1));
        assert!(!wait.set_minimized(true));
        assert!(
            matches!(wait.step(t0 + ms(2)), LoopStep::WaitUntil(_)),
            "no change"
        );
        assert!(wait.set_minimized(false));
        assert_eq!(wait.step(t0 + ms(3)), LoopStep::Resume, "restored: now");
        // Minimized with no frame waiting (a focused window): the state changes, nothing to wake.
        assert!(!wait.set_minimized(true));
        assert!(wait.minimized());
    }

    /// Frames keep an absolute schedule: drawn a little late, the next is still due a period
    /// after the last due time, so 30 FPS stays 30 FPS whatever each frame costs.
    #[test]
    fn the_schedule_is_absolute_and_a_slow_frame_does_not_pile_up() {
        let mut wait = BackgroundWait::default();
        let t0 = Instant::now();
        let period = Duration::from_secs_f64(1.0 / 30.0);
        wait.schedule(30, t0);
        let first = t0 + period;
        assert_eq!(wait.step(t0), LoopStep::WaitUntil(first));
        assert_eq!(wait.step(first), LoopStep::Resume);
        // That frame took 2 ms: the next is still due one period after `first`.
        wait.schedule(30, first + ms(2));
        assert_eq!(
            wait.step(first + ms(2)),
            LoopStep::WaitUntil(first + period)
        );
        assert_eq!(wait.step(first + period), LoopStep::Resume);
        // A frame that took longer than a period: the next one starts at once, no backlog.
        let late = first + period + ms(80);
        wait.schedule(30, late);
        assert!(!wait.active(), "due now: no wait");
        assert_eq!(wait.step(late), LoopStep::Run);
        // And the schedule continues from there.
        wait.schedule(30, late + ms(1));
        assert_eq!(wait.step(late + ms(1)), LoopStep::WaitUntil(late + period));
    }

    /// A frame drawn early (the window repainted by the system mid-wait) never makes the next
    /// one wait more than a period; a new cap starts a new schedule.
    #[test]
    fn an_early_frame_or_a_new_cap_waits_one_period_at_most() {
        let mut wait = BackgroundWait::default();
        let t0 = Instant::now();
        wait.schedule(10, t0);
        // Drawn at 20 ms instead of 100 ms.
        wait.schedule(10, t0 + ms(20));
        assert_eq!(wait.step(t0 + ms(20)), LoopStep::WaitUntil(t0 + ms(120)));
        wait.schedule(30, t0 + ms(120));
        let period = Duration::from_secs_f64(1.0 / 30.0);
        assert_eq!(
            wait.step(t0 + ms(120)),
            LoopStep::WaitUntil(t0 + ms(120) + period)
        );
        // A wake restarts the schedule from the frame it starts.
        assert!(wait.wake());
        assert_eq!(wait.step(t0 + ms(121)), LoopStep::Resume);
        wait.schedule(30, t0 + ms(122));
        assert_eq!(
            wait.step(t0 + ms(122)),
            LoopStep::WaitUntil(t0 + ms(122) + period)
        );
    }
}
