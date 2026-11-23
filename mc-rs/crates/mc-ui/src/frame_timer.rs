//! The F3 lagometer's fixed-size frame history.
//!
//! This is deliberately separate from [`mc_render::FrameStats`].  The latter is an aggregate
//! histogram for reports; the vanilla debug overlay needs the last 240 individual frame times,
//! with `FrameTimer.java`'s slightly surprising cursor behaviour.

/// Number of slots in vanilla's lagometer history.
pub const FRAME_TIMER_SLOTS: usize = 240;

/// Nanoseconds in one 60 FPS frame according to `FrameTimer.getLagometerValue`.
///
/// Keep the Java literal rather than replacing it with `1_000_000_000 / 60`: Java uses the
/// former as an `f64`, and the one-nanosecond distinction is observable at a bar boundary.
pub const LAGOMETER_FRAME_NANOS: f64 = 1.666_666_6E7;

/// `net.minecraft.util.FrameTimer`.
///
/// `last_index` and `index` intentionally do not describe a conventional full ring.  Once the
/// 240th frame is recorded they compare equal, so vanilla's `while (last != next)` graph draws
/// no bars for that one frame; from the 241st frame it draws 239 bars.  It is an old quirk, but
/// reproducing it matters more than making the debug graph look nicer.
// java: net/minecraft/util/FrameTimer.java
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameTimer {
    frames: [i64; FRAME_TIMER_SLOTS],
    last_index: usize,
    counter: usize,
    index: usize,
}

impl Default for FrameTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTimer {
    /// An empty 240-frame history.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frames: [0; FRAME_TIMER_SLOTS],
            last_index: 0,
            counter: 0,
            index: 0,
        }
    }

    /// Record one frame duration in nanoseconds.
    // java: FrameTimer.addFrame
    pub fn add_frame(&mut self, running_time: i64) {
        self.frames[self.index] = running_time;
        self.index += 1;
        if self.index == FRAME_TIMER_SLOTS {
            self.index = 0;
        }

        if self.counter < FRAME_TIMER_SLOTS {
            self.last_index = 0;
            self.counter += 1;
        } else {
            self.last_index = self.parse_index(self.index + 1);
        }
    }

    /// Convert a duration to a lagometer height using Java's floating-point formula.
    // java: FrameTimer.getLagometerValue
    #[must_use]
    pub fn lagometer_value(time: i64, multiplier: i32) -> i32 {
        ((time as f64 / LAGOMETER_FRAME_NANOS) * multiplier as f64) as i32
    }

    /// The lagometer's oldest cursor.
    #[must_use]
    pub const fn last_index(&self) -> usize {
        self.last_index
    }

    /// The next slot that [`Self::add_frame`] will overwrite.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// The backing array, including unused zero-valued slots.
    #[must_use]
    pub const fn frames(&self) -> &[i64; FRAME_TIMER_SLOTS] {
        &self.frames
    }

    /// Java's public modulo helper.
    #[must_use]
    pub const fn parse_index(&self, raw_index: usize) -> usize {
        raw_index % FRAME_TIMER_SLOTS
    }

    /// Frame slots in the exact order `GuiOverlayDebug.renderLagometer` consumes them.
    ///
    /// This is deliberately `last_index..index` *exclusive*, not a conventional history view;
    /// see the type documentation for why the 240/241 boundary looks odd.
    pub fn lagometer_samples(&self) -> LagometerSamples<'_> {
        LagometerSamples {
            timer: self,
            cursor: self.last_index,
            done: self.last_index == self.index,
        }
    }
}

/// Iterator returned by [`FrameTimer::lagometer_samples`].
pub struct LagometerSamples<'a> {
    timer: &'a FrameTimer,
    cursor: usize,
    done: bool,
}

impl Iterator for LagometerSamples<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let value = self.timer.frames[self.cursor];
        self.cursor = self.timer.parse_index(self.cursor + 1);
        self.done = self.cursor == self.timer.index;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagometer_scaling_uses_the_java_sixty_fps_double_literal() {
        assert_eq!(FrameTimer::lagometer_value(0, 30), 0);
        assert_eq!(FrameTimer::lagometer_value(16_666_666, 30), 30);
        assert_eq!(FrameTimer::lagometer_value(33_333_332, 30), 60);
        assert_eq!(FrameTimer::lagometer_value(16_666_666, 60), 60);
    }

    #[test]
    fn first_239_frames_read_from_zero_through_the_next_cursor_exclusive() {
        let mut timer = FrameTimer::new();
        for frame in 1..240 {
            timer.add_frame(frame);
        }
        assert_eq!(timer.last_index(), 0);
        assert_eq!(timer.index(), 239);
        assert_eq!(
            timer.lagometer_samples().collect::<Vec<_>>(),
            (1..240).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_240_and_241_frame_boundary_keeps_the_java_cursor_quirk() {
        let mut timer = FrameTimer::new();
        for frame in 1..=240 {
            timer.add_frame(frame);
        }
        assert_eq!((timer.last_index(), timer.index()), (0, 0));
        assert!(timer.lagometer_samples().next().is_none());

        timer.add_frame(241);
        assert_eq!((timer.last_index(), timer.index()), (2, 1));
        let samples = timer.lagometer_samples().collect::<Vec<_>>();
        assert_eq!(samples.len(), 239);
        assert_eq!(samples.first(), Some(&3));
        assert_eq!(samples.last(), Some(&241));
        assert!(
            !samples.contains(&2),
            "the oldest slot is excluded by Java's loop"
        );
    }
}
