//! The replay's own clock: which tick and partial tick, and which look, a replay time maps to.
//!
//! Every rendered frame of a recording writes a [`FrameRecord`](super::codec::FrameRecord):
//! the replay time it rendered at, the client ticks run so far and its exact partial tick. That
//! makes `virtual ticks = ticks + partial` a known value at every recorded frame. Between two
//! recorded frames the game's timer advanced linearly with time, so playback interpolates
//! linearly between them. At a recorded frame's time the result is exactly that frame's tick
//! count and partial tick, which is what makes a replayed frame identical to the recorded one.
//!
//! Nothing here reads a clock: it is a pure function of the recording, so the exporter can ask
//! for any time in any order and get the same answer.

use mc_replay::ReplayTime;

/// One recorded frame, reduced to what the clock needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrameAnchor {
    /// The replay time it rendered at.
    pub(crate) time: ReplayTime,
    /// Client ticks run before it rendered.
    pub(crate) client_tick: u64,
    /// Its partial tick.
    pub(crate) partial: f32,
    /// The player's look (`rotationYaw`).
    pub(crate) yaw: f32,
    /// The player's look (`rotationPitch`).
    pub(crate) pitch: f32,
    /// The Zoom mod's FOV divisor.
    pub(crate) zoom: f32,
    /// [`FrameRecord::flags`](super::codec::FrameRecord::flags).
    pub(crate) flags: u8,
    /// The enchantment-glint clock this frame drew with (ms, modulo
    /// [`GLINT_CLOCK_PERIOD_MS`](super::codec::GLINT_CLOCK_PERIOD_MS)).
    pub(crate) glint_ms: u64,
    /// `GuiIngame.prevVignetteBrightness` as this frame began.
    pub(crate) vignette: f32,
}

impl FrameAnchor {
    fn virtual_ticks(&self) -> f64 {
        self.client_tick as f64 + f64::from(self.partial)
    }
}

/// Where the replay is at one moment: the tick count to have run, the partial tick to render
/// with, and the player's look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ClockSample {
    /// Every tick record with `client_tick <= ticks` must have been applied.
    pub(crate) ticks: u64,
    /// The partial tick, `0.0..1.0`.
    pub(crate) partial: f32,
    /// The look to render with.
    pub(crate) yaw: f32,
    /// The look to render with.
    pub(crate) pitch: f32,
    /// The zoom divisor to render with.
    pub(crate) zoom: f32,
    /// The nearest earlier recorded frame's flags.
    pub(crate) flags: u8,
    /// The glint clock to draw with: the recorded frame's own, or the earlier frame's advanced
    /// by the elapsed replay time.
    pub(crate) glint_ms: u64,
    /// The vignette ease to start the frame from: the recorded frame's, or the earlier one's.
    pub(crate) vignette: f32,
    /// `Some(i)` when the time is exactly recorded frame `i`'s.
    pub(crate) exact_frame: Option<usize>,
}

/// The recorded frames of a replay, in time order.
#[derive(Debug, Clone, Default)]
pub(crate) struct ReplayClock {
    frames: Vec<FrameAnchor>,
}

impl ReplayClock {
    /// Build from frames in recording order. Frames whose time or virtual tick goes backwards
    /// (which a sane recording never has) are dropped so the clock stays monotonic.
    pub(crate) fn new(frames: impl IntoIterator<Item = FrameAnchor>) -> Self {
        // Filtered in place: a `Vec` passed in keeps its buffer (std's
        // in-place collect), so a long replay's million-frame index is not held twice while the
        // clock is built, nor grown by doubling. Same frames kept as the old copy-and-push: each
        // one is compared with the last one kept.
        let mut frames: Vec<FrameAnchor> = frames.into_iter().collect();
        let mut last_kept: Option<(ReplayTime, f64)> = None;
        frames.retain(|frame| {
            let ticks = frame.virtual_ticks();
            let keep = last_kept.is_none_or(|(time, last_ticks)| {
                let goes_back = frame.time < time || ticks < last_ticks;
                !goes_back
            });
            if keep {
                last_kept = Some((frame.time, ticks));
            }
            keep
        });
        if frames.len() < frames.capacity() / 2 {
            frames.shrink_to_fit();
        }
        Self { frames }
    }

    /// The recorded frames.
    pub(crate) fn frames(&self) -> &[FrameAnchor] {
        &self.frames
    }

    /// The index of the last recorded frame at or before `time`.
    pub(crate) fn frame_at_or_before(&self, time: ReplayTime) -> Option<usize> {
        let after = self.frames.partition_point(|frame| frame.time <= time);
        after.checked_sub(1)
    }

    /// The clock at `time`. Before the first recorded frame it holds the first frame; after the
    /// last it holds the last.
    pub(crate) fn sample(&self, time: ReplayTime) -> Option<ClockSample> {
        let first = self.frames.first()?;
        let after = self.frames.partition_point(|frame| frame.time <= time);
        if after == 0 {
            return Some(Self::exact(first, None));
        }
        let index = after - 1;
        let a = &self.frames[index];
        if a.time == time || after == self.frames.len() {
            return Some(Self::exact(a, (a.time == time).then_some(index)));
        }
        let b = &self.frames[after];
        let span = (b.time - a.time) as f64;
        let t = (time - a.time) as f64 / span;
        let virtual_ticks = a.virtual_ticks() + (b.virtual_ticks() - a.virtual_ticks()) * t;
        let ticks = virtual_ticks.floor();
        // Never step back below `a`'s tick count through a rounding edge.
        let ticks_u = (ticks as u64).max(a.client_tick);
        let partial = if ticks_u as f64 > ticks {
            0.0
        } else {
            ((virtual_ticks - ticks) as f32).clamp(0.0, 0.999_999_9)
        };
        let lerp = |x: f32, y: f32| (f64::from(x) + (f64::from(y) - f64::from(x)) * t) as f32;
        let glint_ms = (a.glint_ms + (time - a.time) / 1_000) % super::codec::GLINT_CLOCK_PERIOD_MS;
        Some(ClockSample {
            ticks: ticks_u,
            partial,
            yaw: lerp(a.yaw, b.yaw),
            pitch: lerp(a.pitch, b.pitch),
            zoom: lerp(a.zoom, b.zoom),
            flags: a.flags,
            glint_ms,
            vignette: a.vignette,
            exact_frame: None,
        })
    }

    fn exact(frame: &FrameAnchor, exact_frame: Option<usize>) -> ClockSample {
        ClockSample {
            ticks: frame.client_tick,
            partial: frame.partial,
            yaw: frame.yaw,
            pitch: frame.pitch,
            zoom: frame.zoom,
            flags: frame.flags,
            glint_ms: frame.glint_ms,
            vignette: frame.vignette,
            exact_frame,
        }
    }

    /// The first time at which `sample(time).ticks >= ticks`, for stepping by one tick. `None`
    /// past the end.
    pub(crate) fn time_of_tick(&self, ticks: u64) -> Option<ReplayTime> {
        let first = self.frames.first()?;
        if first.client_tick >= ticks {
            return Some(first.time);
        }
        let after = self
            .frames
            .partition_point(|frame| frame.virtual_ticks() < ticks as f64);
        let b = self.frames.get(after)?;
        let a = &self.frames[after - 1];
        // Solve a.v + (b.v - a.v) * t = ticks for t, then round up to a whole microsecond so the
        // sample there has crossed the boundary.
        let span_v = b.virtual_ticks() - a.virtual_ticks();
        if span_v <= 0.0 {
            return Some(b.time);
        }
        let t = (ticks as f64 - a.virtual_ticks()) / span_v;
        let mut time = a.time + ((b.time - a.time) as f64 * t).ceil() as u64;
        while time < b.time && self.sample(time).is_some_and(|s| s.ticks < ticks) {
            time += 1;
        }
        Some(time.min(b.time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(time: u64, client_tick: u64, partial: f32, yaw: f32) -> FrameAnchor {
        FrameAnchor {
            time,
            client_tick,
            partial,
            yaw,
            pitch: 0.0,
            zoom: 1.0,
            flags: 0,
            glint_ms: time / 1_000,
            vignette: 0.5,
        }
    }

    /// A 60 fps recording of a 20 TPS timer: three frames per tick.
    fn sixty_fps() -> ReplayClock {
        ReplayClock::new((0..120u64).map(|i| {
            let time = i * 16_667;
            let virtual_ticks = time as f64 / 50_000.0;
            let ticks = virtual_ticks.floor();
            frame(
                time,
                100 + ticks as u64,
                (virtual_ticks - ticks) as f32,
                i as f32,
            )
        }))
    }

    #[test]
    fn a_recorded_frame_time_gives_exactly_its_ticks_partial_and_look() {
        let clock = sixty_fps();
        for (index, recorded) in clock.frames().iter().enumerate() {
            let sample = clock.sample(recorded.time).expect("a sample");
            assert_eq!(sample.ticks, recorded.client_tick);
            assert_eq!(sample.partial.to_bits(), recorded.partial.to_bits());
            assert_eq!(sample.yaw.to_bits(), recorded.yaw.to_bits());
            assert_eq!(sample.exact_frame, Some(index));
        }
    }

    #[test]
    fn between_frames_the_clock_is_monotonic_and_crosses_ticks_in_order() {
        let clock = sixty_fps();
        let mut last = (0u64, 0.0f32);
        let end = clock.frames().last().expect("frames").time;
        for time in (0..end).step_by(997) {
            let sample = clock.sample(time).expect("a sample");
            let now = (sample.ticks, sample.partial);
            assert!(
                now.0 > last.0 || (now.0 == last.0 && now.1 >= last.1),
                "time {time}: {now:?} after {last:?}"
            );
            assert!((0.0..1.0).contains(&sample.partial));
            last = now;
        }
    }

    #[test]
    fn the_look_is_interpolated_between_recorded_frames() {
        let clock = ReplayClock::new([frame(1_000, 10, 0.0, 10.0), frame(3_000, 10, 0.5, 30.0)]);
        let middle = clock.sample(2_000).expect("a sample");
        assert_eq!(middle.yaw, 20.0);
        assert_eq!(middle.ticks, 10);
        assert!((middle.partial - 0.25).abs() < 1e-6);
        // Before the first frame and after the last, the clock holds.
        assert_eq!(clock.sample(0).expect("held").yaw, 10.0);
        assert_eq!(clock.sample(9_000).expect("held").yaw, 30.0);
    }

    #[test]
    fn time_of_tick_finds_the_first_moment_a_tick_is_due() {
        let clock = sixty_fps();
        for ticks in 101..135u64 {
            let time = clock.time_of_tick(ticks).expect("in range");
            assert!(clock.sample(time).expect("sample").ticks >= ticks);
            if time > 0 {
                assert!(clock.sample(time - 1).expect("sample").ticks < ticks);
            }
        }
        assert_eq!(clock.time_of_tick(10_000), None);
    }

    #[test]
    fn frames_that_go_backwards_are_dropped() {
        let clock = ReplayClock::new([
            frame(0, 5, 0.5, 0.0),
            // Its ticks go backwards (5.25 after 5.5): dropped.
            frame(10, 5, 0.25, 0.0),
            frame(5, 6, 0.0, 0.0),
            // Its time goes backwards (3 after 5): dropped.
            frame(3, 7, 0.0, 0.0),
            frame(20, 6, 0.5, 0.0),
        ]);
        let times: Vec<u64> = clock.frames().iter().map(|anchor| anchor.time).collect();
        assert_eq!(times, vec![0, 5, 20]);
    }
}
