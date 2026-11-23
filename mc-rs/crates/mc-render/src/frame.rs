//! Frame timing and the camera.
//!
//! [`Timer`] is a port of `net/minecraft/util/Timer.java` — vanilla's fixed 20 TPS accumulator
//! with render interpolation. [`Camera`] matches vanilla's FOV and orientation semantics but is
//! built **camera-relative**, which the Java client does not need to do and we do.

use glam::{DVec3, Mat4, Vec2, Vec3};

/// Opt-in chronological CPU frame diagnostics with bounded retained samples.
pub mod telemetry;

/// Vanilla's tick rate. Not a tunable: the server ticks at 20 and every animation, physics
/// constant, and interpolation in the game is expressed in ticks (§7 S10 hazard).
pub const TICKS_PER_SECOND: f32 = 20.0;

/// Longest catch-up burst the timer will report from one update, matching `Timer.java`'s cap.
///
/// Time beyond this is **dropped, not carried** — exactly as vanilla drops it — so a long stall
/// (asset load, GC pause, dragging the window) cannot turn into a burst of hundreds of ticks.
pub const MAX_CATCH_UP_TICKS: i32 = 10;

/// One exact diagnostic simulation tick at the vanilla 20 Hz cadence.
pub const DIAGNOSTIC_TICK_MICROS: u64 = 50_000;

/// Why a deterministic diagnostic timer step was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticTimerError {
    /// A diagnostic step must advance time by at least one microsecond.
    ZeroStep,
    /// Diagnostic cadence is defined only for the vanilla 20 TPS timer.
    UnsupportedRate,
    /// Diagnostic cadence does not apply a live timer-speed multiplier.
    UnsupportedSpeed,
    /// Adding the step to the retained fractional microseconds overflowed `u64`.
    Overflow,
    /// A single diagnostic step would exceed vanilla's ten-tick catch-up report limit.
    CatchUpLimit,
}

impl std::fmt::Display for DiagnosticTimerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::ZeroStep => "diagnostic timer step must be positive",
            Self::UnsupportedRate => "diagnostic timer requires 20 TPS",
            Self::UnsupportedSpeed => "diagnostic timer requires timer speed 1.0",
            Self::Overflow => "diagnostic timer microsecond accumulator overflowed",
            Self::CatchUpLimit => "diagnostic timer step exceeds the catch-up limit",
        };
        f.write_str(message)
    }
}

impl std::error::Error for DiagnosticTimerError {}

/// Vanilla's three-register cinematic-camera filter.
///
/// java: `net.minecraft.util.MouseFilter`.  These are deliberately `f32`: the filter's
/// intermediate rounding is visible in a long, slow camera turn.
#[derive(Debug, Clone, Copy, Default)]
pub struct MouseFilter {
    accumulated: f32,
    filtered: f32,
    smoothing: f32,
}

impl MouseFilter {
    /// Add one mouse amount and return its filtered contribution.
    ///
    /// java: `MouseFilter.smooth`.  In particular, the seemingly redundant clamp to the
    /// half-step accumulator is not optional: it is what prevents a sign-changing turn from
    /// overshooting the camera.
    #[must_use]
    pub fn smooth(&mut self, amount: f32, factor: f32) -> f32 {
        self.accumulated += amount;
        let mut filtered = (self.accumulated - self.filtered) * factor;
        self.smoothing += (filtered - self.smoothing) * 0.5;
        if (filtered > 0.0 && filtered > self.smoothing)
            || (filtered < 0.0 && filtered < self.smoothing)
        {
            filtered = self.smoothing;
        }
        self.filtered += filtered;
        filtered
    }

    /// Clear all three registers. java: `MouseFilter.reset`.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Tick-owned state for 1.8.9's smooth ("cinematic") camera.
///
/// java: `EntityRenderer.smoothCamYaw`, `smoothCamPitch`, `smoothCamPartialTicks`, and its two
/// `MouseFilter`s. Raw frame deltas accumulate until `updateRenderer` filters them once per game
/// tick; render frames only integrate that held filter output across partial-tick time.
#[derive(Debug, Clone, Default)]
pub struct SmoothCamera {
    yaw: f32,
    pitch: f32,
    partial_ticks: f32,
    filtered_yaw: f32,
    filtered_pitch: f32,
    filter_x: MouseFilter,
    filter_y: MouseFilter,
}

impl SmoothCamera {
    /// Run `EntityRenderer.updateRenderer`'s once-per-tick filter update.
    ///
    /// Enabled ticks filter every raw delta accumulated by prior render frames, reset the partial
    /// baseline to zero, and clear the accumulators. Disabled ticks clear both filter registers,
    /// exactly like Java's `else` arm.
    pub fn tick(&mut self, sensitivity: f32, enabled: bool) {
        if enabled {
            let f = sensitivity * 0.6 + 0.2;
            let factor = 0.05 * (f * f * f * 8.0);
            self.filtered_yaw = self.filter_x.smooth(self.yaw, factor);
            self.filtered_pitch = self.filter_y.smooth(self.pitch, factor);
            self.partial_ticks = 0.0;
            self.yaw = 0.0;
            self.pitch = 0.0;
        } else {
            self.filtered_yaw = 0.0;
            self.filtered_pitch = 0.0;
            self.filter_x.reset();
            self.filter_y.reset();
        }
    }

    /// Convert one frame's sensitivity-scaled, non-inverted mouse values into `setAngles` input.
    ///
    /// With smooth camera disabled this is an exact pass-through and clears the dormant raw
    /// accumulators. With it enabled, raw values accumulate for the next tick while the current
    /// tick's held filter output is integrated over `partial_ticks - smoothCamPartialTicks`.
    #[must_use]
    pub fn apply(&mut self, yaw: f32, pitch: f32, partial_ticks: f32, enabled: bool) -> (f32, f32) {
        if !enabled {
            self.yaw = 0.0;
            self.pitch = 0.0;
            return (yaw, pitch);
        }

        self.yaw += yaw;
        self.pitch += pitch;
        let elapsed = partial_ticks - self.partial_ticks;
        self.partial_ticks = partial_ticks;
        (self.filtered_yaw * elapsed, self.filtered_pitch * elapsed)
    }
}

/// Vanilla's fixed-tick accumulator with render interpolation.
///
/// java: `util/Timer.java`
///
/// Call [`Timer::update`] once per frame, run [`Timer::elapsed_ticks`] game ticks, then render
/// once using [`Timer::render_partial_ticks`] to interpolate between the previous and current
/// tick states. Render rate and tick rate stay decoupled, which is the whole point: coupling
/// them breaks both animation smoothness and movement parity with the server.
///
/// ```
/// use mc_render::frame::Timer;
/// let mut t = Timer::new(20.0);
/// t.reset_clocks(0, 0);
/// t.update_with(1000, 1000); // exactly one second of wall clock
/// assert_eq!(t.elapsed_ticks(), 10); // 20 ticks accrued, capped at the catch-up limit
/// ```
#[derive(Debug, Clone)]
pub struct Timer {
    ticks_per_second: f32,
    /// Speed multiplier; `0.5` runs the game at half speed. java: `Timer.timerSpeed`
    pub timer_speed: f32,
    elapsed_ticks: i32,
    elapsed_partial_ticks: f32,
    render_partial_ticks: f32,
    last_hr_time: f64,
    last_sync_sys_clock: i64,
    last_sync_hr_clock: i64,
    counter: i64,
    time_sync_adjustment: f64,
    origin: std::time::Instant,
    /// Exact sub-tick remainder retained only by the deterministic diagnostic stepping API.
    diagnostic_remainder_micros: u64,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new(TICKS_PER_SECOND)
    }
}

impl Timer {
    /// A timer running at `tps` ticks per second. Pass [`TICKS_PER_SECOND`] for vanilla parity.
    #[must_use]
    pub fn new(tps: f32) -> Self {
        let origin = std::time::Instant::now();
        Self {
            ticks_per_second: tps,
            timer_speed: 1.0,
            elapsed_ticks: 0,
            elapsed_partial_ticks: 0.0,
            render_partial_ticks: 0.0,
            last_hr_time: 0.0,
            last_sync_sys_clock: 0,
            last_sync_hr_clock: 0,
            counter: 0,
            time_sync_adjustment: 1.0,
            origin,
            diagnostic_remainder_micros: 0,
        }
    }

    /// Whole game ticks that came due at the last update, `0..=`[`MAX_CATCH_UP_TICKS`].
    ///
    /// java: `Timer.elapsedTicks`
    #[must_use]
    pub fn elapsed_ticks(&self) -> i32 {
        self.elapsed_ticks
    }

    /// Fraction of the way from the last tick to the next, in `0.0..1.0`.
    ///
    /// This is the `partial_ticks` every renderer in the codebase interpolates with.
    /// java: `Timer.renderPartialTicks`
    #[must_use]
    pub fn render_partial_ticks(&self) -> f32 {
        self.render_partial_ticks
    }

    /// Freeze `partial_ticks` for a paused frame while still advancing the clock, so that
    /// unpausing does not jolt. java: `Minecraft.runGameLoop` under `isGamePaused`
    pub fn update_paused(&mut self) {
        let frozen = self.render_partial_ticks;
        self.update();
        self.render_partial_ticks = frozen;
    }

    /// Advance using the process monotonic clock. Call once per frame.
    pub fn update(&mut self) {
        let millis = self.origin.elapsed().as_millis().min(i64::MAX as u128) as i64;
        // Java reads two independent clocks here (`Sys.getTime` and `System.nanoTime`) and keeps
        // a running ratio to correct their drift. A single Rust `Instant` is monotonic and needs
        // no correction, so both arguments come from it and `time_sync_adjustment` converges to
        // exactly 1.0 — the drift-correction path below is preserved for parity and for tests.
        self.update_with(millis, millis);
    }

    /// Advance from explicit clock readings, both in milliseconds. Deterministic; used by tests
    /// and by anything that needs to drive the timer from a recorded trace (S22).
    ///
    /// `hr_millis` is the high-resolution clock, `sys_millis` the system clock.
    pub fn update_with(&mut self, hr_millis: i64, sys_millis: i64) {
        let i = sys_millis;
        let j = i - self.last_sync_sys_clock;
        let k = hr_millis;
        let d0 = k as f64 / 1000.0;

        if (0..=1000).contains(&j) {
            self.counter += j;

            if self.counter > 1000 {
                let l = k - self.last_sync_hr_clock;
                // Java divides unguarded here and can poison the ratio with an infinity when the
                // two clocks disagree badly. Guarding costs nothing and cannot change the result
                // in the normal case, where `l` is a full second of elapsed time.
                if l > 0 {
                    let d1 = self.counter as f64 / l as f64;
                    self.time_sync_adjustment +=
                        (d1 - self.time_sync_adjustment) * 0.200_000_002_980_232_24;
                }
                self.last_sync_hr_clock = k;
                self.counter = 0;
            }

            if self.counter < 0 {
                self.last_sync_hr_clock = k;
            }
        } else {
            self.last_hr_time = d0;
        }

        self.last_sync_sys_clock = i;
        let mut d2 = (d0 - self.last_hr_time) * self.time_sync_adjustment;
        self.last_hr_time = d0;
        d2 = d2.clamp(0.0, 1.0);

        self.elapsed_partial_ticks = (f64::from(self.elapsed_partial_ticks)
            + d2 * f64::from(self.timer_speed) * f64::from(self.ticks_per_second))
            as f32;
        self.elapsed_ticks = self.elapsed_partial_ticks as i32;
        self.elapsed_partial_ticks -= self.elapsed_ticks as f32;

        if self.elapsed_ticks > MAX_CATCH_UP_TICKS {
            self.elapsed_ticks = MAX_CATCH_UP_TICKS;
        }

        self.render_partial_ticks = self.elapsed_partial_ticks;
    }

    /// Advance the deterministic diagnostic clock by one explicit microsecond step.
    ///
    /// This method never reads a host clock. It is deliberately restricted to the vanilla
    /// 20 TPS cadence with `timer_speed == 1.0`, so integer microseconds remain the source of
    /// truth: whole ticks are reported in [`Self::elapsed_ticks`], and the exact retained
    /// remainder becomes [`Self::render_partial_ticks`]. A rejected step leaves every timer
    /// field unchanged. Steps over [`MAX_CATCH_UP_TICKS`] are rejected rather than silently
    /// dropping diagnostic time as the live Java-compatible path does.
    pub fn advance_diagnostic_micros(
        &mut self,
        step_micros: u64,
    ) -> Result<(), DiagnosticTimerError> {
        if step_micros == 0 {
            return Err(DiagnosticTimerError::ZeroStep);
        }
        if self.ticks_per_second != TICKS_PER_SECOND || !self.ticks_per_second.is_finite() {
            return Err(DiagnosticTimerError::UnsupportedRate);
        }
        if self.timer_speed != 1.0 || !self.timer_speed.is_finite() {
            return Err(DiagnosticTimerError::UnsupportedSpeed);
        }

        let total_micros = self
            .diagnostic_remainder_micros
            .checked_add(step_micros)
            .ok_or(DiagnosticTimerError::Overflow)?;
        let ticks = total_micros / DIAGNOSTIC_TICK_MICROS;
        if ticks > MAX_CATCH_UP_TICKS as u64 {
            return Err(DiagnosticTimerError::CatchUpLimit);
        }
        let remainder_micros = total_micros % DIAGNOSTIC_TICK_MICROS;

        self.diagnostic_remainder_micros = remainder_micros;
        self.elapsed_ticks = ticks as i32;
        self.elapsed_partial_ticks = remainder_micros as f32 / DIAGNOSTIC_TICK_MICROS as f32;
        self.render_partial_ticks = self.elapsed_partial_ticks;
        Ok(())
    }

    /// Produce a frozen diagnostic frame with no new simulation ticks or interpolation fraction.
    ///
    /// The integer remainder is retained so a later explicit step resumes from the same exact
    /// microsecond boundary. This method performs no host-clock read and is safe to repeat for
    /// every post-warmup frame.
    pub fn update_diagnostic_frozen(&mut self) {
        self.elapsed_ticks = 0;
        self.elapsed_partial_ticks = 0.0;
        self.render_partial_ticks = 0.0;
    }

    /// Seed both internal clocks, discarding accumulated time. Call after a long blocking load
    /// so the first frame afterwards does not report a catch-up burst.
    pub fn reset_clocks(&mut self, hr_millis: i64, sys_millis: i64) {
        self.last_hr_time = hr_millis as f64 / 1000.0;
        self.last_sync_hr_clock = hr_millis;
        self.last_sync_sys_clock = sys_millis;
        self.counter = 0;
        self.elapsed_ticks = 0;
        self.elapsed_partial_ticks = 0.0;
        self.render_partial_ticks = 0.0;
        self.diagnostic_remainder_micros = 0;
        self.origin = std::time::Instant::now();
    }
}

/// Sub-bucket resolution of [`FrameTimeHistogram`]: every microsecond below this value gets a bin
/// of its own.
const HIST_SUB_BITS: u32 = 10;
/// `2^HIST_SUB_BITS` = 1024 µs. Frame times below 1.024 ms are recorded exactly.
const HIST_SUB: u64 = 1 << HIST_SUB_BITS;
/// Largest power-of-two magnitude with its own bins: 2^30 µs ≈ 17.9 minutes. Longer samples
/// saturate into the final bin (and are still exact in [`FrameTimeHistogram::max_micros`]).
const HIST_MAX_MAGNITUDE: u32 = 30;
/// 1024 exact bins + 21 magnitudes × 512 bins = 11 776 bins = 47 104 bytes at `u32` each.
const HIST_BINS: usize =
    HIST_SUB as usize + (HIST_MAX_MAGNITUDE - HIST_SUB_BITS + 1) as usize * (HIST_SUB as usize / 2);

/// Which bin a microsecond value lands in.
fn hist_bin_of(micros: u64) -> usize {
    if micros < HIST_SUB {
        return micros as usize;
    }
    let magnitude = u64::from(63 - micros.leading_zeros());
    if magnitude > u64::from(HIST_MAX_MAGNITUDE) {
        return HIST_BINS - 1;
    }
    let shift = magnitude - u64::from(HIST_SUB_BITS - 1);
    let within = (micros >> shift) - HIST_SUB / 2;
    HIST_SUB as usize
        + (magnitude - u64::from(HIST_SUB_BITS)) as usize * (HIST_SUB as usize / 2)
        + within as usize
}

/// The value a bin reports: exact below [`HIST_SUB`], the bin's midpoint above it.
fn hist_bin_value(bin: usize) -> u64 {
    if bin < HIST_SUB as usize {
        return bin as u64;
    }
    let relative = bin - HIST_SUB as usize;
    let per_magnitude = HIST_SUB as usize / 2;
    let magnitude = u64::from(HIST_SUB_BITS) + (relative / per_magnitude) as u64;
    let within = (relative % per_magnitude) as u64;
    let shift = magnitude - u64::from(HIST_SUB_BITS - 1);
    ((HIST_SUB / 2 + within) << shift) + (1 << shift) / 2
}

/// A whole-run frame-time distribution in a fixed 47 KiB.
///
/// # Why this exists
///
/// A lobby measurement found a mean frame of 0.5 ms and a worst frame of 3.16 ms against a
/// 2.78 ms budget at 360 Hz. **The stutter is invisible in the mean**, so a summary that reports
/// only mean/min/max cannot say whether a change to the chunk pipeline moved the thing the user
/// feels. p99 and p99.9 can.
///
/// # The bound, and what it costs
///
/// Retaining every sample would be unbounded: a 60 000-frame run at 2600 fps is 60 000 samples
/// and a long play session is millions. A ring buffer of the last *N* would be bounded but would
/// **evict the join burst**, which is precisely the tail §1.5 is about. So this is a histogram —
/// bounded *and* whole-run.
///
/// Bins are log-linear (each power-of-two range split into equal-width bins):
///
/// | range | bin width | quantisation error |
/// |---|---|---|
/// | 0 – 1.024 ms | 1 µs | **exact** (integer µs) |
/// | ≥ 1.024 ms | 2^(⌊log2 v⌋ − 9) | half a bin, ≤ 1/1024 of the value = **±0.098 %** |
///
/// Everything in the 2.048–4.096 ms magnitude — which is where both the 2.783 ms frame of §1.3
/// and the 3.16 ms worst frame of §1.5 sit — therefore lands within **±2 µs**. That is two orders
/// of magnitude below the run-to-run spread of anything in §1, so the bound costs nothing that
/// matters here. The cost is real but it is paid somewhere no conclusion depends on it.
///
/// Two further honesty notes:
///
/// * `min`/`max` are kept **exactly**, outside the histogram, and every percentile is clamped
///   into `[min, max]`. A reported p99 can therefore never exceed the reported max, which is the
///   consistency check the plan asks for.
/// * Samples longer than 2^30 µs (≈ 18 minutes) saturate into the last bin, which has no upper
///   edge. A percentile that lands there reports the exact `max` instead of a midpoint, since
///   that is the only exact thing known about the bin's contents.
#[derive(Clone)]
pub struct FrameTimeHistogram {
    bins: Box<[u32]>,
    count: u64,
    sum_us: u128,
    sum_sq_us: u128,
    min_us: u64,
    max_us: u64,
}

impl Default for FrameTimeHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FrameTimeHistogram {
    /// Prints the summary, not 11 776 bins — a derived `Debug` here would make
    /// `{:?}` on a [`FrameStats`] or a [`RunSummary`](crate::shell::RunSummary) unreadable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.summary(), f)
    }
}

impl FrameTimeHistogram {
    /// An empty histogram. Allocates the bins once, up front.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bins: vec![0u32; HIST_BINS].into_boxed_slice(),
            count: 0,
            sum_us: 0,
            sum_sq_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    /// Record one frame time.
    pub fn record(&mut self, frame: std::time::Duration) {
        self.record_micros(frame.as_micros().min(u128::from(u64::MAX)) as u64);
    }

    /// Record one frame time already in microseconds. Deterministic; this is what the tests drive.
    pub fn record_micros(&mut self, micros: u64) {
        self.bins[hist_bin_of(micros)] = self.bins[hist_bin_of(micros)].saturating_add(1);
        self.count += 1;
        self.sum_us += u128::from(micros);
        self.sum_sq_us += u128::from(micros) * u128::from(micros);
        self.min_us = self.min_us.min(micros);
        self.max_us = self.max_us.max(micros);
    }

    /// Samples recorded.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Shortest frame, exactly, in microseconds. Zero when nothing has been recorded.
    #[must_use]
    pub fn min_micros(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.min_us
        }
    }

    /// Longest frame, exactly, in microseconds — never quantised, even past the histogram's top.
    #[must_use]
    pub fn max_micros(&self) -> u64 {
        self.max_us
    }

    /// The `q`-quantile in microseconds, `q` in `0.0..=1.0`, by the **nearest-rank** method:
    /// the smallest value at or below which at least `ceil(q × n)` samples fall.
    ///
    /// Clamped into `[min, max]`, so the result is always consistent with the extremes reported
    /// alongside it. Returns 0 when nothing has been recorded.
    #[must_use]
    pub fn percentile_micros(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let q = q.clamp(0.0, 1.0);
        let rank = (q * self.count as f64).ceil().max(1.0) as u64;
        let rank = rank.min(self.count);
        let mut seen: u64 = 0;
        for (bin, &n) in self.bins.iter().enumerate() {
            if n == 0 {
                continue;
            }
            seen += u64::from(n);
            if seen >= rank {
                // The final bin is the saturating catch-all and has no upper edge, so it has no
                // meaningful midpoint either. The one exact thing known about a sample in it is
                // the run's maximum, so report that rather than 2^31 µs.
                if bin == HIST_BINS - 1 {
                    return self.max_us;
                }
                return hist_bin_value(bin).clamp(self.min_us, self.max_us);
            }
        }
        self.max_us
    }

    /// Arithmetic mean, in microseconds. Computed from the running sum, not the bins, so it
    /// carries no quantisation error at all.
    #[must_use]
    pub fn mean_micros(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_us as f64 / self.count as f64
    }

    /// Population standard deviation, in microseconds. Also exact — from the running sums.
    #[must_use]
    pub fn sd_micros(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.sum_us as f64 / n;
        let variance = (self.sum_sq_us as f64 / n) - mean * mean;
        variance.max(0.0).sqrt()
    }

    /// The whole distribution, in milliseconds, ready to print.
    #[must_use]
    pub fn summary(&self) -> FrameTimeSummary {
        let ms = |us: f64| us / 1000.0;
        FrameTimeSummary {
            frames: self.count,
            mean_ms: ms(self.mean_micros()),
            sd_ms: ms(self.sd_micros()),
            min_ms: ms(self.min_micros() as f64),
            p50_ms: ms(self.percentile_micros(0.50) as f64),
            p95_ms: ms(self.percentile_micros(0.95) as f64),
            p99_ms: ms(self.percentile_micros(0.99) as f64),
            p999_ms: ms(self.percentile_micros(0.999) as f64),
            max_ms: ms(self.max_micros() as f64),
        }
    }
}

/// Mean, spread and tail of a run's frame times, in milliseconds.
///
/// Every field is a millisecond figure so that a reader can compare directly against the frame
/// budget (2.78 ms at 360 Hz). `frames` is the `n` that §0 rule 3 requires be quoted with them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTimeSummary {
    /// Number of frames the distribution covers.
    pub frames: u64,
    /// Arithmetic mean frame time. Exact — not read off the histogram.
    pub mean_ms: f64,
    /// Population standard deviation. Exact.
    pub sd_ms: f64,
    /// Shortest frame. Exact.
    pub min_ms: f64,
    /// Median.
    pub p50_ms: f64,
    /// 95th percentile.
    pub p95_ms: f64,
    /// 99th percentile — **the smoothness number**. One frame in a hundred is worse than this.
    pub p99_ms: f64,
    /// 99.9th percentile. At 360 fps this is roughly the worst frame in every 2.8 seconds.
    pub p999_ms: f64,
    /// Longest frame. Exact.
    pub max_ms: f64,
}

impl std::fmt::Display for FrameTimeSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} mean={:.3} sd={:.3} min={:.3} p50={:.3} p95={:.3} p99={:.3} p99.9={:.3} \
             max={:.3} (ms)",
            self.frames,
            self.mean_ms,
            self.sd_ms,
            self.min_ms,
            self.p50_ms,
            self.p95_ms,
            self.p99_ms,
            self.p999_ms,
            self.max_ms
        )
    }
}

/// Rolling frame-rate counter, updated on one-second boundaries like vanilla's `debugFPS`.
///
/// java: `Minecraft.runGameLoop` (the `debugUpdateTime` block)
///
/// Also accumulates the whole run's frame-time distribution in a [`FrameTimeHistogram`], because
/// the FPS counter vanilla has cannot answer the stutter question — see
/// [`FrameStats::frame_time`].
#[derive(Debug, Clone)]
pub struct FrameStats {
    frames_this_second: u32,
    fps: u32,
    total_frames: u64,
    last_frame: std::time::Instant,
    second_start: std::time::Instant,
    started: std::time::Instant,
    last_frame_secs: f32,
    frame_times: FrameTimeHistogram,
    fps_updates: u64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameStats {
    /// A counter starting now.
    #[must_use]
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            frames_this_second: 0,
            fps: 0,
            total_frames: 0,
            last_frame: now,
            second_start: now,
            started: now,
            last_frame_secs: 0.0,
            frame_times: FrameTimeHistogram::new(),
            fps_updates: 0,
        }
    }

    /// Record one presented frame. Returns `true` on the frame where [`FrameStats::fps`] changed.
    pub fn frame(&mut self) -> bool {
        let now = std::time::Instant::now();
        let delta = now.duration_since(self.last_frame);
        self.last_frame_secs = delta.as_secs_f32();
        self.last_frame = now;
        self.frames_this_second += 1;
        self.total_frames += 1;
        // Recorded for every frame including the first, whose delta runs back to `new()`. That
        // frame is often the most expensive of the run (the deferred asset load), and hiding it
        // would defeat the point of keeping a tail metric at all.
        self.frame_times.record(delta);

        if now.duration_since(self.second_start) >= std::time::Duration::from_secs(1) {
            self.fps = self.frames_this_second;
            self.frames_this_second = 0;
            self.second_start = now;
            self.fps_updates += 1;
            return true;
        }
        false
    }

    /// Frames presented during the last completed one-second window.
    #[must_use]
    pub fn fps(&self) -> u32 {
        self.fps
    }

    /// Frames presented since this counter was created.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// How many one-second windows have closed, i.e. how many times [`FrameStats::fps`] has been
    /// recomputed.
    ///
    /// [`FrameStats::frame`] returns that same edge, but only to the caller that drives the
    /// counter — the shell. A caller further down that accumulates its own per-second number
    /// (vanilla's `RenderChunk.renderChunksUpdated`, which is zeroed in the very block that sets
    /// `debugFPS`) only sees a `&FrameStats`, and cannot spot the edge from [`FrameStats::fps`]
    /// alone because two adjacent windows may hold the same frame rate. Watching this instead
    /// puts its counter on exactly the window the FPS figure came from.
    #[must_use]
    pub fn fps_updates(&self) -> u64 {
        self.fps_updates
    }

    /// Wall time since this counter was created.
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        self.started.elapsed()
    }

    /// Mean frames per second across the counter's whole lifetime — the number to quote when
    /// benchmarking, since [`FrameStats::fps`] is only the most recent one-second sample.
    #[must_use]
    pub fn average_fps(&self) -> f32 {
        let secs = self.elapsed().as_secs_f32();
        if secs <= 0.0 {
            0.0
        } else {
            self.total_frames as f32 / secs
        }
    }

    /// Duration of the frame just recorded, in seconds.
    #[must_use]
    pub fn last_frame_secs(&self) -> f32 {
        self.last_frame_secs
    }

    /// The whole run's frame-time distribution — mean, sd, min, p50, p95, p99, p99.9, max.
    ///
    /// [`FrameStats::average_fps`] is a mean and therefore says nothing about stutter; this is the
    /// number to quote for anything about smoothness.
    #[must_use]
    pub fn frame_time(&self) -> FrameTimeSummary {
        self.frame_times.summary()
    }

    /// The raw histogram, for a caller that wants a percentile this summary does not carry.
    #[must_use]
    pub fn frame_times(&self) -> &FrameTimeHistogram {
        &self.frame_times
    }
}

/// Paces the frame loop to a target frame rate.
///
/// java: `Minecraft.runGameLoop` — `Display.sync(this.getLimitFramerate())`
///
/// Keeps an absolute deadline rather than sleeping for "target minus how long the last frame
/// took". The naive version double-counts its own sleep and settles at roughly twice the
/// requested rate, which is exactly the bug this type exists to not have.
#[derive(Debug, Clone, Default)]
pub struct FrameLimiter {
    next: Option<std::time::Instant>,
    active_limit: Option<u32>,
    oversleep: OversleepEstimate,
}

/// How late `std::thread::sleep` has been waking up recently (PRESENT-8), which is how much of
/// the wait [`FrameLimiter::wait`] must spin instead of sleep.
///
/// The old fixed 1.5 ms margin assumed a 15.6 ms Windows scheduler tick. Rust's `thread::sleep`
/// waits on a high-resolution waitable timer (`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`) on Windows
/// 10 1803+ and falls back to `Sleep` only where that timer is unavailable (std
/// `sys/thread/windows.rs` in the 1.97 toolchain this workspace builds with), so the margin was
/// mostly busy-waiting: up to 1.5 ms of a core per capped frame, 18 % at 120 FPS, taken from
/// boost clocks and from the terrain workers.
///
/// This is LWJGL `Sync`'s idea (it keeps a running average of `Thread.sleep(1)` durations and
/// sleeps only while the remaining time exceeds it) with a decaying **maximum** instead of an
/// average, so one slow wake raises the margin at once and it relaxes only gradually. It starts
/// at the old 1.5 ms, so the first frames pace exactly as before, and it is clamped to
/// `[MIN, MAX]`. Wake-up lateness is at most `oversleep − margin`, which the estimate keeps at
/// zero for any oversleep up to `MAX`; the deadline arithmetic is untouched.
#[derive(Debug, Clone, Copy)]
pub struct OversleepEstimate {
    /// Decaying maximum of observed oversleep.
    estimate: std::time::Duration,
}

impl OversleepEstimate {
    /// The pre-PRESENT-8 fixed margin, and the starting estimate.
    pub const INITIAL: std::time::Duration = std::time::Duration::from_micros(1500);
    /// Never spin less than this: a wake can always be a little later than any recent one.
    pub const MIN_MARGIN: std::time::Duration = std::time::Duration::from_micros(250);
    /// Never spin more than this. A coarse timer (pre-1803 Windows, 15.6 ms) still wakes late
    /// by its excess over this, as it did with the old fixed margin, rather than spinning whole
    /// frames.
    pub const MAX_MARGIN: std::time::Duration = std::time::Duration::from_millis(2);
    /// Added to the estimate, for wakes slightly later than any recently observed.
    pub const SAFETY: std::time::Duration = std::time::Duration::from_micros(100);

    /// The spin margin to use now.
    #[must_use]
    pub fn margin(&self) -> std::time::Duration {
        (self.estimate + Self::SAFETY).clamp(Self::MIN_MARGIN, Self::MAX_MARGIN)
    }

    /// Record one sleep: `requested` was asked for, `actual` elapsed.
    pub fn record(&mut self, requested: std::time::Duration, actual: std::time::Duration) {
        let over = actual.saturating_sub(requested);
        // Decay by 1/16 per sample (≈ 11 samples to halve), but never below the new sample.
        let decayed = self.estimate - self.estimate / 16;
        self.estimate = over.max(decayed).min(Self::MAX_MARGIN);
    }
}

impl Default for OversleepEstimate {
    fn default() -> Self {
        Self {
            estimate: Self::INITIAL,
        }
    }
}

impl FrameLimiter {
    /// A limiter that has not paced a frame yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Block until the next frame is due. `None` means uncapped and returns immediately.
    ///
    /// Call once per frame, after presenting. A frame that overruns its budget returns at once
    /// and the schedule resynchronises to now, so a slow patch costs its own time and no more, and
    /// cannot turn into an unbounded backlog of zero-length frames either.
    ///
    /// java: LWJGL `Sync.sync`, which `Display.sync` calls: wait for `nextFrame`, then
    /// `nextFrame = Math.max(nextFrame + NANOS_IN_SECOND / fps, getTime())`.
    pub fn wait(&mut self, limit: Option<u32>) {
        let limit = limit.filter(|l| *l > 0);
        if self.active_limit != limit {
            self.next = None;
            self.active_limit = limit;
        }
        let Some(limit) = limit else {
            self.next = None;
            return;
        };
        let period = std::time::Duration::from_secs_f64(1.0 / f64::from(limit));
        let deadline = self
            .next
            .unwrap_or_else(|| std::time::Instant::now() + period);

        // `thread::sleep` alone overshoots by however late the OS wakes us. Sleep most of the
        // way, then spin the last fraction of a millisecond; how large that fraction must be is
        // measured rather than assumed (see `OversleepEstimate`, PRESENT-8).
        loop {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline - now;
            let margin = self.oversleep.margin();
            if remaining > margin {
                let request = remaining - margin;
                std::thread::sleep(request);
                self.oversleep.record(request, now.elapsed());
            } else {
                std::hint::spin_loop();
            }
        }
        self.next = Some((deadline + period).max(std::time::Instant::now()));
    }

    /// Forget the deadline — after a load screen or a settings change.
    pub fn reset(&mut self) {
        self.next = None;
        self.active_limit = None;
    }
}

/// Vanilla's default field of view, in **vertical** degrees. java: `GameSettings.fovSetting`
pub const DEFAULT_FOV: f32 = 70.0;

/// The fixed base of the hand projection, in **vertical** degrees.
///
/// java: `EntityRenderer.getFOVModifier(partialTicks, false)` starts from `70.0F` and never reads
/// `gameSettings.fovSetting` or `fovModifierHand`. `renderHand` builds its projection from it, so
/// the held item keeps one size whatever the FOV slider, sprint, Speed or bow pull do.
pub const HAND_FOV: f32 = 70.0;

/// Near plane. java: `EntityRenderer.setupCameraTransform` passes `0.05F` to `gluPerspective`
pub const NEAR_PLANE: f32 = 0.05;

/// The view camera.
///
/// # Camera-relative rendering
///
/// [`Camera::pos`] is [`DVec3`] because the world is `f64`; a vanilla server will happily put you
/// at x = 29,999,984. An `f32` view matrix built from that position loses roughly a metre of
/// precision per 8 million blocks and geometry visibly swims. So [`Camera::view`] omits the
/// world-position translation, and geometry is submitted in camera-relative space. The client
/// may prepend local camera effects, including the first-person 0.1-block translation:
///
/// ```no_run
/// # use mc_render::frame::Camera;
/// # use glam::DVec3;
/// # let camera = Camera::default();
/// let world_pos = DVec3::new(29_999_984.0, 64.0, 12.0);
/// let vertex = camera.relative(world_pos); // f64 subtract, then demote to f32
/// ```
///
/// Every phase must do this. Retrofitting it later is painful, so it is baked into the seam now.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// View matrix without world-position translation. Multiply by camera-relative positions.
    pub view: Mat4,
    /// Right-handed perspective projection with a 0..1 depth range (wgpu convention).
    pub proj: Mat4,
    /// Eye position in world space, `f64`.
    pub pos: DVec3,
    /// Vanilla yaw in degrees: 0 faces +Z (south), increasing turns clockwise seen from above.
    pub yaw: f32,
    /// Vanilla pitch in degrees: negative looks up, positive looks down, clamped to ±90.
    pub pitch: f32,
    /// Field of view in **vertical** degrees, i.e. `GameSettings::fov` (default 70).
    pub fov: f32,
    /// Multiplier applied to [`Camera::fov`] — the hook for vanilla's sprint/speed/bow-pull
    /// modifier (`EntityRenderer.fovModifierHand`) and for the ×60/70 underwater squeeze.
    /// `1.0` is neutral. S13 drives this; S10 never sets it to anything but 1.0.
    pub fov_modifier: f32,
    /// Vertical FOV of the separate projection `EntityRenderer.renderHand` installs for the
    /// first-person hand, its `renderOverlays` and `renderWorldDirections`.
    ///
    /// java: `getFOVModifier(partialTicks, false)`: [`HAND_FOV`] with only the death zoom and the
    /// underwater `60/70` squeeze applied. Neither [`Camera::fov`] nor the sprint/speed/bow part
    /// of [`Camera::fov_modifier`] reaches it.
    pub hand_fov: f32,
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance. See [`Camera::set_view_distance_chunks`].
    pub far: f32,
    /// Viewport aspect ratio, width / height.
    pub aspect: f32,
    /// Sub-pixel projection offset for temporal anti-aliasing, in NDC units
    /// (2 / width = one pixel). [`Vec2::ZERO`], the default, means no jitter.
    ///
    /// Only the GPU camera uniform of the **world** phases sees it
    /// ([`crate::CameraUniform::new`]); [`PhaseList`](crate::PhaseList) uploads the unjittered
    /// matrices again before the first phase at or after
    /// [`PhaseOrder::TEMPORAL_AA`](crate::PhaseOrder::TEMPORAL_AA), so waypoint labels, the hand,
    /// overlays and the UI never move. [`Camera::proj`] and [`Camera::view_proj`] stay
    /// unjittered for CPU work (culling, picking, label placement). It is set per frame only
    /// while TAA runs.
    pub jitter: Vec2,
}

impl Default for Camera {
    fn default() -> Self {
        let mut cam = Self {
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov: DEFAULT_FOV,
            fov_modifier: 1.0,
            hand_fov: HAND_FOV,
            near: NEAR_PLANE,
            far: 256.0,
            aspect: 16.0 / 9.0,
            jitter: Vec2::ZERO,
        };
        cam.update();
        cam
    }
}

impl Camera {
    /// Set the viewport aspect ratio from a pixel size. Zero height is ignored.
    pub fn set_viewport(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }

    /// Set the far plane from a render distance in chunks, the way vanilla does:
    /// `renderDistanceChunks * 16 * sqrt(2)`.
    ///
    /// java: `EntityRenderer.setupCameraTransform` — `farPlaneDistance * MathHelper.SQRT_2`
    pub fn set_view_distance_chunks(&mut self, chunks: u32) {
        // Saturating: an unclamped value from a damaged options.txt
        // never overflows (debug builds panicked on it).
        self.far = (chunks.max(2).saturating_mul(16)) as f32 * std::f32::consts::SQRT_2;
    }

    /// Apply a mouse look delta in degrees, clamping pitch to vanilla's ±90.
    ///
    /// java: `Entity.setAngles` (the `0.15` factor lives in
    /// [`InputState::look_delta`](crate::input::InputState::look_delta))
    pub fn turn(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-90.0, 90.0);
    }

    /// Apply the raw sensitivity-scaled mouse arguments passed to `Entity.setAngles`.
    ///
    /// Java widens both the existing `float` rotation and the raw `float` argument to `double`,
    /// multiplies the latter by the `0.15D` literal, adds in double precision, and only then
    /// narrows the updated rotation. Keeping this operation here avoids losing a low bit by
    /// pre-scaling the delta to `f32` in the input layer.
    pub fn turn_mouse(&mut self, raw_yaw: f32, raw_pitch: f32) {
        self.yaw = (f64::from(self.yaw) + f64::from(raw_yaw) * 0.15) as f32;
        self.pitch =
            ((f64::from(self.pitch) + f64::from(raw_pitch) * 0.15) as f32).clamp(-90.0, 90.0);
    }

    /// The FOV actually used for the projection: [`Camera::fov`] × [`Camera::fov_modifier`].
    #[must_use]
    pub fn effective_fov(&self) -> f32 {
        (self.fov * self.fov_modifier).clamp(1.0, 179.0)
    }

    /// Recompute [`Camera::view`] and [`Camera::proj`] from the scalar fields. Apply any local
    /// camera effects afterward; rebuilding here discards those prepared matrix transforms.
    pub fn update(&mut self) {
        // java: EntityRenderer.orientCamera — glRotatef(pitch, 1,0,0) then glRotatef(yaw+180, 0,1,0),
        // which composes to Rx * Ry. Translation is deliberately omitted; see the type docs.
        self.view = Mat4::from_rotation_x(self.pitch.to_radians())
            * Mat4::from_rotation_y((self.yaw + 180.0).to_radians());
        // wgpu's NDC is Y-up with a 0..1 depth range — the "directx" convention in glam's naming.
        // `Mat4::perspective_rh` is the deprecated spelling of the same thing.
        self.proj = glam::camera::rh::proj::directx::perspective(
            self.effective_fov().to_radians(),
            self.aspect.max(f32::EPSILON),
            self.near,
            self.far,
        );
    }

    /// `proj * view`, ready to multiply camera-relative positions with.
    #[must_use]
    pub fn view_proj(&self) -> Mat4 {
        self.proj * self.view
    }

    /// [`Camera::proj`] shifted by [`Camera::jitter`]: a clip-space translation,
    /// so after the perspective divide every point moves by exactly `jitter` in NDC. Equal to
    /// `proj`, bit for bit, when the jitter is zero.
    #[must_use]
    pub fn jittered_proj(&self) -> Mat4 {
        if self.jitter == Vec2::ZERO {
            return self.proj;
        }
        Mat4::from_translation(self.jitter.extend(0.0)) * self.proj
    }

    /// Convert a world position to the camera-relative `f32` space phases render in.
    #[must_use]
    pub fn relative(&self, world: DVec3) -> Vec3 {
        (world - self.pos).as_vec3()
    }

    /// Unit vector the camera is looking along, in world space.
    ///
    /// java: `Entity.getLook`
    #[must_use]
    pub fn look_vec(&self) -> DVec3 {
        let yaw = f64::from(self.yaw).to_radians();
        let pitch = f64::from(self.pitch).to_radians();
        let cos_pitch = pitch.cos();
        DVec3::new(-yaw.sin() * cos_pitch, -pitch.sin(), yaw.cos() * cos_pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stepped(step_ms: i64, steps: i64) -> (i32, Timer) {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.reset_clocks(0, 0);
        let mut total = 0;
        for n in 1..=steps {
            t.update_with(n * step_ms, n * step_ms);
            total += t.elapsed_ticks();
        }
        (total, t)
    }

    #[test]
    fn twenty_ticks_per_second_of_wall_clock() {
        // 100 frames of 10 ms = 1.0 s => exactly 20 ticks, no catch-up clamping involved.
        let (ticks, timer) = stepped(10, 100);
        assert_eq!(ticks, 20);
        assert!(timer.render_partial_ticks() < 1.0);
    }

    #[test]
    fn partial_ticks_interpolate_between_ticks() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.reset_clocks(0, 0);
        t.update_with(25, 25); // half a tick at 20 TPS
        assert_eq!(t.elapsed_ticks(), 0);
        assert!((t.render_partial_ticks() - 0.5).abs() < 1e-4);
        t.update_with(50, 50);
        assert_eq!(t.elapsed_ticks(), 1);
        assert!(t.render_partial_ticks() < 1e-4);
    }

    #[test]
    fn catch_up_is_clamped_and_excess_is_dropped() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.reset_clocks(0, 0);
        // A 600 ms hitch accrues 12 ticks. Vanilla subtracts all 12 from the accumulator and
        // *then* clamps the report to 10, so the extra 2 are dropped, not carried.
        t.update_with(600, 600);
        assert_eq!(t.elapsed_ticks(), MAX_CATCH_UP_TICKS);
        assert!(t.render_partial_ticks() < 1e-4);
        t.update_with(601, 601);
        assert_eq!(t.elapsed_ticks(), 0);
    }

    #[test]
    fn a_stall_longer_than_a_second_yields_no_ticks_at_all() {
        // java: Timer.updateTimer — when the system-clock delta leaves the 0..1000 ms window,
        // `lastHRTime` is reassigned *before* the delta is taken, so d2 is exactly zero. A long
        // load screen therefore resumes cleanly instead of firing a burst. Surprising, but it is
        // what 1.8.9 does, and S13's movement replay has to match it.
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.reset_clocks(0, 0);
        t.update_with(5_000, 5_000);
        assert_eq!(t.elapsed_ticks(), 0);
        assert_eq!(t.render_partial_ticks(), 0.0);
        // and the clock has resynchronised, so the next normal frame ticks normally
        t.update_with(5_050, 5_050);
        assert_eq!(t.elapsed_ticks(), 1);
    }

    #[test]
    fn timer_speed_scales_tick_rate() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.reset_clocks(0, 0);
        t.timer_speed = 0.5;
        let mut total = 0;
        for n in 1..=100 {
            t.update_with(n * 10, n * 10);
            total += t.elapsed_ticks();
        }
        assert_eq!(total, 10); // half speed over one second
    }

    #[test]
    fn render_rate_is_decoupled_from_tick_rate() {
        // One second of wall clock is one second of game time whether it is spent on 1000 frames
        // or 100. `elapsedPartialTicks` is an f32 in Java too, so the integer counts may land one
        // either side of a tick boundary; the accumulated total is the invariant, not the count.
        let (fast, fast_timer) = stepped(1, 1000);
        let (slow, slow_timer) = stepped(10, 100);
        for (ticks, timer, label) in [
            (fast, &fast_timer, "1000 fps"),
            (slow, &slow_timer, "100 fps"),
        ] {
            let accumulated = ticks as f32 + timer.render_partial_ticks();
            assert!(
                (accumulated - 20.0).abs() < 1e-3,
                "{label}: {accumulated} ticks of game time in one second"
            );
        }
        assert!(fast.abs_diff(slow) <= 1, "fast {fast} vs slow {slow}");
    }

    #[test]
    fn paused_update_freezes_partial_ticks() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.reset_clocks(0, 0);
        t.update_with(25, 25);
        let frozen = t.render_partial_ticks();
        assert!(frozen > 0.0);
        t.update_paused();
        assert_eq!(t.render_partial_ticks(), frozen);
    }

    #[test]
    fn diagnostic_fifty_millisecond_step_is_one_exact_tick() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.advance_diagnostic_micros(DIAGNOSTIC_TICK_MICROS).unwrap();
        assert_eq!(t.elapsed_ticks(), 1);
        assert_eq!(t.render_partial_ticks(), 0.0);
        t.advance_diagnostic_micros(DIAGNOSTIC_TICK_MICROS).unwrap();
        assert_eq!(t.elapsed_ticks(), 1);
        assert_eq!(t.render_partial_ticks(), 0.0);
    }

    #[test]
    fn diagnostic_fractional_steps_accumulate_integer_microseconds() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.advance_diagnostic_micros(25_000).unwrap();
        assert_eq!(t.elapsed_ticks(), 0);
        assert_eq!(t.render_partial_ticks(), 0.5);
        t.advance_diagnostic_micros(25_000).unwrap();
        assert_eq!(t.elapsed_ticks(), 1);
        assert_eq!(t.render_partial_ticks(), 0.0);
        t.advance_diagnostic_micros(1).unwrap();
        assert_eq!(t.elapsed_ticks(), 0);
        assert!((t.render_partial_ticks() - 0.00002).abs() < 1e-7);
    }

    #[test]
    fn diagnostic_frozen_update_is_repeatable_and_preserves_remainder() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.advance_diagnostic_micros(25_000).unwrap();
        t.update_diagnostic_frozen();
        t.update_diagnostic_frozen();
        assert_eq!(t.elapsed_ticks(), 0);
        assert_eq!(t.render_partial_ticks(), 0.0);
        t.advance_diagnostic_micros(25_000).unwrap();
        assert_eq!(t.elapsed_ticks(), 1);
        assert_eq!(t.render_partial_ticks(), 0.0);
    }

    #[test]
    fn diagnostic_invalid_and_overflow_steps_leave_state_unchanged() {
        let mut t = Timer::new(TICKS_PER_SECOND);
        t.diagnostic_remainder_micros = 12_345;
        t.elapsed_ticks = 2;
        t.elapsed_partial_ticks = 0.25;
        t.render_partial_ticks = 0.25;
        let before = (
            t.diagnostic_remainder_micros,
            t.elapsed_ticks,
            t.elapsed_partial_ticks,
            t.render_partial_ticks,
        );
        assert_eq!(
            t.advance_diagnostic_micros(0),
            Err(DiagnosticTimerError::ZeroStep)
        );
        assert_eq!(
            (
                t.diagnostic_remainder_micros,
                t.elapsed_ticks,
                t.elapsed_partial_ticks,
                t.render_partial_ticks,
            ),
            before
        );

        t.diagnostic_remainder_micros = DIAGNOSTIC_TICK_MICROS - 1;
        let before_overflow = (
            t.diagnostic_remainder_micros,
            t.elapsed_ticks,
            t.elapsed_partial_ticks,
            t.render_partial_ticks,
        );
        assert_eq!(
            t.advance_diagnostic_micros(u64::MAX),
            Err(DiagnosticTimerError::Overflow)
        );
        assert_eq!(
            (
                t.diagnostic_remainder_micros,
                t.elapsed_ticks,
                t.elapsed_partial_ticks,
                t.render_partial_ticks,
            ),
            before_overflow
        );

        let mut too_many = Timer::new(TICKS_PER_SECOND);
        assert_eq!(
            too_many.advance_diagnostic_micros(11 * DIAGNOSTIC_TICK_MICROS),
            Err(DiagnosticTimerError::CatchUpLimit)
        );
        assert_eq!(too_many.elapsed_ticks(), 0);
        assert_eq!(too_many.render_partial_ticks(), 0.0);

        let mut wrong_rate = Timer::new(30.0);
        assert_eq!(
            wrong_rate.advance_diagnostic_micros(DIAGNOSTIC_TICK_MICROS),
            Err(DiagnosticTimerError::UnsupportedRate)
        );
        let mut wrong_speed = Timer::new(TICKS_PER_SECOND);
        wrong_speed.timer_speed = 0.5;
        assert_eq!(
            wrong_speed.advance_diagnostic_micros(DIAGNOSTIC_TICK_MICROS),
            Err(DiagnosticTimerError::UnsupportedSpeed)
        );
    }

    #[test]
    fn camera_forward_maps_to_negative_z_in_view_space() {
        for (yaw, pitch) in [(0.0, 0.0), (90.0, 0.0), (-137.0, 22.0), (45.0, -80.0)] {
            let mut cam = Camera {
                yaw,
                pitch,
                ..Camera::default()
            };
            cam.update();
            let look = cam.look_vec().as_vec3();
            let in_view = cam.view.transform_vector3(look);
            assert!(
                (in_view - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4,
                "yaw {yaw} pitch {pitch} -> {in_view:?}"
            );
        }
    }

    #[test]
    fn yaw_zero_faces_south() {
        // java: Entity.getLook — yaw 0 is +Z, which 1.8.9 calls south.
        let cam = Camera::default();
        let look = cam.look_vec();
        assert!((look - DVec3::new(0.0, 0.0, 1.0)).length() < 1e-9);
    }

    #[test]
    fn camera_relative_keeps_precision_far_from_origin() {
        let cam = Camera {
            pos: DVec3::new(29_999_984.0, 64.0, 29_999_984.0),
            ..Camera::default()
        };
        let rel = cam.relative(cam.pos + DVec3::new(0.25, 0.0, -0.5));
        assert_eq!(rel, Vec3::new(0.25, 0.0, -0.5));
    }

    #[test]
    fn pitch_clamps_to_ninety_degrees() {
        let mut cam = Camera::default();
        cam.turn(0.0, 400.0);
        assert_eq!(cam.pitch, 90.0);
        cam.turn(0.0, -400.0);
        assert_eq!(cam.pitch, -90.0);
    }

    #[test]
    fn yaw_accumulates_across_positive_and_negative_revolutions() {
        let mut cam = Camera::default();
        cam.turn(725.0, 0.0);
        assert_eq!(cam.yaw, 725.0, "Entity.setAngles does not reduce yaw");
        cam.turn(-1_090.0, 0.0);
        assert_eq!(cam.yaw, -365.0, "negative yaw is not reduced either");
    }

    #[test]
    fn mouse_turn_adds_the_double_scaled_delta_before_narrowing() {
        let mut cam = Camera {
            yaw: 0.01,
            pitch: -23.75,
            ..Camera::default()
        };
        let expected_yaw = (f64::from(cam.yaw) + 1.0_f64 * 0.15) as f32;
        let prematurely_narrowed = cam.yaw + (1.0_f64 * 0.15) as f32;
        assert_ne!(
            expected_yaw.to_bits(),
            prematurely_narrowed.to_bits(),
            "negative control must distinguish Java's evaluation order"
        );

        cam.turn_mouse(1.0, -3.0);
        assert_eq!(cam.yaw.to_bits(), expected_yaw.to_bits());
        assert_eq!(
            cam.pitch.to_bits(),
            ((f64::from(-23.75_f32) - 3.0_f64 * 0.15) as f32).to_bits()
        );

        // The sign does not turn this into an algebraically equivalent f32 operation.  Keep an
        // opposite-direction control so a future "simplification" cannot pass only because the
        // positive-side rounding happened to agree.
        let mut opposite = Camera {
            yaw: -0.01,
            pitch: 23.75,
            ..Camera::default()
        };
        let expected_yaw = (f64::from(opposite.yaw) - 1.0_f64 * 0.15) as f32;
        let prematurely_narrowed = opposite.yaw - (1.0_f64 * 0.15) as f32;
        assert_ne!(expected_yaw.to_bits(), prematurely_narrowed.to_bits());

        opposite.turn_mouse(-1.0, 3.0);
        assert_eq!(opposite.yaw.to_bits(), expected_yaw.to_bits());
        assert_eq!(
            opposite.pitch.to_bits(),
            ((f64::from(23.75_f32) + 3.0_f64 * 0.15) as f32).to_bits()
        );
    }

    #[test]
    fn mouse_filter_follows_vanillas_three_register_sequence() {
        let mut filter = MouseFilter::default();
        // These are the literal values from `MouseFilter.smooth`, not an exponential-filter
        // approximation: 2 at a 0.05 factor first emits 0.05, then 0.12375.
        assert_eq!(filter.smooth(2.0, 0.05).to_bits(), 0.05_f32.to_bits());
        assert_eq!(filter.smooth(2.0, 0.05).to_bits(), 0.123_75_f32.to_bits());
        // Sign changes use `smoothing` as a clamp and cannot simply return the raw weighted
        // difference. This catches implementations that omit MouseFilter's final branch.
        assert_eq!(
            filter.smooth(-8.0, 0.05).to_bits(),
            f32::from_bits(0xBD2D_F3B8).to_bits()
        );
    }

    #[test]
    fn smooth_camera_filters_once_per_tick_and_partial_wrap_never_runs_backwards() {
        let mut camera = SmoothCamera::default();
        // Frames accumulate raw deltas but cannot manufacture a filtered output before the
        // tick-owned updateRenderer pass.
        assert_eq!(camera.apply(8.0, -4.0, 0.25, true), (0.0, 0.0));
        camera.tick(0.5, true);
        assert_eq!(camera.apply(0.0, 0.0, 0.25, true), (0.05, -0.025));
        assert_eq!(
            camera.apply(8.0, -4.0, 0.75, true),
            (0.1, -0.05),
            "the current tick holds one filter output across every render frame"
        );

        // The next game tick filters the values accumulated above and resets the partial baseline
        // before renderPartialTicks wraps from .75 to 0. A frame-side implementation instead
        // feeds a negative elapsed value here.
        camera.tick(0.5, true);
        assert_eq!(camera.apply(0.0, 0.0, 0.0, true), (0.0, -0.0));
        assert_eq!(
            camera.apply(0.0, 0.0, 0.25, true),
            (0.123_75, -0.061_875),
            "the second MouseFilter update is held and integrated from the reset baseline"
        );
    }

    #[test]
    fn smooth_camera_disable_resets_filters_and_dormant_raw_accumulators() {
        let mut camera = SmoothCamera::default();
        assert_eq!(camera.apply(8.0, -4.0, 0.5, true), (0.0, 0.0));
        camera.tick(0.5, true);
        assert_eq!(camera.apply(0.0, 0.0, 0.5, true), (0.1, -0.05));

        camera.tick(0.5, false);
        assert_eq!(camera.apply(3.0, 7.0, 0.75, false), (3.0, 7.0));
        camera.tick(0.5, true);
        assert_eq!(camera.apply(8.0, -4.0, 0.25, true), (0.0, 0.0));
        camera.tick(0.5, true);
        assert_eq!(
            camera.apply(0.0, 0.0, 0.25, true),
            (0.05, -0.025),
            "re-enabling starts from fresh MouseFilter registers"
        );
    }

    #[test]
    fn fov_modifier_scales_projection() {
        let mut cam = Camera::default();
        cam.update();
        let base = cam.proj;
        cam.fov_modifier = 1.15;
        cam.update();
        assert_ne!(base, cam.proj);
        assert!((cam.effective_fov() - 80.5).abs() < 1e-3);
    }

    #[test]
    fn view_distance_matches_vanilla_far_plane() {
        let mut cam = Camera::default();
        cam.set_view_distance_chunks(8);
        assert!((cam.far - 128.0 * std::f32::consts::SQRT_2).abs() < 1e-3);
    }

    /// `renderDistance:2147483647` in a damaged options.txt overflowed
    /// `chunks * 16` (a panic in debug builds, a garbage far plane in release).
    #[test]
    fn an_absurd_view_distance_saturates_instead_of_overflowing() {
        let mut cam = Camera::default();
        cam.set_view_distance_chunks(i32::MAX as u32);
        assert!(cam.far.is_finite() && cam.far > 0.0);
        cam.set_view_distance_chunks(u32::MAX);
        assert!(cam.far.is_finite() && cam.far > 0.0);
    }

    #[test]
    fn projection_uses_wgpu_depth_range() {
        let mut cam = Camera {
            near: 0.05,
            far: 100.0,
            ..Camera::default()
        };
        cam.update();
        let near_pt = cam.proj * glam::Vec4::new(0.0, 0.0, -0.05, 1.0);
        let far_pt = cam.proj * glam::Vec4::new(0.0, 0.0, -100.0, 1.0);
        assert!((near_pt.z / near_pt.w).abs() < 1e-4, "near should map to 0");
        assert!(
            (far_pt.z / far_pt.w - 1.0).abs() < 1e-3,
            "far should map to 1"
        );
    }

    #[test]
    fn frame_limiter_paces_to_the_requested_rate() {
        // Only a lower bound is asserted: an overloaded machine may be slower, but the limiter
        // must never run *faster* than the cap. Sleeping for "target minus last frame time"
        // fails this by roughly 2x, which is the regression this guards.
        let mut limiter = FrameLimiter::new();
        let start = std::time::Instant::now();
        const FRAMES: u32 = 12;
        for _ in 0..FRAMES {
            limiter.wait(Some(60));
        }
        let elapsed = start.elapsed().as_secs_f64();
        let expected = f64::from(FRAMES) / 60.0;
        assert!(
            elapsed >= expected * 0.9,
            "{FRAMES} frames capped at 60 fps took {elapsed:.4}s, expected ~{expected:.4}s"
        );
    }

    /// PRESENT-8: the spin margin starts at the old fixed 1.5 ms, rises at once to cover a late
    /// wake, relaxes only gradually, and stays inside its clamp.
    #[test]
    fn oversleep_estimate_starts_at_the_old_margin_and_adapts() {
        use std::time::Duration;
        let us = Duration::from_micros;
        let mut est = OversleepEstimate::default();
        assert_eq!(
            est.margin(),
            us(1600),
            "first frames: old 1.5 ms plus safety"
        );

        // A precise timer (50 µs late every time) walks the margin down to the floor.
        for _ in 0..200 {
            est.record(us(5000), us(5050));
        }
        assert_eq!(est.margin(), OversleepEstimate::MIN_MARGIN);

        // One 900 µs-late wake must be covered by the very next margin.
        est.record(us(5000), us(5900));
        assert!(est.margin() >= us(900) + OversleepEstimate::SAFETY);
        // ...and it decays, rather than snapping back after one good sample.
        est.record(us(5000), us(5050));
        assert!(est.margin() > us(800));

        // A coarse timer (15.6 ms late) is clamped: spin at most MAX, as the old fixed margin
        // capped it at 1.5 ms.
        est.record(us(5000), us(20_600));
        assert_eq!(est.margin(), OversleepEstimate::MAX_MARGIN);
        // Waking early (never happens with `thread::sleep`, but saturate anyway) is no oversleep.
        let mut early = OversleepEstimate::default();
        early.record(us(5000), us(1000));
        assert!(early.margin() <= us(1600));
    }

    #[test]
    fn frame_limiter_is_a_no_op_when_uncapped() {
        let mut limiter = FrameLimiter::new();
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            limiter.wait(None);
        }
        assert!(start.elapsed() < std::time::Duration::from_millis(50));
    }

    #[test]
    fn frame_limiter_resyncs_after_an_overrun() {
        use std::time::{Duration, Instant};
        const PERIOD_MS: u64 = 50;
        let limit = Some(1000 / PERIOD_MS as u32);
        let mut limiter = FrameLimiter::new();
        limiter.wait(limit);
        // A frame that blew its budget by several periods.
        std::thread::sleep(Duration::from_millis(PERIOD_MS * 4));

        // It has already paid for itself, so it must not wait a further period on top.
        let start = Instant::now();
        limiter.wait(limit);
        assert!(
            start.elapsed() < Duration::from_millis(PERIOD_MS / 2),
            "a late frame waited {:?} after it had already overrun",
            start.elapsed()
        );

        // Nor may the overrun bank credit: pacing resumes after at most one unthrottled frame.
        // The clock started before that wait set its deadline, so this bound has no slack to allow.
        for _ in 0..3 {
            limiter.wait(limit);
        }
        assert!(
            start.elapsed() >= Duration::from_millis(PERIOD_MS * 2),
            "three frames after an overrun took only {:?}",
            start.elapsed()
        );
    }

    /// `W` of work per frame against a shorter period costs `W`, not `W + period`.
    #[test]
    fn sustained_overrun_adds_no_wait() {
        use std::time::{Duration, Instant};
        let mut limiter = FrameLimiter::new();
        limiter.wait(Some(100));
        let mut waited = Duration::ZERO;
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(30));
            let start = Instant::now();
            limiter.wait(Some(100));
            waited += start.elapsed();
        }
        // 30 ms of work per frame against a 10 ms period. The old limiter waited 5 x 10 ms on top.
        assert!(
            waited < Duration::from_millis(25),
            "five 30 ms frames at a 100 fps cap also waited {waited:?}"
        );
    }

    #[test]
    fn frame_limiter_resets_deadlines_when_the_effective_cap_changes() {
        let mut limiter = FrameLimiter::new();
        limiter.active_limit = Some(60);
        limiter.next = Some(std::time::Instant::now() + std::time::Duration::from_millis(100));

        let start = std::time::Instant::now();
        limiter.wait(Some(1000));
        assert!(
            start.elapsed() < std::time::Duration::from_millis(30),
            "a finite-cap transition must not inherit the old future deadline"
        );
        assert_eq!(limiter.active_limit, Some(1000));
        assert!(limiter.next.is_some());

        limiter.wait(None);
        assert_eq!(limiter.active_limit, None);
        assert!(limiter.next.is_none());

        limiter.wait(Some(120));
        limiter.reset();
        assert_eq!(limiter.active_limit, None);
        assert!(limiter.next.is_none());
    }

    #[test]
    fn frame_stats_counts_frames() {
        let mut s = FrameStats::new();
        for _ in 0..5 {
            s.frame();
        }
        assert_eq!(s.total_frames(), 5);
        assert!(s.average_fps() > 0.0);
        assert_eq!(
            s.frame_time().frames,
            5,
            "every counted frame must also reach the distribution"
        );
    }

    /// The contract the F3 overlay's chunk-update counter leans on: `fps_updates` counts exactly
    /// the frames where `frame()` reported the window rolling, so a caller holding only a
    /// `&FrameStats` can put its own per-second total on the same window the FPS came from.
    #[test]
    fn fps_updates_counts_exactly_the_windows_frame_reports() {
        let mut s = FrameStats::new();
        assert_eq!(s.fps_updates(), 0, "no window has closed yet");
        assert_eq!(s.fps(), 0);

        // Just over a second, in coarse steps — the assertion is the agreement between the two,
        // not how many windows a given machine's sleep granularity fits into the loop.
        let mut rolls = 0;
        for _ in 0..21 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            if s.frame() {
                rolls += 1;
            }
        }
        assert!(rolls >= 1, "a second of sleeping must close a window");
        assert_eq!(s.fps_updates(), rolls);
        assert!(s.fps() > 0, "the closed window reported its frame count");
    }

    /// T0.3. A known distribution with a known answer: the integers 1..=1000 µs, one sample each.
    /// Every one is below `HIST_SUB`, so the histogram is exact here and the expected values are
    /// arithmetic, not approximate — nearest-rank picks the `ceil(q·n)`-th smallest.
    #[test]
    fn percentiles_match_a_known_distribution_exactly() {
        let mut h = FrameTimeHistogram::new();
        for us in 1..=1000u64 {
            h.record_micros(us);
        }
        assert_eq!(h.count(), 1000);
        assert_eq!(h.min_micros(), 1);
        assert_eq!(h.max_micros(), 1000);
        assert_eq!(h.percentile_micros(0.50), 500);
        assert_eq!(h.percentile_micros(0.95), 950);
        assert_eq!(h.percentile_micros(0.99), 990);
        assert_eq!(h.percentile_micros(0.999), 999);
        assert_eq!(h.percentile_micros(1.0), 1000);
        assert_eq!(h.percentile_micros(0.0), 1, "q=0 is the smallest sample");
        // Mean and sd come from the running sums, so they are exact regardless of binning.
        assert!(
            (h.mean_micros() - 500.5).abs() < 1e-9,
            "{}",
            h.mean_micros()
        );
        // Population sd of 1..=1000 is sqrt((n^2-1)/12) = sqrt(83333.25) ≈ 288.675.
        assert!(
            (h.sd_micros() - 288.674_9).abs() < 1e-3,
            "{}",
            h.sd_micros()
        );
    }

    /// The reason T0.3 exists, stated as an assertion: a mean cannot see a tail.
    ///
    /// **Control** — if `percentile_micros` were silently returning the mean (or anything else
    /// insensitive to the tail) the p99.9 assertion below would fail, because the mean of this
    /// distribution is 1.49 ms while its p99.9 is 50 ms. Both are asserted, so the test can tell
    /// the two apart rather than merely agreeing with itself.
    #[test]
    fn the_mean_hides_a_tail_that_the_percentiles_report() {
        let mut h = FrameTimeHistogram::new();
        for _ in 0..990 {
            h.record_micros(1_000);
        }
        for _ in 0..10 {
            h.record_micros(50_000);
        }
        let s = h.summary();
        assert_eq!(s.frames, 1000);
        assert!((s.mean_ms - 1.49).abs() < 1e-6, "{}", s.mean_ms);
        assert!((s.p50_ms - 1.0).abs() < 1e-6, "{}", s.p50_ms);
        assert!((s.p99_ms - 1.0).abs() < 1e-6, "{}", s.p99_ms);
        assert!((s.p999_ms - 50.0).abs() < 1e-6, "{}", s.p999_ms);
        assert!((s.max_ms - 50.0).abs() < 1e-6, "{}", s.max_ms);
        assert!(
            s.p999_ms > 30.0 * s.mean_ms,
            "the tail must be visible in the percentile and invisible in the mean: \
             mean {} ms, p99.9 {} ms",
            s.mean_ms,
            s.p999_ms
        );
    }

    /// A percentile must never contradict the extremes printed beside it.
    #[test]
    fn percentiles_stay_inside_min_and_max() {
        let mut h = FrameTimeHistogram::new();
        // Deliberately spans several magnitudes, including one past the exact region.
        for us in [
            271, 1_050, 2_783, 3_160, 4_001, 9_999, 40_000, 400_000, 4_000_000,
        ] {
            h.record_micros(us);
        }
        let s = h.summary();
        for (label, value) in [
            ("p50", s.p50_ms),
            ("p95", s.p95_ms),
            ("p99", s.p99_ms),
            ("p99.9", s.p999_ms),
        ] {
            assert!(
                value >= s.min_ms && value <= s.max_ms,
                "{label} = {value} ms is outside [{}, {}]",
                s.min_ms,
                s.max_ms
            );
        }
        assert!((s.min_ms - 0.271).abs() < 1e-9);
        assert!((s.max_ms - 4000.0).abs() < 1e-9);
    }

    /// The documented accuracy bound, asserted rather than claimed.
    #[test]
    fn binning_error_stays_within_the_documented_bound() {
        // Exact below 1.024 ms.
        for us in [0u64, 1, 271, 999, 1023] {
            assert_eq!(hist_bin_value(hist_bin_of(us)), us, "{us} must be exact");
        }
        // Within 1/1024 above it, over the range a frame time can plausibly take.
        for us in [
            1_024u64, 1_025, 2_783, 3_160, 16_667, 50_000, 999_999, 60_000_000,
        ] {
            let reported = hist_bin_value(hist_bin_of(us));
            let error = reported.abs_diff(us) as f64 / us as f64;
            assert!(
                error <= 1.0 / 1024.0,
                "{us} µs reported as {reported} µs — {:.4} % error exceeds the documented 0.098 %",
                error * 100.0
            );
        }
    }

    /// Beyond 2^30 µs the bins saturate, but `max` is kept outside the histogram and stays exact.
    #[test]
    fn samples_past_the_top_magnitude_saturate_without_losing_the_maximum() {
        let mut h = FrameTimeHistogram::new();
        h.record_micros(1_000);
        h.record_micros(4_000_000_000);
        assert_eq!(h.max_micros(), 4_000_000_000);
        assert_eq!(h.percentile_micros(1.0), 4_000_000_000);
        assert_eq!(h.percentile_micros(0.5), 1_000);
    }

    #[test]
    fn an_empty_distribution_reports_zeroes_rather_than_panicking() {
        let h = FrameTimeHistogram::new();
        let s = h.summary();
        assert_eq!(s.frames, 0);
        assert_eq!(s.p99_ms, 0.0);
        assert_eq!(s.max_ms, 0.0);
        assert_eq!(s.min_ms, 0.0);
        assert_eq!(s.sd_ms, 0.0);
    }

    /// The histogram is fixed-size by construction; this pins the number the docs quote.
    #[test]
    fn the_histogram_is_bounded() {
        assert_eq!(HIST_BINS, 11_776);
        assert_eq!(HIST_BINS * std::mem::size_of::<u32>(), 47_104);
        let mut h = FrameTimeHistogram::new();
        for us in 0..200_000u64 {
            h.record_micros(us);
        }
        assert_eq!(
            h.bins.len(),
            HIST_BINS,
            "recording must never grow the bins"
        );
        assert_eq!(h.count(), 200_000);
    }
}
