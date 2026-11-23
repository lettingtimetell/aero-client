//! Opt-in, bounded frame timing capture.
//!
//! The recorder is deliberately independent of the shell and UI.  A caller marks phase
//! durations for one draw, reports completed presents, and calls [`FrameTelemetry::finish`] when
//! the process is shutting down.  No file is opened while frames are being recorded: raw samples
//! remain in a bounded, preallocated `Vec` and are written once at finalisation.
//!
//! Set `MC_FRAME_TRACE` to a CSV path to enable capture.  `MC_FRAME_TRACE_MAX_SAMPLES` controls
//! the positive sample bound and defaults to [`DEFAULT_MAX_SAMPLES`].  Values above
//! [`MAX_SAMPLES`] are rejected.  Reaching the bound invalidates the capture explicitly rather
//! than silently dropping data; the metadata sidecar records the overflow and dropped count.
//!
//! The frame CSV is compatible with `mc-rs/optimization/frame_metrics.py`:
//! `frame_index,frame_ms,phase,event`.  A second sidecar, `<path>.phases.csv`, carries the
//! per-draw phase durations, and an optional `<path>.work.csv` carries post-render work counters.
//! `frame_ms` is always the elapsed time between completed presents; phase durations are
//! independent observations and must not be summed to recreate that interval.

use std::env;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Default maximum number of completed present-to-present intervals retained in memory.
pub const DEFAULT_MAX_SAMPLES: usize = 262_144;

/// Largest accepted sample bound, preventing an environment variable from requesting an
/// unreasonably large allocation.  See [`RAW_SAMPLE_BYTES`] and
/// [`DEFAULT_SAMPLE_BUFFER_BYTES`] for the exact target-dependent retention cost.
pub const MAX_SAMPLES: usize = 2_000_000;

const METADATA_VERSION: u32 = 1;

/// The ordered phase names emitted by the phase sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePhase {
    /// Work before fixed-rate ticks are run.
    BeforeTicks,
    /// Fixed-rate game tick processing.
    Ticks,
    /// Application frame preparation and scene submission.
    AppFrame,
    /// Camera update and finalisation.
    Camera,
    /// Swap-chain image acquisition.
    Acquire,
    /// Render command encoding and submission.
    Render,
    /// Work after rendering and before presentation.
    AfterRender,
    /// Presentation of the completed render target.
    Present,
    /// Frame pacing after the completed present.
    Limiter,
}

impl FramePhase {
    const fn name(self) -> &'static str {
        match self {
            Self::BeforeTicks => "before_ticks",
            Self::Ticks => "ticks",
            Self::AppFrame => "app_frame",
            Self::Camera => "camera",
            Self::Acquire => "acquire",
            Self::Render => "render",
            Self::AfterRender => "after_render",
            Self::Present => "present",
            Self::Limiter => "limiter",
        }
    }
}

/// Phase durations for one draw, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhaseTimings {
    /// Duration of work before fixed-rate ticks.
    pub before_ticks_ms: f64,
    /// Duration of fixed-rate game tick processing.
    pub ticks_ms: f64,
    /// Duration of application frame preparation and scene submission.
    pub app_frame_ms: f64,
    /// Duration of camera update and finalisation.
    pub camera_ms: f64,
    /// Duration of swap-chain image acquisition.
    pub acquire_ms: f64,
    /// Duration of render command encoding and submission.
    pub render_ms: f64,
    /// Duration of work after rendering and before presentation.
    pub after_render_ms: f64,
    /// Duration spent presenting the completed render target.
    pub present_ms: f64,
    /// Duration of frame pacing after the completed present.
    pub limiter_ms: f64,
}

/// Optional post-render work counters captured alongside one completed frame interval.
///
/// `None` means that the corresponding instrumentation was unavailable for that frame.  Writers
/// preserve it as an empty CSV cell rather than converting it to zero, because zero is an observed
/// count and absence is a different fact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameWorkStats {
    /// Number of loaded terrain sections.
    pub terrain_loaded_sections: Option<u64>,
    /// Number of terrain sections visible to the renderer.
    pub terrain_visible_sections: Option<u64>,
    /// Number of terrain draw calls.
    pub terrain_draw_calls: Option<u64>,
    /// Number of terrain draw items submitted.
    pub terrain_draw_items: Option<u64>,
    /// Number of terrain vertices or quads emitted by the renderer.
    pub terrain_quads: Option<u64>,
    /// Resident terrain bytes.
    pub terrain_resident_bytes: Option<u64>,
    /// Number of mesh jobs pending admission.
    pub mesh_pending: Option<u64>,
    /// Number of mesh jobs deferred by a budget or queue policy.
    pub mesh_deferred: Option<u64>,
    /// Number of mesh jobs currently in flight.
    pub mesh_in_flight: Option<u64>,
    /// Entity vertices prepared for the frame.
    pub entity_vertices: Option<u64>,
    /// Entity batches prepared for the frame.
    pub entity_batches: Option<u64>,
    /// Entity batch finalizations performed for the frame.
    pub entity_finish_calls: Option<u64>,
    /// Entity upload bytes prepared for the frame.
    pub entity_upload_bytes: Option<u64>,
    /// Inclusive UI preparation/submission CPU nanoseconds after selecting the UI phase.
    pub ui_build_ns: Option<u64>,
    /// UI vertices in submitted foreground and panorama seed lists.
    pub ui_vertices: Option<u64>,
    /// UI batches in submitted foreground and panorama seed lists.
    pub ui_batches: Option<u64>,
    /// Builtin tile items in submitted UI lists.
    pub ui_builtin_items: Option<u64>,
    /// UI vertex payload bytes submitted to the GPU.
    pub ui_vertex_upload_bytes: Option<u64>,
    /// UI depth payload bytes submitted to the GPU.
    pub ui_depth_upload_bytes: Option<u64>,
    /// UI texture queue write calls, including panorama seed resources.
    pub ui_texture_upload_calls: Option<u64>,
    /// UI texture payload bytes submitted to the GPU.
    pub ui_texture_upload_bytes: Option<u64>,
    /// Actual foreground and panorama seed UI draw calls.
    pub ui_draw_calls: Option<u64>,
    /// New logical rebuilds admitted during this diagnostic frame.
    pub mesh_enqueued: Option<u64>,
    /// New rebuilds deferred because active admission was full.
    pub mesh_enqueue_overflow: Option<u64>,
    /// Deferred rebuilds moved into active membership this frame.
    pub mesh_deferred_promoted: Option<u64>,
    /// Rebuilds handed to asynchronous workers this frame.
    pub mesh_dispatched: Option<u64>,
    /// Rebuilds meshed synchronously this frame, including explicit replay.
    pub mesh_synchronous_built: Option<u64>,
    /// Accepted asynchronous worker results this frame.
    pub mesh_completed: Option<u64>,
    /// Accepted worker results exceeding the diagnostic slow threshold.
    pub mesh_slow_completed: Option<u64>,
    /// Logical rebuilds cancelled or superseded this frame.
    pub mesh_cancelled: Option<u64>,
    /// Worker channel results rejected for stale lifecycle revisions.
    pub mesh_stale_result_rejected: Option<u64>,
    /// Oldest active rebuild age in diagnostic frames; absent if unknown or empty.
    pub mesh_oldest_active_age_frames: Option<u64>,
    /// Oldest deferred rebuild age in diagnostic frames; absent if unknown or empty.
    pub mesh_oldest_deferred_age_frames: Option<u64>,
    /// Oldest running rebuild age in diagnostic frames; absent if unknown or empty.
    pub mesh_oldest_in_flight_age_frames: Option<u64>,
    /// Completed meshes retained for a later GPU upload.
    pub mesh_ready_uploads: Option<u64>,
    /// Worker results rejected by renderer revision checks before ready admission.
    pub mesh_input_revision_rejected: Option<u64>,
    /// Retained meshes rejected by renderer revision checks after dequeue.
    pub mesh_retained_revision_rejected: Option<u64>,
    /// One when the upload deadline retained work, zero when it did not.
    pub mesh_upload_deadline_blocked: Option<u64>,
}

impl PhaseTimings {
    /// Set one phase from a standard-library duration.
    pub fn set(&mut self, phase: FramePhase, duration: Duration) {
        self.set_ms(phase, duration.as_secs_f64() * 1_000.0);
    }

    /// Set one phase directly in milliseconds.
    ///
    /// This is useful for deterministic CPU-only tests.  Production callers should prefer
    /// [`PhaseTimings::set`] so a monotonic duration is supplied by the surrounding loop.
    pub fn set_ms(&mut self, phase: FramePhase, milliseconds: f64) {
        match phase {
            FramePhase::BeforeTicks => self.before_ticks_ms = milliseconds,
            FramePhase::Ticks => self.ticks_ms = milliseconds,
            FramePhase::AppFrame => self.app_frame_ms = milliseconds,
            FramePhase::Camera => self.camera_ms = milliseconds,
            FramePhase::Acquire => self.acquire_ms = milliseconds,
            FramePhase::Render => self.render_ms = milliseconds,
            FramePhase::AfterRender => self.after_render_ms = milliseconds,
            FramePhase::Present => self.present_ms = milliseconds,
            FramePhase::Limiter => self.limiter_ms = milliseconds,
        }
    }

    fn validate(self) -> Result<(), TelemetryError> {
        for (phase, milliseconds) in [
            (FramePhase::BeforeTicks, self.before_ticks_ms),
            (FramePhase::Ticks, self.ticks_ms),
            (FramePhase::AppFrame, self.app_frame_ms),
            (FramePhase::Camera, self.camera_ms),
            (FramePhase::Acquire, self.acquire_ms),
            (FramePhase::Render, self.render_ms),
            (FramePhase::AfterRender, self.after_render_ms),
            (FramePhase::Present, self.present_ms),
            (FramePhase::Limiter, self.limiter_ms),
        ] {
            if !milliseconds.is_finite() || milliseconds < 0.0 {
                return Err(TelemetryError::InvalidPhaseDuration(phase.name()));
            }
        }
        Ok(())
    }
}

/// Result of reporting a completed present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentResult {
    /// This was the first present and established the interval anchor.
    Anchored,
    /// A complete interval was retained.
    Recorded {
        /// Zero-based index of the completed interval, excluding the initial anchor.
        frame_index: u64,
    },
    /// A complete interval was observed after the bound was reached and was dropped.
    Dropped {
        /// Index that would have been retained if the sample bound had not been reached.
        frame_index: u64,
    },
}

/// Errors raised while configuring or finalising a frame trace.
#[derive(Debug)]
pub enum TelemetryError {
    /// The configured sample bound was zero, malformed, or above [`MAX_SAMPLES`].
    InvalidMaxSamples(String),
    /// A present timestamp went backwards or produced a zero-length interval.
    NonMonotonicPresent,
    /// A phase duration was not finite or was negative.
    InvalidPhaseDuration(&'static str),
    /// An output file could not be created or written.
    Io(io::Error),
}

impl std::fmt::Display for TelemetryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMaxSamples(value) => {
                write!(formatter, "invalid frame trace sample bound {value:?}")
            }
            Self::NonMonotonicPresent => {
                formatter.write_str("frame trace present timestamp is not increasing")
            }
            Self::InvalidPhaseDuration(phase) => {
                write!(formatter, "invalid frame trace duration for {phase}")
            }
            Self::Io(error) => write!(formatter, "frame trace output failed: {error}"),
        }
    }
}

impl std::error::Error for TelemetryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

/// One finalisation result, including validity facts needed by a benchmark coordinator.
#[derive(Debug, Clone)]
pub struct FinalizeReport {
    /// CSV path selected when the recorder was constructed.
    pub csv_path: PathBuf,
    /// Phase CSV sidecar path.
    pub phases_path: PathBuf,
    /// JSON metadata sidecar path.
    pub metadata_path: PathBuf,
    /// Optional per-frame work CSV sidecar path.
    pub work_path: PathBuf,
    /// Whether output completed after the caller reported a normal run.
    ///
    /// This is a structural output fact, not a benchmark qualification decision; acquire skips
    /// are retained as a separate rejection fact in the metadata.
    pub completed: bool,
    /// Number of frame intervals written to the frame CSV.
    pub count: usize,
    /// Number of completed intervals dropped after overflow.
    pub dropped: u64,
    /// Number of failed acquire attempts observed.
    pub acquire_skips: u64,
    /// Whether the configured bound was exceeded.
    pub overflowed: bool,
    /// Failure recorded during output, if any.
    pub failure: Option<String>,
    /// Configured maximum number of retained samples.
    pub max_samples: usize,
    /// Exact size of one retained raw sample on this target.
    pub raw_sample_bytes: usize,
    /// Exact allocation reserved for the retained sample vector.
    pub sample_buffer_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct RawSample {
    frame_index: u64,
    frame_ms: f64,
    phase: &'static str,
    phases: PhaseTimings,
    acquire_skips: u64,
    work: FrameWorkStats,
}

/// Exact size in bytes of one retained raw sample on the current target.
pub const RAW_SAMPLE_BYTES: usize = std::mem::size_of::<RawSample>();

/// Exact maximum retained-sample allocation at the default bound, in bytes.
pub const DEFAULT_SAMPLE_BUFFER_BYTES: usize = DEFAULT_MAX_SAMPLES * RAW_SAMPLE_BYTES;

/// Exact maximum retained-sample allocation at [`MAX_SAMPLES`], in bytes.
pub const MAX_SAMPLE_BUFFER_BYTES: usize = MAX_SAMPLES * RAW_SAMPLE_BYTES;

/// An opt-in bounded recorder for completed frame intervals and phase timings.
///
/// Construct this only when tracing is enabled.  The normal disabled path can therefore avoid
/// creating an `Instant`, allocating a `PathBuf`, or doing any recorder work.  Call
/// [`FrameTelemetry::begin_frame`] before marking phases, [`FrameTelemetry::present`] after a
/// successful present, and [`FrameTelemetry::record_phase`] after the limiter returns if the
/// limiter is ordered after present.  A later `present` interval still remains the authoritative
/// frame-time sample.
pub struct FrameTelemetry {
    csv_path: PathBuf,
    max_samples: usize,
    samples: Vec<RawSample>,
    current_phases: PhaseTimings,
    current_phase: &'static str,
    current_work: FrameWorkStats,
    last_recorded_slot: Option<usize>,
    last_present: Option<Instant>,
    last_present_duration: Option<Duration>,
    next_frame_index: u64,
    pending_acquire_skips: u64,
    acquire_skips: u64,
    dropped: u64,
    overflowed: bool,
    failure: Option<String>,
}

impl FrameTelemetry {
    /// Construct a recorder from the environment.
    ///
    /// `Ok(None)` means `MC_FRAME_TRACE` is absent or empty.  Invalid configuration is an error
    /// so a benchmark cannot silently run with an unintended bound.
    pub fn from_env() -> Result<Option<Self>, TelemetryError> {
        let Some(path) = env::var_os("MC_FRAME_TRACE") else {
            return Ok(None);
        };
        if path.is_empty() {
            return Ok(None);
        }
        let max_samples = match env::var_os("MC_FRAME_TRACE_MAX_SAMPLES") {
            Some(value) => parse_max_samples(value)?,
            None => DEFAULT_MAX_SAMPLES,
        };
        Ok(Some(Self::new(PathBuf::from(path), max_samples)?))
    }

    /// Construct a recorder for a path and a bounded number of samples.
    ///
    /// File creation is deferred until [`FrameTelemetry::finish`], and then uses exclusive create
    /// for all four output files.  This keeps the hot path free of synchronous file I/O while
    /// ensuring an existing artifact is never overwritten.
    pub fn new(path: impl AsRef<Path>, max_samples: usize) -> Result<Self, TelemetryError> {
        if max_samples == 0 || max_samples > MAX_SAMPLES {
            return Err(TelemetryError::InvalidMaxSamples(max_samples.to_string()));
        }
        Ok(Self {
            csv_path: path.as_ref().to_path_buf(),
            max_samples,
            samples: Vec::with_capacity(max_samples),
            current_phases: PhaseTimings::default(),
            current_phase: "shell",
            current_work: FrameWorkStats::default(),
            last_recorded_slot: None,
            last_present: None,
            last_present_duration: None,
            next_frame_index: 0,
            pending_acquire_skips: 0,
            acquire_skips: 0,
            dropped: 0,
            overflowed: false,
            failure: None,
        })
    }

    /// Begin the current draw and clear phase slots left by the previous draw.
    pub fn begin_frame(&mut self) {
        self.current_phases = PhaseTimings::default();
        self.current_phase = "shell";
        self.current_work = FrameWorkStats::default();
    }

    /// Set the stable diagnostic label written in the frame CSV's `phase` column.
    ///
    /// The label is borrowed as `&'static str` so recording a sample never allocates.  The shell
    /// can pass its `ShellApp::diagnostic_phase` result directly.
    pub fn set_phase_label(&mut self, phase: &'static str) {
        self.current_phase = phase;
    }

    /// Record one phase duration for the current draw.
    pub fn record_phase(
        &mut self,
        phase: FramePhase,
        duration: Duration,
    ) -> Result<(), TelemetryError> {
        let milliseconds = duration.as_secs_f64() * 1_000.0;
        self.record_phase_ms(phase, milliseconds)
    }

    /// Record one phase duration in milliseconds.
    ///
    /// This direct form supports deterministic tests without sleeping or consulting a clock.
    pub fn record_phase_ms(
        &mut self,
        phase: FramePhase,
        milliseconds: f64,
    ) -> Result<(), TelemetryError> {
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return Err(TelemetryError::InvalidPhaseDuration(phase.name()));
        }
        if phase == FramePhase::Limiter {
            if let Some(slot) = self.last_recorded_slot {
                self.samples[slot].phases.set_ms(phase, milliseconds);
            } else {
                self.current_phases.set_ms(phase, milliseconds);
            }
        } else {
            self.current_phases.set_ms(phase, milliseconds);
        }
        Ok(())
    }

    /// Count one failed swap-chain acquire attempt.
    ///
    /// Skips are carried into the next successful interval and are included in its event label and
    /// in the metadata total.  They do not advance the frame index and never reset gameplay time.
    pub fn acquire_skipped(&mut self) {
        self.pending_acquire_skips = self.pending_acquire_skips.saturating_add(1);
        self.acquire_skips = self.acquire_skips.saturating_add(1);
    }

    /// Record a fatal capture or loop failure for the metadata sidecar.
    ///
    /// Recording continues so already captured samples are preserved, but finalisation marks the
    /// artifact incomplete and ineligible for scoring.
    pub fn record_failure(&mut self, reason: impl Into<String>) {
        if self.failure.is_none() {
            self.failure = Some(reason.into());
        }
    }

    /// Report a successful present with an explicit diagnostic label and phase timings.
    pub fn record_present(
        &mut self,
        completed_present: Instant,
        phase: &'static str,
        timings: PhaseTimings,
    ) -> Result<PresentResult, TelemetryError> {
        self.record_present_with_work(completed_present, phase, timings, FrameWorkStats::default())
    }

    /// Report a successful present with phase timings and an optional post-render work snapshot.
    ///
    /// Work values are retained in the same bounded sample as the frame and phase rows.  Missing
    /// values remain missing in the work CSV rather than being changed into zeroes.
    pub fn record_present_with_work(
        &mut self,
        completed_present: Instant,
        phase: &'static str,
        timings: PhaseTimings,
        work: FrameWorkStats,
    ) -> Result<PresentResult, TelemetryError> {
        timings.validate()?;
        self.current_phase = phase;
        self.current_phases = timings;
        self.current_work = work;
        self.present(completed_present)
    }

    /// Report a successful present at an artificial timestamp with an explicit diagnostic label.
    ///
    /// This is the deterministic counterpart to [`FrameTelemetry::record_present`].
    pub fn record_present_at(
        &mut self,
        completed_present: Duration,
        phase: &'static str,
        timings: PhaseTimings,
    ) -> Result<PresentResult, TelemetryError> {
        self.record_present_at_with_work(
            completed_present,
            phase,
            timings,
            FrameWorkStats::default(),
        )
    }

    /// Report a successful present at an artificial timestamp with phase timings and work.
    ///
    /// This deterministic counterpart is intended for CPU-only tests and replay adapters.
    pub fn record_present_at_with_work(
        &mut self,
        completed_present: Duration,
        phase: &'static str,
        timings: PhaseTimings,
        work: FrameWorkStats,
    ) -> Result<PresentResult, TelemetryError> {
        timings.validate()?;
        self.current_phase = phase;
        self.current_phases = timings;
        self.current_work = work;
        self.present_at(completed_present)
    }

    /// Report a successful present using a monotonic clock value.
    pub fn present(&mut self, completed_present: Instant) -> Result<PresentResult, TelemetryError> {
        let Some(previous) = self.last_present else {
            self.last_present = Some(completed_present);
            self.last_present_duration = None;
            self.pending_acquire_skips = 0;
            self.current_phases = PhaseTimings::default();
            self.current_work = FrameWorkStats::default();
            self.last_recorded_slot = None;
            return Ok(PresentResult::Anchored);
        };
        let elapsed = completed_present
            .checked_duration_since(previous)
            .ok_or(TelemetryError::NonMonotonicPresent)?;
        self.last_present = Some(completed_present);
        self.finish_present(elapsed)
    }

    /// Report a successful present at an artificial monotonic timestamp.
    ///
    /// This is intended for CPU-only tests and replay adapters.  The first timestamp establishes
    /// the anchor; each later timestamp produces the exact elapsed present-to-present interval.
    pub fn present_at(
        &mut self,
        completed_present: Duration,
    ) -> Result<PresentResult, TelemetryError> {
        let Some(previous) = self.last_present_duration else {
            self.last_present_duration = Some(completed_present);
            self.last_present = None;
            self.pending_acquire_skips = 0;
            self.current_phases = PhaseTimings::default();
            self.current_work = FrameWorkStats::default();
            self.last_recorded_slot = None;
            return Ok(PresentResult::Anchored);
        };
        let elapsed = completed_present
            .checked_sub(previous)
            .ok_or(TelemetryError::NonMonotonicPresent)?;
        self.last_present_duration = Some(completed_present);
        self.finish_present(elapsed)
    }

    fn finish_present(&mut self, elapsed: Duration) -> Result<PresentResult, TelemetryError> {
        let frame_ms = elapsed.as_secs_f64() * 1_000.0;
        if !frame_ms.is_finite() || frame_ms <= 0.0 {
            return Err(TelemetryError::NonMonotonicPresent);
        }
        let frame_index = self.next_frame_index;
        self.next_frame_index = self.next_frame_index.saturating_add(1);
        let skipped = self.pending_acquire_skips;
        self.pending_acquire_skips = 0;
        let sample = RawSample {
            frame_index,
            frame_ms,
            phase: self.current_phase,
            phases: self.current_phases,
            acquire_skips: skipped,
            work: self.current_work,
        };
        self.current_phases = PhaseTimings::default();
        self.current_phase = "shell";
        self.current_work = FrameWorkStats::default();
        self.last_recorded_slot = None;

        if self.samples.len() >= self.max_samples {
            self.overflowed = true;
            self.dropped = self.dropped.saturating_add(1);
            return Ok(PresentResult::Dropped { frame_index });
        }
        self.samples.push(sample);
        self.last_recorded_slot = Some(self.samples.len() - 1);
        Ok(PresentResult::Recorded { frame_index })
    }

    /// Number of retained frame intervals.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Finalise the trace, writing the frame, phase, work, and JSON metadata sidecars.
    ///
    /// The metadata marks an overflowed or output-failed capture as incomplete and therefore
    /// ineligible for scoring.  Creation of every output uses `create_new`, so an existing CSV or
    /// sidecar causes an error instead of being replaced.
    pub fn finish(self) -> Result<FinalizeReport, TelemetryError> {
        self.finalize(true)
    }

    /// Finalise the trace, explicitly stating whether the surrounding run completed normally.
    ///
    /// Passing `false` preserves all raw samples but marks the metadata invalid for scoring.
    pub fn finalize(self, completed: bool) -> Result<FinalizeReport, TelemetryError> {
        let phases_path = sidecar_path(&self.csv_path, ".phases.csv");
        let metadata_path = sidecar_path(&self.csv_path, ".meta.json");
        let work_path = sidecar_path(&self.csv_path, ".work.csv");
        let mut frames_file = exclusive_create(&self.csv_path)?;
        let mut phases_file = exclusive_create(&phases_path)?;
        let mut metadata_file = exclusive_create(&metadata_path)?;
        let mut work_file = exclusive_create(&work_path)?;

        let mut failure = None;
        if let Err(error) = write_frames(&mut frames_file, &self.samples) {
            failure = Some(error.to_string());
        }
        if let Err(error) = write_phases(&mut phases_file, &self.samples) {
            if failure.is_none() {
                failure = Some(error.to_string());
            }
        }
        if let Err(error) = write_work(&mut work_file, &self.samples) {
            if failure.is_none() {
                failure = Some(error.to_string());
            }
        }

        let report = FinalizeReport {
            csv_path: self.csv_path,
            phases_path,
            metadata_path,
            work_path,
            completed: completed
                && failure.is_none()
                && self.failure.is_none()
                && !self.overflowed
                && !self.samples.is_empty(),
            count: self.samples.len(),
            dropped: self.dropped,
            acquire_skips: self.acquire_skips,
            overflowed: self.overflowed,
            failure: failure.or(self.failure),
            max_samples: self.max_samples,
            raw_sample_bytes: RAW_SAMPLE_BYTES,
            sample_buffer_bytes: self.max_samples.saturating_mul(RAW_SAMPLE_BYTES),
        };
        write_metadata(&mut metadata_file, &report).map_err(TelemetryError::Io)?;
        Ok(report)
    }
}

fn parse_max_samples(value: OsString) -> Result<usize, TelemetryError> {
    let rendered = value.to_string_lossy().into_owned();
    let parsed = rendered
        .parse::<usize>()
        .map_err(|_| TelemetryError::InvalidMaxSamples(rendered.clone()))?;
    if parsed == 0 || parsed > MAX_SAMPLES {
        return Err(TelemetryError::InvalidMaxSamples(rendered));
    }
    Ok(parsed)
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut sidecar = path.to_path_buf();
    let mut file_name = sidecar
        .file_name()
        .map(OsString::from)
        .unwrap_or_else(|| OsString::from("frame-trace"));
    file_name.push(suffix);
    sidecar.set_file_name(file_name);
    sidecar
}

fn exclusive_create(path: &Path) -> Result<File, TelemetryError> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(TelemetryError::Io)
}

fn write_frames(file: &mut File, samples: &[RawSample]) -> io::Result<()> {
    writeln!(file, "frame_index,frame_ms,phase,event")?;
    for sample in samples {
        let event = if sample.acquire_skips == 0 {
            "ok".to_owned()
        } else {
            format!("acquire_skips={}", sample.acquire_skips)
        };
        writeln!(
            file,
            "{},{:.9},{},{}",
            sample.frame_index,
            sample.frame_ms,
            csv_field(sample.phase),
            event
        )?;
    }
    file.flush()
}

fn write_phases(file: &mut File, samples: &[RawSample]) -> io::Result<()> {
    writeln!(
        file,
        "frame_index,before_ticks_ms,ticks_ms,app_frame_ms,camera_ms,acquire_ms,render_ms,after_render_ms,present_ms,limiter_ms"
    )?;
    for sample in samples {
        let p = sample.phases;
        writeln!(
            file,
            "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            sample.frame_index,
            p.before_ticks_ms,
            p.ticks_ms,
            p.app_frame_ms,
            p.camera_ms,
            p.acquire_ms,
            p.render_ms,
            p.after_render_ms,
            p.present_ms,
            p.limiter_ms
        )?;
    }
    file.flush()
}

const WORK_CSV_SCHEMA: &str = "frame_index,terrain_loaded_sections,terrain_visible_sections,terrain_draw_calls,terrain_draw_items,terrain_quads,terrain_resident_bytes,mesh_pending,mesh_deferred,mesh_in_flight,entity_vertices,entity_batches,entity_finish_calls,entity_upload_bytes,ui_build_ns,ui_vertices,ui_batches,ui_builtin_items,ui_vertex_upload_bytes,ui_depth_upload_bytes,ui_texture_upload_calls,ui_texture_upload_bytes,ui_draw_calls,mesh_enqueued,mesh_enqueue_overflow,mesh_deferred_promoted,mesh_dispatched,mesh_synchronous_built,mesh_completed,mesh_slow_completed,mesh_cancelled,mesh_stale_result_rejected,mesh_oldest_active_age_frames,mesh_oldest_deferred_age_frames,mesh_oldest_in_flight_age_frames,mesh_ready_uploads,mesh_input_revision_rejected,mesh_retained_revision_rejected,mesh_upload_deadline_blocked";

fn write_work(file: &mut File, samples: &[RawSample]) -> io::Result<()> {
    writeln!(file, "{WORK_CSV_SCHEMA}")?;
    for sample in samples {
        let work = sample.work;
        write!(file, "{}", sample.frame_index)?;
        for value in [
            work.terrain_loaded_sections,
            work.terrain_visible_sections,
            work.terrain_draw_calls,
            work.terrain_draw_items,
            work.terrain_quads,
            work.terrain_resident_bytes,
            work.mesh_pending,
            work.mesh_deferred,
            work.mesh_in_flight,
            work.entity_vertices,
            work.entity_batches,
            work.entity_finish_calls,
            work.entity_upload_bytes,
            work.ui_build_ns,
            work.ui_vertices,
            work.ui_batches,
            work.ui_builtin_items,
            work.ui_vertex_upload_bytes,
            work.ui_depth_upload_bytes,
            work.ui_texture_upload_calls,
            work.ui_texture_upload_bytes,
            work.ui_draw_calls,
            work.mesh_enqueued,
            work.mesh_enqueue_overflow,
            work.mesh_deferred_promoted,
            work.mesh_dispatched,
            work.mesh_synchronous_built,
            work.mesh_completed,
            work.mesh_slow_completed,
            work.mesh_cancelled,
            work.mesh_stale_result_rejected,
            work.mesh_oldest_active_age_frames,
            work.mesh_oldest_deferred_age_frames,
            work.mesh_oldest_in_flight_age_frames,
            work.mesh_ready_uploads,
            work.mesh_input_revision_rejected,
            work.mesh_retained_revision_rejected,
            work.mesh_upload_deadline_blocked,
        ] {
            match value {
                Some(value) => write!(file, ",{value}")?,
                None => file.write_all(b",")?,
            }
        }
        file.write_all(b"\n")?;
    }
    file.flush()
}

fn csv_field(value: &str) -> String {
    if value.bytes().all(|character| {
        character != b',' && character != b'"' && character != b'\n' && character != b'\r'
    }) {
        return value.to_owned();
    }
    let escaped = value.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

fn write_metadata(file: &mut File, report: &FinalizeReport) -> io::Result<()> {
    let failure = report
        .failure
        .as_deref()
        .map(json_string)
        .unwrap_or_else(|| "null".to_owned());
    writeln!(file, "{{")?;
    writeln!(file, "  \"version\": {},", METADATA_VERSION)?;
    writeln!(
        file,
        "  \"csv_schema\": \"frame_index,frame_ms,phase,event\","
    )?;
    writeln!(file, "  \"phase_schema\": \"frame_index,before_ticks_ms,ticks_ms,app_frame_ms,camera_ms,acquire_ms,render_ms,after_render_ms,present_ms,limiter_ms\",")?;
    writeln!(file, "  \"work_schema\": \"{WORK_CSV_SCHEMA}\",")?;
    writeln!(file, "  \"timing_definition\": \"frame_ms is the completed-present to completed-present interval; the first present establishes the anchor\",")?;
    writeln!(file, "  \"phase_timing_definition\": \"phase values are independent per-draw durations and are not required to sum to frame_ms; limiter may be recorded after present for the draw that just ended\",")?;
    writeln!(file, "  \"work_timing_definition\": \"work values are post-render snapshots for the recorded draw; unavailable instrumentation is an empty cell and work rows share frame_index with frame rows\",")?;
    writeln!(file, "  \"completed\": {},", report.completed)?;
    writeln!(file, "  \"scoring_eligible\": false,")?;
    writeln!(file, "  \"max_samples\": {},", report.max_samples)?;
    writeln!(file, "  \"raw_sample_bytes\": {},", report.raw_sample_bytes)?;
    writeln!(
        file,
        "  \"sample_buffer_bytes\": {},",
        report.sample_buffer_bytes
    )?;
    writeln!(file, "  \"count\": {},", report.count)?;
    writeln!(file, "  \"dropped\": {},", report.dropped)?;
    writeln!(file, "  \"acquire_skips\": {},", report.acquire_skips)?;
    writeln!(file, "  \"overflowed\": {},", report.overflowed)?;
    writeln!(
        file,
        "  \"invalidated_by_acquire_skips\": {},",
        report.acquire_skips > 0
    )?;
    writeln!(
        file,
        "  \"invalidated_by_overflow\": {},",
        report.overflowed
    )?;
    writeln!(file, "  \"invalidated_by_empty\": {},", report.count == 0)?;
    writeln!(file, "  \"failure\": {}", failure)?;
    writeln!(file, "}}")?;
    file.flush()
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn test_path(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "mc-frame-telemetry-{label}-{}-{id}.csv",
            std::process::id()
        ))
    }

    fn remove_outputs(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(sidecar_path(path, ".phases.csv"));
        let _ = std::fs::remove_file(sidecar_path(path, ".meta.json"));
        let _ = std::fs::remove_file(sidecar_path(path, ".work.csv"));
    }

    #[test]
    fn retained_sample_memory_constants_match_layout() {
        assert_eq!(RAW_SAMPLE_BYTES, std::mem::size_of::<RawSample>());
        assert_eq!(
            DEFAULT_SAMPLE_BUFFER_BYTES,
            DEFAULT_MAX_SAMPLES * RAW_SAMPLE_BYTES
        );
        assert_eq!(MAX_SAMPLE_BUFFER_BYTES, MAX_SAMPLES * RAW_SAMPLE_BYTES);
    }

    #[test]
    fn present_intervals_anchor_and_carry_acquire_skips() {
        let path = test_path("intervals");
        let mut telemetry = FrameTelemetry::new(&path, 8).expect("valid bound");
        telemetry.begin_frame();
        telemetry
            .record_phase_ms(FramePhase::Render, 2.5)
            .expect("finite phase");
        assert_eq!(
            telemetry.present_at(Duration::from_millis(100)).unwrap(),
            PresentResult::Anchored
        );
        telemetry.acquire_skipped();
        telemetry.begin_frame();
        telemetry
            .record_phase_ms(FramePhase::Render, 3.5)
            .expect("finite phase");
        assert_eq!(
            telemetry.present_at(Duration::from_millis(350)).unwrap(),
            PresentResult::Recorded { frame_index: 0 }
        );
        let report = telemetry.finish().expect("write trace");
        assert!(report.completed);
        assert_eq!(report.count, 1);
        assert_eq!(report.acquire_skips, 1);
        let frames = std::fs::read_to_string(&path).expect("frame csv");
        assert!(frames.contains("0,250.000000000,shell,acquire_skips=1"));
        let phases = std::fs::read_to_string(&report.phases_path).expect("phase csv");
        assert!(phases
            .contains("0,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,3.500000000"));
        remove_outputs(&path);
    }

    #[test]
    fn overflow_is_explicit_and_invalidates_capture() {
        let path = test_path("overflow");
        let mut telemetry = FrameTelemetry::new(&path, 1).expect("valid bound");
        telemetry.present_at(Duration::from_millis(1)).unwrap();
        telemetry.present_at(Duration::from_millis(2)).unwrap();
        telemetry.acquire_skipped();
        telemetry.present_at(Duration::from_millis(3)).unwrap();
        let report = telemetry.finish().expect("write trace");
        assert!(!report.completed);
        assert!(report.overflowed);
        assert_eq!(report.count, 1);
        assert_eq!(report.dropped, 1);
        assert_eq!(report.acquire_skips, 1);
        let metadata = std::fs::read_to_string(&report.metadata_path).expect("metadata");
        assert!(metadata.contains("\"completed\": false"));
        assert!(metadata.contains("\"dropped\": 1"));
        let work_csv = std::fs::read_to_string(&report.work_path).expect("work csv");
        assert_eq!(work_csv.lines().count(), 2);
        remove_outputs(&path);
    }

    #[test]
    fn work_rows_align_and_preserve_unavailable_counters() {
        let path = test_path("work");
        let work = FrameWorkStats {
            terrain_loaded_sections: Some(7),
            terrain_draw_calls: Some(9),
            entity_upload_bytes: Some(2048),
            ui_build_ns: Some(1000),
            ui_draw_calls: Some(0),
            mesh_enqueued: Some(5),
            mesh_oldest_active_age_frames: Some(0),
            mesh_input_revision_rejected: Some(u64::MAX),
            mesh_upload_deadline_blocked: Some(1),
            ..FrameWorkStats::default()
        };
        let mut telemetry = FrameTelemetry::new(&path, 4).expect("valid bound");
        telemetry
            .record_present_at_with_work(
                Duration::from_millis(10),
                "gameplay",
                PhaseTimings::default(),
                FrameWorkStats::default(),
            )
            .expect("anchor");
        telemetry
            .record_present_at_with_work(
                Duration::from_millis(20),
                "gameplay",
                PhaseTimings::default(),
                work,
            )
            .expect("sample");
        let report = telemetry.finish().expect("write trace");
        let work_csv = std::fs::read_to_string(&report.work_path).expect("work csv");
        let columns: Vec<_> = work_csv
            .lines()
            .nth(1)
            .expect("work row")
            .split(',')
            .collect();
        assert_eq!(columns.len(), 39);
        assert_eq!(columns[0], "0");
        assert_eq!(columns[1], "7");
        assert_eq!(columns[2], "");
        assert_eq!(columns[3], "9");
        assert_eq!(columns[12], "");
        assert_eq!(columns[13], "2048");
        assert_eq!(columns[14], "1000");
        assert_eq!(columns[15], "");
        assert_eq!(columns[22], "0");
        assert_eq!(columns[23], "5");
        assert_eq!(columns[32], "0");
        assert_eq!(columns[33], "");
        assert_eq!(columns[36], u64::MAX.to_string());
        assert_eq!(columns[38], "1");
        let metadata = std::fs::read_to_string(&report.metadata_path).expect("metadata");
        assert!(metadata.contains("\"work_schema\""));
        assert!(metadata.contains("\"work_timing_definition\""));
        remove_outputs(&path);
    }

    #[test]
    fn direct_phase_values_and_invalid_present_times_are_rejected() {
        let path = test_path("validation");
        let mut telemetry = FrameTelemetry::new(&path, 4).expect("valid bound");
        let invalid = PhaseTimings {
            render_ms: f64::NAN,
            ..PhaseTimings::default()
        };
        assert!(matches!(
            telemetry.record_present_at(Duration::from_millis(1), "test", invalid),
            Err(TelemetryError::InvalidPhaseDuration("render"))
        ));
        assert_eq!(
            telemetry
                .record_present_at(Duration::from_millis(10), "test", PhaseTimings::default())
                .unwrap(),
            PresentResult::Anchored
        );
        assert!(matches!(
            telemetry.record_present_at(Duration::from_millis(10), "test", PhaseTimings::default()),
            Err(TelemetryError::NonMonotonicPresent)
        ));
        assert!(matches!(
            telemetry.record_present_at(Duration::from_millis(9), "test", PhaseTimings::default()),
            Err(TelemetryError::NonMonotonicPresent)
        ));
        remove_outputs(&path);
    }

    #[test]
    fn empty_and_unsuccessful_finalizations_are_not_complete() {
        let empty_path = test_path("empty");
        let mut empty = FrameTelemetry::new(&empty_path, 4).expect("valid bound");
        empty.present_at(Duration::from_millis(1)).expect("anchor");
        let empty_report = empty.finish().expect("write empty trace");
        assert!(!empty_report.completed);
        remove_outputs(&empty_path);

        let failed_path = test_path("failed");
        let mut failed = FrameTelemetry::new(&failed_path, 4).expect("valid bound");
        failed
            .record_present_at(Duration::from_millis(1), "test", PhaseTimings::default())
            .expect("anchor");
        failed
            .record_present_at(Duration::from_millis(2), "test", PhaseTimings::default())
            .expect("sample");
        let failed_report = failed.finalize(false).expect("write failed trace");
        assert!(!failed_report.completed);
        remove_outputs(&failed_path);
    }

    #[test]
    fn labels_are_csv_escaped_and_skips_are_retained_as_metadata_facts() {
        let path = test_path("labels");
        let mut telemetry = FrameTelemetry::new(&path, 4).expect("valid bound");
        telemetry.acquire_skipped();
        telemetry
            .record_present_at(
                Duration::from_millis(1),
                "menu,loading",
                PhaseTimings::default(),
            )
            .expect("anchor");
        telemetry.acquire_skipped();
        telemetry
            .record_present_at(
                Duration::from_millis(2),
                "menu,loading",
                PhaseTimings::default(),
            )
            .expect("sample");
        let report = telemetry.finish().expect("write trace");
        assert!(report.completed);
        let frames = std::fs::read_to_string(&path).expect("frame csv");
        assert!(frames.contains("0,1.000000000,\"menu,loading\",acquire_skips=1"));
        let metadata = std::fs::read_to_string(&report.metadata_path).expect("metadata");
        assert!(metadata.contains("\"acquire_skips\": 2"));
        assert!(metadata.contains("\"invalidated_by_acquire_skips\": true"));
        assert!(metadata.contains("\"scoring_eligible\": false"));
        remove_outputs(&path);
    }

    #[test]
    fn exclusive_outputs_are_never_overwritten() {
        let path = test_path("exclusive");
        let mut telemetry = FrameTelemetry::new(&path, 1).expect("valid bound");
        telemetry.present_at(Duration::from_millis(1)).unwrap();
        telemetry.present_at(Duration::from_millis(2)).unwrap();
        telemetry.finish().expect("first write");
        let second = FrameTelemetry::new(&path, 1).expect("construct does not write");
        assert!(matches!(second.finish(), Err(TelemetryError::Io(_))));
        remove_outputs(&path);
    }
}
