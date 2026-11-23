//! The window shell and the game loop.
//!
//! java: `client/Minecraft.java` — `run`, `runGameLoop`, `runTick`, and the display plumbing.
//! Rewritten rather than transliterated: winit 0.30 drives the loop through
//! [`winit::application::ApplicationHandler`], so there is no `while (running)` of our own.
//!
//! What is preserved exactly is the **shape** of vanilla's loop, because everything downstream
//! depends on it:
//!
//! ```text
//! per frame:  timer.update()
//!             app.before_ticks()                                  // drain scheduled work
//!             for _ in 0..timer.elapsed_ticks() { app.tick() }     // fixed 20 TPS
//!             app.frame(partial_ticks = timer.render_partial_ticks())
//!             phases.render()                                       // render rate, uncoupled
//!             present, then optionally sleep to the frame cap
//!             (a window in the background waits in the event loop instead; see
//!             `ShellApp::background_frame_cap`)
//! ```
//!
//! Ticks and frames are independent: at 200 FPS most frames run zero ticks, and at 5 FPS a frame
//! runs several. Interpolating with `partial_ticks` is what keeps motion smooth across that.
//!
//! # Not hanging
//!
//! [`ShellConfig::max_frames`] and [`ShellConfig::max_duration`] bound the loop. Automation must
//! set at least one of them — a windowed event loop with neither will run until the user closes
//! the window, which in a non-interactive session means forever.

// Shell startup and the frame loop expose the same rich public error used by GPU setup. Keeping
// that Result shape avoids an API-breaking box solely for clippy's size heuristic.
#![allow(clippy::result_large_err)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::background_wait::{background_cap, BackgroundWait, LoopStep};
use crate::context::PresentPreference;
use crate::frame::telemetry::{FramePhase, FrameTelemetry, FrameWorkStats, PhaseTimings};
use crate::frame::{
    Camera, DiagnosticTimerError, FrameLimiter, FrameStats, FrameTimeSummary, Timer,
};
use crate::input::{CursorMode, InputState};
use crate::pipeline::PhaseList;
use crate::settings::GameSettings;
use crate::{AcquiredFrame, Gpu, RenderError};

/// How the app wants the swap chain to present, polled once per frame through
/// [`ShellApp::wants_present`] (C1-10 / KNOBS-2).
///
/// Before this existed the shell built the GPU from its startup settings clone and never looked
/// again, so the in-game VSync button changed F3 and `options.txt` but not the swap chain until a
/// restart. MSAA is deliberately absent: a sample-count change invalidates every pipeline and
/// needs a phase re-install, so `msaaSamples` stays a startup-only, `options.txt`-only key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentRequest {
    /// `enableVsync`. Decides the mode while `preference` is [`PresentPreference::Auto`].
    pub vsync: bool,
    /// `presentMode`.
    pub preference: PresentPreference,
    /// `frameLatency`; see [`crate::context::DEFAULT_FRAME_LATENCY`].
    pub frame_latency: u32,
}

impl PresentRequest {
    /// The request `settings` describe.
    #[must_use]
    pub fn from_settings(settings: &GameSettings) -> Self {
        Self {
            vsync: settings.enable_vsync,
            preference: settings.present_mode,
            frame_latency: settings.frame_latency,
        }
    }

    /// Reconfigure `gpu` for this request. Only touches the surface if the resolved mode or the
    /// latency differs from what is configured; returns the mode now configured.
    pub fn apply(self, gpu: &mut Gpu) -> wgpu::PresentMode {
        gpu.set_present_options(self.preference, self.vsync, self.frame_latency)
    }
}

/// How the shell should open its window and when it should stop.
#[derive(Debug, Clone)]
pub struct ShellConfig {
    /// Window title.
    pub title: String,
    /// Initial inner size in logical pixels.
    pub width: u32,
    /// Initial inner height in logical pixels.
    pub height: u32,
    /// Stop after this many presented frames. `None` runs until the window closes.
    pub max_frames: Option<u64>,
    /// Stop after this much wall time. `None` runs until the window closes.
    pub max_duration: Option<Duration>,
    /// Grab the cursor as soon as the window opens.
    pub grab_cursor_on_start: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            title: "Minecraft 1.8.9 (mc-rs)".into(),
            width: 854,
            height: 480,
            max_frames: None,
            max_duration: None,
            grab_cursor_on_start: false,
        }
    }
}

/// State handed to [`ShellApp::tick`], once per fixed 20 TPS tick.
pub struct TickCtx<'a> {
    /// Raw device state. Reading held keys here is how movement input is sampled.
    pub input: &'a mut InputState,
    /// Settings, including [`GameSettings::keys`] for the rebindable actions.
    pub settings: &'a mut GameSettings,
    /// The camera. Set position and orientation from tick-rate state; the shell interpolates
    /// nothing on your behalf, so store previous/current yourself if you need smoothing.
    pub camera: &'a mut Camera,
    /// Ticks elapsed since the shell started.
    pub tick: u64,
}

/// State handed to [`ShellApp::frame`], once per rendered frame before phases record.
pub struct FrameCtx<'a> {
    /// The GPU context.
    pub gpu: &'a Gpu,
    /// The phase list, **mutable**, so phases can be registered, removed or rebuilt after
    /// [`ShellApp::init`] has run.
    ///
    /// This is what makes two things possible that `init`-only registration cannot express:
    ///
    /// * **Loading with the window open.** An app can register just a UI phase in `init`, draw a
    ///   progress screen, and do the expensive asset load in slices across `frame` calls — the
    ///   shell presents between every one of them — then register the rest of the phases here
    ///   once the assets exist. Under the old contract every asset had to be loaded *before*
    ///   `Shell::run` was even called, so there was no window to show progress on.
    /// * **Resource-pack reload.** A re-stitched block atlas cannot be pushed into a phase that
    ///   captured the old one in a bind group; [`PhaseList::clear`] plus re-registration is the
    ///   only honest answer, and it needs `&mut PhaseList` outside `init`.
    ///
    /// Changes take effect for **this** frame: the shell records the list immediately after
    /// `frame` returns. A phase registered here has not had
    /// [`RenderPhase::resize`](crate::RenderPhase::resize) called, so build it at
    /// `gpu.config.width`/`height`.
    pub phases: &'a mut PhaseList,
    /// The camera to finish positioning for this frame. The shell calls
    /// [`Camera::update`](crate::Camera::update) after this returns.
    pub camera: &'a mut Camera,
    /// Raw device state, mutable so a controller can drain
    /// [`InputState::look_delta`](crate::InputState::look_delta) here.
    pub input: &'a mut InputState,
    /// Settings.
    pub settings: &'a GameSettings,
    /// Fraction of the way to the next tick, `0.0..1.0`. Interpolate everything with this.
    pub partial_ticks: f32,
    /// Frame-rate counters.
    pub stats: &'a FrameStats,
}

/// Explicit simulation-clock policy for a deterministic diagnostic run.
///
/// This controls only the game timer that produces fixed ticks and render interpolation. It does
/// not replace the wall clocks used by presentation, frame limiting, frame-trace measurement,
/// shutdown bounds, or event-loop watchdogs. Ordinary replay and live runs return `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticClock {
    /// Advance the simulation by this many exact microseconds, retaining a sub-tick remainder.
    Step {
        /// Exact simulation microseconds to advance for this frame.
        micros: u64,
    },
    /// Run no simulation ticks and render with zero interpolation for this frame.
    Frozen,
    /// Abort the diagnostic run because a declared deterministic barrier is not satisfied.
    ///
    /// The client supplies a stable reason from the state observed by the preceding completed
    /// frame; the shell records it as a frame-trace failure instead of extending warmup based on
    /// variable completion time.
    Invalid(&'static str),
}

/// What the shell drives. Every method has a default, so a smoke test can implement none of them.
pub trait ShellApp {
    /// Called once, after the GPU context exists and before the first frame. Register phases here.
    fn init(&mut self, _gpu: &Gpu, _phases: &mut PhaseList) {}
    /// Apply main-thread work that became ready since the previous frame, before any game tick.
    ///
    /// java: `Minecraft.runGameLoop` drains `scheduledTasks` after updating the timer and before
    /// its `for (elapsedTicks) runTick()` loop. Network packet handlers are scheduled through
    /// that queue, so moving this hook after [`Self::tick`] changes which world state the tick
    /// observes. It is called once per rendered loop even when `elapsed_ticks == 0`.
    fn before_ticks(&mut self, _camera: &mut Camera) {}
    /// Called at a fixed 20 TPS, zero or more times per frame.
    fn tick(&mut self, _ctx: &mut TickCtx<'_>) {}
    /// Called once per raw keyboard event, in the order winit delivered it, before the frame the
    /// event landed in is processed.
    ///
    /// [`InputState`] only remembers whether a key is held **now**, which is enough for the
    /// rebindable actions but not for two things an app may still need: a key pressed *and*
    /// released inside a single frame is invisible to polling no matter how often it checks, and
    /// there is no way to recover the OS's own typed character —
    /// [`KeyEvent::text`](winit::event::KeyEvent::text), which honours the user's real keyboard
    /// layout — from held state at all, because held state was never text to begin with. This
    /// hands the event straight through instead, so neither is lost.
    ///
    /// java: has no analogue of its own — LWJGL 2's buffered `Keyboard.next()` queue is what
    /// this exists to give the app equivalent access to, now that winit's polled `InputState`
    /// has replaced it.
    fn on_key_event(&mut self, _event: &winit::event::KeyEvent, _sequence: u64) {}
    /// Called whenever the OS window gains or loses focus, after the shell has reconciled its
    /// own raw input state and released every game key binding on loss.
    ///
    /// java: `Minecraft.setIngameNotInFocus` releases input and cancels an active block dig;
    /// `setIngameFocus` reinstalls the long left-click lockout. Apps that do not own a player
    /// controller can keep the default no-op.
    fn on_focus_changed(&mut self, _focused: bool) {}
    /// Whether device edges may reach the game key bindings.
    ///
    /// java: `KeyBinding.setKeyBindState`/`onTick` in `Minecraft.runTick`. They sit inside the
    /// `currentScreen == null || currentScreen.allowUserInput` block, but an open screen's
    /// `handleInput()` has already drained every LWJGL event by then, so in effect this is
    /// `currentScreen == null`, `GuiInventory` included. Implementors must call
    /// [`KeyBindings::unpress_all`] when changing this from `true` to `false`; the shell then
    /// discards both press and release edges until game input is allowed again.
    ///
    /// The default keeps non-game shell clients backwards-compatible.
    fn game_input_allowed(&self) -> bool {
        true
    }
    /// The effective frame cap for this frame.
    ///
    /// The app may combine live settings with menu/focus policy. The default preserves the
    /// configured vanilla max-FPS value for apps that own no dynamic UI state.
    fn effective_frame_cap(&self, settings: &GameSettings, _focused: bool) -> Option<u32> {
        settings.framerate_limit()
    }
    /// The frame cap while the window is in the background: another window
    /// has the focus, or (`minimized`) the window is minimized. `None`, the default: no cap of its
    /// own, and the frame is paced exactly as a focused one.
    ///
    /// With a cap, the shell paces the frame at the lower of this and
    /// [`Self::effective_frame_cap`], and waits for the next one in the event loop instead of the
    /// limiter's sleep, so the focus coming back, or a restore, starts it at once. Asked only
    /// while the window is in the background. A window opened with [`BACKGROUND_WINDOW_ENV`]
    /// never hears of its focus, so a benchmark in a background window keeps its frame rate
    /// (minimized, it draws nothing anyway, and is capped like any other).
    fn background_frame_cap(&self, _minimized: bool) -> Option<u32> {
        None
    }
    /// Called once per rendered frame, before phases record.
    fn frame(&mut self, _ctx: &mut FrameCtx<'_>) {}
    /// Label the current application state for an explicitly enabled frame trace.
    ///
    /// This hook is called only when `MC_FRAME_TRACE` is enabled. Labels describe
    /// application state and do not assert benchmark readiness or determinism.
    fn diagnostic_phase(&self) -> &'static str {
        "shell"
    }
    /// Select an explicit simulation clock for this frame.
    ///
    /// The hook is read immediately before the shell updates its timer, so an app can choose
    /// `Frozen` from the state reached by the preceding frame. `None` preserves the ordinary
    /// host-monotonic `Timer::update` path exactly. This hook controls simulation cadence only;
    /// present, limiter, measurement, and run-duration wall clocks remain live.
    fn diagnostic_clock(&self) -> Option<DiagnosticClock> {
        None
    }
    /// Enable application work counters for an explicitly requested diagnostic draw.
    ///
    /// This hook is skipped when tracing is disabled. It must preserve simulation
    /// and rendering behavior; only diagnostic counter collection may change.
    fn begin_diagnostic_frame(&mut self) {}
    /// Snapshot available work counters after rendering an explicitly traced draw.
    /// Missing counters stay absent instead of being represented as zero work.
    fn diagnostic_work(&self) -> FrameWorkStats {
        FrameWorkStats::default()
    }
    /// Finalize the camera before recording phases. Scalar-only apps get the usual matrix
    /// rebuild. Apps that prepare their matrices in `frame` must preserve those matrices here;
    /// rebuilding them would discard view-space effects such as Java's first-person offset.
    fn finalize_camera(&mut self, camera: &mut Camera) {
        camera.update();
    }
    /// Called after every phase has recorded and submitted the frame, but before the acquired
    /// surface texture is presented.
    ///
    /// This narrow seam is intentionally separate from [`Self::frame`]: screenshot/readback
    /// consumers need the composited, post/UI image and must copy the still-acquired surface
    /// texture before [`AcquiredFrame::present`] consumes it. The default is a no-op for apps
    /// that do not need frame readback.
    fn after_render(&mut self, _gpu: &Gpu, _frame: &AcquiredFrame) {}
    /// Video export: render frames offscreen at
    /// this size instead of to the window ([`Gpu::set_render_override`]). Checked once per frame
    /// before anything else; the shell resizes the phases, the camera and the app
    /// ([`Self::resized`]) when it changes, exactly as for a window resize. While it is `Some`,
    /// nothing is presented: [`Self::after_render`] sees the offscreen frame, and the window
    /// shows what [`Self::draw_window_preview`] draws.
    ///
    /// Defaults to `None`: the normal path, unchanged.
    fn offscreen_render_size(&self) -> Option<(u32, u32)> {
        None
    }
    /// Switching to [`Self::offscreen_render_size`] failed: the GPU had no
    /// memory for the targets at that size (or refused them). The shell is back at the window's
    /// size; the app must stop asking for that size (an export fails with `error`), or the next
    /// frame tries again.
    fn offscreen_failed(&mut self, _error: &str) {}
    /// The GPU device was lost and the shell stops this frame with
    /// [`RenderError::DeviceLost`], before any other hook runs: the app ends what it was doing on
    /// the GPU (a video export deletes its unfinished file) and may give the text for the crash
    /// report instead of the driver's bare `reason` (an export names its fix). Default: the
    /// reason alone.
    fn device_lost(&mut self, _reason: &str) -> Option<String> {
        None
    }
    /// While rendering offscreen: whether to refresh the window this iteration.
    fn wants_window_preview(&self) -> bool {
        false
    }
    /// While rendering offscreen: draw the window's picture into `window` (the
    /// window's own image, [`Gpu::window_size`]) after the offscreen `frame` was rendered and
    /// [`Self::after_render`] ran. The shell presents it.
    fn draw_window_preview(&mut self, _gpu: &Gpu, _frame: &AcquiredFrame, _window: &AcquiredFrame) {
    }
    /// Whether the loop should keep going. Checked after every frame.
    ///
    /// java: `Minecraft.running` — the flag `Minecraft.shutdown()` clears, which is how the main
    /// menu's Quit button and `GuiIngameMenu`'s "Save and Quit" actually end the process. Without
    /// this the shell only stops when the *window* closes or a [`ShellConfig`] bound is hit, so an
    /// in-game Quit had nowhere to land.
    ///
    /// Defaults to `true`, so an app that never asks to stop behaves exactly as before.
    fn keep_running(&self) -> bool {
        true
    }
    /// Whether the OS window's fullscreen state should change, checked once per frame.
    ///
    /// `Some(true)`/`Some(false)` requests fullscreen/windowed; `None` leaves the window alone.
    /// It is safe, and expected, to return `Some` unconditionally with whatever the app's own
    /// setting currently is — the shell only calls [`Window::set_fullscreen`] when the request
    /// actually differs from what it last applied, so this does not fight the OS every frame.
    ///
    /// Before this existed there was no way for an app to reach the `winit::window::Window` at
    /// all, so a setting like a fullscreen flag could be flipped and saved but never acted on.
    /// java: this is what lets `Minecraft.toggleFullscreen` (`:1643`) — which reaches AWT's
    /// `Display.setFullscreen` directly — have a real effect here too.
    ///
    /// Defaults to `None`, so an app that never asks for fullscreen behaves exactly as before.
    fn wants_fullscreen(&self) -> Option<bool> {
        None
    }
    /// Whether fullscreen should be an exclusive video-mode switch (LWJGL 2's) rather than the
    /// borderless window, checked with [`Self::wants_fullscreen`]. A change while fullscreen
    /// re-applies it.
    ///
    /// Defaults to `false`, the borderless window every app had before this existed (the client's Borderless Fullscreen mod is ON by default, and only turning it off
    /// asks for exclusive).
    fn wants_exclusive_fullscreen(&self) -> bool {
        false
    }
    /// How the swap chain should present, checked once per frame before acquisition.
    ///
    /// Same contract as [`Self::wants_fullscreen`]: return `Some` unconditionally with the live
    /// settings; the shell reconfigures only when the request differs from what it last applied,
    /// so an unchanged request costs one comparison. `None` (the default) keeps whatever the
    /// shell built the context with, which is the behaviour from before this hook existed.
    fn wants_present(&self) -> Option<PresentRequest> {
        None
    }
    /// Called after the window resizes and the GPU context has been reconfigured.
    fn resized(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {}
    /// The refresh rate of the monitor the window is on, in millihertz (`None` when the platform
    /// does not report one). Called before the first frame and again whenever it changes: the
    /// window moved to another monitor, or a fullscreen switch changed the display mode. The
    /// client's Aero title screen caps its frame rate to it.
    ///
    /// Defaults to doing nothing, so an app that never asks behaves exactly as before.
    fn monitor_refresh_changed(&mut self, _refresh_millihertz: Option<u32>) {}
    /// The window's scale factor (the desktop's display scaling, 1.0 at 100%). Called before the
    /// first frame and again whenever it changes (a move to a monitor with other scaling, or a
    /// change of the setting). The replay viewer draws the recorded mouse pointer at the size
    /// the desktop's own pointer has at it.
    ///
    /// Defaults to doing nothing, so an app that never asks behaves exactly as before.
    fn window_scale_changed(&mut self, _scale_factor: f64) {}
    /// Called once as the loop exits — the place to save settings.
    fn shutdown(&mut self, _settings: &GameSettings) {}
}

/// What a completed [`Shell::run`] reports back.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Frames presented.
    pub frames: u64,
    /// Game ticks run.
    pub ticks: u64,
    /// Wall time the loop ran for.
    pub elapsed: Duration,
    /// Mean frames per second across the run.
    ///
    /// A reciprocal, and therefore a bad unit for reasoning about a frame budget: 1000 → 2000 fps
    /// and 2000 → 4000 fps read as the same doubling but are 0.5 ms and 0.25 ms of reclaimed
    /// budget. Quote [`RunSummary::frame_time`]'s milliseconds for anything but a headline.
    pub average_fps: f32,
    /// Most recent one-second FPS sample.
    pub last_fps: u32,
    /// The whole run's frame-time distribution in milliseconds — mean, sd, min, p50, p95, p99,
    /// p99.9, max, and the `n` they cover.
    ///
    /// `average_fps` is a mean and a mean cannot see a hitch; §1.5 measured a 0.5 ms mean against
    /// a 3.16 ms worst frame. This is the field that can tell those apart.
    pub frame_time: FrameTimeSummary,
    /// Adapter name and backend, for the report.
    pub adapter: String,
    /// Structured adapter facts retained for crash-report enrichment.
    pub adapter_facts: Option<crate::AdapterFacts>,
    /// The swap-chain present mode the run actually used.
    ///
    /// The *resolved* mode, not the requested preference: an unsupported request falls back, and
    /// a measurement that reports what it asked for rather than what it got is not a measurement.
    /// `None` for a run that never got as far as creating a context.
    pub present_mode: Option<wgpu::PresentMode>,
    /// Per-phase GPU timestamps. Empty unless `MC_GPU_TIMING` was set.
    pub gpu: crate::pipeline::GpuTimings,
    /// Per presented frame: from the start of its loop iteration (where ticks and the frame build
    /// read input) until its `present` call returned, in milliseconds. The CPU side of input
    /// latency; GPU execution and scan-out come after it and are not included. Comparable between
    /// the serial loop and the pipelined paths, which is what it is for.
    pub input_to_present: FrameTimeSummary,
}

/// Opens the window, owns the GPU context, and runs the loop.
pub struct Shell;

impl Shell {
    /// Run until the window closes or a [`ShellConfig`] bound is hit.
    ///
    /// Must be called from the main thread — winit requires it on macOS and Windows.
    ///
    /// # Errors
    /// [`RenderError::EventLoop`] if the loop cannot start, [`RenderError::Window`] if the window
    /// cannot be created, or whatever [`Gpu::for_window`] failed with.
    pub fn run<A: ShellApp>(
        config: ShellConfig,
        settings: GameSettings,
        app: &mut A,
    ) -> Result<RunSummary, RenderError> {
        let event_loop = EventLoop::new().map_err(|e| RenderError::EventLoop(e.to_string()))?;
        // Poll, not Wait: the game redraws continuously rather than only on OS-requested repaints.
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut runner = Runner {
            config,
            app,
            settings,
            input: InputState::new(),
            camera: Camera::default(),
            timer: Timer::default(),
            stats: FrameStats::new(),
            limiter: FrameLimiter::new(),
            focused: true,
            background_window: background_window_requested(),
            phases: PhaseList::new(),
            state: None,
            next_input_sequence: 0,
            ticks: 0,
            started: Instant::now(),
            failure: None,
            adapter: String::new(),
            adapter_facts: None,
            present_mode: None,
            trace: FrameTelemetry::from_env()
                .map_err(|error| RenderError::FrameTrace(error.to_string()))?,
            pending_resize: None,
            present_thread: None,
            pipelined_submit: false,
            decouple_target: None,
            drew_frame: false,
            input_to_present: InputToPresent::new(),
            // The first frame reports the monitor the window opened on.
            monitor_dirty: true,
            reported_refresh: None,
            reported_scale: None,
            background: BackgroundWait::default(),
        };

        let loop_result = event_loop.run_app(&mut runner);
        // Archiving an opt-in trace occurs after the run and is not part of its elapsed time.
        let elapsed = runner.started.elapsed();
        // The measurement aid's summaries, in the bench's line format.
        for (label, phase) in &runner.input_to_present.phases {
            eprintln!("mc-rs input-to-present [{label}]: {}", phase.summary());
        }
        if let Some(mut trace) = runner.trace.take() {
            if loop_result.is_err() {
                trace.record_failure("event loop failed");
            } else if runner.failure.is_some() {
                trace.record_failure("shell failed");
            }
            match trace.finalize(loop_result.is_ok() && runner.failure.is_none()) {
                Ok(report) if report.completed => {}
                Ok(_) => {
                    runner.failure.get_or_insert_with(|| {
                        RenderError::FrameTrace(
                            "capture is incomplete; inspect its metadata".into(),
                        )
                    });
                }
                Err(error) => {
                    runner
                        .failure
                        .get_or_insert_with(|| RenderError::FrameTrace(error.to_string()));
                }
            }
        }
        if let Err(error) = loop_result {
            let error = RenderError::EventLoop(error.to_string());
            return Err(match runner.adapter_facts {
                Some(facts) => error.with_adapter(facts),
                None => error,
            });
        }

        if let Some(err) = runner.failure {
            return Err(match runner.adapter_facts {
                Some(facts) => err.with_adapter(facts),
                None => err,
            });
        }
        Ok(RunSummary {
            frames: runner.stats.total_frames(),
            ticks: runner.ticks,
            elapsed,
            average_fps: runner.stats.average_fps(),
            last_fps: runner.stats.fps(),
            frame_time: runner.stats.frame_time(),
            adapter: runner.adapter,
            adapter_facts: runner.adapter_facts,
            present_mode: runner.present_mode,
            gpu: runner.phases.gpu_timings(),
            input_to_present: runner.input_to_present.summary(),
        })
    }
}

struct Live {
    window: Arc<Window>,
    gpu: Gpu,
    applied_cursor: Option<CursorMode>,
    /// The fullscreen state the window was last set to, so `apply_fullscreen` only calls
    /// `Window::set_fullscreen` when [`ShellApp::wants_fullscreen`] actually changes.
    applied_fullscreen: Option<bool>,
    /// The fullscreen kind last *requested* ([`ShellApp::wants_exclusive_fullscreen`]), kept
    /// even when exclusive fell back to borderless, so a missing video mode is not retried
    /// every frame.
    applied_exclusive: bool,
    /// The present request the swap chain was last configured for, so `apply_present` only
    /// reconfigures when [`ShellApp::wants_present`] actually changes.
    applied_present: Option<PresentRequest>,
    /// Whether the OS-pointer look grab (Raw Mouse Input off) is applied; see
    /// `apply_pointer_look`.
    applied_os_pointer: bool,
    /// A fully transparent cursor: the OS-pointer look hides the pointer with it, because winit
    /// clips a *hidden* grabbed pointer to one pixel and it could then not move at all.
    blank_cursor: Option<winit::window::CustomCursor>,
    /// A focus loss released the game's grab ([`release_grab_on_focus_loss`]):
    /// the next grab centres the pointer first, as LWJGL 2's `updateCursor` does on reactivation.
    regrab_centre: bool,
}

struct Runner<'a, A: ShellApp> {
    config: ShellConfig,
    app: &'a mut A,
    settings: GameSettings,
    input: InputState,
    camera: Camera,
    timer: Timer,
    stats: FrameStats,
    limiter: FrameLimiter,
    /// See [`BACKGROUND_WINDOW_ENV`].
    background_window: bool,
    focused: bool,
    phases: PhaseList,
    state: Option<Live>,
    /// One receipt-order counter shared by keyboard, mouse-button, and wheel ingress.
    next_input_sequence: u64,
    ticks: u64,
    started: Instant,
    failure: Option<RenderError>,
    adapter: String,
    adapter_facts: Option<crate::AdapterFacts>,
    present_mode: Option<wgpu::PresentMode>,
    trace: Option<FrameTelemetry>,
    /// The newest `Resized` size not yet applied (PRESENT-9). A drag or a fullscreen switch
    /// delivers a burst of these; only the last one before a frame is applied.
    pending_resize: Option<PhysicalSize<u32>>,
    /// `Some` only when [`PIPELINED_PRESENT_ENV`] or [`PIPELINED_SUBMIT_ENV`] asked for it
    /// (#14 step 1a).
    present_thread: Option<PresentThread>,
    /// [`PIPELINED_SUBMIT_ENV`] (#14 step 1b): frames are finished and submitted on the present
    /// thread. Only acted on while `present_thread` exists.
    pipelined_submit: bool,
    /// Where pipelined frames are recorded before their image exists
    /// ([`W18_OB_DECOUPLE_ENV`]); `None` until the first such frame and whenever that path is off.
    decouple_target: Option<DecoupleTarget>,
    /// The last `draw` presented (or, offscreen, completed) a frame: the
    /// next one may be drawn from `about_to_wait` ([`W18_OB_DIRECT_DRAW_ENV`]).
    drew_frame: bool,
    /// See [`RunSummary::input_to_present`] (and).
    input_to_present: InputToPresent,
    /// The window's monitor may have changed (start, a move, a DPI change, a resize, a fullscreen
    /// switch): re-read its refresh rate before the next frame. See
    /// [`ShellApp::monitor_refresh_changed`].
    monitor_dirty: bool,
    /// The refresh rate last reported to the app (`None`: nothing reported yet).
    reported_refresh: Option<Option<u32>>,
    /// The scale factor last reported to the app (`None`: nothing yet).
    reported_scale: Option<f64>,
    /// Whether the window is minimized, and the next frame's wait while the
    /// window is in the background ([`ShellApp::background_frame_cap`]).
    background: BackgroundWait,
}

impl<A: ShellApp> Runner<'_, A> {
    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RenderError> {
        let attrs = Window::default_attributes()
            .with_active(!self.background_window)
            .with_title(self.config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.width,
                self.config.height,
            ));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .map_err(|e| RenderError::Window(e.to_string()))?,
        );
        // The size a screen-open re-centres the GUI cursor in; `Resized` keeps it current.
        let inner = window.inner_size();
        self.input.set_window_size(inner.width, inner.height);

        // Diagnostic for the frame-pacing question: `shell.acquire` was measured at
        // ~2.8-3 ms/frame under `PresentMode::AutoVsync`, far under a 60 Hz refresh's ~16.7 ms.
        // Logging the monitor winit actually sees turns "is this display just high-refresh" into
        // something a log line answers. `current_monitor` needs the window to exist, so this runs
        // right after `create_window` rather than before.
        if let Some(monitor) = window.current_monitor() {
            tracing::info!(
                monitor = ?monitor.name(),
                refresh_rate_millihertz = ?monitor.refresh_rate_millihertz(),
                size = ?monitor.size(),
                "current monitor",
            );
        } else {
            tracing::info!("window.current_monitor() returned None — no monitor info available");
        }

        // `presentMode` is carried separately from `gpu_options()` because `GpuOptions` is a
        // struct-literal type across five crates' tests; see `Gpu::for_window_with`.
        let mut gpu = Gpu::for_window_with(
            Arc::clone(&window),
            self.settings.gpu_options(),
            self.settings.present_mode,
        )?;
        // `frameLatency` is not a `for_window_with` argument (that signature is shared with
        // tests). With the default latency this is a comparison and no reconfigure, so the
        // default path is exactly the pre-existing one.
        let initial_present = PresentRequest::from_settings(&self.settings);
        initial_present.apply(&mut gpu);
        let info = gpu.adapter_info();
        self.adapter_facts = Some(crate::AdapterFacts::from_info(&info));
        self.adapter = format!("{} ({:?})", info.name, info.backend);
        self.present_mode = Some(gpu.present_mode());
        tracing::info!(
            adapter = %self.adapter,
            present_mode = ?gpu.present_mode(),
            "gpu context ready",
        );

        self.camera
            .set_viewport(gpu.config.width, gpu.config.height);
        // The slider's range, as every other reader of this value:
        // options.txt is read unclamped, as vanilla's `loadOptions` reads it.
        self.camera
            .set_view_distance_chunks(self.settings.render_distance_chunks.clamp(2, 32) as u32);
        self.camera.fov = self.settings.fov;
        self.camera.update();

        self.app.init(&gpu, &mut self.phases);
        self.timer.reset_clocks(0, 0);
        self.started = Instant::now();
        self.stats = FrameStats::new();
        self.limiter.reset();

        if self.config.grab_cursor_on_start {
            self.input.request_cursor(CursorMode::Grabbed);
        }
        // #14 step 1a, opt-in. Not on GL, whose context is the one backend where acquiring and
        // presenting off the creating thread is not something wgpu promises. Step 1b (the frame's
        // finish and submit on the same thread) needs it, so asking for 1b starts it too.
        let pipelined_submit = pipelined_submit_requested();
        if (pipelined_present_requested() || pipelined_submit) && info.backend != wgpu::Backend::Gl
        {
            if let Some(surface) = gpu.shared_surface() {
                match PresentThread::spawn(
                    surface,
                    gpu.device.clone(),
                    gpu.queue.ungated().clone(),
                    gpu.frame_pipe(),
                    Arc::clone(&window),
                ) {
                    Ok(thread) => {
                        tracing::info!(pipelined_submit, "pipelined present thread started");
                        self.present_thread = Some(thread);
                        self.pipelined_submit = pipelined_submit;
                    }
                    Err(error) => {
                        tracing::warn!(%error, "pipelined present thread unavailable; presenting inline");
                    }
                }
            }
        }
        // A 1x1 transparent icon, for Raw Mouse Input off (`apply_pointer_look`). Creating it is
        // the only cost of that mode while it is unused.
        let blank_cursor = winit::window::CustomCursor::from_rgba(vec![0_u8; 4], 1, 1, 0, 0)
            .ok()
            .map(|source| event_loop.create_custom_cursor(source));
        self.state = Some(Live {
            window,
            gpu,
            applied_cursor: None,
            applied_fullscreen: None,
            applied_exclusive: false,
            applied_present: Some(initial_present),
            applied_os_pointer: false,
            blank_cursor,
            regrab_centre: false,
        });
        Ok(())
    }

    /// Apply the newest pending `Resized` size once (PRESENT-9). Phases and the app are resized
    /// only when the GPU actually rebuilt its attachments: a repeated size, or the 0×0 Windows
    /// reports while minimised, used to rebuild fog's bind group and the UI projection anyway.
    fn apply_pending_resize(&mut self) {
        let Some(size) = self.pending_resize.take() else {
            return;
        };
        let Some(state) = self.state.as_mut() else {
            return;
        };
        // `configure` panics while a surface texture is alive; take back any pre-acquired one
        // (only when this size configures the swap chain).
        // If the present thread has not answered yet, keep the size for the next frame.
        let Some(rebuilt) = resize_swap_chain(
            &mut state.gpu,
            self.present_thread.as_mut(),
            size.width,
            size.height,
        ) else {
            self.pending_resize = Some(size);
            return;
        };
        // From the configured size, which never goes to 0×0: a minimised window used to leave
        // the camera with a zero-area viewport until the restore.
        self.camera
            .set_viewport(state.gpu.config.width, state.gpu.config.height);
        // Refresh even while an app is loading and prepares no world camera.
        // Active clients compose their view effects again in the next frame.
        self.camera.update();
        if rebuilt {
            self.phases
                .resize(&state.gpu, state.gpu.config.width, state.gpu.config.height);
            self.app
                .resized(&state.gpu, state.gpu.config.width, state.gpu.config.height);
        }
    }

    /// Switch rendering offscreen, or back to the window, when the app asks
    /// ([`ShellApp::offscreen_render_size`]). One comparison when nothing changes. Returns
    /// `false` when the switch must wait a frame (the present thread still holds an image).
    fn apply_offscreen_request(&mut self) -> bool {
        let wanted = self.app.offscreen_render_size();
        let Some(state) = self.state.as_mut() else {
            return true;
        };
        if wanted == state.gpu.render_override() {
            return true;
        }
        // The window's image is acquired on this thread while offscreen (for the preview), and
        // by the present thread again afterwards: it must hold none across the switch.
        if !self
            .present_thread
            .as_mut()
            .is_none_or(PresentThread::quiesce)
        {
            return false;
        }
        // Going offscreen, the present thread has just dropped any
        // image it held without presenting it, and a dropped image is not always given back:
        // every acquire of the window's preview then waited out its whole timeout, a second per
        // exported frame from the second export of a session on. A fresh swap chain has all its
        // images. Once per export start.
        let rebuilt =
            wanted.is_some() && self.present_thread.is_some() && state.gpu.rebuild_swap_chain();
        // Back to the window, or minimised: an image still lost is given back
        // now, or at the next configure point.
        settle_lost_image(&mut state.gpu, self.present_thread.as_mut(), rebuilt);
        // Every target at the export's size is made here. A size the GPU has
        // no memory for (an 8K export with post effects on a 4 GB card) was a panic in wgpu's
        // default error handler; the scopes catch it, the window's size comes back and the app
        // hears why. Only on a switch, never per frame.
        let device = state.gpu.device.clone();
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
        if state.gpu.set_render_override(wanted) {
            let (width, height) = (state.gpu.config.width, state.gpu.config.height);
            tracing::info!(?wanted, width, height, "offscreen rendering switched");
            self.camera.set_viewport(width, height);
            self.camera.update();
            self.phases.resize(&state.gpu, width, height);
            self.app.resized(&state.gpu, width, height);
        }
        let invalid = pollster::block_on(invalid.pop());
        let failed = pollster::block_on(oom.pop()).or(invalid);
        if let (Some(error), Some(_)) = (failed, wanted) {
            tracing::warn!(%error, ?wanted, "offscreen rendering failed; back to the window");
            if state.gpu.set_render_override(None) {
                let (width, height) = (state.gpu.config.width, state.gpu.config.height);
                self.camera.set_viewport(width, height);
                self.camera.update();
                self.phases.resize(&state.gpu, width, height);
                self.app.resized(&state.gpu, width, height);
            }
            self.app.offscreen_failed(&error.to_string());
        }
        true
    }

    /// Tell the app the window's monitor refresh rate when it may have changed. One `bool` test
    /// on every other frame; the OS query runs only after a move, a DPI change, a resize or a
    /// fullscreen switch, and the app hears only real changes.
    fn report_monitor_refresh(&mut self) {
        if !std::mem::take(&mut self.monitor_dirty) {
            return;
        }
        let Some(state) = self.state.as_ref() else {
            return;
        };
        let refresh = state
            .window
            .current_monitor()
            .and_then(|monitor| monitor.refresh_rate_millihertz());
        // Read on the same occasions (a DPI change sets the flag too).
        let scale = state.window.scale_factor();
        if self.reported_refresh != Some(refresh) {
            self.reported_refresh = Some(refresh);
            tracing::info!(refresh_rate_millihertz = ?refresh, "window monitor refresh rate");
            self.app.monitor_refresh_changed(refresh);
        }
        if self.reported_scale != Some(scale) {
            self.reported_scale = Some(scale);
            self.app.window_scale_changed(scale);
        }
    }

    fn should_stop(&self) -> bool {
        !self.app.keep_running()
            || self
                .config
                .max_frames
                .is_some_and(|n| self.stats.total_frames() >= n)
            || self
                .config
                .max_duration
                .is_some_and(|d| self.started.elapsed() >= d)
    }

    /// One iteration of vanilla's `runGameLoop`.
    ///
    /// Every stage is inside its own [`profile_scope!`](mc_core::profile_scope), because "the
    /// frame takes 14 ms" is not a finding — the five stages below are the first partition that
    /// can *be* a finding, and four of them are outside the app entirely. `shell.acquire` and
    /// `shell.present` in particular are where a swap-chain that is pacing us shows up, and no
    /// amount of instrumenting the client would have found that.
    fn draw(&mut self) {
        mc_core::profile_scope!("shell.draw");
        // Set again once this call presents a frame.
        self.drew_frame = false;
        // The start of the input-to-present span (`RunSummary::input_to_present`): ticks and the
        // frame build below read input after this point.
        let iteration_started = Instant::now();
        // Before the timer and every app hook, exactly where the eager `Resized` handler used to
        // have run by this point (window events are delivered before `RedrawRequested`).
        self.apply_pending_resize();
        // Before this frame's cap is read (one `bool` test unless it moved).
        self.report_monitor_refresh();
        // One comparison unless a video export starts or ends.
        if !self.apply_offscreen_request() {
            return;
        }
        let Some(state) = self.state.as_mut() else {
            return;
        };
        // A lost device draws nothing again, and wgpu raises no error for what is
        // made on it: the next mapped buffer anywhere panicked (an AMD PC's export crash). Stop the loop with the driver's reason instead; `main` turns it into
        // a crash report. One atomic load per frame.
        if state.gpu.is_device_lost() {
            let reason = state
                .gpu
                .device_lost()
                .unwrap_or_else(|| "no reason given".to_owned());
            tracing::error!(%reason, "the GPU device was lost; stopping");
            // The app's words when it has any (a running export names its fix).
            let message = self.app.device_lost(&reason).unwrap_or(reason);
            self.failure = Some(RenderError::DeviceLost(message));
            return;
        }
        // Evaluate this before any timer update or app simulation hook. A diagnostic client can
        // therefore freeze from the state observed at the end of the prior frame.
        let diagnostic_clock = self.app.diagnostic_clock();
        if diagnostic_clock.is_some() {
            // Discard held/raw device state accumulated before entering diagnostic mode.
            // Synthetic diagnostic input, if any, is owned by the client and does not arrive via
            // this OS-facing shell state.
            self.input.clear_held();
            self.settings.keys.unpress_all();
        }
        if self.trace.is_some() {
            self.app.begin_diagnostic_frame();
        }
        // One emptiness test while the measurement aid is unset.
        if !self.input_to_present.phases.is_empty() {
            let current = self.trace.as_ref().map(|_| self.app.diagnostic_phase());
            self.input_to_present.current = current.and_then(|current| {
                self.input_to_present
                    .phases
                    .iter()
                    .position(|(wanted, _)| wanted == current)
            });
        }
        // Clock reads and retained samples exist only in explicitly requested diagnostic runs.
        // Phase durations describe this draw. Present-to-present intervals include the prior
        // draw's limiter and event-loop work, so the phase row is not an additive partition.
        let mut trace_clock = self.trace.as_ref().map(|_| Instant::now());
        let mut trace_phases = PhaseTimings::default();

        if let Some(DiagnosticClock::Invalid(reason)) = diagnostic_clock {
            if let Some(trace) = self.trace.as_mut() {
                trace.record_failure(reason);
            }
            self.failure = Some(RenderError::FrameTrace(format!(
                "diagnostic simulation barrier failed: {reason}"
            )));
            return;
        }

        // Fixed-rate ticks first, then exactly one render. java: Minecraft.runGameLoop. An
        // explicitly selected diagnostic clock never reads a host clock; `None` retains the
        // ordinary live/replay behavior.
        if let Err(error) = advance_simulation_clock(&mut self.timer, diagnostic_clock) {
            if let Some(trace) = self.trace.as_mut() {
                trace.record_failure("diagnostic simulation clock rejected");
            }
            self.failure = Some(RenderError::FrameTrace(format!(
                "diagnostic simulation clock failed: {error}"
            )));
            return;
        }
        {
            mc_core::profile_scope!("shell.before_ticks");
            self.app.before_ticks(&mut self.camera);
        }
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::BeforeTicks);
        {
            mc_core::profile_scope!("shell.ticks");
            for _ in 0..self.timer.elapsed_ticks() {
                self.ticks += 1;
                self.app.tick(&mut TickCtx {
                    input: &mut self.input,
                    settings: &mut self.settings,
                    camera: &mut self.camera,
                    tick: self.ticks,
                });
            }
        }
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Ticks);

        let partial_ticks = self.timer.render_partial_ticks();
        {
            mc_core::profile_scope!("shell.app_frame");
            self.app.frame(&mut FrameCtx {
                gpu: &state.gpu,
                // Disjoint fields of `self`, so the borrow checker splits this cleanly: `state`
                // is `self.state`, the phase list is `self.phases`.
                phases: &mut self.phases,
                camera: &mut self.camera,
                input: &mut self.input,
                settings: &self.settings,
                partial_ticks,
                stats: &self.stats,
            });
        }
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::AppFrame);
        self.app.finalize_camera(&mut self.camera);

        if !self.background_window {
            apply_cursor(state, &mut self.input, self.focused);
            apply_pointer_look(state, &mut self.input, self.focused);
        }
        // A fullscreen switch can change the display mode.
        let fullscreen_before = (state.applied_fullscreen, state.applied_exclusive);
        apply_fullscreen(
            state,
            self.app.wants_fullscreen(),
            self.app.wants_exclusive_fullscreen(),
        );
        if (state.applied_fullscreen, state.applied_exclusive) != fullscreen_before {
            self.monitor_dirty = true;
        }
        // C1-10: before `acquire`, because `Surface::configure` panics while a surface texture
        // is alive. An unchanged request is one comparison.
        if let Some(want) = present_reconciliation(state.applied_present, self.app.wants_present())
        {
            // With a present thread, configure only once it holds no image (else: next frame).
            // A request that configures nothing keeps the image.
            if let Some(mode) = apply_present(state, self.present_thread.as_mut(), want) {
                self.present_mode = Some(mode);
            } else {
                // Retried next frame even if the request goes back to the
                // applied one meanwhile, so an image the late answer dropped is given back.
                state.applied_present = None;
            }
        }
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Camera);

        // While a video export renders offscreen, frames come from (and stay
        // in) the offscreen target; the present thread is left idle.
        let offscreen = state.gpu.render_override().is_some();
        // Record this frame now, into the decoupled target, and wait for its
        // image afterwards (see `W18_OB_DECOUPLE_ENV`). Not with GPU timestamps, whose frames are
        // submitted on this thread as soon as they are recorded.
        let mut early: Option<wgpu::CommandEncoder> = None;
        if !offscreen
            && self.pipelined_submit
            && self.present_thread.is_some()
            && decouple_requested()
            && !crate::w18_ob_ab::off(crate::w18_ob_ab::DECOUPLE)
            && state
                .gpu
                .config
                .usage
                .contains(wgpu::TextureUsages::COPY_DST)
            && !crate::context::gpu_timing_requested()
        {
            let target = match self.decouple_target.take() {
                Some(target) if target.matches(&state.gpu) => target,
                _ => DecoupleTarget::new(&state.gpu),
            };
            {
                mc_core::profile_scope!("shell.phases_render");
                early = self.phases.record_unsubmitted(
                    &state.gpu,
                    &target.view,
                    &self.camera,
                    partial_ticks,
                );
            }
            self.decouple_target = Some(target);
            trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Render);
        } else if !crate::w18_ob_ab::off(crate::w18_ob_ab::DECOUPLE) {
            // The path is off: nothing keeps the target alive. (The measurement aid's off
            // frames keep it, so its on frames do not rebuild it.)
            self.decouple_target = None;
        }
        // Hand the recorded frame to the present thread now, without waiting
        // for its image (see `W19_O1_HANDOFF_ENV`); the image path below otherwise.
        if early.is_some()
            && handoff_requested()
            && self.decouple_target.is_some()
            && self
                .present_thread
                .as_ref()
                .is_some_and(PresentThread::can_hand_off)
        {
            if let (Some(enc), Some(thread), Some(target)) = (
                early.take(),
                self.present_thread.as_mut(),
                self.decouple_target.as_ref(),
            ) {
                // Pending behind the frames already handed off, without waiting for their submits.
                state.gpu.defer_frame_for_handoff(enc);
                trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Acquire);
                {
                    // The frame as recorded holds the bytes its image will hold.
                    mc_core::profile_scope!("shell.after_render");
                    let frame = AcquiredFrame::from_recorded_target(
                        target.texture.clone(),
                        target.view.clone(),
                    );
                    self.app.after_render(&state.gpu, &frame);
                }
                trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::AfterRender);
                let handed = {
                    mc_core::profile_scope!("shell.present");
                    thread.hand_off(
                        &mut state.gpu,
                        &target.texture,
                        iteration_started,
                        &mut self.input_to_present,
                    )
                };
                let skipped = thread.take_skipped();
                let dead = thread.is_dead();
                if let Some(trace) = self.trace.as_mut() {
                    for _ in 0..skipped {
                        trace.acquire_skipped();
                    }
                }
                if dead {
                    tracing::error!(
                        "pipelined present thread stopped; presenting inline from now on"
                    );
                    self.present_thread = None;
                    state.gpu.frame_pipe().abandon();
                }
                if !handed {
                    // The gated queue submits the frame, in order, with no image to show it in,
                    // as for an unavailable image.
                    state.gpu.flush_staged_writes();
                    if let Some(trace) = self.trace.as_mut() {
                        trace.acquire_skipped();
                    }
                    mc_core::profile_scope!("shell.limiter");
                    // Or the event loop's wait, for a window in the background.
                    if !self.pace_in_background() {
                        self.limiter
                            .wait(self.app.effective_frame_cap(&self.settings, self.focused));
                    }
                    return;
                }
                self.end_iteration(trace_clock, trace_phases, true);
                return;
            }
        }
        let acquired = {
            mc_core::profile_scope!("shell.acquire");
            match self.present_thread.as_mut() {
                Some(thread) if !offscreen => thread.take_frame(
                    &mut state.gpu,
                    &mut self.input_to_present,
                    iteration_started.elapsed(),
                ),
                _ => state.gpu.acquire(),
            }
        };
        if self
            .present_thread
            .as_ref()
            .is_some_and(PresentThread::is_dead)
        {
            tracing::error!("pipelined present thread stopped; presenting inline from now on");
            self.present_thread = None;
            // A frame it was handed is gone; queue calls must not wait for its submit.
            state.gpu.frame_pipe().abandon();
        }
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Acquire);
        let frame = match acquired {
            Ok(frame) => frame,
            // Occluded or timed out: skip this frame rather than stalling the loop.
            Err(RenderError::FrameUnavailable) => {
                if let Some(trace) = self.trace.as_mut() {
                    trace.acquire_skipped();
                }
                // A frame already recorded for an image that never came is
                // submitted with its staged writes (it only draws into the decoupled target).
                if let Some(enc) = early.take() {
                    state.gpu.submit_frame(enc.finish());
                }
                // PRESENT-7. `app.frame` has already staged this iteration's buffer writes and
                // nothing will be recorded; submit them now instead of letting every skipped
                // iteration's staging pile up until the window is visible again.
                state.gpu.flush_staged_writes();
                // java: `runGameLoop` calls `Display.sync(getLimitFramerate())` on every loop,
                // presented or not. Skipping it here made an occluded window spin uncapped.
                // Uncapped stays uncapped (`None` returns at once), as in vanilla.
                {
                    mc_core::profile_scope!("shell.limiter");
                    // Or the event loop's wait, for a window in the background
                    // (a minimized window lands here on every frame).
                    if !self.pace_in_background() {
                        self.limiter
                            .wait(self.app.effective_frame_cap(&self.settings, self.focused));
                    }
                }
                return;
            }
            Err(e) => {
                tracing::error!(error = %e, "frame acquisition failed");
                if let Some(enc) = early.take() {
                    state.gpu.submit_frame(enc.finish());
                }
                self.failure = Some(e);
                return;
            }
        };

        let pipelined_submit = self.pipelined_submit && self.present_thread.is_some() && !offscreen;
        // The frame recorded above goes into its image with one copy, in the
        // same encoder, and is left pending exactly as `render_deferred` leaves it (or, if the
        // present thread has just stopped, submitted here as the serial loop does). If the copy
        // cannot be made (the image does not match what was recorded: a reconfiguration between
        // the two), the recorded work is submitted with its staged uploads and this image is
        // skipped, as an unavailable one is; the next frame records at the new size. Phases are
        // never recorded twice in one frame.
        let mut recorded = false;
        if let Some(mut enc) = early.take() {
            let copied = match (
                self.decouple_target.as_ref(),
                frame.color_texture(&state.gpu),
            ) {
                (Some(target), Some(image)) => target.copy_into(&mut enc, image),
                _ => false,
            };
            if !copied {
                state.gpu.submit_frame(enc.finish());
                if let Some(trace) = self.trace.as_mut() {
                    trace.acquire_skipped();
                }
                drop(frame);
                mc_core::profile_scope!("shell.limiter");
                // Or the event loop's wait, for a window in the background.
                if !self.pace_in_background() {
                    self.limiter
                        .wait(self.app.effective_frame_cap(&self.settings, self.focused));
                }
                return;
            }
            if pipelined_submit {
                state.gpu.defer_frame(enc);
            } else {
                state.gpu.submit_frame(enc.finish());
            }
            recorded = true;
        }
        if !recorded {
            {
                mc_core::profile_scope!("shell.phases_render");
                if pipelined_submit {
                    self.phases.render_deferred(
                        &state.gpu,
                        frame.view(),
                        &self.camera,
                        partial_ticks,
                    );
                } else {
                    self.phases
                        .render(&state.gpu, frame.view(), &self.camera, partial_ticks);
                }
            }
            trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Render);
        }
        {
            mc_core::profile_scope!("shell.after_render");
            self.app.after_render(&state.gpu, &frame);
        }
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::AfterRender);
        if offscreen {
            // The frame stays offscreen (presenting it is a no-op); the window
            // shows the app's preview when it asks for one.
            mc_core::profile_scope!("shell.present");
            if self.app.wants_window_preview() {
                if let Ok(window) = state.gpu.acquire_window() {
                    self.app.draw_window_preview(&state.gpu, &frame, &window);
                    state.window.pre_present_notify();
                    window.present(&state.gpu.queue);
                }
            }
            drop(frame);
        } else {
            mc_core::profile_scope!("shell.present");
            match self.present_thread.as_mut() {
                // #14 step 1a: the thread presents and pre-acquires the next image while this
                // thread runs the next iteration's ticks and frame build. Step 1b: it first
                // finishes and submits the frame `render_deferred` left pending (`None` if an
                // `after_render` readback already had to submit it here).
                Some(thread) => {
                    let pending = if pipelined_submit {
                        state.gpu.take_pending_frame()
                    } else {
                        None
                    };
                    thread.present(frame, pending, iteration_started, &state.gpu);
                }
                None => {
                    state.window.pre_present_notify();
                    frame.present(&state.gpu.queue);
                    self.input_to_present.record(iteration_started.elapsed());
                }
            }
        }
        self.end_iteration(trace_clock, trace_phases, pipelined_submit);
    }

    /// The end of a presented iteration, shared by the image path and the hand-off:
    /// the frame counters, the frame cap and the pacing, and the trace row.
    fn end_iteration(
        &mut self,
        mut trace_clock: Option<Instant>,
        mut trace_phases: PhaseTimings,
        pipelined_submit: bool,
    ) {
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Present);
        let completed_present = trace_clock;
        self.drew_frame = true;
        // One cached load unless the measurement aid is on.
        crate::w18_ob_ab::frame_presented();

        if self.stats.frame() {
            tracing::debug!(fps = self.stats.fps(), "fps");
        }
        // java: Minecraft.runGameLoop — Display.sync(getLimitFramerate()). See `frame_cap`.
        {
            mc_core::profile_scope!("shell.limiter");
            // Or the event loop's wait, for a window in the background.
            if !self.pace_in_background() {
                self.limiter
                    .wait(self.app.effective_frame_cap(&self.settings, self.focused));
            }
            // Step 1b pacing, after the cap: hold this iteration's end until the present thread
            // is nearly done, so the event loop pumps the input that arrives meanwhile and the
            // next iteration reads it, instead of reading older input and waiting for the image.
            if pipelined_submit {
                if let Some(thread) = self.present_thread.as_mut() {
                    // At most one handed-off frame ahead of the next one.
                    if let Some(state) = self.state.as_mut() {
                        thread.settle_handoffs(&mut state.gpu, &mut self.input_to_present);
                    }
                    thread.pace();
                }
            }
        }
        trace_phase_boundary(&mut trace_clock, &mut trace_phases, FramePhase::Limiter);
        if let (Some(trace), Some(presented_at)) = (self.trace.as_mut(), completed_present) {
            if let Err(error) = trace.record_present_with_work(
                presented_at,
                self.app.diagnostic_phase(),
                trace_phases,
                self.app.diagnostic_work(),
            ) {
                trace.record_failure("invalid shell timing sequence");
                self.failure = Some(RenderError::FrameTrace(error.to_string()));
            }
        }
    }

    /// The frame cap of a window in the background, at the end of an iteration.
    /// The window is not focused, or it is minimized, and the app has a background cap
    /// ([`ShellApp::background_frame_cap`]): the next frame is scheduled at the lower of the two
    /// caps and waited for in the event loop (`about_to_wait`), where the focus coming back or a
    /// restore starts it at once. Returns `false` after a `bool` test for a focused window, and
    /// whenever the app has no background cap: the limiter then paces the frame as before.
    fn pace_in_background(&mut self) -> bool {
        let (app, settings, focused) = (&self.app, &self.settings, self.focused);
        let Some(cap) = background_cap(
            focused,
            self.background.minimized(),
            |minimized| app.background_frame_cap(minimized),
            || app.effective_frame_cap(settings, focused),
        ) else {
            return false;
        };
        self.background.schedule(cap, Instant::now());
        true
    }
}

/// Dispatch one shell simulation-clock update. The `None` arm is deliberately the ordinary
/// host-clock path; explicit diagnostic modes are the only arms that avoid a clock read.
fn advance_simulation_clock(
    timer: &mut Timer,
    clock: Option<DiagnosticClock>,
) -> Result<(), DiagnosticTimerError> {
    match clock {
        None => {
            timer.update();
            Ok(())
        }
        Some(DiagnosticClock::Step { micros }) => timer.advance_diagnostic_micros(micros),
        Some(DiagnosticClock::Frozen) => {
            timer.update_diagnostic_frozen();
            Ok(())
        }
        Some(DiagnosticClock::Invalid(_)) => {
            unreachable!("invalid diagnostic clocks are rejected by Runner::draw before dispatch")
        }
    }
}

/// Advance an enabled diagnostic clock without reading the clock in the ordinary path.
fn trace_phase_boundary(clock: &mut Option<Instant>, phases: &mut PhaseTimings, phase: FramePhase) {
    if let Some(previous) = clock.as_mut() {
        let now = Instant::now();
        phases.set(phase, now.duration_since(*previous));
        *previous = now;
    }
}

/// Set to anything but empty or `0` for a run that shares a desktop with someone working, such as
/// a benchmark batch: the window opens without taking focus, never captures the cursor, and
/// ignores keyboard and mouse input.
///
/// The cursor is the point of it, but ignoring input matters just as much for the numbers: with a
/// grabbed mouse, the person's own hand moved a "fixed" benchmark camera and changed the scene.
pub const BACKGROUND_WINDOW_ENV: &str = "MC_BACKGROUND_WINDOW";

fn background_window_requested() -> bool {
    std::env::var_os(BACKGROUND_WINDOW_ENV).is_some_and(|value| !value.is_empty() && value != "0")
}

/// The frame cap to hand [`FrameLimiter::wait`] this frame.
///
/// # This is a correctness fix, not an optimisation
///
/// This used to read "`None` when vsync is on, otherwise `maxFps`", on the reasoning that the
/// presentation engine already paces us and sleeping again would only add latency. That reasoning
/// is not vanilla's, and it made a `maxFps:120` setting do nothing: a measurement found
/// three runs at **359.8 / 359.7 / 359.8 fps** — the display's refresh rate — with
/// `enableVsync:true, maxFps:120` set.
///
/// java: `Minecraft.runGameLoop` ends with
///
/// ```java
/// if (this.isFramerateLimitBelowMax()) {
///     Display.sync(this.getLimitFramerate());
/// }
/// ```
///
/// and `isFramerateLimitBelowMax` is `this.gameSettings.limitFramerate < 260`. Vsync is nowhere in
/// that condition. LWJGL's `Display.setVSyncEnabled` and `Display.sync` are independent knobs and
/// 1.8.9 runs both: with vsync on and `maxFps:120` on a 360 Hz display, vanilla renders 120 frames
/// a second, each presented on a vblank. So does this, now.
///
/// The `>= 260` unlimited sentinel is `FRAMERATE_UNLIMITED` and lives in
/// [`GameSettings::framerate_limit`](crate::settings::GameSettings::framerate_limit), which
/// already implements exactly `isFramerateLimitBelowMax`; this function is now only the statement
/// that vsync does not enter into it.
#[cfg(test)]
fn frame_cap(settings: &GameSettings) -> Option<u32> {
    settings.framerate_limit()
}

/// `focused` is the window's focus: An unfocused window never holds the OS
/// pointer, whatever the game asks for ([`cursor_plan`]).
fn apply_cursor(state: &mut Live, input: &mut InputState, focused: bool) {
    let Some(requested) = input.cursor_mode() else {
        return;
    };
    // Taken even when the mode is already applied: a screen closed by a tick and another opened
    // by the same frame's input pass never reached the window as a grab, but `InputState` has
    // already put the GUI cursor at the centre, so the OS pointer must follow it there.
    let recentre = input.take_cursor_recentre();
    let plan = cursor_plan(
        requested,
        focused,
        state.applied_cursor,
        recentre,
        state.regrab_centre,
    );
    match plan {
        CursorPlan::Keep => {}
        CursorPlan::Grab { centre_first } => {
            // Back from another window with the grab still wanted: LWJGL 2's
            // `updateCursor` centres the pointer before it clips it. `Locked` pins the pointer
            // where it is, which after Alt-Tab can be outside the window (another monitor), where
            // a click would land in another application. The release left no clip, so the warp
            // is not clamped.
            state.regrab_centre = false;
            if centre_first {
                warp_to_release_point(&state.window);
            }
            // Locked is the mode we want. winit 0.30 implements it on Windows too, as a 1x1
            // `ClipCursor` rect at the pointer (see the Free arm); Confined is the fallback for
            // backends without it, such as X11.
            let grabbed = state
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(e) = grabbed {
                tracing::warn!(error = %e, "cursor grab unavailable");
            }
            state.window.set_cursor_visible(false);
            state.applied_cursor = Some(CursorMode::Grabbed);
        }
        CursorPlan::Free { recentre } => {
            // java: MouseHelper.ungrabMouseCursor warps to the centre, then releases. Only a
            // focused grabbed -> free edge asks for the warp (`InputState::request_cursor`).
            //
            // The order has to depend on the platform. On Windows, winit 0.30 implements
            // `Locked` as a 1x1 `ClipCursor` rect at the pointer, and `SetCursorPos` is clamped
            // to an active clip while still returning success. So LWJGL's warp-first order moved
            // nothing there, and every screen opened wherever the last grab had caught the
            // pointer (the 2026-11-07 "cursor doesn't go back to the middle" report). Release
            // first on Windows; the cursor is still hidden, so the warp is not seen.
            //
            // Wayland is the opposite: winit moves only a locked pointer there (the position is
            // a hint applied on unlock), so the other platforms keep LWJGL's order. X11 and macOS
            // accept either.
            #[cfg(not(windows))]
            if recentre {
                warp_to_release_point(&state.window);
            }
            // The focus event already released a grab; should one ever still
            // be applied while unfocused, winit cannot drop its clip, so do it here too.
            if !focused && state.applied_cursor == Some(CursorMode::Grabbed) {
                os_cursor::release_clip();
            }
            let _ = state.window.set_cursor_grab(CursorGrabMode::None);
            #[cfg(windows)]
            if recentre {
                warp_to_release_point(&state.window);
            }
            state.window.set_cursor_visible(true);
            state.applied_cursor = Some(CursorMode::Free);
        }
    }
}

/// What [`apply_cursor`] does to the OS pointer this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CursorPlan {
    /// The pointer is already as it should be.
    Keep,
    /// Grab (hide and lock) it, warping it to the window centre first when `centre_first`.
    Grab { centre_first: bool },
    /// Release (show and free) it, warping it to the release point when `recentre`.
    Free { recentre: bool },
}

/// The OS pointer follows the game's request while the window has focus and
/// is free otherwise. `applied` is the mode last applied, `recentre` a pending release re-centre
/// ([`InputState::take_cursor_recentre`]), `regrab_centre` a focus loss released the grab since.
///
/// java (LWJGL 2.9.4, `WindowsDisplay`): `shouldGrab()` is `!isMinimized && isFocused &&
/// Mouse.isGrabbed()`, `updateClipping` clips only while focused (else `resetCursorClipping`), and
/// `updateCursor` centres the pointer whenever it grabs. The request itself (`Mouse.isGrabbed()`,
/// vanilla's `inGameHasFocus`) is not touched by a focus loss: the pause menu that
/// `EntityRenderer.updateCameraAndRender` opens after 500 ms of lost focus is what ungrabs it,
/// and without that menu (`pauseOnLostFocus:false`) refocusing grabs again.
///
/// The centring is Windows-only, where the 1x1 `Locked` clip would otherwise keep the pointer
/// wherever Alt-Tab left it; elsewhere a warp can fail (Wayland) and is not needed.
fn cursor_plan(
    requested: CursorMode,
    focused: bool,
    applied: Option<CursorMode>,
    recentre: bool,
    regrab_centre: bool,
) -> CursorPlan {
    if !focused {
        // Never re-centre for an unfocused window: the pointer is another application's then.
        return if applied == Some(CursorMode::Free) {
            CursorPlan::Keep
        } else {
            CursorPlan::Free { recentre: false }
        };
    }
    match requested {
        CursorMode::Grabbed if applied == Some(CursorMode::Grabbed) => CursorPlan::Keep,
        CursorMode::Grabbed => CursorPlan::Grab {
            centre_first: regrab_centre && cfg!(windows),
        },
        CursorMode::Free if applied == Some(CursorMode::Free) && !recentre => CursorPlan::Keep,
        CursorMode::Free => CursorPlan::Free { recentre },
    }
}

/// Whether a focus loss has a grab to give back: the game's grab, or the
/// OS-pointer look's confinement (Raw Mouse Input off). Without one, any clip is someone else's.
fn focus_loss_releases(applied: Option<CursorMode>, applied_os_pointer: bool) -> bool {
    applied == Some(CursorMode::Grabbed) || applied_os_pointer
}

/// The window lost focus (the Windows key's Start menu, Alt-Tab, a click on
/// another monitor): give the pointer back at once, as LWJGL 2's `appActivate(false)` does
/// through `updateCursor`/`updateClipping`. Bug report: "pressing the
/// windows key traps your mouse in full screen". Returns whether it released a grab.
///
/// Synchronous, from the focus event: a minimised window may get no frame to do it in. winit
/// 0.30's own release is not enough on Windows: its `refresh_os_cursor` changes the clip only while
/// the window is the active one, which it no longer is, so `set_cursor_grab(None)` from here (or
/// from the 500 ms pause menu, as before this fix) left the one-pixel `Locked` clip in place and
/// the pointer trapped. [`os_cursor::release_clip`] is LWJGL's `resetCursorClipping`.
fn release_grab_on_focus_loss(
    window: &Window,
    applied_cursor: &mut Option<CursorMode>,
    applied_os_pointer: &mut bool,
) -> bool {
    if !focus_loss_releases(*applied_cursor, *applied_os_pointer) {
        return false;
    }
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    if *applied_os_pointer {
        window.set_cursor(winit::window::CursorIcon::Default);
        *applied_os_pointer = false;
    }
    window.set_cursor_visible(true);
    os_cursor::release_clip();
    *applied_cursor = Some(CursorMode::Free);
    true
}

/// The one Win32 call winit 0.30 cannot make for an inactive window.
#[cfg(windows)]
mod os_cursor {
    use std::ffi::c_void;

    #[link(name = "user32")]
    extern "system" {
        /// `BOOL ClipCursor(const RECT *lpRect)`.
        fn ClipCursor(rect: *const c_void) -> i32;
    }

    /// `ClipCursor(NULL)`: the pointer may go anywhere on the desktop again.
    pub(super) fn release_clip() {
        // SAFETY: a null rectangle is the documented "no clip" argument; nothing is read or kept.
        if unsafe { ClipCursor(std::ptr::null()) } == 0 {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "could not release the cursor clip"
            );
        }
    }

    /// `RECT`, as `ClipCursor`/`GetClipCursor` take it (the real-window test only).
    #[cfg(test)]
    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub(super) struct Rect {
        pub(super) left: i32,
        pub(super) top: i32,
        pub(super) right: i32,
        pub(super) bottom: i32,
    }

    #[cfg(test)]
    #[link(name = "user32")]
    extern "system" {
        fn GetClipCursor(rect: *mut Rect) -> i32;
    }

    /// Clip the pointer to `rect` (screen pixels). Only the real-window test uses it.
    #[cfg(test)]
    pub(super) fn set_clip(rect: Rect) -> bool {
        // SAFETY: `rect` is a valid, initialised `RECT` that outlives the call; Windows copies it.
        unsafe { ClipCursor(std::ptr::from_ref(&rect).cast()) != 0 }
    }

    /// The current clip rectangle (the whole virtual desktop when nothing is clipped).
    #[cfg(test)]
    pub(super) fn clip() -> Option<Rect> {
        let mut rect = Rect::default();
        // SAFETY: `rect` is a valid, writable `RECT` for the duration of the call.
        (unsafe { GetClipCursor(&mut rect) } != 0).then_some(rect)
    }
}

/// The focus test that leaves a clip behind is in winit's Windows backend; the other platforms
/// need nothing beyond `set_cursor_grab(None)`.
#[cfg(not(windows))]
mod os_cursor {
    pub(super) fn release_clip() {}
}

/// Move the OS pointer to vanilla's release point. `inner_size` is physical pixels, as
/// `Display.getWidth/getHeight` are.
fn warp_to_release_point(window: &Window) {
    let centre = cursor_centre(window.inner_size());
    if let Err(e) = window.set_cursor_position(centre) {
        // Winit cannot move the pointer on several backends. Releasing/showing the cursor must
        // still succeed there.
        tracing::warn!(error = %e, "cursor re-centre unavailable while ungrabbing");
    }
}

/// The physical-pixel target of vanilla's
/// `Mouse.setCursorPosition(Display.getWidth() / 2, Display.getHeight() / 2)`, in winit's
/// top-left frame. It is [`cursor_release_point`](crate::input::cursor_release_point), the same
/// point `InputState` gives the GUI cursor, so the GUI and the OS pointer agree.
///
/// This is deliberately a pure helper: opening a real winit window is not viable in unit tests,
/// while the dimensions and Java's truncation rule are the part of the contract we need to pin.
fn cursor_centre(size: PhysicalSize<u32>) -> PhysicalPosition<i32> {
    let (x, y) = crate::input::cursor_release_point(size.width, size.height);
    // Both coordinates are at most u32::MAX / 2, so they fit an i32.
    PhysicalPosition::new(x as i32, y as i32)
}

/// Whether [`apply_fullscreen`] needs to call `Window::set_fullscreen`, and with what, given the
/// state it last applied and what the app is asking for this frame.
///
/// Split out as a pure function so the dedupe logic — the part a test can prove without opening
/// a real window — has one: `apply_fullscreen` itself needs a live [`Window`] and so cannot be
/// unit tested here.
fn fullscreen_reconciliation(applied: Option<bool>, wants: Option<bool>) -> Option<bool> {
    let want = wants?;
    if applied == Some(want) {
        return None;
    }
    Some(want)
}

/// Reconcile the OS window's fullscreen state with [`ShellApp::wants_fullscreen`].
///
/// java: this is the other half of `Minecraft.toggleFullscreen` (`:1643`) — `mc-client`'s
/// `toggle_fullscreen` only flips and saves the setting; this is what actually reaches the
/// window, every frame, regardless of what triggered the change.
fn apply_fullscreen(state: &mut Live, wants: Option<bool>, exclusive: bool) {
    let Some(kind) = fullscreen_change(
        state.applied_fullscreen,
        state.applied_exclusive,
        wants,
        exclusive,
    ) else {
        return;
    };
    // Borderless is mc-rs's F11 and the default: no mode switch, instant alt-tab, and the
    // measured Independent Flip fast path (the project notes, "Display latency"). Exclusive is the
    // opt-in LWJGL 2 behaviour (Borderless Fullscreen off): `Display.setFullscreen`
    // switches the display mode with `ChangeDisplaySettings(CDS_FULLSCREEN)`.
    let fullscreen = match kind {
        FullscreenKind::Windowed => None,
        FullscreenKind::Borderless => Some(Fullscreen::Borderless(None)),
        FullscreenKind::Exclusive => match exclusive_video_mode(&state.window) {
            Some(mode) => Some(Fullscreen::Exclusive(mode)),
            None => {
                tracing::warn!("no video mode for exclusive fullscreen; using borderless");
                Some(Fullscreen::Borderless(None))
            }
        },
    };
    state.window.set_fullscreen(fullscreen);
    state.applied_fullscreen = Some(kind != FullscreenKind::Windowed);
    state.applied_exclusive = exclusive;
}

/// What [`apply_fullscreen`] hands `Window::set_fullscreen`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FullscreenKind {
    Windowed,
    Borderless,
    Exclusive,
}

/// The fullscreen change [`apply_fullscreen`] must make, if any: [`fullscreen_reconciliation`]'s
/// dedupe, plus a re-apply when the window is already fullscreen and the requested kind
/// ([`ShellApp::wants_exclusive_fullscreen`]) changed. Pure, for the same reason.
fn fullscreen_change(
    applied: Option<bool>,
    applied_exclusive: bool,
    wants: Option<bool>,
    exclusive: bool,
) -> Option<FullscreenKind> {
    let want = match fullscreen_reconciliation(applied, wants) {
        Some(want) => want,
        None if wants == Some(true) && applied_exclusive != exclusive => true,
        None => return None,
    };
    Some(match (want, exclusive) {
        (false, _) => FullscreenKind::Windowed,
        (true, false) => FullscreenKind::Borderless,
        (true, true) => FullscreenKind::Exclusive,
    })
}

/// The window's monitor's video mode for exclusive fullscreen: see [`pick_video_mode`].
fn exclusive_video_mode(window: &Window) -> Option<winit::monitor::VideoModeHandle> {
    let monitor = window.current_monitor()?;
    let modes: Vec<winit::monitor::VideoModeHandle> = monitor.video_modes().collect();
    let keys: Vec<(PhysicalSize<u32>, u16, u32)> = modes
        .iter()
        .map(|mode| {
            (
                mode.size(),
                mode.bit_depth(),
                mode.refresh_rate_millihertz(),
            )
        })
        .collect();
    let index = pick_video_mode(&keys, monitor.size(), monitor.refresh_rate_millihertz())?;
    modes.into_iter().nth(index)
}

/// Index of the exclusive video mode among `(size, bit depth, refresh mHz)` entries: the
/// monitor's own resolution (no rescale, as LWJGL's desktop mode), at its current refresh
/// rate if offered (a 360 Hz display stays at 360 Hz), else the highest refresh, then the
/// highest bit depth. `None` when no mode has the monitor's size.
fn pick_video_mode(
    modes: &[(PhysicalSize<u32>, u16, u32)],
    size: PhysicalSize<u32>,
    current_refresh_millihertz: Option<u32>,
) -> Option<usize> {
    modes
        .iter()
        .enumerate()
        .filter(|(_, (mode_size, _, _))| *mode_size == size)
        .max_by_key(|(_, (_, depth, refresh))| {
            (
                current_refresh_millihertz == Some(*refresh),
                *refresh,
                *depth,
            )
        })
        .map(|(index, _)| index)
}

/// Raw Mouse Input off: LWJGL 2's grabbed mouse on Windows. While the game holds
/// the grab and [`InputState::os_pointer_look`] is on, the pointer is confined to the client
/// rectangle and hidden by a blank icon, so it can move and report `CursorMoved`, and after a
/// frame that read motion it is warped back to the centre. With raw input on (the default) this
/// is one `bool` test per frame.
///
/// winit clips a grabbed pointer to a single pixel whenever it is hidden (`Locked` always;
/// `Confined` with `set_cursor_visible(false)` too: winit-0.30.13 `refresh_os_cursor`), which
/// is why this cannot simply ask for `Confined`.
fn apply_pointer_look(state: &mut Live, input: &mut InputState, focused: bool) {
    if !input.os_pointer_look() && !state.applied_os_pointer {
        return;
    }
    let want = input.os_pointer_look() && state.applied_cursor == Some(CursorMode::Grabbed);
    let recentre = input.take_pointer_recentre();
    if want != state.applied_os_pointer {
        state.applied_os_pointer = want;
        if want {
            if let Err(e) = state.window.set_cursor_grab(CursorGrabMode::Confined) {
                tracing::warn!(error = %e, "pointer confinement unavailable");
            }
            match &state.blank_cursor {
                Some(blank) => state.window.set_cursor(blank.clone()),
                // Look still works; the arrow just stays visible.
                None => tracing::warn!("no blank cursor; the pointer stays visible in game"),
            }
            state.window.set_cursor_visible(true);
            recentre_pointer(&state.window, input);
        } else {
            state.window.set_cursor(winit::window::CursorIcon::Default);
            if state.applied_cursor == Some(CursorMode::Grabbed) {
                // Back to raw input while still in game: redo the ordinary grab.
                state.applied_cursor = None;
                apply_cursor(state, input, focused);
            }
        }
    } else if want && recentre {
        recentre_pointer(&state.window, input);
    }
}

/// Warp the pointer to the window centre and measure the next OS-pointer delta from there
/// (LWJGL's `WindowsDisplay.centerCursor`).
fn recentre_pointer(window: &Window, input: &mut InputState) {
    let centre = cursor_centre(window.inner_size());
    match window.set_cursor_position(centre) {
        Ok(()) => input.pointer_recentred(f64::from(centre.x), f64::from(centre.y)),
        Err(e) => {
            // The pointer stayed put: measure from its last known position.
            let (x, y) = input.cursor();
            input.pointer_recentred(x, y);
            tracing::debug!(error = %e, "pointer re-centre unavailable");
        }
    }
}

/// Whether [`apply_present`] needs to reconfigure, and for what. The same dedupe as
/// [`fullscreen_reconciliation`], kept pure for the same reason.
fn present_reconciliation(
    applied: Option<PresentRequest>,
    wants: Option<PresentRequest>,
) -> Option<PresentRequest> {
    let want = wants?;
    if applied == Some(want) {
        return None;
    }
    Some(want)
}

/// Apply a changed [`ShellApp::wants_present`] request (C1-10), already deduplicated by
/// [`present_reconciliation`]. Returns the newly configured mode, for
/// [`RunSummary::present_mode`], or `None` when the present thread did not give its image back
/// in time (try again next frame; see [`apply_present_request`]).
fn apply_present(
    state: &mut Live,
    thread: Option<&mut PresentThread>,
    want: PresentRequest,
) -> Option<wgpu::PresentMode> {
    let mode = apply_present_request(&mut state.gpu, thread, want)?;
    tracing::info!(
        vsync = want.vsync,
        preference = want.preference.as_str(),
        frame_latency = state.gpu.frame_latency(),
        present_mode = ?mode,
        "present options applied",
    );
    state.applied_present = Some(want);
    Some(mode)
}

/// The swap-chain half of a present request at its configure point:
/// `None` when the present thread did not give its image back in time, else the mode now
/// configured. As for a resize ([`resize_swap_chain`]), the thread's image is taken back only
/// when the request configures the swap chain (another mode or latency) or an image is already
/// lost: the VSync button with an explicit `presentMode` changes the request but not the mode,
/// and dropped an image each time.
fn apply_present_request(
    gpu: &mut Gpu,
    mut thread: Option<&mut PresentThread>,
    want: PresentRequest,
) -> Option<wgpu::PresentMode> {
    let configures = gpu.present_options_configure(want.preference, want.vsync, want.frame_latency);
    if !take_back_image(thread.as_deref_mut(), configures) {
        return None;
    }
    let mode = want.apply(gpu);
    settle_lost_image(gpu, thread, configures);
    Some(mode)
}

/// The swap-chain half of a pending resize (PRESENT-9): `None` when
/// the present thread did not give its image back in time (keep the size for the next frame),
/// else whether the GPU rebuilt its attachments ([`Gpu::resize`]).
///
/// The thread's image is taken back only when this size configures the swap chain (a new size
/// with an area) or an image is already lost. An image dropped without a present is not given
/// back to a Vulkan swap chain (wgpu's discard does nothing there) until the next configure, and
/// a repeated size (Windows sends one on a restore and on frame changes) or a minimised window's
/// 0×0 configures nothing: each dropped one, and from the second on no frame reached the screen.
fn resize_swap_chain(
    gpu: &mut Gpu,
    mut thread: Option<&mut PresentThread>,
    width: u32,
    height: u32,
) -> Option<bool> {
    let configures = gpu.resize_configures(width, height);
    if !take_back_image(thread.as_deref_mut(), configures) {
        return None;
    }
    let rebuilt = gpu.resize(width, height);
    settle_lost_image(gpu, thread, configures);
    Some(rebuilt)
}

/// Before a configure point: take back the present thread's image
/// ([`PresentThread::quiesce`]) when the swap chain is about to be configured (`configures`) or
/// an image is already lost ([`PresentThread::lost_image`]). `false` when the thread did not
/// answer in time. Nothing to take back without a thread (serial presenting, headless).
fn take_back_image(thread: Option<&mut PresentThread>, configures: bool) -> bool {
    match thread {
        Some(thread) if configures || thread.lost_image => thread.quiesce(),
        _ => true,
    }
}

/// After a configure point, once no image is held: the swap chain was
/// configured (`configured`) and has every image again; or an image the present thread dropped
/// since is still lost, and a configure now gives it back. Not while minimised (a zero-area
/// surface cannot be configured): the configure point of the restore does it then.
fn settle_lost_image(gpu: &mut Gpu, thread: Option<&mut PresentThread>, configured: bool) {
    if let Some(thread) = thread {
        if configured || (thread.lost_image && gpu.rebuild_swap_chain()) {
            thread.lost_image = false;
        }
    }
}

/// Set to anything but empty or `0` to present from a dedicated thread (#14 step 1a, opt-in).
///
/// The main thread keeps everything it did before — ticks, `ShellApp::frame`, recording **and
/// `queue.submit`** — so every `queue.write_*` of a frame is flushed by that frame's own submit
/// on this thread before the next frame writes anything (the write-leak hazard that blocks a
/// submit thread until the upload ring exists). Only `present` and the *next* frame's
/// `get_current_texture` (the vblank / frame-latency wait) move to the thread, overlapping the
/// next iteration's ticks and frame build. No frame of latency is added: the image is still
/// taken right before recording, after input was sampled, as `Gpu::acquire` did.
pub const PIPELINED_PRESENT_ENV: &str = "MC_PIPELINED_PRESENT";

fn pipelined_present_requested() -> bool {
    std::env::var_os(PIPELINED_PRESENT_ENV).is_some_and(|value| !value.is_empty() && value != "0")
}

/// Set to anything but empty or `0` to also finish and submit each frame on the present thread
/// (#14 step 1b, opt-in; starts the present thread of [`PIPELINED_PRESENT_ENV`] too).
///
/// The main thread records the frame exactly as before and leaves it pending
/// ([`PhaseList::render_deferred`]); the present thread runs its `finish()` (wgpu's encoding of
/// every recorded pass) and `submit`, then presents, overlapping the next iteration's ticks and
/// frame build. Every main-thread queue call is ordered after that submit by
/// [`crate::frame_pipe::GatedQueue`], so writes and frames reach the GPU in the serial loop's
/// order; see [`crate::frame_pipe`].
pub const PIPELINED_SUBMIT_ENV: &str = "MC_PIPELINED_SUBMIT";

fn pipelined_submit_requested() -> bool {
    std::env::var_os(PIPELINED_SUBMIT_ENV).is_some_and(|value| !value.is_empty() && value != "0")
}

/// Set to `0` to turn off step 1b's iteration pacing ([`PresentThread::pace`]), for A/B runs.
pub const PIPELINED_PACE_ENV: &str = "MC_PIPELINED_PACE";

fn pipelined_pace_requested() -> bool {
    std::env::var_os(PIPELINED_PACE_ENV).is_none_or(|value| value != "0")
}

/// Set to `0` to record each pipelined frame straight into its swap-chain
/// image again. Read once.
///
/// With [`PIPELINED_SUBMIT_ENV`] on, the main thread needed the next swap-chain image before it
/// could record a single pass, and that image comes from the present thread only after it has
/// finished, submitted and presented the previous frame and acquired the next image. So the two
/// threads took turns: the present thread's work, then the main thread's recording, then the
/// present thread again. Now the frame is recorded into [`DecoupleTarget`] (the frame's size and
/// format) before the main thread waits for the image, and the image only receives one copy of
/// it, appended to the same command encoder. Recording overlaps the present thread's work on the
/// previous frame. Every pass writes the same bytes into the same format, and the copy is exact,
/// so the image is byte-for-byte what it was.
pub const W18_OB_DECOUPLE_ENV: &str = "MC_W18_OB_DECOUPLE";

/// Whether pipelined frames are recorded before their image exists: the
/// pipelined submit is on and [`W18_OB_DECOUPLE_ENV`] is not `0`. [`Gpu::for_window_with`] asks
/// the swap chain for `COPY_DST` only when this holds, so with the switch off (or without the
/// pipelined submit) the surface is configured exactly as before.
pub(crate) fn decouple_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pipelined_submit_requested()
            && std::env::var_os(W18_OB_DECOUPLE_ENV).is_none_or(|value| value != "0")
    })
}

/// Set to `0` to have every frame drawn from the window's
/// `RedrawRequested` event again. Read once.
///
/// The loop asked for each frame with `Window::request_redraw` from `about_to_wait`, and the
/// operating system then had to synthesise a paint message, which the event loop fetched and
/// dispatched to the window as `RedrawRequested`: a round trip through the window system per
/// frame, on the main thread, between two frames. Now, while frames are being presented, the next
/// one is drawn straight from `about_to_wait`. That is the same point in the loop (after every
/// pending window event has been delivered, and the paint message has the lowest priority, so it
/// came after them too). The first frame, and every frame after one that presented nothing (a
/// minimised or occluded window), still go through `request_redraw`, and a paint the system
/// sends on its own (an exposed or resized window) still draws.
pub const W18_OB_DIRECT_DRAW_ENV: &str = "MC_W18_OB_DIRECT_DRAW";

/// Whether [`W18_OB_DIRECT_DRAW_ENV`] leaves frames drawn from
/// `about_to_wait` (it is not `0`). Read once.
fn direct_draw_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(W18_OB_DIRECT_DRAW_ENV).is_none_or(|value| value != "0"))
}

/// Set to `0` to have the main thread wait for each pipelined frame's
/// swap-chain image again. Read once.
///
/// With [`W18_OB_DECOUPLE_ENV`] a frame is recorded before its image exists, but the main thread
/// still waited for that image before handing the frame off, and the present thread hands an
/// image over only after it has finished, submitted and presented the previous frame and
/// acquired the next image. So the threads still took turns once per frame: the present
/// thread's work, a wake-up of each thread and the main thread's copy step, in a row. Now the
/// main thread hands the frame off as soon as it is recorded. The present thread keeps the image
/// it acquires after each present, appends the copy of the frame into it (the same exact copy)
/// and presents it. A frame may wait behind the one being presented only while the present
/// thread is expected to be free within [`PresentThread::PACE_LIMIT`] (never with vsync or a
/// GPU-bound frame, whose waits are longer); the pacing steers that wait to the few microseconds
/// it steered the image wait to. Frames that cannot be handed off (the first one, any after an
/// unavailable or outdated image, a resize) take the image path as before.
pub const W19_O1_HANDOFF_ENV: &str = "MC_W19_O1_HANDOFF";

/// Whether pipelined frames are handed off before their image: the decoupled
/// recording is on ([`decouple_requested`]) and [`W19_O1_HANDOFF_ENV`] is not `0`. Read once.
fn handoff_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        decouple_requested()
            && std::env::var_os(W19_O1_HANDOFF_ENV).is_none_or(|value| value != "0")
    })
}

/// Set to `0` so a handed-off frame never waits behind the one being presented
/// (the main thread then waits for that one to be done, as it waited for its image). A
/// measurement aid for [`W19_O1_HANDOFF_ENV`]. Read once.
pub const W19_O1_HANDOFF_QUEUE_ENV: &str = "MC_W19_O1_HANDOFF_QUEUE";

fn handoff_queue_requested() -> bool {
    std::env::var_os(W19_O1_HANDOFF_QUEUE_ENV).is_none_or(|value| value != "0")
}

/// The work a hand-off brings the present thread.
enum HandedFrame {
    /// The frame as recorded: the thread appends the copy into the image and finishes it.
    Pending(crate::frame_pipe::PendingFrame),
    /// A readback on the main thread already submitted the frame: the copy alone, in an encoder
    /// of its own.
    Submitted,
}

/// Set to diagnostic phase labels ([`ShellApp::diagnostic_phase`], such as
/// the SkyWars bench's `sw_join,sw_fight,sw_tail`, comma-separated) to also keep the
/// input-to-present samples that arrive while a traced run (`MC_FRAME_TRACE`) is in each of those
/// phases, and print one summary per phase on stderr when the run ends. A measurement aid: the run-wide summary ([`RunSummary::input_to_present`]) mixes in a
/// bench run's loading frames, which are more than one percent of it and so decide its p99.
/// Unset: nothing is kept, read or printed.
pub const W19_O1_I2P_PHASE_ENV: &str = "MC_W19_O1_I2P_PHASE";

/// The shell's input-to-present samples ([`RunSummary::input_to_present`]), and
/// those of each diagnostic phase [`W19_O1_I2P_PHASE_ENV`] names.
struct InputToPresent {
    all: crate::frame::FrameTimeHistogram,
    phases: Vec<(String, crate::frame::FrameTimeHistogram)>,
    /// The entry of `phases` the frame being drawn belongs to.
    current: Option<usize>,
}

impl InputToPresent {
    fn new() -> Self {
        let phases = std::env::var(W19_O1_I2P_PHASE_ENV)
            .map(|labels| {
                labels
                    .split(',')
                    .map(str::trim)
                    .filter(|label| !label.is_empty())
                    .map(|label| (label.to_owned(), crate::frame::FrameTimeHistogram::new()))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            all: crate::frame::FrameTimeHistogram::new(),
            phases,
            current: None,
        }
    }

    fn record(&mut self, latency: Duration) {
        self.all.record(latency);
        if let Some(index) = self.current {
            self.phases[index].1.record(latency);
        }
    }

    fn summary(&self) -> FrameTimeSummary {
        self.all.summary()
    }
}

/// Whether `image` can receive a copy of `recorded`: the same size and format,
/// and copyable into. False after a reconfiguration between recording and the image.
fn recorded_copy_fits(recorded: &wgpu::Texture, image: &wgpu::Texture) -> bool {
    image.size() == recorded.size()
        && image.format() == recorded.format()
        && image.usage().contains(wgpu::TextureUsages::COPY_DST)
}

/// Append the copy of `recorded` (the frame as recorded) into `image`. The
/// caller has checked [`recorded_copy_fits`].
fn copy_recorded_frame(
    enc: &mut wgpu::CommandEncoder,
    recorded: &wgpu::Texture,
    image: &wgpu::Texture,
) {
    enc.copy_texture_to_texture(
        recorded.as_image_copy(),
        image.as_image_copy(),
        recorded.size(),
    );
}

/// `a - b` in seconds: negative when `a` is earlier than `b`.
fn seconds_between(a: Instant, b: Instant) -> f64 {
    if a >= b {
        (a - b).as_secs_f64()
    } else {
        -(b - a).as_secs_f64()
    }
}

/// The colour target a pipelined frame is recorded into before its image is
/// acquired. Single-sampled like the image it stands in for (with MSAA it is the resolve target,
/// where the image sat), the frame's size and format, and copyable from: for the copy into the
/// image and for scene-mode snapshots, which then draw straight into it as they did into the
/// image. Built on first use, rebuilt when the frame's size or format changes.
struct DecoupleTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl DecoupleTarget {
    fn new(gpu: &Gpu) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-render frame target (recorded before the image)"),
            size: wgpu::Extent3d {
                width: gpu.config.width.max(1),
                height: gpu.config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    fn matches(&self, gpu: &Gpu) -> bool {
        self.texture.format() == gpu.config.format
            && self.texture.width() == gpu.config.width.max(1)
            && self.texture.height() == gpu.config.height.max(1)
    }

    /// Append the copy of this target into `image` to `enc`. `false`, with nothing recorded, when
    /// the two differ in size or format (a resize between recording and the image).
    fn copy_into(&self, enc: &mut wgpu::CommandEncoder, image: &wgpu::Texture) -> bool {
        // The same check and copy the present thread makes for a hand-off.
        if !recorded_copy_fits(&self.texture, image) {
            return false;
        }
        copy_recorded_frame(enc, &self.texture, image);
        true
    }
}

/// A command for the present thread. Each one is answered by exactly one [`Acquired`], except
/// `Exit`.
enum PresentCmd {
    /// Acquire the next image (first frame, or after [`PresentThread::quiesce`]).
    Acquire,
    /// Present this image, then acquire the next one. `frame` is its still-unsubmitted work
    /// (step 1b), which is finished and submitted first; `None` when the main thread already
    /// submitted it. `started` is the iteration's start, for [`RunSummary::input_to_present`].
    Present {
        texture: wgpu::SurfaceTexture,
        frame: Option<crate::frame_pipe::PendingFrame>,
        started: Instant,
    },
    /// Copy `recorded` (the frame as recorded) into an image, submit, present,
    /// then acquire the next image and keep it ([`W19_O1_HANDOFF_ENV`]). The image is `image`
    /// when the main thread brings one (the answer to the frame before), else the one the thread
    /// kept, else a fresh one. `frame` is the frame's work ([`HandedFrame`]).
    /// Answered with [`Acquired::Held`] or why no image is kept, never with an image.
    Handoff {
        frame: HandedFrame,
        recorded: wgpu::Texture,
        image: Option<wgpu::SurfaceTexture>,
        started: Instant,
        handed_at: Instant,
    },
    /// Drop the image the thread keeps, so the main thread may configure the
    /// surface. Answered with [`Acquired::Unavailable`].
    Release,
    Exit,
}

/// The present thread's answer: the next frame's image, or why there is none.
enum Acquired {
    Frame(wgpu::SurfaceTexture),
    /// The thread keeps the next image for the next hand-off.
    Held,
    /// Occluded, timed out or a validation error: skip the frame, as `Gpu::acquire` does.
    Unavailable,
    /// Outdated or lost: the main thread reconfigures (the thread is idle and holds nothing).
    Reconfigure,
}

/// One answer of the present thread, in the order of the commands.
struct Answer {
    /// The next image or why there is none (see [`Acquired`]).
    next: Acquired,
    /// How long the image just presented took from its iteration's start (`None`: nothing
    /// presented).
    presented: Option<Duration>,
    /// How long the thread worked on this command (submit, present, acquire).
    busy: Duration,
    /// When it answered.
    answered: Instant,
    /// For a hand-off: when the thread was free minus when the frame was
    /// handed off, in seconds. Positive: the frame waited behind the one before; negative: the
    /// thread waited for the frame.
    queue_error: Option<f64>,
    /// A hand-off that found no image to show its frame in (its work was still
    /// submitted).
    skipped: bool,
}

/// #14 step 1a: presents frames and pre-acquires the next image off the main thread.
///
/// Protocol: at most one command is outstanding. `outstanding` is true from sending `Acquire`
/// or `Present` until its [`Acquired`] has been received, and the thread holds a surface texture
/// only inside that window. [`Self::quiesce`] closes the window (dropping an unpresented image,
/// which wgpu allows), which is what makes any `Surface::configure` on the main thread safe.
///
/// Hand-offs ([`PresentCmd::Handoff`], [`W19_O1_HANDOFF_ENV`]) are answered
/// without an image, and up to two may be in flight (`in_flight`): one being presented and one
/// waiting. Either `outstanding` or `in_flight` is set, never both (answers come in command
/// order). After a hand-off the thread keeps the next image (`holds_image`); [`Self::quiesce`]
/// collects every answer and has the thread drop it ([`PresentCmd::Release`]).
struct PresentThread {
    cmd: std::sync::mpsc::Sender<PresentCmd>,
    ready: std::sync::mpsc::Receiver<Answer>,
    outstanding: bool,
    /// Hand-offs and releases sent and not yet answered.
    in_flight: u8,
    /// The last answer collected says the thread keeps an image.
    holds_image: bool,
    /// [`Self::quiesce`] dropped an image without presenting it since the shell
    /// last configured the swap chain. A Vulkan swap chain gets such an image back only from a
    /// configure (wgpu's discard does nothing there), so the next configure point makes one
    /// ([`settle_lost_image`]).
    lost_image: bool,
    /// The last answer collected brought or kept an image, so the next frame
    /// may be handed off. False at the start and after an unavailable or outdated image, a skipped
    /// hand-off or a release: the next frame then takes the image path, which deals with those.
    image_ok: bool,
    /// Hand-offs answered as skipped and not yet reported to the frame trace.
    skipped: u32,
    /// [`W19_O1_HANDOFF_QUEUE_ENV`].
    queue: bool,
    /// When the last hand-off answer collected was sent: the thread was done
    /// with that frame then (see [`Self::pace`]).
    last_answered: Option<Instant>,
    dead: bool,
    join: Option<std::thread::JoinHandle<()>>,
    /// Step 1b pacing (see [`Self::pace`]): when the last `Present` was sent, and running means
    /// of the thread's busy time per command and of the main thread's lead (iteration start to
    /// [`Self::take_frame`]), in seconds.
    handed_at: Option<Instant>,
    busy_mean: Option<f64>,
    lead_mean: Option<f64>,
    /// Feedback on top of the two means, in seconds: grows while frames still wait for their
    /// image after reading input, shrinks while the thread sits idle waiting for the frame.
    pace_correction: f64,
    pacing: bool,
}

impl PresentThread {
    fn spawn(
        surface: Arc<wgpu::Surface<'static>>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipe: Arc<crate::frame_pipe::FramePipe>,
        window: Arc<Window>,
    ) -> std::io::Result<Self> {
        let (cmd, commands) = std::sync::mpsc::channel::<PresentCmd>();
        // Room for the answers of two hand-offs in flight.
        let (answers, ready) = std::sync::mpsc::sync_channel::<Answer>(2);
        let join = std::thread::Builder::new()
            .name("mc-render present".into())
            .spawn(move || {
                // However this thread ends (an `Exit`, a closed channel, or a panic in a frame's
                // finish, submit or present), the main thread must stop waiting for submits that
                // will never come.
                struct AbandonOnExit(Arc<crate::frame_pipe::FramePipe>);
                impl Drop for AbandonOnExit {
                    fn drop(&mut self) {
                        self.0.abandon();
                    }
                }
                let _abandon = AbandonOnExit(Arc::clone(&pipe));
                let acquire = || match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(tex)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => Acquired::Frame(tex),
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        Acquired::Reconfigure
                    }
                    _ => Acquired::Unavailable,
                };
                // The image kept after a hand-off, and when this thread last
                // answered (a hand-off's queue error is measured from it).
                let mut held: Option<wgpu::SurfaceTexture> = None;
                let mut free_since: Option<Instant> = None;
                while let Ok(command) = commands.recv() {
                    let began = Instant::now();
                    let mut presented = None;
                    let mut queue_error = None;
                    let mut skipped = false;
                    let next = match command {
                        // An image kept after a hand-off is the next image.
                        PresentCmd::Acquire => match held.take() {
                            Some(texture) => Acquired::Frame(texture),
                            None => {
                                mc_core::profile_scope!("present_thread.acquire");
                                acquire()
                            }
                        },
                        PresentCmd::Present {
                            texture,
                            frame,
                            started,
                        } => {
                            if let Some(frame) = frame {
                                mc_core::profile_scope!("present_thread.submit");
                                pipe.submit_handed(&queue, frame);
                            }
                            {
                                mc_core::profile_scope!("present_thread.present");
                                window.pre_present_notify();
                                queue.present(texture);
                                presented = Some(started.elapsed());
                            }
                            mc_core::profile_scope!("present_thread.acquire");
                            acquire()
                        }
                        PresentCmd::Handoff {
                            frame,
                            recorded,
                            image,
                            started,
                            handed_at,
                        } => {
                            queue_error = free_since.map(|free| seconds_between(free, handed_at));
                            let image = match image.or_else(|| held.take()) {
                                Some(texture) => Acquired::Frame(texture),
                                None => {
                                    mc_core::profile_scope!("present_thread.acquire");
                                    acquire()
                                }
                            };
                            match image {
                                Acquired::Frame(texture)
                                    if recorded_copy_fits(&recorded, &texture.texture) =>
                                {
                                    {
                                        mc_core::profile_scope!("present_thread.submit");
                                        let image = &texture.texture;
                                        match frame {
                                            HandedFrame::Pending(frame) => {
                                                pipe.submit_handed_with(&queue, frame, |enc| {
                                                    copy_recorded_frame(enc, &recorded, image);
                                                });
                                            }
                                            // A readback on the main thread submitted the frame:
                                            // the copy goes on its own, still counted.
                                            HandedFrame::Submitted => {
                                                let mut enc = device.create_command_encoder(
                                                    &wgpu::CommandEncoderDescriptor {
                                                        label: Some("mc-render frame copy"),
                                                    },
                                                );
                                                copy_recorded_frame(&mut enc, &recorded, image);
                                                pipe.submit_bare_handoff(
                                                    &queue,
                                                    Some(enc.finish()),
                                                );
                                            }
                                        }
                                    }
                                    {
                                        mc_core::profile_scope!("present_thread.present");
                                        window.pre_present_notify();
                                        queue.present(texture);
                                        presented = Some(started.elapsed());
                                    }
                                    let next = {
                                        mc_core::profile_scope!("present_thread.acquire");
                                        acquire()
                                    };
                                    match next {
                                        Acquired::Frame(texture) => {
                                            held = Some(texture);
                                            Acquired::Held
                                        }
                                        failure => failure,
                                    }
                                }
                                // No image to show the frame in, or one that no longer fits it (a
                                // reconfiguration in between, dropped unpresented): the frame's
                                // work still goes to the GPU, without the copy, and the main
                                // thread takes the image path next.
                                failure => {
                                    match frame {
                                        HandedFrame::Pending(frame) => {
                                            pipe.submit_handed(&queue, frame);
                                        }
                                        HandedFrame::Submitted => {
                                            pipe.submit_bare_handoff(&queue, None);
                                        }
                                    }
                                    skipped = true;
                                    match failure {
                                        Acquired::Reconfigure => Acquired::Reconfigure,
                                        _ => Acquired::Unavailable,
                                    }
                                }
                            }
                        }
                        PresentCmd::Release => {
                            held = None;
                            Acquired::Unavailable
                        }
                        PresentCmd::Exit => break,
                    };
                    let answered = Instant::now();
                    free_since = Some(answered);
                    let answer = Answer {
                        next,
                        presented,
                        busy: answered - began,
                        answered,
                        queue_error,
                        skipped,
                    };
                    if answers.send(answer).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            cmd,
            ready,
            outstanding: false,
            in_flight: 0,
            holds_image: false,
            lost_image: false,
            image_ok: false,
            skipped: 0,
            queue: handoff_queue_requested(),
            last_answered: None,
            dead: false,
            join: Some(join),
            handed_at: None,
            busy_mean: None,
            lead_mean: None,
            pace_correction: 0.0,
            pacing: pipelined_pace_requested(),
        })
    }

    /// Margin by which [`Self::pace`] starts an iteration before the thread is expected to be
    /// done: a little waiting for the image is cheaper than an idle present thread.
    const PACE_MARGIN: Duration = Duration::from_micros(25);
    /// Longest pacing wait. A longer expected wait (vsync, where the thread's busy time is mostly
    /// the wait for the next refresh) is not paced at all: spinning through it would burn a core,
    /// and the image wait it would move is not a CPU-side cost of this path.
    const PACE_LIMIT: Duration = Duration::from_micros(500);
    /// The wait for the image (after input was read) that the feedback steers towards.
    const PACE_TARGET_WAIT: Duration = Duration::from_micros(5);

    /// Step 1b: hold the end of this iteration until the present thread is about to be done with
    /// the frame just handed to it. Input reaches this loop only through window events, which
    /// winit delivers between iterations, so without this the next iteration starts at once,
    /// reads the input pumped so far, and then waits in [`Self::take_frame`] for its image: that
    /// wait sits between reading input and presenting. With it, the same wait comes before the
    /// event loop pumps input, and the image is ready (within [`Self::PACE_MARGIN`]) when the
    /// next frame needs it. Throughput is the same either way.
    ///
    /// The thread is expected to finish `busy` after the hand-off, and the next iteration needs
    /// `lead` from its start to its `take_frame`; both are running means, plus a feedback term.
    /// A busy-wait: the delays are tens to hundreds of microseconds, far below the OS sleep
    /// granularity, and longer ones ([`Self::PACE_LIMIT`]) are not paced.
    fn pace(&mut self) {
        // A hand-off in flight counts like an outstanding present.
        if !self.pacing || (!self.outstanding && self.in_flight == 0) {
            return;
        }
        let (Some(handed_at), Some(busy), Some(lead)) =
            (self.handed_at, self.busy_mean, self.lead_mean)
        else {
            return;
        };
        let delay = Duration::from_secs_f64((busy - lead + self.pace_correction).max(0.0));
        let Some(delay) = delay.checked_sub(Self::PACE_MARGIN) else {
            return;
        };
        if delay > Self::PACE_LIMIT {
            return;
        }
        // A handed-off frame that waited behind the one before starts when
        // the thread finished that one, not when it was handed off.
        let base = match self.last_answered {
            Some(answered) if self.in_flight > 0 => handed_at.max(answered),
            _ => handed_at,
        };
        let start = base + delay;
        if Instant::now() >= start {
            return;
        }
        mc_core::profile_scope!("shell.pace");
        while Instant::now() < start {
            std::hint::spin_loop();
        }
    }

    /// Fold one sample into a running mean (exponential, 1/16 weight).
    fn fold(mean: &mut Option<f64>, sample: Duration) {
        let sample = sample.as_secs_f64();
        *mean = Some(match *mean {
            Some(mean) => mean + (sample - mean) / 16.0,
            None => sample,
        });
    }

    /// How long the main thread waits for the thread's answer before skipping the frame and
    /// returning to the event loop. A present can need the window thread to pump a message
    /// (DXGI and some Vulkan drivers send window messages); blocking here forever would
    /// deadlock, while returning lets winit pump it. Normal waits are at most one refresh.
    const ANSWER_TIMEOUT: Duration = Duration::from_millis(250);

    /// The image to record this frame into. Blocks only while the thread is still acquiring it —
    /// the same wait `Gpu::acquire` did at this point of the frame.
    fn take_frame(
        &mut self,
        gpu: &mut Gpu,
        input_to_present: &mut InputToPresent,
        lead: Duration,
    ) -> Result<AcquiredFrame, RenderError> {
        Self::fold(&mut self.lead_mean, lead);
        // Answers come in command order: collect the hand-offs first.
        if !self.collect_in_flight(gpu, input_to_present, 0) {
            return Err(RenderError::FrameUnavailable);
        }
        if !self.outstanding {
            if self.cmd.send(PresentCmd::Acquire).is_err() {
                self.dead = true;
                return Err(RenderError::FrameUnavailable);
            }
            self.outstanding = true;
        }
        let asked = Instant::now();
        let answer = match self.ready.recv_timeout(Self::ANSWER_TIMEOUT) {
            Ok(answer) => answer,
            // Still outstanding: skip this frame and collect the answer next time.
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(RenderError::FrameUnavailable);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                self.outstanding = false;
                self.dead = true;
                return Err(RenderError::FrameUnavailable);
            }
        };
        self.outstanding = false;
        // Whatever the thread kept went into this answer.
        self.holds_image = false;
        let Answer {
            next: answer,
            presented,
            busy,
            answered,
            ..
        } = answer;
        if let Some(latency) = presented {
            input_to_present.record(latency);
            Self::fold(&mut self.busy_mean, busy);
            if self.pacing {
                // Positive: this frame read input, then waited for its image (start later).
                // Negative: the image waited for this frame (start earlier).
                let error = if answered > asked {
                    (answered - asked).as_secs_f64()
                } else {
                    -(asked - answered).as_secs_f64()
                };
                // One long frame (a tick, a hitch) must not swing the next hundred: each frame's
                // say is capped.
                let step = (error - Self::PACE_TARGET_WAIT.as_secs_f64()).clamp(-50e-6, 50e-6);
                let limit = Self::PACE_LIMIT.as_secs_f64();
                self.pace_correction = (self.pace_correction + step / 8.0).clamp(-limit, limit);
            }
        }
        // The next frame may be handed off once this one brought an image.
        self.image_ok = matches!(answer, Acquired::Frame(_));
        match answer {
            Acquired::Frame(tex) => {
                gpu.note_acquired();
                Ok(AcquiredFrame::from_surface_texture(tex))
            }
            Acquired::Reconfigure => {
                gpu.reconfigure_after_failed_acquire();
                Err(RenderError::FrameUnavailable)
            }
            // `Held` answers only hand-offs, never an `Acquire` or a `Present`.
            Acquired::Unavailable | Acquired::Held => Err(RenderError::FrameUnavailable),
        }
    }

    /// Whether the next frame may be handed off before its image: the thread
    /// is alive and the last answer brought or kept an image, or an answer that may bring one is
    /// still on its way.
    fn can_hand_off(&self) -> bool {
        !self.dead && (self.image_ok || self.outstanding)
    }

    /// Hand-offs answered as skipped since the last call, for the frame trace.
    fn take_skipped(&mut self) -> u32 {
        std::mem::take(&mut self.skipped)
    }

    /// At the end of an iteration, before the next one reads input: wait until
    /// at most one handed-off frame is ahead of it, the one being presented. Two are in flight
    /// after a hand-off that queued behind the frame before; when the present thread is slow on
    /// that frame (a long present or acquire), starting the next iteration at once would make its
    /// frame wait behind both, a full frame of the thread's work after reading input. Waiting
    /// here, before input is read (as the pacing does), costs no input-to-present time; with one
    /// frame or none in flight it returns at once.
    fn settle_handoffs(&mut self, gpu: &mut Gpu, input_to_present: &mut InputToPresent) {
        if self.in_flight >= 2 {
            mc_core::profile_scope!("shell.settle_handoffs");
            // On a timeout the next hand-off waits as before.
            let _ = self.collect_in_flight(gpu, input_to_present, 1);
        }
    }

    /// Fold one hand-off's (or release's) answer in: its latency sample, the
    /// pacing feedback, and what the thread now keeps.
    fn absorb(&mut self, answer: Answer, gpu: &mut Gpu, input_to_present: &mut InputToPresent) {
        self.in_flight = self.in_flight.saturating_sub(1);
        if let Some(latency) = answer.presented {
            input_to_present.record(latency);
            Self::fold(&mut self.busy_mean, answer.busy);
        }
        if let (true, Some(error)) = (self.pacing, answer.queue_error) {
            // As in `take_frame`: positive, the frame waited behind the one before (start later);
            // negative, the thread waited for the frame (start earlier). Capped per frame.
            let step = (error - Self::PACE_TARGET_WAIT.as_secs_f64()).clamp(-50e-6, 50e-6);
            let limit = Self::PACE_LIMIT.as_secs_f64();
            self.pace_correction = (self.pace_correction + step / 8.0).clamp(-limit, limit);
        }
        if answer.skipped {
            self.skipped = self.skipped.saturating_add(1);
        }
        self.last_answered = Some(answer.answered);
        self.holds_image = matches!(answer.next, Acquired::Held);
        if self.holds_image {
            gpu.note_acquired();
        }
        // An unavailable or outdated image, or a skipped frame: the next frame takes the image
        // path, whose acquire skips or reconfigures as it always did.
        self.image_ok = self.holds_image && !answer.skipped;
    }

    /// Collect answers of hand-offs (and releases) until at most `keep` are in
    /// flight: those already there without waiting, the rest waiting up to
    /// [`Self::ANSWER_TIMEOUT`] each. `false` when one did not come in time (the caller skips the
    /// frame and the event loop pumps messages) or the thread is gone.
    fn collect_in_flight(
        &mut self,
        gpu: &mut Gpu,
        input_to_present: &mut InputToPresent,
        keep: u8,
    ) -> bool {
        while self.in_flight > 0 {
            match self.ready.try_recv() {
                Ok(answer) => self.absorb(answer, gpu, input_to_present),
                Err(_) => break,
            }
        }
        while self.in_flight > keep {
            match self.ready.recv_timeout(Self::ANSWER_TIMEOUT) {
                Ok(answer) => self.absorb(answer, gpu, input_to_present),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => return false,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    self.in_flight = 0;
                    self.dead = true;
                    return false;
                }
            }
        }
        true
    }

    /// Hand the frame left pending on `gpu` (recorded into `recorded`) to the
    /// thread, which copies it into an image and presents it. Returns `false`, with the frame
    /// still pending, when it could not be handed off: the thread did not answer in time, is
    /// gone, or the image it was to bring is unavailable. The caller then submits the frame
    /// through the gated queue (which keeps the order) and skips the image, as for an
    /// unavailable one.
    ///
    /// A frame may wait behind the one being presented only while the thread's work on a frame
    /// is within [`Self::PACE_LIMIT`]; otherwise (vsync, a GPU-bound frame) the main thread waits
    /// here for the frame before to be done, after recording, as it waited for its image.
    fn hand_off(
        &mut self,
        gpu: &mut Gpu,
        recorded: &wgpu::Texture,
        started: Instant,
        input_to_present: &mut InputToPresent,
    ) -> bool {
        Self::fold(&mut self.lead_mean, started.elapsed());
        // The frame before went the image way: its answer brings this frame's image.
        let mut image = None;
        if self.outstanding {
            let asked = Instant::now();
            let answer = match self.ready.recv_timeout(Self::ANSWER_TIMEOUT) {
                Ok(answer) => answer,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => return false,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    self.outstanding = false;
                    self.dead = true;
                    return false;
                }
            };
            self.outstanding = false;
            if let Some(latency) = answer.presented {
                input_to_present.record(latency);
                Self::fold(&mut self.busy_mean, answer.busy);
                if self.pacing {
                    let error = seconds_between(answer.answered, asked);
                    let step = (error - Self::PACE_TARGET_WAIT.as_secs_f64()).clamp(-50e-6, 50e-6);
                    let limit = Self::PACE_LIMIT.as_secs_f64();
                    self.pace_correction = (self.pace_correction + step / 8.0).clamp(-limit, limit);
                }
            }
            match answer.next {
                Acquired::Frame(texture) => {
                    gpu.note_acquired();
                    self.image_ok = true;
                    image = Some(texture);
                }
                Acquired::Reconfigure => {
                    // The thread is idle and holds nothing, as `take_frame` requires.
                    self.image_ok = false;
                    gpu.reconfigure_after_failed_acquire();
                    return false;
                }
                Acquired::Unavailable | Acquired::Held => {
                    self.image_ok = false;
                    return false;
                }
            }
        }
        let room = if self.queue
            && self
                .busy_mean
                .is_some_and(|busy| busy <= Self::PACE_LIMIT.as_secs_f64())
        {
            1
        } else {
            0
        };
        if !self.collect_in_flight(gpu, input_to_present, room) {
            // An image brought above is dropped unpresented, which wgpu allows.
            return false;
        }
        let pipe = gpu.queue.pipe();
        let frame = match pipe.take_for_handoff() {
            Some(frame) => HandedFrame::Pending(frame),
            None => {
                pipe.count_bare_handoff();
                HandedFrame::Submitted
            }
        };
        let handed_at = Instant::now();
        let command = PresentCmd::Handoff {
            frame,
            recorded: recorded.clone(),
            image,
            started,
            handed_at,
        };
        match self.cmd.send(command) {
            Ok(()) => {
                self.in_flight += 1;
                self.handed_at = Some(handed_at);
                true
            }
            Err(std::sync::mpsc::SendError(command)) => {
                // The thread is gone: submit what it would have, so the handed-off count stays
                // honest, and report the frame as not handed (nothing is left to present it).
                if let PresentCmd::Handoff { frame, .. } = command {
                    match frame {
                        HandedFrame::Pending(frame) => {
                            pipe.submit_handed(gpu.queue.ungated(), frame);
                        }
                        HandedFrame::Submitted => {
                            pipe.submit_bare_handoff(gpu.queue.ungated(), None);
                        }
                    }
                }
                self.dead = true;
                false
            }
        }
    }

    /// Present `frame`, whose work the main thread has already submitted, and start acquiring
    /// the next image.
    fn present(
        &mut self,
        frame: AcquiredFrame,
        pending: Option<crate::frame_pipe::PendingFrame>,
        started: Instant,
        gpu: &Gpu,
    ) {
        let Some(texture) = frame.into_surface_texture() else {
            // Headless: nothing to present, but a pending frame still has to be submitted.
            if let Some(pending) = pending {
                gpu.frame_pipe().submit_handed(gpu.queue.ungated(), pending);
            }
            return;
        };
        let command = PresentCmd::Present {
            texture,
            frame: pending,
            started,
        };
        match self.cmd.send(command) {
            Ok(()) => {
                self.outstanding = true;
                self.handed_at = Some(Instant::now());
            }
            Err(std::sync::mpsc::SendError(command)) => {
                // The thread is gone: submit (and drop the image of) what it would have taken, so
                // the handed-off count stays honest and nothing is lost.
                if let PresentCmd::Present {
                    frame: Some(pending),
                    ..
                } = command
                {
                    gpu.frame_pipe().submit_handed(gpu.queue.ungated(), pending);
                }
                self.dead = true;
            }
        }
    }

    /// Wait for the outstanding answer and drop any image in it, so the thread holds no surface
    /// texture. Required before every `Surface::configure`. Returns `false` if the answer did
    /// not arrive within [`Self::ANSWER_TIMEOUT`]; the caller must then not configure this frame
    /// (it retries on the next one, after the event loop has pumped messages).
    ///
    /// A dropped image is lost to a Vulkan swap chain until its next configure,
    /// so this is called only right before one ([`take_back_image`]), and every drop sets
    /// [`Self::lost_image`].
    fn quiesce(&mut self) -> bool {
        // Collect the hand-offs in flight (their latency samples go with them,
        // as the image answer's does below), then have the thread drop the image it keeps. The
        // next frame takes the image path.
        loop {
            while self.in_flight > 0 {
                match self.ready.recv_timeout(Self::ANSWER_TIMEOUT) {
                    Ok(answer) => {
                        self.in_flight -= 1;
                        self.image_ok = false;
                        self.holds_image = matches!(answer.next, Acquired::Held);
                        self.skipped = self.skipped.saturating_add(u32::from(answer.skipped));
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => return false,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        self.in_flight = 0;
                        self.holds_image = false;
                        self.dead = true;
                        return true;
                    }
                }
            }
            if !self.holds_image {
                break;
            }
            if self.cmd.send(PresentCmd::Release).is_err() {
                self.holds_image = false;
                self.dead = true;
                return true;
            }
            // Answered with `Unavailable`: the loop above then sees nothing kept.
            self.in_flight = 1;
            // The thread drops the image it keeps.
            self.lost_image = true;
        }
        if !self.outstanding {
            return true;
        }
        match self.ready.recv_timeout(Self::ANSWER_TIMEOUT) {
            // Dropping an unpresented image discards it, which wgpu allows. The answer's latency
            // sample goes with it: one sample lost around a reconfigure.
            Ok(answer) => {
                self.outstanding = false;
                // The image in the answer is dropped here.
                if matches!(answer.next, Acquired::Frame(_)) {
                    self.lost_image = true;
                }
                true
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => false,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                self.outstanding = false;
                self.dead = true;
                true
            }
        }
    }

    /// The thread exited or panicked; the shell falls back to presenting inline.
    fn is_dead(&self) -> bool {
        self.dead
    }
}

impl Drop for PresentThread {
    fn drop(&mut self) {
        // Shutdown. A thread stuck in a present that waits on this (no longer pumping) window
        // thread is left detached rather than joined forever; the process is exiting anyway.
        // Every answer still owed: an image answer and the hand-offs.
        let owed =
            u8::from(std::mem::take(&mut self.outstanding)) + std::mem::take(&mut self.in_flight);
        for _ in 0..owed {
            if self.ready.recv_timeout(Duration::from_secs(1)).is_err() {
                return;
            }
        }
        let _ = self.cmd.send(PresentCmd::Exit);
        if let Some(join) = self.join.take() {
            // The same for a present that is still running when the
            // exit comes (a driver stuck in it): the settings are saved by now, so wait a moment,
            // then leave the thread to the process exit. It holds its own references to what it
            // uses, so nothing it touches is freed under it.
            if !join_within(join, PRESENT_THREAD_EXIT_WAIT) {
                tracing::warn!("the present thread did not stop in time; exiting without it");
            }
        }
    }
}

/// How long an exiting shell waits for its present thread to stop.
const PRESENT_THREAD_EXIT_WAIT: Duration = Duration::from_secs(2);

/// Join `join` if its thread ends within `wait`; otherwise leave it
/// running (detached) and return `false`.
fn join_within(join: std::thread::JoinHandle<()>, wait: Duration) -> bool {
    let deadline = Instant::now() + wait;
    while !join.is_finished() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let _ = join.join();
    true
}

impl<A: ShellApp> ApplicationHandler for Runner<'_, A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        if let Err(e) = self.start(event_loop) {
            tracing::error!(error = %e, "shell startup failed");
            self.failure = Some(e);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.background_window && is_user_input(&event) {
            return;
        }
        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => event_loop.exit(),

            WindowEvent::Resized(size) => {
                // PRESENT-9: keep only the newest size; `draw` applies it once, first thing.
                // Every `Gpu::resize` drains the GPU and rebuilds the swap chain, depth and MSAA
                // targets, so applying each event of a drag or fullscreen burst was a stall each.
                if self.state.is_some() {
                    self.pending_resize = Some(size);
                }
                // Not deferred: the release re-centre must use the window's real size, the same
                // `inner_size` `apply_cursor` warps within.
                self.input.set_window_size(size.width, size.height);
                // A minimized window reports a zero size. Being minimized or
                // restored ends a background wait: the next frame starts now, at its new cap.
                let minimized = size.width == 0 || size.height == 0;
                if self.background.set_minimized(minimized) {
                    tracing::debug!(minimized, "background frame woken");
                }
                // A fullscreen switch resizes; so does a maximise elsewhere.
                self.monitor_dirty = true;
            }

            // The window may now be on another monitor.
            WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.monitor_dirty = true;
            }

            WindowEvent::Focused(focused) => {
                self.focused = focused;
                // The focus changed: a frame waiting in the background starts
                // now (the first frame after Alt-Tab never waits for the background cap).
                if self.background.wake() {
                    tracing::debug!(focused, "background frame woken");
                }
                self.input.on_focus(focused);
                if !focused {
                    // The pointer goes back to the desktop now; the next
                    // focused frame grabs again if the game still wants it (`apply_cursor`).
                    if let Some(state) = self.state.as_mut() {
                        state.regrab_centre |= release_grab_on_focus_loss(
                            &state.window,
                            &mut state.applied_cursor,
                            &mut state.applied_os_pointer,
                        );
                    }
                    self.settings.keys.unpress_all();
                }
                // Focus still updates shell-owned cursor/input state, but deterministic diagnostic
                // mode must not feed OS focus transitions into player/session decisions.
                if self.app.diagnostic_clock().is_none() {
                    self.app.on_focus_changed(focused);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                // The app sees the raw event first — before it is folded into `InputState`'s
                // polled held-set below — so `ShellApp::on_key_event` observes every press and
                // release in the exact order winit delivered them.
                if self.focused && self.app.diagnostic_clock().is_none() {
                    let sequence = self.next_input_sequence;
                    self.next_input_sequence = self.next_input_sequence.wrapping_add(1);
                    self.app.on_key_event(&event, sequence);
                    if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                        let pressed = event.state.is_pressed();
                        if let Some(input) = self.input.on_key(code, pressed, event.repeat) {
                            self.settings.keys.set_state_when_input_allowed(
                                input,
                                pressed,
                                self.app.game_input_allowed(),
                            );
                        }
                    }
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                if self.focused && self.app.diagnostic_clock().is_none() {
                    let sequence = self.next_input_sequence;
                    self.next_input_sequence = self.next_input_sequence.wrapping_add(1);
                    let pressed = state.is_pressed();
                    let input = self
                        .input
                        .on_mouse_button_with_sequence(button, pressed, sequence);
                    self.settings.keys.set_state_when_input_allowed(
                        input,
                        pressed,
                        self.app.game_input_allowed(),
                    );
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if self.focused && self.app.diagnostic_clock().is_none() {
                    let sequence = self.next_input_sequence;
                    self.next_input_sequence = self.next_input_sequence.wrapping_add(1);
                    self.input.on_scroll_with_sequence(delta, sequence);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.app.diagnostic_clock().is_none() {
                    self.input.on_cursor_moved(position.x, position.y);
                }
            }

            WindowEvent::RedrawRequested => {
                self.draw();
                if self.failure.is_some() || self.should_stop() {
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if self.background_window {
            return;
        }
        if self.focused && self.app.diagnostic_clock().is_none() {
            if let DeviceEvent::MouseMotion { delta } = event {
                self.input.on_raw_motion(delta.0, delta.1);
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // The watchdog also lives here so a run bounded by max_duration still exits when the
        // window is minimised and no redraws are being delivered.
        if self.should_stop() {
            event_loop.exit();
            return;
        }
        // A window in the background waits for its next frame here, not in a
        // sleep, so a window event (the focus coming back, a restore) starts that frame at once.
        // One test while no background frame waits.
        if self.background.active() {
            match self.background.step(Instant::now()) {
                LoopStep::WaitUntil(due) => {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(due));
                    return;
                }
                LoopStep::Resume => {
                    // At most the background cap's rate, and only with this module's debug on.
                    tracing::debug!("background frame");
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
                LoopStep::Run => {}
            }
        }
        // While frames are being presented, draw the next one here, after
        // every pending window event has been delivered, exactly where its `RedrawRequested`
        // would have arrived (see `W18_OB_DIRECT_DRAW_ENV`). Otherwise (the first frame, a
        // minimised or occluded window, a skipped frame) ask for a redraw as before.
        if self.drew_frame
            && direct_draw_requested()
            && !crate::w18_ob_ab::off(crate::w18_ob_ab::DIRECT_DRAW)
        {
            self.draw();
            if self.failure.is_some() || self.should_stop() {
                event_loop.exit();
            }
            return;
        }
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.app.shutdown(&self.settings);
        // Join the present thread (it shares the surface and queue) before the context drops.
        self.present_thread = None;
        self.state = None;
    }
}

/// A minimal free-fly camera driver for development and smoke tests.
///
/// **Not the player controller.** Real movement — collision, friction, step-up, prediction — is
/// S13's `mc-physics`, and this deliberately implements none of it. It exists so the shell and
/// S11's mesher can be flown around before a server connection exists.
#[derive(Debug, Clone)]
pub struct DebugFlyCamera {
    /// Blocks per second at normal speed.
    pub speed: f64,
    /// Multiplier while sprint is held.
    pub sprint_multiplier: f64,
}

impl Default for DebugFlyCamera {
    fn default() -> Self {
        Self {
            speed: 10.0,
            sprint_multiplier: 4.0,
        }
    }
}

impl DebugFlyCamera {
    /// Apply one frame of look and movement. Call from [`ShellApp::frame`].
    pub fn update(&self, ctx: &mut FrameCtx<'_>, dt: f32) {
        let (yaw, pitch) = ctx
            .input
            .look_delta_raw(ctx.settings.mouse_sensitivity, ctx.settings.invert_mouse);
        ctx.camera.turn_mouse(yaw, pitch);

        let keys = &ctx.settings.keys;
        let mut forward = 0.0;
        let mut strafe = 0.0;
        let mut lift = 0.0;
        if keys.is_down(crate::input::keys::FORWARD) {
            forward += 1.0;
        }
        if keys.is_down(crate::input::keys::BACK) {
            forward -= 1.0;
        }
        if keys.is_down(crate::input::keys::RIGHT) {
            strafe += 1.0;
        }
        if keys.is_down(crate::input::keys::LEFT) {
            strafe -= 1.0;
        }
        if keys.is_down(crate::input::keys::JUMP) {
            lift += 1.0;
        }
        if keys.is_down(crate::input::keys::SNEAK) {
            lift -= 1.0;
        }

        if forward == 0.0 && strafe == 0.0 && lift == 0.0 {
            return;
        }
        let speed = if keys.is_down(crate::input::keys::SPRINT) {
            self.speed * self.sprint_multiplier
        } else {
            self.speed
        } * f64::from(dt);

        let yaw_rad = f64::from(ctx.camera.yaw).to_radians();
        let (sin, cos) = yaw_rad.sin_cos();
        // java: Entity.moveFlying — forward is (-sin, cos), strafe is the perpendicular.
        let dx = -sin * forward + cos * strafe;
        let dz = cos * forward + sin * strafe;
        ctx.camera.pos += glam::DVec3::new(dx, lift, dz) * speed;
    }
}

/// Whether a window event comes from the person at the desktop rather than from the window itself.
fn is_user_input(event: &WindowEvent) -> bool {
    matches!(
        event,
        WindowEvent::Focused(_)
            | WindowEvent::KeyboardInput { .. }
            | WindowEvent::ModifiersChanged(_)
            | WindowEvent::Ime(_)
            | WindowEvent::MouseInput { .. }
            | WindowEvent::MouseWheel { .. }
            | WindowEvent::CursorMoved { .. }
            | WindowEvent::CursorEntered { .. }
            | WindowEvent::CursorLeft { .. }
            | WindowEvent::DroppedFile(_)
            | WindowEvent::HoveredFile(_)
            | WindowEvent::HoveredFileCancelled
            | WindowEvent::Touch(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `ShellApp` that overrides nothing — the shape every existing implementor has for the
    /// methods added over time.
    struct Bare;
    impl ShellApp for Bare {}

    /// A present thread that does not stop (a driver stuck in a
    /// present) no longer holds the exit: the join gives up after the wait, where `join()`
    /// waited for as long as the thread ran (the control: 3 s here, forever for a stuck one).
    #[test]
    fn an_exiting_shell_waits_a_bounded_time_for_its_present_thread() {
        let quick = std::thread::spawn(|| std::thread::sleep(Duration::from_millis(20)));
        assert!(
            join_within(quick, Duration::from_secs(2)),
            "a thread that stops is joined"
        );

        let stuck = || std::thread::spawn(|| std::thread::sleep(Duration::from_secs(3)));
        let started = Instant::now();
        assert!(!join_within(stuck(), Duration::from_millis(200)));
        let bounded = started.elapsed();
        assert!(
            bounded < Duration::from_secs(1),
            "gave up after {bounded:?}"
        );

        let started = Instant::now();
        let _ = stuck().join();
        assert!(
            started.elapsed() >= Duration::from_secs(2),
            "the control: a plain join waits for the thread"
        );
    }

    #[test]
    fn scalar_only_apps_still_rebuild_the_camera_after_frame() {
        let mut camera = Camera::default();
        let previous = camera.view;
        camera.yaw = 45.0;
        Bare.finalize_camera(&mut camera);
        assert_ne!(camera.view, previous);
        let mut expected = camera;
        expected.update();
        assert_eq!(camera.view, expected.view);
        assert_eq!(camera.proj, expected.proj);
    }

    /// The additions to this trait are all *provided*, so an app written before any of them
    /// existed is unaffected: it never registers a phase from `frame`, it never asks the loop to
    /// stop, and it never asks for fullscreen — `on_key_event` cannot be exercised the same way
    /// (see below), but its default body is the same empty `{}` as every other provided hook.
    #[test]
    fn the_added_shell_app_methods_default_to_the_old_behaviour() {
        let app = Bare;
        assert!(
            app.keep_running(),
            "an app that does not opt in must run until the window closes or a ShellConfig \
             bound is hit, exactly as before"
        );
        assert_eq!(
            app.wants_fullscreen(),
            None,
            "an app that never asks for fullscreen must leave the window alone, exactly as \
            before this hook existed"
        );
        assert_eq!(
            app.diagnostic_clock(),
            None,
            "ordinary apps, including ordinary replay, must retain the host-clock timer path"
        );
        // A monitor report is a no-op for an app that does not listen.
        let mut listener = Bare;
        listener.monitor_refresh_changed(Some(144_000));
        listener.monitor_refresh_changed(None);
        // No background cap: an unfocused or minimized window is paced by
        // the limiter at `effective_frame_cap`, exactly as before.
        assert_eq!(app.background_frame_cap(false), None);
        assert_eq!(app.background_frame_cap(true), None);
    }

    #[test]
    fn diagnostic_clock_dispatch_uses_exact_step_and_repeatable_freeze() {
        let mut timer = Timer::new(crate::frame::TICKS_PER_SECOND);
        advance_simulation_clock(&mut timer, Some(DiagnosticClock::Step { micros: 25_000 }))
            .expect("valid diagnostic step");
        assert_eq!(timer.elapsed_ticks(), 0);
        assert_eq!(timer.render_partial_ticks(), 0.5);

        advance_simulation_clock(&mut timer, Some(DiagnosticClock::Frozen))
            .expect("freeze has no failure mode");
        advance_simulation_clock(&mut timer, Some(DiagnosticClock::Frozen))
            .expect("freeze is repeatable");
        assert_eq!(timer.elapsed_ticks(), 0);
        assert_eq!(timer.render_partial_ticks(), 0.0);

        advance_simulation_clock(&mut timer, Some(DiagnosticClock::Step { micros: 25_000 }))
            .expect("remainder resumes after freeze");
        assert_eq!(timer.elapsed_ticks(), 1);
        assert_eq!(timer.render_partial_ticks(), 0.0);
    }

    #[test]
    fn diagnostic_clock_rejection_is_mapped_to_frame_trace_failure() {
        let source = include_str!("shell.rs");
        let draw = source
            .split_once("let diagnostic_clock = self.app.diagnostic_clock();")
            .expect("draw evaluates the diagnostic clock")
            .1
            .split_once("fn apply_cursor")
            .expect("end of Runner::draw region")
            .0;
        let update = draw
            .find("advance_simulation_clock")
            .expect("draw dispatches the selected clock");
        let error = draw
            .find("RenderError::FrameTrace(format!(")
            .expect("invalid diagnostic clock becomes a frame-trace error");
        let before_ticks = draw
            .find("self.app.before_ticks")
            .expect("app polling hook");
        assert!(error < update && update < before_ticks);

        let mut timer = Timer::new(crate::frame::TICKS_PER_SECOND);
        assert!(
            advance_simulation_clock(&mut timer, Some(DiagnosticClock::Step { micros: 0 }))
                .is_err()
        );
        assert_eq!(timer.elapsed_ticks(), 0);
    }

    #[test]
    fn diagnostic_mode_keeps_shell_focus_state_but_suppresses_os_input_to_the_app() {
        let source = include_str!("shell.rs");
        let focus = source
            .split_once("WindowEvent::Focused(focused) =>")
            .expect("focus event arm")
            .1
            .split_once("WindowEvent::KeyboardInput")
            .expect("end of focus event arm")
            .0;
        assert!(focus.contains("self.input.on_focus(focused)"));
        assert!(focus.contains("self.app.diagnostic_clock().is_none()"));

        let keyboard = source
            .split_once("WindowEvent::KeyboardInput")
            .expect("keyboard event arm")
            .1
            .split_once("WindowEvent::MouseInput")
            .expect("end of keyboard event arm")
            .0;
        let mouse = source
            .split_once("WindowEvent::MouseInput")
            .expect("mouse event arm")
            .1
            .split_once("WindowEvent::MouseWheel")
            .expect("end of mouse event arm")
            .0;
        let motion = source
            .split_once("fn device_event")
            .expect("device event hook")
            .1
            .split_once("fn about_to_wait")
            .expect("end of device event hook")
            .0;
        assert!(keyboard.contains("self.app.diagnostic_clock().is_none()"));
        assert!(mouse.contains("self.app.diagnostic_clock().is_none()"));
        assert!(motion.contains("self.app.diagnostic_clock().is_none()"));
    }

    /// `Minecraft.runGameLoop:1104-1121` drains scheduled work before entering the elapsed-tick
    /// loop. This source-order test deliberately fails if the hook is removed or put back after
    /// ticks; constructing a real `Runner` here would require a window and GPU for an ordering
    /// property that is entirely local to `Runner::draw`.
    #[test]
    fn scheduled_work_is_applied_before_ticks_and_the_frame() {
        let source = include_str!("shell.rs");
        let draw = source
            .split_once("fn draw(&mut self)")
            .expect("Runner::draw")
            .1
            .split_once("fn apply_cursor")
            .expect("end of Runner::draw region")
            .0;
        let scheduled = draw
            .find("self.app.before_ticks")
            .expect("before-ticks hook");
        let tick = draw.find("self.app.tick").expect("fixed-rate tick call");
        let frame = draw.find("self.app.frame").expect("render-frame call");
        assert!(
            scheduled < tick && tick < frame,
            "{scheduled} {tick} {frame}"
        );
    }

    /// A synthetic winit focus event cannot be built without a real event loop, so keep the
    /// focus boundary as a CPU-only source-order contract. The shell must clear its own state and
    /// binding latches before notifying the app, and unfocused device events must not re-arm them.
    #[test]
    fn focus_loss_reconciles_shell_state_before_the_app_and_gates_device_edges() {
        let source = include_str!("shell.rs");
        let focused = source
            .split_once("WindowEvent::Focused(focused) =>")
            .expect("focus event arm")
            .1
            .split_once("WindowEvent::KeyboardInput")
            .expect("end of focus arm")
            .0;
        let raw = focused
            .find("self.input.on_focus")
            .expect("raw input reset");
        let bindings = focused
            .find("self.settings.keys.unpress_all")
            .expect("binding reset");
        let callback = focused
            .find("self.app.on_focus_changed")
            .expect("client focus callback");
        assert!(raw < bindings && bindings < callback);

        let keyboard = source
            .split_once("WindowEvent::KeyboardInput")
            .expect("keyboard arm")
            .1
            .split_once("WindowEvent::MouseInput")
            .expect("end of keyboard arm")
            .0;
        let mouse = source
            .split_once("WindowEvent::MouseInput")
            .expect("mouse arm")
            .1
            .split_once("WindowEvent::MouseWheel")
            .expect("end of mouse arm")
            .0;
        let motion = source
            .split_once("fn device_event")
            .expect("device-event hook")
            .1
            .split_once("fn about_to_wait")
            .expect("end of device-event hook")
            .0;
        assert!(keyboard.contains("if self.focused"));
        assert!(mouse.contains("if self.focused"));
        assert!(motion.contains("if self.focused"));
    }

    /// The frame cap is app-owned because the live client can switch between the worldless
    /// 30-FPS menu cap and its current settings after the shell's startup settings were cloned.
    /// Keep this CPU-only source contract beside the other `Runner::draw` ordering control so a
    /// cached/static cap cannot silently replace the dynamic hook.
    #[test]
    fn runner_draw_uses_the_dynamic_app_frame_cap_before_waiting() {
        let source = include_str!("shell.rs");
        let draw = source
            .split_once("fn draw(&mut self)")
            .expect("Runner::draw")
            .1
            .split_once("fn apply_cursor")
            .expect("end of Runner::draw region")
            .0;
        let dynamic_cap = draw
            .find("self.app.effective_frame_cap(&self.settings, self.focused)")
            .expect("dynamic app frame-cap hook");
        let limiter = draw.find("self.limiter").expect("frame limiter receiver");
        assert!(limiter < dynamic_cap, "{limiter} {dynamic_cap}");
    }

    /// The monitor refresh rate reaches the app before the frame whose cap
    /// reads it, and every event that can move the window to another monitor or change its mode
    /// re-arms the check. Source-order, like the frame-cap contract above: a `Runner` needs a
    /// real window.
    #[test]
    fn the_monitor_refresh_is_reported_before_the_frame_and_rechecked_after_moves() {
        let source = include_str!("shell.rs");
        let draw = source
            .split_once("fn draw(&mut self)")
            .expect("Runner::draw")
            .1
            .split_once("fn apply_cursor")
            .expect("end of Runner::draw region")
            .0;
        let report = draw
            .find("self.report_monitor_refresh();")
            .expect("monitor report in draw");
        let frame = draw.find("self.app.frame").expect("render-frame call");
        let cap = draw
            .find("self.app.effective_frame_cap(&self.settings, self.focused)")
            .expect("dynamic app frame-cap hook");
        assert!(report < frame && report < cap, "{report} {frame} {cap}");
        let events = source
            .split_once("fn window_event(")
            .expect("window events")
            .1
            .split_once("WindowEvent::Focused(focused) =>")
            .expect("before the focus arm")
            .0;
        assert!(events.contains("WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged"));
        assert_eq!(events.matches("self.monitor_dirty = true;").count(), 2);
        assert!(draw.contains("!= fullscreen_before"));
    }

    /// `winit::event::KeyEvent` cannot be constructed outside the `winit` crate — its
    /// `platform_specific` field is `pub(crate)` there, not merely private to `mc-render` — so
    /// `on_key_event` cannot be driven with a synthetic event from this test module. What *is*
    /// provable without a real window is the reconciliation logic below, and (from
    /// `mc-client`'s side) everything downstream of the event once it has been unpacked into
    /// plain fields.
    #[test]
    fn fullscreen_is_only_reconciled_when_the_request_actually_changes() {
        assert_eq!(
            fullscreen_reconciliation(None, None),
            None,
            "no request at all is a no-op"
        );
        assert_eq!(
            fullscreen_reconciliation(None, Some(true)),
            Some(true),
            "the first request must always be applied"
        );
        assert_eq!(
            fullscreen_reconciliation(Some(true), Some(true)),
            None,
            "already applied — must not call set_fullscreen again every single frame"
        );
        assert_eq!(
            fullscreen_reconciliation(Some(true), Some(false)),
            Some(false),
            "a genuine change must still go through"
        );
        assert_eq!(
            fullscreen_reconciliation(Some(true), None),
            None,
            "the app withdrawing its opinion must not force windowed mode unasked"
        );
    }

    /// Borderless Fullscreen: the default request (borderless) decides exactly as
    /// [`fullscreen_reconciliation`] always did; turning the mod off asks for exclusive, and a
    /// change of kind re-applies only while the window is fullscreen.
    #[test]
    fn fullscreen_kind_is_borderless_by_default_and_reapplied_only_when_it_changes() {
        use FullscreenKind::{Borderless, Exclusive, Windowed};
        assert!(
            !Bare.wants_exclusive_fullscreen(),
            "apps default to borderless"
        );
        for applied in [None, Some(false), Some(true)] {
            for wants in [None, Some(false), Some(true)] {
                let expected = fullscreen_reconciliation(applied, wants).map(|want| {
                    if want {
                        Borderless
                    } else {
                        Windowed
                    }
                });
                assert_eq!(
                    fullscreen_change(applied, false, wants, false),
                    expected,
                    "{applied:?} {wants:?}"
                );
            }
        }
        assert_eq!(
            fullscreen_change(Some(false), false, Some(true), true),
            Some(Exclusive)
        );
        assert_eq!(
            fullscreen_change(Some(true), false, Some(true), true),
            Some(Exclusive),
            "borderless -> exclusive while fullscreen"
        );
        assert_eq!(
            fullscreen_change(Some(true), true, Some(true), false),
            Some(Borderless),
            "and back"
        );
        assert_eq!(fullscreen_change(Some(true), true, Some(true), true), None);
        assert_eq!(
            fullscreen_change(Some(false), false, Some(false), true),
            None,
            "windowed: a kind change waits for the next F11"
        );
        assert_eq!(
            fullscreen_change(Some(true), true, Some(false), true),
            Some(Windowed)
        );
        assert_eq!(fullscreen_change(Some(true), false, None, true), None);
    }

    #[test]
    fn exclusive_fullscreen_keeps_the_monitor_size_and_its_refresh_rate() {
        let size = PhysicalSize::new(1920, 1080);
        let modes = [
            (PhysicalSize::new(1280, 720), 32, 360_000),
            (size, 32, 60_000),
            (size, 32, 360_000),
            (size, 16, 360_000),
            (size, 32, 240_000),
        ];
        assert_eq!(
            pick_video_mode(&modes, size, Some(240_000)),
            Some(4),
            "the current refresh rate wins"
        );
        assert_eq!(
            pick_video_mode(&modes, size, Some(360_000)),
            Some(2),
            "then the deepest colour"
        );
        assert_eq!(
            pick_video_mode(&modes, size, None),
            Some(2),
            "unknown current rate: the highest refresh, then depth"
        );
        assert_eq!(pick_video_mode(&modes, size, Some(144_000)), Some(2));
        assert_eq!(
            pick_video_mode(&modes, PhysicalSize::new(800, 600), None),
            None,
            "no mode at the monitor's size: the caller falls back to borderless"
        );
    }

    /// Source-order contracts for C1-10 and PRESENT-7 inside `Runner::draw`, in the style of the
    /// other draw-ordering tests here (a real `Runner` needs a window and a GPU):
    /// * the present request is applied after the app's frame and before acquisition, because
    ///   `Surface::configure` panics while a surface texture is alive;
    /// * a failed acquisition flushes the staged writes and still runs the dynamic frame cap.
    #[test]
    fn present_options_apply_before_acquire_and_failed_acquire_still_paces() {
        let source = include_str!("shell.rs");
        let draw = source
            .split_once("fn draw(&mut self)")
            .expect("Runner::draw")
            .1
            .split_once("fn apply_cursor")
            .expect("end of Runner::draw region")
            .0;
        let frame = draw.find("self.app.frame").expect("render-frame call");
        let wants = draw
            .find("self.app.wants_present()")
            .expect("present request polled per frame");
        let acquire = draw.find("state.gpu.acquire()").expect("acquire call");
        assert!(
            frame < wants && wants < acquire,
            "{frame} {wants} {acquire}"
        );

        let skipped = draw
            .split_once("Err(RenderError::FrameUnavailable) =>")
            .expect("failed-acquire arm")
            .1
            .split_once("Err(e) =>")
            .expect("end of failed-acquire arm")
            .0;
        let flush = skipped
            .find("flush_staged_writes()")
            .expect("staged writes are flushed");
        let cap = skipped
            .find("self.app.effective_frame_cap(&self.settings, self.focused)")
            .expect("a skipped frame is still paced by the dynamic cap");
        assert!(flush < cap);
        assert!(skipped.contains("self.limiter"));
    }

    /// C1-10: the in-game VSync toggle reaches the swap chain, and an unchanged request does not
    /// reconfigure (each configure drains the GPU).
    #[test]
    fn present_request_is_only_reconciled_when_it_actually_changes() {
        let startup = PresentRequest::from_settings(&GameSettings::default());
        assert_eq!(
            startup,
            PresentRequest {
                vsync: true,
                preference: PresentPreference::Auto,
                frame_latency: crate::context::DEFAULT_FRAME_LATENCY,
            },
            "vanilla defaults: vsync on, auto mode, the latency the window always had"
        );
        assert_eq!(present_reconciliation(Some(startup), None), None);
        assert_eq!(
            present_reconciliation(Some(startup), Some(startup)),
            None,
            "an unchanged request is a comparison, not a reconfigure"
        );
        let vsync_off = PresentRequest {
            vsync: false,
            ..startup
        };
        assert_eq!(
            present_reconciliation(Some(startup), Some(vsync_off)),
            Some(vsync_off),
            "the in-game VSync button must reach the swap chain without a restart"
        );
        let low_latency = PresentRequest {
            frame_latency: 1,
            ..vsync_off
        };
        assert_eq!(
            present_reconciliation(Some(vsync_off), Some(low_latency)),
            Some(low_latency)
        );
        assert_eq!(present_reconciliation(None, Some(startup)), Some(startup));
    }

    /// `MouseHelper.ungrabMouseCursor` uses integer division of the physical Display width and
    /// height before it calls `Mouse.setGrabbed(false)`.  The odd and maximal controls prevent a
    /// tempting logical-pixel or rounding-up replacement from silently changing that target.
    /// LWJGL's `y` counts from the bottom (`WindowsDisplay.setCursorPosition` warps to
    /// `rect.bottom - 1 - y`), so an even height lands one row above `height / 2`.
    #[test]
    fn ungrabbing_recentes_the_cursor_at_the_java_physical_pixel_centre() {
        assert_eq!(
            cursor_centre(PhysicalSize::new(854, 480)),
            PhysicalPosition::new(427, 239),
            "LWJGL's (427, 240) from the bottom is row 480 - 1 - 240 from the top"
        );
        assert_eq!(
            cursor_centre(PhysicalSize::new(855, 481)),
            PhysicalPosition::new(427, 240),
            "Java integer division floors an odd physical window size"
        );
        assert_eq!(
            cursor_centre(PhysicalSize::new(u32::MAX, u32::MAX)),
            PhysicalPosition::new(i32::MAX, i32::MAX),
            "the conversion remains representable for every winit physical size"
        );
    }

    /// `FrameCtx` grew a field. It is constructed in exactly one place (`Runner::draw`), so this
    /// pins that the *shell* is the only owner of that invariant and that the field really is
    /// the live list rather than a copy.
    #[test]
    fn frame_ctx_carries_the_live_phase_list() {
        let mut phases = PhaseList::new();
        phases.insert(crate::PhaseOrder::UI, Box::new(CountingPhase));
        assert_eq!(phases.len(), 1);
        // Handing it out mutably is the whole point: a caller can clear and rebuild.
        let borrowed: &mut PhaseList = &mut phases;
        borrowed.clear();
        assert!(phases.is_empty());
    }

    /// T3.1. The exact decision `Runner::draw` makes, pinned.
    ///
    /// **Control** — the first assertion is the one that fails under the old behaviour: it
    /// returned `None` whenever `enable_vsync` was set, whatever `maxFps` said. If this change
    /// did nothing, `frame_cap` would still answer `None` there and this test would fail on its
    /// first line.
    #[test]
    fn max_fps_is_honoured_with_vsync_on() {
        let capped = GameSettings {
            enable_vsync: true,
            limit_framerate: 120,
            ..GameSettings::default()
        };
        assert_eq!(
            frame_cap(&capped),
            Some(120),
            "vanilla runs Display.sync and vsync together; maxFps:120 must not be silently ignored"
        );

        // ...and the same setting with vsync off, which always worked, must be unchanged.
        assert_eq!(
            frame_cap(&GameSettings {
                enable_vsync: false,
                ..capped.clone()
            }),
            Some(120),
        );

        // maxFps:260 is vanilla's "unlimited" sentinel and stays unlimited with vsync either way.
        for vsync in [true, false] {
            assert_eq!(
                frame_cap(&GameSettings {
                    enable_vsync: vsync,
                    limit_framerate: crate::settings::FRAMERATE_UNLIMITED,
                    ..GameSettings::default()
                }),
                None,
                "maxFps:260 is unlimited (vsync = {vsync})"
            );
        }
    }

    /// The other half of T3.1: that the cap `frame_cap` returns actually paces the loop when
    /// vsync is on. `frame_cap` alone only proves the decision; this proves the limiter acts on it.
    ///
    /// Only a lower bound is asserted, for the same reason
    /// `frame_limiter_paces_to_the_requested_rate` does: an overloaded machine may be slower, but
    /// the limiter must never run *faster* than the cap. Under the old behaviour this loop
    /// returned essentially instantly, which is roughly two orders of magnitude below the bound.
    #[test]
    fn the_limiter_actually_paces_the_loop_with_vsync_on() {
        let settings = GameSettings {
            enable_vsync: true,
            limit_framerate: 120,
            ..GameSettings::default()
        };
        let mut limiter = FrameLimiter::new();
        const FRAMES: u32 = 24;
        let start = Instant::now();
        for _ in 0..FRAMES {
            limiter.wait(frame_cap(&settings));
        }
        let elapsed = start.elapsed().as_secs_f64();
        let expected = f64::from(FRAMES) / 120.0;
        assert!(
            elapsed >= expected * 0.9,
            "{FRAMES} frames capped at 120 fps with vsync on took {elapsed:.4}s, \
             expected ~{expected:.4}s — the limiter is being skipped"
        );
    }

    struct CountingPhase;
    impl crate::RenderPhase for CountingPhase {
        fn record(
            &mut self,
            _: &Gpu,
            _: &mut wgpu::CommandEncoder,
            _: &wgpu::TextureView,
            _: &Camera,
        ) {
        }
        fn name(&self) -> &'static str {
            "counting"
        }
    }

    // ---- The Windows key traps the pointer -------------

    /// The shell's cursor bookkeeping (`Live::applied_cursor`, `applied_os_pointer`,
    /// `regrab_centre`) driven through the real [`cursor_plan`] and [`focus_loss_releases`],
    /// exactly as `apply_cursor` and the `Focused` arm update it, minus the window calls.
    #[derive(Debug, Default)]
    struct PointerModel {
        applied: Option<CursorMode>,
        os_pointer: bool,
        regrab_centre: bool,
        /// Every plan that changed something, plus `"release"` for a focus-loss release.
        log: Vec<String>,
    }

    impl PointerModel {
        /// `WindowEvent::Focused(false)`: `InputState::on_focus`, then the release.
        fn focus_lost(&mut self, input: &mut InputState) {
            input.on_focus(false);
            if focus_loss_releases(self.applied, self.os_pointer) {
                self.applied = Some(CursorMode::Free);
                self.os_pointer = false;
                self.regrab_centre = true;
                self.log.push("release".into());
            }
        }

        /// One frame: the client asks for its mode (`reconcile_cursor_grab`), then `apply_cursor`.
        fn frame(&mut self, input: &mut InputState, wants: CursorMode) -> CursorPlan {
            input.request_cursor(wants);
            let requested = input.cursor_mode().expect("requested");
            let recentre = input.take_cursor_recentre();
            let plan = cursor_plan(
                requested,
                input.focused(),
                self.applied,
                recentre,
                self.regrab_centre,
            );
            match plan {
                CursorPlan::Keep => {}
                CursorPlan::Grab { .. } => {
                    self.regrab_centre = false;
                    self.applied = Some(CursorMode::Grabbed);
                }
                CursorPlan::Free { .. } => self.applied = Some(CursorMode::Free),
            }
            if plan != CursorPlan::Keep {
                self.log.push(format!("{plan:?}"));
            }
            plan
        }
    }

    /// LWJGL's `shouldGrab()`: the OS pointer follows the request only while focused.
    #[test]
    fn the_os_pointer_follows_the_request_only_while_focused() {
        use CursorMode::{Free, Grabbed};
        let centre = cfg!(windows);
        // Focused: the request, with no centring unless a focus loss released the grab.
        assert_eq!(
            cursor_plan(Grabbed, true, None, false, false),
            CursorPlan::Grab {
                centre_first: false
            }
        );
        assert_eq!(
            cursor_plan(Grabbed, true, Some(Free), false, true),
            CursorPlan::Grab {
                centre_first: centre
            }
        );
        assert_eq!(
            cursor_plan(Grabbed, true, Some(Grabbed), false, false),
            CursorPlan::Keep
        );
        assert_eq!(
            cursor_plan(Free, true, Some(Grabbed), true, false),
            CursorPlan::Free { recentre: true }
        );
        assert_eq!(
            cursor_plan(Free, true, Some(Free), false, true),
            CursorPlan::Keep
        );
        // Unfocused: free, whatever the game asks for, and never re-centred.
        for requested in [Grabbed, Free] {
            assert_eq!(
                cursor_plan(requested, false, Some(Free), true, true),
                CursorPlan::Keep,
                "{requested:?}"
            );
            assert_eq!(
                cursor_plan(requested, false, Some(Grabbed), true, false),
                CursorPlan::Free { recentre: false },
                "{requested:?}"
            );
        }
        // Only a grab (the game's, or Raw Mouse Input off's confinement) is given back.
        assert!(focus_loss_releases(Some(Grabbed), false));
        assert!(focus_loss_releases(Some(Free), true));
        assert!(!focus_loss_releases(Some(Free), false));
        assert!(!focus_loss_releases(None, false));
    }

    /// The bug report, step by step: in game, the Windows key (focus lost), the 500 ms pause
    /// menu, back to the game. The pointer is free from the focus loss on although the client
    /// keeps asking for the grab, stays free over the pause menu, and is grabbed again (centred
    /// first) only when the game is back in play. With `pauseOnLostFocus:false` refocusing grabs
    /// at once. The negative control replays the same frames through the old rule ("apply the
    /// request", no focus-loss release): the grab is kept for every unfocused frame and only the
    /// pause menu releases it, which winit cannot do for an inactive window (see
    /// `winit_keeps_an_inactive_windows_clip_and_the_focus_loss_release_frees_it`).
    #[test]
    fn the_windows_key_frees_the_pointer_and_only_the_game_grabs_it_again() {
        use CursorMode::{Free, Grabbed};
        let centre = cfg!(windows);
        let mut input = InputState::new();
        input.set_window_size(1920, 1080);
        input.on_focus(true);
        let mut model = PointerModel::default();
        // Title screen, then in game.
        model.frame(&mut input, Free);
        model.frame(&mut input, Grabbed);
        assert_eq!(model.applied, Some(Grabbed));
        model.log.clear();

        // The Windows key: the Start menu takes focus. The client still asks for the grab every
        // frame (`reconcile_cursor_grab`); the pointer stays the desktop's.
        model.focus_lost(&mut input);
        for _ in 0..30 {
            assert_eq!(model.frame(&mut input, Grabbed), CursorPlan::Keep);
        }
        assert_eq!(model.applied, Some(Free));
        // 500 ms later the pause menu opens: nothing to do, and no warp while unfocused.
        assert_eq!(model.frame(&mut input, Free), CursorPlan::Keep);
        assert_eq!(model.log, ["release"]);

        // Back to the game: the pause menu keeps the pointer free...
        input.on_focus(true);
        assert_eq!(model.frame(&mut input, Free), CursorPlan::Keep);
        // ...until Back to Game grabs it, centred first.
        assert_eq!(
            model.frame(&mut input, Grabbed),
            CursorPlan::Grab {
                centre_first: centre
            }
        );
        // An ordinary screen close later grabs where the pointer is, as before.
        model.frame(&mut input, Free);
        assert_eq!(
            model.frame(&mut input, Grabbed),
            CursorPlan::Grab {
                centre_first: false
            }
        );

        // `pauseOnLostFocus:false`: no menu, so refocusing grabs straight away.
        model.focus_lost(&mut input);
        assert_eq!(model.frame(&mut input, Grabbed), CursorPlan::Keep);
        input.on_focus(true);
        assert_eq!(
            model.frame(&mut input, Grabbed),
            CursorPlan::Grab {
                centre_first: centre
            }
        );

        // A focus loss on the title screen gives nothing back: there is no grab.
        model.frame(&mut input, Free);
        model.log.clear();
        model.focus_lost(&mut input);
        assert!(model.log.is_empty(), "{:?}", model.log);
        input.on_focus(true);

        // Raw Mouse Input off holds a confinement even with the game's grab released.
        model.os_pointer = true;
        model.focus_lost(&mut input);
        assert_eq!(model.log, ["release"]);
        assert!(!model.os_pointer);

        // Negative control: the old rule on the same report.
        let mut input = InputState::new();
        input.on_focus(true);
        let mut applied = None;
        let mut old_frame = |input: &mut InputState, wants: CursorMode| {
            input.request_cursor(wants);
            let mode = input.cursor_mode().expect("requested");
            let recentre = input.take_cursor_recentre();
            let changed = applied != Some(mode) || recentre;
            applied = Some(mode);
            changed.then_some(mode)
        };
        old_frame(&mut input, Grabbed);
        input.on_focus(false);
        let unfocused: Vec<_> = (0..30).map(|_| old_frame(&mut input, Grabbed)).collect();
        assert!(
            unfocused.iter().all(Option::is_none),
            "the old rule never releases while the game asks for the grab"
        );
        assert_eq!(
            old_frame(&mut input, Free),
            Some(Free),
            "only the pause menu released it, through winit, while the window was inactive"
        );
    }

    /// The root cause on this PC, through a real window: winit 0.30 cannot release the pointer of
    /// an inactive window on Windows (`refresh_os_cursor` changes the clip only for the active
    /// window), so a one-pixel clip like `Locked`'s survives `set_cursor_grab(None)`. The focus
    /// loss release frees it.
    ///
    /// Ignored by default: it opens a hidden window and clips the desktop pointer to one pixel
    /// for a few milliseconds (a guard always releases it). Run it on its own, with nobody at
    /// the mouse:
    /// `cargo test -p mc-render --lib winit_keeps_an_inactive -- --ignored`.
    #[cfg(windows)]
    #[test]
    #[ignore = "opens a real window and briefly clips the desktop pointer"]
    fn winit_keeps_an_inactive_windows_clip_and_the_focus_loss_release_frees_it() {
        use winit::platform::pump_events::EventLoopExtPumpEvents;
        use winit::platform::windows::EventLoopBuilderExtWindows;

        /// Opens one hidden, inactive window: never the active one, like the game behind the
        /// Start menu.
        struct Opener(Option<Window>);
        impl ApplicationHandler for Opener {
            fn resumed(&mut self, event_loop: &ActiveEventLoop) {
                if self.0.is_none() {
                    let attrs = Window::default_attributes()
                        .with_visible(false)
                        .with_active(false)
                        .with_inner_size(PhysicalSize::new(320_u32, 240_u32));
                    self.0 = Some(event_loop.create_window(attrs).expect("a window"));
                }
            }
            fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
        }
        /// Whatever happens, the pointer is free again afterwards.
        struct Unclip;
        impl Drop for Unclip {
            fn drop(&mut self) {
                os_cursor::release_clip();
            }
        }

        let free = os_cursor::clip().expect("GetClipCursor");
        if free.right - free.left < 640 || free.bottom - free.top < 480 {
            eprintln!("skip: another application holds a cursor clip ({free:?})");
            return;
        }
        let mut builder = EventLoop::builder();
        builder.with_any_thread(true);
        let mut event_loop = builder.build().expect("an event loop");
        let mut opener = Opener(None);
        for _ in 0..10 {
            let _ = event_loop.pump_app_events(Some(std::time::Duration::ZERO), &mut opener);
            if opener.0.is_some() {
                break;
            }
        }
        let window = opener.0.as_ref().expect("resumed opened the window");
        let _unclip = Unclip;
        let origin = window.inner_position().expect("the client origin");
        let pin = os_cursor::Rect {
            left: origin.x + 160,
            top: origin.y + 120,
            right: origin.x + 161,
            bottom: origin.y + 121,
        };

        // What the grab left behind while the window was active: a one-pixel clip.
        assert!(os_cursor::set_clip(pin), "ClipCursor");
        assert_eq!(os_cursor::clip(), Some(pin));
        // Negative control: the old release (winit only) from an inactive window keeps it.
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
        assert_eq!(
            os_cursor::clip(),
            Some(pin),
            "winit released the clip of an inactive window after all"
        );
        // A focus loss with no grab of ours leaves someone else's clip alone.
        let (mut applied, mut os_pointer) = (Some(CursorMode::Free), false);
        assert!(!release_grab_on_focus_loss(
            window,
            &mut applied,
            &mut os_pointer
        ));
        assert_eq!(os_cursor::clip(), Some(pin));
        // The fix: the focus-loss release with the game's grab applied frees the pointer.
        let mut applied = Some(CursorMode::Grabbed);
        assert!(release_grab_on_focus_loss(
            window,
            &mut applied,
            &mut os_pointer
        ));
        assert_eq!(applied, Some(CursorMode::Free));
        assert_eq!(
            os_cursor::clip(),
            Some(free),
            "the pointer is the desktop's again"
        );
    }

    /// A `PresentThread` whose thread answers every command with `answer` and
    /// counts the commands in `sent`: the protocol's bookkeeping without a GPU or a window.
    fn fake_present_thread(
        answer: fn(&PresentCmd) -> Acquired,
        sent: Arc<std::sync::atomic::AtomicUsize>,
    ) -> PresentThread {
        let (cmd, commands) = std::sync::mpsc::channel::<PresentCmd>();
        let (answers, ready) = std::sync::mpsc::sync_channel::<Answer>(2);
        let join = std::thread::spawn(move || {
            while let Ok(command) = commands.recv() {
                if matches!(command, PresentCmd::Exit) {
                    break;
                }
                sent.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let reply = Answer {
                    next: answer(&command),
                    presented: None,
                    busy: Duration::ZERO,
                    answered: Instant::now(),
                    queue_error: None,
                    skipped: false,
                };
                if answers.send(reply).is_err() {
                    break;
                }
            }
        });
        PresentThread {
            cmd,
            ready,
            outstanding: false,
            in_flight: 0,
            holds_image: false,
            lost_image: false,
            image_ok: false,
            skipped: 0,
            queue: false,
            last_answered: None,
            dead: false,
            join: Some(join),
            handed_at: None,
            busy_mean: None,
            lead_mean: None,
            pace_correction: 0.0,
            pacing: false,
        }
    }

    /// A configure point takes the present thread's image back only
    /// when a configure follows or an image is already lost, and every image `quiesce` drops
    /// stays marked lost until a configure gives it back. (An image answer to a present needs a
    /// window: `reconfigure_points_keep_every_swap_chain_image` covers that one.)
    #[test]
    fn the_present_thread_gives_its_image_back_only_for_a_configure() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let sent = Arc::new(AtomicUsize::new(0));
        let mut thread = fake_present_thread(|_| Acquired::Unavailable, Arc::clone(&sent));
        // After a hand-off the thread keeps the next image.
        thread.holds_image = true;
        assert!(take_back_image(Some(&mut thread), false));
        assert_eq!(
            sent.load(Ordering::SeqCst),
            0,
            "no configure: the thread keeps its image"
        );
        assert!(thread.holds_image && !thread.lost_image);

        assert!(take_back_image(Some(&mut thread), true));
        assert_eq!(
            sent.load(Ordering::SeqCst),
            1,
            "released before the configure"
        );
        assert!(!thread.holds_image, "nothing kept");
        assert!(
            thread.lost_image,
            "the released image is lost until a configure"
        );

        // An image already lost: the next configure point takes back even for no change of its
        // own (and then configures; see `settle_lost_image`).
        thread.holds_image = true;
        assert!(take_back_image(Some(&mut thread), false));
        assert_eq!(sent.load(Ordering::SeqCst), 2);

        let Ok(mut gpu) = Gpu::headless(32, 32, crate::GpuOptions::default()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        // No swap chain to configure (headless here, minimised in a window): still owed.
        settle_lost_image(&mut gpu, Some(&mut thread), false);
        assert!(thread.lost_image, "not configured: still lost");
        settle_lost_image(&mut gpu, Some(&mut thread), true);
        assert!(!thread.lost_image, "a configure gives every image back");

        // Nothing kept or outstanding, or an answer without an image: nothing is dropped.
        assert!(thread.quiesce());
        assert!(thread.cmd.send(PresentCmd::Acquire).is_ok());
        thread.outstanding = true;
        assert!(thread.quiesce());
        assert!(!thread.lost_image);
        assert_eq!(sent.load(Ordering::SeqCst), 3);
        // Serial presenting: nothing to take back.
        assert!(take_back_image(None, true));
    }

    /// In a real window, with the pipelined present thread (#14
    /// step 1a): same-size resizes, 0×0 and back, and present requests that resolve to the
    /// configured mode and latency keep every frame presenting, and a real resize still
    /// configures. The negative control is the order these took before (take the image back,
    /// then a resize that configures nothing): after a few such drops no frame is presented at
    /// all, and one configure (`settle_lost_image`) gives every image back.
    ///
    /// Ignored by default: it opens a small window (not activated) and presents to it for a few
    /// seconds; `MC_GPU_ADAPTER` picks the GPU. Run it on its own:
    /// `cargo test -p mc-render --lib reconfigure_points_keep -- --ignored`.
    #[cfg(windows)]
    #[test]
    #[ignore = "opens a real window and presents to it"]
    fn reconfigure_points_keep_every_swap_chain_image() {
        use winit::platform::pump_events::EventLoopExtPumpEvents;
        use winit::platform::windows::EventLoopBuilderExtWindows;

        struct Opener(Option<Arc<Window>>);
        impl ApplicationHandler for Opener {
            fn resumed(&mut self, event_loop: &ActiveEventLoop) {
                if self.0.is_none() {
                    let attrs = Window::default_attributes()
                        .with_active(false)
                        .with_inner_size(PhysicalSize::new(320_u32, 240_u32));
                    let window = event_loop.create_window(attrs).expect("a window");
                    self.0 = Some(Arc::new(window));
                }
            }
            fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
        }

        /// `frames` iterations of the image path (`take_frame`, a clear, `present`), pumping
        /// the window's messages before each as the event loop does: the frames presented.
        fn run(
            pump: &mut dyn FnMut(),
            gpu: &mut Gpu,
            thread: &mut PresentThread,
            latency: &mut InputToPresent,
            frames: u32,
        ) -> u32 {
            let mut presented = 0;
            for _ in 0..frames {
                pump();
                let started = Instant::now();
                if let Ok(frame) = thread.take_frame(gpu, latency, Duration::ZERO) {
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    drop(gpu.begin_clear_color_pass(&mut enc, frame.view(), wgpu::Color::BLUE));
                    let _ = gpu.submit_frame(enc.finish());
                    thread.present(frame, None, started, gpu);
                    presented += 1;
                }
            }
            presented
        }

        /// `quiesce`, retried while the thread is still in a (timing-out) acquire.
        fn quiesce(pump: &mut dyn FnMut(), thread: &mut PresentThread) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !thread.quiesce() {
                assert!(
                    Instant::now() < deadline,
                    "the present thread never answered"
                );
                pump();
            }
        }

        let mut builder = EventLoop::builder();
        builder.with_any_thread(true);
        let mut event_loop = builder.build().expect("an event loop");
        let mut opener = Opener(None);
        for _ in 0..10 {
            let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut opener);
            if opener.0.is_some() {
                break;
            }
        }
        let window = opener.0.clone().expect("resumed opened the window");
        let mut pump = || {
            let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut opener);
        };
        let options = crate::GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        };
        // An explicit mode, so a VSync change resolves to the same one.
        let mut gpu =
            Gpu::for_window_with(Arc::clone(&window), options, PresentPreference::Immediate)
                .expect("a GPU for the window");
        let surface = gpu.shared_surface().expect("a window surface");
        let mut thread = PresentThread::spawn(
            surface,
            gpu.device.clone(),
            gpu.queue.ungated().clone(),
            gpu.frame_pipe(),
            Arc::clone(&window),
        )
        .expect("the present thread");
        let mut latency = InputToPresent::new();
        let (width, height) = (gpu.config.width, gpu.config.height);
        let mode = gpu.present_mode();
        eprintln!(
            "{:?} at {width}x{height}, {mode:?}",
            gpu.adapter_info().name
        );

        let presented = run(&mut pump, &mut gpu, &mut thread, &mut latency, 40);
        assert!(presented >= 30, "the window presents: {presented} of 40");
        for i in 0..4 {
            let same = resize_swap_chain(&mut gpu, Some(&mut thread), width, height);
            assert_eq!(same, Some(false), "a repeated size rebuilds nothing");
            let presented = run(&mut pump, &mut gpu, &mut thread, &mut latency, 20);
            assert!(
                presented >= 18,
                "after same-size resize {i}: {presented} of 20"
            );
        }
        assert_eq!(
            resize_swap_chain(&mut gpu, Some(&mut thread), 0, 0),
            Some(false)
        );
        let _ = run(&mut pump, &mut gpu, &mut thread, &mut latency, 5);
        let back = resize_swap_chain(&mut gpu, Some(&mut thread), width, height);
        assert_eq!(back, Some(false));
        let presented = run(&mut pump, &mut gpu, &mut thread, &mut latency, 20);
        assert!(presented >= 18, "after 0x0 and back: {presented} of 20");
        let explicit = |vsync, frame_latency| PresentRequest {
            vsync,
            preference: PresentPreference::Immediate,
            frame_latency,
        };
        let requests = [explicit(false, 0), explicit(true, 0), explicit(true, 2)];
        for (i, want) in requests.into_iter().enumerate() {
            let applied = apply_present_request(&mut gpu, Some(&mut thread), want);
            assert_eq!(
                applied,
                Some(mode),
                "request {i} resolves to the configured mode"
            );
            let presented = run(&mut pump, &mut gpu, &mut thread, &mut latency, 20);
            assert!(
                presented >= 18,
                "after present request {i}: {presented} of 20"
            );
        }
        assert!(!thread.lost_image, "no image was dropped");

        // A real resize still configures, after taking the image back (else wgpu panics).
        let _ = window.request_inner_size(PhysicalSize::new(width + 40, height + 20));
        for _ in 0..100 {
            pump();
            if window.inner_size() != PhysicalSize::new(width, height) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let size = window.inner_size();
        assert_ne!(
            (size.width, size.height),
            (width, height),
            "the window resized"
        );
        let resized = resize_swap_chain(&mut gpu, Some(&mut thread), size.width, size.height);
        assert_eq!(resized, Some(true));
        assert!(
            !thread.lost_image,
            "the configure gave the dropped image back"
        );
        let presented = run(&mut pump, &mut gpu, &mut thread, &mut latency, 20);
        assert!(presented >= 18, "after a real resize: {presented} of 20");

        // The negative control: the orders before (take the image back, then the same size or a
        // request for the same mode, which configure nothing), taken in turn. Each drop loses an
        // image; once too few are left, none presents (2 on the RTX). An AMD iGPU's driver gives
        // such images back by itself, so there the control has nothing to show.
        let mut drops = 0;
        let mut stalled = false;
        while drops < 8 {
            if run(&mut pump, &mut gpu, &mut thread, &mut latency, 5) == 0 {
                stalled = true;
                break;
            }
            quiesce(&mut pump, &mut thread);
            if drops % 2 == 0 {
                assert!(!gpu.resize(size.width, size.height));
            } else {
                assert_eq!(explicit(drops % 4 == 1, 2).apply(&mut gpu), mode);
            }
            drops += 1;
        }
        if stalled {
            let after = run(&mut pump, &mut gpu, &mut thread, &mut latency, 4);
            eprintln!("the control: no frame after {drops} dropped images ({after} of 4 after)");
            assert_eq!(after, 0);
        } else {
            eprintln!("the control: this GPU still presents after {drops} dropped images");
        }
        assert!(drops >= 1 && thread.lost_image);
        // One configure gives them all back.
        quiesce(&mut pump, &mut thread);
        settle_lost_image(&mut gpu, Some(&mut thread), false);
        assert!(!thread.lost_image);
        let presented = run(&mut pump, &mut gpu, &mut thread, &mut latency, 20);
        assert!(presented >= 18, "after the rebuild: {presented} of 20");
        drop(thread);
        drop(gpu);
    }
}
