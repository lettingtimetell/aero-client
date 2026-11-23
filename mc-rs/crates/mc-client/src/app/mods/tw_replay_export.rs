//! Video export. A replay (or a clip) becomes an MP4 at any
//! resolution and any frame rate, with the game's sound and the microphone.
//!
//! # How a frame is exported
//!
//! An export takes over the frame loop until it ends. Every loop iteration renders one frame:
//!
//! 1. **Time.** the `replay_render_at(t)` puts the world, the ticks and the camera exactly
//!    at replay time `t` (from `before_ticks`), and the shell's own clock is frozen
//!    ([`Client::tw_export_clock`]). Rendered frame `k` is at `start + k / (fps x blur)`: a fixed
//!    timestep whatever the PC's speed, so 2000 fps is as easy as 30 (just more frames).
//! 2. **Resolution.** The shell renders the whole frame graph (shaders, mods, the HUD as the
//!    camera mode shows it) offscreen at the export's size (`Gpu::set_render_override`), not at
//!    the window's ([`Client::tw_export_offscreen_size`]).
//! 3. **Capture.** `mc_render::export::ExportCapture` blends `blur` rendered frames into one
//!    (motion blur, averaged in linear light), converts to NV12 on the GPU and reads it back
//!    through a ring of buffers, so the GPU never waits ([`Client::tw_export_after_render`]).
//! 4. **Encode.** `mc_video::EncoderThread` encodes on its own thread (Media Foundation, the
//!    GPU's H.264/H.265 encoder when it has one) and pulls the offline audio mix as it goes.
//! 5. **Window.** The window shows the latest frame scaled to fit, with a progress panel
//!    ([`Client::tw_export_draw_preview`]); Esc cancels.
//!
//! With a maximum file size the bitrate comes from `mc_video::size::plan`; a file that still
//! comes out too big is rendered again at a lower bitrate (at most `size::MAX_PASSES` passes).
//! A cancelled or failed export leaves no file (the encoder writes `<name>.partial`).
//!
//! # Without the playback
//!
//! The same pipeline runs on any session: with no replay open, the shell clock steps by exactly
//! one rendered frame per loop (`DiagnosticClock::Step`), which is what the `--replay <.mccap>`
//! benchmark captures use. That is the "legacy" clock; `MC_RD_EXPORT` (see [`Harness`]) starts
//! such an export for testing.
//!
//! # Cost when not exporting
//!
//! Every hook tests one `Option` and returns.

mod audio;
mod clips;
// GPU refusals and device loss, as messages and logs.
mod gpu_report;
mod panel;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_mods::mods::replay as replay_mod;
use mc_render::export::{ExportCapture, PreviewCompositor, PreviewLayout};
use mc_render::shell::DiagnosticClock;
use mc_render::{AcquiredFrame, Gpu};
use mc_replay::ReplayTime;
use mc_video::{
    size, AacBitrate, AudioConfig, EncodeSummary, EncoderConfig, EncoderPreference, EncoderThread,
    FrameRate, RateControl, VideoCodec, VideoError,
};

use super::super::Client;

/// Readback buffers in flight.
const READBACK_RING: usize = 4;
/// NV12 frames that may wait for the encoder.
const ENCODER_QUEUE: usize = 6;
/// Frames rendered at the range start before capturing, at least (TAA and other temporal effects
/// converge), and at most (terrain that never settles must not hang the export).
const SETTLE_MIN_FRAMES: u32 = 12;
const SETTLE_MAX_FRAMES: u32 = 2400;
/// How long (since the export began) the settle also waits for skin and
/// cape downloads still on their way, so players appear in their own skins from the first frame
/// of a replay this PC has not shown before (a friend's). An offline PC or a slow profile server
/// delays the start at most this much.
const SKIN_WAIT: Duration = Duration::from_secs(4);
/// How often the window's preview refreshes.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(50);

/// What to export. The dialog fills this in ([`Client::export_defaults`] gives a start).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExportRequest {
    /// First replay µs exported (included).
    pub(crate) start: ReplayTime,
    /// Last replay µs exported (excluded).
    pub(crate) end: ReplayTime,
    /// Output size; rounded down to even numbers.
    pub(crate) width: u32,
    /// Output size; rounded down to even numbers.
    pub(crate) height: u32,
    /// Frames per second, any value (24 to 2000 and beyond).
    pub(crate) fps: FrameRate,
    /// H.264 or H.265.
    pub(crate) codec: VideoCodec,
    /// A quality preset or a size limit.
    pub(crate) target: ExportTarget,
    /// Rendered frames blended into each output frame (1 = no motion blur).
    pub(crate) motion_blur: u32,
    /// Include the game's sound.
    pub(crate) game_audio: bool,
    /// Include the recorded microphone.
    pub(crate) mic: bool,
    /// Microphone level, 0..=2 (the `mic_volume` option over 100).
    pub(crate) mic_volume: f32,
    /// The game's sound level, 0..=2 (the `game_volume` option over 100).
    pub(crate) game_volume: f32,
    /// Include recorded desktop audio.
    pub(crate) desktop: bool,
    /// Desktop audio level, 0..=2 (the `desktop_volume` option over 100).
    pub(crate) desktop_volume: f32,
    /// One more audio track per included sound (game, microphone, desktop audio)
    /// after the mixed one, for video editors.
    pub(crate) separate_tracks: bool,
    /// The finished file. Its folder is created if missing.
    pub(crate) path: PathBuf,
}

/// Quality or size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ExportTarget {
    /// An `export_quality` choice: 0 small file, 1 balanced, 2 high, 3 near lossless.
    Quality(usize),
    /// Stay under this many bytes (a 10 MB upload limit...).
    MaxBytes(u64),
}

/// Where an export is.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExportPhase {
    /// Waiting for the frame loop to switch to the export's resolution.
    Starting,
    /// Rendering the range's first moment until the world has loaded.
    Settling,
    /// Rendering and encoding.
    Rendering,
    /// The encoder is finishing the file.
    Finishing,
    /// Written.
    Done {
        /// The file.
        path: PathBuf,
        /// Its size.
        bytes: u64,
    },
    /// Stopped by an error (the message says what).
    Failed(String),
    /// Stopped by the player.
    Cancelled,
}

impl ExportPhase {
    /// Whether the export has ended.
    #[must_use]
    pub(crate) fn is_finished(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Failed(_) | Self::Cancelled)
    }
}

/// Progress for a progress bar (the dialog, the clip toast, the window panel).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExportProgress {
    /// Where it is.
    pub(crate) phase: ExportPhase,
    /// Output frames rendered in this pass.
    pub(crate) frames_done: u64,
    /// Output frames in the range.
    pub(crate) frames_total: u64,
    /// 1 normally; 2 or 3 when a size limit needed another pass.
    pub(crate) pass: u32,
    /// Wall time since the export started.
    pub(crate) elapsed: Duration,
    /// Estimated time left, once rendering has started.
    pub(crate) eta: Option<Duration>,
    /// For example "NVIDIA H.264 Encoder MFT (GPU)".
    pub(crate) encoder: String,
    /// 0..=1 over the whole export.
    pub(crate) fraction: f32,
}

/// Which clock drives the export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Clock {
    /// The replay: `replay_render_at` per rendered frame.
    Replay,
    /// Any live session (the `.mccap` benchmark path): the shell steps the simulation.
    Legacy,
}

/// The audio inputs, kept so a second pass can build the mix again.
struct AudioPlan {
    sounds: Vec<audio::PlacedSound>,
    listeners: Vec<mc_audio::Listener>,
    mic: Option<mc_video::mix::Voice>,
    /// Desktop audio, a centred voice like the microphone.
    desktop: Option<mc_video::mix::Voice>,
    /// Whether the game's sound is included, and its level.
    game: bool,
    game_gain: f32,
    /// Write each included channel as a track of its own too.
    tracks: bool,
}

impl AudioPlan {
    /// The names of the tracks after the mixed one, in the order
    /// [`audio::MixSource`] fills them (game, microphone, desktop audio; the ones included).
    /// Names fit the 12 bytes Media Foundation leaves (`mc_video::tracks`).
    fn track_names(&self) -> Vec<String> {
        if !self.tracks {
            return Vec::new();
        }
        let mut names = Vec::new();
        if self.game {
            names.push("Game sound".to_owned());
        }
        if self.mic.is_some() {
            names.push("Microphone".to_owned());
        }
        if self.desktop.is_some() {
            names.push("Desktop".to_owned());
        }
        names
    }

    fn source(&self, start: ReplayTime) -> Box<dyn mc_video::AudioSource> {
        let sounds = self
            .sounds
            .iter()
            .map(|s| audio::PlacedSound {
                time: s.time,
                clip: s.clip.clone(),
                gain: s.gain,
                pitch: s.pitch,
                placement: s.placement,
            })
            .collect();
        let voices = [self.mic.clone(), self.desktop.clone()]
            .into_iter()
            .flatten()
            .map(|voice| audio::VoiceChannel {
                voice,
                track: self.tracks,
            })
            .collect();
        Box::new(audio::MixSource::with_channels(
            start,
            sounds,
            self.listeners.clone(),
            self.game_gain,
            self.tracks && self.game,
            voices,
        ))
    }
}

/// What the export changed on the client, put back when it ends.
struct Saved {
    deterministic_mesh: bool,
}

/// One running export.
struct Job {
    request: ExportRequest,
    clock: Clock,
    phase: ExportPhase,
    started: Instant,
    render_started: Option<Instant>,
    pass: u32,
    /// The constant bitrate of a size-limited pass.
    bitrate: Option<u32>,
    aac: Option<AacBitrate>,
    audio: Option<AudioPlan>,
    frames_total: u64,
    /// Rendered frames per output frame.
    blur: u32,
    /// Rendered frames this pass needs.
    sub_total: u64,
    /// The next rendered frame's index (its time is `sub_time(index)`).
    sub_next: u64,
    /// Rendered frames captured this pass.
    sub_done: u64,
    settle_frames: u32,
    encoder: Option<EncoderThread>,
    capture: Option<ExportCapture>,
    finishing: Option<std::thread::JoinHandle<Result<EncodeSummary, VideoError>>>,
    preview: Option<PreviewCompositor>,
    last_preview: Option<Instant>,
    panel: Option<(panel::PanelText, u32)>,
    encoder_name: String,
    saved: Saved,
    /// Frame grabs to write (output frame index → file), for checks.
    grabs: Vec<(u64, PathBuf)>,
    /// `MC_RD_EXPORT`'s `raw`: every output frame's NV12 bytes appended here (calibration).
    raw: Option<std::io::BufWriter<std::fs::File>>,
    /// The replay time of the frame being rendered (set in `before_ticks`); the legacy clock's
    /// render clock ([`Client::render_clock`]).
    frame_time: ReplayTime,
    /// The terrain renderer's refused uploads when the export started: more during
    /// the export means the GPU ran out of memory for the world, and the video would have holes.
    refused_at_start: u64,
    /// The encoder runs on the graphics card: its failures name Processor only.
    hardware_encoder: bool,
}

impl Job {
    /// Replay time of rendered frame `index`: `start + index / (fps x blur)`, exactly.
    fn sub_time(&self, index: u64) -> ReplayTime {
        let rate = FrameRate::new(
            self.request.fps.num.saturating_mul(self.blur),
            self.request.fps.den,
        );
        self.request.start + rate.frame_start_micros(index)
    }

    fn progress(&self) -> ExportProgress {
        let frames_done = self.sub_done / u64::from(self.blur.max(1));
        let within = if self.sub_total == 0 {
            0.0
        } else {
            self.sub_done as f64 / self.sub_total as f64
        };
        let fraction = match self.phase {
            ExportPhase::Done { .. } => 1.0,
            ExportPhase::Finishing => 1.0,
            _ => within,
        };
        // A later pass is shown as starting over: the bar would otherwise sit at 100%.
        let eta = self.render_started.and_then(|started| {
            (within > 0.02).then(|| {
                let spent = started.elapsed().as_secs_f64();
                Duration::from_secs_f64(spent / within * (1.0 - within))
            })
        });
        ExportProgress {
            phase: self.phase.clone(),
            frames_done,
            frames_total: self.frames_total,
            pass: self.pass,
            elapsed: self.started.elapsed(),
            eta,
            encoder: self.encoder_name.clone(),
            fraction: fraction as f32,
        }
    }
}

/// `MC_RD_EXPORT`: start an export by itself (tests and sample exports). `key=value` pairs
/// separated by `;`: `seconds` (5), `start` (µs into the range, 0), `fps` (60), `size`
/// (`1280x720`), `blur` (1), `codec` (`h264`/`h265`), `quality` (0-3, 2) or `maxmb` (a size
/// limit in MB), `audio` (1), `out` (file), `grabs` (folder for PNG frame grabs), `exit` (1:
/// quit when done), `wait` (frames to wait after the world is ready, 60).
#[derive(Debug, Clone)]
pub(crate) struct Harness {
    request: ExportRequest,
    wait_frames: u32,
    exit: bool,
    grabs: Option<PathBuf>,
    raw: Option<PathBuf>,
    /// Cancel after this many rendered frames (tests the cancel path).
    cancel: Option<u64>,
    /// Save the window's preview (frame + panel) here once, halfway.
    preview: Option<PathBuf>,
    /// `MC_RD_REPLAY`: open this replay from the title screen first (the playback), so
    /// the export runs on the replay clock with the recorded sounds.
    replay: Option<PathBuf>,
    /// `repass=1`: treat a size-limited first pass as over the limit, to exercise the second pass.
    repass: bool,
    seconds: f64,
    offset: u64,
    started: bool,
    waited: u32,
    /// `starts=10,60,5`: after this export, more of the same length at these
    /// offsets (seconds into the range), one after another in the same session: GPU memory
    /// across exports and the seeks between them (a backward one rebuilds the world).
    more_starts: Vec<u64>,
    /// `camera=pov|follow|free`: the replay camera to export through.
    camera: Option<super::tw_replay_play::ReplayCameraMode>,
    /// `lose=<n>`: destroy the GPU device after `n` rendered frames, to play a
    /// driver reset (the AMD export crash) through the real client.
    lose_at: Option<u64>,
    /// `options=main|video|chat` with `shots=<folder>`: instead of exporting,
    /// watch the replay with the viewer's controls (at `start`), open the game's options over it
    /// (and that page), go back, and save a picture of the window at each step.
    options: Option<String>,
    shots: Option<PathBuf>,
    options_step: u8,
    step_frames: u32,
    pending_shot: Option<PathBuf>,
}

impl Harness {
    fn from_env() -> Option<Self> {
        let spec = std::env::var("MC_RD_EXPORT").ok()?;
        let mut harness = Harness {
            request: ExportRequest {
                start: 0,
                end: 0,
                width: 1280,
                height: 720,
                fps: FrameRate::whole(60),
                codec: VideoCodec::H264,
                target: ExportTarget::Quality(2),
                motion_blur: 1,
                game_audio: true,
                mic: false,
                mic_volume: 1.0,
                game_volume: 1.0,
                desktop: false,
                desktop_volume: 1.0,
                separate_tracks: false,
                path: std::env::temp_dir().join("mc-rd-export.mp4"),
            },
            wait_frames: 60,
            exit: false,
            grabs: None,
            raw: None,
            cancel: None,
            preview: None,
            replay: std::env::var_os("MC_RD_REPLAY").map(PathBuf::from),
            repass: false,
            seconds: 5.0,
            offset: 0,
            started: false,
            waited: 0,
            more_starts: Vec::new(),
            camera: None,
            lose_at: None,
            options: None,
            shots: None,
            options_step: 0,
            step_frames: 0,
            pending_shot: None,
        };
        for pair in spec.split(';').filter(|p| !p.trim().is_empty()) {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            let value = value.trim();
            match key.trim() {
                "seconds" => harness.seconds = value.parse().unwrap_or(5.0),
                "start" => harness.offset = value.parse().unwrap_or(0),
                "fps" => {
                    harness.request.fps = FrameRate::whole(value.parse().unwrap_or(60).max(1));
                }
                "size" => {
                    if let Some((w, h)) = value.split_once('x') {
                        harness.request.width = w.parse().unwrap_or(1280);
                        harness.request.height = h.parse().unwrap_or(720);
                    }
                }
                "blur" => harness.request.motion_blur = value.parse().unwrap_or(1).max(1),
                "codec" => {
                    harness.request.codec = if value.eq_ignore_ascii_case("h265") {
                        VideoCodec::H265
                    } else {
                        VideoCodec::H264
                    };
                }
                "quality" => {
                    harness.request.target = ExportTarget::Quality(value.parse().unwrap_or(2));
                }
                "maxmb" => {
                    let mb: f64 = value.parse().unwrap_or(10.0);
                    harness.request.target = ExportTarget::MaxBytes((mb * 1_000_000.0) as u64);
                }
                "audio" => harness.request.game_audio = value != "0",
                "mic" => harness.request.mic = value == "1",
                // The sound channels and separate tracks.
                "desktop" => harness.request.desktop = value == "1",
                "tracks" => harness.request.separate_tracks = value == "1",
                "game_volume" => {
                    harness.request.game_volume = value.parse().unwrap_or(100.0) / 100.0;
                }
                "out" => harness.request.path = PathBuf::from(value),
                "grabs" => harness.grabs = Some(PathBuf::from(value)),
                "raw" => harness.raw = Some(PathBuf::from(value)),
                "cancel" => harness.cancel = value.parse().ok(),
                "preview" => harness.preview = Some(PathBuf::from(value)),
                "repass" => harness.repass = value == "1",
                "exit" => harness.exit = value == "1",
                "wait" => harness.wait_frames = value.parse().unwrap_or(60),
                "starts" => {
                    harness.more_starts = value
                        .split(',')
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .map(|s| (s.max(0.0) * 1e6) as u64)
                        .collect();
                }
                "lose" => harness.lose_at = value.parse().ok(),
                "options" => harness.options = Some(value.to_ascii_lowercase()),
                "shots" => harness.shots = Some(PathBuf::from(value)),
                "camera" => {
                    use super::tw_replay_play::ReplayCameraMode as Mode;
                    harness.camera = match value.to_ascii_lowercase().as_str() {
                        "pov" | "player" => Some(Mode::Pov),
                        "follow" | "third" => Some(Mode::ThirdPerson),
                        "free" => Some(Mode::Free),
                        _ => None,
                    };
                }
                other => tracing::warn!(key = other, "MC_RD_EXPORT: unknown key"),
            }
        }
        Some(harness)
    }
}

/// The clock render-time animations read: shader-pack time (waving plants, water waves), the
/// Dreamy pack's fog wind and the volumetric clouds' wind and drift, and the eases that follow the
/// player (export determinism).
///
/// * Normal play: seconds since the client started, on the wall clock (exactly what the shader
///   packs read before this existed, so normal play is unchanged).
/// * A watched replay: the replay's own time, so its speed and pause apply to animations too.
/// * A video export: the replay time of the frame being rendered, exact for every output frame
///   and every motion-blur sub-frame (the `replay_render_at` sets it); the legacy export
///   clock likewise.
///
/// `epoch` changes whenever the clock jumps (a replay opened or closed, a seek, each export pass):
/// a consumer that integrates restarts from the clock instead of adding the jump. While `replay`
/// is set, consumers derive their state from `seconds` alone where they can, so the same replay
/// moment looks the same at any export frame rate and in every run.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct RenderClock {
    /// Seconds.
    pub(crate) seconds: f64,
    /// Bumped at every jump of the clock.
    pub(crate) epoch: u32,
    /// The clock follows a replay or an export (deterministic), not the wall clock.
    pub(crate) replay: bool,
}

/// A replay-time step larger than this (or backward) starts a new [`RenderClock`] epoch: a seek,
/// not playback (8x speed at 30 fps moves 0.27 s a frame).
const CLOCK_JUMP_SECONDS: f64 = 1.0;

/// The field on [`Client`] (`tw_export`).
#[derive(Default)]
pub(crate) struct ReplayExportState {
    job: Option<Box<Job>>,
    /// The last export's outcome, until [`Client::export_take_finished`].
    finished: Option<ExportProgress>,
    harness: Option<Option<Harness>>,
    /// The window's size, as last seen (for "Window size" exports).
    window: (u32, u32),
    /// Clips to video: the open clip of a clip export, the idle title screen's automatic export.
    clips: clips::ClipExports,
    /// The render clock ([`Client::render_clock`]): this frame's value once someone asked, the
    /// frame it was read in, and a pending jump (each export pass starts a new epoch).
    clock: std::cell::Cell<RenderClock>,
    clock_frame: std::cell::Cell<u64>,
    clock_jump: std::cell::Cell<bool>,
    /// Frames seen (`tw_export_frame`), so the clock is read at most once a frame.
    frame: u64,
    /// The adapter the game renders with (read once, on the first frame): the
    /// hardware encoder is put on it.
    adapter: Option<mc_video::GpuAdapter>,
    /// An export that ended because the device was lost: its message, for the
    /// crash report the shell's stop makes on the next frame ([`Client::tw_export_device_lost`]).
    lost_export: Option<String>,
}

impl std::fmt::Debug for ReplayExportState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplayExportState")
            .field("running", &self.job.is_some())
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

/// The `export_resolution` / `clip_resolution` choice as a size (`window` for "Window size").
#[must_use]
pub(crate) fn resolution_for_choice(index: usize, window: (u32, u32)) -> (u32, u32) {
    match index {
        1 => (1280, 720),
        2 => (1920, 1080),
        3 => (2560, 1440),
        4 => (3840, 2160),
        _ => (window.0.max(16) & !1, window.1.max(16) & !1),
    }
}

/// Rendered frames per output frame for an `export_motion_blur` choice.
#[must_use]
pub(crate) fn blur_for_choice(index: usize) -> u32 {
    1 << index.min(5)
}

/// The adapter the game renders with, as the encoder looks it up (its PCI
/// vendor and device IDs, the same under Vulkan and Direct3D 12, and its name).
fn render_adapter(gpu: &Gpu) -> mc_video::GpuAdapter {
    let info = gpu.adapter_info();
    mc_video::GpuAdapter {
        vendor: info.vendor,
        device: info.device,
        name: info.name,
    }
}

/// The encoder's constant-quality value for an `export_quality` choice. Calibrated on real
/// SkyWars frames (720p60, moving camera; `mc-video/examples/calibrate.rs`), luma PSNR with
/// NVENC H.264: 55 = 34.7 dB (small), 68 = 38 dB (balanced), 80 = 40.5 dB (high, the default),
/// 96 = about 44 dB (near lossless). H.265 is about 0.8 dB better at 9% less bitrate.
fn quality_for_choice(index: usize) -> u8 {
    match index {
        0 => 55,
        1 => 68,
        2 => 80,
        _ => 96,
    }
}

/// `%USERPROFILE%\Videos\Aero Client`, or `<gameDir>\exports` without a user profile.
#[must_use]
pub(crate) fn default_export_folder(game_dir: &Path) -> PathBuf {
    match std::env::var_os("USERPROFILE") {
        Some(profile) => PathBuf::from(profile).join("Videos").join("Aero Client"),
        None => game_dir.join("exports"),
    }
}

/// A file name without characters Windows refuses.
#[must_use]
pub(crate) fn safe_file_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let trimmed = cleaned.trim().trim_end_matches('.').trim();
    let name: String = if trimmed.is_empty() {
        "Replay".to_owned()
    } else {
        trimmed.chars().take(120).collect()
    };
    // The title is the replay file's own: a name Windows reads as a
    // device (`NUL`, `CON`, `COM1`..., also with an extension on older Windows) would export
    // into the device instead of a file.
    if mc_assets::is_windows_device_name(&name) {
        format!("_{name}")
    } else {
        name
    }
}

/// A file's (or folder's) own name for logs and messages, never the folders
/// above it: they hold the player's account name, and players share their logs.
pub(crate) fn nw_rq_file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || String::from("(no name)"),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// `path`, or `path (2)`, `path (3)`... whichever does not exist yet.
fn unused_path(path: PathBuf) -> PathBuf {
    if !path.exists() {
        return path;
    }
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mp4".to_owned());
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    (2..1000)
        .map(|n| parent.join(format!("{stem} ({n}).{ext}")))
        .find(|candidate| !candidate.exists())
        .unwrap_or(path)
}

impl Client {
    // -----------------------------------------------------------------------------------------
    // API for the export dialog and the clip export
    // -----------------------------------------------------------------------------------------

    /// Whether an export is running (hide the viewer's overlay and ignore its input
    /// while this is true; the frames being rendered are the video).
    #[must_use]
    pub(crate) fn export_running(&self) -> bool {
        self.tw_export.job.is_some()
    }

    /// The running export's progress, or the last finished one's until it is taken (for a
    /// progress view of the own; the window shows the export's panel either way).
    #[must_use]
    #[allow(dead_code)] // part of the export API; RC shows only the outcome today
    pub(crate) fn export_progress(&self) -> Option<ExportProgress> {
        match &self.tw_export.job {
            Some(job) => Some(job.progress()),
            None => self.tw_export.finished.clone(),
        }
    }

    /// The last finished export's outcome (done, failed or cancelled), once.
    pub(crate) fn export_take_finished(&mut self) -> Option<ExportProgress> {
        self.tw_export.finished.take()
    }

    /// Stop the running export. The partial file is deleted.
    pub(crate) fn export_cancel(&mut self) {
        if let Some(job) = self.tw_export.job.as_mut() {
            job.phase = ExportPhase::Cancelled;
        }
        self.tw_export_end();
    }

    /// The dialog's starting values: the open replay's visible range (or clip window) and the
    /// `replay` mod's export defaults, into `%USERPROFILE%\Videos\Aero Client`.
    #[must_use]
    pub(crate) fn export_defaults(&self) -> ExportRequest {
        use replay_mod::ids;
        let settings = &self.mods.settings;
        let (start, end) = if self.replay_is_active() {
            self.replay_view_range()
        } else {
            (0, 0)
        };
        let fps = replay_mod::EXPORT_FPS_CHOICES
            .get(settings.choice("replay", ids::EXPORT_FPS))
            .and_then(|label| label.parse().ok())
            .unwrap_or(60);
        let (width, height) = resolution_for_choice(
            settings.choice("replay", ids::EXPORT_RESOLUTION),
            self.tw_export.window,
        );
        let codec = if settings.choice("replay", ids::EXPORT_CODEC) == 1 {
            VideoCodec::H265
        } else {
            VideoCodec::H264
        };
        let title = self
            .replay_meta()
            .map(|meta| meta.title.clone())
            .filter(|title| !title.trim().is_empty())
            .unwrap_or_else(|| "Replay".to_owned());
        let folder = default_export_folder(&self.config.game_dir);
        ExportRequest {
            start,
            end,
            width,
            height,
            fps: FrameRate::whole(fps),
            codec,
            target: ExportTarget::Quality(settings.choice("replay", ids::EXPORT_QUALITY)),
            motion_blur: blur_for_choice(settings.choice("replay", ids::EXPORT_MOTION_BLUR)),
            game_audio: settings.bool("replay", ids::EXPORT_AUDIO),
            mic: settings.bool("replay", ids::EXPORT_MIC),
            mic_volume: settings.f32("replay", ids::MIC_VOLUME) / 100.0,
            game_volume: settings.f32("replay", ids::GAME_VOLUME) / 100.0,
            desktop: settings.bool("replay", ids::EXPORT_DESKTOP),
            desktop_volume: settings.f32("replay", ids::DESKTOP_VOLUME) / 100.0,
            separate_tracks: settings.bool("replay", ids::EXPORT_TRACKS),
            path: folder.join(format!("{}.mp4", safe_file_name(&title))),
        }
    }

    /// Whether H.265 can be exported on this PC (a GPU encoder or the HEVC Video Extensions).
    #[must_use]
    pub(crate) fn export_h265_available() -> bool {
        mc_video::probe::codec_support(VideoCodec::H265).is_some()
    }

    /// Start exporting the open replay (or, with no replay open, the live session from now, the
    /// benchmark path). Fails without starting when something is wrong with the request.
    ///
    /// # Errors
    /// A message for the dialog.
    pub(crate) fn export_start(&mut self, request: ExportRequest) -> Result<(), String> {
        if self.tw_export.job.is_some() {
            return Err("An export is already running.".to_owned());
        }
        // The loading screen draws over everything until it has faded out: it must not end
        // up in the video.
        if self.tw_loading_frame_cap().is_some() {
            return Err("The game is still loading.".to_owned());
        }
        let clock = if self.replay_is_active() {
            Clock::Replay
        } else if self.session.is_some() {
            Clock::Legacy
        } else {
            return Err("Open a replay first.".to_owned());
        };
        let mut request = request;
        request.width &= !1;
        request.height &= !1;
        request.motion_blur = request.motion_blur.clamp(1, 64);
        if clock == Clock::Replay {
            // Never past the replay's ends: frames there would only repeat its last moment.
            let (first, last) = self.replay_view_range();
            request.start = request.start.clamp(first, last);
            request.end = request.end.clamp(first, last);
        }
        if request.end <= request.start {
            return Err("The range is empty.".to_owned());
        }
        if !request.fps.is_valid() || request.fps.as_f64() > 10_000.0 {
            return Err("Choose a frame rate between 1 and 10000.".to_owned());
        }
        if request.width < 16
            || request.height < 16
            || request.width > 7680
            || request.height > 4320
        {
            return Err("Choose a size between 16x16 and 7680x4320.".to_owned());
        }
        if request.codec == VideoCodec::H265 && !Self::export_h265_available() {
            return Err("This PC cannot encode H.265. Choose H.264.".to_owned());
        }
        if !mc_video::probe::media_foundation_available() {
            return Err("Video export needs Windows Media Foundation (on Windows N, install the Media Feature Pack).".to_owned());
        }
        let span = request.end - request.start;
        let frames_total = request.fps.frames_in_micros(span).max(1);
        let seconds = span as f64 / 1e6;
        let blur = request.motion_blur;
        // Decode and place the sounds first: no sound at all (the legacy clock, a silent range
        // with the mic off) means no audio track rather than a silent one.
        let audio = self.tw_export_audio_plan(&request, clock);
        let aac = audio.as_ref().map(|_| AacBitrate::Kbps128);
        // Separate tracks are more AAC streams at the mix's rate (one per channel, plus
        // mc-video's copy of the mix at the end); a size limit leaves room for them first.
        let names = audio.as_ref().map_or(0, |plan| plan.track_names().len()) as u64;
        let extra_tracks = if names > 0 { names + 1 } else { 0 };
        let per_second = u64::from(AacBitrate::Kbps128.bytes_per_second());
        let extra_bytes = extra_tracks * per_second * (seconds.ceil() as u64);
        let (bitrate, aac) = match request.target {
            ExportTarget::MaxBytes(max) => {
                let plan = size::plan(max.saturating_sub(extra_bytes), seconds, frames_total, aac);
                if !plan.fits {
                    tracing::warn!(
                        max,
                        seconds,
                        "export: the size limit is too small; the file will be over"
                    );
                }
                (Some(plan.video_bitrate), plan.audio)
            }
            ExportTarget::Quality(_) => (None, aac),
        };
        if let Some(parent) = request.path.parent() {
            // The folder's name, not its whole path (the account name).
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Could not create the folder \u{201c}{}\u{201d}: {e}",
                    nw_rq_file_name(parent)
                )
            })?;
        }
        // Never overwrite an earlier export: "name (2).mp4" instead (the outcome names the file).
        request.path = unused_path(request.path);
        let saved = Saved {
            deterministic_mesh: self.replay_deterministic_mesh,
        };
        self.set_replay_deterministic_mesh(true);
        // The post blurs' notion of a frame's duration: one rendered frame.
        let rendered_fps = request.fps.as_f64() * f64::from(request.motion_blur);
        self.tw_export_set_fixed_dt(Some((1.0 / rendered_fps) as f32));
        let job = Job {
            sub_total: frames_total * u64::from(blur),
            request,
            clock,
            phase: ExportPhase::Starting,
            started: Instant::now(),
            render_started: None,
            pass: 1,
            bitrate,
            aac,
            audio,
            frames_total,
            blur,
            sub_next: 0,
            sub_done: 0,
            settle_frames: 0,
            encoder: None,
            capture: None,
            finishing: None,
            preview: None,
            last_preview: None,
            panel: None,
            encoder_name: String::new(),
            saved,
            grabs: Vec::new(),
            raw: None,
            frame_time: 0,
            refused_at_start: self.nw_rd_refused_uploads(),
            hardware_encoder: false,
        };
        tracing::info!(
            width = job.request.width,
            height = job.request.height,
            fps = job.request.fps.as_f64(),
            frames = job.frames_total,
            blur = job.blur,
            codec = job.request.codec.label(),
            ?bitrate,
            file = %nw_rq_file_name(&job.request.path),
            ?clock,
            "export started",
        );
        self.tw_export.finished = None;
        self.tw_export.job = Some(Box::new(job));
        if let Err(error) = self.tw_export_open_encoder() {
            // Nothing was written; put everything back and tell the dialog directly.
            self.tw_export_fail(error.clone());
            self.tw_export.finished = None;
            return Err(error);
        }
        Ok(())
    }

    /// The sounds and listener of `request`'s range, decoded and placed (nothing for the legacy
    /// clock, which has no recorded sounds).
    fn tw_export_audio_plan(&mut self, request: &ExportRequest, clock: Clock) -> Option<AudioPlan> {
        if clock != Clock::Replay || !(request.game_audio || request.mic || request.desktop) {
            return None;
        }
        let started = Instant::now();
        let mut sounds = Vec::new();
        let mut cache = audio::ClipCache::default();
        if request.game_audio {
            let from = request.start.saturating_sub(audio::LOOKBACK_MICROS);
            for sound in self.replay_sounds_in(from, request.end) {
                // Effects are short: one that started well before the range is over by then.
                // Only music and records (streamed) reach back further.
                if !sound.streaming && sound.time + audio::EFFECT_LOOKBACK_MICROS < request.start {
                    continue;
                }
                let Some(file) = sound.file.as_deref() else {
                    continue;
                };
                let Some(clip) = cache.get(&self.assets.resources, file) else {
                    continue;
                };
                let category = sound
                    .category
                    .map_or(1.0, |category| self.audio.categories.effective(category));
                let event_gain = mc_assets::ResourceLocation::parse(&sound.event)
                    .map_or(1.0, |event| self.audio.event_gains().get(&event));
                let gain =
                    mc_audio::normalized_volume(sound.volume, sound.variant_volume, category)
                        * event_gain;
                if gain <= 0.0 {
                    continue;
                }
                sounds.push(audio::PlacedSound {
                    time: sound.time,
                    clip,
                    gain,
                    pitch: mc_audio::normalized_pitch(sound.pitch, sound.variant_pitch),
                    placement: audio::Placement {
                        position: sound.position,
                        linear: sound.attenuation_linear,
                        volume: sound.volume,
                    },
                });
            }
        }
        // The listener, every 10 ms: the export camera. In the POV and third-person cameras that
        // is the recorded player's eyes (third person sits a few blocks behind them); a free
        // camera does not move during an export (nobody flies it), so its pose holds throughout.
        let step = u64::from(audio::LISTENER_STEP) * 1_000_000 / u64::from(audio::RATE);
        let free_camera = (self.replay_camera_mode()
            == super::tw_replay_play::ReplayCameraMode::Free)
            .then(|| self.replay_last_camera())
            .flatten()
            .map(|pose| listener_from(pose.pos, pose.yaw, pose.pitch));
        let mut listeners = Vec::new();
        let mut t = request.start;
        let mut last = None;
        let mut posed = 0_usize;
        while t <= request.end {
            if let Some(fixed) = free_camera {
                last = Some(fixed);
                posed += 1;
            } else if let Some(pose) = self.replay_camera_at(t) {
                last = Some(listener_from(pose.pos, pose.yaw, pose.pitch));
                posed += 1;
            }
            listeners.push(last);
            t += step;
        }
        // Times before the first recorded pose hear from the first one.
        let first = listeners.iter().flatten().next().copied();
        let listeners: Vec<mc_audio::Listener> = listeners
            .into_iter()
            .map(|l| l.or(first).unwrap_or_else(mc_audio::Listener::origin))
            .collect();
        let near = |listener: &mc_audio::Listener, sound: &audio::PlacedSound| {
            sound
                .placement
                .position
                .is_none_or(|p| listener.distance_to(p) < 16.0)
        };
        let audible = listeners
            .first()
            .map_or(0, |l| sounds.iter().filter(|s| near(l, s)).count());
        tracing::info!(
            posed,
            samples = listeners.len(),
            listener = ?listeners.first().map(|l| l.position),
            audible_from_start = audible,
            "export: listener track",
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            // The loudest sounds of the range as heard: gain x distance x the clip's own peak.
            let mut heard: Vec<(f32, String)> = sounds
                .iter()
                .filter(|s| s.time >= request.start && s.time < request.end)
                .map(|s| {
                    let index = ((s.time - request.start) / step) as usize;
                    let listener = listeners[index.min(listeners.len() - 1)];
                    let distance = s
                        .placement
                        .position
                        .map_or(0.0, |p| listener.distance_to(p));
                    let kind = if s.placement.linear {
                        mc_audio::AttenuationType::Linear
                    } else {
                        mc_audio::AttenuationType::None
                    };
                    let atten = mc_audio::attenuate(distance, kind, s.placement.volume);
                    let peak = s.clip.samples.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                    (
                        s.gain * atten * peak,
                        format!("gain {:.2} dist {distance:.1} peak {peak:.2}", s.gain),
                    )
                })
                .collect();
            heard.sort_by(|a, b| b.0.total_cmp(&a.0));
            tracing::debug!(in_range = heard.len(), loudest = ?&heard[..heard.len().min(8)], "export: sounds heard");
        }
        let mic = if request.mic {
            let (rate, frames) = self.tw_export_mic_frames(request.start, request.end);
            audio::mic_voice(
                request.start,
                request.end,
                rate,
                &frames,
                request.mic_volume,
            )
        } else {
            None
        };
        // Desktop audio, placed exactly like the microphone (same record layout).
        let desktop = if request.desktop {
            let (rate, frames) = self.tw_export_desktop_frames(request.start, request.end);
            audio::mic_voice(
                request.start,
                request.end,
                rate,
                &frames,
                request.desktop_volume,
            )
        } else {
            None
        };
        tracing::info!(
            sounds = sounds.len(),
            files = cache.len(),
            listeners = listeners.len(),
            mic = mic.is_some(),
            desktop = desktop.is_some(),
            tracks = request.separate_tracks,
            ms = started.elapsed().as_millis(),
            "export: audio mix prepared",
        );
        Some(AudioPlan {
            sounds,
            listeners,
            mic,
            desktop,
            game: request.game_audio,
            game_gain: request.game_volume.clamp(0.0, 2.0),
            tracks: request.separate_tracks,
        })
    }

    /// The recorded desktop audio frames in `[start, end)` (plus one second before),
    /// decoded, and their sample rate.
    fn tw_export_desktop_frames(
        &self,
        start: ReplayTime,
        end: ReplayTime,
    ) -> (u32, Vec<(ReplayTime, Vec<i16>)>) {
        let rate = self
            .replay_meta()
            .and_then(|meta| meta.desktop_format.as_ref())
            .map_or(mc_replay::desktop::SAMPLE_RATE, |format| format.sample_rate);
        let frames = self
            .replay_desktop_in(start.saturating_sub(1_000_000), end)
            .into_iter()
            .filter_map(|(time, payload)| {
                let mut pcm = Vec::new();
                mc_replay::desktop::decode_record(payload, &mut pcm)
                    .ok()
                    .map(|lead| (time.saturating_sub(u64::from(lead)), pcm))
            })
            .collect();
        (rate, frames)
    }

    /// The recorded microphone frames in `[start, end)` (plus one second before, for a frame
    /// that straddles the start), decoded from the IMA ADPCM, and their sample rate.
    fn tw_export_mic_frames(
        &self,
        start: ReplayTime,
        end: ReplayTime,
    ) -> (u32, Vec<(ReplayTime, Vec<i16>)>) {
        let rate = self
            .replay_meta()
            .and_then(|meta| meta.mic_format.as_ref())
            .map_or(24_000, |format| format.sample_rate);
        let frames = self
            .replay_mic_in(start.saturating_sub(1_000_000), end)
            .into_iter()
            .filter_map(|(time, payload)| {
                let mut pcm = Vec::new();
                // a record is a 4-byte lead plus the ADPCM frame; the frame's first
                // sample is at `time - lead` (see `mc_replay::mic`).
                mc_replay::mic::decode_record(payload, &mut pcm)
                    .ok()
                    .map(|lead| (time.saturating_sub(u64::from(lead)), pcm))
            })
            .collect();
        (rate, frames)
    }

    /// Start this pass's encoder.
    fn tw_export_open_encoder(&mut self) -> Result<(), String> {
        let recording_started_ms = self.replay_meta().map(|meta| meta.started_unix_ms);
        // "Video encoder: Processor only" (a graphics driver whose encoder
        // misbehaves); Automatic is the GPU's encoder when there is one, as before.
        let encoder_preference = if self
            .mods
            .settings
            .choice("replay", replay_mod::ids::EXPORT_ENCODER)
            == 1
        {
            EncoderPreference::SoftwareOnly
        } else {
            EncoderPreference::Auto
        };
        // The hardware encoder on the adapter the export renders with.
        let adapter = self.tw_export.adapter.clone();
        let Some(job) = self.tw_export.job.as_deref_mut() else {
            return Ok(());
        };
        let request = &job.request;
        let mut config = EncoderConfig::new(
            request.path.clone(),
            request.width,
            request.height,
            request.fps,
            request.codec,
        );
        // The file's "Media created" is when the range was played (the recording's clock), not
        // when it was exported: the same in every export of the range, so the files are equal.
        config.created = match job.clock {
            // The start is the replay file's: added checked, never a
            // panic on a value past what a clock holds.
            Clock::Replay => recording_started_ms.and_then(|ms| {
                std::time::UNIX_EPOCH
                    .checked_add(Duration::from_millis(ms))?
                    .checked_add(Duration::from_micros(request.start))
            }),
            Clock::Legacy => None,
        };
        config.rate = match (job.bitrate, request.target) {
            (Some(bitrate), _) => RateControl::ConstantBitrate { bitrate },
            (None, ExportTarget::Quality(index)) => RateControl::Quality {
                quality: quality_for_choice(index),
                max_bitrate: None,
            },
            (None, ExportTarget::MaxBytes(_)) => RateControl::Quality {
                quality: 70,
                max_bitrate: None,
            },
        };
        config.encoder = encoder_preference;
        config.adapter = adapter;
        config.audio = job.aac.map(|bitrate| AudioConfig {
            sample_rate: audio::RATE,
            channels: 2,
            bitrate,
        });
        // The separate tracks after the mixed one (none unless asked for).
        if config.audio.is_some() {
            config.audio_tracks = job
                .audio
                .as_ref()
                .map_or_else(Vec::new, AudioPlan::track_names);
        }
        let source = job.audio.as_ref().map(|plan| plan.source(request.start));
        // H.264 encoders stop at 4096 pixels across (the render matrix: 8K with
        // H.264 failed with only "no encoder accepts these settings"); say what to change.
        let too_wide_for_h264 =
            request.codec == VideoCodec::H264 && (request.width > 4096 || request.height > 4096);
        let encoder = EncoderThread::spawn(config, source, ENCODER_QUEUE).map_err(|error| {
            if too_wide_for_h264 {
                format!(
                    "H.264 cannot encode {}\u{00D7}{} (it stops at 4096 pixels across). Choose \
                     H.265 or a smaller size.",
                    request.width, request.height
                )
            } else {
                // Automatic may have tried the graphics card's encoder.
                gpu_report::encoder_failure(
                    "Could not start the encoder",
                    &error,
                    encoder_preference != EncoderPreference::SoftwareOnly,
                )
            }
        })?;
        let info = encoder.info();
        job.encoder_name = format!(
            "{} ({})",
            info.name,
            if info.hardware { "GPU" } else { "CPU" }
        );
        job.hardware_encoder = info.hardware;
        tracing::info!(encoder = %job.encoder_name, nominal_fps = info.nominal_rate.as_f64(), pass = job.pass, "export encoder ready");
        job.encoder = Some(encoder);
        job.capture = None;
        job.sub_next = 0;
        job.sub_done = 0;
        job.settle_frames = 0;
        job.render_started = None;
        job.phase = ExportPhase::Starting;
        Ok(())
    }

    fn tw_export_fail(&mut self, error: String) {
        tracing::warn!(%error, "export failed");
        if let Some(job) = self.tw_export.job.as_mut() {
            job.phase = ExportPhase::Failed(error);
        }
        self.tw_export_end();
    }

    /// End the job (any phase): drop the encoder (deleting a partial file), put back what the
    /// export changed, and keep the outcome for [`Self::export_take_finished`].
    fn tw_export_end(&mut self) {
        let Some(mut job) = self.tw_export.job.take() else {
            return;
        };
        if let Some(encoder) = job.encoder.take() {
            encoder.cancel();
        }
        if let Some(finishing) = job.finishing.take() {
            // Cancelled while the encoder was finishing the file: wait for it, then delete what
            // it wrote (the player asked for no file).
            if let Ok(Ok(summary)) = finishing.join() {
                let _ = std::fs::remove_file(&summary.path);
            }
        }
        if !job.phase.is_finished() {
            job.phase = ExportPhase::Cancelled;
        }
        self.set_replay_deterministic_mesh(job.saved.deterministic_mesh);
        self.tw_export_set_fixed_dt(None);
        tracing::info!(phase = ?job.phase, elapsed = ?job.started.elapsed(), "export ended");
        let progress = job.progress();
        // `starts`: the next export of the series, once the world is ready.
        let mut rearmed = false;
        if let Some(Some(harness)) = self.tw_export.harness.as_mut() {
            if !harness.more_starts.is_empty() {
                harness.offset = harness.more_starts.remove(0);
                harness.started = false;
                harness.waited = 0;
                rearmed = true;
                tracing::info!(
                    offset = harness.offset,
                    "MC_RD_EXPORT: next export of the series"
                );
            }
        }
        if let Some(Some(harness)) = self.tw_export.harness.as_ref() {
            if harness.exit && !rearmed {
                tracing::info!("MC_RD_EXPORT: done, quitting");
                self.request_quit_after_export();
            }
        }
        // A clip export closes its clip again on the next frame (`tw_export_frame`), outside the
        // render hooks this can run from.
        self.tw_export.finished = Some(progress);
    }

    /// The per-pass time step for the motion-blur post phases (their history blend), so they
    /// see the export's frame time instead of the wall clock.
    fn tw_export_set_fixed_dt(&mut self, dt: Option<f32>) {
        // Setters added to x9_aa.rs and m7_postfx.rs.
        self.x9_set_fixed_dt(dt);
        self.m7_set_fixed_dt(dt);
    }

    fn request_quit_after_export(&mut self) {
        self.quit();
    }

    /// This frame's [`RenderClock`]. Computed on the first call of a frame and cached, so it
    /// costs nothing in a frame where no animation asks for it.
    pub(crate) fn render_clock(&self) -> RenderClock {
        let state = &self.tw_export;
        if state.frame != 0 && state.clock_frame.get() == state.frame {
            return state.clock.get();
        }
        let (seconds, replay) = match state.job.as_deref() {
            Some(job) if job.clock == Clock::Legacy => (job.frame_time as f64 / 1e6, true),
            _ if self.replay_is_active() => (self.replay_time() as f64 / 1e6, true),
            _ => (self.started.elapsed().as_secs_f64(), false),
        };
        let previous = state.clock.get();
        let jumped = state.clock_jump.replace(false)
            || replay != previous.replay
            || (replay
                && (seconds < previous.seconds || seconds - previous.seconds > CLOCK_JUMP_SECONDS));
        let clock = RenderClock {
            seconds,
            epoch: previous.epoch.wrapping_add(u32::from(jumped)),
            replay,
        };
        state.clock.set(clock);
        state.clock_frame.set(state.frame);
        clock
    }

    // -----------------------------------------------------------------------------------------
    // Hooks (driver.rs)
    // -----------------------------------------------------------------------------------------

    /// `ShellApp::diagnostic_clock`, first: frozen during a replay export (`replay_render_at`
    /// moves the world), one rendered frame's time per loop on the legacy clock.
    pub(crate) fn tw_export_clock(&self) -> Option<DiagnosticClock> {
        let job = self.tw_export.job.as_deref()?;
        match (job.clock, &job.phase) {
            (Clock::Legacy, ExportPhase::Rendering) => {
                let micros = job.sub_time(job.sub_next + 1) - job.sub_time(job.sub_next);
                Some(DiagnosticClock::Step {
                    micros: micros.max(1),
                })
            }
            _ => Some(DiagnosticClock::Frozen),
        }
    }

    /// `ShellApp::effective_frame_cap`, first: an export renders as fast as it can.
    pub(crate) fn tw_export_uncapped(&self) -> bool {
        self.tw_export.job.is_some()
    }

    /// `ShellApp::offscreen_render_size`.
    pub(crate) fn tw_export_offscreen_size(&self) -> Option<(u32, u32)> {
        let job = self.tw_export.job.as_deref()?;
        Some((job.request.width, job.request.height))
    }

    /// `ShellApp::before_ticks`, after the loading check. Returns `true` when the export moved the
    /// world itself (the rest of `before_ticks` is skipped).
    pub(crate) fn tw_export_before_ticks(&mut self) -> bool {
        if self.tw_export.job.is_none() {
            return false;
        }
        // A shader pack's shadow map is drawn afresh for every exported frame (the cache would
        // keep an older frame's map, so a frame would depend on the frames before it).
        if let Some(render) = &self.render {
            render.terrain.borrow_mut().invalidate_pack_shadows();
        }
        let Some(job) = self.tw_export.job.as_deref_mut() else {
            return false;
        };
        let time = match job.phase {
            ExportPhase::Rendering => job.sub_time(job.sub_next),
            ExportPhase::Starting | ExportPhase::Settling => job.request.start,
            // Finishing: the world stays where the last frame left it (a seek back to the start
            // would rebuild it for nothing); a second pass goes back to `Starting`.
            _ => return job.clock == Clock::Replay,
        };
        job.frame_time = time;
        match job.clock {
            Clock::Replay => {
                // `Client::frame` takes the partial tick from the clock itself.
                let _partial = self.replay_render_at(time);
                true
            }
            // The legacy clock: packets and ticks advance as in any frame (the shell's clock
            // steps by one rendered frame, `tw_export_clock`).
            Clock::Legacy => false,
        }
    }

    /// `ShellApp::after_render`: capture the frame, feed the encoder, move the job along.
    pub(crate) fn tw_export_after_render(&mut self, gpu: &Gpu, frame: &AcquiredFrame) {
        if self.tw_export.job.is_none() {
            // The options pictures (an `Option` test without the harness).
            if matches!(&self.tw_export.harness, Some(Some(h)) if h.pending_shot.is_some()) {
                self.nw_rd_save_shot(gpu, frame);
            }
            return;
        }
        let (cancel_at, lose_at) = match self.tw_export.harness.as_ref() {
            Some(Some(harness)) => (harness.cancel, harness.lose_at),
            _ => (None, None),
        };
        let settled = self.replay_terrain_settled();
        // Skins requested by the settle frames, still downloading.
        let downloads_pending = self.appearance.worker_running()
            && (self.appearance.skins.inflight_len() > 0
                || self.appearance.capes.inflight_len() > 0);
        let Some(job) = self.tw_export.job.as_deref_mut() else {
            return;
        };
        match job.phase {
            ExportPhase::Starting => {
                // The first frame at the export's size has been drawn.
                if gpu.render_override() == Some((job.request.width, job.request.height)) {
                    job.phase = ExportPhase::Settling;
                }
                return;
            }
            ExportPhase::Settling => {
                job.settle_frames += 1;
                let skins_ready = !downloads_pending || job.started.elapsed() >= SKIN_WAIT;
                let ready = (settled && skins_ready && job.settle_frames >= SETTLE_MIN_FRAMES)
                    || job.settle_frames >= SETTLE_MAX_FRAMES;
                if ready {
                    if !settled {
                        tracing::warn!("export: terrain still loading; starting anyway");
                    }
                    // Every target exists at the export's size now.
                    gpu_report::log_memory(gpu, "export: GPU memory at the export's size");
                    job.phase = ExportPhase::Rendering;
                    job.render_started = Some(Instant::now());
                    // The first captured frame starts the render clock's integrators afresh (the
                    // clouds' history and jitter, the fog's ease): what the video shows then
                    // depends on the replay alone, not on how many frames the world took to
                    // load. Two exports of the same range (and both passes of one) match.
                    self.tw_export.clock_jump.set(true);
                    // The same for what vanilla keeps from *drawn*
                    // frames: the boss bar's countdown and the wither sky tint it asks for.
                    // Whether a settle frame drew a boss depends on how far the terrain had
                    // meshed (entities behind unmeshed sections are culled), so two exports of
                    // one range could start with a different bar. Cleared as every jump clears
                    // it; a boss in view brings it back in the first captured frame.
                    self.replay_debug_forget_drawn_frames();
                }
                return;
            }
            ExportPhase::Rendering => {}
            _ => return,
        }
        let Some(texture) = frame.color_texture(gpu) else {
            return;
        };
        let (w, h) = (job.request.width, job.request.height);
        // The capture's buffers (the blend accumulator is 16 bytes a pixel) are
        // refused with a message instead of a panic when the GPU cannot hold them.
        if job.capture.is_none() {
            match ExportCapture::try_new(gpu, w, h, gpu.config.format, job.blur, READBACK_RING) {
                Ok(capture) => job.capture = Some(capture),
                Err(error) => {
                    let message = gpu_report::export_failure(&error, (w, h), job.blur);
                    // For the crash report of the shell's stop next frame.
                    if gpu.is_device_lost() {
                        self.tw_export.lost_export = Some(message.clone());
                    }
                    self.tw_export_fail(message);
                    return;
                }
            }
        }
        let mut failure = None;
        {
            let Job {
                capture,
                encoder,
                grabs,
                raw,
                blur,
                ..
            } = &mut *job;
            let blur = *blur;
            let capture = capture.get_or_insert_with(|| {
                ExportCapture::new(gpu, w, h, gpu.config.format, blur, READBACK_RING)
            });
            let mut sink = |index: u64, nv12: &[u8]| {
                deliver(encoder, grabs, raw, (w, h), index, nv12, &mut failure);
            };
            capture.capture(gpu, texture, &mut sink);
            capture.poll(gpu, &mut sink);
        }
        job.sub_next += 1;
        job.sub_done += 1;
        if cancel_at == Some(job.sub_done) {
            tracing::info!(frames = job.sub_done, "MC_RD_EXPORT: cancelling as asked");
            self.export_cancel();
            return;
        }
        if let Some(error) = failure {
            let hardware = job.hardware_encoder;
            self.tw_export_fail(gpu_report::encoder_failure(
                "The encoder stopped",
                &error,
                hardware,
            ));
            return;
        }
        // Frames the GPU could not hand back, or world the GPU had no memory for,
        // would make a video with missing frames or holes: stop and say why instead.
        let failed_readbacks = job.capture.as_ref().map_or(0, |c| c.stats().failed);
        let refused_at_start = job.refused_at_start;
        let every = gpu_report::memory_log_every();
        if every.is_some_and(|n| job.sub_done % n == 0) {
            gpu_report::log_memory(gpu, "export: GPU memory");
        }
        if lose_at == Some(job.sub_done) {
            tracing::warn!(
                frames = job.sub_done,
                "MC_RD_EXPORT: destroying the GPU device (lose=)"
            );
            gpu.device.destroy();
        }
        let finished = job.sub_done >= job.sub_total;
        if let Some(message) = self.nw_rd_export_health(gpu, failed_readbacks, refused_at_start) {
            // For the crash report of the shell's stop next frame.
            if gpu.is_device_lost() {
                self.tw_export.lost_export = Some(message.clone());
            }
            self.tw_export_fail(message);
            return;
        }
        if finished {
            gpu_report::log_memory(gpu, "export: GPU memory at the end of the pass");
            self.tw_export_finish_pass(gpu);
        }
    }

    /// `options=`: one step per frame once the replay is ready. Seek to
    /// `start`, picture the replay with its controls, open the game's options over it (and the
    /// page asked for), picture it, go back with Done, picture the replay again, then quit (with
    /// `exit=1`).
    fn nw_rd_options_pictures(&mut self) {
        let settled = self.replay_terrain_settled();
        let Some(Some(harness)) = self.tw_export.harness.as_mut() else {
            return;
        };
        harness.step_frames += 1;
        let (step, frames, offset) = (harness.options_step, harness.step_frames, harness.offset);
        let page = harness.options.clone().unwrap_or_default();
        let dir = harness.shots.clone().unwrap_or_else(std::env::temp_dir);
        let advance = |harness: &mut Harness, shot: Option<String>| {
            harness.options_step += 1;
            harness.step_frames = 0;
            if let Some(name) = shot {
                harness.pending_shot = Some(dir.join(format!("{name}.png")));
            }
        };
        match step {
            0 => {
                advance(harness, None);
                if offset > 0 {
                    let base = self.replay_view_range().0;
                    self.replay_seek(base + offset);
                }
            }
            1 if frames >= 120 && settled => {
                advance(harness, Some("1-viewer".to_owned()));
            }
            2 if frames >= 5 => {
                advance(harness, None);
                self.nw_rd_open_replay_options();
                use mc_ui::screens::options::OptionsPage;
                match page.as_str() {
                    "video" => self
                        .handle_action(mc_ui::screens::Action::OpenOptionsPage(OptionsPage::Video)),
                    "chat" => self
                        .handle_action(mc_ui::screens::Action::OpenOptionsPage(OptionsPage::Chat)),
                    _ => {}
                }
            }
            3 if frames >= 40 => advance(harness, Some(format!("2-options-{page}"))),
            4 if frames >= 5 => {
                advance(harness, None);
                // Done until the replay is back (a page goes back to the options first).
                for _ in 0..4 {
                    if matches!(self.ui, super::super::Ui::InGame) {
                        break;
                    }
                    self.go_back();
                }
            }
            5 if frames >= 40 => advance(harness, Some("3-back-to-the-replay".to_owned())),
            6 if frames >= 5 => {
                harness.started = true;
                let exit = harness.exit;
                tracing::info!(
                    ui_in_game = matches!(self.ui, super::super::Ui::InGame),
                    replay = self.replay_is_active(),
                    "MC_RD_EXPORT: options pictures done"
                );
                if exit {
                    self.request_quit_after_export();
                }
            }
            _ => {}
        }
    }

    /// Save this frame (the options pictures).
    fn nw_rd_save_shot(&mut self, gpu: &Gpu, frame: &AcquiredFrame) {
        let Some(Some(harness)) = self.tw_export.harness.as_mut() else {
            return;
        };
        let Some(path) = harness.pending_shot.take() else {
            return;
        };
        let (w, h) = (gpu.config.width, gpu.config.height);
        match frame.read_pixels(gpu) {
            Ok(rgba) => {
                let _ = std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")));
                let saved = image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8);
                tracing::info!(
                    file = %nw_rq_file_name(&path),
                    ok = saved.is_ok(),
                    "MC_RD_EXPORT: picture"
                );
            }
            Err(error) => tracing::warn!(%error, "MC_RD_EXPORT: picture failed"),
        }
    }

    /// `ShellApp::offscreen_failed`: the export's size did not fit the GPU.
    pub(crate) fn tw_export_offscreen_failed(&mut self, error: &str) {
        let Some(job) = self.tw_export.job.as_deref() else {
            return;
        };
        let message = gpu_report::offscreen_failure(error, (job.request.width, job.request.height));
        self.tw_export_fail(message);
    }

    /// `ShellApp::device_lost`: the shell stops on a lost device at the start of a
    /// frame, before [`Self::tw_export_frame`]'s own check can run (a friend's AMD export crash
    /// stopped that way, and its crash report never named the fix). A running export fails here,
    /// its unfinished file deleted, and the crash report gets the export's message.
    pub(crate) fn tw_export_device_lost(&mut self, reason: &str) -> Option<String> {
        if self.tw_export.job.is_none() {
            // The export's own check saw the loss first (last frame) and already ended it.
            return self.tw_export.lost_export.take();
        }
        let message = gpu_report::export_device_lost_text(reason);
        self.tw_export_fail(message.clone());
        Some(message)
    }

    /// Whether the running export can go on: `Some(message)` when the device is
    /// lost, readbacks failed or the terrain was refused memory.
    fn nw_rd_export_health(
        &self,
        gpu: &Gpu,
        failed_readbacks: u64,
        refused_at_start: u64,
    ) -> Option<String> {
        if gpu.is_device_lost() {
            return Some(gpu_report::device_lost_message(gpu, true));
        }
        if failed_readbacks > 0 {
            // With the graphics card's encoder on the same chip, it may be why.
            let hint = match self.tw_export.job.as_deref() {
                Some(job) if job.hardware_encoder => gpu_report::PROCESSOR_ONLY_HINT,
                _ => "",
            };
            return Some(
                format!(
                    "The graphics card could not hand back {failed_readbacks} exported frame(s). \
                     Nothing was saved. {hint}"
                )
                .trim_end()
                .to_owned(),
            );
        }
        let refused = self.nw_rd_refused_uploads();
        if refused > refused_at_start {
            return Some(
                "The graphics card ran out of memory for the world. Try a smaller size, a lower \
                 render distance or turning shader packs off. Nothing was saved."
                    .to_owned(),
            );
        }
        None
    }

    /// The terrain renderer's refused uploads so far (0 without a world).
    fn nw_rd_refused_uploads(&self) -> u64 {
        self.render
            .as_ref()
            .map_or(0, |render| render.terrain.borrow().refused_uploads())
    }

    /// The device is lost (a driver reset or hang): end a running export with the
    /// reason and stop the client cleanly. Nothing can be drawn any more, and before this the
    /// next buffer creation panicked somewhere random (the AMD export crash).
    fn nw_rd_device_lost(&mut self, gpu: &Gpu) {
        // The crash report says what the export window would have: the fix.
        let exporting = self.tw_export.job.is_some() || self.tw_export.lost_export.is_some();
        if self.tw_export.job.is_some() {
            self.tw_export_fail(gpu_report::device_lost_message(gpu, true));
        }
        if self.fatal.is_none() {
            let message = gpu_report::device_lost_message(gpu, exporting);
            tracing::error!(%message, "stopping: the GPU device was lost");
            self.fatal = Some(super::super::ClientError::Io(std::io::Error::other(
                message,
            )));
        }
    }

    /// Every rendered frame is captured: wait for the readbacks, then let the encoder finish
    /// the file on a thread of its own.
    fn tw_export_finish_pass(&mut self, gpu: &Gpu) {
        let Some(job) = self.tw_export.job.as_deref_mut() else {
            return;
        };
        let size = (job.request.width, job.request.height);
        let Job {
            capture,
            encoder,
            grabs,
            raw,
            ..
        } = &mut *job;
        let mut failure = None;
        if let Some(capture) = capture.as_mut() {
            capture.finish(gpu, &mut |index, nv12: &[u8]| {
                deliver(encoder, grabs, raw, size, index, nv12, &mut failure);
            });
            tracing::info!(stats = ?capture.stats(), "export: pass rendered");
        }
        if let Some(error) = failure {
            let hardware = job.hardware_encoder;
            self.tw_export_fail(gpu_report::encoder_failure(
                "The encoder stopped",
                &error,
                hardware,
            ));
            return;
        }
        if let Some(encoder) = job.encoder.take() {
            // A panic here reaches `tw_export_frame` through `join` ("the
            // finishing thread panicked": a failed export), so it is not the game's crash.
            job.finishing = std::thread::Builder::new()
                .name("export finish".to_owned())
                .spawn(move || mc_core::panic_scope::contain_for_join(move || encoder.finish()))
                .ok();
        }
        job.phase = ExportPhase::Finishing;
    }

    /// Once per frame (from `ShellApp::frame`): the `MC_RD_EXPORT` harness, and a finished
    /// encode's outcome (done, or another pass for a size limit).
    pub(crate) fn tw_export_frame(&mut self, gpu: &Gpu, input: &mc_render::InputState) {
        // A lost GPU device (driver reset or hang): stop cleanly with the driver's
        // reason. One atomic load per frame.
        if gpu.is_device_lost() {
            self.nw_rd_device_lost(gpu);
            return;
        }
        // A new frame: the render clock is read afresh by the first animation that asks.
        self.tw_export.frame = self.tw_export.frame.wrapping_add(1).max(1);
        if self.tw_export.job.is_none() {
            // Not exporting: two stores, plus the clip watch on the title screen only.
            self.tw_export.window = gpu.window_size();
            // Once per run (an `Option` test after that).
            if self.tw_export.adapter.is_none() {
                self.tw_export.adapter = Some(render_adapter(gpu));
            }
            if self.tw_export.clips.is_open() {
                let phase = self
                    .tw_export
                    .finished
                    .as_ref()
                    .map_or(ExportPhase::Cancelled, |progress| progress.phase.clone());
                self.tw_export_after_clip(&phase);
            }
            if matches!(self.ui, super::super::Ui::MainMenu(_)) {
                if input.button_down(0) || input.button_down(1) {
                    self.tw_export.clips.note_input();
                }
                self.tw_export_clips_idle(input.cursor());
            }
            self.tw_export_harness();
            return;
        }
        // A click stops an automatic clip export (keys go through `tw_export_key`).
        if self.tw_export.clips.automatic() && (input.button_down(0) || input.button_down(1)) {
            tracing::info!("automatic clip export stopped by a click");
            self.export_cancel();
            return;
        }
        let Some(job) = self.tw_export.job.as_deref_mut() else {
            return;
        };
        let done = job
            .finishing
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished);
        if !done {
            return;
        }
        let Some(handle) = job.finishing.take() else {
            return;
        };
        let hardware = job.hardware_encoder;
        let result = handle.join().unwrap_or_else(|_| {
            Err(VideoError::WorkerGone(
                "the finishing thread panicked".to_owned(),
            ))
        });
        match result {
            Ok(summary) => self.tw_export_pass_done(summary),
            Err(error) => self.tw_export_fail(gpu_report::encoder_failure(
                "The file could not be finished",
                &error,
                hardware,
            )),
        }
    }

    fn tw_export_pass_done(&mut self, summary: EncodeSummary) {
        let repass = matches!(
            self.tw_export.harness.as_ref(),
            Some(Some(harness)) if harness.repass
        );
        let Some(job) = self.tw_export.job.as_deref_mut() else {
            return;
        };
        tracing::info!(
            bytes = summary.bytes,
            frames = summary.frames,
            pass = job.pass,
            file = %nw_rq_file_name(&summary.path),
            "export: file written",
        );
        if let (ExportTarget::MaxBytes(max), Some(bitrate)) = (job.request.target, job.bitrate) {
            let seconds = (job.request.end - job.request.start) as f64 / 1e6;
            // `repass=1` (tests): pretend the first pass came out 20% over the limit.
            let measured = if repass && job.pass == 1 {
                summary.bytes.max(max + max / 5)
            } else {
                summary.bytes
            };
            let retry = size::retry(max, measured, bitrate, seconds, job.frames_total, job.aac);
            if let (Some(next), true) = (
                retry,
                job.pass < size::MAX_PASSES && job.clock == Clock::Replay,
            ) {
                tracing::info!(
                    over = summary.bytes,
                    max,
                    next,
                    "export: over the size limit; another pass"
                );
                // The next pass replaces it; a cancelled pass must not leave an over-size file.
                let _ = std::fs::remove_file(&summary.path);
                job.pass += 1;
                job.bitrate = Some(next);
                if let Err(error) = self.tw_export_open_encoder() {
                    self.tw_export_fail(error);
                }
                return;
            }
        }
        job.phase = ExportPhase::Done {
            path: summary.path,
            bytes: summary.bytes,
        };
        self.tw_export_end();
    }

    /// `MC_RD_EXPORT`: start once the world is ready.
    fn tw_export_harness(&mut self) {
        if self.tw_export.harness.is_none() {
            self.tw_export.harness = Some(Harness::from_env());
        }
        let waiting = matches!(
            self.tw_export.harness.as_ref(),
            Some(Some(harness)) if !harness.started
        );
        if !waiting || self.tw_export.job.is_some() {
            return;
        }
        // `MC_RD_REPLAY`: open the replay once the title screen is up.
        let replay = match self.tw_export.harness.as_mut() {
            Some(Some(harness)) => harness.replay.take(),
            _ => None,
        };
        if let Some(path) = replay {
            let title_up = matches!(self.ui, super::super::Ui::MainMenu(_))
                && self.tw_loading_frame_cap().is_none();
            let with_viewer = matches!(
                self.tw_export.harness.as_ref(),
                Some(Some(harness)) if harness.options.is_some()
            );
            // File names only in logs (players share them).
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            if title_up && with_viewer {
                // With the viewer's controls, as the browser's Watch opens it.
                self.replay_view_watch_file(&path);
                if !self.replay_is_active() {
                    self.nw_rq_harness_give_up(&name);
                    return;
                }
                tracing::info!(file = %name, "MC_RD_REPLAY: watching");
            } else if title_up {
                match self.replay_open(&path) {
                    Ok(()) => tracing::info!(file = %name, "MC_RD_REPLAY: opened"),
                    Err(error) => {
                        tracing::warn!(%error, "MC_RD_REPLAY: could not open");
                        self.nw_rq_harness_give_up(&name);
                        return;
                    }
                }
            } else if let Some(Some(harness)) = self.tw_export.harness.as_mut() {
                // Not on the title screen yet (or the loading screen still covers it): try again
                // next frame.
                harness.replay = Some(path);
                return;
            }
        }
        let ready =
            (self.replay_is_active() || self.session.is_some()) && self.replay_terrain_settled();
        let Some(Some(harness)) = self.tw_export.harness.as_mut() else {
            return;
        };
        if !ready {
            return;
        }
        harness.waited += 1;
        if harness.waited < harness.wait_frames {
            return;
        }
        if harness.options.is_some() {
            // The options pictures instead of an export.
            self.nw_rd_options_pictures();
            return;
        }
        harness.started = true;
        let mut request = harness.request.clone();
        let seconds = harness.seconds;
        let grabs = harness.grabs.clone();
        let raw = harness.raw.clone();
        let offset = harness.offset;
        let camera = harness.camera;
        // A series writes one file per export.
        if offset > 0 || !harness.more_starts.is_empty() {
            let stem = request
                .path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            request.path = request
                .path
                .with_file_name(format!("{stem}-at{}s.mp4", offset / 1_000_000));
        }
        if let Some(mode) = camera {
            self.replay_set_camera_mode(mode);
        }
        let base = if self.replay_is_active() {
            self.replay_view_range().0
        } else {
            0
        };
        request.start = base + offset;
        request.end = request.start + (seconds * 1e6) as u64;
        if let Err(error) = self.export_start(request) {
            tracing::warn!(%error, "MC_RD_EXPORT: could not start");
            return;
        }
        if let (Some(path), Some(job)) = (raw, self.tw_export.job.as_deref_mut()) {
            job.raw = std::fs::File::create(&path)
                .ok()
                .map(|file| std::io::BufWriter::with_capacity(1 << 22, file));
        }
        if let (Some(dir), Some(job)) = (grabs, self.tw_export.job.as_deref_mut()) {
            let _ = std::fs::create_dir_all(&dir);
            let last = job.frames_total.saturating_sub(1);
            job.grabs = [(0, "first"), (last / 2, "middle"), (last, "last")]
                .into_iter()
                .map(|(frame, name)| (frame, dir.join(format!("export_{name}_{frame}.png"))))
                .collect();
        }
    }

    /// `MC_RD_REPLAY` could not be opened: the library moved it into its date
    /// folder (renamed to local time), or it was renamed or deleted. The harness used to wait for
    /// a replay that would never open, so the client never quit (the runs hung twice). Stop
    /// waiting, and quit when `exit=1`, so a script sees the failure instead of hanging.
    fn nw_rq_harness_give_up(&mut self, file: &str) {
        let exit = match self.tw_export.harness.as_mut() {
            Some(Some(harness)) => {
                harness.started = true;
                harness.exit
            }
            _ => false,
        };
        tracing::warn!(
            file,
            "MC_RD_EXPORT: the replay could not be opened (moved, renamed or deleted?); nothing \
             to export"
        );
        if exit {
            self.request_quit_after_export();
        }
    }

    /// `ShellApp::wants_window_preview`.
    pub(crate) fn tw_export_wants_preview(&self) -> bool {
        self.tw_export.job.as_deref().is_some_and(|job| {
            job.last_preview
                .is_none_or(|last| last.elapsed() >= PREVIEW_INTERVAL)
        })
    }

    /// `ShellApp::draw_window_preview`: the latest frame, letterboxed, with the progress panel.
    pub(crate) fn tw_export_draw_preview(
        &mut self,
        gpu: &Gpu,
        frame: &AcquiredFrame,
        window: &AcquiredFrame,
    ) {
        let state = &mut self.tw_export;
        let Some(job) = state.job.as_deref_mut() else {
            return;
        };
        job.last_preview = Some(Instant::now());
        let Some(texture) = frame.color_texture(gpu) else {
            return;
        };
        let window_size = gpu.window_size();
        let text = panel_text(job);
        let compositor = job
            .preview
            .get_or_insert_with(|| PreviewCompositor::new(gpu, gpu.config.format));
        let stale = job
            .panel
            .as_ref()
            .is_none_or(|(old, height)| *old != text || *height != window_size.1);
        if stale {
            let canvas = panel::draw(&text, window_size.1);
            compositor.set_panel(gpu, canvas.width, canvas.height, &canvas.pixels);
            job.panel = Some((text, window_size.1));
        }
        let k = panel::scale(window_size.1);
        let (panel_w, panel_h) = (560.0 * k, 132.0 * k);
        let panel_at = (
            ((window_size.0 as f32 - panel_w) * 0.5).round(),
            (window_size.1 as f32 - panel_h - 32.0 * k).round(),
        );
        compositor.draw(
            gpu,
            texture,
            (gpu.config.width, gpu.config.height),
            window.view(),
            window_size,
            PreviewLayout {
                dim: 0.9,
                panel_at: Some(panel_at),
            },
        );
        // `MC_RD_EXPORT`'s `preview`: one picture of the window, halfway through.
        let halfway = job.sub_done * 2 >= job.sub_total && job.sub_total > 0;
        if let (Some(Some(harness)), true) = (state.harness.as_mut(), halfway) {
            if let (Some(path), Some(image)) = (harness.preview.take(), window.color_texture(gpu)) {
                match mc_render::export::read_rgba(gpu, image, window_size, gpu.config.format) {
                    Some(rgba) => {
                        let saved = image::save_buffer(
                            &path,
                            &rgba,
                            window_size.0,
                            window_size.1,
                            image::ColorType::Rgba8,
                        );
                        tracing::info!(
                            file = %nw_rq_file_name(&path),
                            ok = saved.is_ok(),
                            "export: window preview saved"
                        );
                    }
                    None => tracing::warn!("export: window preview readback failed"),
                }
            }
        }
    }

    /// `ShellApp::on_key_event`, first: while exporting every key is the export's; Esc cancels.
    pub(crate) fn tw_export_key(&mut self, code: winit::keyboard::KeyCode, pressed: bool) -> bool {
        if self.tw_export.job.is_none() {
            // The idle title screen's clip export waits for a quiet moment: note key presses there.
            if pressed && matches!(self.ui, super::super::Ui::MainMenu(_)) {
                self.tw_export.clips.note_input();
            }
            return false;
        }
        let automatic = self.tw_export.clips.automatic();
        if pressed && (code == winit::keyboard::KeyCode::Escape || automatic) {
            tracing::info!(?code, automatic, "export cancelled by a key");
            self.export_cancel();
        }
        true
    }
}

/// One finished output frame: to the encoder, and to the `MC_RD_EXPORT` grabs and raw dump.
fn deliver(
    encoder: &mut Option<EncoderThread>,
    grabs: &mut Vec<(u64, PathBuf)>,
    raw: &mut Option<std::io::BufWriter<std::fs::File>>,
    (w, h): (u32, u32),
    index: u64,
    nv12: &[u8],
    failure: &mut Option<String>,
) {
    if let Some(encoder) = encoder.as_mut() {
        let mut buffer = encoder.buffer();
        buffer.copy_from_slice(nv12);
        if let Err(error) = encoder.submit(buffer) {
            failure.get_or_insert(error.to_string());
        }
    }
    if let Some(raw) = raw.as_mut() {
        use std::io::Write as _;
        let _ = raw.write_all(nv12);
    }
    if let Some(at) = grabs.iter().position(|(frame, _)| *frame == index) {
        let (_, path) = grabs.swap_remove(at);
        let rgba = mc_video::nv12::to_rgba(nv12, w, h);
        if let Err(error) = image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8) {
            tracing::warn!(%error, file = %nw_rq_file_name(&path), "export: frame grab not saved");
        }
    }
}

/// A listener at `eye` looking along `yaw`/`pitch` (degrees), as `SoundManager.setListener`
/// builds it.
fn listener_from(eye: mc_math::DVec3, yaw: f32, pitch: f32) -> mc_audio::Listener {
    let mut listener = mc_audio::Listener::interpolated(eye, eye, 0.0, pitch, pitch, yaw, yaw, 1.0);
    listener.position = eye;
    listener
}

fn panel_text(job: &Job) -> panel::PanelText {
    let progress = job.progress();
    let request = &job.request;
    let fps = request.fps.as_f64();
    let fps_label = if (fps - fps.round()).abs() < 1e-6 {
        format!("{} fps", fps.round() as u64)
    } else {
        format!("{fps:.2} fps")
    };
    let mut detail = format!(
        "{}\u{00D7}{} \u{00B7} {} \u{00B7} {}",
        request.width,
        request.height,
        fps_label,
        if job.encoder_name.is_empty() {
            request.codec.label().to_owned()
        } else {
            format!(
                "{} {}",
                request.codec.label(),
                if job.encoder_name.ends_with("(GPU)") {
                    "(GPU)"
                } else {
                    "(CPU)"
                }
            )
        }
    );
    if job.blur > 1 {
        detail.push_str(&format!(" \u{00B7} motion blur {}\u{00D7}", job.blur));
    }
    if let ExportTarget::MaxBytes(max) = request.target {
        detail.push_str(&format!(" \u{00B7} under {:.0} MB", max as f64 / 1e6));
    }
    let title = match (&job.phase, job.pass) {
        (ExportPhase::Starting | ExportPhase::Settling, 1) => "Getting the world ready".to_owned(),
        (ExportPhase::Starting | ExportPhase::Settling, pass) => {
            format!("Pass {pass}: making the file smaller")
        }
        (ExportPhase::Finishing, _) => "Finishing the file".to_owned(),
        (_, 1) => "Exporting video".to_owned(),
        (_, pass) => format!("Exporting video (pass {pass}, smaller file)"),
    };
    let mut status = Vec::new();
    if let Some(eta) = progress.eta {
        status.push(format!("{} left", panel::clock(eta.as_secs_f64())));
    }
    status.push(format!(
        "frame {} of {}",
        panel::thousands(progress.frames_done.min(progress.frames_total)),
        panel::thousands(progress.frames_total)
    ));
    status.push("Esc to cancel".to_owned());
    panel::PanelText {
        title,
        detail,
        status: status.join(" \u{00B7} "),
        fraction: progress.fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_map_to_sizes_and_blur() {
        assert_eq!(resolution_for_choice(2, (800, 600)), (1920, 1080));
        assert_eq!(resolution_for_choice(4, (800, 600)), (3840, 2160));
        assert_eq!(resolution_for_choice(0, (1365, 767)), (1364, 766));
        assert_eq!(blur_for_choice(0), 1);
        assert_eq!(blur_for_choice(3), 8);
        assert_eq!(blur_for_choice(5), 32);
    }

    #[test]
    fn file_names_are_safe() {
        assert_eq!(
            safe_file_name("mc.example.net: SkyWars?"),
            "mc.example.net_ SkyWars_"
        );
        assert_eq!(safe_file_name("  ..  "), "Replay");
        assert_eq!(safe_file_name("a/b\\c"), "a_b_c");
    }

    /// A replay's title is the file's own: a device name never becomes
    /// the export's name, and ordinary titles are unchanged.
    #[test]
    fn a_device_name_title_never_names_the_export() {
        for title in [
            "NUL",
            "nul",
            "Con",
            "aux",
            "PRN",
            "COM1",
            "com9",
            "LPT1",
            "lpt0",
            "COM\u{b9}",
            "CONOUT$",
            "NUL.tar",
            "con .x",
            "  NUL  ",
        ] {
            let name = safe_file_name(title);
            assert!(name.starts_with('_'), "{title:?} -> {name:?}");
            assert!(
                !mc_assets::is_windows_device_name(&name),
                "{title:?} -> {name:?}"
            );
        }
        for title in [
            "Sky Wars \u{b7} Nov 12",
            "CONSOLE",
            "COM10",
            "LPT",
            "null",
            "Replay",
            "nul_",
        ] {
            assert_eq!(safe_file_name(title), title, "{title:?}");
        }
    }
}
