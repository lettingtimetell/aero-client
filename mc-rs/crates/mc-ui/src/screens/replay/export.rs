//! The export window: what part, how big, how smooth, how heavy, which sounds, what name, then
//! a compact progress panel (so the live preview behind it stays visible) with the time left and
//! Cancel, and at the end "Show in folder".
//!
//! One screen of choices, each a row of plain buttons rather than a hidden list: **Range**
//! (the selection or the whole replay), **Resolution**, **Frame rate** (common rates up to 2000
//! as buttons, and a box that takes any number), **Quality**, **Max file size** (common upload
//! limits called out), **Format** (H.264 or H.265), **Motion blur** (frames blended into each
//! one), **Sound** (game sound, microphone) and the **File name**. A line at the bottom adds it
//! up ("12.7 s · 762 frames · about 18 MB"). The defaults come from the `replay` mod's export
//! settings (or its clip settings for a one-click clip export).

use std::time::Instant;

use mc_mods::mods::replay as options;

use super::format;
use super::icons::{self, Icon};
use super::style::{self, ms_since, styles, Fonts, Hits, Look, Tip, TipClock};
use super::{key, ReplayEnv};
use crate::screens::modmenu::anim;
use crate::screens::modmenu::glass::{GlassFrame, GlassRect};
use crate::screens::modmenu::paint::{white, Painter, Rect};
use crate::screens::modmenu::text::{TextStyle, Weight};
use crate::widget::keys::Modifiers;
use crate::widget::DrawList;

const WINDOW_W: f32 = 700.0;
const RADIUS: f32 = 25.0;
const TITLE_H: f32 = 78.0;
const FOOTER_H: f32 = 68.0;
const LABEL_W: f32 = 168.0;
const PAD: f32 = 26.0;
const ROW: f32 = 50.0;
const HINT: f32 = 17.0;
const CONTROL_H: f32 = 34.0;
const PANEL_W: f32 = 640.0;
const PANEL_H: f32 = 176.0;
/// The frame-rate buttons (the box takes any other number).
pub const FPS_PRESETS: [u32; 6] = [30, 60, 120, 240, 1000, 2000];
/// The highest frame rate the box takes.
pub const FPS_MAX: u32 = 9999;
/// The longest file name the box keeps.
pub const NAME_MAX_CHARS: usize = 80;
/// Short labels for the resolution buttons ([`options::RESOLUTION_CHOICES`] order).
const RESOLUTION_LABELS: [&str; 5] = ["Window", "720p", "1080p", "1440p", "4K"];
/// Short labels for the quality buttons ([`options::EXPORT_QUALITY_CHOICES`] order).
const QUALITY_LABELS: [&str; 4] = ["Small", "Balanced", "High", "Best"];
/// Short labels for the size buttons ([`options::CLIP_MAX_SIZE_CHOICES`] order).
const SIZE_LABELS: [&str; 7] = [
    "No limit", "8 MB", "10 MB", "25 MB", "50 MB", "100 MB", "500 MB",
];
/// Frames blended per exported frame ([`options::EXPORT_MOTION_BLUR_CHOICES`] order).
pub const BLEND_FRAMES: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// A rough video bitrate per quality at 1080p and 60 fps, in megabits per second (only for the
/// size estimate; the encoder picks the real rate).
const QUALITY_MBPS: [f32; 4] = [4.0, 8.0, 16.0, 45.0];

/// Everything the form asks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSettings {
    /// Export the selection (`true`) or the whole replay.
    pub selection: bool,
    /// An index into [`options::RESOLUTION_CHOICES`] (0 is the window's size).
    pub resolution: usize,
    /// Frames per second, `1..=FPS_MAX`.
    pub fps: u32,
    /// An index into [`options::EXPORT_QUALITY_CHOICES`].
    pub quality: usize,
    /// An index into [`options::CLIP_MAX_SIZE_CHOICES`] (0 is no limit).
    pub max_size: usize,
    /// H.265 rather than H.264.
    pub h265: bool,
    /// An index into [`options::EXPORT_MOTION_BLUR_CHOICES`] (0 is off).
    pub motion_blur: usize,
    /// Include the game's sound.
    pub game_audio: bool,
    /// Include the microphone.
    pub mic: bool,
    /// Include desktop audio (other programs' sound, such as a voice call).
    pub desktop: bool,
    /// One more audio track per sound after the mixed one, for video editors.
    pub separate_tracks: bool,
    /// The file name without `.mp4`.
    pub file_name: String,
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            selection: false,
            resolution: 2,
            fps: 60,
            quality: 2,
            max_size: 0,
            h265: false,
            motion_blur: 0,
            game_audio: true,
            mic: true,
            desktop: true,
            separate_tracks: false,
            file_name: "Replay".to_owned(),
        }
    }
}

impl ExportSettings {
    /// The output size for a `window`-sized game window.
    #[must_use]
    pub fn size(&self, window: (u32, u32)) -> (u32, u32) {
        match self.resolution {
            1 => (1280, 720),
            2 => (1920, 1080),
            3 => (2560, 1440),
            4 => (3840, 2160),
            // The window, rounded down to even numbers (H.264 needs them).
            _ => ((window.0.max(2)) & !1, (window.1.max(2)) & !1),
        }
    }

    /// The byte limit, if any.
    #[must_use]
    pub fn max_bytes(&self) -> Option<u64> {
        options::CLIP_MAX_SIZE_BYTES
            .get(self.max_size)
            .copied()
            .flatten()
    }

    /// Frames blended into each exported frame (1 = no motion blur).
    #[must_use]
    pub fn blend_frames(&self) -> u32 {
        BLEND_FRAMES.get(self.motion_blur).copied().unwrap_or(1)
    }

    /// A rough size for `duration_us` of video, in bytes.
    #[must_use]
    pub fn estimate_bytes(&self, duration_us: u64, window: (u32, u32)) -> u64 {
        let (w, h) = self.size(window);
        let pixels = (w as f32 * h as f32) / (1920.0 * 1080.0);
        let rate = (self.fps as f32 / 60.0).sqrt();
        let codec = if self.h265 { 0.6 } else { 1.0 };
        let mbps = QUALITY_MBPS.get(self.quality).copied().unwrap_or(16.0) * pixels * rate * codec;
        let channels = [self.game_audio, self.mic, self.desktop];
        let audio_kbps = if channels.contains(&true) {
            // Each separate track is another AAC stream at about the mix's rate, and
            // the mix's copy at the end (for Windows' players) is one more.
            let extra = if self.separate_tracks {
                channels.iter().filter(|&&on| on).count() as f32 + 1.0
            } else {
                0.0
            };
            192.0 + 128.0 * extra
        } else {
            0.0
        };
        let seconds = duration_us as f32 / 1e6;
        let bytes = ((mbps * 1e6 + audio_kbps * 1e3) / 8.0 * seconds) as u64;
        match self.max_bytes() {
            Some(limit) => bytes.min(limit),
            None => bytes,
        }
    }
}

/// What the client should export (the form, resolved).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRequest {
    /// Where to start, microseconds on the timeline.
    pub start_us: u64,
    /// Where to stop.
    pub end_us: u64,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Frames per second.
    pub fps: u32,
    /// An index into [`options::EXPORT_QUALITY_CHOICES`].
    pub quality: usize,
    /// Stay under this many bytes.
    pub max_bytes: Option<u64>,
    /// H.265 rather than H.264.
    pub h265: bool,
    /// Frames rendered and blended per exported frame (1 = no motion blur).
    pub blend_frames: u32,
    /// Include the game's sound.
    pub game_audio: bool,
    /// Include the microphone.
    pub mic: bool,
    /// Include desktop audio.
    pub desktop: bool,
    /// One more audio track per included sound after the mixed one.
    pub separate_tracks: bool,
    /// The file name, `.mp4` included.
    pub file_name: String,
}

/// Where an export job is.
#[derive(Debug, Clone, PartialEq)]
pub enum ExportState {
    /// Rendering and encoding.
    Running,
    /// Writing the file's end (or re-encoding to fit the size limit).
    Finishing,
    /// Written.
    Done {
        /// The file.
        path: String,
        /// Its size.
        bytes: u64,
    },
    /// It failed, with a reason for the player.
    Failed(String),
    /// The player cancelled it.
    Cancelled,
}

/// An export job's progress, reported by the client every frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportProgress {
    /// Frames written.
    pub frames_done: u64,
    /// Frames in the whole export.
    pub frames_total: u64,
    /// Seconds since it started.
    pub elapsed_s: f32,
    /// Where it is.
    pub state: ExportState,
}

/// What the export window asks the client to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportEvent {
    /// Start this export.
    Start(ExportRequest),
    /// Stop the running export.
    Cancel,
    /// Close the window (the form, or a finished export).
    Close,
    /// Show the written file in Explorer.
    ShowFile(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    Window,
    Range(bool),
    Resolution(usize),
    FpsPreset(u32),
    FpsField,
    Quality(usize),
    MaxSize(usize),
    Codec(bool),
    Blur(usize),
    GameAudio,
    Mic,
    Desktop,
    Tracks,
    NameField,
    Export,
    Cancel,
    Done,
    ShowFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    None,
    Fps,
    Name,
}

/// The export window. The viewer holds one while it is open.
#[derive(Debug)]
pub struct ExportDialog {
    /// The form's values.
    pub settings: ExportSettings,
    opened_at: Instant,
    last_frame: Option<Instant>,
    /// The selection, if there is one: `(in, out)` in microseconds.
    selection: Option<(u64, u64)>,
    /// The whole timeline's length.
    duration_us: u64,
    window: (u32, u32),
    has_mic: bool,
    /// The replay holds desktop audio ([`ExportDialog::set_has_desktop`]).
    has_desktop: bool,
    clip: bool,
    /// This PC can encode H.265.
    h265_available: bool,
    /// The fps box's text while it is being typed in.
    fps_text: String,
    focus: Focus,
    /// The progress of the job this window started.
    progress: Option<ExportProgress>,
    /// When the job's progress view appeared.
    running_since: Option<Instant>,
    /// A smoothed frames-per-second rate for the time left.
    rate: Option<f32>,
    hits: Hits<Target>,
    tips: TipClock,
    cursor: (f32, f32),
    pressed: Option<Target>,
    events: Vec<ExportEvent>,
}

impl ExportDialog {
    /// A form for a timeline `duration_us` long with this `selection`, starting from
    /// `settings`. `window` is the game window's size (the "Window" resolution).
    #[must_use]
    pub fn new(
        settings: ExportSettings,
        selection: Option<(u64, u64)>,
        duration_us: u64,
        window: (u32, u32),
        has_mic: bool,
        clip: bool,
        now: Instant,
    ) -> ExportDialog {
        let mut settings = settings;
        settings.selection = selection.is_some();
        settings.fps = settings.fps.clamp(1, FPS_MAX);
        if !has_mic {
            settings.mic = false;
        }
        ExportDialog {
            fps_text: settings.fps.to_string(),
            settings,
            opened_at: now,
            last_frame: None,
            selection,
            duration_us,
            window,
            has_mic,
            has_desktop: false,
            clip,
            h265_available: true,
            focus: Focus::None,
            progress: None,
            running_since: None,
            rate: None,
            hits: Hits::default(),
            tips: TipClock::default(),
            cursor: (-1.0, -1.0),
            pressed: None,
            events: Vec::new(),
        }
    }

    /// The events since the last call.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, ExportEvent> {
        self.events.drain(..)
    }

    /// Whether this PC can encode H.265; when it cannot, the choice is greyed out and the form
    /// falls back to H.264.
    pub fn set_h265_available(&mut self, available: bool) {
        self.h265_available = available;
        if !available {
            self.settings.h265 = false;
        }
    }

    /// Whether the replay holds desktop audio. Without it the switch is greyed out and
    /// off.
    pub fn set_has_desktop(&mut self, has: bool) {
        self.has_desktop = has;
        if !has {
            self.settings.desktop = false;
        }
    }

    /// The frame-rate box in the last frame (where a click starts typing a rate).
    #[must_use]
    pub fn fps_field_rect(&self) -> Option<Rect> {
        self.hits.rect_of(Target::FpsField)
    }

    /// Whether a job is running or finished (the progress view shows).
    #[must_use]
    pub fn started(&self) -> bool {
        self.progress.is_some()
    }

    /// Whether the job is still going.
    #[must_use]
    pub fn running(&self) -> bool {
        matches!(
            self.progress.as_ref().map(|p| &p.state),
            Some(ExportState::Running | ExportState::Finishing)
        )
    }

    /// Whether a text box has the keyboard.
    #[must_use]
    pub fn typing(&self) -> bool {
        self.focus != Focus::None
    }

    /// The range the form would export: `(start, end)` microseconds.
    #[must_use]
    pub fn range(&self) -> (u64, u64) {
        match (self.settings.selection, self.selection) {
            (true, Some(range)) => range,
            _ => (0, self.duration_us),
        }
    }

    /// The request the form describes.
    #[must_use]
    pub fn request(&self) -> ExportRequest {
        let (start_us, end_us) = self.range();
        let (width, height) = self.settings.size(self.window);
        let stem = format::file_stem(&self.settings.file_name);
        ExportRequest {
            start_us,
            end_us,
            width,
            height,
            fps: self.settings.fps,
            quality: self.settings.quality,
            max_bytes: self.settings.max_bytes(),
            h265: self.settings.h265,
            blend_frames: self.settings.blend_frames(),
            game_audio: self.settings.game_audio,
            mic: self.settings.mic && self.has_mic,
            desktop: self.settings.desktop && self.has_desktop,
            separate_tracks: self.settings.separate_tracks,
            file_name: format!("{stem}.mp4"),
        }
    }

    /// Start at once (a one-click clip export): the window goes straight to its progress view
    /// and the client gets [`ExportEvent::Start`].
    pub fn start_now(&mut self, now: Instant) {
        self.start(now);
    }

    fn start(&mut self, now: Instant) {
        if self.focus == Focus::Fps {
            self.commit_fps();
        }
        let request = self.request();
        let frames =
            (request.end_us.saturating_sub(request.start_us)) * u64::from(request.fps) / 1_000_000;
        self.progress = Some(ExportProgress {
            frames_done: 0,
            frames_total: frames.max(1),
            elapsed_s: 0.0,
            state: ExportState::Running,
        });
        self.running_since = Some(now);
        self.rate = None;
        self.focus = Focus::None;
        self.events.push(ExportEvent::Start(request));
    }

    /// The client's report on the job.
    pub fn set_progress(&mut self, progress: ExportProgress) {
        if progress.elapsed_s > 0.5 && progress.frames_done > 0 {
            let rate = progress.frames_done as f32 / progress.elapsed_s;
            self.rate = Some(match self.rate {
                Some(old) => old + (rate - old) * 0.1,
                None => rate,
            });
        }
        self.progress = Some(progress);
    }

    fn commit_fps(&mut self) {
        if let Ok(fps) = self.fps_text.trim().parse::<u32>() {
            self.settings.fps = fps.clamp(1, FPS_MAX);
        }
        self.fps_text = self.settings.fps.to_string();
    }

    // ---- input ----------------------------------------------------------------------------

    /// The pointer moved.
    pub fn pointer_moved(&mut self, x: f32, y: f32) {
        self.cursor = (x, y);
    }

    /// A mouse button went down; returns whether it hit the window (the window is modal: the
    /// caller gives it every click while it is open).
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8) -> bool {
        self.cursor = (x, y);
        self.tips.suppress();
        let target = self.hits.at(x, y);
        if button != 0 {
            return target.is_some();
        }
        // A click outside a box ends its editing.
        match (self.focus, target) {
            (Focus::Fps, Some(Target::FpsField)) | (Focus::Name, Some(Target::NameField)) => {}
            (Focus::Fps, _) => {
                self.commit_fps();
                self.focus = Focus::None;
            }
            (Focus::Name, _) => self.focus = Focus::None,
            (Focus::None, _) => {}
        }
        self.pressed = target;
        target.is_some()
    }

    /// A mouse button went up.
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8, now: Instant) {
        self.cursor = (x, y);
        if button != 0 {
            return;
        }
        let Some(pressed) = self.pressed.take() else {
            return;
        };
        if self.hits.at(x, y) == Some(pressed) {
            self.activate(pressed, now);
        }
    }

    /// Paste into the focused box.
    pub fn paste(&mut self, text: &str) -> bool {
        match self.focus {
            Focus::Fps => {
                for c in text.chars().filter(char::is_ascii_digit) {
                    if self.fps_text.len() < 4 {
                        self.fps_text.push(c);
                    }
                }
                true
            }
            Focus::Name => {
                for c in text.chars().filter(|c| !c.is_control()) {
                    if self.settings.file_name.chars().count() < NAME_MAX_CHARS {
                        self.settings.file_name.push(c);
                    }
                }
                true
            }
            Focus::None => false,
        }
    }

    /// A key. Returns whether the window used it (it takes every key while open).
    pub fn key(&mut self, code: i32, typed: Option<char>, mods: Modifiers, now: Instant) -> bool {
        match self.focus {
            Focus::Fps => {
                match code {
                    key::ESCAPE => {
                        self.fps_text = self.settings.fps.to_string();
                        self.focus = Focus::None;
                    }
                    key::ENTER | key::NUMPAD_ENTER => {
                        self.commit_fps();
                        self.focus = Focus::None;
                    }
                    key::BACKSPACE => {
                        self.fps_text.pop();
                    }
                    _ => {
                        if let Some(c) = typed.filter(char::is_ascii_digit) {
                            if self.fps_text.len() < 4 && !mods.ctrl {
                                self.fps_text.push(c);
                            }
                        }
                    }
                }
                return true;
            }
            Focus::Name => {
                match code {
                    key::ESCAPE | key::ENTER | key::NUMPAD_ENTER => self.focus = Focus::None,
                    key::BACKSPACE => {
                        self.settings.file_name.pop();
                    }
                    _ => {
                        if let Some(c) = typed.filter(|c| !c.is_control()) {
                            if !mods.ctrl
                                && self.settings.file_name.chars().count() < NAME_MAX_CHARS
                            {
                                self.settings.file_name.push(c);
                            }
                        }
                    }
                }
                return true;
            }
            Focus::None => {}
        }
        match code {
            key::ESCAPE => {
                if self.running() {
                    // Esc never throws away a running export: it needs the Cancel button.
                } else {
                    self.events.push(ExportEvent::Close);
                }
            }
            key::ENTER | key::NUMPAD_ENTER if !self.started() => self.start(now),
            _ => {}
        }
        true
    }

    fn activate(&mut self, target: Target, now: Instant) {
        let s = &mut self.settings;
        match target {
            Target::Window => {}
            Target::Range(selection) => {
                if !selection || self.selection.is_some() {
                    s.selection = selection;
                }
            }
            Target::Resolution(i) => s.resolution = i,
            Target::FpsPreset(fps) => {
                s.fps = fps;
                self.fps_text = fps.to_string();
                self.focus = Focus::None;
            }
            Target::FpsField => {
                self.focus = Focus::Fps;
                self.fps_text.clear();
            }
            Target::Quality(i) => s.quality = i,
            Target::MaxSize(i) => s.max_size = i,
            Target::Codec(h265) => {
                if !h265 || self.h265_available {
                    s.h265 = h265;
                }
            }
            Target::Blur(i) => s.motion_blur = i,
            Target::GameAudio => s.game_audio = !s.game_audio,
            Target::Mic => {
                if self.has_mic {
                    s.mic = !s.mic;
                }
            }
            Target::Desktop => {
                if self.has_desktop {
                    s.desktop = !s.desktop;
                }
            }
            Target::Tracks => s.separate_tracks = !s.separate_tracks,
            Target::NameField => self.focus = Focus::Name,
            Target::Export => self.start(now),
            Target::Cancel => {
                if self.running() {
                    self.events.push(ExportEvent::Cancel);
                } else {
                    self.events.push(ExportEvent::Close);
                }
            }
            Target::Done => self.events.push(ExportEvent::Close),
            Target::ShowFile => {
                if let Some(ExportState::Done { path, .. }) =
                    self.progress.as_ref().map(|p| &p.state)
                {
                    self.events.push(ExportEvent::ShowFile(path.clone()));
                }
            }
        }
    }

    // ---- drawing --------------------------------------------------------------------------

    /// Draw one frame into `list`; the glass panes go into `glass`.
    pub fn build(&mut self, env: &mut ReplayEnv<'_>, list: &mut DrawList, glass: &mut GlassFrame) {
        self.hits.clear();
        let now = env.now;
        self.last_frame = Some(now);
        let s = style::scale(env.screen);
        let mut fonts = Fonts {
            text: &mut *env.text,
            mc: &mut *env.mc_font,
        };
        let mut painter = Painter::new(list);
        if self.progress.is_some() {
            self.draw_progress(&mut painter, &mut fonts, glass, env.screen, s, now);
        } else {
            self.draw_form(&mut painter, &mut fonts, glass, env.screen, s, now);
        }
        let hovered = self
            .hits
            .entry_at(self.cursor.0, self.cursor.1)
            .and_then(|(r, _, tip)| tip.map(|tip| (r, tip)));
        if let Some((anchor, tip, alpha)) = self.tips.update(hovered, now) {
            style::tooltip(
                &mut painter,
                &mut fonts,
                anchor,
                tip.label,
                tip.key,
                env.screen,
                alpha,
                s,
            );
        }
    }

    /// The form's height in design pixels.
    fn form_h(&self) -> f32 {
        TITLE_H + 14.0 + ROW * 8.0 + HINT * 3.0 + 64.0 + 34.0 + FOOTER_H
    }

    #[allow(clippy::too_many_lines)]
    fn draw_form(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        glass: &mut GlassFrame,
        screen: (f32, f32),
        s: f32,
        now: Instant,
    ) {
        let open = anim::window_open_progress(ms_since(self.opened_at, now), 0);
        if open <= 0.0 {
            return;
        }
        let dip = (screen.0 / s, screen.1 / s);
        let w = WINDOW_W.min(dip.0 - 40.0) * s;
        let h = self.form_h().min(dip.1 - 24.0) * s;
        let slide = ((1.0 - open) * anim::SLIDE_DIP * s).round();
        let rect = Rect::new(
            ((screen.0 - w) * 0.5).round(),
            ((screen.1 - h) * 0.5).round() + slide,
            w.round(),
            h.round(),
        );
        glass.rects.push(GlassRect {
            rect: [rect.x, rect.y, rect.w, rect.h],
            radius: RADIUS * s,
            alpha: open,
        });
        glass.dim = glass.dim.max(0.30 * open);
        painter.alpha = open;
        style::panel(painter, rect, RADIUS * s, TITLE_H * s, s);
        self.hits.push(rect, Target::Window, None);

        // ---- title ----
        let cy = rect.y + TITLE_H * s * 0.5;
        let pad = PAD * s;
        icons::draw(
            painter,
            Icon::Export,
            (rect.x + pad + 12.0 * s, cy),
            24.0 * s,
            1.9 * s,
            white(0.95),
        );
        let tx = rect.x + pad + 36.0 * s;
        fonts.shadowed(
            painter,
            "Export video",
            tx,
            cy - 21.0 * s,
            TextStyle::new(19.0 * s).weight(Weight::Semibold),
            white(1.0),
            s,
        );
        let (start, end) = self.range();
        let summary = format!(
            "{} to {}  \u{b7}  {}",
            format::clock_tenths(start),
            format::clock_tenths(end),
            format::span(end.saturating_sub(start))
        );
        fonts.shadowed(
            painter,
            &summary,
            tx,
            cy + 3.0 * s,
            styles::subtitle(s),
            white(0.62),
            s,
        );

        // ---- rows ----
        let label_x = rect.x + pad;
        let cx = rect.x + pad + LABEL_W * s;
        let cw = rect.right() - pad - cx;
        let mut y = rect.y + TITLE_H * s + 14.0 * s;
        let label = |fonts: &mut Fonts<'_>, painter: &mut Painter<'_>, text: &str, cy: f32| {
            fonts.shadowed_mid(painter, text, label_x, cy, styles::body(s), white(0.86), s);
        };
        let hint = |fonts: &mut Fonts<'_>, painter: &mut Painter<'_>, text: &str, y: f32| {
            fonts.shadowed(
                painter,
                text,
                cx + 4.0 * s,
                y,
                styles::small(s),
                white(0.55),
                s,
            );
        };

        // Range.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Range", row_cy);
        let control = Rect::new(cx, row_cy - CONTROL_H * s * 0.5, cw, CONTROL_H * s);
        let sel_label = match self.selection {
            Some((a, b)) => format!("Selection  {}", format::span(b.saturating_sub(a))),
            None => "Selection".to_owned(),
        };
        let whole_label = format!(
            "{}  {}",
            if self.clip {
                "Whole clip"
            } else {
                "Whole replay"
            },
            format::clock(self.duration_us)
        );
        self.segmented_row(
            painter,
            fonts,
            control,
            &[(None, sel_label.as_str()), (None, whole_label.as_str())],
            usize::from(!self.settings.selection),
            &[self.selection.is_none(), false],
            &[Target::Range(true), Target::Range(false)],
            &[
                Tip::new("Only the part between In and Out", Some("I / O")),
                Tip::new("From start to end", None),
            ],
            s,
        );
        y += ROW * s;

        // Resolution.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Resolution", row_cy);
        let control = Rect::new(cx, row_cy - CONTROL_H * s * 0.5, cw, CONTROL_H * s);
        let labels: Vec<(Option<icons::Icon>, &str)> =
            RESOLUTION_LABELS.iter().map(|l| (None, *l)).collect();
        let targets: Vec<Target> = (0..5).map(Target::Resolution).collect();
        self.segmented_row(
            painter,
            fonts,
            control,
            &labels,
            self.settings.resolution,
            &[],
            &targets,
            &[],
            s,
        );
        y += ROW * s;
        let (ow, oh) = self.settings.size(self.window);
        hint(
            fonts,
            painter,
            &format!("{ow} \u{d7} {oh} pixels"),
            y - 6.0 * s,
        );
        y += HINT * s;

        // Frame rate: preset buttons and a box for any number.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Frame rate", row_cy);
        let field_w = 104.0 * s;
        let presets = Rect::new(
            cx,
            row_cy - CONTROL_H * s * 0.5,
            cw - field_w - 10.0 * s,
            CONTROL_H * s,
        );
        let labels: Vec<String> = FPS_PRESETS.iter().map(u32::to_string).collect();
        let label_refs: Vec<(Option<icons::Icon>, &str)> =
            labels.iter().map(|l| (None, l.as_str())).collect();
        let selected = FPS_PRESETS
            .iter()
            .position(|f| *f == self.settings.fps && self.focus != Focus::Fps)
            .unwrap_or(usize::MAX);
        let targets: Vec<Target> = FPS_PRESETS.iter().map(|f| Target::FpsPreset(*f)).collect();
        self.segmented_row(
            painter,
            fonts,
            presets,
            &label_refs,
            selected,
            &[],
            &targets,
            &[],
            s,
        );
        let field = Rect::new(
            presets.right() + 10.0 * s,
            presets.y,
            field_w,
            CONTROL_H * s,
        );
        let caret = (ms_since(self.opened_at, now) as u64 / 530).is_multiple_of(2);
        let focused = self.focus == Focus::Fps;
        let text = if focused {
            self.fps_text.clone()
        } else {
            self.settings.fps.to_string()
        };
        style::text_field(painter, fonts, field, &text, "Any", focused, caret, s);
        let fps_style = styles::small(s);
        fonts.plain_mid(
            painter,
            "fps",
            field.right() - 30.0 * s,
            field.y + field.h * 0.5,
            fps_style,
            white(0.5),
        );
        self.hits.push(
            field,
            Target::FpsField,
            Some(Tip::new("Type any frame rate", None)),
        );
        y += ROW * s;

        // Quality.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Quality", row_cy);
        let control = Rect::new(cx, row_cy - CONTROL_H * s * 0.5, cw, CONTROL_H * s);
        let labels: Vec<(Option<icons::Icon>, &str)> =
            QUALITY_LABELS.iter().map(|l| (None, *l)).collect();
        let targets: Vec<Target> = (0..4).map(Target::Quality).collect();
        self.segmented_row(
            painter,
            fonts,
            control,
            &labels,
            self.settings.quality,
            &[],
            &targets,
            &[
                Tip::new("Smallest files", None),
                Tip::new("Good for sharing", None),
                Tip::new("Sharp, bigger files", None),
                Tip::new("Near lossless, for editing", None),
            ],
            s,
        );
        y += ROW * s;

        // Max file size.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Max file size", row_cy);
        let control = Rect::new(cx, row_cy - CONTROL_H * s * 0.5, cw, CONTROL_H * s);
        let labels: Vec<(Option<icons::Icon>, &str)> =
            SIZE_LABELS.iter().map(|l| (None, *l)).collect();
        let targets: Vec<Target> = (0..SIZE_LABELS.len()).map(Target::MaxSize).collect();
        self.segmented_row(
            painter,
            fonts,
            control,
            &labels,
            self.settings.max_size,
            &[],
            &targets,
            &[],
            s,
        );
        y += ROW * s;
        // With a limit, the encoder aims at the size: the quality adapts to fit it.
        let size_hint = match self.settings.max_size {
            0 => "The quality decides the size",
            2 => "Fits a 10 MB upload limit; the quality adapts",
            4 => "Fits a 50 MB upload limit; the quality adapts",
            6 => "Fits a 500 MB upload limit; the quality adapts",
            _ => "The video stays under this size; the quality adapts",
        };
        hint(fonts, painter, size_hint, y - 6.0 * s);
        y += HINT * s;

        // Format.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Format", row_cy);
        let control = Rect::new(cx, row_cy - CONTROL_H * s * 0.5, cw, CONTROL_H * s);
        let h265_label = if self.h265_available {
            "H.265"
        } else {
            "H.265 (not on this PC)"
        };
        self.segmented_row(
            painter,
            fonts,
            control,
            &[(None, "H.264"), (None, h265_label)],
            usize::from(self.settings.h265),
            &[false, !self.h265_available],
            &[Target::Codec(false), Target::Codec(true)],
            &[
                Tip::new("Plays everywhere", None),
                Tip::new("Smaller files; some players can't open them", None),
            ],
            s,
        );
        y += ROW * s;
        hint(
            fonts,
            painter,
            if self.settings.h265 {
                "Smaller files; some players and apps can't play them"
            } else {
                "Plays everywhere"
            },
            y - 6.0 * s,
        );
        y += HINT * s;

        // Motion blur.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Motion blur", row_cy);
        let control = Rect::new(cx, row_cy - CONTROL_H * s * 0.5, cw, CONTROL_H * s);
        let labels: Vec<(Option<icons::Icon>, &str)> = options::EXPORT_MOTION_BLUR_CHOICES
            .iter()
            .map(|l| (None, *l))
            .collect();
        let targets: Vec<Target> = (0..labels.len()).map(Target::Blur).collect();
        self.segmented_row(
            painter,
            fonts,
            control,
            &labels,
            self.settings.motion_blur,
            &[],
            &targets,
            &[],
            s,
        );
        y += ROW * s;

        // Sound.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "Sound", row_cy);
        let mut x = cx;
        // Four switches share the row (the window has no room for another row): the
        // three sound channels, then separate tracks. Short labels; the tips say the rest.
        for (name, on, target, enabled, tip) in [
            (
                "Game",
                self.settings.game_audio,
                Target::GameAudio,
                true,
                "The game's sound",
            ),
            (
                "Mic",
                self.settings.mic && self.has_mic,
                Target::Mic,
                self.has_mic,
                if self.has_mic {
                    "Your microphone"
                } else {
                    "This replay has no microphone track"
                },
            ),
            (
                "Desktop",
                self.settings.desktop && self.has_desktop,
                Target::Desktop,
                self.has_desktop,
                if self.has_desktop {
                    "Desktop audio: other programs, such as a voice call"
                } else {
                    "This replay has no desktop audio"
                },
            ),
            (
                "Tracks",
                self.settings.separate_tracks,
                Target::Tracks,
                true,
                "Separate tracks: one more audio track per sound after the mixed one, for \
                 video editors",
            ),
        ] {
            let sw = Rect::new(x, row_cy - 10.0 * s, 38.0 * s, 20.0 * s);
            let hit = Rect::new(x - 4.0 * s, row_cy - 16.0 * s, 116.0 * s, 32.0 * s);
            let hover = hit.contains(self.cursor.0, self.cursor.1) && enabled;
            style::switch(painter, sw, if on { 1.0 } else { 0.0 }, hover, !enabled);
            let tw = fonts.shadowed_mid(
                painter,
                name,
                sw.right() + 8.0 * s,
                row_cy,
                styles::body(s),
                white(if enabled { 0.92 } else { 0.4 }),
                s,
            );
            self.hits.push(
                Rect::new(hit.x, hit.y, 38.0 * s + 16.0 * s + tw, hit.h),
                target,
                Some(Tip::new(tip, None)),
            );
            x += 120.0 * s;
        }
        y += ROW * s;

        // File name.
        let row_cy = y + ROW * s * 0.5;
        label(fonts, painter, "File name", row_cy);
        let field = Rect::new(
            cx,
            row_cy - CONTROL_H * s * 0.5,
            cw - 52.0 * s,
            CONTROL_H * s,
        );
        let focused = self.focus == Focus::Name;
        let name = self.settings.file_name.clone();
        style::text_field(painter, fonts, field, &name, "Replay", focused, caret, s);
        self.hits.push(field, Target::NameField, None);
        fonts.shadowed_mid(
            painter,
            ".mp4",
            field.right() + 10.0 * s,
            row_cy,
            styles::body(s),
            white(0.6),
            s,
        );
        y += ROW * s;
        hint(
            fonts,
            painter,
            "Saved in Videos \u{203a} Aero Client",
            y - 6.0 * s,
        );

        // ---- the sum, and the buttons ----
        let footer_top = rect.bottom() - FOOTER_H * s;
        painter.rect(
            Rect::new(rect.x + 1.0, footer_top.round(), rect.w - 2.0, 1.0),
            white(0.10),
        );
        let fcy = footer_top + FOOTER_H * s * 0.5;
        let length = end.saturating_sub(start);
        let frames = length * u64::from(self.settings.fps) / 1_000_000;
        let bytes = self.settings.estimate_bytes(length, self.window);
        let blend = self.settings.blend_frames();
        let mut sum = format!(
            "{}  \u{b7}  {} frames  \u{b7}  about {}",
            format::span(length),
            format::count(frames),
            format::size(bytes)
        );
        if blend > 1 {
            sum.push_str(&format!("  \u{b7}  renders {blend}x the frames"));
        }
        let sum_room = rect.w * 0.55;
        let sum = fonts.ellipsize(&sum, styles::small(s), sum_room);
        fonts.shadowed_mid(
            painter,
            &sum,
            rect.x + pad,
            fcy,
            styles::small(s),
            white(0.72),
            s,
        );
        let bh = 38.0 * s;
        let export = Rect::new(
            rect.right() - pad - 132.0 * s,
            fcy - bh * 0.5,
            132.0 * s,
            bh,
        );
        let look = Look {
            primary: true,
            ..self.look(export, Target::Export)
        };
        style::pill(
            painter,
            fonts,
            export,
            Some(Icon::Export),
            "Export",
            look,
            s,
        );
        self.hits.push(
            export,
            Target::Export,
            Some(Tip::new("Start the export", Some("Enter"))),
        );
        let cancel = Rect::new(export.x - 10.0 * s - 108.0 * s, export.y, 108.0 * s, bh);
        let look = self.look(cancel, Target::Cancel);
        style::pill(painter, fonts, cancel, None, "Cancel", look, s);
        self.hits
            .push(cancel, Target::Cancel, Some(Tip::new("Close", Some("Esc"))));
    }

    fn look(&self, rect: Rect, target: Target) -> Look {
        let hover = rect.contains(self.cursor.0, self.cursor.1);
        Look::new(hover, hover && self.pressed == Some(target))
    }

    #[allow(clippy::too_many_arguments)]
    fn segmented_row(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        rect: Rect,
        labels: &[(Option<icons::Icon>, &str)],
        selected: usize,
        disabled: &[bool],
        targets: &[Target],
        tips: &[Tip],
        s: f32,
    ) {
        let hover = style::segment_at(rect, labels.len(), s, self.cursor);
        let segments = style::segmented(painter, fonts, rect, labels, selected, hover, disabled, s);
        for (i, segment) in segments.into_iter().enumerate() {
            if disabled.get(i).copied().unwrap_or(false) {
                continue;
            }
            if let Some(target) = targets.get(i) {
                self.hits.push(segment, *target, tips.get(i).copied());
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn draw_progress(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        glass: &mut GlassFrame,
        screen: (f32, f32),
        s: f32,
        now: Instant,
    ) {
        let Some(progress) = self.progress.clone() else {
            return;
        };
        let since = self.running_since.unwrap_or(now);
        let open = anim::window_open_progress(ms_since(since, now), 0);
        let w = PANEL_W.min(screen.0 / s - 32.0) * s;
        let h = PANEL_H * s;
        let slide = ((1.0 - open) * anim::SLIDE_DIP * s).round();
        let rect = Rect::new(
            ((screen.0 - w) * 0.5).round(),
            (screen.1 - h - 36.0 * s).round() + slide,
            w.round(),
            h.round(),
        );
        glass.rects.push(GlassRect {
            rect: [rect.x, rect.y, rect.w, rect.h],
            radius: 22.0 * s,
            alpha: open,
        });
        painter.alpha = open;
        style::panel(painter, rect, 22.0 * s, 0.0, s);
        self.hits.push(rect, Target::Window, None);
        let pad = 22.0 * s;
        let request_name = self.request().file_name;
        let (title, icon, color) = match &progress.state {
            ExportState::Running => ("Exporting video", Icon::Export, white(0.95)),
            ExportState::Finishing => ("Finishing the file", Icon::Export, white(0.95)),
            ExportState::Done { .. } => ("Video saved", Icon::Check, style::SUCCESS),
            ExportState::Failed(_) => ("Export failed", Icon::Warning, style::DANGER),
            ExportState::Cancelled => ("Export cancelled", Icon::Close, white(0.8)),
        };
        let ty = rect.y + 34.0 * s;
        icons::draw(
            painter,
            icon,
            (rect.x + pad + 11.0 * s, ty),
            22.0 * s,
            2.0 * s,
            color,
        );
        let title_style = TextStyle::new(17.0 * s).weight(Weight::Semibold);
        let tw = fonts.shadowed_mid(
            painter,
            title,
            rect.x + pad + 32.0 * s,
            ty,
            title_style,
            white(1.0),
            s,
        );
        let fraction = match &progress.state {
            ExportState::Done { .. } => 1.0,
            _ => progress.frames_done as f32 / progress.frames_total.max(1) as f32,
        };
        let percent = format!("{}%", (fraction * 100.0).floor() as u32);
        let ps = TextStyle::new(17.0 * s).weight(Weight::Semibold);
        let pw = fonts.width(&percent, ps);
        fonts.shadowed_mid(
            painter,
            &percent,
            rect.right() - pad - pw,
            ty,
            ps,
            white(0.95),
            s,
        );
        let name_x = rect.x + pad + 32.0 * s + tw + 12.0 * s;
        let name_room = rect.right() - pad - pw - 16.0 * s - name_x;
        let name = fonts.ellipsize(&request_name, styles::small(s), name_room);
        fonts.shadowed_mid(
            painter,
            &name,
            name_x,
            ty + 1.0 * s,
            styles::small(s),
            white(0.55),
            s,
        );

        let bar = Rect::new(
            rect.x + pad,
            rect.y + 62.0 * s,
            rect.w - 2.0 * pad,
            10.0 * s,
        );
        let bar_color = match &progress.state {
            ExportState::Done { .. } => style::SUCCESS,
            ExportState::Failed(_) => style::DANGER,
            ExportState::Cancelled => white(0.5),
            _ => style::ACCENT,
        };
        style::progress_bar(painter, bar, fraction, bar_color);

        let line_y = rect.y + 98.0 * s;
        let small = styles::body(s);
        let line = match &progress.state {
            ExportState::Running | ExportState::Finishing => {
                let mut line = format!(
                    "Frame {} of {}",
                    format::count(progress.frames_done),
                    format::count(progress.frames_total)
                );
                if let Some(rate) = self.rate.filter(|r| *r > 0.0) {
                    line.push_str(&format!("  \u{b7}  {rate:.0} frames/s"));
                    let left =
                        progress.frames_total.saturating_sub(progress.frames_done) as f32 / rate;
                    line.push_str(&format!(
                        "  \u{b7}  about {} left",
                        format::clock((left * 1e6) as u64 + 999_999)
                    ));
                }
                line.push_str(&format!(
                    "  \u{b7}  {} elapsed",
                    format::clock((progress.elapsed_s * 1e6) as u64)
                ));
                line
            }
            ExportState::Done { bytes, .. } => format!(
                "{}  \u{b7}  {} frames in {}",
                format::size(*bytes),
                format::count(progress.frames_total),
                format::clock((progress.elapsed_s * 1e6) as u64)
            ),
            ExportState::Failed(reason) => reason.clone(),
            ExportState::Cancelled => "Nothing was saved.".to_owned(),
        };
        let line = fonts.ellipsize(&line, small, rect.w - 2.0 * pad);
        fonts.shadowed_mid(painter, &line, rect.x + pad, line_y, small, white(0.78), s);

        // Buttons.
        let bh = 36.0 * s;
        let by = rect.bottom() - pad - bh + 4.0 * s;
        match &progress.state {
            ExportState::Running | ExportState::Finishing => {
                let cancel = Rect::new(rect.right() - pad - 120.0 * s, by, 120.0 * s, bh);
                let look = self.look(cancel, Target::Cancel);
                style::pill(painter, fonts, cancel, Some(Icon::Close), "Cancel", look, s);
                self.hits.push(
                    cancel,
                    Target::Cancel,
                    Some(Tip::new("Stop the export", None)),
                );
                fonts.shadowed_mid(
                    painter,
                    "The world behind shows the frames as they render.",
                    rect.x + pad,
                    by + bh * 0.5,
                    styles::small(s),
                    white(0.5),
                    s,
                );
            }
            ExportState::Done { .. } => {
                let done = Rect::new(rect.right() - pad - 100.0 * s, by, 100.0 * s, bh);
                let look = Look {
                    primary: true,
                    ..self.look(done, Target::Done)
                };
                style::pill(painter, fonts, done, Some(Icon::Check), "Done", look, s);
                self.hits
                    .push(done, Target::Done, Some(Tip::new("Close", Some("Esc"))));
                let show = Rect::new(done.x - 10.0 * s - 156.0 * s, by, 156.0 * s, bh);
                let look = self.look(show, Target::ShowFile);
                style::pill(
                    painter,
                    fonts,
                    show,
                    Some(Icon::Folder),
                    "Show in folder",
                    look,
                    s,
                );
                self.hits.push(show, Target::ShowFile, None);
            }
            ExportState::Failed(_) | ExportState::Cancelled => {
                let close = Rect::new(rect.right() - pad - 100.0 * s, by, 100.0 * s, bh);
                let look = self.look(close, Target::Done);
                style::pill(painter, fonts, close, None, "Close", look, s);
                self.hits
                    .push(close, Target::Done, Some(Tip::new("Close", Some("Esc"))));
            }
        }
    }
}
