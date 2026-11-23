//! The controls over a replay, laid out like a video player so nothing needs explaining: a glass
//! bar along the bottom with the **timeline** on top (drag it to scrub; hovering shows the time;
//! the In-Out **selection** is a blue window around it with handles to drag), and under it, from
//! left to right, **play/pause**, **back and forward 5 seconds**, **step**, the time, the
//! **speed**; the **camera** (Player, Follow, Free) in the middle; and on the right **In**,
//! **Out**, the selection's length with **Preview** (loops it) and clear, **Render settings**
//! (the mod menu over the replay), **Export**, **Hide** and **Exit**. Every button's tooltip
//! names its key; the keys are those of video players (Space or K, the arrows, `,` and `.`).
//!
//! While the replay plays, the controls fade out after a few seconds without the mouse moving
//! and come back as soon as it moves (H hides them until H is pressed again). A small chip at
//! the top left says what is playing; with the free camera a hint says how to fly it.

use std::time::Instant;

use super::export::{ExportDialog, ExportEvent, ExportRequest, ExportSettings};
use super::format;
use super::icons::{self, Icon};
use super::style::{
    self, fade, ms_since, styles, Fonts, Hits, Look, Tip, TipClock, ACCENT, ACCENT_TEXT,
};
use super::{key, CameraMode, PlaybackStatus, ReplayEnv, ReplayInfo, SPEEDS};
use crate::screens::modmenu::anim;
use crate::screens::modmenu::glass::{GlassFrame, GlassRect};
use crate::screens::modmenu::paint::{black, css, white, Fill, Painter, Rect};
use crate::screens::modmenu::text::{TextStyle, Weight};
use crate::widget::keys::Modifiers;
use crate::widget::DrawList;

const BAR_MARGIN: f32 = 24.0;
const BAR_BOTTOM: f32 = 20.0;
const BAR_H: f32 = 128.0;
const BAR_RADIUS: f32 = 22.0;
const BAR_PAD: f32 = 26.0;
const TRACK_Y: f32 = 34.0;
const TRACK_H: f32 = 6.0;
const CONTROLS_Y: f32 = 88.0;
const BUTTON: f32 = 40.0;
/// How far one press of the arrow keys (and the ±5 s buttons) jumps.
pub const JUMP_US: u64 = 5_000_000;
/// How long the controls stay after the mouse stops while the replay plays.
pub const IDLE_HIDE_MS: f32 = 2600.0;
const TOAST_MS: f32 = 1800.0;
const HIDE_HINT_MS: f32 = 2200.0;
const FREE_HINT_MS: f32 = 7000.0;

/// What the viewer asks the client (and the playback engine) to do.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewerEvent {
    /// Start playing.
    Play,
    /// Pause.
    Pause,
    /// Jump to this time on the timeline (microseconds).
    Seek(u64),
    /// Step one tick back (-1) or forward (1), paused.
    Step(i32),
    /// Play at this speed.
    Speed(f32),
    /// Watch through this camera.
    Camera(CameraMode),
    /// Open the mod menu over the replay (shaders and other render mods).
    RenderSettings,
    /// Open the game's own options (video, chat, controls...) over the replay;
    /// Done comes back to it.
    Options,
    /// Start this export.
    Export(ExportRequest),
    /// Stop the running export.
    CancelExport,
    /// Show this exported file in Explorer.
    ShowFile(String),
    /// Leave the replay (back to the browser).
    Exit,
    /// The Sound menu changed a channel (0 game, 1 microphone, 2 desktop
    /// audio): its mute for this viewing and its volume (0..=2, the Replay mod's option).
    Sound {
        /// Which channel.
        channel: usize,
        /// Muted while watching.
        muted: bool,
        /// Its volume, 1 = 100%.
        volume: f32,
    },
}

/// The Sound menu's rows: label, and the tip of a channel the replay lacks.
const SOUND_CHANNELS: [(&str, &str); 3] = [
    ("Game sound", ""),
    ("Microphone", "This replay has no microphone track"),
    ("Desktop audio", "This replay has no desktop audio"),
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    Bar,
    Track,
    InHandle,
    OutHandle,
    Play,
    Back5,
    Forward5,
    StepBack,
    StepForward,
    Speed,
    /// The speed list's panel (between its items: keeps it open).
    SpeedMenu,
    SpeedOption(usize),
    Camera(CameraMode),
    MarkIn,
    MarkOut,
    Preview,
    ClearSelection,
    RenderSettings,
    /// The Options button: the game's own options over the replay.
    Options,
    Export,
    Hide,
    Exit,
    Marker(usize),
    /// The Sound button, its menu's panel, a row's switch, its - and +.
    Sound,
    SoundMenu,
    SoundMute(usize),
    SoundDown(usize),
    SoundUp(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Drag {
    /// Scrubbing the playhead; playback resumes on release if it was playing.
    Playhead { resume: bool },
    /// Dragging the In handle.
    In,
    /// Dragging the Out handle.
    Out,
}

/// The overlay's state. The client keeps one while a replay is open (it survives the mod menu
/// opening over the replay).
#[derive(Debug)]
pub struct ReplayViewerOverlay {
    info: ReplayInfo,
    status: PlaybackStatus,
    opened_at: Instant,
    last_frame: Option<Instant>,
    mark_in: Option<u64>,
    mark_out: Option<u64>,
    preview: bool,
    /// H: hidden until H again.
    hidden: bool,
    hidden_at: Option<Instant>,
    /// The last pointer move, click or key: the controls fade out a while after it.
    last_activity: Instant,
    /// The controls' eased visibility.
    shown: f32,
    hits: Hits<Target>,
    tips: TipClock,
    cursor: (f32, f32),
    pressed: Option<Target>,
    drag: Option<Drag>,
    /// The latest scrub position, sent once per frame.
    pending_seek: Option<u64>,
    speed_menu: bool,
    /// The press closed the speed list: its release does nothing else (no play/pause).
    swallow_release: bool,
    /// The timeline's track in the last frame.
    track: Rect,
    /// Eased hover growth of the track.
    track_hover: f32,
    toast: Option<(String, Instant)>,
    export: Option<ExportDialog>,
    /// The form's starting values (the `replay` mod's export settings).
    export_defaults: ExportSettings,
    /// A one-click clip export: leave the replay when its window closes.
    exit_after_export: bool,
    /// When the free camera was chosen (its hint shows for a while).
    free_since: Option<Instant>,
    /// This PC can encode H.265 (the export window offers it).
    h265_available: bool,
    /// The replay holds desktop audio (the export window offers it).
    has_desktop: bool,
    /// The Sound menu is open; each channel's `(muted, volume)`.
    sound_menu: bool,
    sound: [(bool, f32); 3],
    events: Vec<ViewerEvent>,
}

impl ReplayViewerOverlay {
    /// The controls for the replay described by `info`, appearing from `now`.
    #[must_use]
    pub fn new(info: ReplayInfo, now: Instant) -> ReplayViewerOverlay {
        let export_defaults = ExportSettings {
            file_name: format::file_stem(&info.title),
            ..ExportSettings::default()
        };
        ReplayViewerOverlay {
            info,
            status: PlaybackStatus::default(),
            opened_at: now,
            last_frame: None,
            mark_in: None,
            mark_out: None,
            preview: false,
            hidden: false,
            hidden_at: None,
            last_activity: now,
            // Shown from the first frame (the open animation fades the controls in).
            shown: 1.0,
            hits: Hits::default(),
            tips: TipClock::default(),
            cursor: (-1.0, -1.0),
            pressed: None,
            drag: None,
            pending_seek: None,
            speed_menu: false,
            swallow_release: false,
            track: Rect::default(),
            track_hover: 0.0,
            toast: None,
            export: None,
            export_defaults,
            exit_after_export: false,
            free_since: None,
            h265_available: true,
            has_desktop: false,
            sound_menu: false,
            sound: [(false, 1.0); 3],
            events: Vec::new(),
        }
    }

    /// Whether this PC can encode H.265 (when it cannot, the export window greys it out).
    pub fn set_h265_available(&mut self, available: bool) {
        self.h265_available = available;
    }

    /// Whether the replay holds desktop audio (the export window's switch).
    pub fn set_has_desktop(&mut self, has: bool) {
        self.has_desktop = has;
    }

    /// The sound channels' `(muted, volume)` (game, microphone, desktop audio) for the
    /// Sound menu; the client sets them when the replay opens.
    pub fn set_sound_channels(&mut self, channels: [(bool, f32); 3]) {
        self.sound = channels;
    }

    /// Whether the replay holds channel `index` (the game's sound always).
    fn has_channel(&self, index: usize) -> bool {
        match index {
            0 => true,
            1 => self.info.has_mic,
            _ => self.has_desktop,
        }
    }

    /// Change a channel and tell the client.
    fn set_channel(&mut self, index: usize, muted: bool, volume: f32) {
        if !self.has_channel(index) {
            return;
        }
        let Some(slot) = self.sound.get_mut(index) else {
            return;
        };
        // 5% steps from 0% to 200%, as the Replay mod's sliders.
        let volume = ((volume * 20.0).round() / 20.0).clamp(0.0, 2.0);
        *slot = (muted, volume);
        self.events.push(ViewerEvent::Sound {
            channel: index,
            muted,
            volume,
        });
    }

    /// The export window's starting values (the client reads them from the `replay` mod). The
    /// file name defaults to the replay's title.
    pub fn set_export_defaults(&mut self, mut settings: ExportSettings) {
        settings.file_name = format::file_stem(&self.info.title);
        self.export_defaults = settings;
    }

    /// What is playing.
    #[must_use]
    pub fn info(&self) -> &ReplayInfo {
        &self.info
    }

    /// The game window's size now (the export window's "Window" resolution).
    pub fn set_window_size(&mut self, size: (u32, u32)) {
        self.info.window_size = size;
    }

    /// Open the speed list, as a click on the speed button does (the review pictures).
    pub fn open_speed_menu(&mut self) {
        self.speed_menu = true;
    }

    /// Open the Sound menu (as its button does).
    pub fn open_sound_menu(&mut self) {
        self.sound_menu = true;
        self.speed_menu = false;
    }

    /// The events since the last call.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, ViewerEvent> {
        self.events.drain(..)
    }

    /// The selection, if both ends are set: `(in, out)` microseconds.
    #[must_use]
    pub fn selection(&self) -> Option<(u64, u64)> {
        match (self.mark_in, self.mark_out) {
            (Some(a), Some(b)) if b > a => Some((a, b)),
            _ => None,
        }
    }

    /// Whether the selection loops.
    #[must_use]
    pub fn previewing(&self) -> bool {
        self.preview && self.selection().is_some()
    }

    /// Whether the export window is open.
    #[must_use]
    pub fn exporting(&self) -> bool {
        self.export.is_some()
    }

    /// Whether an export job is running (the client drives playback for it).
    #[must_use]
    pub fn export_running(&self) -> bool {
        self.export.as_ref().is_some_and(ExportDialog::running)
    }

    /// The open export window (the client reports progress to it).
    pub fn export_dialog(&mut self) -> Option<&mut ExportDialog> {
        self.export.as_mut()
    }

    /// Whether the controls are showing (not hidden or faded out).
    #[must_use]
    pub fn controls_visible(&self) -> bool {
        self.shown > 0.01
    }

    /// Whether a text box has the keyboard (the client then sends typed text here first).
    #[must_use]
    pub fn typing(&self) -> bool {
        self.export.as_ref().is_some_and(ExportDialog::typing)
    }

    /// The playback state this frame (from the engine). Also loops the selection while
    /// Preview is on.
    pub fn set_status(&mut self, status: PlaybackStatus) {
        if status.camera == CameraMode::Free && self.status.camera != CameraMode::Free {
            self.free_since = Some(self.last_frame.unwrap_or(self.opened_at));
        }
        self.status = status;
        if self.export_running() || self.drag.is_some() {
            return;
        }
        if let (true, Some((a, b))) = (self.preview, self.selection()) {
            if status.playing && status.seeking.is_none() && status.time_us >= b {
                self.events.push(ViewerEvent::Seek(a));
            }
        }
    }

    /// Set In and Out (tests, and the client restoring a selection).
    pub fn set_selection(&mut self, mark_in: Option<u64>, mark_out: Option<u64>) {
        self.mark_in = mark_in;
        self.mark_out = mark_out;
    }

    /// Export at once with `settings` (a one-click clip export): the window opens on its
    /// progress view, and the replay closes when the player closes it.
    pub fn export_now(&mut self, settings: ExportSettings, now: Instant) {
        let mut dialog = ExportDialog::new(
            settings,
            None,
            self.status.duration_us,
            self.info.window_size,
            self.info.has_mic,
            self.info.clip,
            now,
        );
        dialog.set_h265_available(self.h265_available);
        dialog.set_has_desktop(self.has_desktop);
        dialog.start_now(now);
        self.export = Some(dialog);
        self.exit_after_export = true;
        self.forward_export_events();
    }

    // ---- input ----------------------------------------------------------------------------

    /// The pointer moved.
    pub fn pointer_moved(&mut self, x: f32, y: f32, now: Instant) {
        if (x, y) != self.cursor {
            self.last_activity = now;
        }
        self.cursor = (x, y);
        if let Some(export) = self.export.as_mut() {
            export.pointer_moved(x, y);
            return;
        }
        match self.drag {
            Some(Drag::Playhead { .. }) => self.pending_seek = Some(self.time_at(x)),
            Some(Drag::In) => {
                let t = self.time_at(x);
                let limit = self
                    .mark_out
                    .map_or(u64::MAX, |o| o.saturating_sub(100_000));
                self.mark_in = Some(t.min(limit));
            }
            Some(Drag::Out) => {
                let t = self.time_at(x);
                let limit = self.mark_in.map_or(0, |i| i + 100_000);
                self.mark_out = Some(t.max(limit).min(self.status.duration_us));
            }
            None => {}
        }
    }

    /// The tooltip label of the control under `(x, y)` in the last frame (what the pointer
    /// would explain there; the tests find controls by it).
    #[must_use]
    pub fn tooltip_at(&self, x: f32, y: f32) -> Option<&'static str> {
        self.hits
            .entry_at(x, y)
            .and_then(|(_, _, tip)| tip.map(|t| t.label))
    }

    /// The rectangles of the buttons drawn in the last frame (not the bar, the timeline or its
    /// handles): the tests check that none overlaps another or leaves the screen.
    #[must_use]
    pub fn control_rects(&self) -> Vec<Rect> {
        self.hits
            .targets()
            .into_iter()
            .filter(|t| {
                !matches!(
                    t,
                    Target::Bar
                        | Target::Track
                        | Target::InHandle
                        | Target::OutHandle
                        | Target::Marker(_)
                        | Target::SpeedMenu
                        | Target::SpeedOption(_)
                        | Target::SoundMenu
                        | Target::SoundMute(_)
                        | Target::SoundDown(_)
                        | Target::SoundUp(_)
                )
            })
            .filter_map(|t| self.hits.rect_of(t))
            .collect()
    }

    /// The timeline time under screen x.
    fn time_at(&self, x: f32) -> u64 {
        if self.track.w <= 0.0 {
            return 0;
        }
        let t = ((x - self.track.x) / self.track.w).clamp(0.0, 1.0);
        (f64::from(t) * self.status.duration_us as f64) as u64
    }

    /// A mouse button went down. Returns whether the overlay used it (a click on the world
    /// outside the controls plays or pauses, as in a video player, so this is `true` for any
    /// left click while the controls show).
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8, now: Instant) -> bool {
        self.cursor = (x, y);
        self.last_activity = now;
        self.tips.suppress();
        if let Some(export) = self.export.as_mut() {
            export.mouse_pressed(x, y, button);
            return true;
        }
        if button != 0 {
            return false;
        }
        let target = if self.shown > 0.5 {
            self.hits.at(x, y)
        } else {
            None
        };
        if self.speed_menu
            && !matches!(
                target,
                Some(Target::SpeedOption(_) | Target::SpeedMenu | Target::Speed)
            )
        {
            // A click outside the open list only closes it.
            self.speed_menu = false;
            self.pressed = None;
            self.swallow_release = true;
            return true;
        }
        // The same for the Sound menu.
        if self.sound_menu
            && !matches!(
                target,
                Some(
                    Target::Sound
                        | Target::SoundMenu
                        | Target::SoundMute(_)
                        | Target::SoundDown(_)
                        | Target::SoundUp(_)
                )
            )
        {
            self.sound_menu = false;
            self.pressed = None;
            self.swallow_release = true;
            return true;
        }
        self.pressed = target;
        match target {
            Some(Target::Track) => {
                let resume = self.status.playing;
                if resume {
                    self.events.push(ViewerEvent::Pause);
                }
                self.drag = Some(Drag::Playhead { resume });
                self.pending_seek = Some(self.time_at(x));
            }
            Some(Target::InHandle) => self.drag = Some(Drag::In),
            Some(Target::OutHandle) => self.drag = Some(Drag::Out),
            _ => {}
        }
        true
    }

    /// A mouse button went up.
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8, now: Instant) {
        self.cursor = (x, y);
        self.last_activity = now;
        if let Some(export) = self.export.as_mut() {
            export.mouse_released(x, y, button, now);
            self.forward_export_events();
            return;
        }
        if button != 0 {
            return;
        }
        if std::mem::take(&mut self.swallow_release) {
            return;
        }
        if let Some(drag) = self.drag.take() {
            if let Drag::Playhead { resume } = drag {
                let t = self.time_at(x);
                self.pending_seek = None;
                self.events.push(ViewerEvent::Seek(t));
                if resume {
                    self.events.push(ViewerEvent::Play);
                }
            }
            self.pressed = None;
            return;
        }
        let Some(pressed) = self.pressed.take() else {
            // A click on the world itself: play or pause, as a video player does, whether the
            // controls show or not (unless the release lands on a showing control).
            if self.shown <= 0.5 || self.hits.at(x, y).is_none() {
                self.toggle_play();
            }
            return;
        };
        if self.hits.at(x, y) == Some(pressed) {
            self.activate(pressed, now);
        }
    }

    /// The wheel turned over `(x, y)`: over the timeline it jumps 5 seconds, over the speed
    /// button it changes the speed. Returns whether it was used.
    pub fn scrolled(&mut self, x: f32, y: f32, delta: f32, now: Instant) -> bool {
        self.last_activity = now;
        if self.export.is_some() || self.shown < 0.5 || delta == 0.0 {
            return self.export.is_some();
        }
        match self.hits.at(x, y) {
            Some(Target::Track | Target::InHandle | Target::OutHandle) => {
                self.jump(if delta > 0.0 { 1 } else { -1 });
                true
            }
            Some(Target::Speed | Target::SpeedOption(_)) => {
                self.change_speed(if delta > 0.0 { 1 } else { -1 }, now);
                true
            }
            Some(_) => true,
            None => false,
        }
    }

    /// Paste into the export window's focused box.
    pub fn paste(&mut self, text: &str) -> bool {
        self.export.as_mut().is_some_and(|e| e.paste(text))
    }

    /// A key went down. Returns whether the overlay used it.
    pub fn key(&mut self, code: i32, typed: Option<char>, mods: Modifiers, now: Instant) -> bool {
        self.last_activity = now;
        if let Some(export) = self.export.as_mut() {
            export.key(code, typed, mods, now);
            self.forward_export_events();
            return true;
        }
        // Ctrl+, (the usual settings shortcut): the game's options. Plain `,`
        // stays one tick back.
        if mods.ctrl && code == key::COMMA {
            self.speed_menu = false;
            self.events.push(ViewerEvent::Options);
            return true;
        }
        match code {
            key::ESCAPE => {
                if self.sound_menu {
                    self.sound_menu = false;
                } else if self.speed_menu {
                    self.speed_menu = false;
                } else if self.hidden {
                    self.hidden = false;
                } else {
                    self.events.push(ViewerEvent::Exit);
                }
            }
            key::SPACE | key::K => self.toggle_play(),
            key::LEFT => self.jump(-1),
            key::RIGHT => self.jump(1),
            key::COMMA => self.step(-1),
            key::PERIOD => self.step(1),
            key::MINUS | key::SUBTRACT => self.change_speed(-1, now),
            key::EQUALS | key::ADD => self.change_speed(1, now),
            key::C => self
                .events
                .push(ViewerEvent::Camera(self.status.camera.next())),
            key::I => self.mark(true, now),
            key::O => self.mark(false, now),
            key::P => self.toggle_preview(now),
            key::E => self.open_export(now),
            key::H => self.toggle_hidden(now),
            key::HOME => self.seek(0),
            key::END => self.seek(self.status.duration_us),
            key::BACKSPACE | key::DELETE => self.clear_selection(now),
            // The previous and next bookmark (or any marker).
            key::LBRACKET => self.jump_marker(false, now),
            key::RBRACKET => self.jump_marker(true, now),
            // 1..9 and 0: jump to 10%..90% and the start.
            2..=11 => {
                let tenth = if code == 11 { 0 } else { (code - 1) as u64 };
                self.seek(self.status.duration_us / 10 * tenth);
            }
            _ => return false,
        }
        true
    }

    fn forward_export_events(&mut self) {
        let Some(export) = self.export.as_mut() else {
            return;
        };
        let events: Vec<ExportEvent> = export.drain_events().collect();
        for event in events {
            match event {
                ExportEvent::Start(request) => self.events.push(ViewerEvent::Export(request)),
                ExportEvent::Cancel => self.events.push(ViewerEvent::CancelExport),
                ExportEvent::ShowFile(path) => self.events.push(ViewerEvent::ShowFile(path)),
                ExportEvent::Close => {
                    self.export = None;
                    if std::mem::take(&mut self.exit_after_export) {
                        self.events.push(ViewerEvent::Exit);
                    }
                }
            }
        }
    }

    fn toggle_play(&mut self) {
        if self.status.playing {
            self.events.push(ViewerEvent::Pause);
        } else {
            // At the end, play starts over (or at In while previewing).
            if self.status.time_us >= self.status.duration_us.saturating_sub(50_000) {
                let start = self
                    .selection()
                    .filter(|_| self.preview)
                    .map_or(0, |(a, _)| a);
                self.events.push(ViewerEvent::Seek(start));
            }
            self.events.push(ViewerEvent::Play);
        }
    }

    fn seek(&mut self, t: u64) {
        self.events
            .push(ViewerEvent::Seek(t.min(self.status.duration_us)));
    }

    fn jump(&mut self, direction: i64) {
        let t = if direction < 0 {
            self.status.time_us.saturating_sub(JUMP_US)
        } else {
            self.status.time_us.saturating_add(JUMP_US)
        };
        self.seek(t);
    }

    fn step(&mut self, ticks: i32) {
        if self.status.playing {
            self.events.push(ViewerEvent::Pause);
        }
        self.events.push(ViewerEvent::Step(ticks));
    }

    /// `]` (`forward`) and `[`: seek to the next or previous marker (bookmarks and clip
    /// markers alike), exactly on it as a click on its diamond does, and say which ("Bookmark 2
    /// of 5"). Previous from just after a marker (within a second) goes to the one before, as a
    /// media player's previous does.
    fn jump_marker(&mut self, forward: bool, now: Instant) {
        let t = self.status.time_us;
        let mut times: Vec<(u64, usize)> = self
            .info
            .markers
            .iter()
            .enumerate()
            .map(|(i, (time, _))| (*time, i))
            .filter(|(time, _)| *time <= self.status.duration_us)
            .collect();
        times.sort_unstable();
        let found = if forward {
            times.iter().find(|(time, _)| *time > t + 50_000)
        } else {
            times.iter().rev().find(|(time, _)| *time + 1_000_000 < t)
        };
        let Some(&(time, index)) = found else {
            let text = if times.is_empty() {
                "No bookmarks in this replay"
            } else if forward {
                "No more bookmarks"
            } else {
                "No earlier bookmarks"
            };
            self.toast = Some((text.to_owned(), now));
            return;
        };
        let position = times.iter().position(|(_, i)| *i == index).unwrap_or(0) + 1;
        let label = self.info.markers[index].1.as_str();
        let label = if label.is_empty() { "Marker" } else { label };
        self.toast = Some((
            format!(
                "{label} {position} of {}  \u{b7}  {}",
                times.len(),
                format::clock_tenths(time)
            ),
            now,
        ));
        self.seek(time);
    }

    fn change_speed(&mut self, direction: i32, now: Instant) {
        let current = SPEEDS
            .iter()
            .position(|s| (s - self.status.speed).abs() < 1e-3)
            .unwrap_or(4);
        let next = (current as i32 + direction).clamp(0, SPEEDS.len() as i32 - 1) as usize;
        if next != current {
            self.events.push(ViewerEvent::Speed(SPEEDS[next]));
            self.toast = Some((format!("Speed {}", format::speed(SPEEDS[next])), now));
        }
    }

    fn mark(&mut self, is_in: bool, now: Instant) {
        let t = self.status.time_us;
        if is_in {
            self.mark_in = Some(t);
            if self.mark_out.is_some_and(|o| o <= t) {
                self.mark_out = None;
            }
            self.toast = Some((format!("In at {}", format::clock_tenths(t)), now));
        } else {
            self.mark_out = Some(t);
            if self.mark_in.is_some_and(|i| i >= t) {
                self.mark_in = None;
            }
            self.toast = Some((format!("Out at {}", format::clock_tenths(t)), now));
        }
        if self.mark_in.is_none() && !is_in {
            // An Out with no In: the selection starts at the beginning.
            self.mark_in = (t > 0).then_some(0);
        }
    }

    fn clear_selection(&mut self, now: Instant) {
        if self.mark_in.is_some() || self.mark_out.is_some() {
            self.mark_in = None;
            self.mark_out = None;
            self.preview = false;
            self.toast = Some(("Selection cleared".to_owned(), now));
        }
    }

    fn toggle_preview(&mut self, now: Instant) {
        let Some((a, _)) = self.selection() else {
            self.toast = Some(("Set In (I) and Out (O) first".to_owned(), now));
            return;
        };
        self.preview = !self.preview;
        if self.preview {
            self.events.push(ViewerEvent::Seek(a));
            self.events.push(ViewerEvent::Play);
        }
    }

    fn toggle_hidden(&mut self, now: Instant) {
        self.hidden = !self.hidden;
        self.speed_menu = false;
        self.hidden_at = self.hidden.then_some(now);
    }

    fn open_export(&mut self, now: Instant) {
        if self.status.playing {
            self.events.push(ViewerEvent::Pause);
        }
        self.speed_menu = false;
        let mut dialog = ExportDialog::new(
            self.export_defaults.clone(),
            self.selection(),
            self.status.duration_us,
            self.info.window_size,
            self.info.has_mic,
            self.info.clip,
            now,
        );
        dialog.set_h265_available(self.h265_available);
        dialog.set_has_desktop(self.has_desktop);
        self.export = Some(dialog);
    }

    fn activate(&mut self, target: Target, now: Instant) {
        match target {
            Target::Bar
            | Target::Track
            | Target::InHandle
            | Target::OutHandle
            | Target::SpeedMenu
            | Target::SoundMenu => {}
            // The Sound menu.
            Target::Sound => {
                self.sound_menu = !self.sound_menu;
                self.speed_menu = false;
            }
            Target::SoundMute(i) => {
                if let Some(&(muted, volume)) = self.sound.get(i) {
                    self.set_channel(i, !muted, volume);
                }
            }
            Target::SoundDown(i) | Target::SoundUp(i) => {
                if let Some(&(muted, volume)) = self.sound.get(i) {
                    let step = if matches!(target, Target::SoundUp(_)) {
                        0.05
                    } else {
                        -0.05
                    };
                    self.set_channel(i, muted, volume + step);
                }
            }
            Target::Marker(i) => {
                if let Some((t, _)) = self.info.markers.get(i) {
                    let t = *t;
                    self.seek(t);
                }
            }
            Target::Play => self.toggle_play(),
            Target::Back5 => self.jump(-1),
            Target::Forward5 => self.jump(1),
            Target::StepBack => self.step(-1),
            Target::StepForward => self.step(1),
            Target::Speed => self.speed_menu = !self.speed_menu,
            Target::SpeedOption(i) => {
                self.speed_menu = false;
                if let Some(speed) = SPEEDS.get(i) {
                    if (speed - self.status.speed).abs() > 1e-3 {
                        self.events.push(ViewerEvent::Speed(*speed));
                    }
                }
            }
            Target::Camera(mode) => {
                if mode != self.status.camera {
                    self.events.push(ViewerEvent::Camera(mode));
                }
            }
            Target::MarkIn => self.mark(true, now),
            Target::MarkOut => self.mark(false, now),
            Target::Preview => self.toggle_preview(now),
            Target::ClearSelection => self.clear_selection(now),
            Target::RenderSettings => self.events.push(ViewerEvent::RenderSettings),
            Target::Options => self.events.push(ViewerEvent::Options),
            Target::Export => self.open_export(now),
            Target::Hide => self.toggle_hidden(now),
            Target::Exit => self.events.push(ViewerEvent::Exit),
        }
    }

    // ---- drawing --------------------------------------------------------------------------

    /// Draw one frame into `list` and return its glass.
    #[allow(clippy::too_many_lines)]
    pub fn build(&mut self, env: &mut ReplayEnv<'_>, list: &mut DrawList) -> GlassFrame {
        self.hits.clear();
        let now = env.now;
        let dt = self.last_frame.map_or(0.0, |last| {
            now.saturating_duration_since(last).as_secs_f32()
        });
        self.last_frame = Some(now);
        if let Some(t) = self.pending_seek.take() {
            self.events.push(ViewerEvent::Seek(t));
        }
        let s = style::scale(env.screen);
        let mut glass = GlassFrame {
            px_per_dip: s,
            time: env.time_s,
            dim: 0.0,
            strength: 1.0,
            // Frostier than the menu: the game moves behind these panes.
            frost: 3.2,
            rects: Vec::with_capacity(4),
        };

        // Visibility: hidden by H, while the export window is up (its panel takes the bar's
        // place), or idle while playing.
        let idle = ms_since(self.last_activity, now) > IDLE_HIDE_MS;
        let over_bar = Self::bar_rect(env.screen)
            .inset(-12.0 * s)
            .contains(self.cursor.0, self.cursor.1);
        let want = !(self.hidden
            || self.export.is_some()
            || (self.status.playing
                && idle
                && !over_bar
                && self.drag.is_none()
                && !self.speed_menu
                && !self.sound_menu
                && self.export.is_none()));
        let open = anim::window_open_progress(ms_since(self.opened_at, now), 0);
        // In quickly; out gently when idle, quickly when hidden on purpose (H, the export panel).
        let rate = if want || self.hidden || self.export.is_some() {
            14.0
        } else {
            6.0
        };
        let k = anim::approach(dt, rate);
        self.shown += (f32::from(u8::from(want)) - self.shown) * k;
        if (f32::from(u8::from(want)) - self.shown).abs() < 0.003 {
            self.shown = f32::from(u8::from(want));
        }
        let alpha = self.shown * open;
        let hover_track = self
            .track
            .inset(-10.0 * s)
            .contains(self.cursor.0, self.cursor.1)
            || matches!(self.drag, Some(Drag::Playhead { .. }));
        let kt = anim::approach(dt, 18.0);
        self.track_hover += (f32::from(u8::from(hover_track)) - self.track_hover) * kt;

        let mut fonts = Fonts {
            text: &mut *env.text,
            mc: &mut *env.mc_font,
        };
        let mut painter = Painter::new(list);

        if alpha > 0.001 {
            painter.alpha = alpha;
            // A soft scrim behind the bar so white text reads over a noon sky.
            let scrim_h = (BAR_H + BAR_BOTTOM + 90.0) * s;
            style::scrim(
                &mut painter,
                Rect::new(0.0, env.screen.1 - scrim_h, env.screen.0, scrim_h),
                0.38,
            );
            self.draw_chip(&mut painter, &mut fonts, &mut glass, s, alpha);
            self.draw_bar(&mut painter, &mut fonts, &mut glass, env.screen, s, alpha);
            if self.status.camera == CameraMode::Free && !self.status.flying {
                self.draw_free_hint(&mut painter, &mut fonts, env.screen, s, now);
            }
        }
        painter.alpha = 1.0;

        // A seek that rebuilds the world: a small progress chip in the middle.
        if let Some(progress) = self.status.seeking {
            let w = 260.0 * s;
            let rect = Rect::new(
                ((env.screen.0 - w) * 0.5).round(),
                (env.screen.1 * 0.5 - 30.0 * s).round(),
                w.round(),
                (60.0 * s).round(),
            );
            glass.rects.push(GlassRect {
                rect: [rect.x, rect.y, rect.w, rect.h],
                radius: 16.0 * s,
                alpha: 1.0,
            });
            style::panel(&mut painter, rect, 16.0 * s, 0.0, s);
            fonts.shadowed_mid(
                &mut painter,
                "Seeking\u{2026}",
                rect.x + 18.0 * s,
                rect.y + 21.0 * s,
                styles::button(s),
                white(0.95),
                s,
            );
            style::progress_bar(
                &mut painter,
                Rect::new(
                    rect.x + 18.0 * s,
                    rect.y + 38.0 * s,
                    rect.w - 36.0 * s,
                    6.0 * s,
                ),
                progress,
                ACCENT,
            );
        }

        // Toasts (speed, In/Out) and the "press H" hint.
        let message = match (&self.toast, self.hidden_at) {
            (Some((text, at)), _) if ms_since(*at, now) < TOAST_MS => {
                Some((text.clone(), ms_since(*at, now), TOAST_MS))
            }
            (_, Some(at)) if self.hidden && ms_since(at, now) < HIDE_HINT_MS => Some((
                "Controls hidden \u{b7} press H to show them".to_owned(),
                ms_since(at, now),
                HIDE_HINT_MS,
            )),
            _ => None,
        };
        if let Some((text, age, life)) = message {
            let a = (age / 120.0).min(1.0) * ((life - age) / 300.0).clamp(0.0, 1.0);
            let y = if self.shown > 0.5 && !self.hidden {
                env.screen.1 - (BAR_H + BAR_BOTTOM + 58.0) * s
            } else {
                env.screen.1 - 80.0 * s
            };
            style::toast(
                &mut painter,
                &mut fonts,
                &text,
                None,
                env.screen.0 * 0.5,
                y,
                a,
                s,
            );
        }

        // The export window over everything.
        if let Some(export) = self.export.as_mut() {
            let mut sub = ReplayEnv {
                text: &mut *fonts.text,
                mc_font: &mut *fonts.mc,
                screen: env.screen,
                now,
                time_s: env.time_s,
            };
            export.build(&mut sub, painter.list, &mut glass);
        } else {
            let hovered = self
                .hits
                .entry_at(self.cursor.0, self.cursor.1)
                .filter(|_| self.shown > 0.5 && self.drag.is_none())
                .and_then(|(r, _, tip)| tip.map(|tip| (r, tip)));
            if let Some((anchor, tip, a)) = self.tips.update(hovered, now) {
                style::tooltip(
                    &mut painter,
                    &mut fonts,
                    anchor,
                    tip.label,
                    tip.key,
                    env.screen,
                    a,
                    s,
                );
            }
        }
        glass
    }

    /// The bar's rectangle (before any fade) for a `screen`-sized window.
    #[must_use]
    pub fn bar_rect(screen: (f32, f32)) -> Rect {
        let s = style::scale(screen);
        let w = screen.0 - 2.0 * BAR_MARGIN * s;
        let w = w.min(1720.0 * s);
        Rect::new(
            ((screen.0 - w) * 0.5).round(),
            (screen.1 - (BAR_H + BAR_BOTTOM) * s).round(),
            w.round(),
            (BAR_H * s).round(),
        )
    }

    fn look(&self, rect: Rect, target: Target) -> Look {
        let hover = rect.contains(self.cursor.0, self.cursor.1) && self.drag.is_none();
        Look::new(hover, hover && self.pressed == Some(target))
    }

    /// The top-left chip: what is playing.
    fn draw_chip(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        glass: &mut GlassFrame,
        s: f32,
        alpha: f32,
    ) {
        let badge_style = TextStyle::new(10.5 * s).weight(Weight::Bold).caps(0.12);
        let badge = if self.info.clip { "Clip" } else { "Replay" };
        let bw = fonts.width(badge, badge_style) + 16.0 * s;
        let title_style = TextStyle::new(14.0 * s).weight(Weight::Semibold);
        let title = fonts.ellipsize(&self.info.title, title_style, 520.0 * s);
        let tw = fonts.width(&title, title_style);
        // A recording titled with its server (the default) does not say it twice.
        let sub = if self.info.server.is_empty() || self.info.title.contains(&self.info.server) {
            String::new()
        } else {
            format!("  \u{b7}  {}", self.info.server)
        };
        let sw = fonts.width(&sub, styles::small(s));
        let h = 40.0 * s;
        let w = 14.0 * s + bw + 10.0 * s + tw + sw + 16.0 * s;
        let rect = Rect::new((20.0 * s).round(), (18.0 * s).round(), w.round(), h.round());
        glass.rects.push(GlassRect {
            rect: [rect.x, rect.y, rect.w, rect.h],
            radius: h * 0.5,
            alpha,
        });
        style::panel(painter, rect, h * 0.5, 0.0, s);
        let cy = rect.y + h * 0.5;
        let badge_rect = Rect::new(rect.x + 12.0 * s, cy - 10.0 * s, bw, 20.0 * s);
        painter.round_rect(badge_rect, 6.0 * s, Fill::Solid(fade(ACCENT, 0.85)));
        fonts.plain_mid(
            painter,
            badge,
            badge_rect.x + 8.0 * s,
            cy,
            badge_style,
            white(1.0),
        );
        let x = badge_rect.right() + 10.0 * s;
        let w = fonts.shadowed_mid(painter, &title, x, cy, title_style, white(0.97), s);
        fonts.shadowed_mid(painter, &sub, x + w, cy, styles::small(s), white(0.62), s);
    }

    fn draw_free_hint(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        screen: (f32, f32),
        s: f32,
        now: Instant,
    ) {
        let age = self.free_since.map_or(0.0, |at| ms_since(at, now));
        if age > FREE_HINT_MS {
            return;
        }
        let a = ((FREE_HINT_MS - age) / 600.0).clamp(0.0, 1.0);
        let text = "Hold the right mouse button to look around  \u{b7}  WASD to move  \u{b7}  \
                    Space and Shift for up and down  \u{b7}  scroll for speed";
        // Under the title chip, so the two never overlap on a narrow window.
        style::toast(
            painter,
            fonts,
            text,
            Some((Icon::CameraFree, white(0.9))),
            screen.0 * 0.5,
            72.0 * s,
            a,
            s,
        );
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn draw_bar(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        glass: &mut GlassFrame,
        screen: (f32, f32),
        s: f32,
        alpha: f32,
    ) {
        let bar = Self::bar_rect(screen);
        glass.rects.push(GlassRect {
            rect: [bar.x, bar.y, bar.w, bar.h],
            radius: BAR_RADIUS * s,
            alpha,
        });
        style::panel(painter, bar, BAR_RADIUS * s, 0.0, s);
        self.hits.push(bar, Target::Bar, None);

        // ---- the timeline ----
        let pad = BAR_PAD * s;
        let grow = self.track_hover * 2.0 * s;
        let track = Rect::new(
            bar.x + pad,
            bar.y + TRACK_Y * s - grow * 0.5,
            bar.w - 2.0 * pad,
            TRACK_H * s + grow,
        );
        self.track = Rect::new(track.x, bar.y + TRACK_Y * s, track.w, TRACK_H * s);
        let duration = self.status.duration_us.max(1);
        let x_of = |t: u64| track.x + track.w * (t.min(duration) as f32 / duration as f32);
        self.hits.push(
            Rect::new(
                track.x - 8.0 * s,
                track.y - 14.0 * s,
                track.w + 16.0 * s,
                track.h + 28.0 * s,
            ),
            Target::Track,
            None,
        );

        // The selection: a blue window around the track, handles at In and Out.
        let selection = self.selection();
        let (sel_a, sel_b) = (self.mark_in, self.mark_out);
        if let Some((a, b)) = selection {
            let (xa, xb) = (x_of(a), x_of(b));
            let band = Rect::new(
                xa,
                track.y - 9.0 * s,
                (xb - xa).max(2.0),
                track.h + 18.0 * s,
            );
            painter.round_rect(band, 6.0 * s, Fill::Solid(fade(ACCENT, 0.30)));
            painter.round_rect_stroke(band, 6.0 * s, 1.0, 0.5, fade(ACCENT_TEXT, 0.65));
        }
        // Base track, then what has played. While scrubbing, the playhead follows the pointer
        // at once (the seek itself lands a frame or more later).
        let r = track.h * 0.5;
        painter.round_rect(track, r, Fill::Solid(white(0.24)));
        let played_x = match self.drag {
            Some(Drag::Playhead { .. }) => self.cursor.0.clamp(track.x, track.right()),
            _ => x_of(self.status.time_us),
        };
        if played_x > track.x {
            painter.round_rect(
                Rect::new(track.x, track.y, (played_x - track.x).max(track.h), track.h),
                r,
                Fill::Solid(white(0.92)),
            );
        }
        // The selection over the track: blue, so it reads inside the played part too.
        if let Some((a, b)) = selection {
            let (xa, xb) = (x_of(a), x_of(b));
            painter.round_rect(
                Rect::new(xa, track.y, (xb - xa).max(2.0), track.h),
                r,
                Fill::Solid(fade(ACCENT, 0.80)),
            );
        }
        // Handles for a set In or Out (also a lone one).
        for (mark, target, tip) in [
            (
                sel_a,
                Target::InHandle,
                Tip::new("In: drag to move", Some("I")),
            ),
            (
                sel_b,
                Target::OutHandle,
                Tip::new("Out: drag to move", Some("O")),
            ),
        ] {
            let Some(t) = mark else {
                continue;
            };
            let x = x_of(t);
            let handle = Rect::new(x - 2.0 * s, track.y - 11.0 * s, 4.0 * s, track.h + 22.0 * s);
            let hover = handle
                .inset(-6.0 * s)
                .contains(self.cursor.0, self.cursor.1)
                || self.drag
                    == Some(if target == Target::InHandle {
                        Drag::In
                    } else {
                        Drag::Out
                    });
            painter.round_rect_shadow(handle, 2.0 * s, [3.0 * s; 4], black(0.35));
            painter.round_rect(
                handle,
                2.0 * s,
                Fill::Solid(if hover { white(1.0) } else { ACCENT_TEXT }),
            );
            // A small flag pointing into the selection.
            let dir = if target == Target::InHandle {
                1.0
            } else {
                -1.0
            };
            style::fill_convex(
                painter,
                &[
                    [x, handle.y],
                    [x + dir * 7.0 * s, handle.y + 4.0 * s],
                    [x, handle.y + 8.0 * s],
                ],
                if hover { white(1.0) } else { ACCENT_TEXT },
            );
            self.hits.push(handle.inset(-6.0 * s), target, Some(tip));
        }
        // Markers from the recording: small amber diamonds above the track (a click jumps there).
        for (i, (t, _)) in self.info.markers.iter().enumerate() {
            let x = x_of(*t);
            let cy = track.y - 9.0 * s;
            let r = 4.0 * s;
            style::fill_convex(
                painter,
                &[[x, cy - r], [x + r, cy], [x, cy + r], [x - r, cy]],
                css(255, 206, 110, 0.95),
            );
            let mark = Rect::new(x - r, cy - r, 2.0 * r, 2.0 * r);
            // The tooltip names the keys that step between them.
            let tip = match self.info.markers[i].1.as_str() {
                "Bookmark" => Tip::new("Bookmark", Some("[ ]")),
                "Clip" => Tip::new("Clip saved here", Some("[ ]")),
                _ => Tip::new("Marker", Some("[ ]")),
            };
            self.hits
                .push(mark.inset(-4.0 * s), Target::Marker(i), Some(tip));
        }
        // The playhead.
        let knob_r = (6.5 + 1.5 * self.track_hover) * s;
        let cy = track.y + track.h * 0.5;
        painter.circle(played_x, cy + 1.0 * s, knob_r + 1.0 * s, black(0.30));
        painter.circle(played_x, cy, knob_r, white(1.0));
        // Hovering: a hairline and the time under the pointer.
        if self.track_hover > 0.05 && self.drag.is_none() && !self.speed_menu {
            let x = self.cursor.0.clamp(track.x, track.right());
            let t = self.time_at(x);
            painter.rect(
                Rect::new(
                    x.round(),
                    track.y - 6.0 * s,
                    1.0 * s.max(1.0),
                    track.h + 12.0 * s,
                ),
                white(0.7 * self.track_hover),
            );
            self.time_bubble(
                painter,
                fonts,
                x,
                track.y - 16.0 * s,
                t,
                s,
                self.track_hover,
            );
        } else if let Some(Drag::Playhead { .. }) = self.drag {
            let x = self.cursor.0.clamp(track.x, track.right());
            self.time_bubble(
                painter,
                fonts,
                x,
                track.y - 16.0 * s,
                self.time_at(x),
                s,
                1.0,
            );
        }

        // ---- the controls row ----
        let cy = bar.y + CONTROLS_Y * s;
        let b = BUTTON * s;
        let mut x = bar.x + pad - 4.0 * s;
        // Play/pause: bigger, lit.
        let play_r = Rect::new(x, cy - 24.0 * s, 48.0 * s, 48.0 * s);
        let playing = self.status.playing;
        let look = Look {
            active: true,
            ..self.look(play_r, Target::Play)
        };
        style::icon_button(
            painter,
            play_r,
            if playing { Icon::Pause } else { Icon::Play },
            22.0 * s,
            look,
            s,
        );
        self.hits.push(
            play_r,
            Target::Play,
            Some(if playing {
                Tip::new("Pause", Some("Space"))
            } else {
                Tip::new("Play", Some("Space"))
            }),
        );
        x = play_r.right() + 10.0 * s;
        for (icon, target, tip, size) in [
            (
                Icon::Back5,
                Target::Back5,
                Tip::new("Back 5 seconds", Some("\u{2190}")),
                b,
            ),
            (
                Icon::Forward5,
                Target::Forward5,
                Tip::new("Forward 5 seconds", Some("\u{2192}")),
                b,
            ),
            (
                Icon::StepBack,
                Target::StepBack,
                Tip::new("Back one tick", Some(",")),
                36.0 * s,
            ),
            (
                Icon::StepForward,
                Target::StepForward,
                Tip::new("Forward one tick", Some(".")),
                36.0 * s,
            ),
        ] {
            let rect = Rect::new(x, cy - size * 0.5, size, size);
            let look = self.look(rect, target);
            let icon_size = if matches!(icon, Icon::StepBack | Icon::StepForward) {
                15.0 * s
            } else {
                21.0 * s
            };
            style::icon_button(painter, rect, icon, icon_size, look, s);
            self.hits.push(rect, target, Some(tip));
            x = rect.right() + 4.0 * s;
        }
        x += 10.0 * s;
        // The time (while scrubbing, the time under the pointer).
        let readout = styles::readout(s);
        let mut time = String::with_capacity(24);
        let shown_time = match self.drag {
            Some(Drag::Playhead { .. }) => self.time_at(self.cursor.0),
            _ => self.status.time_us,
        };
        format::write_clock_tenths(&mut time, shown_time.min(self.status.duration_us));
        let tw = fonts.shadowed_mid(painter, &time, x, cy, readout, white(0.98), s);
        let total = format!(" / {}", format::clock(self.status.duration_us));
        let total_w = fonts.shadowed_mid(
            painter,
            &total,
            x + tw,
            cy,
            readout.weight(Weight::Regular),
            white(0.6),
            s,
        );
        x += tw + total_w + 14.0 * s;
        // Speed.
        let speed_text = format::speed(self.status.speed);
        let sw = fonts.width(&speed_text, styles::button(s)) + 40.0 * s;
        let speed_r = Rect::new(x, cy - 17.0 * s, sw.max(64.0 * s), 34.0 * s);
        let look = Look {
            active: self.speed_menu,
            ..self.look(speed_r, Target::Speed)
        };
        style::pill_body(painter, speed_r, 17.0 * s, look, s);
        fonts.shadowed_mid(
            painter,
            &speed_text,
            speed_r.x + 14.0 * s,
            cy,
            styles::button(s),
            white(0.97),
            s,
        );
        style::chevron_down(
            painter,
            (speed_r.right() - 15.0 * s, cy),
            3.5 * s,
            1.4 * s,
            white(0.8),
        );
        self.hits.push(
            speed_r,
            Target::Speed,
            Some(Tip::new("Speed", Some("- / ="))),
        );
        let left_end = speed_r.right();

        // ---- right group, from the right edge ----
        let mut right = bar.right() - pad + 4.0 * s;
        let mut sound_r = Rect::default();
        for (icon, target, tip) in [
            (
                Icon::Exit,
                Target::Exit,
                Tip::new("Exit the replay", Some("Esc")),
            ),
            (
                Icon::Hide,
                Target::Hide,
                Tip::new("Hide the controls", Some("H")),
            ),
            (
                Icon::Gear,
                Target::Options,
                Tip::new("Options: video, chat, controls...", Some("Ctrl+,")),
            ),
            (
                Icon::Speaker,
                Target::Sound,
                Tip::new("Sound: each channel's volume and mute", None),
            ),
        ] {
            let rect = Rect::new(right - b, cy - b * 0.5, b, b);
            let look = Look {
                active: target == Target::Sound && self.sound_menu,
                ..self.look(rect, target)
            };
            style::icon_button(painter, rect, icon, 20.0 * s, look, s);
            self.hits.push(rect, target, Some(tip));
            if target == Target::Sound {
                sound_r = rect;
            }
            right = rect.x - 4.0 * s;
        }
        right -= 10.0 * s;
        // Decide what fits: the labels of Render settings and the camera go first.
        let has_sel = selection.is_some();
        let sel_w = if has_sel {
            128.0 + 4.0 + 40.0 + 2.0 + 40.0
        } else {
            0.0
        };
        let wide = 112.0 + 10.0 + 176.0 + 14.0 + 40.0 + 4.0 + 40.0 + 10.0 + sel_w + 18.0 + 330.0;
        let available = (right - left_end) / s - 24.0;
        let compact = available < wide;
        // Export (primary).
        let export_w = 112.0 * s;
        let export_r = Rect::new(right - export_w, cy - 18.0 * s, export_w, 36.0 * s);
        let look = Look {
            primary: true,
            ..self.look(export_r, Target::Export)
        };
        style::pill(
            painter,
            fonts,
            export_r,
            Some(Icon::Export),
            "Export",
            look,
            s,
        );
        self.hits.push(
            export_r,
            Target::Export,
            Some(Tip::new(
                if has_sel {
                    "Export the selection"
                } else {
                    "Export to a video"
                },
                Some("E"),
            )),
        );
        right = export_r.x - 10.0 * s;
        // Render settings.
        let rs_w = if compact { b } else { 176.0 * s };
        let rs_r = Rect::new(right - rs_w, cy - 18.0 * s, rs_w, 36.0 * s);
        let look = self.look(rs_r, Target::RenderSettings);
        if compact {
            style::icon_button(painter, rs_r, Icon::Sliders, 19.0 * s, look, s);
        } else {
            style::pill(
                painter,
                fonts,
                rs_r,
                Some(Icon::Sliders),
                "Render settings",
                look,
                s,
            );
        }
        self.hits.push(
            rs_r,
            Target::RenderSettings,
            Some(Tip::new(
                "Shaders and mods, live over the replay",
                Some("Right Shift"),
            )),
        );
        right = rs_r.x - 14.0 * s;
        // The selection's own controls.
        if let Some((a, b_us)) = selection {
            let clear = Rect::new(right - 30.0 * s, cy - 15.0 * s, 30.0 * s, 30.0 * s);
            let look = self.look(clear, Target::ClearSelection);
            style::icon_button(painter, clear, Icon::Close, 13.0 * s, look, s);
            self.hits.push(
                clear,
                Target::ClearSelection,
                Some(Tip::new("Clear the selection", Some("Delete"))),
            );
            let preview = Rect::new(clear.x - 2.0 * s - b, cy - b * 0.5, b, b);
            let look = Look {
                active: self.preview,
                ..self.look(preview, Target::Preview)
            };
            style::icon_button(painter, preview, Icon::Loop, 19.0 * s, look, s);
            self.hits.push(
                preview,
                Target::Preview,
                Some(Tip::new(
                    if self.preview {
                        "Stop previewing"
                    } else {
                        "Preview the selection on a loop"
                    },
                    Some("P"),
                )),
            );
            let text = format::span(b_us - a);
            let ts = styles::button(s);
            let tw = fonts.width(&text, ts);
            let tx = preview.x - 6.0 * s - tw;
            fonts.shadowed_mid(painter, &text, tx, cy, ts, ACCENT_TEXT, s);
            right = tx - 10.0 * s;
        }
        for (icon, target, tip) in [
            (
                Icon::MarkOut,
                Target::MarkOut,
                Tip::new("Set Out here", Some("O")),
            ),
            (
                Icon::MarkIn,
                Target::MarkIn,
                Tip::new("Set In here", Some("I")),
            ),
        ] {
            let rect = Rect::new(right - b, cy - b * 0.5, b, b);
            let look = self.look(rect, target);
            style::icon_button(painter, rect, icon, 19.0 * s, look, s);
            self.hits.push(rect, target, Some(tip));
            right = rect.x - 4.0 * s;
        }
        let right_start = right - 14.0 * s;

        // ---- the camera, centred in what is left ----
        let modes = CameraMode::ALL;
        let cam_w = if compact { 150.0 * s } else { 330.0 * s };
        let room = (right_start - left_end).max(cam_w);
        let mut cam_x = bar.x + (bar.w - cam_w) * 0.5;
        if cam_x < left_end + 14.0 * s || cam_x + cam_w > right_start {
            cam_x = left_end + (room - cam_w) * 0.5;
        }
        let cam = Rect::new(cam_x, cy - 18.0 * s, cam_w, 36.0 * s);
        let labels: Vec<(Option<Icon>, &str)> = modes
            .iter()
            .map(|m| {
                let icon = match m {
                    CameraMode::Pov => Icon::CameraPov,
                    CameraMode::Follow => Icon::CameraFollow,
                    CameraMode::Free => Icon::CameraFree,
                };
                (Some(icon), if compact { "" } else { m.label() })
            })
            .collect();
        let hover = style::segment_at(cam, 3, s, self.cursor);
        let segments = style::segmented(
            painter,
            fonts,
            cam,
            &labels,
            self.status.camera.index(),
            hover,
            &[],
            s,
        );
        for (segment, mode) in segments.into_iter().zip(modes) {
            let tip = match mode {
                CameraMode::Pov => Tip::new("Player view: exactly as you played", Some("C")),
                CameraMode::Follow => Tip::new("Follow: third person behind you", Some("C")),
                CameraMode::Free => Tip::new("Free camera: hold right mouse to fly", Some("C")),
            };
            self.hits.push(segment, Target::Camera(mode), Some(tip));
        }

        // ---- the speed list, above the speed button ----
        if self.speed_menu {
            // Above the whole bar, so it never covers the timeline.
            let item_h = 32.0 * s;
            let w = speed_r.w.max(92.0 * s);
            let h = item_h * SPEEDS.len() as f32 + 12.0 * s;
            let menu = Rect::new(speed_r.x, bar.y - h - 10.0 * s, w, h);
            glass.rects.push(GlassRect {
                rect: [menu.x, menu.y, menu.w, menu.h],
                radius: 14.0 * s,
                alpha,
            });
            style::panel(painter, menu, 14.0 * s, 0.0, s);
            self.hits.push(menu, Target::SpeedMenu, None);
            for (i, speed) in SPEEDS.iter().enumerate().rev() {
                // Fastest at the top.
                let row = SPEEDS.len() - 1 - i;
                let item = Rect::new(
                    menu.x + 6.0 * s,
                    menu.y + 6.0 * s + item_h * row as f32,
                    menu.w - 12.0 * s,
                    item_h,
                );
                let current = (speed - self.status.speed).abs() < 1e-3;
                let hover = item.contains(self.cursor.0, self.cursor.1);
                if hover || current {
                    painter.round_rect(
                        item,
                        9.0 * s,
                        Fill::Solid(white(if hover { 0.16 } else { 0.09 })),
                    );
                }
                let text = format::speed(*speed);
                fonts.shadowed_mid(
                    painter,
                    &text,
                    item.x + 30.0 * s,
                    item.y + item_h * 0.5,
                    styles::button(s),
                    white(if current { 1.0 } else { 0.85 }),
                    s,
                );
                if current {
                    icons::draw(
                        painter,
                        Icon::Check,
                        (item.x + 15.0 * s, item.y + item_h * 0.5),
                        14.0 * s,
                        1.8 * s,
                        ACCENT_TEXT,
                    );
                }
                self.hits.push(item, Target::SpeedOption(i), None);
            }
        }

        // ---- The Sound menu, above the Sound button ----
        if self.sound_menu {
            let row_h = 40.0 * s;
            let w = 330.0 * s;
            let h = row_h * SOUND_CHANNELS.len() as f32 + 12.0 * s;
            let menu = Rect::new((sound_r.right() - w).max(bar.x), bar.y - h - 10.0 * s, w, h);
            glass.rects.push(GlassRect {
                rect: [menu.x, menu.y, menu.w, menu.h],
                radius: 14.0 * s,
                alpha,
            });
            style::panel(painter, menu, 14.0 * s, 0.0, s);
            self.hits.push(menu, Target::SoundMenu, None);
            for (i, &(label, missing)) in SOUND_CHANNELS.iter().enumerate() {
                let present = self.has_channel(i);
                let (muted, volume) = self.sound[i];
                let cy = menu.y + 6.0 * s + row_h * (i as f32 + 0.5);
                // The switch: on = heard.
                let sw = Rect::new(menu.x + 16.0 * s, cy - 10.0 * s, 38.0 * s, 20.0 * s);
                let row = Rect::new(menu.x + 8.0 * s, cy - 16.0 * s, 190.0 * s, 32.0 * s);
                let hover = present && row.contains(self.cursor.0, self.cursor.1);
                let on = if present && !muted { 1.0 } else { 0.0 };
                style::switch(painter, sw, on, hover, !present);
                fonts.shadowed_mid(
                    painter,
                    label,
                    sw.right() + 10.0 * s,
                    cy,
                    styles::body(s),
                    white(if present { 0.92 } else { 0.4 }),
                    s,
                );
                let tip = if present {
                    Tip::new(if muted { "Unmute" } else { "Mute" }, None)
                } else {
                    Tip::new(missing, None)
                };
                self.hits.push(row, Target::SoundMute(i), Some(tip));
                // - 100% +
                let side = 26.0 * s;
                let plus = Rect::new(menu.right() - 14.0 * s - side, cy - side * 0.5, side, side);
                let minus = Rect::new(plus.x - 58.0 * s - side, plus.y, side, side);
                for (rect, text, target) in [
                    (minus, "-", Target::SoundDown(i)),
                    (plus, "+", Target::SoundUp(i)),
                ] {
                    let hover = present && rect.contains(self.cursor.0, self.cursor.1);
                    let fill = if hover { 0.18 } else { 0.08 };
                    painter.round_rect(rect, 8.0 * s, Fill::Solid(white(fill)));
                    let tw = fonts.width(text, styles::button(s));
                    fonts.shadowed_mid(
                        painter,
                        text,
                        rect.x + (rect.w - tw) * 0.5,
                        cy,
                        styles::button(s),
                        white(if present { 0.95 } else { 0.4 }),
                        s,
                    );
                    if present {
                        self.hits.push(rect, target, None);
                    }
                }
                let percent = format!("{}%", (volume * 100.0).round() as i32);
                let tw = fonts.width(&percent, styles::button(s));
                fonts.shadowed_mid(
                    painter,
                    &percent,
                    minus.right() + (plus.x - minus.right() - tw) * 0.5,
                    cy,
                    styles::button(s),
                    white(if present && !muted { 0.95 } else { 0.45 }),
                    s,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn time_bubble(
        &self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        x: f32,
        bottom: f32,
        t: u64,
        s: f32,
        alpha: f32,
    ) {
        let text = format::clock_tenths(t);
        let style = TextStyle::new(12.5 * s).weight(Weight::Semibold);
        let tw = fonts.width(&text, style);
        let w = tw + 16.0 * s;
        let h = 24.0 * s;
        let rect = Rect::new(
            (x - w * 0.5).round(),
            (bottom - h).round(),
            w.round(),
            h.round(),
        );
        let saved = painter.alpha;
        painter.alpha = saved * alpha;
        painter.round_rect(rect, 7.0 * s, Fill::Solid(css(16, 20, 28, 0.92)));
        painter.round_rect_stroke(rect, 7.0 * s, 1.0, 0.5, white(0.25));
        fonts.plain_mid(
            painter,
            &text,
            rect.x + 8.0 * s,
            rect.y + h * 0.5,
            style,
            white(0.97),
        );
        painter.alpha = saved;
    }
}
