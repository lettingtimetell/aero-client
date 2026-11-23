//! The replay browser and the replay viewer's controls.
//!
//! A **simple to use** replay system: watch a recording from the main menu,
//! scrub through it, select a section, preview it with other render mods on, and export it at
//! any frame rate. This module is that UI, as plain state plus arithmetic in the style of the
//! mod menu ([`super::modmenu`]): each screen draws one frame into a [`DrawList`] in **physical
//! pixels** and returns the [`GlassFrame`] for its glass phase; input methods change the state and
//! queue events the client drains. Hit regions are recorded while drawing and used by the next
//! input event, so layout lives in exactly one place. Nothing here touches files, the network or
//! the GPU, so every screen is tested headless.
//!
//! * [`browser::ReplayBrowserScreen`]: recordings and clips in two tabs, each with its title,
//!   server, date, length and size; Watch, Rename, Delete (two clicks), Show in folder, and for
//!   clips a one-click Export with the clip settings. In one list or by folder as on disk, in
//!   the order the player picks ([`arrange`]).
//! * [`viewer::ReplayViewerOverlay`]: the controls over a replay: a timeline along the bottom to
//!   drag (hover shows the time), play/pause, speed, ±5 s, step, the camera, In/Out with the
//!   selection drawn on the timeline, Preview selection (loops it), Render settings (the mod
//!   menu over the replay), Export, hide the controls, Exit. Every button's tooltip names its key.
//! * [`export::ExportDialog`]: range, resolution, frame rate (presets up to 2000 plus any typed
//!   number), quality or a maximum file size, format, motion blur, sound and microphone, the file
//!   name; then a progress view with the time left and Cancel.
//!
//! The client half is `mc-client/src/app/mods/tw_replay_view.rs`: it fills the view models
//! below from the library, the playback engine and the export jobs, and turns
//! the events into calls on them.
//!
//! [`DrawList`]: crate::widget::DrawList
//! [`GlassFrame`]: crate::screens::modmenu::glass::GlassFrame

// The browser's folders and sort orders.
pub mod arrange;
pub mod browser;
pub mod export;
pub mod format;
// The browser's day groups and short recordings.
pub mod groups;
pub mod icons;
pub mod style;
pub mod viewer;

// The screens' behaviour tests are `mc-ui/tests/tw_replay_ui_logic.rs` (headless) and
// `tw_replay_ui_render.rs` (GPU pictures).

use std::time::Instant;

use crate::font::FontRenderer;
use crate::screens::modmenu::text::MenuText;

pub use browser::{BrowserEvent, ReplayBrowserScreen};
pub use export::{ExportDialog, ExportEvent, ExportProgress, ExportRequest, ExportSettings};
pub use viewer::{ReplayViewerOverlay, ViewerEvent};

/// LWJGL key codes the replay screens use (beyond `widget::keys`).
pub mod key {
    /// Esc.
    pub const ESCAPE: i32 = 1;
    /// `-` (speed down).
    pub const MINUS: i32 = 12;
    /// `=` (speed up).
    pub const EQUALS: i32 = 13;
    /// Backspace.
    pub const BACKSPACE: i32 = 14;
    /// Tab (next tab in the browser).
    pub const TAB: i32 = 15;
    /// E (export).
    pub const E: i32 = 18;
    /// R (rename).
    pub const R: i32 = 19;
    /// `[` (the previous bookmark).
    pub const LBRACKET: i32 = 26;
    /// `]` (the next bookmark).
    pub const RBRACKET: i32 = 27;
    /// S (show or hide short recordings in the browser).
    pub const S: i32 = 31;
    /// F (the browser's List or Folders view).
    pub const F: i32 = 33;
    /// I (set In).
    pub const I: i32 = 23;
    /// O (set Out; in the browser, the Sort menu).
    pub const O: i32 = 24;
    /// P (preview the selection).
    pub const P: i32 = 25;
    /// Enter.
    pub const ENTER: i32 = 28;
    /// A (select all, with Ctrl).
    pub const A: i32 = 30;
    /// H (hide the controls).
    pub const H: i32 = 35;
    /// K (play/pause).
    pub const K: i32 = 37;
    /// C (camera).
    pub const C: i32 = 46;
    /// V (paste, with Ctrl).
    pub const V: i32 = 47;
    /// `,` (step back).
    pub const COMMA: i32 = 51;
    /// `.` (step forward).
    pub const PERIOD: i32 = 52;
    /// Space (play/pause).
    pub const SPACE: i32 = 57;
    /// F2 (rename, as in Explorer).
    pub const F2: i32 = 60;
    /// Numpad `-`.
    pub const SUBTRACT: i32 = 74;
    /// Numpad `+`.
    pub const ADD: i32 = 78;
    /// Numpad Enter.
    pub const NUMPAD_ENTER: i32 = 156;
    /// Home.
    pub const HOME: i32 = 199;
    /// Up.
    pub const UP: i32 = 200;
    /// Page Up.
    pub const PAGE_UP: i32 = 201;
    /// Left.
    pub const LEFT: i32 = 203;
    /// Right.
    pub const RIGHT: i32 = 205;
    /// End.
    pub const END: i32 = 207;
    /// Down.
    pub const DOWN: i32 = 208;
    /// Page Down.
    pub const PAGE_DOWN: i32 = 209;
    /// Delete.
    pub const DELETE: i32 = 211;
}

/// What the client hands a screen's `build` each frame.
pub struct ReplayEnv<'a> {
    /// Segoe UI (or Minecraft-font) text.
    pub text: &'a mut MenuText,
    /// The Minecraft font (fallback text and measuring).
    pub mc_font: &'a mut FontRenderer,
    /// Window size in physical pixels.
    pub screen: (f32, f32),
    /// Real time now (animations, tooltips, the caret).
    pub now: Instant,
    /// The glass `uTime` in seconds.
    pub time_s: f32,
}

/// One recording or clip in the browser (the client converts the library entries).
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayEntry {
    /// The client's id for it (stable while the browser is open).
    pub id: u64,
    /// The title (the file's own, or the player's rename).
    pub title: String,
    /// The server address, for example `mc.example.net`.
    pub server: String,
    /// When it was recorded, in milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// Its length in microseconds (a clip: the clip window's length).
    pub duration_us: u64,
    /// The file's size in bytes.
    pub size_bytes: u64,
    /// A clip rather than a full recording.
    pub clip: bool,
    /// It holds a microphone track.
    pub has_mic: bool,
    /// Why it cannot be watched (an unreadable or newer file), if so.
    pub problem: Option<String>,
    /// Bookmarks set while recording (the Bookmark key).
    pub bookmarks: u32,
    /// The folder it is in under its tab's root (`replays\` for recordings, `replays\clips\` for
    /// clips): the folder names on disk joined by `/`, `""` for the root itself (the browser's
    /// Folders view). See [`arrange`].
    pub folder: String,
    /// The account that recorded it (the Account order), `""` when unknown.
    pub player: String,
}

/// The library as the browser shows it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LibraryView {
    /// Full recordings, newest first.
    pub recordings: Vec<ReplayEntry>,
    /// Clips, newest first.
    pub clips: Vec<ReplayEntry>,
    /// Every folder the listing found under the recordings' root, empty ones too, as
    /// [`ReplayEntry::folder`] keys.
    pub recording_folders: Vec<String>,
    /// The same under the clips' root.
    pub clip_folders: Vec<String>,
    /// The storage limit for full recordings (the `replay` mod's "Replay storage limit"), bytes.
    pub cap_bytes: u64,
    /// Whether recording is switched on (the `replay` mod), for the empty-state hint.
    pub recording_on: bool,
    /// Whether a clip key is set, for the empty clips tab's hint.
    pub clip_key_set: bool,
}

/// The camera a replay is watched through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraMode {
    /// The recorded first-person view, exactly as it was played.
    #[default]
    Pov,
    /// A third-person camera following the player.
    Follow,
    /// A free camera (hold the right mouse button to fly).
    Free,
}

impl CameraMode {
    /// In the order the camera control shows them.
    pub const ALL: [CameraMode; 3] = [CameraMode::Pov, CameraMode::Follow, CameraMode::Free];

    /// The short label on the camera control.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            CameraMode::Pov => "Player",
            CameraMode::Follow => "Follow",
            CameraMode::Free => "Free",
        }
    }

    /// The next mode (the C key).
    #[must_use]
    pub fn next(self) -> CameraMode {
        match self {
            CameraMode::Pov => CameraMode::Follow,
            CameraMode::Follow => CameraMode::Free,
            CameraMode::Free => CameraMode::Pov,
        }
    }

    /// Its index in [`CameraMode::ALL`].
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            CameraMode::Pov => 0,
            CameraMode::Follow => 1,
            CameraMode::Free => 2,
        }
    }
}

/// The speeds the speed control offers (playback runs from 0.1x to 8x).
pub const SPEEDS: [f32; 8] = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0];

/// The playback state the client reports every frame (cheap to copy: no strings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackStatus {
    /// The replay's (or clip window's) length, microseconds.
    pub duration_us: u64,
    /// The position, microseconds from the start of the timeline.
    pub time_us: u64,
    /// Playing (not paused).
    pub playing: bool,
    /// The speed.
    pub speed: f32,
    /// The camera in use.
    pub camera: CameraMode,
    /// A seek is rebuilding the world: its progress, `0..=1`.
    pub seeking: Option<f32>,
    /// The free camera is flying (the right mouse button is held).
    pub flying: bool,
}

impl Default for PlaybackStatus {
    fn default() -> Self {
        PlaybackStatus {
            duration_us: 0,
            time_us: 0,
            playing: false,
            speed: 1.0,
            camera: CameraMode::Pov,
            seeking: None,
            flying: false,
        }
    }
}

/// What the viewer shows about the replay it is playing (set once when it opens).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReplayInfo {
    /// The title.
    pub title: String,
    /// The server address.
    pub server: String,
    /// When it was recorded, milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// A clip.
    pub clip: bool,
    /// It holds a microphone track.
    pub has_mic: bool,
    /// Markers on the timeline: time (microseconds from the timeline's start) and label.
    pub markers: Vec<(u64, String)>,
    /// The window size in pixels (the export dialog's "Window size" choice).
    pub window_size: (u32, u32),
}
