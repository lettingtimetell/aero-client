//! Replay (id `replay`): record games, save clips and export them to video.
//!
//! One option list for the recorder, the viewer and the video export:
//! - the recorder uses the recording, clip and storage options;
//! - the viewer reads `open_viewer`;
//! - the video export reads the `clip_*` and `export_*` options.
//!
//! Keep every `id` stable: they are the settings-file keys. Off by default: nothing is recorded
//! until the player turns the mod on.
//! The viewer works whether or not the mod is on.

use std::borrow::Cow;

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// Option ids, shared by the client and the menu.
pub mod ids {
    /// Start recording on every server join.
    pub const AUTO_START: &str = "auto_start";
    /// Key that starts a recording.
    pub const KEY_START: &str = "key_start";
    /// Key that pauses or resumes the recording.
    pub const KEY_PAUSE: &str = "key_pause";
    /// Key that stops and saves the recording.
    pub const KEY_STOP: &str = "key_stop";
    /// Small red dot while recording.
    pub const INDICATOR: &str = "indicator";
    /// Record the microphone.
    pub const MIC: &str = "mic";
    /// Microphone volume in the replay.
    pub const MIC_VOLUME: &str = "mic_volume";
    /// Key that saves the last `clip_length` seconds as a clip.
    pub const KEY_CLIP: &str = "key_clip";
    /// Clip length in seconds.
    pub const CLIP_LENGTH: &str = "clip_length";
    /// Largest file a clip export may produce.
    pub const CLIP_MAX_SIZE: &str = "clip_max_size";
    /// Export clips to video automatically.
    pub const CLIP_AUTO_EXPORT: &str = "clip_auto_export";
    /// Clip export frame rate.
    pub const CLIP_FPS: &str = "clip_fps";
    /// Clip export resolution.
    pub const CLIP_RESOLUTION: &str = "clip_resolution";
    /// Default export frame rate.
    pub const EXPORT_FPS: &str = "export_fps";
    /// Default export resolution.
    pub const EXPORT_RESOLUTION: &str = "export_resolution";
    /// Default export quality.
    pub const EXPORT_QUALITY: &str = "export_quality";
    /// Default export codec.
    pub const EXPORT_CODEC: &str = "export_codec";
    /// Frames blended into each exported frame.
    pub const EXPORT_MOTION_BLUR: &str = "export_motion_blur";
    /// Include game sound in exports.
    pub const EXPORT_AUDIO: &str = "export_audio";
    /// Include the microphone in exports.
    pub const EXPORT_MIC: &str = "export_mic";
    /// Storage cap for full recordings, in GB.
    pub const MAX_STORAGE_GB: &str = "max_storage_gb";
    /// Button that opens the replay viewer.
    pub const OPEN_VIEWER: &str = "open_viewer";
    // Each replay lane adds its option ids at its own marker.
    /// Key that bookmarks the moment while recording.
    pub const KEY_BOOKMARK: &str = "key_bookmark";
    /// Show the screens you had open (inventory, chests, chat, menus) and your mouse pointer
    /// when you watch a replay in the Player camera.
    pub const SHOW_SCREENS: &str = "show_screens";
    /// How much of the chat you were typing a recording keeps ([`super::CHAT_TEXT_CHOICES`]).
    pub const CHAT_TEXT: &str = "chat_text";
    /// Which video encoder exports use: the graphics card's when it has one, or only the
    /// processor.
    pub const EXPORT_ENCODER: &str = "export_encoder";
    /// Record other programs' sound (desktop audio) as its own channel.
    pub const DESKTOP_AUDIO: &str = "desktop_audio";
    /// Which programs desktop audio records ([`super::DESKTOP_SOURCE_CHOICES`]).
    pub const DESKTOP_SOURCE: &str = "desktop_source";
    /// Desktop audio volume in replays and exports.
    pub const DESKTOP_VOLUME: &str = "desktop_volume";
    /// Game sound volume in replays and exports.
    pub const GAME_VOLUME: &str = "game_volume";
    /// Include desktop audio in exports.
    pub const EXPORT_DESKTOP: &str = "export_desktop";
    /// Exports also get one audio track per channel after the mixed one.
    pub const EXPORT_TRACKS: &str = "export_tracks";
    // The replay browser's order and view, remembered (not drawn in the menu).
    /// The browser's sort order (`newest`, `oldest`, `name`, `server`, `longest`, `largest`,
    /// `bookmarks`, `account`).
    pub const BROWSER_SORT: &str = "browser_sort";
    /// The browser's view (`list` or `folders`).
    pub const BROWSER_VIEW: &str = "browser_view";
}

/// `chat_text` choices: what a recording keeps of the chat you were typing.
pub const CHAT_TEXT_CHOICES: &[&str] = &["Everything", "Hide commands", "Nothing"];

/// `desktop_source` choices: 0 every program except the game, 1 the Discord app.
pub const DESKTOP_SOURCE_CHOICES: &[&str] = &["All other programs", "Discord only"];

/// `clip_max_size` choices; [`CLIP_MAX_SIZE_BYTES`] has the byte limits in the same order.
pub const CLIP_MAX_SIZE_CHOICES: &[&str] = &[
    "No limit", "8 MB", "10 MB", "25 MB", "50 MB", "100 MB", "500 MB",
];

/// Byte limits for [`CLIP_MAX_SIZE_CHOICES`] (`None` = no limit). Upload limits count MB as 10^6
/// bytes in some places and 2^20 in others, so the limits use the smaller, 10^6.
pub const CLIP_MAX_SIZE_BYTES: &[Option<u64>] = &[
    None,
    Some(8_000_000),
    Some(10_000_000),
    Some(25_000_000),
    Some(50_000_000),
    Some(100_000_000),
    Some(500_000_000),
];

/// `clip_fps` choices.
pub const CLIP_FPS_CHOICES: &[&str] = &["30", "60", "120", "144", "240"];

/// `clip_resolution` and `export_resolution` choices.
pub const RESOLUTION_CHOICES: &[&str] = &["Window size", "720p", "1080p", "1440p", "4K"];

/// `export_fps` choices. The export dialog also takes any number typed in.
pub const EXPORT_FPS_CHOICES: &[&str] = &[
    "24", "30", "60", "120", "144", "240", "360", "480", "1000", "2000",
];

/// `export_quality` choices.
pub const EXPORT_QUALITY_CHOICES: &[&str] = &["Small file", "Balanced", "High", "Near lossless"];

/// `export_codec` choices.
pub const EXPORT_CODEC_CHOICES: &[&str] = &["H.264 (plays everywhere)", "H.265 (smaller files)"];

/// `export_motion_blur` choices: frames rendered and blended per exported frame.
pub const EXPORT_MOTION_BLUR_CHOICES: &[&str] = &["Off", "2x", "4x", "8x", "16x", "32x"];

/// The options, in menu order.
static OPTIONS: &[OptionDescriptor] = &[
    // ---- Recording ----
    OptionDescriptor {
        id: ids::AUTO_START,
        label: "Record every game",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Records each game: a new file every time you join a server or switch to a \
                      new game. Off: only the start key records.",
    },
    OptionDescriptor {
        id: ids::KEY_START,
        label: "Start recording key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "Starts a recording, or resumes a paused one. Not set by default.",
    },
    OptionDescriptor {
        id: ids::KEY_PAUSE,
        label: "Pause recording key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "Pauses the recording, and resumes it when pressed again. Not set by default.",
    },
    OptionDescriptor {
        id: ids::KEY_STOP,
        label: "Stop recording key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "Stops and saves the recording. Not set by default.",
    },
    OptionDescriptor {
        id: ids::INDICATOR,
        label: "Recording dot",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Shows a small red dot in the top right corner while recording, and two \
                      bars while paused. Hidden with F1 and F3.",
    },
    OptionDescriptor {
        id: ids::MIC,
        label: "Record microphone",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Records your default microphone while a recording runs, for commentary. \
                      Silence is left out.",
    },
    OptionDescriptor {
        id: ids::MIC_VOLUME,
        label: "Microphone volume",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How loud the microphone is in replays and exports.",
    },
    // ---- Clips ----
    OptionDescriptor {
        id: ids::KEY_CLIP,
        label: "Clip key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "Saves the last moments of play as a clip. Not set by default; while it is \
                      set, the game keeps the current server's recording in memory.",
    },
    OptionDescriptor {
        id: ids::CLIP_LENGTH,
        label: "Clip length",
        kind: OptionKind::Slider {
            min: 5.0,
            max: 300.0,
            step: 5.0,
            unit: " s",
            decimals: 0,
        },
        default: OptionValue::Float(30.0),
        description: "How many seconds before the key press a clip keeps.",
    },
    OptionDescriptor {
        id: ids::CLIP_MAX_SIZE,
        label: "Clip max file size",
        kind: OptionKind::Choice {
            choices: CLIP_MAX_SIZE_CHOICES,
        },
        default: OptionValue::Choice(2),
        description: "Exported clips stay under this size, so they fit an upload limit.",
    },
    OptionDescriptor {
        id: ids::CLIP_AUTO_EXPORT,
        label: "Export clips to video",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Turns each clip into a video file by itself when you are back on the title \
                      screen. Off: export clips from the replay viewer.",
    },
    OptionDescriptor {
        id: ids::CLIP_FPS,
        label: "Clip frame rate",
        kind: OptionKind::Choice {
            choices: CLIP_FPS_CHOICES,
        },
        default: OptionValue::Choice(1),
        description: "Frames per second of exported clips.",
    },
    OptionDescriptor {
        id: ids::CLIP_RESOLUTION,
        label: "Clip resolution",
        kind: OptionKind::Choice {
            choices: RESOLUTION_CHOICES,
        },
        default: OptionValue::Choice(2),
        description: "Size of exported clips.",
    },
    // ---- Export defaults ----
    OptionDescriptor {
        id: ids::EXPORT_FPS,
        label: "Export frame rate",
        kind: OptionKind::Choice {
            choices: EXPORT_FPS_CHOICES,
        },
        default: OptionValue::Choice(2),
        description: "The frame rate the export window starts with. Any rate can be typed there, \
                      up to 2000 and beyond.",
    },
    OptionDescriptor {
        id: ids::EXPORT_RESOLUTION,
        label: "Export resolution",
        kind: OptionKind::Choice {
            choices: RESOLUTION_CHOICES,
        },
        default: OptionValue::Choice(2),
        description: "The size the export window starts with.",
    },
    OptionDescriptor {
        id: ids::EXPORT_QUALITY,
        label: "Export quality",
        kind: OptionKind::Choice {
            choices: EXPORT_QUALITY_CHOICES,
        },
        default: OptionValue::Choice(2),
        description: "The quality the export window starts with. Higher makes bigger files.",
    },
    OptionDescriptor {
        id: ids::EXPORT_CODEC,
        label: "Export format",
        kind: OptionKind::Choice {
            choices: EXPORT_CODEC_CHOICES,
        },
        default: OptionValue::Choice(0),
        description: "H.264 plays everywhere. H.265 makes smaller files.",
    },
    OptionDescriptor {
        id: ids::EXPORT_MOTION_BLUR,
        label: "Export motion blur",
        kind: OptionKind::Choice {
            choices: EXPORT_MOTION_BLUR_CHOICES,
        },
        default: OptionValue::Choice(0),
        description: "Renders several frames for each exported one and blends them, for smooth \
                      motion blur.",
    },
    OptionDescriptor {
        id: ids::EXPORT_AUDIO,
        label: "Export game sound",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Includes the game's sound in exported videos.",
    },
    OptionDescriptor {
        id: ids::EXPORT_MIC,
        label: "Export microphone",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Includes the recorded microphone in exported videos.",
    },
    // ---- Storage ----
    OptionDescriptor {
        id: ids::MAX_STORAGE_GB,
        label: "Replay storage limit",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 200.0,
            step: 1.0,
            unit: " GB",
            decimals: 0,
        },
        default: OptionValue::Float(20.0),
        description: "When full recordings use more than this, the oldest are deleted. A \
                      recording that reaches it is saved and stopped. Clips are never deleted.",
    },
    OptionDescriptor {
        id: ids::OPEN_VIEWER,
        label: "Replay viewer",
        kind: OptionKind::Button { label: "Open" },
        default: OptionValue::Action,
        description: "Watch, cut and export your replays and clips.",
    },
    // Each replay lane adds its options at its own marker (a slice now, so
    // there is no length to keep in step). Menu order is decided by the groups in menu_rules.rs.
    OptionDescriptor {
        id: ids::KEY_BOOKMARK,
        label: "Bookmark key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "Bookmarks the moment while recording, so you can find it again: the replay \
                      list counts them and the viewer jumps between them with [ and ]. Not set \
                      by default.",
    },
    OptionDescriptor {
        id: ids::SHOW_SCREENS,
        label: "Menus and chat in replays",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Shows the screens you had open while recording (inventory, chests, chat, \
                      the pause menu) and your mouse pointer when you watch a replay in the \
                      Player camera, and in its exports. Off shows only the game.",
    },
    OptionDescriptor {
        id: ids::CHAT_TEXT,
        label: "Typed chat in recordings",
        kind: OptionKind::Choice {
            choices: CHAT_TEXT_CHOICES,
        },
        default: OptionValue::Choice(1),
        description: "What a recording keeps of the chat you type before you send it. Hide \
                      commands keeps only a command's name, so passwords (/login) and private \
                      messages (/msg) stay out of replays you share. Nothing shows the chat box \
                      empty.",
    },
    OptionDescriptor {
        id: ids::EXPORT_ENCODER,
        label: "Video encoder",
        kind: OptionKind::Choice {
            choices: &["Automatic", "Processor only"],
        },
        default: OptionValue::Choice(0),
        description: "Which part of the PC encodes exported videos and clips. Automatic uses the \
                      graphics card's encoder when it has one (much faster). Processor only is \
                      slower but avoids graphics-driver encoder problems: try it if exporting \
                      fails or crashes on this PC.",
    },
    OptionDescriptor {
        id: ids::DESKTOP_AUDIO,
        label: "Record desktop audio",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Records other programs' sound, such as a voice call or music, as its own \
                      channel next to the game and your microphone. Off by default: it records \
                      other people's voices, so ask them before you share. Needs Windows 10 \
                      version 2004 or newer.",
    },
    OptionDescriptor {
        id: ids::DESKTOP_SOURCE,
        label: "Desktop audio source",
        kind: OptionKind::Choice {
            choices: DESKTOP_SOURCE_CHOICES,
        },
        default: OptionValue::Choice(0),
        description: "Which programs desktop audio records: every program except the game, or \
                      only the Discord app.",
    },
    OptionDescriptor {
        id: ids::DESKTOP_VOLUME,
        label: "Desktop audio volume",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How loud desktop audio is in replays and exports.",
    },
    OptionDescriptor {
        id: ids::GAME_VOLUME,
        label: "Game sound volume",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 200.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How loud the game's own sounds are in replays and exports. While watching, \
                      it cannot go above your sound settings.",
    },
    OptionDescriptor {
        id: ids::EXPORT_DESKTOP,
        label: "Export desktop audio",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Includes recorded desktop audio in exported videos.",
    },
    OptionDescriptor {
        id: ids::EXPORT_TRACKS,
        label: "Separate audio tracks",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Adds one audio track per channel (game, microphone, desktop audio) after \
                      the mixed track, for video editors. Players and chat apps play the mixed \
                      track.",
    },
    // Not drawn: the replay browser's order and view, as the player left
    // them (the browser's own Sort menu and List / Folders switch change them).
    OptionDescriptor {
        id: ids::BROWSER_SORT,
        label: "Replay list order",
        kind: OptionKind::HiddenText { max_len: 16 },
        default: OptionValue::Text(Cow::Borrowed("newest")),
        description: "",
    },
    OptionDescriptor {
        id: ids::BROWSER_VIEW,
        label: "Replay list view",
        kind: OptionKind::HiddenText { max_len: 16 },
        default: OptionValue::Text(Cow::Borrowed("list")),
        description: "",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "replay",
    name: "Replay",
    category: Category::Replay,
    description: "Records your games so you can watch them again from any angle, then export \
                  them to video at any frame rate. The clip key saves the last moments of play.",
    default_enabled: false,
    options: OPTIONS,
    hud_elements: &[],
};
