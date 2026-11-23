//! Now Playing (id `spotify`, its name until 2026-11-22; searching "spotify" still finds it): a
//! now-playing HUD widget with cover art, progress and playback
//! controls.
//!
//! Keep the `id` stable (it is the settings-file key); the `category` line must name the
//! mod's home in `crate::menu_layout`, which decides where the menu lists it (a test keeps the
//! two in step).
//!
//! * The data source is the `mc-media` crate: the Windows media session, which needs
//!   no login and works with Spotify Free. The client starts it only while this mod is on.
//! * The widget is `mc-ui/src/modhud/elements/spotify.rs`; the client glue (hub lifecycle, the
//!   cover texture, clicks and keys) is `mc-client/src/app/mods/w9_spotify.rs`.
//! * The buttons are clickable while chat is open (a free cursor over an undimmed HUD, where a
//!   click outside the chat box does nothing in vanilla). The three keys are optional and
//!   unbound by default; they work in game only.

use crate::category::Category;
use crate::descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue,
};
use crate::hud::{HudAnchor, HudPlacement};

/// Option ids (the widget and the client read them by these names).
pub mod ids {
    /// `0` Minecraft, `1` Aero (the mod menu's glass look).
    pub const STYLE: &str = "style";
    /// When the control row shows: `0` always, `1` only while chat is open, `2` never.
    pub const CONTROLS: &str = "controls";
    /// The background panel's opacity as a percentage ("Background opacity"; the id is older
    /// than that label, from when it faded the whole widget).
    pub const OPACITY: &str = "opacity";
    /// Draw the album cover.
    pub const SHOW_COVER: &str = "show_cover";
    /// Draw the progress bar and the times.
    pub const SHOW_PROGRESS: &str = "show_progress";
    /// `0` Spotify's desktop app only, `1` whatever Windows considers the current player.
    pub const SOURCE: &str = "source";
    /// Optional play/pause key (LWJGL code, `0` = unbound).
    pub const KEY_PLAY_PAUSE: &str = "key_play_pause";
    /// Optional next-track key.
    pub const KEY_NEXT: &str = "key_next";
    /// Optional previous-track key.
    pub const KEY_PREVIOUS: &str = "key_previous";
}

/// The `style` choices, in index order.
pub const STYLES: &[&str] = &["Minecraft", "Aero"];
/// The `controls` choices, in index order.
pub const CONTROLS: &[&str] = &["Always", "Chat open", "Never"];
/// The `source` choices, in index order.
pub const SOURCES: &[&str] = &["Spotify", "Any player"];

static OPTIONS: [OptionDescriptor; 9] = [
    OptionDescriptor {
        id: ids::STYLE,
        label: "Style",
        kind: OptionKind::Choice { choices: STYLES },
        default: OptionValue::Choice(0),
        description: "Minecraft: the font and a dark panel like chat. Aero: the menu's glass.",
    },
    OptionDescriptor {
        id: ids::OPACITY,
        label: "Background opacity",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How see-through the panel behind the widget is. The text, cover, \
                      progress bar and controls always stay fully visible.",
    },
    OptionDescriptor {
        id: ids::CONTROLS,
        label: "Controls",
        kind: OptionKind::Choice { choices: CONTROLS },
        default: OptionValue::Choice(0),
        description: "Previous, play/pause, next and volume. Open chat to click them.",
    },
    OptionDescriptor {
        id: ids::SHOW_COVER,
        label: "Album cover",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: ids::SHOW_PROGRESS,
        label: "Progress bar",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "",
    },
    OptionDescriptor {
        id: ids::SOURCE,
        label: "Source",
        kind: OptionKind::Choice { choices: SOURCES },
        default: OptionValue::Choice(0),
        description: "Spotify only, or whatever Windows shows in its media controls.",
    },
    OptionDescriptor {
        id: ids::KEY_PLAY_PAUSE,
        label: "Play/pause key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "Optional. Works in game, not while typing.",
    },
    OptionDescriptor {
        id: ids::KEY_NEXT,
        label: "Next track key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "",
    },
    OptionDescriptor {
        id: ids::KEY_PREVIOUS,
        label: "Previous track key",
        kind: OptionKind::Keybind,
        default: OptionValue::Key(0),
        description: "",
    },
];

static HUD: [HudElementDescriptor; 1] = [HudElementDescriptor {
    id: "spotify",
    name: "Now Playing",
    // Bottom right, clear of the chat input line (the bottom 14 GUI pixels while chat is
    // open, which is when the buttons are clickable).
    default_placement: HudPlacement::new(HudAnchor::BottomRight, -2.0, -16.0),
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "spotify",
    name: "Now Playing",
    category: Category::Hud,
    description: "The song that's playing, with the cover and progress: from Spotify, or from \
                  whatever player your media controls show. Open chat to click play/pause, skip \
                  or the volume.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &HUD,
};
