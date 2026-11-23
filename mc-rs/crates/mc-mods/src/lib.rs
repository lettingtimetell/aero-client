#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! The PvP mod framework's data half: what mods exist, what options they have, the user's
//! settings for them, and where their HUD elements sit.
//!
//! Everything here is plain data plus arithmetic. Nothing draws, nothing touches the GPU and
//! nothing knows about the client, so the menu (`mc-ui::screens::modmenu`), the HUD pipeline
//! (`mc-ui::modhud`) and the client (`mc-client::app::mods`) can all depend on it, and every
//! rule is unit-testable on its own.
//!
//! | Module | What it holds |
//! |---|---|
//! | [`category`] | [`Category`]: a mod's home category (the same as in [`menu_layout`]) |
//! | [`descriptor`] | [`ModDescriptor`], [`OptionDescriptor`], [`OptionKind`], [`OptionValue`] |
//! | [`presets`] | Ready-made option descriptors shared by the HUD mods (text colour, shadow...) |
//! | [`catalog`] | Every mod, in menu order. One file per mod under [`mods`] |
//! | [`settings`] | [`ModSettings`]: switches, options, HUD placements, profiles, menu key, JSON |
//! | [`hud`] | [`HudAnchor`], [`HudPlacement`], [`HudRect`] and the placement maths |
//! | [`dynamic`] | Runtime menu rows ([`DynamicRows`]) and menu events ([`ModEvent`]) |
//! | [`hud_style`] | The shared HUD style: one text style for every readout ("HUD style / Custom") |
//! | [`profiles`] | Whole-setup profiles: save and load every mod's settings under a name |
//! | [`menu_layout`] | The menu's categories, sections, one-line summaries and search keywords |
//! | [`menu_rules`] | The settings window's hide/dim rules, named groups and special layouts |
//! | [`menu_search`] | The menu's search: both spellings, optional spaces, keywords, ranking |
//!
//! # Rules every mod follows
//!
//! * **Off by default.** `default_enabled` is `false` for every mod in the catalog except
//!   `raw_input` and `borderless_fullscreen`, whose ON state is today's behaviour, and the mods
//!   named after a vanilla feature (`fog`, `vignette`, `tall_grass`, `dynamic_fov`), whose ON
//!   state is vanilla. One exception by the user's request: `aero_title`, the custom title
//!   screen, is on by default and its OFF state is vanilla.
//! * **Off means vanilla.** A disabled mod must not change a pixel, a packet or the timing.
//! * **Hot paths read a snapshot, never the store.** Keep a small typed struct with the values
//!   you need and refresh it only when [`ModSettings::generation`] changes. See `API.md`.

pub mod catalog;
pub mod category;
pub mod descriptor;
pub mod dynamic;
pub mod hud;
// The shared HUD style and whole-setup profiles.
pub mod hud_style;
pub mod profiles;
// The mod menu's layout and rules tables.
pub mod menu_layout;
pub mod menu_rules;
pub mod menu_search;
pub mod mods;
pub mod presets;
pub mod settings;

pub use category::Category;
pub use descriptor::{
    HudElementDescriptor, ModDescriptor, OptionDescriptor, OptionKind, OptionValue, SliderLabel,
};
pub use dynamic::{DynamicRow, DynamicRows, ModEvent, RowButton, RowIcon};
pub use hud::{
    placement_from_rect, resolve_rect, snap_rect, snap_rect_with, HudAnchor, HudPlacement, HudRect,
    SnapGuides, HUD_MAX_SCALE, HUD_MIN_SCALE,
};
pub use settings::{
    LoadReport, ModProfile, ModSettings, DEFAULT_MENU_KEY, MAX_PROFILES, PROFILE_NAME_MAX,
};

/// The settings file, relative to the game directory (`--gameDir`).
pub const SETTINGS_FILE_NAME: &str = "mc-rs-mods.json";
