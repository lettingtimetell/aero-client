//! [`ModSettings`]: the runtime store for every mod's enabled flag, option values and HUD
//! placements, plus its tolerant JSON form (`<gameDir>/mc-rs-mods.json`).
//!
//! # Reading settings on a hot path
//!
//! Every getter here is a short linear search over the catalog plus a string compare. That is
//! fine for the menu and for a refresh, and **not** fine per frame or per particle. Keep a
//! typed snapshot instead and refresh it when [`ModSettings::generation`] moves:
//!
//! ```
//! use mc_mods::ModSettings;
//!
//! #[derive(Default)]
//! struct FpsSnapshot {
//!     seen: u64, // 0 = never synced; the store's generation starts at 1
//!     enabled: bool,
//!     shadow: bool,
//! }
//!
//! impl FpsSnapshot {
//!     fn sync(&mut self, settings: &ModSettings) {
//!         if self.seen == settings.generation() {
//!             return; // the common case: one u64 compare
//!         }
//!         self.seen = settings.generation();
//!         self.enabled = settings.enabled("fps");
//!         self.shadow = settings.bool("fps", "text_shadow");
//!     }
//! }
//!
//! let mut settings = ModSettings::new();
//! let mut snap = FpsSnapshot::default();
//! snap.sync(&settings);
//! assert!(!snap.enabled);
//! settings.set_enabled("fps", true);
//! snap.sync(&settings);
//! assert!(snap.enabled);
//! ```
//!
//! # The file
//!
//! ```json
//! {
//!   "format": 1,
//!   "menu": { "open_key": 54 },
//!   "mods": {
//!     "fps": {
//!       "enabled": true,
//!       "options": { "text_color": "#FFFFFFFF", "text_shadow": true },
//!       "hud": { "fps": { "anchor": "top_left", "x": 2.0, "y": 2.0, "scale": 1.0 } }
//!     },
//!     "color_grading": {
//!       "enabled": true,
//!       "options": { "saturation": 1.2 },
//!       "profiles": [ { "name": "Warm", "options": { "saturation": 1.2 } } ]
//!     }
//!   }
//! }
//! ```
//!
//! * `menu.open_key`: the key that opens and closes the mod menu, as an LWJGL code. Missing means
//!   Right Shift ([`DEFAULT_MENU_KEY`]).
//! * `profiles`: named sets of one mod's option values (colour grading profiles,
//!   [`ModProfile`]). Written only when a mod has some.
//! * `pinned`: a top-level list of mod ids pinned to the menu's My setup, e.g.
//!   `"pinned": ["zoom", "fps"]`. Written only when something is pinned;
//!   older files lack it and load with no pins. See [`ModSettings::pinned_at`].
//!
//! Loading never fails: a missing file, a missing key or an unknown key leaves the default; a
//! wrong-typed or out-of-range value is coerced or clamped ([`OptionKind::sanitize`]) and noted
//! in the [`LoadReport`]. Keys this build does not know (a mod, an option, a key inside a mod's
//! entry or at the root, from a newer or older build) are kept and written back, so switching
//! builds does not lose them.

use std::borrow::Cow;
use std::path::Path;

use serde_json::{Map, Number, Value};

use crate::catalog;
use crate::descriptor::{ModDescriptor, OptionKind, OptionValue};
use crate::hud::{HudAnchor, HudPlacement};
// The shared HUD style and whole-setup
// profiles.
use crate::hud_style;
use crate::profiles::SetupDiff;

/// The `"format"` number written to the file.
pub const FORMAT_VERSION: u64 = 1;

/// The key that opens and closes the mod menu until the user picks another: Right Shift
/// (LWJGL 54). Neither half of the old Shift+Tab is a key vanilla leaves alone
/// (Shift sneaks, Tab shows the player list); Right Shift is unbound in vanilla.
pub const DEFAULT_MENU_KEY: i32 = 54;

/// The most profiles the menu lets one mod save (a file with more keeps them all).
pub const MAX_PROFILES: usize = 12;

/// The longest profile name, in characters.
pub const PROFILE_NAME_MAX: usize = 24;

/// A named set of one mod's option values: a colour grading profile.
///
/// It holds every persisted option of the mod, so loading it reproduces the same look whatever
/// the current values are; an option the profile does not mention (one added after it was saved)
/// loads as its default.
#[derive(Debug, Clone, PartialEq)]
pub struct ModProfile {
    name: String,
    /// One value per option of the mod, aligned with its descriptor's `options`.
    values: Vec<OptionValue>,
    /// Option ids this build does not know, written back as they were.
    unknown: Map<String, Value>,
}

impl ModProfile {
    /// The name the user gave it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The stored value of the option at `option_index` in the mod's descriptor.
    #[must_use]
    pub fn value(&self, option_index: usize) -> Option<&OptionValue> {
        self.values.get(option_index)
    }
}

/// `name` as a profile name: control characters removed, trimmed, at most
/// [`PROFILE_NAME_MAX`] characters. May be empty.
#[must_use]
pub fn clean_profile_name(name: &str) -> String {
    let clean: String = name.chars().filter(|c| !c.is_control()).collect();
    let short: String = clean.trim().chars().take(PROFILE_NAME_MAX).collect();
    short.trim_end().to_owned()
}

/// What happened while loading, for the log.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LoadReport {
    /// There was no file: a first run, every mod at its default.
    pub file_missing: bool,
    /// The file existed but was not valid JSON (or not an object). Everything is at its default;
    /// the caller should keep a copy of the bad file before the next save overwrites it.
    pub corrupt: bool,
    /// The file held bytes that are not UTF-8: they were read as U+FFFD
    /// and everything else loaded (one odd byte used to make the whole file "corrupt" and every
    /// mod go back to its defaults). The next save writes the replacement characters, so the
    /// caller keeps a copy of the file first, as for [`LoadReport::corrupt`].
    pub reencoded: bool,
    /// One line per coerced, clamped or ignored value.
    pub warnings: Vec<String>,
    /// One line per mod entry saved by an older build and rewritten into today's meaning
    /// ([`MIGRATIONS`]). Not a problem, just worth a log line.
    pub migrated: Vec<String>,
}

/// Rewrites a mod's entry saved by an older build into today's meaning, before it is read:
/// `None` for a current entry, otherwise the rewritten entry and a line for
/// [`LoadReport::migrated`]. The file on disk is upgraded by the next save.
pub type Migration = fn(&Map<String, Value>) -> Option<(Map<String, Value>, String)>;

/// Every per-mod migration, by mod id.
pub const MIGRATIONS: &[(&str, Migration)] = &[
    // "No Dynamic FOV" became "Dynamic FOV": `enabled` flipped meaning, `keep_bow` became
    // `bow_zoom`.
    ("dynamic_fov", crate::mods::dynamic_fov::migrate_legacy),
];

/// Options that hold what a mod is doing rather than how it is set up,
/// as `(mod id, option id)`: saved with the settings so they survive a restart, but not settings.
/// A whole-setup profile neither saves nor loads them ([`ModSettings::setup_json`],
/// [`ModSettings::apply_setup`]), they never make a profile look changed
/// ([`ModSettings::setup_diff`]) and they never make a mod count as changed
/// ([`ModSettings::options_are_default_at`]). "Reset settings" still resets them.
///
/// Toggle Sprint's toggle was the one: a profile saved while sprinting turned the sprint toggle
/// on when it was loaded, and tapping Sprint afterwards made the loaded profile read as changed.
pub const RUNTIME_STATE: &[(&str, &str)] = &[
    (
        "toggle_sprint",
        crate::mods::toggle_sprint::SPRINT_TOGGLED_ID,
    ),
    // The replay browser's order and view: how the player left the
    // window, not how the mod is set up.
    ("replay", crate::mods::replay::ids::BROWSER_SORT),
    ("replay", crate::mods::replay::ids::BROWSER_VIEW),
];

/// Whether option `option_id` of `mod_id` is in [`RUNTIME_STATE`].
#[must_use]
pub fn is_runtime_state(mod_id: &str, option_id: &str) -> bool {
    RUNTIME_STATE
        .iter()
        .any(|(m, o)| *m == mod_id && *o == option_id)
}

#[derive(Debug, Clone)]
struct ModState {
    enabled: bool,
    values: Vec<OptionValue>,
    placements: Vec<HudPlacement>,
    profiles: Vec<ModProfile>,
    unknown_options: Map<String, Value>,
    unknown_hud: Map<String, Value>,
    /// Keys of the mod's entry other than `enabled`, `options`, `hud` and `profiles` (and a
    /// readout's `style`).
    unknown_entry: Map<String, Value>,
    // A readout's style options come from the shared HUD style (only ever
    // `true` for `hud_style::READOUTS`).
    hud_style: bool,
}

impl ModState {
    fn defaults(descriptor: &ModDescriptor) -> ModState {
        ModState {
            hud_style: hud_style::uses_hud_style_by_default(descriptor),
            enabled: descriptor.default_enabled,
            values: default_values(descriptor),
            placements: descriptor
                .hud_elements
                .iter()
                .map(|element| element.default_placement)
                .collect(),
            profiles: Vec::new(),
            unknown_options: Map::new(),
            unknown_hud: Map::new(),
            unknown_entry: Map::new(),
        }
    }
}

fn default_values(descriptor: &ModDescriptor) -> Vec<OptionValue> {
    descriptor
        .options
        .iter()
        .map(|option| option.default.clone())
        .collect()
}

/// Every mod's settings. One instance lives on the client (`Client::mods.settings`).
#[derive(Debug, Clone)]
pub struct ModSettings {
    catalog: &'static [&'static ModDescriptor],
    mods: Vec<ModState>,
    generation: u64,
    unknown_mods: Map<String, Value>,
    /// The mod menu's open/close key (LWJGL code, never 0).
    menu_key: i32,
    /// Keys of the `"menu"` object other than `open_key`.
    unknown_menu: Map<String, Value>,
    /// Root keys other than `format`, `menu`, `mods` and `pinned`.
    unknown_root: Map<String, Value>,
    /// Mods pinned to the menu's My setup, by id, in the order they were pinned. Ids this build
    /// does not know are kept and written back.
    pinned: Vec<String>,
    /// The shared HUD style: one value per `hud_style::STYLE_IDS`.
    hud_style: [OptionValue; 5],
    /// Keys of the root `"hud_style"` object this build does not know.
    unknown_hud_style: Map<String, Value>,
}

impl Default for ModSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl ModSettings {
    /// Every mod in [`catalog::all`] at its defaults.
    #[must_use]
    pub fn new() -> ModSettings {
        ModSettings::with_catalog(catalog::all())
    }

    /// Defaults for an explicit catalog (tests use small fake catalogs).
    #[must_use]
    pub fn with_catalog(catalog: &'static [&'static ModDescriptor]) -> ModSettings {
        ModSettings {
            catalog,
            mods: catalog.iter().map(|d| ModState::defaults(d)).collect(),
            generation: 1,
            unknown_mods: Map::new(),
            menu_key: DEFAULT_MENU_KEY,
            unknown_menu: Map::new(),
            unknown_root: Map::new(),
            pinned: Vec::new(),
            hud_style: hud_style::defaults(),
            unknown_hud_style: Map::new(),
        }
    }

    /// The catalog this store was built over, in menu order.
    #[must_use]
    pub fn catalog(&self) -> &'static [&'static ModDescriptor] {
        self.catalog
    }

    /// A counter that moves on every change (and on load). Starts at 1, so a snapshot that
    /// starts at 0 always refreshes on first use. Comparing it is the whole per-frame cost of a
    /// mod that is off.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn bump(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    /// Position of a mod in [`ModSettings::catalog`].
    #[must_use]
    pub fn index_of(&self, mod_id: &str) -> Option<usize> {
        self.catalog.iter().position(|d| d.id == mod_id)
    }

    /// A mod's descriptor.
    #[must_use]
    pub fn descriptor(&self, mod_id: &str) -> Option<&'static ModDescriptor> {
        self.catalog.iter().copied().find(|d| d.id == mod_id)
    }

    // ---- enabled flags -------------------------------------------------------------------

    /// Whether the mod is on. An unknown id is off.
    #[must_use]
    pub fn enabled(&self, mod_id: &str) -> bool {
        self.index_of(mod_id)
            .is_some_and(|index| self.mods[index].enabled)
    }

    /// [`ModSettings::enabled`] by catalog index.
    #[must_use]
    pub fn enabled_at(&self, index: usize) -> bool {
        self.mods.get(index).is_some_and(|state| state.enabled)
    }

    /// Turn a mod on or off. Returns whether anything changed.
    pub fn set_enabled(&mut self, mod_id: &str, on: bool) -> bool {
        match self.index_of(mod_id) {
            Some(index) => self.set_enabled_at(index, on),
            None => false,
        }
    }

    /// [`ModSettings::set_enabled`] by catalog index.
    pub fn set_enabled_at(&mut self, index: usize, on: bool) -> bool {
        match self.mods.get_mut(index) {
            Some(state) if state.enabled != on => {
                state.enabled = on;
                self.bump();
                true
            }
            _ => false,
        }
    }

    /// Flip a mod. Returns its new state.
    pub fn toggle(&mut self, mod_id: &str) -> bool {
        let on = !self.enabled(mod_id);
        self.set_enabled(mod_id, on);
        self.enabled(mod_id)
    }

    // ---- option values -------------------------------------------------------------------

    fn locate(&self, mod_id: &str, option_id: &str) -> Option<(usize, usize)> {
        let mod_index = self.index_of(mod_id)?;
        let option_index = self.catalog[mod_index].option_index(option_id)?;
        Some((mod_index, option_index))
    }

    /// The stored value of an option, or `None` for an unknown mod or option. Never asserts, so
    /// generic code (like a shared text-style reader) can probe for optional options.
    #[must_use]
    pub fn value(&self, mod_id: &str, option_id: &str) -> Option<&OptionValue> {
        let (m, o) = self.locate(mod_id, option_id)?;
        self.value_at(m, o)
    }

    /// [`ModSettings::value`] by catalog index and option index.
    ///
    /// A readout that uses the shared HUD style reads its style options
    /// from it ([`crate::hud_style`]): this is the value that is drawn.
    #[must_use]
    pub fn value_at(&self, mod_index: usize, option_index: usize) -> Option<&OptionValue> {
        let state = self.mods.get(mod_index)?;
        if state.hud_style {
            if let Some(slot) = self.shared_style_slot(mod_index, option_index) {
                return Some(&self.hud_style[slot]);
            }
        }
        state.values.get(option_index)
    }

    /// The place in the shared HUD style of option `o` of mod `m`, if it is a style option.
    fn shared_style_slot(&self, m: usize, o: usize) -> Option<usize> {
        hud_style::slot(self.catalog.get(m)?.options.get(o)?.id)
    }

    fn typed(&self, mod_id: &str, option_id: &str) -> Option<&OptionValue> {
        let value = self.value(mod_id, option_id);
        debug_assert!(
            value.is_some(),
            "mc-mods: no option {mod_id}/{option_id} in the catalog"
        );
        value
    }

    /// A toggle's value. `false` for an unknown or non-toggle option (debug builds assert the
    /// option exists, so a typo fails a test instead of silently reading the default).
    #[must_use]
    pub fn bool(&self, mod_id: &str, option_id: &str) -> bool {
        matches!(self.typed(mod_id, option_id), Some(OptionValue::Bool(true)))
    }

    /// A slider's value. `0.0` for an unknown or non-slider option.
    #[must_use]
    pub fn f32(&self, mod_id: &str, option_id: &str) -> f32 {
        match self.typed(mod_id, option_id) {
            Some(OptionValue::Float(v)) => *v,
            _ => 0.0,
        }
    }

    /// A choice's index. `0` for an unknown or non-choice option.
    #[must_use]
    pub fn choice(&self, mod_id: &str, option_id: &str) -> usize {
        match self.typed(mod_id, option_id) {
            Some(OptionValue::Choice(index)) => *index,
            _ => 0,
        }
    }

    /// A choice's label, `""` for an unknown or non-choice option.
    #[must_use]
    pub fn choice_label(&self, mod_id: &str, option_id: &str) -> &'static str {
        let index = self.choice(mod_id, option_id);
        match self.descriptor(mod_id).and_then(|d| d.option(option_id)) {
            Some(option) => match option.kind {
                OptionKind::Choice { choices } => choices.get(index).copied().unwrap_or(""),
                _ => "",
            },
            None => "",
        }
    }

    /// A colour as `0xAARRGGBB`. Opaque white for an unknown or non-colour option.
    #[must_use]
    pub fn color(&self, mod_id: &str, option_id: &str) -> u32 {
        match self.typed(mod_id, option_id) {
            Some(OptionValue::Color(argb)) => *argb,
            _ => 0xFFFF_FFFF,
        }
    }

    /// A key option in vanilla's LWJGL encoding (`> 0` key, `< 0` mouse button - 100, `0`
    /// unbound). Unbound for an unknown or non-key option.
    #[must_use]
    pub fn key(&self, mod_id: &str, option_id: &str) -> i32 {
        match self.typed(mod_id, option_id) {
            Some(OptionValue::Key(code)) => *code,
            _ => 0,
        }
    }

    /// A text option. `""` for an unknown or non-text option.
    #[must_use]
    pub fn text(&self, mod_id: &str, option_id: &str) -> &str {
        match self.typed(mod_id, option_id) {
            Some(OptionValue::Text(text)) => text,
            _ => "",
        }
    }

    /// Store a value after [`OptionKind::sanitize`]. A wrong-typed value is replaced by the
    /// option's default. Returns whether the stored value changed.
    pub fn set_value(&mut self, mod_id: &str, option_id: &str, value: OptionValue) -> bool {
        match self.locate(mod_id, option_id) {
            Some((m, o)) => self.set_value_at(m, o, value),
            None => false,
        }
    }

    /// [`ModSettings::set_value`] by catalog index and option index.
    pub fn set_value_at(
        &mut self,
        mod_index: usize,
        option_index: usize,
        value: OptionValue,
    ) -> bool {
        let Some(descriptor) = self.catalog.get(mod_index) else {
            return false;
        };
        let Some(option) = descriptor.options.get(option_index) else {
            return false;
        };
        // A readout that uses the HUD style edits the shared value.
        if self.mods[mod_index].hud_style {
            if let Some(slot) = hud_style::slot(option.id) {
                let default = &hud_style::STYLE_OPTIONS[slot].default;
                let value = option.kind.sanitize(value, default);
                if self.hud_style[slot] == value {
                    return false;
                }
                self.hud_style[slot] = value;
                self.bump();
                return true;
            }
        }
        let value = option.kind.sanitize(value, &option.default);
        let slot = &mut self.mods[mod_index].values[option_index];
        if *slot == value {
            return false;
        }
        *slot = value;
        self.bump();
        true
    }

    /// Set a toggle.
    pub fn set_bool(&mut self, mod_id: &str, option_id: &str, on: bool) -> bool {
        self.set_value(mod_id, option_id, OptionValue::Bool(on))
    }

    /// Set a slider (clamped and snapped).
    pub fn set_f32(&mut self, mod_id: &str, option_id: &str, value: f32) -> bool {
        self.set_value(mod_id, option_id, OptionValue::Float(value))
    }

    /// Set a choice by index.
    pub fn set_choice(&mut self, mod_id: &str, option_id: &str, index: usize) -> bool {
        self.set_value(mod_id, option_id, OptionValue::Choice(index))
    }

    /// Set a colour (`0xAARRGGBB`).
    pub fn set_color(&mut self, mod_id: &str, option_id: &str, argb: u32) -> bool {
        self.set_value(mod_id, option_id, OptionValue::Color(argb))
    }

    /// Set a key (LWJGL encoding).
    pub fn set_key(&mut self, mod_id: &str, option_id: &str, code: i32) -> bool {
        self.set_value(mod_id, option_id, OptionValue::Key(code))
    }

    /// Set a text option (truncated to its `max_len`).
    pub fn set_text(&mut self, mod_id: &str, option_id: &str, text: &str) -> bool {
        self.set_value(
            mod_id,
            option_id,
            OptionValue::Text(Cow::Owned(text.to_owned())),
        )
    }

    /// Put every option of one mod back to its default (enabled flag and HUD placements are
    /// left alone).
    pub fn reset_options(&mut self, mod_id: &str) {
        if let Some(index) = self.index_of(mod_id) {
            self.reset_options_at(index);
        }
    }

    /// [`ModSettings::reset_options`] by catalog index: the menu's "Reset to defaults". Hidden
    /// options (state a mod keeps for its dynamic rows, e.g. the particle table) reset too.
    /// Returns whether anything changed.
    pub fn reset_options_at(&mut self, index: usize) -> bool {
        let Some(descriptor) = self.catalog.get(index) else {
            return false;
        };
        let defaults = default_values(descriptor);
        // A readout's choice of style resets too.
        let style = hud_style::uses_hud_style_by_default(descriptor);
        let state = &mut self.mods[index];
        if state.values == defaults && state.hud_style == style {
            return false;
        }
        state.values = defaults;
        state.hud_style = style;
        self.bump();
        true
    }

    /// Whether every option of the mod at `index` holds its default (the menu greys "Reset to
    /// defaults" out then). Buttons hold nothing and always count as default.
    #[must_use]
    pub fn options_are_default_at(&self, index: usize) -> bool {
        match (self.catalog.get(index), self.mods.get(index)) {
            (Some(descriptor), Some(state)) => {
                // A readout's choice of style counts as a setting.
                // What the mod is doing (`RUNTIME_STATE`) does not.
                state.hud_style == hud_style::uses_hud_style_by_default(descriptor)
                    && descriptor
                        .options
                        .iter()
                        .zip(&state.values)
                        .all(|(option, value)| {
                            *value == option.default || is_runtime_state(descriptor.id, option.id)
                        })
            }
            _ => true,
        }
    }

    // ---- the menu key --------------------------------------------------------------------

    /// The key that opens and closes the mod menu, as an LWJGL key code (never 0: see
    /// [`ModSettings::set_menu_key`]).
    #[must_use]
    pub fn menu_key(&self) -> i32 {
        self.menu_key
    }

    /// Set the menu key. It is keyboard-only and never unbound, or nothing could open the menu
    /// again: `0`, a mouse code (negative), Escape (`1`, the pause menu's) or a code above 255
    /// sets [`DEFAULT_MENU_KEY`]. Returns whether anything changed.
    pub fn set_menu_key(&mut self, code: i32) -> bool {
        let code = sanitize_menu_key(code);
        if self.menu_key == code {
            return false;
        }
        self.menu_key = code;
        self.bump();
        true
    }

    // ---- pins ----------------------------------------------

    /// Whether the mod at catalog `index` is pinned to the menu's My setup (it then shows there
    /// even with nothing changed).
    #[must_use]
    pub fn pinned_at(&self, index: usize) -> bool {
        self.catalog
            .get(index)
            .is_some_and(|descriptor| self.pinned.iter().any(|id| id == descriptor.id))
    }

    /// Pin or unpin the mod at catalog `index`; a new pin goes last. Returns whether anything
    /// changed.
    pub fn set_pinned_at(&mut self, index: usize, on: bool) -> bool {
        let Some(descriptor) = self.catalog.get(index) else {
            return false;
        };
        match (self.pinned.iter().position(|id| id == descriptor.id), on) {
            (None, true) => self.pinned.push(descriptor.id.to_owned()),
            (Some(at), false) => {
                self.pinned.remove(at);
            }
            _ => return false,
        }
        self.bump();
        true
    }

    /// The pinned mod ids in the order they were pinned, ids this build does not know included
    /// (whole-setup profiles save and restore these).
    #[must_use]
    pub fn pinned_ids(&self) -> &[String] {
        &self.pinned
    }

    /// Replace the pinned list (loading a whole-setup profile). Empty, over-long and repeated
    /// ids are dropped, as on load. Returns whether anything changed.
    pub fn set_pinned_ids<I, S>(&mut self, ids: I) -> bool
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut clean: Vec<String> = Vec::new();
        for id in ids {
            if let Some(id) = clean_pin_id(id.as_ref()) {
                if !clean.contains(&id) {
                    clean.push(id);
                }
            }
        }
        if clean == self.pinned {
            return false;
        }
        self.pinned = clean;
        self.bump();
        true
    }

    // ---- the shared HUD style (see `hud_style`) -------------

    /// Whether the mod at catalog `index` is a text readout that can use the shared HUD style
    /// ([`hud_style::READOUTS`]).
    #[must_use]
    pub fn hud_style_capable_at(&self, index: usize) -> bool {
        self.catalog
            .get(index)
            .is_some_and(|descriptor| hud_style::is_readout(descriptor.id))
    }

    /// Whether the readout at catalog `index` uses the shared HUD style ("HUD style") rather
    /// than its own style settings ("Custom"). Always `false` for a mod that is not a readout.
    #[must_use]
    pub fn uses_hud_style_at(&self, index: usize) -> bool {
        self.mods.get(index).is_some_and(|state| state.hud_style)
    }

    /// Switch the readout at catalog `index` to the shared HUD style (`true`) or to its own
    /// style settings. Its own values are kept while it uses the HUD style. A readout that
    /// never had its own look (its own defaults are the HUD style's, and nothing was changed)
    /// starts Custom from the HUD style's current look, so switching never makes it jump. Does
    /// nothing for a mod that is not a readout. Returns whether anything changed.
    pub fn set_uses_hud_style_at(&mut self, index: usize, on: bool) -> bool {
        let Some(descriptor) = self.catalog.get(index).copied() else {
            return false;
        };
        if !hud_style::is_readout(descriptor.id) || self.mods[index].hud_style == on {
            return false;
        }
        let state = &mut self.mods[index];
        if !on && hud_style::derived_uses_hud_style(descriptor, &state.values) {
            for (o, option) in descriptor.options.iter().enumerate() {
                if let Some(slot) = hud_style::slot(option.id) {
                    let shared = self.hud_style[slot].clone();
                    state.values[o] = option.kind.sanitize(shared, &option.default);
                }
            }
        }
        state.hud_style = on;
        self.bump();
        true
    }

    /// The shared HUD style's value of style option `option_id` (`None` for an id that is not
    /// one of [`hud_style::STYLE_IDS`]).
    #[must_use]
    pub fn hud_style_value(&self, option_id: &str) -> Option<&OptionValue> {
        hud_style::slot(option_id).map(|slot| &self.hud_style[slot])
    }

    /// Set one of the shared HUD style's values (sanitized like the option). Returns whether it
    /// changed.
    pub fn set_hud_style_value(&mut self, option_id: &str, value: OptionValue) -> bool {
        let Some(slot) = hud_style::slot(option_id) else {
            return false;
        };
        let option = &hud_style::STYLE_OPTIONS[slot];
        let value = option.kind.sanitize(value, &option.default);
        if self.hud_style[slot] == value {
            return false;
        }
        self.hud_style[slot] = value;
        self.bump();
        true
    }

    /// Whether every value of the shared HUD style is at its default.
    #[must_use]
    pub fn hud_style_is_default(&self) -> bool {
        self.hud_style
            .iter()
            .zip(&hud_style::STYLE_OPTIONS)
            .all(|(value, option)| *value == option.default)
    }

    /// How many readouts use the shared HUD style.
    #[must_use]
    pub fn hud_style_users(&self) -> usize {
        self.mods.iter().filter(|state| state.hud_style).count()
    }

    // ---- whole-setup profiles (see `crate::profiles`) -----

    /// The whole mod setup as JSON, the form a setup profile saves: [`ModSettings::to_json`]
    /// without the menu key (`"menu"`), each mod's own saved profiles (Color Grading's looks),
    /// [`RUNTIME_STATE`] and root keys this build does not know. Every mod is written, defaults
    /// included, so loading it later reproduces this setup whatever the store holds then.
    #[must_use]
    pub fn setup_json(&self) -> Value {
        let mut root = Map::new();
        if let Value::Object(full) = self.to_json() {
            for (key, value) in full {
                match (key.as_str(), value) {
                    ("mods", Value::Object(mut mods)) => {
                        for (mod_id, entry) in &mut mods {
                            if let Value::Object(entry) = entry {
                                entry.remove("profiles");
                                // Not a setting: a profile never
                                // carries it.
                                if let Some(Value::Object(options)) = entry.get_mut("options") {
                                    for (_, option_id) in
                                        RUNTIME_STATE.iter().filter(|(m, _)| m == mod_id)
                                    {
                                        options.remove(*option_id);
                                    }
                                }
                            }
                        }
                        root.insert(key, Value::Object(mods));
                    }
                    ("format" | "pinned" | hud_style::ROOT_KEY, value) => {
                        root.insert(key, value);
                    }
                    _ => {}
                }
            }
        }
        Value::Object(root)
    }

    /// Load a whole setup (a profile parsed with [`ModSettings::from_json_str`], or a fresh
    /// store): every mod's switch, settings, HUD positions and choice of HUD style, the pins and
    /// the shared HUD style, all in one step (one generation, so the client saves it once). The
    /// menu key, each mod's own saved profiles, [`RUNTIME_STATE`] and root keys this build does
    /// not know stay as they are. Returns whether anything changed; does nothing for a store over
    /// another catalog.
    pub fn apply_setup(&mut self, setup: &ModSettings) -> bool {
        if !self.same_catalog(setup) || self.setup_eq(setup) {
            return false;
        }
        for (descriptor, (state, from)) in self
            .catalog
            .iter()
            .zip(self.mods.iter_mut().zip(&setup.mods))
        {
            state.enabled = from.enabled;
            // What the mod is doing stays as it is (an older profile
            // file may still hold it).
            let mut values = from.values.clone();
            for (o, option) in descriptor.options.iter().enumerate() {
                if is_runtime_state(descriptor.id, option.id) {
                    if let (Some(slot), Some(now)) = (values.get_mut(o), state.values.get(o)) {
                        slot.clone_from(now);
                    }
                }
            }
            state.values = values;
            state.placements.clone_from(&from.placements);
            state.hud_style = from.hud_style;
            state.unknown_options.clone_from(&from.unknown_options);
            state.unknown_hud.clone_from(&from.unknown_hud);
            state.unknown_entry.clone_from(&from.unknown_entry);
        }
        self.pinned.clone_from(&setup.pinned);
        self.hud_style.clone_from(&setup.hud_style);
        self.unknown_hud_style.clone_from(&setup.unknown_hud_style);
        self.bump();
        true
    }

    /// Whether `setup` holds the same whole setup as this store (everything
    /// [`ModSettings::apply_setup`] copies).
    #[must_use]
    pub fn setup_eq(&self, setup: &ModSettings) -> bool {
        self.same_catalog(setup) && self.setup_diff(setup).is_empty()
    }

    /// What loading `setup` would change, mod by mod (the menu's profile preview). Empty for a
    /// store over another catalog.
    #[must_use]
    pub fn setup_diff(&self, setup: &ModSettings) -> SetupDiff {
        let mut diff = SetupDiff::default();
        if !self.same_catalog(setup) {
            return diff;
        }
        for (m, (now, then)) in self.mods.iter().zip(&setup.mods).enumerate() {
            match (now.enabled, then.enabled) {
                (false, true) => diff.turns_on.push(m),
                (true, false) => diff.turns_off.push(m),
                _ => {}
            }
            // What the mod is doing is not a setting.
            let descriptor = self.catalog[m];
            let values_differ = descriptor
                .options
                .iter()
                .zip(now.values.iter().zip(&then.values))
                .any(|(option, (a, b))| a != b && !is_runtime_state(descriptor.id, option.id));
            if values_differ
                || now.hud_style != then.hud_style
                || now.unknown_options != then.unknown_options
                || now.unknown_entry != then.unknown_entry
            {
                diff.settings.push(m);
            }
            if now.placements != then.placements || now.unknown_hud != then.unknown_hud {
                diff.moved.push(m);
            }
        }
        diff.pins = self.pinned != setup.pinned;
        diff.hud_style =
            self.hud_style != setup.hud_style || self.unknown_hud_style != setup.unknown_hud_style;
        diff
    }

    /// Whether `other` was built over the same catalog (the same mods in the same order).
    fn same_catalog(&self, other: &ModSettings) -> bool {
        self.catalog.len() == other.catalog.len()
            && self
                .catalog
                .iter()
                .zip(other.catalog)
                .all(|(a, b)| a.id == b.id)
    }

    // ---- profiles ------------------------------------------------------------------------

    /// The profiles saved for the mod at catalog `index`, in the order they were made.
    #[must_use]
    pub fn profiles_at(&self, index: usize) -> &[ModProfile] {
        self.mods.get(index).map_or(&[], |state| &state.profiles)
    }

    /// The profile of mod `index` with this name (compared trimmed and case-insensitively).
    #[must_use]
    pub fn profile_named_at(&self, index: usize, name: &str) -> Option<usize> {
        let wanted = clean_profile_name(name);
        self.profiles_at(index)
            .iter()
            .position(|profile| profile.name.eq_ignore_ascii_case(&wanted))
    }

    /// Save the current option values of mod `index` as a profile called `name` (cleaned with
    /// [`clean_profile_name`]; an empty name becomes "Profile N"). A profile with the same name
    /// is updated in place; otherwise a new one is added, unless the mod already has
    /// [`MAX_PROFILES`]. Returns the profile's index, or `None` when nothing was saved.
    pub fn save_profile_at(&mut self, index: usize, name: &str) -> Option<usize> {
        let state = self.mods.get(index)?;
        let mut name = clean_profile_name(name);
        if name.is_empty() {
            let mut n = state.profiles.len() + 1;
            loop {
                let candidate = format!("Profile {n}");
                if !state
                    .profiles
                    .iter()
                    .any(|profile| profile.name.eq_ignore_ascii_case(&candidate))
                {
                    name = candidate;
                    break;
                }
                n += 1;
            }
        }
        let values = state.values.clone();
        let existing = self.profile_named_at(index, &name);
        let state = &mut self.mods[index];
        let slot = match existing {
            Some(slot) => {
                let profile = &mut state.profiles[slot];
                if profile.values == values && profile.name == name {
                    return Some(slot);
                }
                profile.values = values;
                profile.name = name;
                slot
            }
            None if state.profiles.len() < MAX_PROFILES => {
                state.profiles.push(ModProfile {
                    name,
                    values,
                    unknown: Map::new(),
                });
                state.profiles.len() - 1
            }
            None => return None,
        };
        self.bump();
        Some(slot)
    }

    /// Apply profile `profile` of mod `index` to the mod's options. Returns whether any value
    /// changed.
    pub fn load_profile_at(&mut self, index: usize, profile: usize) -> bool {
        let Some(values) = self
            .mods
            .get(index)
            .and_then(|state| state.profiles.get(profile))
            .map(|profile| profile.values.clone())
        else {
            return false;
        };
        let state = &mut self.mods[index];
        if state.values == values {
            return false;
        }
        state.values = values;
        self.bump();
        true
    }

    /// Delete profile `profile` of mod `index`. Returns whether it existed.
    pub fn delete_profile_at(&mut self, index: usize, profile: usize) -> bool {
        let Some(state) = self.mods.get_mut(index) else {
            return false;
        };
        if profile >= state.profiles.len() {
            return false;
        }
        state.profiles.remove(profile);
        self.bump();
        true
    }

    /// Whether mod `index`'s options are exactly profile `profile`'s values right now (the menu
    /// marks that profile).
    #[must_use]
    pub fn profile_is_current_at(&self, index: usize, profile: usize) -> bool {
        self.mods.get(index).is_some_and(|state| {
            state
                .profiles
                .get(profile)
                .is_some_and(|profile| profile.values == state.values)
        })
    }

    // ---- HUD placements ------------------------------------------------------------------

    /// Where a HUD element sits. The element's default for an unknown element, or top-left 0,0
    /// for an unknown mod.
    #[must_use]
    pub fn placement(&self, mod_id: &str, element_id: &str) -> HudPlacement {
        self.index_of(mod_id)
            .and_then(|m| {
                let e = self.catalog[m].hud_element_index(element_id)?;
                Some(self.mods[m].placements[e])
            })
            .unwrap_or(HudPlacement::new(HudAnchor::TopLeft, 0.0, 0.0))
    }

    /// [`ModSettings::placement`] by catalog index and element index.
    #[must_use]
    pub fn placement_at(&self, mod_index: usize, element_index: usize) -> Option<HudPlacement> {
        self.mods
            .get(mod_index)?
            .placements
            .get(element_index)
            .copied()
    }

    /// Move a HUD element. Returns whether anything changed.
    pub fn set_placement(
        &mut self,
        mod_id: &str,
        element_id: &str,
        placement: HudPlacement,
    ) -> bool {
        let Some(m) = self.index_of(mod_id) else {
            return false;
        };
        let Some(e) = self.catalog[m].hud_element_index(element_id) else {
            return false;
        };
        self.set_placement_at(m, e, placement)
    }

    /// [`ModSettings::set_placement`] by catalog index and element index.
    pub fn set_placement_at(
        &mut self,
        mod_index: usize,
        element_index: usize,
        placement: HudPlacement,
    ) -> bool {
        let placement = placement.sanitized();
        let Some(slot) = self
            .mods
            .get_mut(mod_index)
            .and_then(|state| state.placements.get_mut(element_index))
        else {
            return false;
        };
        if *slot == placement {
            return false;
        }
        *slot = placement;
        self.bump();
        true
    }

    /// Put a HUD element back where its descriptor says.
    pub fn reset_placement_at(&mut self, mod_index: usize, element_index: usize) -> bool {
        let Some(default) = self
            .catalog
            .get(mod_index)
            .and_then(|d| d.hud_elements.get(element_index))
            .map(|element| element.default_placement)
        else {
            return false;
        };
        self.set_placement_at(mod_index, element_index, default)
    }

    // ---- JSON ----------------------------------------------------------------------------

    /// The whole store as JSON (every mod, defaults included, so the file is self-documenting).
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut mods = self.unknown_mods.clone();
        for (descriptor, state) in self.catalog.iter().zip(&self.mods) {
            let mut entry = state.unknown_entry.clone();
            entry.insert("enabled".into(), Value::Bool(state.enabled));
            if !descriptor.options.is_empty() || !state.unknown_options.is_empty() {
                let options = options_to_json(descriptor, &state.values, &state.unknown_options);
                entry.insert("options".into(), Value::Object(options));
            }
            if !descriptor.hud_elements.is_empty() || !state.unknown_hud.is_empty() {
                let mut hud = state.unknown_hud.clone();
                for (element, placement) in descriptor.hud_elements.iter().zip(&state.placements) {
                    hud.insert(element.id.into(), placement_to_json(placement));
                }
                entry.insert("hud".into(), Value::Object(hud));
            }
            if !state.profiles.is_empty() {
                let profiles = state
                    .profiles
                    .iter()
                    .map(|profile| {
                        let mut map = Map::new();
                        map.insert("name".into(), Value::String(profile.name.clone()));
                        let options =
                            options_to_json(descriptor, &profile.values, &profile.unknown);
                        map.insert("options".into(), Value::Object(options));
                        Value::Object(map)
                    })
                    .collect();
                entry.insert("profiles".into(), Value::Array(profiles));
            }
            // Which style a readout uses, only when its saved values alone
            // would read otherwise (`hud_style::derived_uses_hud_style`).
            if hud_style::is_readout(descriptor.id)
                && state.hud_style != hud_style::derived_uses_hud_style(descriptor, &state.values)
            {
                let style = if state.hud_style {
                    hud_style::ENTRY_HUD
                } else {
                    hud_style::ENTRY_CUSTOM
                };
                entry.insert(hud_style::ENTRY_KEY.into(), Value::String(style.into()));
            }
            mods.insert(descriptor.id.into(), Value::Object(entry));
        }
        let mut root = self.unknown_root.clone();
        root.insert("format".into(), Value::Number(Number::from(FORMAT_VERSION)));
        let mut menu = self.unknown_menu.clone();
        menu.insert(
            "open_key".into(),
            Value::Number(Number::from(self.menu_key)),
        );
        root.insert("menu".into(), Value::Object(menu));
        // Written only when something is pinned, so a file without pins saves
        // exactly as before.
        if !self.pinned.is_empty() {
            let pinned = self.pinned.iter().cloned().map(Value::String).collect();
            root.insert("pinned".into(), Value::Array(pinned));
        }
        // The shared HUD style, only while a value differs from its default:
        // a file nobody styled saves exactly as before.
        if !self.hud_style_is_default() || !self.unknown_hud_style.is_empty() {
            let mut style = self.unknown_hud_style.clone();
            for (option, value) in hud_style::STYLE_OPTIONS.iter().zip(&self.hud_style) {
                style.insert(option.id.into(), value_to_json(&option.kind, value));
            }
            root.insert(hud_style::ROOT_KEY.into(), Value::Object(style));
        }
        root.insert("mods".into(), Value::Object(mods));
        Value::Object(root)
    }

    /// [`ModSettings::to_json`], pretty-printed.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.to_json()).unwrap_or_else(|_| "{}".into())
    }

    /// Parse a settings file over the default catalog. Never fails; see [`LoadReport`].
    #[must_use]
    pub fn from_json_str(text: &str) -> (ModSettings, LoadReport) {
        ModSettings::from_json_str_with_catalog(text, catalog::all())
    }

    /// [`ModSettings::from_json_str`] over an explicit catalog.
    #[must_use]
    pub fn from_json_str_with_catalog(
        text: &str,
        catalog: &'static [&'static ModDescriptor],
    ) -> (ModSettings, LoadReport) {
        let mut settings = ModSettings::with_catalog(catalog);
        let mut report = LoadReport::default();
        match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(root)) => settings.apply_json(&root, &mut report),
            Ok(_) => {
                report.corrupt = true;
                report
                    .warnings
                    .push("settings root is not a JSON object".into());
            }
            Err(error) => {
                report.corrupt = true;
                report
                    .warnings
                    .push(format!("settings are not valid JSON: {error}"));
            }
        }
        settings.bump();
        (settings, report)
    }

    fn apply_json(&mut self, root: &Map<String, Value>, report: &mut LoadReport) {
        for (key, value) in root {
            if !matches!(
                key.as_str(),
                "format" | "menu" | "mods" | "pinned" | hud_style::ROOT_KEY
            ) {
                self.unknown_root.insert(key.clone(), value.clone());
            }
        }
        // The shared HUD style: missing values keep their default.
        match root.get(hud_style::ROOT_KEY) {
            Some(Value::Object(style)) => {
                for (key, json) in style {
                    let Some(slot) = hud_style::slot(key) else {
                        self.unknown_hud_style.insert(key.clone(), json.clone());
                        continue;
                    };
                    let option = &hud_style::STYLE_OPTIONS[slot];
                    match value_from_json(&option.kind, json) {
                        Some(value) => {
                            let clean = option.kind.sanitize(value.clone(), &option.default);
                            if clean != value {
                                report
                                    .warnings
                                    .push(format!("hud_style.{key}: out of range, adjusted"));
                            }
                            self.hud_style[slot] = clean;
                        }
                        None => report.warnings.push(format!(
                            "hud_style.{key}: unreadable value {json}, default kept"
                        )),
                    }
                }
            }
            Some(other) => report.warnings.push(format!(
                "\"hud_style\" is not an object ({other}), defaults kept"
            )),
            None => {}
        }
        match root.get("menu") {
            Some(Value::Object(menu)) => {
                for (key, value) in menu {
                    if key != "open_key" {
                        self.unknown_menu.insert(key.clone(), value.clone());
                    }
                }
                if let Some(json) = menu.get("open_key") {
                    let code = json_f64(json)
                        .filter(|v| v.fract() == 0.0 && v.abs() < 1.0e6)
                        .map(|v| v as i32);
                    let clean = code.map(sanitize_menu_key);
                    if clean.is_none() || clean != code {
                        report.warnings.push(format!(
                            "menu.open_key: unusable key {json}, Right Shift kept"
                        ));
                    }
                    self.menu_key = clean.unwrap_or(DEFAULT_MENU_KEY);
                }
            }
            Some(other) => report.warnings.push(format!(
                "\"menu\" is not an object ({other}), defaults kept"
            )),
            None => {}
        }
        // Mods pinned to My setup: a top-level list that older files lack.
        match root.get("pinned") {
            Some(Value::Array(list)) => {
                for (i, json) in list.iter().enumerate() {
                    match json.as_str().and_then(clean_pin_id) {
                        Some(id) => {
                            if !self.pinned.contains(&id) {
                                self.pinned.push(id);
                            }
                        }
                        None => report
                            .warnings
                            .push(format!("pinned[{i}]: not a mod id ({json}), skipped")),
                    }
                }
            }
            Some(other) => report
                .warnings
                .push(format!("\"pinned\" is not a list ({other}), ignored")),
            None => {}
        }
        let Some(mods) = root.get("mods") else {
            return;
        };
        let Some(mods) = mods.as_object() else {
            report.warnings.push("\"mods\" is not an object".into());
            return;
        };
        // 256-character chat is always on, so neither the old
        // standalone `long_chat` mod nor Chat's `long_chat` option exists any more. Both are
        // ordinary unknown keys now: kept for the round trip, never a warning.
        for (mod_id, entry) in mods {
            let Some(index) = self.index_of(mod_id) else {
                self.unknown_mods.insert(mod_id.clone(), entry.clone());
                continue;
            };
            let Some(entry) = entry.as_object() else {
                report
                    .warnings
                    .push(format!("{mod_id}: entry is not an object"));
                continue;
            };
            // An entry saved by an older build is read in today's meaning (`MIGRATIONS`).
            let migrated = MIGRATIONS
                .iter()
                .find(|(id, _)| id == mod_id)
                .and_then(|(_, migrate)| migrate(entry));
            let entry = match &migrated {
                Some((rewritten, note)) => {
                    report.migrated.push(note.clone());
                    rewritten
                }
                None => entry,
            };
            let descriptor = self.catalog[index];
            let state = &mut self.mods[index];
            let readout = hud_style::is_readout(descriptor.id);
            for (key, value) in entry {
                if !matches!(key.as_str(), "enabled" | "options" | "hud" | "profiles")
                    && !(readout && key == hud_style::ENTRY_KEY)
                {
                    state.unknown_entry.insert(key.clone(), value.clone());
                }
            }
            if let Some(profiles) = entry.get("profiles") {
                match profiles.as_array() {
                    Some(list) => {
                        for (i, json) in list.iter().enumerate() {
                            match profile_from_json(descriptor, json) {
                                Some(profile) => state.profiles.push(profile),
                                None => report.warnings.push(format!(
                                    "{mod_id}.profiles[{i}]: not a named profile, skipped"
                                )),
                            }
                        }
                    }
                    None => report
                        .warnings
                        .push(format!("{mod_id}.profiles: not a list, ignored")),
                }
            }
            if let Some(enabled) = entry.get("enabled") {
                match json_bool(enabled) {
                    Some(on) => state.enabled = on,
                    // The default stays (on for Fog, Sky and the
                    // other default-on mods), so the warning no longer says "left off".
                    None => report
                        .warnings
                        .push(format!("{mod_id}.enabled: not a boolean, default kept")),
                }
            }
            if let Some(options) = entry.get("options").and_then(Value::as_object) {
                for (option_id, json) in options {
                    let Some(o) = descriptor.option_index(option_id) else {
                        state
                            .unknown_options
                            .insert(option_id.clone(), json.clone());
                        continue;
                    };
                    let option = &descriptor.options[o];
                    if !option.kind.is_persisted() {
                        continue;
                    }
                    match value_from_json(&option.kind, json) {
                        Some(value) => {
                            let clean = option.kind.sanitize(value.clone(), &option.default);
                            if clean != value {
                                report
                                    .warnings
                                    .push(format!("{mod_id}.{option_id}: out of range, adjusted"));
                            }
                            state.values[o] = clean;
                        }
                        None => report.warnings.push(format!(
                            "{mod_id}.{option_id}: unreadable value {json}, default kept"
                        )),
                    }
                }
            }
            if let Some(hud) = entry.get("hud").and_then(Value::as_object) {
                for (element_id, json) in hud {
                    let Some(e) = descriptor.hud_element_index(element_id) else {
                        state.unknown_hud.insert(element_id.clone(), json.clone());
                        continue;
                    };
                    let default = descriptor.hud_elements[e].default_placement;
                    match placement_from_json(json, default) {
                        Some(placement) => state.placements[e] = placement,
                        None => report.warnings.push(format!(
                            "{mod_id}.hud.{element_id}: unreadable placement, default kept"
                        )),
                    }
                }
            }
            // An `options` or `hud` that is not an object (a hand edit)
            // loads as the defaults: say so in the log instead of dropping it without a word.
            for key in ["options", "hud"] {
                if entry.get(key).is_some_and(|json| !json.is_object()) {
                    report
                        .warnings
                        .push(format!("{mod_id}.{key}: not an object, defaults kept"));
                }
            }
            // Which style a readout uses: its entry says, or else its saved
            // values decide (a readout someone styled loads as Custom, so it keeps its look).
            if readout {
                let derived = hud_style::derived_uses_hud_style(descriptor, &state.values);
                state.hud_style = match entry.get(hud_style::ENTRY_KEY) {
                    None => derived,
                    Some(Value::String(s)) if s == hud_style::ENTRY_HUD => true,
                    Some(Value::String(s)) if s == hud_style::ENTRY_CUSTOM => false,
                    Some(other) => {
                        report.warnings.push(format!(
                            "{mod_id}.style: not \"hud\" or \"custom\" ({other}), read from its \
                             settings"
                        ));
                        derived
                    }
                };
            }
        }
        // Modern Keybinds' retired
        // "Eating stops mining" became 1.7 Animations' "Using items stops mining". A saved "off"
        // carries over while the new option has never been saved
        // (`modern_keybinds::eat_stops_mining_was_off`); the old key stays an unknown option.
        if crate::mods::modern_keybinds::eat_stops_mining_was_off(mods) {
            if let Some(index) = self.index_of("old_animations") {
                if let Some(o) = self.catalog[index].option_index("use_stops_mining") {
                    self.mods[index].values[o] = OptionValue::Bool(false);
                    report.migrated.push(
                        "modern_keybinds.eat_stops_mining: off, so old_animations.\
                         use_stops_mining loads off"
                            .into(),
                    );
                }
            }
        }
    }

    /// Load `path`. A missing file is a first run; an unreadable or invalid one loads defaults.
    #[must_use]
    pub fn load(path: &Path) -> (ModSettings, LoadReport) {
        match std::fs::read(path) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => ModSettings::from_json_str(&text),
                // One odd byte costs its own characters, not the file.
                Err(error) => {
                    let text = String::from_utf8_lossy(error.as_bytes()).into_owned();
                    let (settings, mut report) = ModSettings::from_json_str(&text);
                    report.reencoded = true;
                    report.warnings.push(format!(
                        "{} holds bytes that are not UTF-8; they were read as U+FFFD",
                        path.display()
                    ));
                    (settings, report)
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let report = LoadReport {
                    file_missing: true,
                    ..LoadReport::default()
                };
                (ModSettings::new(), report)
            }
            Err(error) => {
                let report = LoadReport {
                    corrupt: true,
                    warnings: vec![format!("could not read {}: {error}", path.display())],
                    ..LoadReport::default()
                };
                (ModSettings::new(), report)
            }
        }
    }

    /// Write the store to `path` through a temporary file, flushed, and a rename, so neither a
    /// crash nor a power cut mid-write leaves a truncated settings file
    /// ([`mc_core::durable::write_atomic`]).
    ///
    /// # Errors
    ///
    /// The directory is not writable.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut text = self.to_json_string();
        text.push('\n');
        mc_core::durable::write_atomic(path, text.as_bytes())
    }
}

// ---- value <-> JSON ------------------------------------------------------------------------

/// The longest pinned mod id the file keeps (catalog ids are far shorter).
const PIN_ID_MAX: usize = 64;

/// A pinned mod id as stored: trimmed, not empty, at most [`PIN_ID_MAX`] bytes, no control
/// characters. Ids the catalog does not know are fine (a newer build's mods).
fn clean_pin_id(id: &str) -> Option<String> {
    let id = id.trim();
    (!id.is_empty() && id.len() <= PIN_ID_MAX && !id.chars().any(char::is_control))
        .then(|| id.to_owned())
}

/// The menu key as stored: a keyboard LWJGL code `2..=255`, anything else Right Shift.
/// `1` is Escape: the menu's capture never takes it (Escape cancels), but
/// a hand-edited file could, and the menu key is swallowed in game before anything else sees it,
/// so Escape would open this menu instead of the pause menu, which then had no key at all.
fn sanitize_menu_key(code: i32) -> i32 {
    if (2..=255).contains(&code) {
        code
    } else {
        DEFAULT_MENU_KEY
    }
}

/// One mod's option values as the `"options"` object: `unknown` first, then every persisted
/// option.
fn options_to_json(
    descriptor: &ModDescriptor,
    values: &[OptionValue],
    unknown: &Map<String, Value>,
) -> Map<String, Value> {
    let mut options = unknown.clone();
    for (option, value) in descriptor.options.iter().zip(values) {
        if option.kind.is_persisted() {
            options.insert(option.id.into(), value_to_json(&option.kind, value));
        }
    }
    options
}

/// A `{"name": .., "options": {..}}` profile. `None` without a usable name. Options it does
/// not mention hold their default; unreadable values fall back to the default too.
fn profile_from_json(descriptor: &ModDescriptor, json: &Value) -> Option<ModProfile> {
    let map = json.as_object()?;
    let name = clean_profile_name(map.get("name")?.as_str()?);
    if name.is_empty() {
        return None;
    }
    let mut profile = ModProfile {
        name,
        values: default_values(descriptor),
        unknown: Map::new(),
    };
    if let Some(options) = map.get("options").and_then(Value::as_object) {
        for (option_id, json) in options {
            let Some(o) = descriptor.option_index(option_id) else {
                profile.unknown.insert(option_id.clone(), json.clone());
                continue;
            };
            let option = &descriptor.options[o];
            if !option.kind.is_persisted() {
                continue;
            }
            if let Some(value) = value_from_json(&option.kind, json) {
                profile.values[o] = option.kind.sanitize(value, &option.default);
            }
        }
    }
    Some(profile)
}

pub(crate) fn json_bool(json: &Value) -> Option<bool> {
    match json {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "on" | "yes" | "1" => Some(true),
            "false" | "off" | "no" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn json_f64(json: &Value) -> Option<f64> {
    match json {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Parse `#AARRGGBB`, `#RRGGBB` (opaque), with or without `#` or `0x`.
#[must_use]
pub fn parse_color(text: &str) -> Option<u32> {
    let hex = text.trim();
    let hex = hex
        .strip_prefix('#')
        .or_else(|| hex.strip_prefix("0x"))
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    match hex.len() {
        8 => u32::from_str_radix(hex, 16).ok(),
        6 => u32::from_str_radix(hex, 16)
            .ok()
            .map(|rgb| 0xFF00_0000 | rgb),
        _ => None,
    }
}

/// A choice label lower case and in US spelling, so a label saved before the wave's US-spelling
/// pass ("Custom colour") still loads as today's ("Custom color").
fn us_spelling(label: &str) -> String {
    let mut out = label.to_ascii_lowercase();
    for (uk, us) in [
        ("colour", "color"),
        ("armour", "armor"),
        ("centre", "center"),
        ("grey", "gray"),
        ("favourite", "favorite"),
    ] {
        if out.contains(uk) {
            out = out.replace(uk, us);
        }
    }
    out
}

fn value_from_json(kind: &OptionKind, json: &Value) -> Option<OptionValue> {
    match kind {
        OptionKind::Toggle => json_bool(json).map(OptionValue::Bool),
        OptionKind::Slider { .. } | OptionKind::SliderLabeled { .. } => {
            json_f64(json).map(|v| OptionValue::Float(v as f32))
        }
        OptionKind::Choice { choices } => match json {
            Value::String(label) => {
                let wanted = label.trim();
                choices
                    .iter()
                    .position(|choice| choice.eq_ignore_ascii_case(wanted))
                    .or_else(|| {
                        let wanted = us_spelling(wanted);
                        choices
                            .iter()
                            .position(|choice| us_spelling(choice) == wanted)
                    })
                    .map(OptionValue::Choice)
            }
            Value::Number(n) => n
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .map(OptionValue::Choice),
            _ => None,
        },
        OptionKind::Color { .. } => match json {
            Value::String(text) => parse_color(text).map(OptionValue::Color),
            Value::Number(n) => n
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .map(OptionValue::Color),
            _ => None,
        },
        OptionKind::Keybind => json_f64(json).and_then(|v| {
            (v.fract() == 0.0 && v.abs() < 1.0e6).then_some(OptionValue::Key(v as i32))
        }),
        OptionKind::Text { .. } | OptionKind::HiddenText { .. } => json
            .as_str()
            .map(|text| OptionValue::Text(Cow::Owned(text.to_owned()))),
        OptionKind::Button { .. } => Some(OptionValue::Action),
    }
}

/// An `f32` as a JSON number with its shortest decimal form (`0.3`, not `0.30000001192...`).
fn f32_json(v: f32) -> Value {
    // `+ 0.0` turns -0.0 into 0.0: with serde_json's `arbitrary_precision` a `-0` in the file
    // would never compare equal to the `0` a fresh store writes.
    let shortest: f64 = format!("{}", v + 0.0).parse().unwrap_or(0.0);
    Number::from_f64(shortest).map_or(Value::Null, Value::Number)
}

fn value_to_json(kind: &OptionKind, value: &OptionValue) -> Value {
    match value {
        OptionValue::Bool(b) => Value::Bool(*b),
        OptionValue::Float(v) => f32_json(*v),
        OptionValue::Choice(index) => match kind {
            OptionKind::Choice { choices } => choices
                .get(*index)
                .map_or(Value::Null, |label| Value::String((*label).into())),
            _ => Value::Null,
        },
        OptionValue::Color(argb) => Value::String(format!("#{argb:08X}")),
        OptionValue::Key(code) => Value::Number(Number::from(*code)),
        OptionValue::Text(text) => Value::String(text.to_string()),
        OptionValue::Action => Value::Null,
    }
}

fn placement_to_json(placement: &HudPlacement) -> Value {
    let mut map = Map::new();
    map.insert("anchor".into(), Value::String(placement.anchor.id().into()));
    map.insert("x".into(), f32_json(placement.x));
    map.insert("y".into(), f32_json(placement.y));
    map.insert("scale".into(), f32_json(placement.scale));
    Value::Object(map)
}

fn placement_from_json(json: &Value, default: HudPlacement) -> Option<HudPlacement> {
    let map = json.as_object()?;
    let anchor = match map.get("anchor") {
        Some(Value::String(id)) => HudAnchor::from_id(id)?,
        Some(_) => return None,
        None => default.anchor,
    };
    let number = |key: &str, fallback: f32| -> Option<f32> {
        match map.get(key) {
            Some(value) => json_f64(value).map(|v| v as f32),
            None => Some(fallback),
        }
    };
    Some(
        HudPlacement {
            anchor,
            x: number("x", default.x)?,
            y: number("y", default.y)?,
            scale: number("scale", default.scale)?,
        }
        .sanitized(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::Category;
    use crate::descriptor::{HudElementDescriptor, OptionDescriptor};

    static TEST_OPTIONS: [OptionDescriptor; 6] = [
        OptionDescriptor {
            id: "flag",
            label: "Flag",
            kind: OptionKind::Toggle,
            default: OptionValue::Bool(true),
            description: "",
        },
        OptionDescriptor {
            id: "size",
            label: "Size",
            kind: OptionKind::Slider {
                min: 0.5,
                max: 2.0,
                step: 0.25,
                unit: "x",
                decimals: 2,
            },
            default: OptionValue::Float(1.0),
            description: "",
        },
        OptionDescriptor {
            id: "mode",
            label: "Mode",
            kind: OptionKind::Choice {
                choices: &["Off", "Low", "High"],
            },
            default: OptionValue::Choice(1),
            description: "",
        },
        OptionDescriptor {
            id: "tint",
            label: "Tint",
            kind: OptionKind::Color { alpha: false },
            default: OptionValue::Color(0xFF11_2233),
            description: "",
        },
        OptionDescriptor {
            id: "bind",
            label: "Bind",
            kind: OptionKind::Keybind,
            default: OptionValue::Key(46),
            description: "",
        },
        OptionDescriptor {
            id: "name",
            label: "Name",
            kind: OptionKind::Text { max_len: 5 },
            default: OptionValue::Text(Cow::Borrowed("abc")),
            description: "",
        },
    ];

    static TEST_ELEMENTS: [HudElementDescriptor; 1] = [HudElementDescriptor {
        id: "main",
        name: "Main",
        default_placement: HudPlacement::new(HudAnchor::TopRight, -2.0, 2.0),
    }];

    static TEST_MOD: ModDescriptor = ModDescriptor {
        id: "test",
        name: "Test",
        category: Category::World,
        description: "",
        default_enabled: false,
        options: &TEST_OPTIONS,
        hud_elements: &TEST_ELEMENTS,
    };

    static OTHER_MOD: ModDescriptor = ModDescriptor {
        id: "other",
        name: "Other",
        category: Category::Hud,
        description: "",
        default_enabled: false,
        options: &[],
        hud_elements: &[],
    };

    static TEST_CATALOG: [&ModDescriptor; 2] = [&TEST_MOD, &OTHER_MOD];

    fn fresh() -> ModSettings {
        ModSettings::with_catalog(&TEST_CATALOG)
    }

    #[test]
    fn defaults_come_from_the_descriptors() {
        let s = fresh();
        assert!(!s.enabled("test"));
        assert!(s.bool("test", "flag"));
        assert_eq!(s.f32("test", "size"), 1.0);
        assert_eq!(s.choice("test", "mode"), 1);
        assert_eq!(s.choice_label("test", "mode"), "Low");
        assert_eq!(s.color("test", "tint"), 0xFF11_2233);
        assert_eq!(s.key("test", "bind"), 46);
        assert_eq!(s.text("test", "name"), "abc");
        assert_eq!(
            s.placement("test", "main"),
            HudPlacement::new(HudAnchor::TopRight, -2.0, 2.0)
        );
        assert!(!s.enabled("no_such_mod"));
    }

    #[test]
    fn generation_moves_on_every_change_and_only_on_change() {
        let mut s = fresh();
        let g0 = s.generation();
        assert!(g0 >= 1);
        assert!(s.set_enabled("test", true));
        let g1 = s.generation();
        assert!(g1 > g0);
        assert!(!s.set_enabled("test", true), "no-op set");
        assert_eq!(s.generation(), g1);
        assert!(s.set_f32("test", "size", 1.5));
        assert!(s.generation() > g1);
        let g2 = s.generation();
        s.set_placement(
            "test",
            "main",
            HudPlacement::new(HudAnchor::Center, 1.0, 1.0),
        );
        assert!(s.generation() > g2);
    }

    #[test]
    fn setters_clamp_snap_and_coerce() {
        let mut s = fresh();
        s.set_f32("test", "size", 9.0);
        assert_eq!(s.f32("test", "size"), 2.0);
        s.set_f32("test", "size", 1.1);
        assert_eq!(s.f32("test", "size"), 1.0, "snapped to the 0.25 grid");
        s.set_f32("test", "size", f32::NAN);
        assert_eq!(s.f32("test", "size"), 1.0, "NaN falls back to the default");
        s.set_choice("test", "mode", 7);
        assert_eq!(s.choice("test", "mode"), 1, "out-of-range index -> default");
        s.set_color("test", "tint", 0x0012_3456);
        assert_eq!(
            s.color("test", "tint"),
            0xFF12_3456,
            "no-alpha option forces opaque"
        );
        s.set_key("test", "bind", 9999);
        assert_eq!(s.key("test", "bind"), 0, "absurd key code -> unbound");
        s.set_text("test", "name", "abcdefgh\n");
        assert_eq!(s.text("test", "name"), "abcde");
        // Wrong variant: replaced by the default.
        s.set_value("test", "flag", OptionValue::Float(3.0));
        assert!(s.bool("test", "flag"));
    }

    #[test]
    fn json_round_trip_preserves_everything() {
        let mut s = fresh();
        s.set_enabled("test", true);
        s.set_bool("test", "flag", false);
        s.set_f32("test", "size", 1.75);
        s.set_choice("test", "mode", 2);
        s.set_color("test", "tint", 0xFFAB_CDEF);
        s.set_key("test", "bind", -99);
        s.set_text("test", "name", "zz");
        s.set_placement(
            "test",
            "main",
            HudPlacement::new(HudAnchor::BottomLeft, 3.5, -7.0).scaled(1.5),
        );
        let text = s.to_json_string();
        let (back, report) = ModSettings::from_json_str_with_catalog(&text, &TEST_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(!report.corrupt);
        assert_eq!(back.to_json(), s.to_json());
        assert!(back.enabled("test"));
        assert!(!back.bool("test", "flag"));
        assert_eq!(back.f32("test", "size"), 1.75);
        assert_eq!(back.choice_label("test", "mode"), "High");
        assert_eq!(back.color("test", "tint"), 0xFFAB_CDEF);
        assert_eq!(back.key("test", "bind"), -99);
        assert_eq!(back.text("test", "name"), "zz");
        assert_eq!(
            back.placement("test", "main"),
            HudPlacement::new(HudAnchor::BottomLeft, 3.5, -7.0).scaled(1.5)
        );
        // Floats are written in their shortest form.
        assert!(text.contains("1.75"), "{text}");
        assert!(!text.contains("1.7500000"), "{text}");
    }

    /// The real catalog: every mod switched away from its default, every
    /// setting moved off its default and every HUD element moved, saved and loaded again, comes
    /// back exactly, with no warning and no migration note. Catches a choice label that does not
    /// find itself again (a duplicate, or one the spelling fold maps elsewhere), a slider stop
    /// that does not survive the decimal round trip, and a kind that is not written back.
    #[test]
    fn every_setting_of_the_real_catalog_survives_a_save_and_load() {
        let mut s = ModSettings::new();
        let catalog = s.catalog();
        for (m, descriptor) in catalog.iter().enumerate() {
            s.set_enabled_at(m, !descriptor.default_enabled);
            for (o, option) in descriptor.options.iter().enumerate() {
                let moved = match (option.kind, &option.default) {
                    (OptionKind::Toggle, OptionValue::Bool(on)) => OptionValue::Bool(!on),
                    (
                        OptionKind::Slider { min, max, .. }
                        | OptionKind::SliderLabeled { min, max, .. },
                        OptionValue::Float(v),
                    ) => OptionValue::Float(if *v == max { min } else { max }),
                    (OptionKind::Choice { choices }, OptionValue::Choice(i)) => {
                        OptionValue::Choice((i + 1) % choices.len())
                    }
                    (OptionKind::Color { .. }, OptionValue::Color(argb)) => {
                        OptionValue::Color(argb ^ 0x00A5_5A3C)
                    }
                    (OptionKind::Keybind, OptionValue::Key(code)) => {
                        OptionValue::Key(if *code == 47 { 48 } else { 47 })
                    }
                    (OptionKind::Text { .. } | OptionKind::HiddenText { .. }, _) => {
                        OptionValue::Text("ab".into())
                    }
                    _ => continue,
                };
                s.set_value_at(m, o, moved);
            }
            for e in 0..descriptor.hud_elements.len() {
                let placement = HudPlacement::new(HudAnchor::BottomRight, -7.25, -3.5);
                s.set_placement_at(m, e, placement.scaled(1.25));
            }
        }
        let (back, report) = ModSettings::from_json_str(&s.to_json_string());
        assert!(!report.corrupt);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(report.migrated.is_empty(), "{:?}", report.migrated);
        for (m, descriptor) in catalog.iter().enumerate() {
            assert_eq!(back.enabled_at(m), s.enabled_at(m), "{}", descriptor.id);
            for (o, option) in descriptor.options.iter().enumerate() {
                assert_eq!(
                    back.value_at(m, o),
                    s.value_at(m, o),
                    "{}.{}",
                    descriptor.id,
                    option.id
                );
            }
            for e in 0..descriptor.hud_elements.len() {
                assert_eq!(
                    back.placement_at(m, e),
                    s.placement_at(m, e),
                    "{}",
                    descriptor.id
                );
            }
        }
        assert_eq!(back.to_json(), s.to_json());
    }

    #[test]
    fn missing_unknown_and_bad_values_are_tolerated() {
        let text = r##"{
            "format": 7,
            "extra_root_key": [1, 2],
            "mods": {
                "test": {
                    "enabled": "yes",
                    "options": {
                        "flag": "maybe",
                        "size": 50,
                        "mode": "HIGH",
                        "tint": "#123",
                        "bind": 1.5,
                        "name": 42,
                        "from_the_future": {"a": 1}
                    },
                    "hud": {
                        "main": {"anchor": "nowhere", "x": 1},
                        "ghost": {"anchor": "center"}
                    }
                },
                "other": 12,
                "renamed_mod": {"enabled": true}
            }
        }"##;
        let (s, report) = ModSettings::from_json_str_with_catalog(text, &TEST_CATALOG);
        assert!(!report.corrupt);
        assert!(s.enabled("test"), "\"yes\" reads as true");
        assert!(s.bool("test", "flag"), "unreadable -> default");
        assert_eq!(s.f32("test", "size"), 2.0, "clamped");
        assert_eq!(
            s.choice("test", "mode"),
            2,
            "labels match case-insensitively"
        );
        assert_eq!(s.color("test", "tint"), 0xFF11_2233, "bad hex -> default");
        assert_eq!(s.key("test", "bind"), 46, "fractional key -> default");
        assert_eq!(s.text("test", "name"), "abc", "non-string -> default");
        assert_eq!(
            s.placement("test", "main"),
            HudPlacement::new(HudAnchor::TopRight, -2.0, 2.0),
            "bad anchor -> default placement"
        );
        assert!(report.warnings.len() >= 6, "{:?}", report.warnings);
        // Unknown keys survive a save.
        let saved = s.to_json();
        assert!(saved["mods"]["renamed_mod"]["enabled"].as_bool().unwrap());
        assert_eq!(saved["mods"]["test"]["options"]["from_the_future"]["a"], 1);
        assert_eq!(saved["mods"]["test"]["hud"]["ghost"]["anchor"], "center");
    }

    /// A mod entry whose `options` or `hud` is not an object, or whose
    /// `enabled` is not a boolean, loads its defaults, and the log says so in words that are
    /// true (the default of a default-on mod is on, not "left off").
    #[test]
    fn a_misshapen_entry_loads_its_defaults_and_says_so() {
        let text = r#"{"mods":{"test":{"enabled":"perhaps","options":[1,2],"hud":"top"}}}"#;
        let (s, report) = ModSettings::from_json_str_with_catalog(text, &TEST_CATALOG);
        assert!(!report.corrupt);
        let fresh = fresh();
        assert_eq!(
            s.enabled("test"),
            fresh.enabled("test"),
            "the default stays"
        );
        assert!(s.options_are_default_at(s.index_of("test").unwrap()));
        assert_eq!(s.placement("test", "main"), fresh.placement("test", "main"));
        assert_eq!(
            report.warnings,
            [
                "test.enabled: not a boolean, default kept",
                "test.options: not an object, defaults kept",
                "test.hud: not an object, defaults kept",
            ],
        );
    }

    /// 256-character chat is always on, so the `long_chat` option is
    /// gone. Every older shape of it (the standalone mod of 2026-11-09 and Chat's option) still
    /// loads cleanly: no warning, no migration note, the rest of the entry intact, and the old
    /// key kept for the round trip like any unknown key.
    #[test]
    fn long_chat_is_gone_and_old_files_with_it_still_load() {
        assert!(catalog::all().iter().all(|m| m.id != "long_chat"));
        let chat = catalog::find("chat").expect("chat");
        assert!(chat.option("long_chat").is_none());
        for old in [
            r#"{"mods":{"long_chat":{"enabled":false},"chat":{"enabled":true}}}"#,
            r#"{"mods":{"chat":{"enabled":true,"options":{"long_chat":false,"raise":true}}}}"#,
            r#"{"mods":{"long_chat":{"enabled":true},"chat":{"options":{"long_chat":true}}}}"#,
        ] {
            let (settings, report) = ModSettings::from_json_str(old);
            assert!(report.warnings.is_empty(), "{old}: {:?}", report.warnings);
            assert!(report.migrated.is_empty(), "{old}: {:?}", report.migrated);
            let saved = settings.to_json();
            if old.contains(r#""options":{"long_chat":false,"raise":true}"#) {
                assert!(settings.enabled("chat"));
                assert!(
                    settings.bool("chat", "raise"),
                    "the rest of the entry loads"
                );
                assert_eq!(saved["mods"]["chat"]["options"]["long_chat"], false);
            }
            let (again, report) = ModSettings::from_json_str(&settings.to_json_string());
            assert!(report.warnings.is_empty() && report.migrated.is_empty());
            assert_eq!(again.to_json(), saved, "{old}: a stable round trip");
        }
    }

    #[test]
    fn garbage_and_missing_files_load_defaults() {
        let (s, report) = ModSettings::from_json_str_with_catalog("{not json", &TEST_CATALOG);
        assert!(report.corrupt);
        assert!(!s.enabled("test"));
        let (_, report) = ModSettings::from_json_str_with_catalog("[1,2,3]", &TEST_CATALOG);
        assert!(report.corrupt);

        let dir = std::env::temp_dir().join(format!("mc-mods-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("missing.json");
        let _ = std::fs::remove_file(&path);
        let (s, report) = ModSettings::load(&path);
        assert!(report.file_missing && !report.corrupt);
        assert!(s.generation() >= 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One byte that is not UTF-8 no longer makes the whole file
    /// "corrupt" and every mod go back to its default: the rest loads, and the report asks the
    /// caller to keep a copy. Negative control: the old `read_to_string` refuses the file.
    #[test]
    fn a_non_utf8_byte_costs_its_characters_not_the_settings() {
        let dir = std::env::temp_dir().join(format!("mc-mods-encoding-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(crate::SETTINGS_FILE_NAME);
        let mut s = ModSettings::new();
        let first = s.catalog()[0].id;
        s.set_enabled(first, true);
        let text = s.to_json_string();
        let open = text.find('{').expect("a JSON object");
        let mut bytes = text.as_bytes()[..=open].to_vec();
        bytes.extend_from_slice(b"\"note\":\"caf\xE9\",");
        bytes.extend_from_slice(&text.as_bytes()[open + 1..]);
        std::fs::write(&path, &bytes).unwrap();
        assert!(std::fs::read_to_string(&path).is_err(), "negative control");

        let (back, report) = ModSettings::load(&path);
        assert!(back.enabled(first), "the settings themselves loaded");
        assert!(report.reencoded && !report.corrupt, "{report:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_and_load_through_a_real_file() {
        let dir = std::env::temp_dir().join(format!("mc-mods-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(crate::SETTINGS_FILE_NAME);
        let mut s = ModSettings::new();
        let first = s.catalog()[0].id;
        s.set_enabled(first, true);
        s.save(&path).unwrap();
        let (back, report) = ModSettings::load(&path);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(back.enabled(first));
        assert_eq!(back.to_json(), s.to_json());
        let _ = std::fs::remove_dir_all(&dir);
    }

    static ACTION_OPTIONS: [OptionDescriptor; 2] = [
        OptionDescriptor {
            id: "test_mic",
            label: "Test mic",
            kind: OptionKind::Button { label: "Test" },
            default: OptionValue::Action,
            description: "",
        },
        OptionDescriptor {
            id: "device",
            label: "Device",
            kind: OptionKind::HiddenText { max_len: 64 },
            default: OptionValue::Text(Cow::Borrowed("")),
            description: "",
        },
    ];

    static ACTION_MOD: ModDescriptor = ModDescriptor {
        id: "voice",
        name: "Voice",
        category: Category::Players,
        description: "",
        default_enabled: false,
        options: &ACTION_OPTIONS,
        hud_elements: &[],
    };

    static ACTION_CATALOG: [&ModDescriptor; 1] = [&ACTION_MOD];

    #[test]
    fn buttons_are_not_persisted_and_hidden_text_is() {
        let mut s = ModSettings::with_catalog(&ACTION_CATALOG);
        assert!(s.set_text("voice", "device", "Headset (USB)"));
        assert!(
            !s.set_value("voice", "test_mic", OptionValue::Action),
            "nothing to change"
        );
        let json = s.to_json();
        assert!(json["mods"]["voice"]["options"].get("test_mic").is_none());
        assert_eq!(json["mods"]["voice"]["options"]["device"], "Headset (USB)");
        let text = r#"{"mods":{"voice":{"options":{"test_mic":5,"device":"Mic 2"}}}}"#;
        let (back, report) = ModSettings::from_json_str_with_catalog(text, &ACTION_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(back.text("voice", "device"), "Mic 2");
        assert_eq!(back.value("voice", "test_mic"), Some(&OptionValue::Action));
        assert!(!OptionKind::HiddenText { max_len: 4 }.is_visible());
        assert!(!OptionKind::Button { label: "x" }.is_persisted());
    }

    fn clock_label(value: f32, out: &mut String) {
        use std::fmt::Write as _;
        if value >= 24.0 {
            out.push_str("Off");
        } else {
            let minutes = (value * 60.0).round() as u32;
            let _ = write!(out, "{:02}:{:02}", minutes / 60, minutes % 60);
        }
    }

    static LABELED_OPTIONS: [OptionDescriptor; 2] = [
        OptionDescriptor {
            id: "time",
            label: "Time",
            kind: OptionKind::SliderLabeled {
                min: 0.0,
                max: 24.0,
                step: 0.25,
                label: crate::SliderLabel(clock_label),
            },
            default: OptionValue::Float(12.0),
            description: "",
        },
        OptionDescriptor {
            id: "offset",
            label: "Offset",
            kind: OptionKind::Slider {
                min: -0.2,
                max: 0.2,
                step: 0.005,
                unit: "",
                decimals: 3,
            },
            default: OptionValue::Float(0.0),
            description: "",
        },
    ];

    static LABELED_MOD: ModDescriptor = ModDescriptor {
        id: "clock",
        name: "Clock",
        category: Category::World,
        description: "",
        default_enabled: false,
        options: &LABELED_OPTIONS,
        hud_elements: &[],
    };

    static LABELED_CATALOG: [&ModDescriptor; 1] = [&LABELED_MOD];

    #[test]
    fn labeled_sliders_snap_like_sliders_and_badge_through_their_formatter() {
        let mut s = ModSettings::with_catalog(&LABELED_CATALOG);
        let kind = LABELED_OPTIONS[0].kind;
        assert_eq!(kind.format_value(&OptionValue::Float(13.5)), "13:30");
        assert_eq!(kind.format_value(&OptionValue::Float(24.0)), "Off");
        // Clamped and snapped to the 0.25 grid exactly as a plain slider.
        assert!(s.set_f32("clock", "time", 6.13));
        assert_eq!(s.f32("clock", "time"), 6.25);
        assert!(s.set_f32("clock", "time", 99.0));
        assert_eq!(s.f32("clock", "time"), 24.0);
        assert_eq!(kind, kind, "fn-pointer labels compare by identity");
        // Persisted as a plain number and read back the same.
        let json = s.to_json();
        assert_eq!(json["mods"]["clock"]["options"]["time"], 24.0);
        let text = r#"{"mods":{"clock":{"options":{"time":"7.5"}}}}"#;
        let (back, report) = ModSettings::from_json_str_with_catalog(text, &LABELED_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(back.f32("clock", "time"), 7.5);
    }

    #[test]
    fn a_symmetric_slider_never_stores_or_writes_negative_zero() {
        let kind = LABELED_OPTIONS[1].kind;
        let OptionValue::Float(zero) =
            kind.sanitize(OptionValue::Float(0.0), &OptionValue::Float(0.0))
        else {
            panic!("a slider sanitises to a float");
        };
        assert!(
            zero.is_sign_positive(),
            "snapped 0 must be +0, got {zero:?}"
        );
        let mut s = ModSettings::with_catalog(&LABELED_CATALOG);
        assert!(s.set_f32("clock", "offset", 0.05));
        assert!(s.set_f32("clock", "offset", -0.0001));
        assert!(s.f32("clock", "offset").is_sign_positive());
        let text = s.to_json_string();
        assert!(!text.contains("-0"), "{text}");
        let (back, _) = ModSettings::from_json_str_with_catalog(&text, &LABELED_CATALOG);
        assert_eq!(back.to_json(), s.to_json());
    }

    /// "No Dynamic FOV" became "Dynamic FOV", default on. The
    /// user's file had `"dynamic_fov": {"enabled": true, "options": {"keep_bow": false,
    /// "strength": 0.0}}`, a FOV that never moved, so it must load as Dynamic FOV **off**, keep
    /// loading that way after a save, and leave every other mod as it was.
    ///
    /// With `MC_MODS_MIGRATION_FILE` set to a **copy** of a settings file, the same checks run
    /// on that file (it is only read; the re-save goes to a temp dir).
    #[test]
    fn a_no_dynamic_fov_file_loads_as_dynamic_fov_off_and_stays_so() {
        let users_entry = r#"{
            "format": 1,
            "mods": {
                "dynamic_fov": {"enabled": true, "options": {"keep_bow": false, "strength": 0.0}},
                "fps": {"enabled": true, "hud": {"fps": {"anchor": "top_left", "scale": 0.8,
                        "x": 2.0, "y": 2.0}}},
                "zoom": {"enabled": true}
            }
        }"#;
        let file = std::env::var_os("MC_MODS_MIGRATION_FILE");
        let text = match &file {
            Some(path) => std::fs::read_to_string(path).expect("read the settings copy"),
            None => users_entry.to_owned(),
        };
        let raw: Value = serde_json::from_str(&text).expect("valid JSON");
        let (s, report) = ModSettings::from_json_str(&text);
        assert!(!report.corrupt);
        assert!(
            !s.enabled("dynamic_fov"),
            "a FOV that never moved is Dynamic FOV off"
        );
        assert_eq!(
            s.f32("dynamic_fov", "strength"),
            100.0,
            "back to vanilla when turned on"
        );
        assert!(s.bool("dynamic_fov", "bow_zoom"));
        assert_eq!(report.migrated.len(), 1, "{:?}", report.migrated);
        // Every other mod keeps its saved switch (or its default when it has none).
        for descriptor in s.catalog() {
            if descriptor.id == "dynamic_fov" {
                continue;
            }
            let saved = raw["mods"][descriptor.id]["enabled"].as_bool();
            assert_eq!(
                s.enabled(descriptor.id),
                saved.unwrap_or(descriptor.default_enabled),
                "{}",
                descriptor.id
            );
        }
        // The upgraded entry: no `keep_bow`, a `bow_zoom`, and no second migration.
        let saved = s.to_json();
        let entry = &saved["mods"]["dynamic_fov"];
        assert_eq!(entry["enabled"], false);
        assert!(entry["options"].get("keep_bow").is_none(), "{entry}");
        assert_eq!(entry["options"]["bow_zoom"], true);
        let dir = std::env::temp_dir().join(format!("mc-mods-migrate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(crate::SETTINGS_FILE_NAME);
        s.save(&path).unwrap();
        let (back, report) = ModSettings::load(&path);
        assert!(report.migrated.is_empty(), "{:?}", report.migrated);
        assert!(!back.enabled("dynamic_fov"));
        assert_eq!(back.to_json(), saved);
        let _ = std::fs::remove_dir_all(&dir);
        if let Some(path) = file {
            eprintln!("checked {}", std::path::Path::new(&path).display());
        }
    }

    /// A fresh store is current: Dynamic FOV on at the vanilla defaults, and its file needs no
    /// migration when read back.
    #[test]
    fn a_fresh_store_has_dynamic_fov_on_and_round_trips_without_migration() {
        let s = ModSettings::new();
        assert!(s.enabled("dynamic_fov"));
        assert_eq!(s.f32("dynamic_fov", "strength"), 100.0);
        assert!(s.bool("dynamic_fov", "bow_zoom"));
        let (back, report) = ModSettings::from_json_str(&s.to_json_string());
        assert!(report.migrated.is_empty() && report.warnings.is_empty());
        assert_eq!(back.to_json(), s.to_json());
        // A file from before the mod existed has no entry at all: the new default, no note.
        let (old, report) = ModSettings::from_json_str(r#"{"format":1,"mods":{}}"#);
        assert!(old.enabled("dynamic_fov") && report.migrated.is_empty());
    }

    #[test]
    fn color_parsing() {
        assert_eq!(parse_color("#80FF0000"), Some(0x80FF_0000));
        assert_eq!(parse_color("ff0000"), Some(0xFFFF_0000));
        assert_eq!(parse_color("0x01020304"), Some(0x0102_0304));
        assert_eq!(parse_color("#12345"), None);
        assert_eq!(parse_color("#GG0000"), None);
    }


    /// Item 11: the menu key is Right Shift until changed, persists at the root, and can never be
    /// unbound or put on a mouse button (nothing else opens the menu).
    #[test]
    fn the_menu_key_defaults_to_right_shift_persists_and_is_never_unbound() {
        let mut s = fresh();
        assert_eq!(s.menu_key(), DEFAULT_MENU_KEY);
        assert_eq!(DEFAULT_MENU_KEY, 54, "LWJGL RSHIFT");
        let g = s.generation();
        assert!(s.set_menu_key(50), "M");
        assert!(s.generation() > g, "a change is saved like any other");
        assert!(!s.set_menu_key(50), "no-op");
        let json = s.to_json();
        assert_eq!(json["menu"]["open_key"], 50);
        let (back, report) =
            ModSettings::from_json_str_with_catalog(&s.to_json_string(), &TEST_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(back.menu_key(), 50);
        // 1 is Escape, which must keep opening the pause menu.
        for bad in [0, 1, -100, -98, 256, 9999] {
            let mut s = fresh();
            s.set_menu_key(50);
            s.set_menu_key(bad);
            assert_eq!(
                s.menu_key(),
                DEFAULT_MENU_KEY,
                "{bad} falls back to Right Shift"
            );
        }
        let (escape, report) = ModSettings::from_json_str_with_catalog(
            r#"{"menu":{"open_key":1},"mods":{}}"#,
            &TEST_CATALOG,
        );
        assert_eq!(escape.menu_key(), DEFAULT_MENU_KEY, "a hand-edited Escape");
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(s.set_menu_key(2), "1 on the main row is still a menu key");
        // A file without the key, or with a broken one, opens on Right Shift.
        let (old, report) =
            ModSettings::from_json_str_with_catalog(r#"{"format":1,"mods":{}}"#, &TEST_CATALOG);
        assert!(report.warnings.is_empty());
        assert_eq!(old.menu_key(), DEFAULT_MENU_KEY);
        let (broken, report) = ModSettings::from_json_str_with_catalog(
            r#"{"menu":{"open_key":"none","theme":"dark"},"mods":{}}"#,
            &TEST_CATALOG,
        );
        assert_eq!(broken.menu_key(), DEFAULT_MENU_KEY);
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert_eq!(
            broken.to_json()["menu"]["theme"],
            "dark",
            "unknown menu keys survive"
        );
    }

    /// Items 12/43: profiles hold every option, load back exactly, update by name, delete, mark
    /// the current one, and survive the file.
    #[test]
    fn profiles_save_load_delete_and_round_trip_through_json() {
        let mut s = fresh();
        let m = s.index_of("test").unwrap();
        assert!(s.profiles_at(m).is_empty());
        s.set_f32("test", "size", 1.5);
        s.set_bool("test", "flag", false);
        let g = s.generation();
        assert_eq!(s.save_profile_at(m, "  Warm\n"), Some(0));
        assert!(s.generation() > g, "saving a profile is saved to the file");
        assert_eq!(
            s.profiles_at(m)[0].name(),
            "Warm",
            "trimmed, no control characters"
        );
        assert!(s.profile_is_current_at(m, 0));
        // An empty name gets a numbered one; the same name (any case) updates in place.
        s.set_f32("test", "size", 0.5);
        assert_eq!(s.save_profile_at(m, ""), Some(1));
        assert_eq!(s.profiles_at(m)[1].name(), "Profile 2");
        assert!(!s.profile_is_current_at(m, 0));
        assert!(s.profile_is_current_at(m, 1));
        assert_eq!(s.profile_named_at(m, "WARM"), Some(0));
        // Load: every option comes back, including ones changed since.
        s.set_choice("test", "mode", 2);
        assert!(s.load_profile_at(m, 0));
        assert_eq!(s.f32("test", "size"), 1.5);
        assert!(!s.bool("test", "flag"));
        assert_eq!(s.choice("test", "mode"), 1, "the default it was saved with");
        assert!(!s.load_profile_at(m, 0), "already current: no change");
        // Update by name.
        s.set_f32("test", "size", 2.0);
        assert_eq!(s.save_profile_at(m, "warm"), Some(0));
        assert_eq!(s.profiles_at(m).len(), 2);
        assert_eq!(s.profiles_at(m)[0].name(), "warm");
        assert_eq!(s.profiles_at(m)[0].value(1), Some(&OptionValue::Float(2.0)));
        // Through the file, with an option from another build kept.
        let mut json = s.to_json();
        json["mods"]["test"]["profiles"][0]["options"]["from_the_future"] = Value::from(3);
        let text = serde_json::to_string(&json).unwrap();
        let (back, report) = ModSettings::from_json_str_with_catalog(&text, &TEST_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(back.profiles_at(m).len(), 2);
        assert_eq!(back.profiles_at(m)[0].name(), "warm");
        assert_eq!(
            back.to_json()["mods"]["test"]["profiles"][0]["options"]["from_the_future"],
            3
        );
        assert_eq!(back.profiles_at(m)[1], s.profiles_at(m)[1]);
        for o in 0..TEST_OPTIONS.len() {
            assert_eq!(
                back.profiles_at(m)[0].value(o),
                s.profiles_at(m)[0].value(o)
            );
        }
        // Delete.
        let g = s.generation();
        assert!(s.delete_profile_at(m, 0));
        assert!(s.generation() > g);
        assert_eq!(s.profiles_at(m).len(), 1);
        assert_eq!(s.profiles_at(m)[0].name(), "Profile 2");
        assert!(!s.delete_profile_at(m, 5));
        // No profiles: no key in the file.
        assert!(s.delete_profile_at(m, 0));
        assert!(s.to_json()["mods"]["test"].get("profiles").is_none());
        // A mod with no options can still hold an (empty) profile; unknown index is harmless.
        assert!(s.profiles_at(99).is_empty());
        assert_eq!(s.save_profile_at(99, "x"), None);
    }

    #[test]
    fn profiles_stop_at_the_limit_and_bad_entries_are_skipped() {
        let mut s = fresh();
        let m = s.index_of("test").unwrap();
        for i in 0..MAX_PROFILES {
            assert_eq!(s.save_profile_at(m, &format!("P{i}")), Some(i));
        }
        assert_eq!(s.save_profile_at(m, "one too many"), None);
        assert_eq!(s.save_profile_at(m, "p0"), Some(0), "updating still works");
        let long = "x".repeat(PROFILE_NAME_MAX + 10);
        assert_eq!(clean_profile_name(&long).chars().count(), PROFILE_NAME_MAX);
        let text = r#"{"mods":{"test":{"profiles":[
            {"name":"Good","options":{"size":"1.25","mode":"High","bind":"x"}},
            {"name":"   "},
            "not an object",
            {"options":{}}
        ]}}}"#;
        let (back, report) = ModSettings::from_json_str_with_catalog(text, &TEST_CATALOG);
        assert_eq!(back.profiles_at(m).len(), 1);
        let good = &back.profiles_at(m)[0];
        assert_eq!(good.value(1), Some(&OptionValue::Float(1.25)));
        assert_eq!(good.value(2), Some(&OptionValue::Choice(2)));
        assert_eq!(
            good.value(4),
            Some(&OptionValue::Key(46)),
            "unreadable -> default"
        );
        assert_eq!(report.warnings.len(), 3, "{:?}", report.warnings);
    }

    /// Toggle Sprint's saved toggle is what the player is doing, not
    /// a setting. A whole-setup profile neither saves nor loads it (one saved while sprinting
    /// used to switch the toggle on when loaded), tapping Sprint after loading a profile leaves
    /// that profile current, and the toggle alone never counts as a changed setting. Real
    /// settings of the same mod still go with the profile.
    #[test]
    fn runtime_state_is_not_part_of_a_setup() {
        use crate::mods::toggle_sprint::SPRINT_TOGGLED_ID as TOGGLED;
        const ID: &str = "toggle_sprint";
        let mut s = ModSettings::new();
        let m = s.index_of(ID).expect("toggle sprint");
        s.set_text(ID, TOGGLED, "on");
        assert!(s.options_are_default_at(m), "a toggled sprint is no change");
        let json = s.setup_json();
        assert!(json["mods"][ID]["options"].get(TOGGLED).is_none());
        assert!(json["mods"][ID]["options"].get("show_text").is_some());
        let (profile, _) = ModSettings::from_json_str(&json.to_string());
        assert!(s.setup_eq(&profile), "the profile just saved is current");
        s.set_text(ID, TOGGLED, "off");
        assert!(s.setup_eq(&profile), "tapping Sprint keeps it current");
        // A real setting differs, and loading puts it back without touching the toggle.
        s.set_bool(ID, "show_text", false);
        assert!(!s.setup_eq(&profile));
        assert_eq!(s.setup_diff(&profile).settings, vec![m]);
        assert!(s.apply_setup(&profile));
        assert!(s.bool(ID, "show_text"));
        assert_eq!(s.text(ID, TOGGLED), "off");
        // A profile file from before this rule, which still holds the toggle, does not set it.
        let mut old = ModSettings::new();
        old.set_text(ID, TOGGLED, "on");
        old.set_bool(ID, "show_text", false);
        let (old_profile, _) = ModSettings::from_json_str(&old.to_json_string());
        assert!(s.apply_setup(&old_profile));
        assert!(!s.bool(ID, "show_text"));
        assert_eq!(s.text(ID, TOGGLED), "off", "the live toggle stays");
        // The settings file itself still keeps it across a restart.
        s.set_text(ID, TOGGLED, "on");
        let (restarted, _) = ModSettings::from_json_str(&s.to_json_string());
        assert_eq!(restarted.text(ID, TOGGLED), "on");
        assert!(is_runtime_state(ID, TOGGLED) && !is_runtime_state(ID, "show_text"));
    }

    /// Keys this build does not know, at the root and inside a mod's entry, are written back.
    #[test]
    fn unknown_root_and_entry_keys_survive_a_save() {
        let text = r#"{
            "format": 1,
            "launcher": {"last": "x"},
            "mods": {"test": {"enabled": true, "pinned": [1, 2]}}
        }"#;
        let (s, report) = ModSettings::from_json_str_with_catalog(text, &TEST_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let json = s.to_json();
        assert_eq!(json["launcher"]["last"], "x");
        assert_eq!(json["mods"]["test"]["pinned"][1], 2);
        assert_eq!(json["mods"]["test"]["enabled"], true);
        assert_eq!(json["format"], 1);
    }

    /// Item 12: "Reset to defaults" puts the options back and nothing else.
    #[test]
    fn reset_to_defaults_resets_options_only() {
        let mut s = fresh();
        let m = s.index_of("test").unwrap();
        assert!(s.options_are_default_at(m));
        s.set_enabled("test", true);
        s.set_f32("test", "size", 2.0);
        s.set_text("test", "name", "zz");
        let moved = HudPlacement::new(HudAnchor::Center, 5.0, 5.0);
        s.set_placement("test", "main", moved);
        s.save_profile_at(m, "Mine");
        assert!(!s.options_are_default_at(m));
        let g = s.generation();
        assert!(s.reset_options_at(m));
        assert!(s.generation() > g);
        assert!(s.options_are_default_at(m));
        assert_eq!(s.f32("test", "size"), 1.0);
        assert_eq!(s.text("test", "name"), "abc");
        assert!(s.enabled("test"), "the switch stays");
        assert_eq!(s.placement("test", "main"), moved, "the HUD position stays");
        assert_eq!(s.profiles_at(m).len(), 1, "profiles stay");
        assert!(!s.reset_options_at(m), "nothing left to reset");
        assert!(s.options_are_default_at(s.index_of("other").unwrap()));
    }

    /// The user's real `mc-rs-mods.json` (a **copy**: `MC_MODS_USER_FILE=<path>`; an inline
    /// sample of it otherwise) loads with no warning, and saving it again keeps every value it
    /// had: the wave only adds the menu key and entries for mods it did not know.
    #[test]
    fn the_users_file_loads_unchanged_and_gains_only_the_menu_key() {
        let sample = r##"{
            "format": 1,
            "mods": {
                "color_grading": {"enabled": false, "options": {"master_slope": 1.12,
                    "offset_g": -0.06, "saturation": 1.26, "slope_r": 0.93}},
                "fps": {"enabled": true, "hud": {"fps": {"anchor": "top_left", "scale": 0.8,
                    "x": 2.0, "y": 2.0}}, "options": {"text_color": "#FFFFFFFF"}},
                "voice_chat": {"enabled": false, "options": {"device": ""}},
                "zoom": {"enabled": true}
            }
        }"##;
        let file = std::env::var_os("MC_MODS_USER_FILE");
        let text = match &file {
            Some(path) => std::fs::read_to_string(path).expect("read the settings copy"),
            None => sample.to_owned(),
        };
        let original: Value = serde_json::from_str(&text).expect("valid JSON");
        let (s, report) = ModSettings::from_json_str(&text);
        assert!(!report.corrupt);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(s.menu_key(), DEFAULT_MENU_KEY);
        let saved = s.to_json();
        assert_subset(&original, &saved, "$");
        assert_eq!(saved["menu"]["open_key"], DEFAULT_MENU_KEY);
        // Stable: a second load and save writes the same file.
        let (again, _) = ModSettings::from_json_str(&s.to_json_string());
        assert_eq!(again.to_json(), saved);
        if let Some(path) = file {
            eprintln!("checked {}", std::path::Path::new(&path).display());
        }
    }

    /// Pins are a top-level list that older files simply lack.
    /// A file without one loads with no pins and saves without the key; pinning moves the
    /// generation (so the client saves it) and survives a round trip; ids this build does not
    /// know are kept, junk is skipped with a warning.
    #[test]
    fn pins_are_a_top_level_list_that_older_files_lack() {
        let old = r#"{"format":1,"mods":{"test":{"enabled":true}}}"#;
        let (mut s, report) = ModSettings::from_json_str_with_catalog(old, &TEST_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let m = s.index_of("test").unwrap();
        assert!(!s.pinned_at(m));
        assert!(s.to_json().get("pinned").is_none(), "no pins: no key");
        let g = s.generation();
        assert!(s.set_pinned_at(m, true));
        assert!(s.generation() > g, "a pin is saved like any change");
        assert!(!s.set_pinned_at(m, true), "already pinned");
        assert!(s.pinned_at(m) && !s.pinned_at(s.index_of("other").unwrap()));
        assert_eq!(s.to_json()["pinned"], Value::from(vec!["test"]));
        let (back, report) =
            ModSettings::from_json_str_with_catalog(&s.to_json_string(), &TEST_CATALOG);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(back.pinned_at(m));
        assert_eq!(back.to_json(), s.to_json(), "stable");
        // A newer build's mod stays pinned; junk and repeats are dropped.
        let text = r#"{"pinned": ["future_mod", "test", 7, "", "test"], "mods": {}}"#;
        let (mut s, report) = ModSettings::from_json_str_with_catalog(text, &TEST_CATALOG);
        assert_eq!(s.pinned_ids(), ["future_mod", "test"]);
        assert_eq!(report.warnings.len(), 2, "{:?}", report.warnings);
        assert_eq!(
            s.to_json()["pinned"],
            Value::from(vec!["future_mod", "test"])
        );
        assert!(s.set_pinned_at(m, false));
        assert_eq!(
            s.pinned_ids(),
            ["future_mod"],
            "unpinning removes only that id"
        );
        // Profiles replace the whole list.
        assert!(s.set_pinned_ids(["test", " test ", "other"]));
        assert_eq!(s.pinned_ids(), ["test", "other"]);
        assert!(!s.set_pinned_ids(vec!["test".to_owned(), "other".to_owned()]));
        // Not a list: ignored with a warning, nothing else lost.
        let (s, report) =
            ModSettings::from_json_str_with_catalog(r#"{"pinned": "test"}"#, &TEST_CATALOG);
        assert!(s.pinned_ids().is_empty());
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    }

    /// Every key of `old` is in `new` with the same value (objects recursively).
    fn assert_subset(old: &Value, new: &Value, at: &str) {
        match (old, new) {
            (Value::Object(old), Value::Object(new)) => {
                for (key, value) in old {
                    let Some(kept) = new.get(key) else {
                        panic!("{at}.{key} was dropped");
                    };
                    assert_subset(value, kept, &format!("{at}.{key}"));
                }
            }
            (Value::Number(a), Value::Number(b)) => {
                assert_eq!(a.as_f64(), b.as_f64(), "{at} changed");
            }
            _ => assert_eq!(old, new, "{at} changed"),
        }
    }
}
