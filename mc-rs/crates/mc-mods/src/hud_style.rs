//! The shared HUD style: one
//! text color, text shadow, background, background color and brackets setting for every text
//! readout, instead of the same five settings repeated in each.
//!
//! * Each readout ([`READOUTS`]) uses either the HUD style or its own settings ("Custom").
//! * [`ModSettings`](crate::ModSettings) resolves it. While a readout uses the HUD style, the
//!   value of one of its style options ([`STYLE_IDS`]) **is** the shared value, for reading and
//!   for writing. So every reader (the HUD elements' `HudTextStyle::read`, Armor Status, the
//!   menu's rows) sees what is drawn without knowing the shared style exists, and the menu's
//!   ordinary rows edit the shared style while the readout uses it.
//! * A readout keeps its own values while it uses the HUD style: switching it back to Custom
//!   brings them back. A readout that never had its own look starts Custom from the HUD style's
//!   look, so switching never makes it jump.
//!
//! # Defaults, and files saved before the shared style existed
//!
//! * A readout whose own defaults are the HUD style's defaults uses the HUD style on a fresh
//!   install ([`uses_hud_style_by_default`]). Toggle Sprint (plain text in brackets, no box)
//!   starts as Custom, so a fresh install looks exactly as before.
//! * An entry without a `"style"` key reads as [`derived_uses_hud_style`]: a readout whose style
//!   settings were changed loads as Custom, so nothing on screen changes ("readouts you've
//!   already styled load as Custom").
//!
//! # The file
//!
//! * `"hud_style": {"text_color": "#FFFFFFFF", ...}` at the root, written only while some value
//!   differs from its default (a missing value loads as its default);
//! * `"style": "hud"` or `"style": "custom"` in a readout's entry, written only when it differs
//!   from what [`derived_uses_hud_style`] reads from the saved values.
//!
//! So a file nobody styled saves exactly as before, and every file round-trips exactly.

use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionValue};
use crate::presets;

/// The root key of the shared style's values.
pub const ROOT_KEY: &str = "hud_style";
/// The key in a readout's entry that says which style it uses.
pub const ENTRY_KEY: &str = "style";
/// [`ENTRY_KEY`]'s value for "uses the HUD style".
pub const ENTRY_HUD: &str = "hud";
/// [`ENTRY_KEY`]'s value for "uses its own settings".
pub const ENTRY_CUSTOM: &str = "custom";

/// The shared style's settings, in order: the [`presets`] every readout uses, with their
/// labels, kinds and defaults.
pub static STYLE_OPTIONS: [OptionDescriptor; 5] = [
    presets::TEXT_COLOR,
    presets::TEXT_SHADOW,
    presets::BACKGROUND,
    presets::BACKGROUND_COLOR,
    presets::BRACKETS,
];

/// The ids of [`STYLE_OPTIONS`]: the option ids a readout takes from the HUD style.
pub const STYLE_IDS: [&str; 5] = [
    presets::TEXT_COLOR_ID,
    presets::TEXT_SHADOW_ID,
    presets::BACKGROUND_ID,
    presets::BACKGROUND_COLOR_ID,
    presets::BRACKETS_ID,
];

/// The text readouts that can use the HUD style. Each has some of [`STYLE_IDS`] among its
/// options (a test checks it) and takes only those from the HUD style: Armor Status and Held
/// Item Info have no box, so the HUD style's background never adds one.
pub const READOUTS: &[&str] = &[
    "fps",
    "cps",
    "ping",
    "reach_display",
    "coordinates",
    "potion_status",
    "armor_status",
    "item_info",
    "toggle_sprint",
];

/// Mods with a text color that are not text readouts, so they keep their own look: Keystrokes'
/// "Key color" fills the key boxes. A test makes every mod with a `text_color` setting pick a
/// side, so a new readout cannot be forgotten. (The vanilla HUD pieces, such as Boss Bar's
/// `text_shadow` or Tab List's `background_color`, have no text color and are not readouts.)
pub const NOT_READOUTS: &[&str] = &["keystrokes"];

/// The place of a style option in [`STYLE_IDS`].
#[must_use]
pub fn slot(option_id: &str) -> Option<usize> {
    STYLE_IDS.iter().position(|id| *id == option_id)
}

/// Whether the mod `id` can use the HUD style.
#[must_use]
pub fn is_readout(id: &str) -> bool {
    READOUTS.contains(&id)
}

/// The shared style's values on a fresh install.
#[must_use]
pub fn defaults() -> [OptionValue; 5] {
    std::array::from_fn(|i| STYLE_OPTIONS[i].default.clone())
}

/// Whether readout `d` uses the HUD style on a fresh install: it is a readout and its own
/// defaults for the style options it has are the HUD style's defaults.
#[must_use]
pub fn uses_hud_style_by_default(d: &ModDescriptor) -> bool {
    is_readout(d.id)
        && d.options.iter().all(|option| {
            slot(option.id).is_none_or(|s| option.default == STYLE_OPTIONS[s].default)
        })
}

/// Whether a readout whose entry does not say ([`ENTRY_KEY`]) uses the HUD style, from its own
/// saved values: only while it would on a fresh install and none of its style settings was
/// changed. A readout someone styled loads as Custom, so it keeps its look.
#[must_use]
pub fn derived_uses_hud_style(d: &ModDescriptor, values: &[OptionValue]) -> bool {
    uses_hud_style_by_default(d)
        && d.options
            .iter()
            .zip(values)
            .all(|(option, value)| slot(option.id).is_none() || *value == option.default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;

    /// Every readout has style options, every mod with a style option is a readout or is listed
    /// as not one, and only Toggle Sprint (its own plain look) starts as Custom.
    #[test]
    fn every_mod_with_style_settings_is_classified() {
        let all = catalog::all();
        for id in READOUTS.iter().chain(NOT_READOUTS) {
            let d = all
                .iter()
                .find(|d| d.id == *id)
                .unwrap_or_else(|| panic!("{id} is not in the catalog"));
            assert!(
                d.options.iter().any(|option| slot(option.id).is_some()),
                "{id} has no style setting"
            );
        }
        // The vanilla HUD pieces (Boss Bar, Scoreboard, Titles, Chat, Tab List...) have their
        // own `text_shadow` or `background_color`; a readout is a mod with a text color.
        for d in all {
            let readout_like = d.option(presets::TEXT_COLOR_ID).is_some();
            let listed = READOUTS.contains(&d.id) || NOT_READOUTS.contains(&d.id);
            assert_eq!(
                readout_like, listed,
                "{}: a mod with a text_color setting must be in READOUTS or NOT_READOUTS",
                d.id
            );
        }
        let custom: Vec<&str> = all
            .iter()
            .filter(|d| is_readout(d.id) && !uses_hud_style_by_default(d))
            .map(|d| d.id)
            .collect();
        assert_eq!(custom, ["toggle_sprint"]);
        // The style options are the presets, in order.
        for (option, id) in STYLE_OPTIONS.iter().zip(STYLE_IDS) {
            assert_eq!(option.id, id);
        }
    }

    #[test]
    fn a_styled_readout_derives_custom() {
        let fps = catalog::all().iter().find(|d| d.id == "fps").unwrap();
        let mut values: Vec<OptionValue> = fps
            .options
            .iter()
            .map(|option| option.default.clone())
            .collect();
        assert!(derived_uses_hud_style(fps, &values));
        let color = fps.option_index(presets::TEXT_COLOR_ID).unwrap();
        values[color] = OptionValue::Color(0xFFFF_0000);
        assert!(!derived_uses_hud_style(fps, &values), "styled: Custom");
        // A setting that is not a style setting does not matter.
        let mut values: Vec<OptionValue> = fps
            .options
            .iter()
            .map(|option| option.default.clone())
            .collect();
        if let Some(other) = fps.options.iter().position(|o| slot(o.id).is_none()) {
            values[other] = OptionValue::Action;
            assert!(derived_uses_hud_style(fps, &values));
        }
    }

    use crate::settings::ModSettings;
    use serde_json::Value;

    const RED: u32 = 0xFFFF_5555;
    const GREEN: u32 = 0xFF55_FF55;

    /// Reads and writes of a readout's style options go to the shared style while it uses it:
    /// every readout on the HUD style changes together, a Custom one does not, and switching to
    /// Custom keeps the look (a readout never styled copies the HUD style).
    #[test]
    fn readouts_on_the_hud_style_read_and_write_the_shared_values() {
        let mut s = ModSettings::new();
        let (fps, cps, sprint, keys) = (
            s.index_of("fps").unwrap(),
            s.index_of("cps").unwrap(),
            s.index_of("toggle_sprint").unwrap(),
            s.index_of("keystrokes").unwrap(),
        );
        assert!(s.uses_hud_style_at(fps) && s.uses_hud_style_at(cps));
        assert!(
            !s.uses_hud_style_at(sprint),
            "Toggle Sprint keeps its own plain look"
        );
        assert!(!s.hud_style_capable_at(keys) && s.hud_style_capable_at(sprint));
        assert_eq!(s.hud_style_users(), READOUTS.len() - 1);
        let g = s.generation();
        assert!(s.set_hud_style_value(presets::TEXT_COLOR_ID, OptionValue::Color(RED)));
        assert!(s.generation() > g);
        assert_eq!(s.color("fps", "text_color"), RED);
        assert_eq!(s.color("cps", "text_color"), RED);
        assert_eq!(
            s.color("toggle_sprint", "text_color"),
            0xFFFF_FFFF,
            "Custom: its own"
        );
        assert_eq!(
            s.color("keystrokes", "text_color"),
            0xFFFF_FFFF,
            "not a readout"
        );
        // The menu's rows write through the readout: the shared value changes.
        let o = s.catalog()[fps].option_index("text_color").unwrap();
        assert!(s.set_value_at(fps, o, OptionValue::Color(GREEN)));
        assert_eq!(s.color("cps", "text_color"), GREEN);
        assert!(
            s.options_are_default_at(fps),
            "its own settings are untouched"
        );
        // Custom keeps the look it had (it was never styled, so it starts from the HUD style).
        assert!(s.set_uses_hud_style_at(fps, false));
        assert_eq!(s.color("fps", "text_color"), GREEN);
        assert!(
            !s.options_are_default_at(fps),
            "Custom with its own color: changed"
        );
        assert!(s.set_hud_style_value(presets::TEXT_COLOR_ID, OptionValue::Color(RED)));
        assert_eq!(
            s.color("fps", "text_color"),
            GREEN,
            "Custom ignores the HUD style"
        );
        assert_eq!(s.color("cps", "text_color"), RED);
        // Back on the HUD style it follows it again, and its own color comes back on Custom.
        assert!(s.set_uses_hud_style_at(fps, true));
        assert_eq!(s.color("fps", "text_color"), RED);
        assert!(s.set_uses_hud_style_at(fps, false));
        assert_eq!(s.color("fps", "text_color"), GREEN, "its own look returns");
        assert!(!s.set_uses_hud_style_at(keys, true), "not a readout");
        // Reset puts the readout back on the HUD style with its own defaults.
        assert!(s.reset_options_at(fps));
        assert!(s.uses_hud_style_at(fps) && s.options_are_default_at(fps));
        assert_eq!(
            s.color("fps", "text_color"),
            RED,
            "the HUD style itself is not reset"
        );
        // Armor Status has no box: the HUD style never adds one.
        assert!(s.value("armor_status", presets::BACKGROUND_ID).is_none());
    }

    /// A file saved before the shared style: a readout that was styled loads as Custom (the
    /// value drawn is the file's), the others use the HUD style, and the file saves without
    /// any new key. Every later state round-trips exactly.
    #[test]
    fn old_files_keep_their_look_and_every_file_round_trips() {
        let old = r##"{"format": 1, "mods": {
            "fps": {"enabled": true, "options": {"text_color": "#FFFF5555", "brackets": true}},
            "cps": {"enabled": true, "options": {"text_color": "#FFFFFFFF"}},
            "toggle_sprint": {"enabled": true}
        }}"##;
        let (s, report) = ModSettings::from_json_str(old);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let (fps, cps, sprint) = (
            s.index_of("fps").unwrap(),
            s.index_of("cps").unwrap(),
            s.index_of("toggle_sprint").unwrap(),
        );
        assert!(!s.uses_hud_style_at(fps), "styled: Custom");
        assert_eq!(s.color("fps", "text_color"), RED);
        assert!(s.bool("fps", "brackets"));
        assert!(s.uses_hud_style_at(cps) && !s.uses_hud_style_at(sprint));
        let saved = s.to_json();
        assert!(saved.get(ROOT_KEY).is_none(), "no shared style written");
        for id in READOUTS {
            assert!(
                saved["mods"][id].get(ENTRY_KEY).is_none(),
                "{id}: no style key written"
            );
        }
        // Every combination round-trips.
        let mut s = s;
        assert!(s.set_uses_hud_style_at(fps, true));
        assert!(s.set_uses_hud_style_at(cps, false));
        assert!(s.set_uses_hud_style_at(sprint, true));
        assert!(s.set_hud_style_value(presets::BACKGROUND_ID, OptionValue::Bool(false)));
        let saved = s.to_json();
        assert_eq!(saved["mods"]["fps"][ENTRY_KEY], ENTRY_HUD);
        assert_eq!(saved["mods"]["cps"][ENTRY_KEY], ENTRY_CUSTOM);
        assert_eq!(saved["mods"]["toggle_sprint"][ENTRY_KEY], ENTRY_HUD);
        assert_eq!(saved[ROOT_KEY][presets::BACKGROUND_ID], false);
        let (back, report) = ModSettings::from_json_str(&s.to_json_string());
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(back.to_json(), saved, "stable");
        assert!(back.uses_hud_style_at(fps) && !back.uses_hud_style_at(cps));
        assert!(back.uses_hud_style_at(sprint));
        assert!(
            !back.bool("fps", presets::BACKGROUND_ID),
            "from the HUD style"
        );
        assert!(back.setup_eq(&s));
        // Junk is tolerated: a bad style word reads from the settings, a bad root is ignored.
        let junk = r#"{"hud_style": 7, "mods": {"fps": {"style": "sparkly"}}}"#;
        let (j, report) = ModSettings::from_json_str(junk);
        assert_eq!(report.warnings.len(), 2, "{:?}", report.warnings);
        assert!(j.uses_hud_style_at(fps) && j.hud_style_is_default());
        // Unknown keys inside the shared style survive.
        let future = r#"{"hud_style": {"glow": 3, "brackets": true}, "mods": {}}"#;
        let (f, report) = ModSettings::from_json_str(future);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(f.to_json()[ROOT_KEY]["glow"], 3);
        assert!(
            f.bool("fps", "brackets"),
            "fps uses the HUD style's brackets"
        );
        // A mod that is not a readout keeps a "style" key as an unknown key.
        let other = r#"{"mods": {"keystrokes": {"style": "hud"}}}"#;
        let (k, _) = ModSettings::from_json_str(other);
        assert_eq!(k.to_json()["mods"]["keystrokes"][ENTRY_KEY], "hud");
    }

    /// The user's real settings file (a **copy**: `MC_MODS_USER_FILE=<path>`; an inline sample
    /// of it otherwise) loads with no warning, every value it holds is the value drawn (so
    /// nothing on screen changes), and it saves without a shared-style key.
    #[test]
    fn the_users_file_draws_every_value_it_holds() {
        let sample = r##"{"format": 1, "mods": {
            "fps": {"enabled": true, "options": {"background": true,
                "background_color": "#6F000000", "brackets": false,
                "text_color": "#FFFFFFFF", "text_shadow": true}},
            "toggle_sprint": {"enabled": true, "options": {"background": false,
                "brackets": true, "text_color": "#FFFFFFFF", "text_shadow": true}},
            "keystrokes": {"enabled": false, "options": {"text_shadow": false}}
        }}"##;
        let file = std::env::var_os("MC_MODS_USER_FILE");
        let text = match &file {
            Some(path) => std::fs::read_to_string(path).expect("read the settings copy"),
            None => sample.to_owned(),
        };
        let (s, report) = ModSettings::from_json_str(&text);
        assert!(!report.corrupt && report.warnings.is_empty(), "{report:?}");
        let original: Value = serde_json::from_str(&text).unwrap();
        let mut checked = 0;
        for (mod_id, entry) in original["mods"].as_object().unwrap() {
            let Some(m) = s.index_of(mod_id) else {
                continue;
            };
            let Some(options) = entry.get("options").and_then(Value::as_object) else {
                continue;
            };
            let saved = s.to_json();
            for (option_id, json) in options {
                let Some(o) = s.catalog()[m].option_index(option_id) else {
                    continue;
                };
                if !s.catalog()[m].options[o].kind.is_persisted() {
                    continue;
                }
                // It saves unchanged...
                let file_value = &saved["mods"][mod_id]["options"][option_id];
                match (file_value.as_f64(), json.as_f64()) {
                    (Some(a), Some(b)) => assert_eq!(a, b, "{mod_id}.{option_id} changed"),
                    _ => assert_eq!(file_value, json, "{mod_id}.{option_id} changed"),
                }
                // ...and what a reader gets (the value drawn) is what the file says.
                let drawn = s.value(mod_id, option_id).unwrap();
                let same = match drawn {
                    OptionValue::Bool(b) => json.as_bool() == Some(*b),
                    OptionValue::Float(v) => json
                        .as_f64()
                        .is_some_and(|f| (f - f64::from(*v)).abs() < 1e-4),
                    OptionValue::Color(c) => {
                        json.as_str().and_then(crate::settings::parse_color) == Some(*c)
                    }
                    OptionValue::Key(k) => json.as_i64() == Some(i64::from(*k)),
                    OptionValue::Text(t) => json.as_str() == Some(t.as_ref()),
                    OptionValue::Choice(_) | OptionValue::Action => true,
                };
                assert!(same, "{mod_id}.{option_id}: drawn {drawn:?}, file {json}");
                checked += 1;
            }
        }
        assert!(checked > 0);
        let saved = s.to_json();
        assert!(saved.get(ROOT_KEY).is_none());
        for id in READOUTS {
            assert!(saved["mods"][id].get(ENTRY_KEY).is_none(), "{id}");
        }
        let (again, _) = ModSettings::from_json_str(&s.to_json_string());
        assert_eq!(again.to_json(), saved, "stable");
        assert!(again.setup_eq(&s));
        if let Some(path) = file {
            eprintln!(
                "checked {checked} values in {}",
                std::path::Path::new(&path).display()
            );
        }
    }

    /// A whole setup leaves out the menu key and Color Grading's saved looks, and loading one
    /// keeps them.
    #[test]
    fn a_setup_is_every_mod_but_not_the_menu_key_or_saved_looks() {
        let mut s = ModSettings::new();
        s.set_menu_key(88);
        let grading = s.index_of("color_grading").unwrap();
        assert!(s.save_profile_at(grading, "Warm").is_some());
        s.set_enabled("fps", true);
        s.set_hud_style_value(presets::TEXT_SHADOW_ID, OptionValue::Bool(false));
        let setup = s.setup_json();
        assert!(setup.get("menu").is_none());
        assert!(setup["mods"]["color_grading"].get("profiles").is_none());
        assert_eq!(setup["mods"]["fps"]["enabled"], true);
        assert_eq!(setup[ROOT_KEY][presets::TEXT_SHADOW_ID], false);
        let mut other = ModSettings::new();
        other.set_menu_key(30);
        assert!(other.save_profile_at(grading, "Cold").is_some());
        let (parsed, _) = ModSettings::from_json_str(&setup.to_string());
        assert!(other.apply_setup(&parsed));
        assert!(other.setup_eq(&s));
        assert_eq!(other.menu_key(), 30, "kept");
        assert_eq!(other.profiles_at(grading).len(), 1, "kept");
        assert_eq!(other.profiles_at(grading)[0].name(), "Cold");
        assert!(!other.bool("fps", presets::TEXT_SHADOW_ID));
    }
}
