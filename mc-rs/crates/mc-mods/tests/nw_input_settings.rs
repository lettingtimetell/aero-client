//! Modern Keybinds' "Eating stops mining"
//! (`modern_keybinds.eat_stops_mining`) became 1.7 Animations' "Using items
//! stops mining" (`old_animations.use_stops_mining`), which also covers sword blocking and bow
//! drawing.
//!
//! * The old option id is retired: no mod declares it, and a file that has it loads without a
//!   warning, keeping it as an unknown option for the round trip.
//! * A saved "off" carries over to the new option once (only while the new key was never saved),
//!   so a player who turned the packet-timing feature off does not find it back on. A saved "on"
//!   was the default and changes nothing.
//!
//! `NW_INPUT_MODS_JSON=<path>` also loads a real settings file: a **copy** (never the user's own
//! file). It must load with no warnings, and the two options must come out as the rule says.

use mc_mods::{catalog, ModSettings, OptionValue};
use serde_json::Value;

const OLD: (&str, &str) = ("modern_keybinds", "eat_stops_mining");
const NEW: (&str, &str) = ("old_animations", "use_stops_mining");

fn file(eat: Option<bool>, use_stops: Option<bool>) -> String {
    let mut old_options = serde_json::Map::new();
    if let Some(on) = eat {
        old_options.insert(OLD.1.into(), Value::Bool(on));
    }
    let mut new_options = serde_json::Map::new();
    new_options.insert("blockhit".into(), Value::Bool(true));
    if let Some(on) = use_stops {
        new_options.insert(NEW.1.into(), Value::Bool(on));
    }
    serde_json::json!({
        "format": 1,
        "mods": {
            "modern_keybinds": {"enabled": true, "options": old_options},
            "old_animations": {"enabled": true, "options": new_options},
        }
    })
    .to_string()
}

#[test]
fn the_old_id_is_retired_and_the_new_option_is_on_in_1_7_animations() {
    let modern = catalog::find(OLD.0).expect("Modern Keybinds");
    assert!(modern.option(OLD.1).is_none(), "the old option is gone");
    for descriptor in catalog::all() {
        assert!(
            descriptor.option(OLD.1).is_none(),
            "{} reuses the retired id",
            descriptor.id
        );
    }
    let option = catalog::find(NEW.0)
        .and_then(|descriptor| descriptor.option(NEW.1))
        .expect("1.7 Animations' Using items stops mining");
    assert_eq!(option.default, OptionValue::Bool(true));
    // 1.7 Animations itself starts off, so the feature is opt-in.
    let settings = ModSettings::new();
    assert!(!settings.enabled(NEW.0) && settings.bool(NEW.0, NEW.1));
    // The descriptions no longer claim the mod is only visual.
    let animations = catalog::find(NEW.0).expect("1.7 Animations");
    assert!(
        !animations.description.ends_with("Visual only."),
        "{}",
        animations.description
    );
    assert!(
        option.description.contains("packet"),
        "{}",
        option.description
    );
}

#[test]
fn a_saved_off_carries_over_once_and_a_later_choice_sticks() {
    let (settings, report) = ModSettings::from_json_str(&file(Some(false), None));
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    assert_eq!(report.migrated.len(), 1, "{:?}", report.migrated);
    assert!(!settings.bool(NEW.0, NEW.1), "the off carried over");
    assert!(settings.enabled(NEW.0) && settings.bool(NEW.0, "blockhit"));
    // The old key round-trips as an unknown option, and the new one is now saved.
    let saved = settings.to_json();
    assert_eq!(saved["mods"][OLD.0]["options"][OLD.1], Value::Bool(false));
    assert_eq!(saved["mods"][NEW.0]["options"][NEW.1], Value::Bool(false));
    let (mut again, report) = ModSettings::from_json_str(&settings.to_json_string());
    assert!(report.migrated.is_empty() && report.warnings.is_empty());
    assert!(!again.bool(NEW.0, NEW.1));
    // The player turns it back on: the old "off" (still in the file) never undoes that.
    assert!(again.set_bool(NEW.0, NEW.1, true));
    let (third, report) = ModSettings::from_json_str(&again.to_json_string());
    assert!(report.migrated.is_empty(), "{:?}", report.migrated);
    assert!(third.bool(NEW.0, NEW.1));
}

#[test]
fn a_saved_on_or_no_old_key_keeps_the_default_and_a_saved_new_value_wins() {
    for (eat, use_stops, want) in [
        (Some(true), None, true),
        (None, None, true),
        (Some(true), Some(false), false),
        (Some(false), Some(true), true),
    ] {
        let (settings, report) = ModSettings::from_json_str(&file(eat, use_stops));
        assert!(
            report.migrated.is_empty() && report.warnings.is_empty(),
            "{eat:?} {use_stops:?}: {report:?}"
        );
        assert_eq!(settings.bool(NEW.0, NEW.1), want, "{eat:?} {use_stops:?}");
    }
}

/// The same rule on a real settings file (a copy): it loads cleanly and the new option comes out
/// as the old one says.
#[test]
fn a_real_settings_file_copy_loads_with_the_option_where_the_rule_puts_it() {
    let Ok(path) = std::env::var("NW_INPUT_MODS_JSON") else {
        eprintln!("skip: NW_INPUT_MODS_JSON is not set");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("the copy reads");
    let raw: Value = serde_json::from_str(&text).expect("the copy is JSON");
    let old = raw["mods"][OLD.0]["options"][OLD.1].as_bool();
    let new_saved = raw["mods"][NEW.0]["options"][NEW.1].as_bool();
    let (settings, report) = ModSettings::from_json_str(&text);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    let want = new_saved.unwrap_or(old != Some(false));
    assert_eq!(settings.bool(NEW.0, NEW.1), want);
    eprintln!(
        "copy: eat_stops_mining={old:?}, use_stops_mining saved={new_saved:?} -> {}; 1.7 \
         Animations enabled={}; migrated={:?}",
        settings.bool(NEW.0, NEW.1),
        settings.enabled(NEW.0),
        report.migrated
    );
}
