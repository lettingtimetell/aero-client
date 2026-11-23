//! The settings side of the bug list.
//!
//! * **B4.** Nametags can be made smaller and lighter than vanilla, never larger or darker:
//!   Scale tops out at 1.00x and Background darkness at vanilla's 25%. A file saved by an older
//!   build with bigger values clamps on load (the descriptor's `sanitize`), with a warning, and
//!   saves back clamped.
//! * **B1.** Chat's "Keep chat between servers" (on) and "History length" (100 to 5000 lines,
//!   default 1000).
//! * **B3.** Spotify's `opacity` keeps its settings key; it is now "Background opacity".
//! * **B6.** `long_chat` is gone (the full load check is in `settings.rs`'s unit tests).
//!
//! `TW_B_MODS_JSON=<path>` also runs the B4 check on a real settings file: a **copy** (never the
//! user's own file), into which the test writes oversized nametag values next to the copy
//! (`<path>.b4.json`); every other value in the copy must load unchanged.

use mc_mods::{catalog, ModSettings, OptionKind, OptionValue};
use serde_json::Value;

fn slider_range(mod_id: &str, option_id: &str) -> (f32, f32) {
    let option = catalog::find(mod_id)
        .and_then(|descriptor| descriptor.option(option_id))
        .unwrap_or_else(|| panic!("{mod_id}.{option_id} is in the catalog"));
    match option.kind {
        OptionKind::Slider { min, max, .. } => (min, max),
        other => panic!("{mod_id}.{option_id} is a slider, not {other:?}"),
    }
}

#[test]
fn nametag_sliders_stop_at_vanilla_size_and_darkness() {
    assert_eq!(slider_range("nametags", "scale"), (0.5, 1.0));
    assert_eq!(slider_range("nametags", "background"), (0.0, 25.0));
    // The defaults are vanilla's, and they are the maximums.
    let settings = ModSettings::new();
    assert_eq!(settings.f32("nametags", "scale"), 1.0);
    assert_eq!(settings.f32("nametags", "background"), 25.0);
}

#[test]
fn oversized_nametags_in_an_old_file_clamp_on_load_and_save_back_clamped() {
    let old = r#"{"mods":{"nametags":{"enabled":true,"options":{"scale":3.0,"background":100.0,
        "text_shadow":false,"own_nametag":false}}}}"#;
    let (settings, report) = ModSettings::from_json_str(old);
    assert_eq!(settings.f32("nametags", "scale"), 1.0);
    assert_eq!(settings.f32("nametags", "background"), 25.0);
    assert!(settings.enabled("nametags"));
    assert!(
        !settings.bool("nametags", "text_shadow"),
        "the rest loads as saved"
    );
    assert!(!settings.bool("nametags", "own_nametag"));
    for option in ["scale", "background"] {
        let note = format!("nametags.{option}: out of range, adjusted");
        assert!(report.warnings.contains(&note), "{:?}", report.warnings);
    }
    assert_eq!(report.warnings.len(), 2, "{:?}", report.warnings);
    let saved = settings.to_json();
    assert_eq!(saved["mods"]["nametags"]["options"]["scale"], 1.0);
    assert_eq!(saved["mods"]["nametags"]["options"]["background"], 25.0);
    let (again, report) = ModSettings::from_json_str(&settings.to_json_string());
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    assert_eq!(again.to_json(), saved);

    // Values inside the range are untouched, and so is everything in between.
    let (smaller, report) = ModSettings::from_json_str(
        r#"{"mods":{"nametags":{"options":{"scale":0.75,"background":10.0}}}}"#,
    );
    assert!(report.warnings.is_empty());
    assert_eq!(smaller.f32("nametags", "scale"), 0.75);
    assert_eq!(smaller.f32("nametags", "background"), 10.0);
    // An edit past the end clamps too (the menu's slider cannot get there, a profile might).
    let mut edited = ModSettings::new();
    edited.set_value("nametags", "scale", OptionValue::Float(2.5));
    edited.set_value("nametags", "background", OptionValue::Float(80.0));
    assert_eq!(edited.f32("nametags", "scale"), 1.0);
    assert_eq!(edited.f32("nametags", "background"), 25.0);
}

/// Every key under `want` is in `have` with an equal value (numbers to f32 precision: the store
/// keeps sliders as `f32`).
fn contains(want: &Value, have: &Value, path: &str, problems: &mut Vec<String>) {
    match (want, have) {
        (Value::Object(want), Value::Object(have)) => {
            for (key, value) in want {
                let at = format!("{path}.{key}");
                match have.get(key) {
                    Some(found) => contains(value, found, &at, problems),
                    None => problems.push(format!("{at}: missing after load")),
                }
            }
        }
        (Value::Number(want), Value::Number(have)) => {
            let (want, have) = (want.as_f64().unwrap(), have.as_f64().unwrap());
            if (want - have).abs() > 1e-5 * want.abs().max(1.0) {
                problems.push(format!("{path}: {want} became {have}"));
            }
        }
        (want, have) if want == have => {}
        (want, have) => problems.push(format!("{path}: {want} became {have}")),
    }
}

#[test]
fn a_real_settings_file_copy_with_oversized_nametags_clamps_on_load() {
    let Ok(copy) = std::env::var("TW_B_MODS_JSON") else {
        eprintln!("SKIP: set TW_B_MODS_JSON to a copy of a real mc-rs-mods.json");
        return;
    };
    let copy = std::path::PathBuf::from(copy);
    let text = std::fs::read_to_string(&copy).expect("read the copy");
    let original: Value = serde_json::from_str(&text).expect("the copy is JSON");
    let mut oversized = original.clone();
    oversized["mods"]["nametags"] = serde_json::json!({
        "enabled": true,
        "options": {"scale": 3.0, "background": 100.0}
    });
    let target = copy.with_extension("b4.json");
    std::fs::write(&target, serde_json::to_string_pretty(&oversized).unwrap()).expect("write");

    let (settings, report) = ModSettings::load(&target);
    assert!(!report.corrupt && !report.file_missing);
    assert_eq!(settings.f32("nametags", "scale"), 1.0);
    assert_eq!(settings.f32("nametags", "background"), 25.0);
    let nametag_warnings: Vec<_> = report
        .warnings
        .iter()
        .filter(|warning| warning.starts_with("nametags."))
        .collect();
    assert_eq!(nametag_warnings.len(), 2, "{:?}", report.warnings);

    // Everything else the copy holds comes back out unchanged (minus keys this wave removed).
    let mut want = original.clone();
    if let Some(chat) = want["mods"]["chat"]["options"].as_object_mut() {
        chat.remove("long_chat");
    }
    let mut problems = Vec::new();
    contains(
        &want["mods"],
        &settings.to_json()["mods"],
        "mods",
        &mut problems,
    );
    assert!(problems.is_empty(), "{problems:#?}");

    settings.save(&target).expect("save");
    let (reloaded, report) = ModSettings::load(&target);
    assert!(
        report.warnings.iter().all(|w| !w.starts_with("nametags.")),
        "{:?}",
        report.warnings
    );
    assert_eq!(reloaded.f32("nametags", "scale"), 1.0);
    assert_eq!(reloaded.f32("nametags", "background"), 25.0);
    eprintln!(
        "B4: {} loaded with {} warnings ({} about nametags), saved back clamped",
        target.display(),
        report.warnings.len(),
        nametag_warnings.len()
    );
}

#[test]
fn chat_history_options_have_their_ranges_and_defaults() {
    assert_eq!(slider_range("chat", "history_length"), (100.0, 5000.0));
    let settings = ModSettings::new();
    assert!(
        settings.bool("chat", "keep_chat"),
        "kept between servers by default"
    );
    assert_eq!(settings.f32("chat", "history_length"), 1000.0);
    assert!(
        !settings.enabled("chat"),
        "the Chat mod itself stays off by default"
    );
    let (settings, report) = ModSettings::from_json_str(
        r#"{"mods":{"chat":{"options":{"history_length":99999,"keep_chat":false}}}}"#,
    );
    assert_eq!(settings.f32("chat", "history_length"), 5000.0);
    assert!(!settings.bool("chat", "keep_chat"));
    assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    let (settings, _) =
        ModSettings::from_json_str(r#"{"mods":{"chat":{"options":{"history_length":10}}}}"#);
    assert_eq!(
        settings.f32("chat", "history_length"),
        100.0,
        "vanilla is the floor"
    );
    assert!(catalog::find("chat").unwrap().option("long_chat").is_none());
}

#[test]
fn spotify_background_opacity_keeps_the_old_settings_key() {
    let option = catalog::find("spotify")
        .and_then(|descriptor| descriptor.option("opacity"))
        .expect("the key is unchanged");
    assert_eq!(option.label, "Background opacity");
    let (settings, report) =
        ModSettings::from_json_str(r#"{"mods":{"spotify":{"options":{"opacity":40.0}}}}"#);
    assert!(report.warnings.is_empty());
    assert_eq!(settings.f32("spotify", "opacity"), 40.0);
}
