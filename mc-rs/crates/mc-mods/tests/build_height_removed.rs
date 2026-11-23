//! The Build Height mod is gone. A settings file (or a setup profile) saved while it existed must
//! still load: no warning, nothing corrupt, every other mod's values intact, the old
//! `build_height` entry and a pin of it kept verbatim (the store writes unknown mods back, so an
//! older build that still has the mod finds its settings again), and nothing of it in the
//! catalog, the menu or its search.
//!
//! `BUILD_HEIGHT_MODS_JSON=<path>` also checks a real file (a copy, never the user's own): every
//! value in it must come back out of `to_json()` unchanged.

use mc_mods::{catalog, menu_layout, menu_rules, ModSettings};
use serde_json::Value;

const ID: &str = "build_height";

/// The `build_height` entry as the last build with the mod saved it, every option set away from
/// its default.
const BUILD_HEIGHT_ENTRY: &str = r#"{"enabled": true, "options": {"darkness": 75.0,
    "height": 110.0, "manual_height": true}}"#;

/// Every key under `input` is in `output` with an equal value (numbers to f32 precision: the
/// store keeps sliders as `f32`). `path` names the spot for the failure message.
fn assert_contains(input: &Value, output: &Value, path: &str, problems: &mut Vec<String>) {
    match (input, output) {
        (Value::Object(want), Value::Object(have)) => {
            for (key, value) in want {
                let at = format!("{path}.{key}");
                match have.get(key) {
                    Some(found) => assert_contains(value, found, &at, problems),
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

fn check_file(text: &str) -> (ModSettings, Value) {
    let input: Value = serde_json::from_str(text).expect("the test file is JSON");
    let (settings, report) = ModSettings::from_json_str(text);
    assert!(!report.corrupt && !report.file_missing, "{report:?}");
    assert!(
        report.warnings.is_empty(),
        "load warnings: {:?}",
        report.warnings
    );
    assert!(settings.index_of(ID).is_none(), "the mod is gone");
    assert!(!settings.enabled(ID), "an unknown id reads as off");
    let output = settings.to_json();
    let mut problems = Vec::new();
    assert_contains(&input, &output, "$", &mut problems);
    assert!(problems.is_empty(), "values changed on load: {problems:#?}");
    // The build_height entry is written back exactly as it was read.
    assert_eq!(output["mods"][ID], input["mods"][ID]);
    // And a second save/load round trip is stable.
    let (again, report) = ModSettings::from_json_str(&settings.to_json_string());
    assert!(report.warnings.is_empty(), "{report:?}");
    assert_eq!(again.to_json(), output, "round trip changed the file");
    (settings, input)
}

#[test]
fn a_file_with_the_removed_build_height_mod_loads_everything_else() {
    let text = format!(
        r#"{{
            "format": 1,
            "pinned": ["zoom", "build_height"],
            "mods": {{
                "fps": {{ "enabled": true }},
                "block_overlay": {{ "enabled": true }},
                "zoom": {{ "enabled": true, "options": {{ "key": 47 }} }},
                "build_height": {BUILD_HEIGHT_ENTRY},
                "tall_grass": {{ "enabled": true }}
            }}
        }}"#
    );
    let (settings, _) = check_file(&text);
    for id in ["fps", "block_overlay", "zoom", "tall_grass"] {
        assert!(settings.enabled(id), "{id} still on");
    }
    assert_eq!(
        settings.key("zoom", "key"),
        47,
        "an option next to build_height was read"
    );
    // The pin is kept for the round trip but names no mod: My setup shows only zoom.
    assert_eq!(settings.pinned_ids(), ["zoom", ID]);
    let pinned: Vec<_> = (0..settings.catalog().len())
        .filter(|&index| settings.pinned_at(index))
        .map(|index| settings.catalog()[index].id)
        .collect();
    assert_eq!(pinned, ["zoom"]);
}

/// The same entry in a setup profile (the settings file's format): loading the profile applies
/// every known mod and leaves the unknown one alone.
#[test]
fn a_profile_with_the_removed_mod_applies_cleanly() {
    let profile = format!(
        r#"{{"profile": {{"name": "Old setup", "saved": 1762990000}}, "format": 1,
            "mods": {{"fps": {{"enabled": true}}, "build_height": {BUILD_HEIGHT_ENTRY}}}}}"#
    );
    let (setup, report) = ModSettings::from_json_str(&profile);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    let mut store = ModSettings::new();
    assert!(!store.enabled("fps"));
    assert!(store.apply_setup(&setup), "the profile changes something");
    assert!(store.enabled("fps"));
    assert!(store.index_of(ID).is_none());
    assert!(store.setup_eq(&setup), "applied in full");
}

/// Nothing of the mod is left in the catalog, the menu's placement, summaries, rules or search.
#[test]
fn the_catalog_and_menu_no_longer_know_it() {
    assert!(catalog::find(ID).is_none());
    assert!(catalog::all().iter().all(|descriptor| descriptor.id != ID));
    assert!(menu_layout::meta(ID).is_none());
    assert!(menu_layout::home_category(ID).is_none());
    assert!(menu_rules::RULES.iter().all(|rule| rule.mod_id != ID));
    assert!(menu_rules::GROUPS.iter().all(|group| group.mod_id != ID));
    let words = ["build limit", "height limit", "bed wars", "bedwars"];
    for (mod_id, _, keywords) in menu_layout::OPTION_KEYWORDS {
        assert_ne!(*mod_id, ID);
        for word in words {
            assert!(!keywords.contains(&word), "{mod_id}: {word}");
        }
    }
    for meta in menu_layout::MOD_META {
        for word in words {
            assert!(!meta.keywords.contains(&word), "{}: {word}", meta.id);
        }
    }
}

#[test]
fn a_real_settings_file_copy_loads_without_build_height() {
    let Some(path) = std::env::var_os("BUILD_HEIGHT_MODS_JSON") else {
        eprintln!("SKIP: BUILD_HEIGHT_MODS_JSON unset");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("readable copy");
    let (settings, input) = check_file(&text);
    let mods = input["mods"].as_object().expect("mods object");
    let mut enabled = 0;
    let mut values = 0;
    for (id, entry) in mods {
        if id == ID || settings.index_of(id).is_none() {
            continue;
        }
        if entry["enabled"].as_bool() == Some(true) {
            assert!(settings.enabled(id), "{id} still on");
            enabled += 1;
        }
        values += entry
            .get("options")
            .and_then(Value::as_object)
            .map_or(0, serde_json::Map::len);
    }
    eprintln!(
        "loaded the copy: {} mods in the file ({enabled} known ones on), {values} option values \
         intact, build_height kept verbatim: {}",
        mods.len(),
        mods.contains_key(ID)
    );
}
