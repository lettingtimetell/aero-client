//! A settings file written before the Fog and Vignette mods
//! existed loads both of them **on**, which is vanilla, with "Clear water and lava" off. So a
//! user who never opens the menu sees exactly what they saw before.
//!
//! `W4_USER_MODS_JSON=<path>` also checks a real file (a copy, never the user's own), e.g. the
//! scratch copy of the user's `mc-rs-mods.json`.

use mc_mods::mods::fog;
use mc_mods::ModSettings;

fn assert_vanilla(settings: &ModSettings) {
    assert!(settings.enabled("fog"), "Fog loads on (vanilla)");
    assert!(settings.enabled("vignette"), "Vignette loads on (vanilla)");
    assert!(
        !settings.bool("fog", fog::CLEAR_LIQUIDS),
        "liquids keep their fog"
    );
}

#[test]
fn a_file_from_before_the_mods_existed_loads_them_vanilla() {
    let text = r#"{
        "version": 1,
        "mods": {
            "fps": { "enabled": true, "options": {} },
            "fullbright": { "enabled": true, "options": { "brightness": 1500.0 } }
        }
    }"#;
    let (settings, report) = ModSettings::from_json_str(text);
    assert!(!report.corrupt, "{report:?}");
    assert!(settings.enabled("fullbright"), "the file was read");
    assert_vanilla(&settings);
}

#[test]
fn the_settings_round_trip_through_the_file() {
    let mut settings = ModSettings::new();
    settings.set_enabled("fog", false);
    settings.set_bool("fog", fog::CLEAR_LIQUIDS, true);
    settings.set_enabled("vignette", false);
    let (loaded, report) = ModSettings::from_json_str(&settings.to_json_string());
    assert!(!report.corrupt, "{report:?}");
    assert!(!loaded.enabled("fog"));
    assert!(loaded.bool("fog", fog::CLEAR_LIQUIDS));
    assert!(!loaded.enabled("vignette"));
}

#[test]
fn a_real_settings_file_copy_loads_them_vanilla() {
    let Some(path) = std::env::var_os("W4_USER_MODS_JSON") else {
        eprintln!("SKIP: W4_USER_MODS_JSON unset");
        return;
    };
    let (settings, report) = ModSettings::load(std::path::Path::new(&path));
    assert!(!report.corrupt && !report.file_missing, "{report:?}");
    eprintln!(
        "loaded {:?}: fullbright {}, fps {}",
        path,
        settings.enabled("fullbright"),
        settings.enabled("fps")
    );
    assert_vanilla(&settings);
}
