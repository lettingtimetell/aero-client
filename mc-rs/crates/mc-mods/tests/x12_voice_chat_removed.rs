//! The Voice Chat mod is gone. A settings file saved while
//! it existed must still load: no warning, every other mod's values intact, and the old
//! `voice_chat` entry kept verbatim (the store writes unknown mods back, so an older build that
//! still has the mod finds its settings again).
//!
//! `X12_USER_MODS_JSON=<path>` also checks a real file (a copy, never the user's own): every
//! value in it must come back out of `to_json()` unchanged.

use mc_mods::ModSettings;
use serde_json::Value;

/// The `voice_chat` entry exactly as the last build with the mod saved it (the user's file).
const VOICE_CHAT_ENTRY: &str = r#"{"enabled": false, "hud": {"indicator": {"anchor": "bottom_left",
    "scale": 1.0, "x": 2.0, "y": -2.0}}, "options": {"activation": "Push to Talk",
    "input_device": "", "key_ptt": 47, "key_whisper": 0, "mic_gain": 100.0, "muted": false,
    "output_device": "", "show_indicator": true, "vad_threshold": -50.0,
    "voice_volume": 100.0}}"#;

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
    assert!(settings.index_of("voice_chat").is_none(), "the mod is gone");
    let output = settings.to_json();
    let mut problems = Vec::new();
    assert_contains(&input, &output, "$", &mut problems);
    assert!(problems.is_empty(), "values changed on load: {problems:#?}");
    // The voice_chat entry is written back exactly as it was read.
    assert_eq!(output["mods"]["voice_chat"], input["mods"]["voice_chat"]);
    // And a second save/load round trip is stable.
    let (again, report) = ModSettings::from_json_str(&settings.to_json_string());
    assert!(report.warnings.is_empty(), "{report:?}");
    assert_eq!(again.to_json(), output, "round trip changed the file");
    (settings, input)
}

#[test]
fn a_file_with_the_removed_voice_chat_mod_loads_everything_else() {
    let text = format!(
        r#"{{
            "format": 1,
            "mods": {{
                "fps": {{ "enabled": true }},
                "fullbright": {{ "enabled": true, "options": {{ "brightness": 1500.0 }} }},
                "zoom": {{ "enabled": true, "options": {{ "key": 47 }} }},
                "voice_chat": {VOICE_CHAT_ENTRY}
            }}
        }}"#
    );
    let (settings, _) = check_file(&text);
    assert!(settings.enabled("fps"));
    assert!(settings.enabled("fullbright"));
    assert!(settings.enabled("zoom"));
    assert_eq!(
        settings.key("zoom", "key"),
        47,
        "an option next to voice_chat was read"
    );
    assert!(
        !settings.enabled("voice_chat"),
        "an unknown id reads as off"
    );
}

#[test]
fn a_real_settings_file_copy_loads_without_voice_chat() {
    let Some(path) = std::env::var_os("X12_USER_MODS_JSON") else {
        eprintln!("SKIP: X12_USER_MODS_JSON unset");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("readable copy");
    let (settings, input) = check_file(&text);
    let mods = input["mods"].as_object().expect("mods object");
    let mut enabled = 0;
    let mut values = 0;
    for (id, entry) in mods {
        if id == "voice_chat" {
            continue;
        }
        assert!(
            settings.index_of(id).is_some(),
            "{id} is still in the catalog"
        );
        if entry["enabled"].as_bool() == Some(true) {
            assert!(settings.enabled(id), "{id} still on");
            enabled += 1;
        }
        values += entry
            .get("options")
            .and_then(Value::as_object)
            .map_or(0, serde_json::Map::len);
        values += entry
            .get("hud")
            .and_then(Value::as_object)
            .map_or(0, serde_json::Map::len);
    }
    eprintln!(
        "loaded {path:?}: {} mods in the file ({} besides voice_chat on), {values} option and HUD \
         values intact, voice_chat kept verbatim",
        mods.len(),
        enabled
    );
}
