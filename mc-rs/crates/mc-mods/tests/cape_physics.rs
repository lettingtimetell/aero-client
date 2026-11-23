use mc_mods::{catalog, Category, ModSettings, OptionValue};

const ID: &str = "cape_physics";

#[test]
fn legacy_numeric_cape_choices_keep_enable_state_and_use_new_slider_defaults() {
    for enabled in [false, true] {
        for (index, label) in ["Wavy", "Flowing", "Windy"].into_iter().enumerate() {
            let saved = serde_json::json!({
                "mods": {
                    "cape_physics": {"enabled": enabled, "options": {"motion": index}},
                    "fps": {"enabled": true}
                }
            });
            let (settings, report) = ModSettings::from_json_str(&saved.to_string());
            assert!(report.warnings.is_empty());
            assert_eq!(settings.enabled(ID), enabled);
            assert_eq!(settings.choice(ID, "motion"), index);
            assert_eq!(settings.choice_label(ID, "motion"), label);
            assert_eq!(settings.f32(ID, "strength"), 60.0);
            assert_eq!(settings.f32(ID, "speed"), 100.0);
            assert_eq!(settings.f32(ID, "stiffness"), 50.0);
            assert!(settings.enabled("fps"));
        }
    }
    let settings = ModSettings::new();
    assert!(!settings.enabled(ID));
    assert_eq!(settings.choice_label(ID, "motion"), "Wavy");
    assert_eq!(catalog::find(ID).unwrap().category, Category::Players);
}

#[test]
fn removed_cape_choice_labels_follow_the_normal_unknown_choice_policy() {
    for old_label in ["No sideways swing", "Vertical only", "Still"] {
        let saved = serde_json::json!({
            "mods": {"cape_physics": {"enabled": true, "options": {"motion": old_label}}}
        });
        let (settings, report) = ModSettings::from_json_str(&saved.to_string());
        assert!(settings.enabled(ID));
        assert_eq!(settings.choice_label(ID, "motion"), "Wavy");
        assert!(!report.warnings.is_empty());
    }
}

#[test]
fn cape_wave_styles_and_slider_preferences_round_trip() {
    for style in 0..4 {
        let mut settings = ModSettings::new();
        settings.set_enabled(ID, true);
        settings.set_value(ID, "motion", OptionValue::Choice(style));
        settings.set_value(ID, "strength", OptionValue::Float(85.0));
        settings.set_value(ID, "speed", OptionValue::Float(175.0));
        settings.set_value(ID, "stiffness", OptionValue::Float(20.0));
        let (loaded, report) = ModSettings::from_json_str(&settings.to_json_string());
        assert!(report.warnings.is_empty());
        assert!(loaded.enabled(ID));
        assert_eq!(loaded.choice(ID, "motion"), style);
        assert_eq!(loaded.f32(ID, "strength"), 85.0);
        assert_eq!(loaded.f32(ID, "speed"), 175.0);
        assert_eq!(loaded.f32(ID, "stiffness"), 20.0);
    }
}

#[test]
fn draped_is_appended_without_changing_existing_style_indices() {
    assert_eq!(
        mc_mods::mods::cape_physics::MOTIONS,
        ["Wavy", "Flowing", "Windy", "Draped"]
    );
    let saved = serde_json::json!({
        "mods": {"cape_physics": {"enabled": true, "options": {"motion": "Draped"}}}
    });
    let (settings, report) = ModSettings::from_json_str(&saved.to_string());
    assert!(report.warnings.is_empty());
    assert!(settings.enabled(ID));
    assert_eq!(settings.choice(ID, "motion"), 3);
    assert_eq!(settings.choice_label(ID, "motion"), "Draped");
}

#[test]
fn cape_wave_sliders_clamp_snap_and_reject_invalid_saved_values() {
    for (values, expected) in [
        (
            serde_json::json!({"strength": -10, "speed": 0, "stiffness": -5}),
            [0.0, 25.0, 0.0],
        ),
        (
            serde_json::json!({"strength": 500, "speed": 500, "stiffness": 500}),
            [100.0, 200.0, 100.0],
        ),
        (
            serde_json::json!({"strength": 62, "speed": 103, "stiffness": 48}),
            [60.0, 105.0, 50.0],
        ),
        (
            serde_json::json!({"strength": "invalid", "speed": null, "stiffness": false}),
            [60.0, 100.0, 50.0],
        ),
    ] {
        let saved = serde_json::json!({"mods": {"cape_physics": {"options": values}}});
        let (settings, _) = ModSettings::from_json_str(&saved.to_string());
        for (id, expected) in ["strength", "speed", "stiffness"].into_iter().zip(expected) {
            assert_eq!(settings.f32(ID, id), expected, "{id}: {saved}");
        }
        assert!(!settings.enabled(ID));
    }
}
