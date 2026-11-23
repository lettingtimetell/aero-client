use mc_mods::{catalog, Category, ModSettings, OptionValue};

#[test]
fn auto_gg_stays_opt_in_when_loading_existing_settings() {
    let (settings, report) = ModSettings::from_json_str(
        r#"{"mods":{"chat":{"enabled":true,"options":{"long_chat":false}},"fps":{"enabled":true}}}"#,
    );
    assert!(report.warnings.is_empty());
    assert!(!settings.enabled("auto_gg"));
    assert_eq!(settings.text("auto_gg", "message"), "gg");
    assert_eq!(settings.f32("auto_gg", "delay_ms"), 500.0);
    assert!(settings.enabled("chat"));
    // `long_chat` is no longer an option; the old key just survives.
    assert_eq!(
        settings.to_json()["mods"]["chat"]["options"]["long_chat"],
        false
    );
    assert!(settings.enabled("fps"));

    let descriptor = catalog::find("auto_gg").expect("Auto GG must be available in the menu");
    assert_eq!(descriptor.category, Category::Interface);
    assert_eq!(descriptor.name, "Auto GG");
    assert!(descriptor.hud_elements.is_empty());
    let (loaded, _) = ModSettings::from_json_str(&settings.to_json_string());
    assert!(!loaded.enabled("auto_gg"));
}

#[test]
fn auto_gg_enabled_message_and_delay_preferences_round_trip() {
    for (delay, message) in [
        (0.0, "gg"),
        (500.0, "Good game, everyone!"),
        (1700.0, "/ac gg"),
        (3000.0, ""),
    ] {
        let mut settings = ModSettings::new();
        settings.set_enabled("auto_gg", true);
        settings.set_value("auto_gg", "delay_ms", OptionValue::Float(delay));
        settings.set_text("auto_gg", "message", message);
        let (loaded, report) = ModSettings::from_json_str(&settings.to_json_string());
        assert!(report.warnings.is_empty());
        assert!(loaded.enabled("auto_gg"));
        assert_eq!(loaded.f32("auto_gg", "delay_ms"), delay);
        assert_eq!(loaded.text("auto_gg", "message"), message);
    }
}

#[test]
fn auto_gg_saved_messages_follow_the_chat_macro_character_limit() {
    let message = "é".repeat(300);
    let input = serde_json::json!({
        "mods": {"auto_gg": {"enabled": true, "options": {"message": message}}}
    });
    let (settings, _) = ModSettings::from_json_str(&input.to_string());
    assert_eq!(settings.text("auto_gg", "message"), "é".repeat(256));
    let (settings, _) =
        ModSettings::from_json_str(r#"{"mods":{"auto_gg":{"options":{"message":123}}}}"#);
    assert_eq!(settings.text("auto_gg", "message"), "gg");
}

#[test]
fn auto_gg_saved_delays_are_bounded_and_invalid_values_use_the_default() {
    for (saved, expected) in [
        ("-100", 0.0),
        ("425", 400.0),
        ("575", 600.0),
        ("100000", 3000.0),
        ("\"fast\"", 500.0),
        ("null", 500.0),
    ] {
        let input = format!(
            r#"{{"mods":{{"auto_gg":{{"enabled":true,"options":{{"delay_ms":{saved}}}}}}}}}"#
        );
        let (settings, _) = ModSettings::from_json_str(&input);
        assert!(settings.enabled("auto_gg"));
        assert_eq!(settings.f32("auto_gg", "delay_ms"), expected, "{saved}");
    }
}
