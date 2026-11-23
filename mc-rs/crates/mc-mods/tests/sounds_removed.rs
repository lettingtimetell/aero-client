use mc_mods::ModSettings;

#[test]
fn removed_sounds_settings_are_inert_and_preserved_for_older_clients() {
    let saved = r#"{"mods":{"sounds":{"enabled":true,"options":{"volumes":"random.bow=0m"}},"fps":{"enabled":true}}}"#;
    let (settings, report) = ModSettings::from_json_str(saved);
    assert!(report.warnings.is_empty());
    assert!(settings.index_of("sounds").is_none());
    assert!(settings.enabled("fps"));
    let original: serde_json::Value = serde_json::from_str(saved).unwrap();
    assert_eq!(
        settings.to_json()["mods"]["sounds"],
        original["mods"]["sounds"]
    );
}
