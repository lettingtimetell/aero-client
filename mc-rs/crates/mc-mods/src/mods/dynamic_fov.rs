//! Dynamic FOV (id `dynamic_fov`): vanilla's field-of-view changes from sprinting, Speed,
//! Slowness, flying, drawing a bow, being underwater and dying, with a slider per source, a
//! strength slider over the movement ones and a bow switch. Off keeps the FOV still while
//! moving.
//!
//! This mod began as "No Dynamic FOV" (ON meant the FOV stays put) and was renamed to "Dynamic
//! FOV", on by default: the row now names the vanilla feature, so ON, the default, is vanilla
//! (like Fog, Vignette and Tall Grass), and OFF is no change at all, the bow's included. Keep the
//! `id`
//! stable (it is the settings-file key); the `category` line must name the mod's home in
//! `crate::menu_layout`, which decides where the menu lists it (a test keeps the two in step).
//!
//! * `strength`: the share of vanilla's movement change (sprint, Speed, Slowness, flying) that
//!   is kept, 0 to 100 %. Same key and meaning as before the rename; the default moved from 0
//!   to 100.
//! * `bow_zoom`: the bow's narrowing, in full (vanilla) or not at all, whatever the strength.
//!   It replaces `keep_bow`, whose "off" scaled the bow by the strength and so did nothing at
//!   100 %.
//! * One slider per vanilla source, 0 to 100 %, 100 % being
//!   vanilla: `sprinting`, `speed_effect`, `slowness_effect` and `flying` (the movement part
//!   of `AbstractClientPlayer.getFovModifier`), `underwater` and `dying` (the two squeezes of
//!   `EntityRenderer.getFOVModifier`). The movement attribute is split by its modifiers: each
//!   slider scales the *amount* of its own modifier (sprint `+0.3`, Speed `+0.2` per level,
//!   Slowness `-0.15` per level, all multiply-total) or of flying's `x1.1`, and the attribute is
//!   resolved again in Java's order, so 100 % reproduces vanilla bit for bit and two sources
//!   combine as they do in vanilla. `strength` then scales the whole movement part, as before.
//!   Off ignores every slider: the movement and the bow are flattened and the underwater and
//!   dying squeezes stay vanilla (what off did before the sliders existed).
//!
//! Client half: `mc-client/src/app/mods/m3_controls.rs` scales the per-tick target of
//! `EntityRenderer.updateFovModifierHand`, so vanilla's easing still smooths every change, and
//! `mc-client/src/app/mods/nw_dynamic_fov.rs` adds the per-source sliders. At the defaults it
//! keeps no state and the target is vanilla's, bit for bit.
//!
//! # Settings saved before the rename
//!
//! [`migrate_legacy`] rewrites an entry saved by an older build into today's meaning when the
//! file loads (`ModSettings::apply_json` calls it), so nobody's FOV changes behind their back.

use serde_json::{Map, Number, Value};

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The bow switch. Its presence marks an entry saved after the rename.
const BOW_ZOOM: &str = "bow_zoom";
/// The bow switch of the "No Dynamic FOV" builds: on kept the bow's narrowing in full.
const LEGACY_KEEP_BOW: &str = "keep_bow";

/// The per-source sliders. Never rename these ids: they are the keys
/// the settings file stores.
pub const SPRINTING: &str = "sprinting";
/// See [`SPRINTING`].
pub const SPEED_EFFECT: &str = "speed_effect";
/// See [`SPRINTING`].
pub const SLOWNESS_EFFECT: &str = "slowness_effect";
/// See [`SPRINTING`].
pub const FLYING: &str = "flying";
/// See [`SPRINTING`].
pub const UNDERWATER: &str = "underwater";
/// See [`SPRINTING`].
pub const DYING: &str = "dying";

/// A 0 to 100 % slider whose 100 % is vanilla.
const fn share(
    id: &'static str,
    label: &'static str,
    description: &'static str,
) -> OptionDescriptor {
    OptionDescriptor {
        id,
        label,
        kind: OptionKind::Slider {
            min: 0.0,
            max: 100.0,
            step: 5.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description,
    }
}

static OPTIONS: [OptionDescriptor; 8] = [
    share(
        "strength",
        "Strength",
        "Scales every movement change below at once: sprinting, Speed, Slowness and flying. \
         100% is vanilla; 0% keeps the FOV still while moving.",
    ),
    share(
        SPRINTING,
        "Sprinting",
        "How much sprinting widens the FOV. 100% is vanilla; 0% leaves it as when walking.",
    ),
    share(
        SPEED_EFFECT,
        "Speed effect",
        "How much the Speed effect widens the FOV. 100% is vanilla; 0% ignores the effect.",
    ),
    share(
        SLOWNESS_EFFECT,
        "Slowness effect",
        "How much the Slowness effect narrows the FOV. 100% is vanilla; 0% ignores the effect.",
    ),
    share(
        FLYING,
        "Flying",
        "How much flying widens the FOV. 100% is vanilla; 0% leaves it as on the ground.",
    ),
    OptionDescriptor {
        id: BOW_ZOOM,
        label: "Bow zoom",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Drawing a bow narrows the view, as in vanilla.",
    },
    share(
        UNDERWATER,
        "Underwater",
        "How much being underwater narrows the view. 100% is vanilla; 0% keeps it as on land.",
    ),
    share(
        DYING,
        "Dying",
        "How much the view zooms in while you die. 100% is vanilla; 0% keeps it still.",
    ),
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "dynamic_fov",
    name: "Dynamic FOV",
    category: Category::Controls,
    description: "Vanilla's FOV changes when you sprint, fly, get Speed or Slowness, draw a bow, \
                  go underwater or die, each with its own slider. Turn it off to keep the FOV \
                  still while moving.",
    default_enabled: true,
    options: &OPTIONS,
    hud_elements: &[],
};

/// Rewrite a settings-file entry saved before the rename into today's meaning. `None` when the
/// entry is already current; otherwise the rewritten entry and a line for the load report.
///
/// Every build writes every option, so an entry without `bow_zoom` comes from before the
/// rename. There `enabled` meant "No Dynamic FOV" (default off), `strength` the share of the
/// change kept (default 0 %) and `keep_bow` whether the bow still narrowed in full (default
/// off; otherwise the bow followed the strength):
///
/// | before (No Dynamic FOV) | after (Dynamic FOV) |
/// |---|---|
/// | off | on, 100 %, bow zoom on: vanilla, as before |
/// | on, strength s > 0 | on, s, bow zoom = `keep_bow` (on at s = 100 %, where the bow was full) |
/// | on, 0 %, `keep_bow` on | on, 0 %, bow zoom on: only the bow moved the FOV |
/// | on, 0 %, `keep_bow` off | off: nothing moved the FOV (the options go back to their defaults) |
///
/// `keep_bow` is dropped in every case, so it is not carried along as an unknown key.
#[must_use]
pub fn migrate_legacy(entry: &Map<String, Value>) -> Option<(Map<String, Value>, String)> {
    let options = entry.get("options").and_then(Value::as_object);
    if options.is_some_and(|options| options.contains_key(BOW_ZOOM)) {
        return None;
    }
    let option = |id: &str| options.and_then(|options| options.get(id));
    // The old meaning, with the old defaults for anything missing or unreadable.
    let no_dynamic_fov = entry
        .get("enabled")
        .and_then(crate::settings::json_bool)
        .unwrap_or(false);
    let strength = option("strength")
        .and_then(crate::settings::json_f64)
        .filter(|s| s.is_finite())
        .map_or(0.0, |s| s.clamp(0.0, 100.0));
    let keep_bow = option(LEGACY_KEEP_BOW)
        .and_then(crate::settings::json_bool)
        .unwrap_or(false);
    let (on, strength, bow_zoom) = if !no_dynamic_fov {
        (true, 100.0, true)
    } else if strength > 0.0 {
        (true, strength, keep_bow || strength >= 100.0)
    } else if keep_bow {
        (true, 0.0, true)
    } else {
        (false, 100.0, true)
    };
    let mut rewritten = entry.clone();
    rewritten.insert("enabled".into(), Value::Bool(on));
    let mut new_options = options.cloned().unwrap_or_default();
    new_options.remove(LEGACY_KEEP_BOW);
    new_options.insert(
        "strength".into(),
        Number::from_f64(strength).map_or(Value::Null, Value::Number),
    );
    new_options.insert(BOW_ZOOM.into(), Value::Bool(bow_zoom));
    rewritten.insert("options".into(), Value::Object(new_options));
    let switch = |on: bool| if on { "on" } else { "off" };
    let now = if on {
        format!("on at {strength} %, bow zoom {}", switch(bow_zoom))
    } else {
        "off".to_owned()
    };
    let note = format!(
        "dynamic_fov: an older build saved \"No Dynamic FOV\" {} ({} %, keep bow zoom {}); \
         read as Dynamic FOV {now}",
        switch(no_dynamic_fov),
        option("strength")
            .and_then(crate::settings::json_f64)
            .unwrap_or(0.0),
        switch(keep_bow),
    );
    Some((rewritten, note))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(json: &str) -> Map<String, Value> {
        match serde_json::from_str::<Value>(json) {
            Ok(Value::Object(map)) => map,
            other => panic!("not an object: {other:?}"),
        }
    }

    /// `(enabled, strength, bow_zoom)` of a migrated entry, and that `keep_bow` is gone.
    fn migrated(json: &str) -> (bool, f64, bool) {
        let (new, note) = migrate_legacy(&entry(json)).expect("an old entry migrates");
        assert!(note.starts_with("dynamic_fov: "), "{note}");
        let options = new["options"].as_object().expect("options");
        assert!(!options.contains_key(LEGACY_KEEP_BOW), "{new:?}");
        assert!(
            migrate_legacy(&new).is_none(),
            "a migrated entry is current"
        );
        (
            new["enabled"].as_bool().expect("enabled"),
            options["strength"].as_f64().expect("strength"),
            options[BOW_ZOOM].as_bool().expect("bow_zoom"),
        )
    }

    #[test]
    fn the_row_names_the_vanilla_feature_and_defaults_to_it() {
        assert_eq!(DESCRIPTOR.name, "Dynamic FOV");
        assert!(DESCRIPTOR.default_enabled, "on is vanilla");
        assert_eq!(
            DESCRIPTOR.id, "dynamic_fov",
            "the settings key never changes"
        );
        assert_eq!(OPTIONS[0].label, "Strength");
        assert_eq!(
            OPTIONS[0].default,
            OptionValue::Float(100.0),
            "100 % is vanilla"
        );
        let bow = OPTIONS
            .iter()
            .find(|option| option.id == BOW_ZOOM)
            .expect("the bow switch");
        assert_eq!(
            bow.default,
            OptionValue::Bool(true),
            "the bow zooms in vanilla"
        );
        assert!(!DESCRIPTOR.name.starts_with("No "));
    }

    /// One slider per vanilla source, each a 0 to 100 % share that
    /// defaults to 100 % (vanilla), with ids that never collide with the old ones.
    #[test]
    fn every_vanilla_source_has_a_slider_that_defaults_to_vanilla() {
        let ids: Vec<&str> = OPTIONS.iter().map(|option| option.id).collect();
        assert_eq!(
            ids,
            [
                "strength",
                SPRINTING,
                SPEED_EFFECT,
                SLOWNESS_EFFECT,
                FLYING,
                BOW_ZOOM,
                UNDERWATER,
                DYING
            ]
        );
        assert!(
            !ids.contains(&LEGACY_KEEP_BOW),
            "a retired id is never reused"
        );
        for option in OPTIONS.iter().filter(|option| option.id != BOW_ZOOM) {
            assert_eq!(
                option.kind,
                OptionKind::Slider {
                    min: 0.0,
                    max: 100.0,
                    step: 5.0,
                    unit: "%",
                    decimals: 0,
                },
                "{}",
                option.id
            );
            assert_eq!(option.default, OptionValue::Float(100.0), "{}", option.id);
            assert!(
                option.description.contains("100% is vanilla"),
                "{}",
                option.id
            );
        }
    }

    #[test]
    fn old_entries_keep_their_behaviour() {
        // The user's own file on 2026-11-07: the FOV never moved, so Dynamic FOV is off.
        assert_eq!(
            migrated(r#"{"enabled":true,"options":{"keep_bow":false,"strength":0.0}}"#),
            (false, 100.0, true)
        );
        // Old off was vanilla: new on at the vanilla defaults.
        assert_eq!(
            migrated(r#"{"enabled":false,"options":{"keep_bow":false,"strength":0.0}}"#),
            (true, 100.0, true)
        );
        assert_eq!(
            migrated(r#"{"enabled":false,"options":{"keep_bow":true,"strength":35.0}}"#),
            (true, 100.0, true)
        );
        // Old on with some strength: the strength carries over, the bow switch too.
        assert_eq!(
            migrated(r#"{"enabled":true,"options":{"keep_bow":true,"strength":40.0}}"#),
            (true, 40.0, true)
        );
        assert_eq!(
            migrated(r#"{"enabled":true,"options":{"keep_bow":false,"strength":40.0}}"#),
            (true, 40.0, false)
        );
        // At 100 % the old bow followed the strength in full: vanilla.
        assert_eq!(
            migrated(r#"{"enabled":true,"options":{"keep_bow":false,"strength":100.0}}"#),
            (true, 100.0, true)
        );
        // Static FOV with the bow kept: only the bow moves the FOV.
        assert_eq!(
            migrated(r#"{"enabled":true,"options":{"keep_bow":true,"strength":0.0}}"#),
            (true, 0.0, true)
        );
        // Missing pieces read as the old defaults (on, 0 %, no bow: off).
        assert_eq!(migrated(r#"{"enabled":true}"#), (false, 100.0, true));
        assert_eq!(migrated(r#"{}"#), (true, 100.0, true));
    }

    #[test]
    fn current_entries_and_unknown_keys_are_left_alone() {
        assert!(migrate_legacy(&entry(
            r#"{"enabled":false,"options":{"strength":100.0,"bow_zoom":true}}"#
        ))
        .is_none());
        let (new, _) = migrate_legacy(&entry(
            r#"{"enabled":true,"options":{"strength":0,"keep_bow":false,"future":7},"x":1}"#,
        ))
        .expect("old");
        assert_eq!(new["options"]["future"], 7, "unknown options survive");
        assert_eq!(new["x"], 1, "unknown entry keys survive");
    }
}
