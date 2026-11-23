//! Aero Title Screen (id `aero_title`): the custom title screen, **on by default**. Keep the `id`
//! stable (it is the settings-file key); the `category` line must name the mod's home in
//! `crate::menu_layout`, which decides where the menu lists it (a test keeps the two in step).
//!
//! On: the "Aero Client" title screen. One of three animated scenes behind it ("Menu scene"): a block greenhouse drawn by the real terrain renderer with vanilla
//! textures (`mc_render::menu_scene`), or the earlier Glass and Shore scenes
//! (`mc_render::loading_scene`); the title in Inter, and Multiplayer, Options, Language and Quit
//! Game as Aero glass buttons that pop in like the Shift+Tab menu's windows. While it shows, the
//! frame cap is the monitor's refresh rate instead of vanilla's 30 FPS. The loading screen before
//! it is the logo and a progress bar over a dark backdrop, with no minimum time.
//!
//! Off: vanilla's title screen exactly (`mc_ui::screens::menu::MainMenuScreen`: panorama,
//! splash, every button, the 30 FPS cap).
//!
//! UI half: `mc_ui::screens::aero_title`. Client half: `mc-client/src/app/mods/aero_title.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// "Logo style": how the "Aero Client" logo is drawn on the title
/// and loading screens. The choices are `mc_ui::screens::aero_title::font::LogoStyle`, in its
/// order (a test there checks they match). Italic, the default, is the user's "remove the shadow
/// and make it italics"; Classic is the logo before that.
pub const LOGO_STYLE: OptionDescriptor = OptionDescriptor {
    id: "logo_style",
    label: "Logo style",
    kind: OptionKind::Choice {
        choices: &["Italic", "Light italic", "Gradient", "Underline", "Classic"],
    },
    default: OptionValue::Choice(0),
    description: "How the Aero Client logo looks on the title and loading screens. Classic is \
                  the original upright logo with a drop shadow.",
};

/// "Menu scene": the animated scene behind the title screen.
/// Requested: "loading screens were supposed to be main menu screens, it just increases loading
/// time like this. please make it into 3 randomly selected main menus". The earlier two
/// loading scenes (Glass, Shore) joined the greenhouse as title-screen scenes. The choices are
/// `MenuChoice` in `mc-client/src/app/mods/nw_menu.rs`, in this order; Random picks one of the
/// three each time the game starts.
pub const MENU_SCENE: OptionDescriptor = OptionDescriptor {
    id: "menu_scene",
    label: "Menu scene",
    kind: OptionKind::Choice {
        choices: &["Random", "Greenhouse", "Glass", "Shore"],
    },
    default: OptionValue::Choice(0),
    description: "The animated scene behind the title screen: Greenhouse (a glass greenhouse \
                  built of blocks), Glass (a plant swaying behind frosted, wired glass), Shore \
                  (neon waves on a beach at night), or Random (one of the three, picked each \
                  time the game starts).",
};

/// Option ids this mod no longer has. **Never reuse one**: a settings file saved by an older
/// build may still hold its value, which loads as an unknown key (kept for the round trip,
/// never a warning). `loading_screen` (Random, Glass, Shore, Classic) and `loading_min_time`
/// (default 2 s) were the earlier animated loading screen (2026-11-12); the night
/// wave of 2026-11-14 made its scenes title-screen scenes ([`MENU_SCENE`]) and the
/// loading screen fast again, with no minimum time. `client_brand` ("Tell servers you use Aero
/// Client", a toggle) existed for one build on 2026-11-22: the client now always tells servers
/// its name (`mc-client/src/app/mods/client_brand.rs`), so the option was removed.
pub const RETIRED_OPTION_IDS: &[&str] = &["loading_screen", "loading_min_time", "client_brand"];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "aero_title",
    name: "Aero Title Screen",
    category: Category::Interface,
    description: "The Aero Client title screen: an animated scene (a greenhouse, a plant behind \
                  glass, or a neon night shore), glass buttons and smooth animation at your \
                  monitor's refresh rate, after a quick loading screen. Off shows the vanilla \
                  title and loading screens.",
    default_enabled: true,
    options: &[
        // (the earlier loading-screen options were here; their ids are
        // in `RETIRED_OPTION_IDS`)
        MENU_SCENE,
        LOGO_STYLE,
        ACCOUNTS,
        CHECK_FOR_UPDATES,
    ],
    hud_elements: &[],
};

/// "Accounts": opens the Accounts screen (add, switch or remove
/// Minecraft accounts). The title screen's account corner opens it too; this button keeps it
/// reachable with this mod off (vanilla's title screen) and from a world. A button stores
/// nothing, so the id never appears in the settings file.
pub const ACCOUNTS: OptionDescriptor = OptionDescriptor {
    id: "accounts",
    label: "Accounts",
    kind: OptionKind::Button { label: "Open" },
    default: OptionValue::Action,
    description: "Add, switch or remove your Minecraft accounts. The title screen's account \
                  corner (top right) opens the same screen.",
};

/// "Check for updates": whether the portable Aero Client asks GitHub
/// for a newer release when the Aero title screen opens (at most once every 6 hours) and shows a
/// card at the bottom left when there is one. On by default. Nothing is downloaded until the
/// player clicks Update (and a build without an update signing key only links the release page).
/// Builds from source never check. Client half: `mc-client/src/app/mods/w21_updater.rs`.
pub const CHECK_FOR_UPDATES: OptionDescriptor = OptionDescriptor {
    id: "check_for_updates",
    label: "Check for updates",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(true),
    description: "Look for a new Aero Client version on GitHub when the title screen opens (at \
                  most every 6 hours) and show it at the bottom left. Nothing is downloaded \
                  unless you click Update.",
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::ModSettings;

    #[test]
    fn check_for_updates_is_a_toggle_on_by_default_under_updates() {
        assert_eq!(CHECK_FOR_UPDATES.id, "check_for_updates");
        assert_eq!(CHECK_FOR_UPDATES.kind, OptionKind::Toggle);
        assert!(DESCRIPTOR.option("check_for_updates").is_some());
        let settings = ModSettings::new();
        assert!(
            settings.bool("aero_title", "check_for_updates"),
            "on by default"
        );
        let groups: Vec<_> = crate::menu_rules::groups_for("aero_title")
            .filter(|group| group.options.contains(&"check_for_updates"))
            .map(|group| group.title)
            .collect();
        assert_eq!(groups, ["Updates"]);
    }

    /// The client always tells servers its name, so the one-build "Tell servers you use Aero
    /// Client" toggle is gone: no option, no menu row or heading, no search entry. A settings
    /// file saved by that build (the toggle off) still loads with no warning and keeps the old
    /// key for the round trip, like any retired id.
    #[test]
    fn the_retired_client_brand_toggle_is_gone_and_old_files_still_load() {
        assert!(DESCRIPTOR.option("client_brand").is_none());
        assert!(crate::menu_rules::groups_for("aero_title")
            .all(|group| !group.options.contains(&"client_brand") && group.title != "Privacy"));
        assert!(!crate::menu_rules::applies_while_off(
            "aero_title",
            "client_brand"
        ));
        let old = r#"{"mods":{"aero_title":{"enabled":true,"options":{"client_brand":false,"logo_style":"Gradient"}}}}"#;
        let (settings, report) = ModSettings::from_json_str(old);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(report.migrated.is_empty(), "{:?}", report.migrated);
        assert_eq!(
            settings.choice("aero_title", "logo_style"),
            2,
            "Gradient kept"
        );
        let saved = settings.to_json();
        assert_eq!(
            saved["mods"]["aero_title"]["options"]["client_brand"], false,
            "kept for the round trip"
        );
    }

    #[test]
    fn the_menu_scene_is_random_by_default_and_lists_the_three_scenes() {
        assert_eq!(MENU_SCENE.id, "menu_scene");
        assert_eq!(MENU_SCENE.default, OptionValue::Choice(0));
        match MENU_SCENE.kind {
            OptionKind::Choice { choices } => {
                assert_eq!(choices, ["Random", "Greenhouse", "Glass", "Shore"]);
            }
            _ => panic!("Menu scene is a choice"),
        }
        let settings = ModSettings::new();
        assert!(settings.enabled("aero_title"), "on by default");
        assert_eq!(settings.choice("aero_title", "menu_scene"), 0, "Random");
    }

    /// The retired ids are gone and stay gone, and a settings entry written before then (both
    /// at their defaults, read from their settings file) still loads: no warning, no
    /// migration note, the new option at Random, the logo style kept, and the old keys kept for
    /// the round trip like any unknown key.
    #[test]
    fn retired_ids_are_never_reused_and_old_files_with_them_still_load() {
        for id in RETIRED_OPTION_IDS {
            assert!(DESCRIPTOR.option(id).is_none(), "{id} is retired");
        }
        for old in [
            r#"{"mods":{"aero_title":{"enabled":true,"options":{"loading_min_time":2.0,"loading_screen":"Random","logo_style":"Italic"}}}}"#,
            r#"{"mods":{"aero_title":{"enabled":true,"options":{"loading_min_time":0.5,"loading_screen":"Classic","logo_style":"Gradient"}}}}"#,
        ] {
            let (settings, report) = ModSettings::from_json_str(old);
            assert!(report.warnings.is_empty(), "{old}: {:?}", report.warnings);
            assert!(report.migrated.is_empty(), "{old}: {:?}", report.migrated);
            assert!(settings.enabled("aero_title"));
            assert_eq!(settings.choice("aero_title", "menu_scene"), 0, "Random");
            let style = if old.contains("Gradient") { 2 } else { 0 };
            assert_eq!(settings.choice("aero_title", "logo_style"), style);
            let saved = settings.to_json();
            let options = &saved["mods"]["aero_title"]["options"];
            assert_eq!(options["menu_scene"], "Random");
            assert!(options["loading_screen"].is_string(), "kept: {options}");
            assert!(options["loading_min_time"].is_number(), "kept: {options}");
            let (again, report) = ModSettings::from_json_str(&settings.to_json_string());
            assert!(report.warnings.is_empty() && report.migrated.is_empty());
            assert_eq!(again.to_json(), saved, "{old}: a stable round trip");
        }
    }
}
