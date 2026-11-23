//! The replay browser's remembered order and view (`replay.browser_sort`, `replay.browser_view`)
//! are how the player left the window, not settings: they survive a restart in the settings file,
//! but they never make the Replay mod count as changed, and a whole-setup profile neither saves
//! nor loads them.

use mc_mods::mods::replay::ids::{BROWSER_SORT, BROWSER_VIEW};
use mc_mods::settings::is_runtime_state;
use mc_mods::ModSettings;

const ID: &str = "replay";

#[test]
fn the_browsers_order_and_view_are_not_settings() {
    let mut s = ModSettings::new();
    let m = s.index_of(ID).expect("the replay mod");
    assert_eq!(s.text(ID, BROWSER_SORT), "newest", "the default order");
    assert_eq!(s.text(ID, BROWSER_VIEW), "list", "the default view");
    let json = s.setup_json();
    let (profile, _) = ModSettings::from_json_str(&json.to_string());
    // Picking an order and a view in the browser changes no setting...
    s.set_text(ID, BROWSER_SORT, "server");
    s.set_text(ID, BROWSER_VIEW, "folders");
    assert!(s.options_are_default_at(m), "the Replay mod is not changed");
    assert!(s.setup_eq(&profile), "a loaded profile stays current");
    // ...and loading a profile leaves them as the player left them.
    assert!(s.setup_json()["mods"][ID]["options"]
        .get(BROWSER_SORT)
        .is_none());
    s.set_bool(ID, "auto_start", false);
    assert!(s.apply_setup(&profile));
    assert!(s.bool(ID, "auto_start"));
    assert_eq!(s.text(ID, BROWSER_SORT), "server");
    // The settings file keeps them across a restart.
    let (restarted, _) = ModSettings::from_json_str(&s.to_json_string());
    assert_eq!(restarted.text(ID, BROWSER_SORT), "server");
    assert_eq!(restarted.text(ID, BROWSER_VIEW), "folders");
    assert!(is_runtime_state(ID, BROWSER_SORT) && is_runtime_state(ID, BROWSER_VIEW));
    assert!(!is_runtime_state(ID, "auto_start"));
    // A word the browser does not know is kept as text; the browser falls back to its default.
    s.set_text(ID, BROWSER_SORT, "a much longer word than sixteen");
    assert_eq!(s.text(ID, BROWSER_SORT).chars().count(), 16, "capped at 16");
}
