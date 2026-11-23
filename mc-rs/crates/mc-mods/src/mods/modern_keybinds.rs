//! Modern Keybinds: keys you are still holding work again as soon as a screen closes.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! 1.8.9 releases every key binding when a screen opens and never re-reads the keyboard when it
//! closes, so W held through the inventory does nothing until it is pressed again. With this on
//! (the default, and vanilla's behaviour since 1.13) the held-type
//! bindings (movement, jump, sneak, sprint, player list) read as held again at the first tick
//! back in game. Edge bindings (inventory, drop, chat, hotbar, perspective) never fire from it.
//! The client half is `mc-client/src/app/mods/x4_input.rs`.
//!
//! "Eating stops mining" ([`EAT_STOPS_MINING`]) lived here before it moved to 1.7 Animations as
//! "Using items stops mining" (`old_animations.use_stops_mining`), which also covers sword
//! blocking and bow drawing. The old id is retired: never reuse it. A saved "off" carries over once
//! ([`eat_stops_mining_was_off`]); the key itself stays in the file as an unknown option.

use serde_json::{Map, Value};

use crate::category::Category;
use crate::descriptor::ModDescriptor;

/// The retired "Eating stops mining" option id. Retired: never reuse it.
pub const EAT_STOPS_MINING: &str = "eat_stops_mining";

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "modern_keybinds",
    name: "Modern Keybinds",
    category: Category::Controls,
    description: "Keys you are still holding (WASD, jump, sneak, sprint) work again as soon as \
                  a screen closes, as in Minecraft 1.13 and later. Off: they work as in 1.8.9.",
    default_enabled: true,
    options: &[],
    hud_elements: &[],
};

/// Whether a settings file's `mods` object turned the retired
/// "Eating stops mining" off and has never saved its replacement, 1.7 Animations' "Using items
/// stops mining". Then the replacement loads off too: someone who turned off a feature that
/// changes when packets are sent should not find it back on, and on for swords and bows too.
///
/// A saved `true` was the default, so the replacement keeps its own default. Once the store saves,
/// the new key is in the file and this never applies again, so a later choice is never undone.
#[must_use]
pub fn eat_stops_mining_was_off(mods: &Map<String, Value>) -> bool {
    let old = mods
        .get(DESCRIPTOR.id)
        .and_then(|entry| entry.get("options"))
        .and_then(|options| options.get(EAT_STOPS_MINING));
    let new_saved = mods
        .get("old_animations")
        .and_then(|entry| entry.get("options"))
        .and_then(|options| options.get("use_stops_mining"))
        .is_some();
    old == Some(&Value::Bool(false)) && !new_saved
}
