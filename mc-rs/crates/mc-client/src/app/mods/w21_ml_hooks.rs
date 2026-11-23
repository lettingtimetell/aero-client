//! Hooks for the leak harness (`tests/ml_leak_harness.rs`).
//!
//! The harness drives the client the way a player does, but an integration test cannot press
//! the mod menu's key. These two calls reach exactly the paths that key and Escape reach. Nothing
//! in the client calls them, so they cost nothing.

use std::time::Instant;

use crate::app::Client;

impl Client {
    /// Open the mod menu (its glass included), as its key binding does.
    #[doc(hidden)]
    pub fn leak_harness_open_mod_menu(&mut self) {
        self.open_mod_menu(Instant::now());
    }

    /// Close the mod menu, as Escape does; its closing animation plays on the next frames.
    #[doc(hidden)]
    pub fn leak_harness_close_mod_menu(&mut self) {
        self.close_mod_menu(Instant::now());
    }

    /// Entries in the appearance state's per-player maps (cape chase histories, skin and cape
    /// profile keys), which must follow the players present, not every player ever seen.
    #[doc(hidden)]
    #[must_use]
    pub fn leak_harness_per_player_entries(&self) -> usize {
        self.appearance.per_player_entries()
    }
}
