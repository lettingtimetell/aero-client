//! One file per HUD element. The lane that owns the mod owns its file.
//!
//! [`registry`] lists every element in draw order (later draws on top). It already lists an
//! element for every `HudElementDescriptor` in the catalog, so a lane only edits its own file.
//! A test in `modhud` fails if the catalog and this list disagree.

pub mod armor_status;
pub mod boss_bar;
pub mod coordinates;
pub mod cps;
pub mod fps;
pub mod item_info;
pub mod keystrokes;
pub mod ping;
pub mod potion_status;
pub mod reach_display;
pub mod scoreboard;
// The Spotify widget
pub mod spotify;
mod spotify_aero;
pub mod toggle_sprint;
// Action bar and titles
pub mod action_bar;
pub mod titles;

use crate::modhud::HudElement;

/// Every HUD element, in draw order.
#[must_use]
pub fn registry() -> Vec<Box<dyn HudElement>> {
    vec![
        Box::<fps::FpsElement>::default(),
        Box::<cps::CpsElement>::default(),
        Box::<ping::PingElement>::default(),
        Box::<coordinates::CoordinatesElement>::default(),
        Box::<armor_status::ArmorStatusElement>::default(),
        Box::<potion_status::PotionStatusElement>::default(),
        Box::<reach_display::ReachDisplayElement>::default(),
        Box::<item_info::ItemInfoElement>::default(),
        Box::<keystrokes::KeystrokesElement>::default(),
        Box::<boss_bar::BossBarElement>::default(),
        Box::<scoreboard::ScoreboardElement>::default(),
        Box::<toggle_sprint::ToggleSprintElement>::default(),
        Box::<spotify::SpotifyElement>::default(),
        Box::<action_bar::ActionBarElement>::default(),
        Box::<titles::TitlesElement>::default(),
    ]
}
