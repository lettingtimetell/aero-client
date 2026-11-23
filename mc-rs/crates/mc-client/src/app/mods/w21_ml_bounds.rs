//! Per-player state that must not outlive its player.
//!
//! The leak harness (`tests/ml_leak_harness.rs`, scenario `players`: 50 new players a cycle,
//! then gone) showed the client keeping about one block per player ever seen: the cape chase
//! history and the profile payloads behind the skin and cape keys
//! ([`crate::app::appearance::AppearanceState::retain_players`] has the details). This prunes
//! them for players who are gone, off the frame path: one compare per tick, a pass over the
//! maps every [`PRUNE_EVERY_TICKS`] ticks.
//!
//! Kill switch: `MC_W21_ML_PRUNE=0` (read once) restores the old keep-everything behaviour.

use std::collections::HashSet;

use uuid::Uuid;

use crate::app::Client;

/// Five seconds of ticks: rare enough to cost nothing, often enough that a busy lobby's churn
/// (NPCs respawned with fresh UUIDs, players coming and going) never piles up.
const PRUNE_EVERY_TICKS: u64 = 100;

fn prune_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("MC_W21_ML_PRUNE").is_ok_and(|value| value == "0"))
}

impl Client {
    /// Forget the appearance state of players who are gone: not in
    /// the tab list, no tracked player entity, and not the local player. A player who is only
    /// out of sight (still listed) or only unlisted (an NPC entity) keeps everything.
    pub(crate) fn w21_ml_prune_departed_players(&mut self) {
        if !self.tick.is_multiple_of(PRUNE_EVERY_TICKS)
            || self.appearance.per_player_entries() == 0
            || !prune_enabled()
        {
            return;
        }
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let state = session.state();
        let entities: HashSet<Uuid> = state
            .entities
            .iter()
            .filter(|(_, tracked)| tracked.kind == crate::sync::entities::TrackedKind::Player)
            .filter_map(|(_, tracked)| tracked.uuid)
            .collect();
        let local = state.local.uuid;
        let players = &state.players;
        self.appearance.retain_players(|uuid| {
            *uuid == local || entities.contains(uuid) || players.get(uuid).is_some()
        });
    }
}
