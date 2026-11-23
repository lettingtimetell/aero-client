//! Replay QA: playback fixes that live beside the player.
//!
//! # The world border runs on the recording's clock
//!
//! Vanilla's `WorldBorder` stamps `System.currentTimeMillis()` when an `S44` is applied and reads
//! the clock again to place a moving border. A replay applies its packets whenever the viewer
//! reaches them, so on the wall clock a shrinking border kept shrinking while the replay was
//! paused, landed in a different place after every seek (the rebuild, a checkpoint restore and
//! playing through all stamped other times) and differed between two exports of the same range.
//!
//! While a replay plays, its session's [`Environment`](crate::sync::env::Environment) carries
//! the recording's clock instead ([`Environment::set_replay_clock`]): the recording's start in
//! Unix milliseconds plus the replay time of the record being applied, and of the frame being
//! drawn. Every border stamp and read (packets, the local player's collision, clicks, the wall
//! and its warning vignette, the forcefield's scroll) then depends on replay time only. A live
//! session's environment never has the clock set, so live play keeps vanilla's wall clock.
//!
//! [`Environment::set_replay_clock`]: crate::sync::env::Environment::set_replay_clock

use mc_replay::{ReplayMeta, ReplayTime};

use super::super::super::Client;

/// The recording's clock at `time`: its start (Unix milliseconds) plus the replay time.
/// Pauses are cut out of replay time, so this is the clock the recording client had with its
/// pauses removed; only differences between two readings matter to the border.
pub(super) fn replay_clock_millis(meta: &ReplayMeta, time: ReplayTime) -> i64 {
    let start = i64::try_from(meta.started_unix_ms).unwrap_or(0);
    start.saturating_add(i64::try_from(time / 1_000).unwrap_or(i64::MAX))
}

impl Client {
    /// Put the replayed world on the recording's clock at `time` (the
    /// record about to be applied, or the frame about to be drawn).
    pub(super) fn nw_rq_replay_clock(&mut self, time: ReplayTime) {
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return;
        };
        let millis = replay_clock_millis(&playback.meta, time);
        if let Some(session) = self.session.as_mut() {
            session.state_mut().env.set_replay_clock(Some(millis));
        }
    }
}

#[cfg(test)]
mod tests {
    use mc_protocol::play::s2c::WorldBorder;

    use super::*;
    use crate::sync::env::Environment;

    /// A JoinGame or a dimension change replaces the world's `Environment`; the border packet
    /// that follows in the same batch must still be stamped with the replay's clock (the
    /// determinism test caught wall-clock stamps there).
    #[test]
    fn the_replay_clock_survives_a_world_reset() {
        let mut state = crate::sync::state::ClientState::new();
        state.env.set_replay_clock(Some(1_000_123));
        state.reset_world_for_join(0);
        assert_eq!(state.env.replay_clock(), Some(1_000_123), "a JoinGame");
        state.reset_world_for_dimension_change(-1);
        assert_eq!(
            state.env.replay_clock(),
            Some(1_000_123),
            "a dimension change"
        );
        state
            .env
            .apply_world_border(&WorldBorder::SetSize { diameter: 100.0 });
        assert_eq!(state.env.border.start_time_millis, 1_000_123);
        // Live play: no clock, and a reset does not invent one.
        let mut live = crate::sync::state::ClientState::new();
        live.reset_world_for_join(0);
        assert_eq!(live.env.replay_clock(), None);
    }

    #[test]
    fn the_replay_clock_is_the_recordings_start_plus_replay_time() {
        let meta = ReplayMeta {
            started_unix_ms: 1_794_621_923_186,
            ..ReplayMeta::default()
        };
        assert_eq!(replay_clock_millis(&meta, 0), 1_794_621_923_186);
        assert_eq!(replay_clock_millis(&meta, 2_500_999), 1_794_621_925_686);
    }

    /// A shrinking border on the replay clock is a function of replay time: applying its packet
    /// at two different wall-clock moments (a rebuild, a restore, two exports) gives the same
    /// diameter at the same replay time, and the diameter does not move while the replay is
    /// paused. On the wall clock (the control) neither holds.
    #[test]
    fn a_moving_border_follows_replay_time_not_the_wall_clock() {
        let meta = ReplayMeta {
            started_unix_ms: 1_000_000,
            ..ReplayMeta::default()
        };
        let shrink = WorldBorder::LerpSize {
            old_diameter: 200.0,
            new_diameter: 100.0,
            speed_millis: 10_000,
        };
        // The packet was recorded at 3 s; the frame is drawn at 8 s (half way through).
        let at = |time: ReplayTime| replay_clock_millis(&meta, time);
        let diameter_after_applying = |env: &mut Environment| {
            env.set_replay_clock(Some(at(3_000_000)));
            env.apply_world_border(&shrink);
            env.set_replay_clock(Some(at(8_000_000)));
            env.border.current_diameter(env.border_now_millis())
        };
        let mut first = Environment::default();
        let first_pass = diameter_after_applying(&mut first);
        std::thread::sleep(std::time::Duration::from_millis(30));
        let mut second = Environment::default();
        let second_pass = diameter_after_applying(&mut second);
        assert_eq!(first_pass, 150.0);
        assert_eq!(
            first_pass, second_pass,
            "the same replay time, the same border"
        );
        // Paused: the wall clock moves on, the replay clock does not.
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(
            second.border.current_diameter(second.border_now_millis()),
            150.0
        );

        // Control: on the wall clock the packet is stamped "now" and read "now", so the
        // diameter depends on when it was applied and keeps moving while nothing is replayed.
        let mut live = Environment::default();
        live.apply_world_border(&shrink);
        let early = live.border.current_diameter(live.border_now_millis());
        std::thread::sleep(std::time::Duration::from_millis(30));
        let later = live.border.current_diameter(live.border_now_millis());
        assert!(
            early > 199.0 && later < early,
            "wall clock: {early} then {later}"
        );
    }
}
