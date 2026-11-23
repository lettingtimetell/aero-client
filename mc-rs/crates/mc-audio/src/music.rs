//! The music ticker: menu/game/biome track selection with vanilla's timing.
//!
// java: MusicTicker.java

use mc_assets::ResourceLocation;
use mc_math::JavaRandom;

/// Which background track should be playing, mirroring `Minecraft.getAmbientMusicType()`'s
/// possible results. Selecting *which* variant applies (menu vs. in-game, dimension, creative
/// mode, the end fight) is downstream game-state logic S19 doesn't own; [`MusicTicker::update`]
/// just takes whichever [`MusicType`] the caller determines is current each tick.
// java: MusicTicker.MusicType
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicType {
    /// Main menu.
    Menu,
    /// Overworld survival/adventure.
    Game,
    /// Overworld, creative mode.
    Creative,
    /// End-game credits roll.
    Credits,
    /// The Nether.
    Nether,
    /// Fighting the Ender Dragon.
    EndBoss,
    /// The End, once the dragon is dead.
    End,
}

impl MusicType {
    /// The `sounds.json` event this track plays.
    pub fn event(self) -> ResourceLocation {
        ResourceLocation::minecraft(match self {
            MusicType::Menu => "music.menu",
            MusicType::Game => "music.game",
            MusicType::Creative => "music.game.creative",
            MusicType::Credits => "music.game.end.credits",
            MusicType::Nether => "music.game.nether",
            MusicType::EndBoss => "music.game.end.dragon",
            MusicType::End => "music.game.end",
        })
    }

    /// Minimum ticks (20 TPS) before a new track of this type can start after the previous one
    /// ends, inclusive.
    pub const fn min_delay(self) -> i32 {
        match self {
            MusicType::Menu => 20,
            MusicType::Game => 12_000,
            MusicType::Creative => 1_200,
            MusicType::Credits => i32::MAX,
            MusicType::Nether => 1_200,
            MusicType::EndBoss => 0,
            MusicType::End => 6_000,
        }
    }

    /// Maximum ticks before a new track of this type must start, inclusive.
    pub const fn max_delay(self) -> i32 {
        match self {
            MusicType::Menu => 600,
            MusicType::Game => 24_000,
            MusicType::Creative => 3_600,
            MusicType::Credits => i32::MAX,
            MusicType::Nether => 3_600,
            MusicType::EndBoss => 0,
            MusicType::End => 24_000,
        }
    }
}

/// What [`MusicTicker`] currently believes about the one track it may be playing.
#[derive(Debug, Clone, PartialEq)]
struct Current {
    music_type: MusicType,
    /// Opaque handle from [`MusicPlayback::play`], passed back to `is_playing`/`stop`.
    handle: u64,
}

/// Abstraction over "start/query/stop a sound", so [`MusicTicker`]'s tick logic — the part with
/// vanilla timing to get right — is testable without a real [`crate::AudioEngine`] or backend.
/// `AudioEngine` itself is the real implementor in `mc-client`'s integration once S14 lands.
pub trait MusicPlayback {
    /// Starts `event` playing (ambient, non-positional — `PositionedSoundRecord.create(loc)`)
    /// and returns an opaque handle to it.
    fn play(&mut self, event: &ResourceLocation) -> u64;
    /// Whether the sound behind `handle` is still audible.
    fn is_playing(&self, handle: u64) -> bool;
    /// Stops the sound behind `handle` immediately.
    fn stop(&mut self, handle: u64);
}

/// Ports `MusicTicker.java` exactly: a track plays to completion, then a random cooldown (in
/// `[min_delay, max_delay]` ticks) before the next one starts; switching *which* [`MusicType`]
/// is current (e.g. walking into the Nether) cuts the old track early with a short cooldown
/// instead of the type's normal one.
// java: MusicTicker
#[derive(Debug)]
pub struct MusicTicker {
    rand: JavaRandom,
    current: Option<Current>,
    time_until_next: i32,
}

impl MusicTicker {
    /// A fresh ticker, matching `MusicTicker`'s field initialisers
    /// (`timeUntilNextMusic = 100`, no `currentMusic`).
    pub fn new(seed: i64) -> Self {
        Self {
            rand: JavaRandom::new(seed),
            current: None,
            time_until_next: 100,
        }
    }

    /// Whether a track is currently believed to be playing.
    pub fn is_playing(&self) -> bool {
        self.current.is_some()
    }

    /// One 20-TPS tick. `wanted` is whatever [`MusicType`] the caller's game-state logic says
    /// should be playing right now (`Minecraft.getAmbientMusicType()`'s result).
    ///
    /// # A vanilla off-by-one, kept intentionally
    /// Java's check is `timeUntilNextMusic-- <= 0` — a *post*-decrement: the comparison sees the
    /// value from *before* this tick's decrement. Starting from the field initialiser of `100`,
    /// that means the first tick where the pre-decrement value is `<= 0` is tick **101**, not
    /// tick 100 (`100, 99, ..., 1` on ticks 1-100, then `0 <= 0` is finally true on tick 101).
    /// Reproduced exactly here rather than "fixed", since a server-driven multiplayer session
    /// has no way to notice a one-tick difference in menu-music timing, but a systematically
    /// different accumulator would be a real (if tiny) parity bug.
    // java: MusicTicker.update
    pub fn update(&mut self, wanted: MusicType, playback: &mut impl MusicPlayback) {
        if let Some(current) = &self.current {
            if current.music_type != wanted {
                playback.stop(current.handle);
                // java: `MathHelper.getRandomIntegerInRange(rand, 0, minDelay / 2)`
                self.time_until_next = java_random_range(&mut self.rand, 0, wanted.min_delay() / 2);
            }
        }

        if let Some(current) = &self.current {
            if !playback.is_playing(current.handle) {
                self.current = None;
                self.time_until_next = self.time_until_next.min(java_random_range(
                    &mut self.rand,
                    wanted.min_delay(),
                    wanted.max_delay(),
                ));
            }
        }

        if self.current.is_none() {
            let fire = self.time_until_next <= 0;
            self.time_until_next -= 1;
            if fire {
                self.start(wanted, playback);
            }
        }
    }

    /// `MusicTicker.func_181558_a`: starts `music_type` now, unconditionally.
    ///
    /// The End-poem screen calls this directly for `MusicType::Credits`; unlike the ordinary
    /// [`Self::update`] path, that screen is paused and therefore never asks the ticker to select
    /// an ambient type while its forced track is running.
    pub fn start(&mut self, music_type: MusicType, playback: &mut impl MusicPlayback) {
        let handle = playback.play(&music_type.event());
        self.current = Some(Current { music_type, handle });
        self.time_until_next = i32::MAX;
    }

    /// `MusicTicker.func_181557_a`: stops whatever is playing (e.g. on disconnect) and resets
    /// the cooldown to zero, so a rejoin can start music immediately rather than waiting out
    /// whatever cooldown was in flight.
    pub fn stop(&mut self, playback: &mut impl MusicPlayback) {
        if let Some(current) = self.current.take() {
            playback.stop(current.handle);
            self.time_until_next = 0;
        }
    }
}

/// `MathHelper.getRandomIntegerInRange(random, min, max)`: `random.nextInt(max - min + 1) + min`.
/// `mc-math` doesn't expose this exact helper name, but `JavaRandom::range_i32` is the same
/// inclusive-range convention (confirmed against `mc-math`'s own differential tests).
fn java_random_range(rand: &mut JavaRandom, min: i32, max: i32) -> i32 {
    rand.range_i32(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct FakePlayback {
        next_handle: u64,
        /// handle -> remaining ticks until it reports "not playing"
        playing: HashMap<u64, u32>,
        log: Vec<(&'static str, ResourceLocation)>,
    }

    impl MusicPlayback for FakePlayback {
        fn play(&mut self, event: &ResourceLocation) -> u64 {
            self.next_handle += 1;
            let h = self.next_handle;
            // Default duration "forever" (test overrides via `finish`); tests call `tick`
            // explicitly and use `finish` to end a track.
            self.playing.insert(h, u32::MAX);
            self.log.push(("play", event.clone()));
            h
        }
        fn is_playing(&self, handle: u64) -> bool {
            self.playing.get(&handle).is_some_and(|&t| t > 0)
        }
        fn stop(&mut self, handle: u64) {
            self.playing.remove(&handle);
        }
    }

    #[test]
    fn starts_at_tick_101_with_no_prior_state() {
        // See `MusicTicker::update`'s doc comment: the field starts at 100 but the
        // post-decrement check means the first possible fire is tick 101, not 100.
        let mut ticker = MusicTicker::new(1);
        let mut playback = FakePlayback::default();
        assert!(!ticker.is_playing());
        for _ in 0..100 {
            ticker.update(MusicType::Menu, &mut playback);
        }
        assert!(!ticker.is_playing(), "must not start before tick 101");
        ticker.update(MusicType::Menu, &mut playback);
        assert!(ticker.is_playing(), "must start at tick 101");
        assert_eq!(playback.log, vec![("play", MusicType::Menu.event())]);
    }

    #[test]
    fn does_not_restart_while_still_playing() {
        let mut ticker = MusicTicker::new(1);
        let mut playback = FakePlayback::default();
        for _ in 0..101 {
            ticker.update(MusicType::Menu, &mut playback);
        }
        assert_eq!(playback.log.len(), 1);
        for _ in 0..500 {
            ticker.update(MusicType::Menu, &mut playback);
        }
        assert_eq!(playback.log.len(), 1, "should not re-trigger while playing");
    }

    #[test]
    fn schedules_a_cooldown_after_the_track_ends() {
        let mut ticker = MusicTicker::new(1);
        let mut playback = FakePlayback::default();
        for _ in 0..101 {
            ticker.update(MusicType::Menu, &mut playback);
        }
        assert_eq!(playback.log.len(), 1);
        let handle = *playback.playing.keys().next().unwrap();
        playback.playing.insert(handle, 0); // finish the track

        // One more update notices it stopped and schedules a cooldown in
        // [min_delay, max_delay] = [20, 600] ticks; it must not restart immediately.
        ticker.update(MusicType::Menu, &mut playback);
        assert!(!ticker.is_playing());
        assert_eq!(
            playback.log.len(),
            1,
            "must not restart on the same tick it stopped"
        );

        // ...but must restart within min_delay..=max_delay ticks of that.
        let mut restarted_at = None;
        for tick in 1..=MusicType::Menu.max_delay() {
            ticker.update(MusicType::Menu, &mut playback);
            if playback.log.len() == 2 {
                restarted_at = Some(tick);
                break;
            }
        }
        let restarted_at = restarted_at.expect("must restart within max_delay ticks");
        assert!(
            restarted_at >= MusicType::Menu.min_delay(),
            "restarted too early at {restarted_at}"
        );
    }

    #[test]
    fn switching_music_type_cuts_the_old_track_and_reschedules() {
        let mut ticker = MusicTicker::new(1);
        let mut playback = FakePlayback::default();
        for _ in 0..101 {
            ticker.update(MusicType::Menu, &mut playback);
        }
        assert!(ticker.is_playing());
        assert_eq!(playback.log[0].1, MusicType::Menu.event());

        // Switch to Game while Menu is still "playing" -> old track stopped immediately.
        ticker.update(MusicType::Game, &mut playback);
        assert!(!ticker.is_playing(), "old track must be cut on type switch");
    }

    #[test]
    fn switching_music_type_consumes_both_vanilla_random_draws() {
        let seed = 1;
        let mut ticker = MusicTicker::new(seed);
        let mut playback = FakePlayback::default();
        for _ in 0..101 {
            ticker.update(MusicType::Menu, &mut playback);
        }

        ticker.update(MusicType::Game, &mut playback);

        // MusicTicker.java keeps currentMusic non-null after stopSound, so the following
        // isSoundPlaying test observes the stopped handle and consumes a second random draw.
        // The cooldown value cannot prove this: [0, min/2] is always below [min, max], so
        // Math.min keeps the first draw. Pin the next draw instead, which makes deleting either
        // call (or accidentally using the old MENU delay range) bite.
        let mut oracle = JavaRandom::new(seed);
        let first = java_random_range(&mut oracle, 0, MusicType::Game.min_delay() / 2);
        let second = java_random_range(
            &mut oracle,
            MusicType::Game.min_delay(),
            MusicType::Game.max_delay(),
        );
        assert_eq!(ticker.time_until_next, first.min(second) - 1);
        assert_eq!(
            ticker.rand.next_i32_bound(10_000),
            oracle.next_i32_bound(10_000)
        );
    }

    #[test]
    fn stop_clears_state_and_allows_immediate_restart() {
        let mut ticker = MusicTicker::new(1);
        let mut playback = FakePlayback::default();
        for _ in 0..101 {
            ticker.update(MusicType::Menu, &mut playback);
        }
        assert!(ticker.is_playing());
        ticker.stop(&mut playback);
        assert!(!ticker.is_playing());

        ticker.update(MusicType::Menu, &mut playback);
        assert!(
            ticker.is_playing(),
            "must restart on the very next tick after stop()"
        );
    }

    #[test]
    fn sound_manager_reset_leaves_current_until_the_next_update() {
        let mut ticker = MusicTicker::new(1);
        let mut playback = FakePlayback::default();
        for _ in 0..101 {
            ticker.update(MusicType::Game, &mut playback);
        }
        assert!(ticker.is_playing());

        // Minecraft.loadWorld calls SoundHandler.stopSounds, which clears SoundManager's
        // playingSounds map without calling MusicTicker.func_181557_a. The next update observes
        // the stale logical handle as not playing and performs the normal cooldown transition.
        playback.playing.clear();
        assert!(
            ticker.is_playing(),
            "the ticker is not reset by stopAllSounds"
        );
        ticker.update(MusicType::Game, &mut playback);
        assert!(!ticker.is_playing());
    }

    #[test]
    fn end_boss_has_zero_delay_both_ways() {
        assert_eq!(MusicType::EndBoss.min_delay(), 0);
        assert_eq!(MusicType::EndBoss.max_delay(), 0);
    }

    #[test]
    fn every_music_type_has_a_registered_event_name() {
        for (mt, expected) in [
            (MusicType::Menu, "minecraft:music.menu"),
            (MusicType::Game, "minecraft:music.game"),
            (MusicType::Creative, "minecraft:music.game.creative"),
            (MusicType::Credits, "minecraft:music.game.end.credits"),
            (MusicType::Nether, "minecraft:music.game.nether"),
            (MusicType::EndBoss, "minecraft:music.game.end.dragon"),
            (MusicType::End, "minecraft:music.game.end"),
        ] {
            assert_eq!(mt.event().to_string(), expected);
        }
    }
}
