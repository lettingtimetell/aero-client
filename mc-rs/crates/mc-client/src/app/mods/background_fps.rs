//! Background FPS (`mc_mods::mods::background_fps`, on by default): a low frame cap while the game
//! window is in the background, to spare the CPU and the GPU.
//!
//! The shell asks [`Client::background_fps_cap`] (as `ShellApp::background_frame_cap`) only while
//! another window has the focus or the game window is minimized. It paces such a frame at the
//! lower of this cap and the usual one (`ShellApp::effective_frame_cap`: Max Framerate, the menus'
//! 30, the title screen's refresh rate), and waits for the next frame in its event loop, so the
//! focus coming back or a restore starts that frame at once. A focused window never asks: while
//! the game is played, the shell spends two `bool` tests a frame on it and nothing changes.
//!
//! # What never slows down
//!
//! * **A video export** renders as fast as it can, in front or not (the same test that comes first
//!   in `effective_frame_cap`).
//! * **The loading screen.** The asset load advances one stage per frame (`Client::step_loading`),
//!   so a cap would slow start-up for a player who switches away while the game loads. The loading
//!   screen's fade into the title screen counts as loading.
//! * **A diagnostic run** (`--diagnosticSimulation`): a measurement is never paced by the desktop.
//!   Benchmarks open their window with `MC_BACKGROUND_WINDOW`, which never reports a focus change,
//!   and their own `ShellApp` wrappers keep the default (no background cap).
//!
//! # Ticks and packets
//!
//! The game clock (`mc_render::frame::Timer`) still owes 20 ticks a second whatever the frame
//! rate, and catches up at most ten a frame, as vanilla's `Timer.elapsedTicks` does, so every stop
//! of the sliders (5 and up) keeps all 20. At the defaults (30 frames a second, 33 ms apart) a
//! frame runs at most one tick, so the player's movement packets keep one per tick, as at any
//! normal frame rate (two frames in three run one; the gaps alternate 33 and 67 ms around the
//! 50 ms tick, as for a player at 30 FPS). Below 20 frames a second a frame runs two or more ticks
//! back to back, exactly as vanilla 1.8.9 does on a slow PC: at 10, the packets leave in pairs
//! every 100 ms instead of one every 50 ms. Packets from the server are applied once per frame
//! (vanilla's `scheduledTasks`), so a reply the server waits for (a transaction confirm) can leave
//! up to one background frame later. Keep-alives are answered by the network task at once, frame
//! rate or not.
//!
//! # Cost
//!
//! The settings are read through a snapshot that is refreshed only when the settings generation
//! moves ([`BackgroundFpsSnapshot`]): with the mod off, a window in the background costs a few
//! `Option` tests and one `u64` compare a frame.

use std::cell::Cell;

use mc_mods::mods::background_fps::{ID, MINIMIZED_FPS, UNFOCUSED_FPS};
use mc_mods::ModSettings;

use super::super::Client;

/// The mod's two caps as the frame cap reads them, refreshed when the settings generation moves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BackgroundFpsSnapshot {
    /// The settings generation this was read at (`0`: never; the store starts at 1).
    seen: u64,
    /// `(unfocused, minimized)` frames a second, while the mod is on.
    caps: Option<(u32, u32)>,
}

impl BackgroundFpsSnapshot {
    /// The snapshot in `cache`, read again from `settings` if they changed since.
    fn read(cache: &Cell<Self>, settings: &ModSettings) -> Self {
        let mut value = cache.get();
        if value.seen != settings.generation() {
            value = Self {
                seen: settings.generation(),
                caps: settings
                    .enabled(ID)
                    .then(|| (fps(settings, UNFOCUSED_FPS), fps(settings, MINIMIZED_FPS))),
            };
            cache.set(value);
        }
        value
    }

    /// The cap for a window in the background (`minimized` or not), `None` with the mod off.
    fn cap(self, minimized: bool) -> Option<u32> {
        let (unfocused, minimized_cap) = self.caps?;
        Some(if minimized { minimized_cap } else { unfocused })
    }
}

/// A slider's frames a second. The store keeps it inside the slider's range (5 and up).
fn fps(settings: &ModSettings, option: &str) -> u32 {
    settings.f32(ID, option).round().max(1.0) as u32
}

/// Whether nothing may slow the frames down right now, whatever the window does: a video export,
/// the loading screen or a diagnostic run (see the module docs).
fn always_full_speed(exporting: bool, loading: bool, diagnostic: bool) -> bool {
    exporting || loading || diagnostic
}

impl Client {
    /// `ShellApp::background_frame_cap`: the frame cap while the window is in the background
    /// (`minimized`, or another window has the focus), or `None` when it has none (the mod off,
    /// a video export, the loading screen, a diagnostic run). Called only for a window in the
    /// background.
    pub(crate) fn background_fps_cap(&self, minimized: bool) -> Option<u32> {
        let loading = self.is_loading() || self.tw_loading_frame_cap().is_some();
        let diagnostic = self.config.diagnostic_simulation.is_some();
        if always_full_speed(self.tw_export_uncapped(), loading, diagnostic) {
            return None;
        }
        BackgroundFpsSnapshot::read(&self.background_fps, &self.mods.settings).cap(minimized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_render::ShellApp;

    #[test]
    fn the_snapshot_follows_the_switch_and_the_sliders_through_the_generation() {
        let mut settings = ModSettings::new();
        let cache = Cell::default();
        let on = BackgroundFpsSnapshot::read(&cache, &settings);
        assert_eq!(on.cap(false), Some(30), "unfocused, by default");
        assert_eq!(on.cap(true), Some(30), "minimized, by default");
        settings.set_f32(ID, UNFOCUSED_FPS, 60.0);
        settings.set_f32(ID, MINIMIZED_FPS, 5.0);
        let moved = BackgroundFpsSnapshot::read(&cache, &settings);
        assert_eq!((moved.cap(false), moved.cap(true)), (Some(60), Some(5)));
        settings.set_enabled(ID, false);
        let off = BackgroundFpsSnapshot::read(&cache, &settings);
        assert_eq!((off.cap(false), off.cap(true)), (None, None), "off: no cap");
        // Same generation: the cached value is used as it is (the store is not read again).
        let stale = BackgroundFpsSnapshot {
            seen: settings.generation(),
            caps: Some((1, 1)),
        };
        cache.set(stale);
        assert_eq!(BackgroundFpsSnapshot::read(&cache, &settings), stale);
    }

    /// What a background frame rate does to the game's ticks, through the shell's real clock
    /// (one movement packet goes out per tick): at the defaults (30) no frame runs two ticks; at
    /// 10 every frame runs two, back to back, and all 20 a second still run either way.
    #[test]
    fn the_defaults_never_run_two_ticks_in_one_frame() {
        let ticks_per_frame = |fps: i64| -> Vec<i32> {
            let mut timer = mc_render::frame::Timer::new(mc_render::frame::TICKS_PER_SECOND);
            timer.reset_clocks(0, 0);
            (1..=fps * 4)
                .map(|frame| {
                    let ms = frame * 1000 / fps;
                    timer.update_with(ms, ms);
                    timer.elapsed_ticks()
                })
                .collect()
        };
        let default = ticks_per_frame(30);
        assert_eq!(default.iter().sum::<i32>(), 80, "20 ticks a second");
        assert!(default.iter().all(|&ticks| ticks <= 1), "{default:?}");
        let ten = ticks_per_frame(10);
        assert_eq!(ten.iter().sum::<i32>(), 80, "20 ticks a second");
        assert!(ten.iter().all(|&ticks| ticks == 2), "{ten:?}");
    }

    #[test]
    fn exports_loads_and_diagnostic_runs_always_run_at_full_speed() {
        assert!(!always_full_speed(false, false, false));
        assert!(always_full_speed(true, false, false), "a video export");
        assert!(always_full_speed(false, true, false), "the loading screen");
        assert!(always_full_speed(false, false, true), "a diagnostic run");
    }

    /// A client over the real assets in a temp game dir. Skips without the vanilla jar (set
    /// `MC_JARS_DIR`).
    fn config() -> Option<(crate::app::config::ClientConfig, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = crate::app::config::ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: crate::app::config::LoginMode::Offline("Dev".to_owned()),
            ..crate::app::config::ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        Some((config, dir))
    }

    /// The real client on its title screen: the background caps, under the title screen's own
    /// cap (the monitor's refresh rate), and the focused cap unchanged; the mod off gives none.
    #[test]
    fn the_title_screen_runs_at_the_background_caps_and_off_is_unchanged() {
        let Some((config, _dir)) = config() else {
            return;
        };
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        client.monitor_refresh_changed(Some(359_966));
        let settings = client.settings.clone();
        assert!(!client.is_loading());
        assert_eq!(client.effective_frame_cap(&settings, true), Some(360));
        assert_eq!(
            client.effective_frame_cap(&settings, false),
            Some(360),
            "the usual cap does not read the focus; the shell takes the lower of the two"
        );
        assert_eq!(client.background_frame_cap(false), Some(30));
        assert_eq!(client.background_frame_cap(true), Some(30));
        client.mods.settings.set_f32(ID, UNFOCUSED_FPS, 45.0);
        client.mods.settings.set_f32(ID, MINIMIZED_FPS, 10.0);
        assert_eq!(client.background_frame_cap(false), Some(45));
        assert_eq!(client.background_frame_cap(true), Some(10));
        client.mods.settings.set_enabled(ID, false);
        assert_eq!(client.background_frame_cap(false), None, "off: as before");
        assert_eq!(client.background_frame_cap(true), None);
        assert_eq!(client.effective_frame_cap(&settings, false), Some(360));
    }

    /// While the game loads (the deferred load of the windowed client), a window in the
    /// background is never slowed down: each frame advances the load by one stage.
    #[test]
    fn the_loading_screen_is_never_capped() {
        let Some((config, _dir)) = config() else {
            return;
        };
        let client = Client::new_deferred(config).expect("boot assets load");
        assert!(client.is_loading(), "control: the load has not run yet");
        assert_eq!(client.background_frame_cap(false), None);
        assert_eq!(client.background_frame_cap(true), None);
    }

    /// A diagnostic run is never paced by the desktop. (A real one also needs `--replay` and a
    /// capture; the setting is what the cap reads.)
    #[test]
    fn a_diagnostic_run_is_never_capped() {
        let Some((config, _dir)) = config() else {
            return;
        };
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        assert_eq!(client.background_frame_cap(true), Some(30), "control");
        client.config.diagnostic_simulation = Some(crate::app::config::DiagnosticSimulation {
            seed: 1,
            warmup_frames: 1,
        });
        assert_eq!(client.background_frame_cap(true), None);
        assert_eq!(client.background_frame_cap(false), None);
    }
}
