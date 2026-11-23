//! Graceful shutdown.
//!
//! java: `Minecraft.shutdownMinecraftApplet` (`:1049`) — stop the stream, `loadWorld(null)`
//! (which calls `NetHandlerPlayClient.cleanup`), `unloadSounds()`, then `Display.destroy()` in a
//! `finally` so the window goes even if the rest threw.
//!
//! # Two deliberate differences from vanilla, both improvements
//!
//! 1. **Settings are saved first, not never.** Vanilla calls `saveOptions()` every time an
//!    option changes and never at exit, so a crash between the last change and the exit loses
//!    nothing — but so does an option changed by code rather than by the options screen. Writing
//!    the file first here means a failure in any later step cannot lose it.
//! 2. **No `System.exit`.** Vanilla's `finally` block ends with `System.exit(0)`, which skips
//!    every remaining destructor. [`run`] returns and lets the caller drop things in order,
//!    which is what makes the leak check in this section meaningful at all.
//!
//! The ordering that is *not* negotiable is vanilla's: the connection closes before the world is
//! unloaded (a socket that is still delivering packets into a world being torn down is a use of
//! freed state), and the GPU is released last (every other subsystem may hold a texture).

use crate::error::ClientError;

/// One step of the shutdown, in the order [`run`] performs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShutdownStep {
    /// `options.txt` written. java: `GameSettings.saveOptions`.
    SavedSettings,
    /// `servers.dat` written. java: `ServerList.saveServerList`.
    SavedServerList,
    /// The connection task asked to close. java: `NetworkManager.closeChannel`.
    ClosedConnection,
    /// The world, the ECS and every chunk mesh dropped. java: `loadWorld(null)`.
    UnloadedWorld,
    /// Playing sounds stopped and buffers released. java: `mcSoundHandler.unloadSounds`.
    StoppedSounds,
    /// GPU resources released. java: `Display.destroy`.
    ReleasedGpu,
}

/// What a shutdown did, including anything that failed.
#[derive(Debug, Default)]
pub struct ShutdownReport {
    /// The steps that ran, in order.
    pub steps: Vec<ShutdownStep>,
    /// Failures, paired with the step they happened in. A failure never stops the shutdown —
    /// vanilla's `finally` guarantees the same thing.
    pub failures: Vec<(ShutdownStep, String)>,
}

impl ShutdownReport {
    /// Whether every step succeeded.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty()
    }
}

/// The pieces a shutdown touches.
///
/// A trait rather than a concrete `&mut Client` so the *ordering* — the part that is actually
/// load-bearing — can be tested without a GPU, a socket or a filesystem.
pub trait ShutdownTarget {
    /// Write `options.txt`.
    ///
    /// # Errors
    /// The file could not be written.
    fn save_settings(&mut self) -> Result<(), ClientError>;
    /// Write `servers.dat`.
    ///
    /// # Errors
    /// The file could not be written.
    fn save_server_list(&mut self) -> Result<(), ClientError>;
    /// Ask the connection task to close. Must be safe to call with no connection.
    fn close_connection(&mut self);
    /// Drop the world, the ECS and the chunk meshes.
    fn unload_world(&mut self);
    /// Stop and release audio.
    fn stop_sounds(&mut self);
    /// Release GPU resources: phases, pipelines, buffers, textures.
    fn release_gpu(&mut self);
}

/// Shut the client down in vanilla's order, never giving up part-way.
pub fn run(target: &mut dyn ShutdownTarget) -> ShutdownReport {
    mc_core::profile_scope!("shutdown");
    let mut report = ShutdownReport::default();

    if let Err(error) = target.save_settings() {
        report
            .failures
            .push((ShutdownStep::SavedSettings, error.to_string()));
    }
    report.steps.push(ShutdownStep::SavedSettings);

    if let Err(error) = target.save_server_list() {
        report
            .failures
            .push((ShutdownStep::SavedServerList, error.to_string()));
    }
    report.steps.push(ShutdownStep::SavedServerList);

    target.close_connection();
    report.steps.push(ShutdownStep::ClosedConnection);

    target.unload_world();
    report.steps.push(ShutdownStep::UnloadedWorld);

    target.stop_sounds();
    report.steps.push(ShutdownStep::StoppedSounds);

    target.release_gpu();
    report.steps.push(ShutdownStep::ReleasedGpu);

    tracing::info!(
        steps = report.steps.len(),
        failures = report.failures.len(),
        "stopping"
    );
    report
}

/// The order [`run`] performs its steps in, exposed so a test can assert against it by name
/// rather than by repeating the list.
pub const ORDER: [ShutdownStep; 6] = [
    ShutdownStep::SavedSettings,
    ShutdownStep::SavedServerList,
    ShutdownStep::ClosedConnection,
    ShutdownStep::UnloadedWorld,
    ShutdownStep::StoppedSounds,
    ShutdownStep::ReleasedGpu,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Recorder {
        calls: Vec<ShutdownStep>,
        settings_fail: bool,
        servers_fail: bool,
    }

    impl ShutdownTarget for Recorder {
        fn save_settings(&mut self) -> Result<(), ClientError> {
            self.calls.push(ShutdownStep::SavedSettings);
            if self.settings_fail {
                return Err(ClientError::Io(std::io::Error::other("disk full")));
            }
            Ok(())
        }
        fn save_server_list(&mut self) -> Result<(), ClientError> {
            self.calls.push(ShutdownStep::SavedServerList);
            if self.servers_fail {
                return Err(ClientError::Io(std::io::Error::other("read-only")));
            }
            Ok(())
        }
        fn close_connection(&mut self) {
            self.calls.push(ShutdownStep::ClosedConnection);
        }
        fn unload_world(&mut self) {
            self.calls.push(ShutdownStep::UnloadedWorld);
        }
        fn stop_sounds(&mut self) {
            self.calls.push(ShutdownStep::StoppedSounds);
        }
        fn release_gpu(&mut self) {
            self.calls.push(ShutdownStep::ReleasedGpu);
        }
    }

    #[test]
    fn every_step_runs_exactly_once_in_the_documented_order() {
        let mut recorder = Recorder::default();
        let report = run(&mut recorder);
        assert_eq!(recorder.calls, ORDER.to_vec());
        assert_eq!(report.steps, ORDER.to_vec());
        assert!(report.is_clean());
    }

    /// The two constraints that are correctness rather than taste: the socket closes before the
    /// world it feeds is torn down, and the GPU is released after everything that holds a
    /// texture.
    #[test]
    fn the_connection_closes_before_the_world_and_the_gpu_goes_last() {
        let mut recorder = Recorder::default();
        run(&mut recorder);
        let index = |step: ShutdownStep| recorder.calls.iter().position(|s| *s == step).unwrap();
        assert!(index(ShutdownStep::ClosedConnection) < index(ShutdownStep::UnloadedWorld));
        assert_eq!(
            index(ShutdownStep::ReleasedGpu),
            recorder.calls.len() - 1,
            "nothing may run after the GPU is gone"
        );
        assert!(index(ShutdownStep::SavedSettings) < index(ShutdownStep::ClosedConnection));
    }

    /// java: `shutdownMinecraftApplet`'s `finally` — `Display.destroy()` happens even when the
    /// earlier block threw. A failed save must not leak the GPU.
    #[test]
    fn a_failing_step_is_recorded_and_the_rest_still_runs() {
        let mut recorder = Recorder {
            settings_fail: true,
            servers_fail: true,
            ..Recorder::default()
        };
        let report = run(&mut recorder);
        assert_eq!(recorder.calls, ORDER.to_vec(), "nothing was skipped");
        assert!(!report.is_clean());
        assert_eq!(report.failures.len(), 2);
        assert_eq!(report.failures[0].0, ShutdownStep::SavedSettings);
        assert!(report.failures[0].1.contains("disk full"));
        assert_eq!(report.failures[1].0, ShutdownStep::SavedServerList);
    }

    /// Shutting down twice must be harmless: the run loop can reach it by the quit button and
    /// by the window closing in the same frame.
    #[test]
    fn shutting_down_twice_is_harmless() {
        let mut recorder = Recorder::default();
        run(&mut recorder);
        run(&mut recorder);
        assert_eq!(recorder.calls.len(), ORDER.len() * 2);
    }
}
