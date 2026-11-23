//! Surviving the audio output device going away.
//!
//! The output library restarts its stream by itself when the device disappears or the default
//! device changes (it checks twice a second, on a thread of its own). A restart that fails, for
//! example because the new device vanished again or refused the format, is an `unwrap` on that
//! thread (`backend/cpal/desktop/stream_manager.rs` in the 0.12 line), so the thread panics and
//! the output is gone for the rest of the session.
//!
//! That thread is not ours and cannot run inside `mc_core::panic_scope::contain`, so the game's
//! panic hook asks [`is_output_backend_panic`] instead: such a panic is not the game's crash
//! (vanilla's sound system logs its errors and the game goes on), and it is counted so that every
//! [`crate::AudioEngine`] rebuilds its output on whatever device is now the default at its next
//! tick ([`output_backend_panics`]).

use std::panic::PanicHookInfo;
use std::sync::atomic::{AtomicU64, Ordering};

/// Output-thread panics since start. A count rather than a flag, so each engine reacts once
/// without taking the news away from another.
static OUTPUT_BACKEND_PANICS: AtomicU64 = AtomicU64::new(0);

/// Whether `info` is a panic on the output library's own stream thread, counting it for the
/// engines, which restart their output on their next tick. Called from the panic hook only.
#[must_use]
pub fn is_output_backend_panic(info: &PanicHookInfo<'_>) -> bool {
    let matched = info
        .location()
        .is_some_and(|location| is_output_backend_source(location.file()));
    if matched {
        OUTPUT_BACKEND_PANICS.fetch_add(1, Ordering::AcqRel);
    }
    matched
}

/// Whether a panic location's source file belongs to the output library's stream backend. The
/// path is the crate's own (`.../kira-<version>/src/backend/cpal/...`), absolute in a
/// development build and starting at the remapped `cargo` folder in a packaged one.
fn is_output_backend_source(file: &str) -> bool {
    let after_crate = file.rfind("kira-").map_or("", |start| &file[start..]);
    after_crate.contains("/src/backend/cpal/") || after_crate.contains("\\src\\backend\\cpal\\")
}

/// Output-thread panics since start; one atomic load (an engine compares it with the count it
/// last acted on). Engine tests move their own last-seen count instead of this one, so a test
/// never restarts another test's engine.
pub(crate) fn output_backend_panics() -> u64 {
    OUTPUT_BACKEND_PANICS.load(Ordering::Acquire)
}

// ---- The output coming back with a device ---------------------------
//
// When the output device disappears, the output library stops its stream and opens one on the
// new default device; when there is no device at all, it stops the stream and never looks again
// (`check_stream` only acts while a stream runs). So after every device was gone (headphones
// unplugged on a PC with no speakers, a USB DAC or a monitor's speakers switched off, a driver
// reset), sound never came back. An engine that cannot open an output at start (no device) or
// at a restart stayed silent too. The engine's output watch looks every few seconds and opens a
// new output once the default device is there.

/// Ticks between looks at the output (3 s at 20 ticks a second).
pub(crate) const WATCH_TICKS: u32 = 60;
/// The longest wait between attempts while reopening does not bring the output back (1 minute).
pub(crate) const WATCH_MAX_TICKS: u32 = 1200;

/// An output backend's sign of life, for the engine's output watch.
pub trait OutputHeartbeat {
    /// Audio periods rendered since the last call; `None` when the backend cannot tell (the test
    /// backend renders only when a test drives it).
    fn periods_since_last_look(&mut self) -> Option<usize>;
}

impl OutputHeartbeat for kira::backend::cpal::CpalBackend {
    /// The output library queues a CPU-usage figure for every period its stream renders (a
    /// 100-entry queue nothing else reads): any figure since the last look means it runs.
    fn periods_since_last_look(&mut self) -> Option<usize> {
        let mut periods = 0;
        while self.pop_cpu_usage().is_some() {
            periods += 1;
        }
        Some(periods)
    }
}

impl OutputHeartbeat for kira::backend::mock::MockBackend {
    fn periods_since_last_look(&mut self) -> Option<usize> {
        None
    }
}

/// What a look at the output decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Look {
    /// The output renders (or its backend cannot tell): nothing to do.
    Fine,
    /// No running output and no device to open one on: look again later.
    NoDevice,
    /// No running output, and a device is there: open a new output on it.
    Reopen,
}

/// When the engine looks at its output, and what it then does. Counting is one decrement per
/// tick; a look asks the backend for its periods (a short queue drain) and, only when the output
/// does not run, whether a default device exists.
#[derive(Debug)]
pub(crate) struct OutputWatch {
    countdown: u32,
    interval: u32,
}

impl OutputWatch {
    pub(crate) fn new() -> Self {
        Self {
            countdown: WATCH_TICKS,
            interval: WATCH_TICKS,
        }
    }

    /// Count one tick; `true` when it is time to look.
    pub(crate) fn due(&mut self) -> bool {
        self.countdown = self.countdown.saturating_sub(1);
        if self.countdown > 0 {
            return false;
        }
        self.countdown = self.interval;
        true
    }

    /// Decide from what the output did since the last look (`None`: there is no output) and,
    /// only if it is not running, whether a default device is there now.
    pub(crate) fn look(
        &mut self,
        periods: Option<Option<usize>>,
        device_present: impl FnOnce() -> bool,
    ) -> Look {
        let running = matches!(periods, Some(None) | Some(Some(1..)));
        if running {
            self.interval = WATCH_TICKS;
            return Look::Fine;
        }
        if !device_present() {
            // Probing is cheap: keep looking every few seconds so sound returns promptly.
            self.interval = WATCH_TICKS;
            self.countdown = WATCH_TICKS;
            return Look::NoDevice;
        }
        // If the new output does not run either (a device that refuses to open), wait twice as
        // long before each further attempt.
        self.interval = self.interval.saturating_mul(2).min(WATCH_MAX_TICKS);
        self.countdown = self.interval;
        Look::Reopen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stream_backend_is_recognised_in_every_path_shape() {
        for file in [
            // a development build on Windows
            r"C:\Users\someone\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\kira-0.12.3\src\backend\cpal\desktop\stream_manager.rs",
            // a packaged build (`--remap-path-prefix` maps the cargo home to `cargo`)
            r"cargo\registry\src\index.crates.io-1949cf8c6b5b557f\kira-0.12.3\src\backend\cpal\desktop\stream_manager.rs",
            // Unix
            "/home/someone/.cargo/registry/src/index.crates.io-x/kira-0.12.3/src/backend/cpal/desktop/stream_manager.rs",
        ] {
            assert!(is_output_backend_source(file), "{file}");
        }
    }

    fn ticks_to_next_look(watch: &mut OutputWatch) -> u32 {
        (1..=10_000).find(|_| watch.due()).expect("a look comes")
    }

    /// A look every few seconds; a running output (or a backend that
    /// cannot tell) is left alone without asking for a device; a dead output, or none at all,
    /// waits for a device and is reopened once one is there; while reopening does not help the
    /// wait doubles up to a minute, and it is back to every few seconds once the output runs.
    #[test]
    fn the_output_watch_reopens_a_dead_output_once_a_device_is_there() {
        let mut watch = OutputWatch::new();
        assert_eq!((0..WATCH_TICKS * 3).filter(|_| watch.due()).count(), 3);
        let never = || -> bool { panic!("a running output never asks for a device") };
        assert_eq!(watch.look(Some(Some(37)), never), Look::Fine);
        assert_eq!(watch.look(Some(None), never), Look::Fine);
        // Every device gone: nothing renders and there is nothing to open.
        assert_eq!(watch.look(Some(Some(0)), || false), Look::NoDevice);
        assert_eq!(watch.look(None, || false), Look::NoDevice);
        assert_eq!(ticks_to_next_look(&mut watch), WATCH_TICKS);
        // A device is back.
        assert_eq!(watch.look(Some(Some(0)), || true), Look::Reopen);
        assert_eq!(ticks_to_next_look(&mut watch), 2 * WATCH_TICKS);
        // The new output does not run either: ever longer waits, at most a minute.
        for _ in 0..10 {
            assert_eq!(watch.look(None, || true), Look::Reopen);
        }
        assert_eq!(ticks_to_next_look(&mut watch), WATCH_MAX_TICKS);
        // It runs: every few seconds again.
        assert_eq!(watch.look(Some(Some(1)), never), Look::Fine);
        let _ = ticks_to_next_look(&mut watch);
        assert_eq!(ticks_to_next_look(&mut watch), WATCH_TICKS);
    }

    /// Negative controls: the game's own audio code, the library's other modules, and a path
    /// that merely mentions the backend outside the library are all real crashes.
    #[test]
    fn everything_else_is_not_the_stream_backend() {
        for file in [
            r"mc-rs\crates\mc-audio\src\engine.rs",
            r"C:\x\kira-0.12.3\src\sound\static_sound\handle.rs",
            r"C:\x\cpal-0.15.3\src\host\wasapi\stream.rs",
            r"crates\mc-client\src\backend\cpal\kira.rs",
            "",
        ] {
            assert!(!is_output_backend_source(file), "{file}");
        }
    }
}
