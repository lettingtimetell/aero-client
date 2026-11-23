#![warn(missing_docs)]
// The FFI calls that need `unsafe` are all in `ffi` (Windows only), which allows it.
#![deny(unsafe_code)]

//! Now-playing information and media controls for the Spotify mod.
//!
//! Lane **W8** owns this crate and implements the real source: on Windows, the system media
//! session (`GlobalSystemMediaTransportControlsSessionManager`, the same source as the volume
//! flyout) filtered to Spotify's desktop app, plus that app's own audio-session volume. Lane
//! **W9** (the HUD widget) codes against the public API below from the start, using
//! [`MediaHub::fake`] until the real worker lands.
//!
//! **Contract:** add to this API freely, but don't rename or remove anything, and don't change a
//! signature, without updating W9's code in the same change. Both lanes record API changes in
//! their notes.
//!
//! Threading: the game thread only ever calls the cheap, non-blocking methods of [`MediaHub`]
//! ([`MediaHub::state`], [`MediaHub::generation`], [`MediaHub::send`]). All Windows calls,
//! waiting and image decoding happen on the hub's own worker thread.
//!
//! The worker (W8, 2026-11-07) follows the session through its change events, with a 1 s poll
//! as a backstop while a player is followed and a 5 s poll while none is. It publishes a new
//! snapshot only when something shown changes (text, play state, a seek, the controls, the
//! volume or the cover), so [`MediaHub::generation`] stays still while a track simply plays on.
//! Covers are decoded and shrunk to at most 128x128 on the worker, only when their bytes change.
//! A panic on the worker is contained: the state goes empty and the game goes on (the client's
//! panic hook checks [`panic_is_being_contained`]).

mod apps;
mod art;
#[cfg(windows)]
mod ffi;
#[cfg(windows)]
mod gsmtc;
#[cfg_attr(not(windows), allow(dead_code))]
mod timeline;
#[cfg_attr(not(windows), allow(dead_code))]
mod worker;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use worker::Msg;

/// Whether the current thread is a media worker, whose panics are contained (the hub's state
/// goes empty and the game goes on). The client's panic hook checks this before recording a
/// crash, as it does for the voice thread.
#[must_use]
pub fn panic_is_being_contained() -> bool {
    worker::panic_is_being_contained()
}

/// A decoded album cover: straight (not premultiplied) RGBA8, row-major, no row padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlbumArt {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

/// One snapshot of the track the followed player is on.
#[derive(Debug, Clone)]
pub struct NowPlaying {
    /// The player's app id as Windows reports it (for example `Spotify.exe`).
    pub source_app: String,
    /// Track title.
    pub title: String,
    /// Artist line.
    pub artist: String,
    /// Album title (may be empty).
    pub album: String,
    /// True while playing; false while paused or stopped.
    pub playing: bool,
    /// Playback position at [`NowPlaying::sampled_at`], if the player reports a timeline.
    pub position_ms: Option<u64>,
    /// Track length, if known.
    pub duration_ms: Option<u64>,
    /// When `position_ms` was read. Use [`NowPlaying::position_at`] for a smooth progress bar.
    pub sampled_at: Instant,
    /// The player's own volume in its Windows audio session, `0.0..=1.0`, if it was found.
    pub volume: Option<f32>,
    /// The cover, if the player provides one. Shared, so cloning a snapshot stays cheap.
    pub art: Option<Arc<AlbumArt>>,
    /// Changes whenever `art` changes, so the widget re-uploads its texture only then.
    pub art_generation: u64,
    /// The player currently accepts play/pause.
    pub can_play_pause: bool,
    /// The player currently accepts "next track".
    pub can_next: bool,
    /// The player currently accepts "previous track".
    pub can_previous: bool,
}

impl NowPlaying {
    /// The playback position extrapolated to `now` while playing, clamped to the duration.
    #[must_use]
    pub fn position_at(&self, now: Instant) -> Option<u64> {
        let base = self.position_ms?;
        let elapsed = if self.playing {
            u64::try_from(now.saturating_duration_since(self.sampled_at).as_millis())
                .unwrap_or(u64::MAX)
        } else {
            0
        };
        let position = base.saturating_add(elapsed);
        Some(match self.duration_ms {
            Some(duration) => position.min(duration),
            None => position,
        })
    }
}

/// A control request for the followed player.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MediaCommand {
    /// Toggle between playing and paused.
    PlayPause,
    /// Skip to the next track.
    Next,
    /// Go to the previous track (or the start of this one, as the player decides).
    Previous,
    /// Set the player's own audio-session volume, `0.0..=1.0`.
    SetVolume(f32),
}

/// Which media sessions the hub follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceFilter {
    /// Only Spotify's desktop app.
    #[default]
    Spotify,
    /// Whatever Windows considers the current media session (browsers, other players).
    Any,
}

/// What the game thread reads each frame.
#[derive(Debug, Clone, Default)]
pub struct MediaState {
    /// Bumps on every change. Compare it to skip work when nothing changed.
    pub generation: u64,
    /// `None` when no matching player has a session, or the source is unavailable.
    pub now_playing: Option<Arc<NowPlaying>>,
}

/// State shared between a hub handle and whoever feeds it (the worker, or a [`FakeControl`]).
#[derive(Debug, Default)]
struct Shared {
    generation: AtomicU64,
    state: Mutex<MediaState>,
    filter: Mutex<SourceFilter>,
}

impl Shared {
    fn publish(&self, now_playing: Option<Arc<NowPlaying>>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.generation = state.generation.wrapping_add(1);
        state.now_playing = now_playing;
        self.generation.store(state.generation, Ordering::Release);
    }

    fn filter(&self) -> SourceFilter {
        *self
            .filter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Handle to the media worker. Creating one never blocks; dropping it stops the worker.
#[derive(Debug)]
pub struct MediaHub {
    shared: Arc<Shared>,
    commands: mpsc::Sender<Msg>,
}

impl MediaHub {
    /// Starts following media sessions that match `filter` on a background thread.
    ///
    /// On Windows this spawns the worker (named `mc-media`), which connects to the system media
    /// sessions on its own; the state stays empty until it has read something. Elsewhere there
    /// is no source: the state stays empty and commands are dropped.
    #[must_use]
    pub fn start(filter: SourceFilter) -> Self {
        let shared = Arc::new(Shared::default());
        *shared
            .filter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = filter;
        let (commands, receiver) = mpsc::channel();
        #[cfg(windows)]
        worker::spawn(
            Arc::clone(&shared),
            commands.clone(),
            receiver,
            gsmtc::WindowsBackend::new,
        );
        #[cfg(not(windows))]
        drop(receiver);
        Self { shared, commands }
    }

    /// A hub whose worker runs `make_backend`'s backend: the real loop over a mock source.
    #[cfg(test)]
    fn with_backend<B, F>(filter: SourceFilter, make_backend: F) -> Self
    where
        B: worker::Backend,
        F: FnOnce(worker::Waker) -> Option<B> + Send + 'static,
    {
        let shared = Arc::new(Shared::default());
        *shared
            .filter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = filter;
        let (commands, receiver) = mpsc::channel();
        worker::spawn(
            Arc::clone(&shared),
            commands.clone(),
            receiver,
            make_backend,
        );
        Self { shared, commands }
    }

    /// A hub fed by hand, for tests and for building the widget before the real source exists.
    #[must_use]
    pub fn fake() -> (Self, FakeControl) {
        let shared = Arc::new(Shared::default());
        let (commands, receiver) = mpsc::channel();
        let control = FakeControl {
            shared: Arc::clone(&shared),
            commands: Mutex::new(receiver),
        };
        (Self { shared, commands }, control)
    }

    /// Changes which sessions are followed. The worker picks it up within about 50 ms.
    pub fn set_filter(&self, filter: SourceFilter) {
        *self
            .shared
            .filter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = filter;
        let _ = self.commands.send(Msg::Wake);
    }

    /// The latest state: a short lock and an `Arc` clone.
    #[must_use]
    pub fn state(&self) -> MediaState {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The current generation, without locking or cloning.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(Ordering::Acquire)
    }

    /// Queues a command for the worker and returns at once.
    ///
    /// `SetVolume` levels are clamped to `0.0..=1.0` (a NaN is ignored), and a burst of them (a
    /// dragged slider) reaches Windows at most every 30 ms, latest value first.
    pub fn send(&self, command: MediaCommand) {
        // A stopped worker is not an error for the caller: the button just does nothing.
        let _ = self.commands.send(Msg::Command(command));
    }
}

impl Drop for MediaHub {
    /// Tells the worker to stop and returns at once; the worker exits after its current call.
    fn drop(&mut self) {
        let _ = self.commands.send(Msg::Stop);
    }
}

/// The feeding side of [`MediaHub::fake`].
#[derive(Debug)]
pub struct FakeControl {
    shared: Arc<Shared>,
    commands: Mutex<mpsc::Receiver<Msg>>,
}

impl FakeControl {
    /// Publishes a new snapshot (bumps the generation).
    pub fn set(&self, now_playing: Option<NowPlaying>) {
        self.shared.publish(now_playing.map(Arc::new));
    }

    /// The filter the hub was last given.
    #[must_use]
    pub fn filter(&self) -> SourceFilter {
        *self
            .shared
            .filter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Every command sent since the last call, oldest first.
    #[must_use]
    pub fn take_commands(&self) -> Vec<MediaCommand> {
        let receiver = self
            .commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        receiver
            .try_iter()
            .filter_map(|message| match message {
                Msg::Command(command) => Some(command),
                Msg::Wake | Msg::Stop => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn track(playing: bool, position_ms: u64, at: Instant) -> NowPlaying {
        NowPlaying {
            source_app: "Spotify.exe".to_owned(),
            title: "Title".to_owned(),
            artist: "Artist".to_owned(),
            album: String::new(),
            playing,
            position_ms: Some(position_ms),
            duration_ms: Some(10_000),
            sampled_at: at,
            volume: Some(0.5),
            art: None,
            art_generation: 0,
            can_play_pause: true,
            can_next: true,
            can_previous: true,
        }
    }

    #[test]
    fn fake_hub_publishes_snapshots_and_records_commands() {
        let (hub, control) = MediaHub::fake();
        assert_eq!(hub.generation(), 0);
        assert!(hub.state().now_playing.is_none());
        control.set(Some(track(true, 1_000, Instant::now())));
        assert_eq!(hub.generation(), 1);
        assert_eq!(hub.state().now_playing.unwrap().title, "Title");
        hub.send(MediaCommand::Next);
        hub.send(MediaCommand::SetVolume(0.25));
        assert_eq!(
            control.take_commands(),
            vec![MediaCommand::Next, MediaCommand::SetVolume(0.25)]
        );
        hub.set_filter(SourceFilter::Any);
        assert_eq!(control.filter(), SourceFilter::Any);
    }

    #[test]
    fn position_extrapolates_only_while_playing_and_clamps_to_the_duration() {
        let at = Instant::now();
        let later = at + Duration::from_millis(1_500);
        assert_eq!(track(true, 1_000, at).position_at(later), Some(2_500));
        assert_eq!(track(false, 1_000, at).position_at(later), Some(1_000));
        assert_eq!(track(true, 9_900, at).position_at(later), Some(10_000));
    }
}
