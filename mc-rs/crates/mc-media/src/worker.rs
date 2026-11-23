//! The media worker: one thread per real hub. It owns the platform backend, runs the commands the
//! game sends, refreshes when the backend reports a change, polls as a backstop, and publishes a
//! new snapshot only when something the widget shows has changed.
//!
//! Everything here is portable; the Windows specifics are behind [`Backend`]. The tests drive this
//! loop with a mock backend, so no test ever reaches a real media session.

use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use crate::{art, AlbumArt, MediaCommand, NowPlaying, Shared, SourceFilter};

/// Poll interval while a player is followed: the backstop for changes a player sends no event for.
pub(crate) const ACTIVE_POLL: Duration = Duration::from_secs(1);

/// Poll interval while no player is followed. The backend's session events wake the worker sooner
/// when a player appears.
pub(crate) const IDLE_POLL: Duration = Duration::from_secs(5);

/// The refresh runs this long after an OS event, so a burst of events costs one refresh.
const EVENT_SETTLE: Duration = Duration::from_millis(50);

/// After a transport command, refresh this soon even if the player sends no event for it.
const COMMAND_SETTLE: Duration = Duration::from_millis(300);

/// Volume changes reach Windows at most this often; a dragged slider may send one every frame.
const VOLUME_INTERVAL: Duration = Duration::from_millis(30);

/// After a track change the cover is read on this many refreshes: a player may attach the new
/// thumbnail a moment after the new title.
const COVER_READS: u8 = 3;

/// A position this far from the extrapolated one is a seek, and is published.
const POSITION_TOLERANCE_MS: u64 = 1_500;

/// Reads that fail this many times in a row clear the state (the player hangs or is gone).
const FAILURES_BEFORE_CLEAR: u32 = 3;

/// At most this many queued messages are handled before the worker refreshes again.
const BATCH: usize = 256;

thread_local! {
    static CONTAINING_PANIC: Cell<bool> = const { Cell::new(false) };
}

/// See [`crate::panic_is_being_contained`].
#[must_use]
pub(crate) fn panic_is_being_contained() -> bool {
    CONTAINING_PANIC.with(Cell::get)
}

/// What hub handles and OS event callbacks send to the worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Msg {
    /// A request from the game.
    Command(MediaCommand),
    /// Something may have changed (an OS event, or a new filter): refresh soon.
    Wake,
    /// The hub was dropped: stop.
    Stop,
}

/// Lets the backend's event callbacks (on OS thread-pool threads) wake the worker. At most one
/// wake is queued at a time, so a flood of events never floods the queue.
#[derive(Debug, Clone)]
pub(crate) struct Waker {
    sender: mpsc::Sender<Msg>,
    queued: Arc<AtomicBool>,
}

impl Waker {
    /// Asks for a refresh. Cheap, never blocks, and does nothing once the worker has stopped.
    pub(crate) fn wake(&self) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            let _ = self.sender.send(Msg::Wake);
        }
    }

    /// A waker for driving a backend by hand in a test, with the receiving end of its wakes.
    #[cfg(test)]
    pub(crate) fn detached() -> (Self, mpsc::Receiver<Msg>) {
        let (sender, receiver) = mpsc::channel();
        let waker = Self {
            sender,
            queued: Arc::new(AtomicBool::new(false)),
        };
        (waker, receiver)
    }
}

/// One reading of the followed player, before cover handling and change detection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Reading {
    pub(crate) source_app: String,
    pub(crate) title: String,
    pub(crate) artist: String,
    pub(crate) album: String,
    /// `None` while the player is between states (changing tracks): keep the last known one.
    pub(crate) playing: Option<bool>,
    /// The position at `sampled_at`.
    pub(crate) position_ms: Option<u64>,
    pub(crate) duration_ms: Option<u64>,
    pub(crate) sampled_at: Instant,
    pub(crate) can_play_pause: bool,
    pub(crate) can_next: bool,
    pub(crate) can_previous: bool,
    /// The player announced new media properties since the last reading, so the cover may have
    /// changed even if the title did not.
    pub(crate) media_changed: bool,
}

/// The result of one poll of the backend.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Poll {
    /// The followed player's state.
    Player(Reading),
    /// No player matches the filter (or the platform has no media sessions).
    NoPlayer,
    /// The player could not be read this time; keep what was published.
    Failed,
}

/// The platform side of the worker. Every method runs on the worker thread.
pub(crate) trait Backend {
    /// Finds the player `filter` selects and reads it.
    fn poll(&mut self, filter: SourceFilter) -> Poll;

    /// The encoded cover (JPEG or PNG) of the player last polled, if it has one.
    fn cover(&mut self) -> Option<Vec<u8>>;

    /// The audio-session volume of the player last polled, `0.0..=1.0`.
    fn volume(&mut self) -> Option<f32>;

    /// Sends `command` to the player last polled. `SetVolume` levels arrive finite and clamped.
    /// Returns whether the player accepted it.
    fn command(&mut self, command: MediaCommand) -> bool;

    /// How long to sleep between polls while no player is followed.
    fn idle_poll(&self) -> Duration {
        IDLE_POLL
    }
}

/// Starts the worker thread. `make_backend` runs on that thread (COM is per thread); `None` means
/// the platform has no source, and the hub stays empty.
pub(crate) fn spawn<B, F>(
    shared: Arc<Shared>,
    sender: mpsc::Sender<Msg>,
    receiver: mpsc::Receiver<Msg>,
    make_backend: F,
) where
    B: Backend,
    F: FnOnce(Waker) -> Option<B> + Send + 'static,
{
    let queued = Arc::new(AtomicBool::new(false));
    let waker = Waker {
        sender,
        queued: Arc::clone(&queued),
    };
    let spawned = std::thread::Builder::new()
        .name("mc-media".to_owned())
        .spawn(move || {
            // Every panic on this thread is contained: the widget goes blank, the game goes on.
            CONTAINING_PANIC.with(|flag| flag.set(true));
            let panic_shared = Arc::clone(&shared);
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(move || {
                if let Some(backend) = make_backend(waker) {
                    Worker::new(backend, shared, queued).run(&receiver);
                }
            }));
            if outcome.is_err() {
                tracing::warn!("the media worker stopped after an internal error");
                panic_shared.publish(None);
            }
        });
    if let Err(error) = spawned {
        tracing::warn!("the media worker could not start: {error}");
    }
}

/// The cover shown with the current snapshot.
#[derive(Debug, Default)]
struct Cover {
    /// The track the cover was last read for.
    track: Option<TrackKey>,
    /// Cover reads left for this track (see `COVER_READS`).
    reads_left: u8,
    /// Fingerprint of the encoded bytes behind `art` (or of the bytes that failed to decode).
    fingerprint: Option<u64>,
    art: Option<Arc<AlbumArt>>,
    /// Bumped whenever `art` changes; never reused.
    generation: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct TrackKey {
    source_app: String,
    title: String,
    artist: String,
    album: String,
}

impl TrackKey {
    fn of(reading: &Reading) -> Self {
        Self {
            source_app: reading.source_app.clone(),
            title: reading.title.clone(),
            artist: reading.artist.clone(),
            album: reading.album.clone(),
        }
    }

    fn matches(&self, reading: &Reading) -> bool {
        self.source_app == reading.source_app
            && self.title == reading.title
            && self.artist == reading.artist
            && self.album == reading.album
    }
}

struct Worker<B> {
    backend: B,
    shared: Arc<Shared>,
    /// The waker's "a wake is queued" flag, cleared when the wake is handled.
    queued: Arc<AtomicBool>,
    published: Option<Arc<NowPlaying>>,
    /// The last known play state, kept while the player reports "changing".
    playing: bool,
    failures: u32,
    cover: Cover,
    /// The latest volume asked for and not yet applied (rate limit).
    pending_volume: Option<f32>,
    last_volume_set: Option<Instant>,
}

impl<B: Backend> Worker<B> {
    fn new(backend: B, shared: Arc<Shared>, queued: Arc<AtomicBool>) -> Self {
        Self {
            backend,
            shared,
            queued,
            published: None,
            playing: false,
            failures: 0,
            cover: Cover::default(),
            pending_volume: None,
            last_volume_set: None,
        }
    }

    fn run(mut self, receiver: &mpsc::Receiver<Msg>) {
        let mut next_refresh = Instant::now();
        loop {
            let deadline = self
                .volume_due()
                .map_or(next_refresh, |due| due.min(next_refresh));
            match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(first) => {
                    // Take what is queued too, so a burst (a dragged slider) costs one pass.
                    let queued: Vec<Msg> = receiver.try_iter().take(BATCH).collect();
                    for message in std::iter::once(first).chain(queued) {
                        if !self.handle(message, &mut next_refresh) {
                            return;
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            let now = Instant::now();
            self.apply_volume(now);
            if now >= next_refresh {
                let interval = self.refresh();
                next_refresh = Instant::now() + interval;
            }
        }
    }

    /// Handles one message; `false` means stop.
    fn handle(&mut self, message: Msg, next_refresh: &mut Instant) -> bool {
        match message {
            Msg::Stop => return false,
            Msg::Wake => {
                self.queued.store(false, Ordering::Release);
                *next_refresh = (*next_refresh).min(Instant::now() + EVENT_SETTLE);
            }
            Msg::Command(MediaCommand::SetVolume(level)) => {
                // A NaN never reaches Windows; anything else is clamped to the valid range.
                if level.is_finite() {
                    self.pending_volume = Some(level.clamp(0.0, 1.0));
                }
            }
            Msg::Command(command) => {
                let accepted = self.backend.command(command);
                tracing::debug!(?command, accepted, "media command");
                *next_refresh = (*next_refresh).min(Instant::now() + COMMAND_SETTLE);
            }
        }
        true
    }

    /// When the pending volume may be applied.
    fn volume_due(&self) -> Option<Instant> {
        self.pending_volume?;
        Some(
            self.last_volume_set
                .map_or_else(Instant::now, |at| at + VOLUME_INTERVAL),
        )
    }

    fn apply_volume(&mut self, now: Instant) {
        let Some(level) = self.pending_volume else {
            return;
        };
        if self
            .last_volume_set
            .is_some_and(|at| now < at + VOLUME_INTERVAL)
        {
            return;
        }
        self.pending_volume = None;
        self.last_volume_set = Some(now);
        if self.backend.command(MediaCommand::SetVolume(level)) {
            // Show the new level now rather than on the next poll.
            self.republish_volume();
        }
    }

    fn republish_volume(&mut self) {
        let Some(current) = self.published.clone() else {
            return;
        };
        let volume = self.backend.volume();
        if volume_moved(current.volume, volume) {
            let mut next = NowPlaying::clone(&current);
            next.volume = volume;
            self.publish(Some(next));
        }
    }

    /// Polls the backend and publishes what changed. Returns how long until the next poll.
    fn refresh(&mut self) -> Duration {
        match self.backend.poll(self.shared.filter()) {
            Poll::Player(reading) => {
                self.failures = 0;
                self.show(reading);
                ACTIVE_POLL
            }
            Poll::NoPlayer => {
                self.failures = 0;
                self.clear();
                self.backend.idle_poll()
            }
            Poll::Failed => {
                self.failures = self.failures.saturating_add(1);
                if self.failures >= FAILURES_BEFORE_CLEAR {
                    self.clear();
                }
                ACTIVE_POLL
            }
        }
    }

    fn show(&mut self, reading: Reading) {
        if let Some(playing) = reading.playing {
            self.playing = playing;
        }
        self.update_cover(&reading);
        let next = NowPlaying {
            source_app: reading.source_app,
            title: reading.title,
            artist: reading.artist,
            album: reading.album,
            playing: self.playing,
            position_ms: reading.position_ms,
            duration_ms: reading.duration_ms,
            sampled_at: reading.sampled_at,
            volume: self.backend.volume(),
            art: self.cover.art.clone(),
            art_generation: self.cover.generation,
            can_play_pause: reading.can_play_pause,
            can_next: reading.can_next,
            can_previous: reading.can_previous,
        };
        if self
            .published
            .as_deref()
            .is_none_or(|shown| differs(shown, &next))
        {
            self.publish(Some(next));
        }
    }

    fn update_cover(&mut self, reading: &Reading) {
        let same_track = self
            .cover
            .track
            .as_ref()
            .is_some_and(|track| track.matches(reading));
        if reading.media_changed || !same_track {
            self.cover.track = Some(TrackKey::of(reading));
            self.cover.reads_left = COVER_READS;
        }
        if self.cover.reads_left == 0 {
            return;
        }
        self.cover.reads_left -= 1;
        let bytes = self.backend.cover();
        let fingerprint = bytes.as_deref().map(art::fingerprint);
        if fingerprint == self.cover.fingerprint {
            // The same bytes as the cover shown, or still no cover.
            return;
        }
        self.cover.fingerprint = fingerprint;
        let art = bytes.as_deref().and_then(art::decode_cover).map(Arc::new);
        if art.is_none() && self.cover.art.is_none() {
            return;
        }
        self.cover.art = art;
        self.cover.generation = self.cover.generation.wrapping_add(1);
    }

    /// No player: forget the cover and publish `None` once.
    fn clear(&mut self) {
        self.cover.track = None;
        self.cover.reads_left = 0;
        self.cover.fingerprint = None;
        if self.cover.art.take().is_some() {
            self.cover.generation = self.cover.generation.wrapping_add(1);
        }
        if self.published.is_some() {
            self.publish(None);
        }
    }

    fn publish(&mut self, now_playing: Option<NowPlaying>) {
        let now_playing = now_playing.map(Arc::new);
        self.published.clone_from(&now_playing);
        self.shared.publish(now_playing);
    }
}

/// Whether `next` shows something `shown` does not: any text, flag, cover or volume change, or a
/// position that moved other than by playing on.
fn differs(shown: &NowPlaying, next: &NowPlaying) -> bool {
    shown.source_app != next.source_app
        || shown.title != next.title
        || shown.artist != next.artist
        || shown.album != next.album
        || shown.playing != next.playing
        || shown.duration_ms != next.duration_ms
        || shown.can_play_pause != next.can_play_pause
        || shown.can_next != next.can_next
        || shown.can_previous != next.can_previous
        || shown.art_generation != next.art_generation
        || volume_moved(shown.volume, next.volume)
        || position_jumped(shown, next)
}

fn position_jumped(shown: &NowPlaying, next: &NowPlaying) -> bool {
    match (shown.position_at(next.sampled_at), next.position_ms) {
        (Some(expected), Some(actual)) => expected.abs_diff(actual) > POSITION_TOLERANCE_MS,
        (None, None) => false,
        _ => true,
    }
}

fn volume_moved(shown: Option<f32>, next: Option<f32>) -> bool {
    match (shown, next) {
        (Some(shown), Some(next)) => (shown - next).abs() > 0.001,
        (None, None) => false,
        _ => true,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::MediaHub;
    use std::io::Cursor;
    use std::sync::Mutex;

    /// What the mock backend reports, and what it was asked to do.
    #[derive(Debug)]
    pub(crate) struct Script {
        pub(crate) poll: Poll,
        pub(crate) cover: Option<Vec<u8>>,
        pub(crate) volume: Option<f32>,
        pub(crate) accept: bool,
        pub(crate) commands: Vec<MediaCommand>,
        pub(crate) command_threads: Vec<Option<String>>,
        pub(crate) filters: Vec<SourceFilter>,
        pub(crate) cover_reads: usize,
        pub(crate) panic_on_poll: bool,
        /// Each poll takes this long (a slow player), without holding the script.
        pub(crate) poll_delay: Option<Duration>,
        pub(crate) contained_while_polling: Option<bool>,
        pub(crate) dropped: bool,
    }

    impl Default for Script {
        fn default() -> Self {
            Self {
                poll: Poll::NoPlayer,
                cover: None,
                volume: None,
                accept: true,
                commands: Vec::new(),
                command_threads: Vec::new(),
                filters: Vec::new(),
                cover_reads: 0,
                panic_on_poll: false,
                poll_delay: None,
                contained_while_polling: None,
                dropped: false,
            }
        }
    }

    pub(crate) struct MockBackend(pub(crate) Arc<Mutex<Script>>);

    impl MockBackend {
        fn script(&self) -> std::sync::MutexGuard<'_, Script> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }
    }

    impl Backend for MockBackend {
        fn poll(&mut self, filter: SourceFilter) -> Poll {
            let delay = {
                let mut script = self.script();
                script.filters.push(filter);
                script.contained_while_polling = Some(panic_is_being_contained());
                if script.panic_on_poll {
                    drop(script);
                    panic!("simulated backend bug (expected in this test)");
                }
                script.poll_delay
            };
            if let Some(delay) = delay {
                std::thread::sleep(delay);
            }
            self.script().poll.clone()
        }

        fn cover(&mut self) -> Option<Vec<u8>> {
            let mut script = self.script();
            script.cover_reads += 1;
            script.cover.clone()
        }

        fn volume(&mut self) -> Option<f32> {
            self.script().volume
        }

        fn command(&mut self, command: MediaCommand) -> bool {
            let mut script = self.script();
            script.commands.push(command);
            let thread = std::thread::current().name().map(str::to_owned);
            script.command_threads.push(thread);
            if let (MediaCommand::SetVolume(level), true) = (command, script.accept) {
                script.volume = Some(level);
            }
            script.accept
        }
    }

    impl Drop for MockBackend {
        fn drop(&mut self) {
            self.script().dropped = true;
        }
    }

    pub(crate) fn reading(title: &str, playing: bool, position_ms: u64, at: Instant) -> Reading {
        Reading {
            source_app: "Spotify.exe".to_owned(),
            title: title.to_owned(),
            artist: "Artist".to_owned(),
            album: "Album".to_owned(),
            playing: Some(playing),
            position_ms: Some(position_ms),
            duration_ms: Some(200_000),
            sampled_at: at,
            can_play_pause: true,
            can_next: true,
            can_previous: true,
            media_changed: false,
        }
    }

    fn cover_png(shade: u8) -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(300, 300, image::Rgba([shade, 0, 0, 255]));
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode test PNG");
        bytes
    }

    fn worker(script: &Arc<Mutex<Script>>) -> (Worker<MockBackend>, Arc<Shared>) {
        let shared = Arc::new(Shared::default());
        let worker = Worker::new(
            MockBackend(Arc::clone(script)),
            Arc::clone(&shared),
            Arc::new(AtomicBool::new(false)),
        );
        (worker, shared)
    }

    fn set(script: &Arc<Mutex<Script>>, edit: impl FnOnce(&mut Script)) {
        edit(&mut script.lock().unwrap());
    }

    fn generation(shared: &Shared) -> u64 {
        shared.generation.load(Ordering::Acquire)
    }

    fn shown(shared: &Shared) -> Option<Arc<NowPlaying>> {
        shared.state.lock().unwrap().now_playing.clone()
    }

    /// Waits up to 10 s (the machine may be busy with other builds) for `done`.
    pub(crate) fn eventually(mut done: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        done()
    }

    #[test]
    fn a_snapshot_is_published_only_when_something_shown_changes() {
        let script = Arc::new(Mutex::new(Script::default()));
        let (mut worker, shared) = worker(&script);
        let t0 = Instant::now();
        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 1_000, t0));
            s.volume = Some(0.5);
        });
        assert_eq!(worker.refresh(), ACTIVE_POLL);
        assert_eq!(generation(&shared), 1);
        assert_eq!(shown(&shared).unwrap().title, "One");

        // Playing on: two seconds later the position is two seconds further. Nothing to publish.
        let t1 = t0 + Duration::from_secs(2);
        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 3_000, t1))
        });
        worker.refresh();
        assert_eq!(
            generation(&shared),
            1,
            "a position that only played on is not news"
        );

        set(&script, |s| s.volume = Some(0.5004));
        worker.refresh();
        assert_eq!(generation(&shared), 1, "volume noise is not news");

        set(&script, |s| s.volume = Some(0.6));
        worker.refresh();
        assert_eq!(generation(&shared), 2, "a volume change is");
        assert_eq!(shown(&shared).unwrap().volume, Some(0.6));

        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 60_000, t1))
        });
        worker.refresh();
        assert_eq!(generation(&shared), 3, "a seek is");

        set(&script, |s| {
            s.poll = Poll::Player(reading("One", false, 60_000, t1))
        });
        worker.refresh();
        assert_eq!(generation(&shared), 4, "a pause is");
        assert!(!shown(&shared).unwrap().playing);

        set(&script, |s| {
            s.poll = Poll::Player(reading("Two", false, 0, t1))
        });
        worker.refresh();
        assert_eq!(generation(&shared), 5, "a new track is");
        assert_eq!(shown(&shared).unwrap().title, "Two");
    }

    #[test]
    fn a_changing_status_keeps_the_last_play_state() {
        let script = Arc::new(Mutex::new(Script::default()));
        let (mut worker, shared) = worker(&script);
        let now = Instant::now();
        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 0, now))
        });
        worker.refresh();
        let mut changing = reading("One", true, 0, now);
        changing.playing = None;
        set(&script, |s| s.poll = Poll::Player(changing));
        worker.refresh();
        assert!(shown(&shared).unwrap().playing);
        set(&script, |s| {
            s.poll = Poll::Player(reading("One", false, 0, now))
        });
        worker.refresh();
        assert!(!shown(&shared).unwrap().playing);
    }

    #[test]
    fn the_art_generation_moves_only_when_the_cover_does() {
        let script = Arc::new(Mutex::new(Script::default()));
        let (mut worker, shared) = worker(&script);
        let now = Instant::now();
        let (a, b, c) = (cover_png(10), cover_png(20), cover_png(30));
        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 0, now));
            s.cover = Some(a.clone());
        });
        worker.refresh();
        let first = shown(&shared).unwrap();
        let art = first.art.clone().expect("cover decoded");
        assert_eq!((art.width, art.height), (128, 128), "shrunk from 300x300");
        assert_eq!(first.art_generation, 1);

        // The cover is read a few times after a track change, then left alone.
        for _ in 0..5 {
            worker.refresh();
        }
        assert_eq!(script.lock().unwrap().cover_reads, COVER_READS as usize);
        assert_eq!(shown(&shared).unwrap().art_generation, 1);

        // Next track, same album cover: new title, same art and generation, no new decode.
        set(&script, |s| {
            s.poll = Poll::Player(reading("Two", true, 0, now))
        });
        worker.refresh();
        let second = shown(&shared).unwrap();
        assert_eq!(second.title, "Two");
        assert_eq!(second.art_generation, 1);
        assert!(
            Arc::ptr_eq(second.art.as_ref().unwrap(), &art),
            "the decoded cover is reused"
        );

        set(&script, |s| {
            s.poll = Poll::Player(reading("Three", true, 0, now));
            s.cover = Some(b.clone());
        });
        worker.refresh();
        assert_eq!(shown(&shared).unwrap().art_generation, 2, "a new cover");

        set(&script, |s| {
            s.poll = Poll::Player(reading("Four", true, 0, now));
            s.cover = None;
        });
        worker.refresh();
        let no_cover = shown(&shared).unwrap();
        assert!(no_cover.art.is_none());
        assert_eq!(
            no_cover.art_generation, 3,
            "losing the cover is a change too"
        );

        set(&script, |s| {
            s.poll = Poll::Player(reading("Five", true, 0, now))
        });
        worker.refresh();
        assert_eq!(shown(&shared).unwrap().art_generation, 3, "still no cover");

        // The player swaps the thumbnail without a title change, and says so.
        let mut announced = reading("Five", true, 0, now);
        announced.media_changed = true;
        set(&script, |s| {
            s.poll = Poll::Player(announced);
            s.cover = Some(c.clone());
        });
        worker.refresh();
        assert_eq!(shown(&shared).unwrap().art_generation, 4);

        // Undecodable bytes give no cover (and one change), and are not decoded again.
        set(&script, |s| {
            s.poll = Poll::Player(reading("Six", true, 0, now));
            s.cover = Some(b"not an image".to_vec());
        });
        worker.refresh();
        worker.refresh();
        let broken = shown(&shared).unwrap();
        assert!(broken.art.is_none());
        assert_eq!(broken.art_generation, 5);
    }

    #[test]
    fn no_player_clears_the_state_and_repeated_failures_clear_it_too() {
        let script = Arc::new(Mutex::new(Script::default()));
        let (mut worker, shared) = worker(&script);
        let now = Instant::now();
        assert_eq!(
            worker.refresh(),
            IDLE_POLL,
            "idles while there is nothing to follow"
        );
        assert_eq!(
            generation(&shared),
            0,
            "nothing was ever shown, nothing to clear"
        );

        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 0, now));
            s.cover = Some(cover_png(1));
        });
        worker.refresh();
        let generation_before = shown(&shared).unwrap().art_generation;
        set(&script, |s| s.poll = Poll::NoPlayer);
        assert_eq!(worker.refresh(), IDLE_POLL);
        assert!(shown(&shared).is_none(), "the player closed");
        let cleared_at = generation(&shared);
        worker.refresh();
        assert_eq!(generation(&shared), cleared_at, "cleared once");

        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 0, now))
        });
        worker.refresh();
        let back = shown(&shared).unwrap();
        assert!(
            back.art.is_some(),
            "the cover is read again for a returning player"
        );
        assert!(
            back.art_generation > generation_before,
            "generations are never reused"
        );

        set(&script, |s| s.poll = Poll::Failed);
        worker.refresh();
        worker.refresh();
        assert!(shown(&shared).is_some(), "a hiccup keeps the last state");
        worker.refresh();
        assert!(
            shown(&shared).is_none(),
            "a player that stays unreadable is dropped"
        );
    }

    #[test]
    fn volume_requests_are_sanitised_and_rate_limited() {
        let script = Arc::new(Mutex::new(Script::default()));
        let (mut worker, shared) = worker(&script);
        let now = Instant::now();
        set(&script, |s| {
            s.poll = Poll::Player(reading("One", true, 0, now));
            s.volume = Some(0.5);
        });
        worker.refresh();
        let mut next_refresh = now + ACTIVE_POLL;

        assert!(worker.handle(
            Msg::Command(MediaCommand::SetVolume(1.5)),
            &mut next_refresh
        ));
        assert!(worker.handle(
            Msg::Command(MediaCommand::SetVolume(f32::NAN)),
            &mut next_refresh
        ));
        assert_eq!(
            worker.pending_volume,
            Some(1.0),
            "clamped, and the NaN dropped"
        );
        worker.apply_volume(now);
        assert_eq!(
            script.lock().unwrap().commands,
            vec![MediaCommand::SetVolume(1.0)]
        );
        assert_eq!(
            shown(&shared).unwrap().volume,
            Some(1.0),
            "republished at once"
        );

        // A burst inside the rate-limit window: only the last level is applied, once, later.
        for level in [0.9, 0.8, -3.0] {
            worker.handle(
                Msg::Command(MediaCommand::SetVolume(level)),
                &mut next_refresh,
            );
        }
        worker.apply_volume(now + Duration::from_millis(10));
        assert_eq!(script.lock().unwrap().commands.len(), 1, "too soon");
        assert_eq!(worker.volume_due(), Some(now + VOLUME_INTERVAL));
        worker.apply_volume(now + VOLUME_INTERVAL);
        assert_eq!(
            script.lock().unwrap().commands,
            vec![MediaCommand::SetVolume(1.0), MediaCommand::SetVolume(0.0)]
        );
        assert_eq!(worker.volume_due(), None);
        assert_eq!(shown(&shared).unwrap().volume, Some(0.0));
    }

    #[test]
    fn transport_commands_run_and_bring_the_next_refresh_forward() {
        let script = Arc::new(Mutex::new(Script::default()));
        let (mut worker, _shared) = worker(&script);
        let far = Instant::now() + Duration::from_secs(60);
        let mut next_refresh = far;
        for command in [
            MediaCommand::PlayPause,
            MediaCommand::Next,
            MediaCommand::Previous,
        ] {
            assert!(worker.handle(Msg::Command(command), &mut next_refresh));
        }
        assert_eq!(
            script.lock().unwrap().commands,
            vec![
                MediaCommand::PlayPause,
                MediaCommand::Next,
                MediaCommand::Previous
            ]
        );
        assert!(next_refresh < far);
        assert!(!worker.handle(Msg::Stop, &mut next_refresh));
    }

    #[test]
    fn hub_commands_reach_the_backend_on_the_worker_thread_in_order() {
        let script = Arc::new(Mutex::new(Script::default()));
        let backend_script = Arc::clone(&script);
        let hub = MediaHub::with_backend(SourceFilter::Spotify, move |_| {
            Some(MockBackend(backend_script))
        });
        hub.send(MediaCommand::PlayPause);
        hub.send(MediaCommand::Next);
        hub.send(MediaCommand::Previous);
        hub.send(MediaCommand::SetVolume(0.25));
        assert!(eventually(|| script.lock().unwrap().commands.len() == 4));
        let script = script.lock().unwrap();
        assert_eq!(
            script.commands,
            vec![
                MediaCommand::PlayPause,
                MediaCommand::Next,
                MediaCommand::Previous,
                MediaCommand::SetVolume(0.25),
            ]
        );
        assert!(script
            .command_threads
            .iter()
            .all(|t| t.as_deref() == Some("mc-media")));
    }

    #[test]
    fn a_new_filter_reaches_the_backend_promptly() {
        let script = Arc::new(Mutex::new(Script::default()));
        let backend_script = Arc::clone(&script);
        let hub = MediaHub::with_backend(SourceFilter::Spotify, move |_| {
            Some(MockBackend(backend_script))
        });
        assert!(eventually(|| !script.lock().unwrap().filters.is_empty()));
        // The idle poll is 5 s; the filter change wakes the worker long before that.
        let changed_at = Instant::now();
        hub.set_filter(SourceFilter::Any);
        assert!(eventually(
            || script.lock().unwrap().filters.last() == Some(&SourceFilter::Any)
        ));
        assert!(changed_at.elapsed() < IDLE_POLL);
    }

    #[test]
    fn backend_events_wake_the_worker() {
        let script = Arc::new(Mutex::new(Script::default()));
        let backend_script = Arc::clone(&script);
        let (waker_tx, waker_rx) = mpsc::channel();
        let hub = MediaHub::with_backend(SourceFilter::Spotify, move |waker| {
            let _ = waker_tx.send(waker);
            Some(MockBackend(backend_script))
        });
        let waker: Waker = waker_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("worker up");
        assert!(eventually(|| !script.lock().unwrap().filters.is_empty()));
        assert!(hub.state().now_playing.is_none());
        set(&script, |s| {
            s.poll = Poll::Player(reading("Woken", true, 0, Instant::now()))
        });
        // Many events at once (as a player sends on a track change) are one refresh.
        for _ in 0..50 {
            waker.wake();
        }
        assert!(eventually(|| hub.generation() == 1));
        assert_eq!(hub.state().now_playing.unwrap().title, "Woken");
        // Normally 2 polls (start, then one for the burst); generous for a loaded machine.
        let polls = script.lock().unwrap().filters.len();
        assert!(
            polls < 10,
            "50 events coalesce into a few refreshes, got {polls}"
        );
    }

    #[test]
    fn dropping_the_hub_never_waits_for_a_busy_worker() {
        // The worker is stuck in a slow 2 s poll (a hung player) when the hub is dropped.
        let script = Arc::new(Mutex::new(Script {
            poll_delay: Some(Duration::from_secs(2)),
            ..Script::default()
        }));
        let backend_script = Arc::clone(&script);
        let hub = MediaHub::with_backend(SourceFilter::Spotify, move |_| {
            Some(MockBackend(backend_script))
        });
        assert!(
            eventually(|| !script.lock().unwrap().filters.is_empty()),
            "poll started"
        );
        let started = Instant::now();
        drop(hub);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "drop returned at once"
        );
        assert!(
            eventually(|| script.lock().unwrap().dropped),
            "the worker exits after its current call"
        );
    }

    #[test]
    fn a_panic_on_the_worker_is_contained_and_clears_the_state() {
        assert!(!panic_is_being_contained(), "not on the test thread");
        let script = Arc::new(Mutex::new(Script {
            panic_on_poll: true,
            ..Script::default()
        }));
        let backend_script = Arc::clone(&script);
        let hub = MediaHub::with_backend(SourceFilter::Spotify, move |_| {
            Some(MockBackend(backend_script))
        });
        // Unwinding drops the backend, so `dropped` means the thread has finished.
        assert!(eventually(|| script.lock().unwrap().dropped));
        assert_eq!(script.lock().unwrap().contained_while_polling, Some(true));
        assert!(hub.state().now_playing.is_none());
        hub.send(MediaCommand::Next);
        assert!(
            script.lock().unwrap().commands.is_empty(),
            "a dead worker ignores commands"
        );
    }
}
