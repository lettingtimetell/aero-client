//! The Windows source, in safe Rust over the WinRT bindings: the system media session
//! (`GlobalSystemMediaTransportControlsSessionManager`, the same source as the media controls in
//! the volume flyout) gives the track, timeline, play state, enabled controls and cover. It needs
//! no login or API key. The volume comes from Core Audio in [`crate::ffi`].
//!
//! Every call here runs on the media worker thread, which is in the multithreaded apartment.
//! WinRT async calls are waited for with a timeout, so a hung player can only make the widget's
//! state stale, never stall the worker for long.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use windows::core::{RuntimeType, HSTRING};
use windows::Foundation::TypedEventHandler;
use windows::Media::Control::{
    CurrentSessionChangedEventArgs, GlobalSystemMediaTransportControlsSession as Session,
    GlobalSystemMediaTransportControlsSessionManager as Manager,
    GlobalSystemMediaTransportControlsSessionMediaProperties as MediaProperties,
    GlobalSystemMediaTransportControlsSessionPlaybackStatus as Status,
    MediaPropertiesChangedEventArgs, PlaybackInfoChangedEventArgs, SessionsChangedEventArgs,
    TimelinePropertiesChangedEventArgs,
};
use windows::Storage::Streams::{DataReader, IRandomAccessStreamReference};
use windows::Win32::Foundation::ERROR_TIMEOUT;
use windows_future::{AsyncOperationCompletedHandler, AsyncStatus, IAsyncOperation};

use crate::ffi::{ComApartment, VolumeFinder};
use crate::timeline::{filetime_now, Timeline};
use crate::worker::{Backend, Poll, Reading, Waker};
use crate::{apps, art, MediaCommand, SourceFilter};

/// The longest a single WinRT call may take before it is abandoned.
const CALL_TIMEOUT: Duration = Duration::from_secs(2);

/// How long to wait before asking for the session manager again after it was unavailable.
const MANAGER_RETRY: Duration = Duration::from_secs(30);

/// The Windows backend. Created on the worker thread, which it never leaves.
pub(crate) struct WindowsBackend {
    waker: Waker,
    link: Option<ManagerLink>,
    /// When to ask for the session manager again after a failure.
    retry_at: Option<Instant>,
    warned_unavailable: bool,
    followed: Option<Followed>,
    volume: VolumeFinder,
    /// Declared last: fields drop in declaration order, so every COM object above is released
    /// before the thread leaves the apartment.
    _apartment: ComApartment,
}

impl WindowsBackend {
    /// Enters the COM apartment for this (worker) thread. The session manager is requested on
    /// the first poll, so this never blocks.
    pub(crate) fn new(waker: Waker) -> Option<Self> {
        let apartment = ComApartment::enter_mta();
        Some(Self {
            waker,
            link: None,
            retry_at: None,
            warned_unavailable: false,
            followed: None,
            volume: VolumeFinder::default(),
            _apartment: apartment,
        })
    }

    /// Makes sure the session manager is there. `false` while it is unavailable.
    fn connect(&mut self) -> bool {
        if self.link.is_some() {
            return true;
        }
        let now = Instant::now();
        if self.retry_at.is_some_and(|at| now < at) {
            return false;
        }
        match Manager::RequestAsync().and_then(|request| wait(&request, CALL_TIMEOUT)) {
            Ok(manager) => {
                self.link = Some(ManagerLink::attach(manager, &self.waker));
                self.retry_at = None;
                true
            }
            Err(error) => {
                if self.warned_unavailable {
                    tracing::debug!("Windows media sessions still unavailable: {error}");
                } else {
                    tracing::warn!("Windows media sessions unavailable ({error}); retrying");
                    self.warned_unavailable = true;
                }
                self.retry_at = Some(now + MANAGER_RETRY);
                false
            }
        }
    }
}

impl Backend for WindowsBackend {
    fn poll(&mut self, filter: SourceFilter) -> Poll {
        if !self.connect() {
            return Poll::NoPlayer;
        }
        let Some(link) = &self.link else {
            return Poll::NoPlayer;
        };
        // Choose the session afresh when the session list or the filter changed. Otherwise keep
        // the one followed, with its event subscriptions.
        let sessions_changed = link.sessions_changed.swap(false, Ordering::AcqRel);
        if sessions_changed || self.followed.as_ref().is_some_and(|f| f.filter != filter) {
            self.followed = None;
        }
        if self.followed.is_none() {
            match pick(&link.manager, filter) {
                Ok(Some((session, app_id))) => {
                    tracing::debug!("following the media session of {app_id}");
                    self.followed = Some(Followed::attach(session, app_id, filter, &self.waker));
                }
                Ok(None) => return Poll::NoPlayer,
                Err(error) => {
                    // The manager itself failed (the media service restarted?): start over.
                    tracing::debug!("media session list failed: {error}");
                    self.link = None;
                    return Poll::Failed;
                }
            }
        }
        let Some(followed) = self.followed.as_mut() else {
            return Poll::NoPlayer;
        };
        match followed.read() {
            Ok(reading) => Poll::Player(reading),
            Err(error) => {
                // The session may have gone with its player: choose afresh next time.
                tracing::debug!("media session read failed: {error}");
                self.followed = None;
                Poll::Failed
            }
        }
    }

    fn cover(&mut self) -> Option<Vec<u8>> {
        let properties = self.followed.as_ref()?.properties.as_ref()?;
        // Fails when the player gives no thumbnail (a null reference).
        let thumbnail = properties.Thumbnail().ok()?;
        match read_all(&thumbnail) {
            Ok(bytes) if !bytes.is_empty() => Some(bytes),
            Ok(_) => None,
            Err(error) => {
                tracing::debug!("cover read failed: {error}");
                None
            }
        }
    }

    fn volume(&mut self) -> Option<f32> {
        let followed = self.followed.as_mut()?;
        if std::mem::take(&mut followed.nudge_volume) {
            self.volume.nudge();
        }
        self.volume.level(&followed.app_id)
    }

    fn command(&mut self, command: MediaCommand) -> bool {
        let Some(followed) = self.followed.as_ref() else {
            return false;
        };
        let operation = match command {
            MediaCommand::SetVolume(level) => {
                return self.volume.set_level(&followed.app_id, level)
            }
            MediaCommand::PlayPause => followed.play_pause(),
            MediaCommand::Next => followed.session.TrySkipNextAsync(),
            MediaCommand::Previous => followed.session.TrySkipPreviousAsync(),
        };
        operation
            .and_then(|operation| wait(&operation, CALL_TIMEOUT))
            .unwrap_or(false)
    }
}

/// The session manager and its two subscriptions.
struct ManagerLink {
    manager: Manager,
    /// Set by the subscriptions when the session list or the current session changes.
    sessions_changed: Arc<AtomicBool>,
    sessions_token: Option<i64>,
    current_token: Option<i64>,
}

impl ManagerLink {
    fn attach(manager: Manager, waker: &Waker) -> Self {
        let sessions_changed = Arc::new(AtomicBool::new(false));
        let sessions_token = {
            let flag = Arc::clone(&sessions_changed);
            let waker = waker.clone();
            let handler =
                TypedEventHandler::<Manager, SessionsChangedEventArgs>::new(move |_, _| {
                    flag.store(true, Ordering::Release);
                    waker.wake();
                    Ok(())
                });
            manager.SessionsChanged(&handler).ok()
        };
        let current_token = {
            let flag = Arc::clone(&sessions_changed);
            let waker = waker.clone();
            let handler =
                TypedEventHandler::<Manager, CurrentSessionChangedEventArgs>::new(move |_, _| {
                    flag.store(true, Ordering::Release);
                    waker.wake();
                    Ok(())
                });
            manager.CurrentSessionChanged(&handler).ok()
        };
        Self {
            manager,
            sessions_changed,
            sessions_token,
            current_token,
        }
    }
}

impl Drop for ManagerLink {
    fn drop(&mut self) {
        if let Some(token) = self.sessions_token {
            let _ = self.manager.RemoveSessionsChanged(token);
        }
        if let Some(token) = self.current_token {
            let _ = self.manager.RemoveCurrentSessionChanged(token);
        }
    }
}

/// Which transport controls the player enabled at the last read.
#[derive(Debug, Clone, Copy, Default)]
struct Controls {
    toggle: bool,
    play: bool,
    pause: bool,
}

/// The followed session, its subscriptions, and what the last read found.
struct Followed {
    session: Session,
    app_id: String,
    filter: SourceFilter,
    /// Set by the media-properties subscription.
    media_changed: Arc<AtomicBool>,
    media_token: Option<i64>,
    playback_token: Option<i64>,
    timeline_token: Option<i64>,
    /// The properties of the last read, kept for the cover.
    properties: Option<MediaProperties>,
    controls: Controls,
    playing: bool,
    /// The player started playing or announced new media since the volume was last read: a
    /// moment when it may have gained an audio session.
    nudge_volume: bool,
}

impl Followed {
    fn attach(session: Session, app_id: String, filter: SourceFilter, waker: &Waker) -> Self {
        let media_changed = Arc::new(AtomicBool::new(true));
        let media_token = {
            let flag = Arc::clone(&media_changed);
            let waker = waker.clone();
            let handler =
                TypedEventHandler::<Session, MediaPropertiesChangedEventArgs>::new(move |_, _| {
                    flag.store(true, Ordering::Release);
                    waker.wake();
                    Ok(())
                });
            session.MediaPropertiesChanged(&handler).ok()
        };
        let playback_token = {
            let waker = waker.clone();
            let handler =
                TypedEventHandler::<Session, PlaybackInfoChangedEventArgs>::new(move |_, _| {
                    waker.wake();
                    Ok(())
                });
            session.PlaybackInfoChanged(&handler).ok()
        };
        let timeline_token = {
            let waker = waker.clone();
            let handler = TypedEventHandler::<Session, TimelinePropertiesChangedEventArgs>::new(
                move |_, _| {
                    waker.wake();
                    Ok(())
                },
            );
            session.TimelinePropertiesChanged(&handler).ok()
        };
        Self {
            session,
            app_id,
            filter,
            media_changed,
            media_token,
            playback_token,
            timeline_token,
            properties: None,
            controls: Controls::default(),
            playing: false,
            nudge_volume: false,
        }
    }

    fn read(&mut self) -> windows::core::Result<Reading> {
        let properties = wait(&self.session.TryGetMediaPropertiesAsync()?, CALL_TIMEOUT)?;
        let info = self.session.GetPlaybackInfo()?;
        let playing = match info.PlaybackStatus()? {
            Status::Playing => Some(true),
            // Between tracks: keep the last known state rather than flicker to "paused".
            Status::Changing => None,
            _ => Some(false),
        };
        if let Some(playing) = playing {
            self.nudge_volume |= playing && !self.playing;
            self.playing = playing;
        }
        let controls = info.Controls()?;
        self.controls = Controls {
            toggle: controls.IsPlayPauseToggleEnabled().unwrap_or(false),
            play: controls.IsPlayEnabled().unwrap_or(false),
            pause: controls.IsPauseEnabled().unwrap_or(false),
        };
        let timeline = self.session.GetTimelineProperties()?;
        let timeline = Timeline {
            start: timeline.StartTime()?.Duration,
            end: timeline.EndTime()?.Duration,
            position: timeline.Position()?.Duration,
            last_updated: timeline.LastUpdatedTime()?.UniversalTime,
        };
        let sampled_at = Instant::now();
        let (position_ms, duration_ms) = timeline.sample(filetime_now(), playing == Some(true));
        let reading = Reading {
            source_app: self.app_id.clone(),
            title: text(properties.Title()),
            artist: text(properties.Artist()),
            album: text(properties.AlbumTitle()),
            playing,
            position_ms,
            duration_ms,
            sampled_at,
            can_play_pause: self.controls.toggle || self.controls.play || self.controls.pause,
            can_next: controls.IsNextEnabled().unwrap_or(false),
            can_previous: controls.IsPreviousEnabled().unwrap_or(false),
            media_changed: self.media_changed.swap(false, Ordering::AcqRel),
        };
        self.nudge_volume |= reading.media_changed;
        self.properties = Some(properties);
        Ok(reading)
    }

    /// The toggle when the player offers it (Spotify does); otherwise play or pause by state.
    fn play_pause(&self) -> windows::core::Result<IAsyncOperation<bool>> {
        let controls = self.controls;
        if !controls.toggle && self.playing && controls.pause {
            return self.session.TryPauseAsync();
        }
        if !controls.toggle && !self.playing && controls.play {
            return self.session.TryPlayAsync();
        }
        self.session.TryTogglePlayPauseAsync()
    }
}

impl Drop for Followed {
    fn drop(&mut self) {
        if let Some(token) = self.media_token {
            let _ = self.session.RemoveMediaPropertiesChanged(token);
        }
        if let Some(token) = self.playback_token {
            let _ = self.session.RemovePlaybackInfoChanged(token);
        }
        if let Some(token) = self.timeline_token {
            let _ = self.session.RemoveTimelinePropertiesChanged(token);
        }
    }
}

/// The session `filter` selects: the current session if it qualifies, otherwise the first that
/// does. `Err` only when the session list itself cannot be read.
fn pick(
    manager: &Manager,
    filter: SourceFilter,
) -> windows::core::Result<Option<(Session, String)>> {
    let wanted = |app_id: &str| match filter {
        SourceFilter::Spotify => apps::is_spotify(app_id),
        SourceFilter::Any => true,
    };
    // Fails (a null session) when nothing is current.
    if let Ok(current) = manager.GetCurrentSession() {
        let app_id = app_id_of(&current);
        if wanted(&app_id) {
            return Ok(Some((current, app_id)));
        }
    }
    // Indexed rather than iterated: the bindings' iterator panics if the list cannot be read.
    let sessions = manager.GetSessions()?;
    for index in 0..sessions.Size()? {
        let Ok(session) = sessions.GetAt(index) else {
            continue;
        };
        let app_id = app_id_of(&session);
        if wanted(&app_id) {
            return Ok(Some((session, app_id)));
        }
    }
    Ok(None)
}

fn app_id_of(session: &Session) -> String {
    text(session.SourceAppUserModelId())
}

fn text(value: windows::core::Result<HSTRING>) -> String {
    value
        .map(|value| value.to_string_lossy().trim().to_owned())
        .unwrap_or_default()
}

/// Reads a whole (cover) stream. Covers above the decoder's limit are not read.
fn read_all(reference: &IRandomAccessStreamReference) -> windows::core::Result<Vec<u8>> {
    let stream = wait(&reference.OpenReadAsync()?, CALL_TIMEOUT)?;
    let size = stream.Size()?;
    let Some(size) = u32::try_from(size)
        .ok()
        .filter(|&size| size > 0 && size as usize <= art::MAX_ENCODED_BYTES)
    else {
        return Ok(Vec::new());
    };
    let reader = DataReader::CreateDataReader(&stream)?;
    let mut bytes = Vec::with_capacity(size as usize);
    while bytes.len() < size as usize {
        let remaining = size - bytes.len() as u32;
        let loaded = wait(&reader.LoadAsync(remaining)?, CALL_TIMEOUT)?;
        if loaded == 0 {
            break;
        }
        let start = bytes.len();
        bytes.resize(start + loaded.min(remaining) as usize, 0);
        reader.ReadBytes(&mut bytes[start..])?;
    }
    Ok(bytes)
}

/// Waits for a WinRT async operation for at most `timeout`; on timeout it is cancelled.
fn wait<T: RuntimeType + 'static>(
    operation: &IAsyncOperation<T>,
    timeout: Duration,
) -> windows::core::Result<T> {
    if operation.Status()? == AsyncStatus::Started {
        let (done, finished) = mpsc::sync_channel::<()>(1);
        // Called at once if the operation completed in the meantime (the async contract).
        operation.SetCompleted(&AsyncOperationCompletedHandler::new(move |_, _| {
            let _ = done.try_send(());
            Ok(())
        }))?;
        if finished.recv_timeout(timeout).is_err() {
            let _ = operation.Cancel();
            return Err(ERROR_TIMEOUT.into());
        }
    }
    operation.GetResults()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;

    /// **Read-only**, against this PC's real media sessions: polls Spotify's session, reads its
    /// cover and volume, prints them with timings. It never sends a command. Run with
    /// `cargo test -p mc-media --lib gsmtc::tests::live_read_only -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads this PC's real media sessions (read-only)"]
    fn live_read_only() {
        let (waker, _wakes) = Waker::detached();
        let mut backend = WindowsBackend::new(waker).expect("backend");
        let started = Instant::now();
        let poll = backend.poll(SourceFilter::Spotify);
        println!(
            "first poll (connect + pick + subscribe + read): {:?}",
            started.elapsed()
        );
        println!("{poll:#?}");
        let Poll::Player(_) = poll else {
            println!("no Spotify session: nothing more to read");
            return;
        };
        let started = Instant::now();
        let cover = backend.cover();
        println!("cover read: {:?}", started.elapsed());
        if let Some(bytes) = &cover {
            let head: Vec<String> = bytes.iter().take(4).map(|b| format!("{b:02X}")).collect();
            println!("cover bytes: {} (starts {})", bytes.len(), head.join(" "));
            let mut decoder = JpegDecoder::new(ZCursor::new(bytes.as_slice()));
            if decoder.decode_headers().is_ok() {
                println!("JPEG source size: {:?}", decoder.dimensions());
            }
            if let (Some(width), Some(height)) = (bytes.get(16..20), bytes.get(20..24)) {
                if bytes.starts_with(b"\x89PNG") {
                    let side = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
                    println!("PNG source size: {}x{}", side(width), side(height));
                }
            }
            let started = Instant::now();
            let art = art::decode_cover(bytes);
            println!(
                "decode + shrink: {:?} -> {:?}",
                started.elapsed(),
                art.map(|art| (art.width, art.height))
            );
        }
        let started = Instant::now();
        let volume = backend.volume();
        let elapsed = started.elapsed();
        println!("volume: {volume:?} (first read, with the session search: {elapsed:?})");
        let started = Instant::now();
        let volume = backend.volume();
        println!("volume: {volume:?} (cached: {:?})", started.elapsed());
        let rounds = 20;
        let started = Instant::now();
        for _ in 0..rounds {
            let _ = backend.poll(SourceFilter::Spotify);
        }
        println!(
            "steady poll, mean of {rounds}: {:?}",
            started.elapsed() / rounds
        );
        let started = Instant::now();
        let any = backend.poll(SourceFilter::Any);
        println!("SourceFilter::Any poll: {:?}", started.elapsed());
        if let Poll::Player(reading) = any {
            println!(
                "  current session: {} ({})",
                reading.source_app, reading.title
            );
        }
    }

    /// **Read-only**: counts the change events Spotify's session raises in 30 s (subscribing
    /// sends nothing to the player). Run with
    /// `cargo test -p mc-media --lib gsmtc::tests::live_event_rates -- --ignored --nocapture`.
    #[test]
    #[ignore = "subscribes to this PC's real media session for 30 s (read-only)"]
    fn live_event_rates() {
        use std::sync::atomic::AtomicU32;
        let _apartment = ComApartment::enter_mta();
        let request = Manager::RequestAsync().expect("request");
        let manager = wait(&request, CALL_TIMEOUT).expect("manager");
        let picked = pick(&manager, SourceFilter::Spotify).expect("sessions");
        let Some((session, app_id)) = picked else {
            println!("no Spotify session");
            return;
        };
        let counts: Arc<[AtomicU32; 3]> = Arc::new(Default::default());
        let counter = |index: usize| {
            let counts = Arc::clone(&counts);
            move || {
                counts[index].fetch_add(1, Ordering::Relaxed);
            }
        };
        let (media, playback, timeline) = (counter(0), counter(1), counter(2));
        let media = session
            .MediaPropertiesChanged(
                &TypedEventHandler::<Session, MediaPropertiesChangedEventArgs>::new(move |_, _| {
                    media();
                    Ok(())
                }),
            )
            .expect("subscribe");
        let playback = session
            .PlaybackInfoChanged(
                &TypedEventHandler::<Session, PlaybackInfoChangedEventArgs>::new(move |_, _| {
                    playback();
                    Ok(())
                }),
            )
            .expect("subscribe");
        let timeline = session
            .TimelinePropertiesChanged(&TypedEventHandler::<
                Session,
                TimelinePropertiesChangedEventArgs,
            >::new(move |_, _| {
                timeline();
                Ok(())
            }))
            .expect("subscribe");
        let status = session
            .GetPlaybackInfo()
            .and_then(|info| info.PlaybackStatus());
        println!("{app_id}: status {status:?}; counting events for 30 s");
        std::thread::sleep(Duration::from_secs(30));
        let _ = session.RemoveMediaPropertiesChanged(media);
        let _ = session.RemovePlaybackInfoChanged(playback);
        let _ = session.RemoveTimelinePropertiesChanged(timeline);
        let [media, playback, timeline] = &*counts;
        println!(
            "events in 30 s: media properties {}, playback info {}, timeline {}",
            media.load(Ordering::Relaxed),
            playback.load(Ordering::Relaxed),
            timeline.load(Ordering::Relaxed)
        );
    }
}
