//! Desktop audio while recording.
//!
//! Other programs' sound (a voice call, music; [`mc_replay::desktop`]) is recorded as its own
//! stream, [`StreamId::Desktop`], the way the parent module records the microphone:
//!
//! * The capture is open while **Record desktop audio** is on (off by default: it records other
//!   people's voices) and either a recording runs unpaused (records go into the file) or the clip
//!   key's session buffer runs (they go into a 300 s ring, so a clip has them even with no
//!   recording running). It closes when neither holds, with the session, and when the source
//!   setting changes. With the option off nothing is opened and the hook is two tests a frame.
//! * Records are pushed in the frame that received their samples, with a lead saying how much
//!   older their first sample is ([`mc_replay::desktop::decode_record`]), on the microphone's
//!   smoothed timeline, so consecutive frames sit exactly 20 ms apart in the file.
//! * The file's metadata gets `has_desktop` and `desktop_format` with its first record.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use mc_mods::ModSettings;
use mc_replay::desktop::{DesktopCapture, DesktopEncoder, DesktopSource, RECORD_BYTES};
use mc_replay::{Record, ReplayMeta, ReplayTime, StreamId};

use super::{ids, Client, Conn, Input, MicSlot, KEY_CLIP, MIC_RING_FRAMES, MIC_RING_SPAN, MOD_ID};

/// The desktop audio settings, read with the recorder's other options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct DesktopOptions {
    on: bool,
    source: DesktopSource,
}

impl DesktopOptions {
    pub(super) fn read(settings: &ModSettings) -> DesktopOptions {
        DesktopOptions {
            on: settings.bool(MOD_ID, ids::DESKTOP_AUDIO),
            source: DesktopSource::from_choice(settings.choice(MOD_ID, ids::DESKTOP_SOURCE)),
        }
    }
}

/// Desktop audio while something records.
#[derive(Debug)]
pub(super) struct Desktop {
    capture: DesktopCapture,
    source: DesktopSource,
    encoder: DesktopEncoder,
    pending: Vec<i16>,
    /// Samples received since the timeline began.
    received: u64,
    /// Session time of the timeline's first sample (smoothed), µs.
    anchor: Option<f64>,
    /// The lowest anchor estimate this window (the delivery with the least delay).
    window_min: f64,
    window_started: Option<Instant>,
    /// This frame's records, back to back ([`RECORD_BYTES`] each).
    records: Vec<u8>,
    warned: bool,
    /// The last 300 s of records while the clip key's buffer runs (session clock).
    ring: VecDeque<MicSlot>,
}

impl Desktop {
    fn new(capture: DesktopCapture, source: DesktopSource) -> Desktop {
        Desktop {
            capture,
            source,
            encoder: DesktopEncoder::default(),
            pending: Vec::new(),
            received: 0,
            anchor: None,
            window_min: f64::MAX,
            window_started: None,
            records: Vec::new(),
            warned: false,
            ring: VecDeque::new(),
        }
    }

    /// Take what the capture delivered, encode whole frames, and leave this frame's records in
    /// `self.records` (the gate drops silence). They are pushed at `session_now`; a record's
    /// first four bytes say how long before that its first sample was captured. The same
    /// timeline as the microphone's (`Mic::frame` in the parent): no allocation once warm.
    fn frame(&mut self, session_now: ReplayTime, now: Instant) {
        self.records.clear();
        if let Some(error) = self.capture.error() {
            if !self.warned {
                tracing::warn!(%error, "replay desktop audio: nothing will be recorded");
                self.warned = true;
            }
            return;
        }
        let before = self.pending.len();
        self.capture.drain(&mut self.pending);
        let got = (self.pending.len() - before) as u64;
        if got == 0 && self.anchor.is_none() {
            return;
        }
        self.received += got;
        let rate = f64::from(mc_replay::desktop::SAMPLE_RATE);
        // When sample 0 was captured, if the newest one was captured right now.
        let estimate = session_now as f64 - self.received as f64 * 1e6 / rate;
        let anchor = *self.anchor.get_or_insert(estimate);
        self.window_min = self.window_min.min(estimate);
        let started = *self.window_started.get_or_insert(now);
        if now.saturating_duration_since(started) >= Duration::from_secs(1) {
            // Drift and delivery jitter: move toward the least-delayed estimate, 2 ms a second
            // at most, so consecutive frames stay exactly 20 ms apart.
            let anchor = anchor + (self.window_min - anchor).clamp(-2000.0, 2000.0);
            self.anchor = Some(anchor);
            self.window_min = f64::MAX;
            self.window_started = Some(now);
        }
        let anchor = self.anchor.unwrap_or(estimate);
        let samples = mc_replay::desktop::FRAME_SAMPLES;
        let mut consumed = 0;
        while self.pending.len() - consumed >= samples {
            let frame = &self.pending[consumed..consumed + samples];
            let records = &mut self.records;
            self.encoder.push(frame, |index, bytes| {
                let at = anchor + index as f64 * samples as f64 * 1e6 / rate;
                let lead = (session_now as f64 - at).clamp(0.0, f64::from(u32::MAX)) as u32;
                records.extend_from_slice(&lead.to_le_bytes());
                records.extend_from_slice(bytes);
            });
            consumed += samples;
        }
        self.pending.drain(..consumed);
    }
}

/// Mark `meta` as holding desktop audio.
fn mark(meta: &mut ReplayMeta) {
    if !meta.has_desktop {
        meta.has_desktop = true;
        meta.desktop_format = Some(mc_replay::desktop::format());
    }
}

/// The desktop records of a clip's `window` (session clock), from the ring, appended to `extra`
/// (the writer sorts and merges them); marks the clip's metadata when there are any.
pub(super) fn add_to_clip(
    conn: &Conn,
    window: (ReplayTime, ReplayTime),
    extra: &mut Vec<Record>,
    meta: &mut ReplayMeta,
) {
    let Some(desktop) = conn.desktop.as_ref() else {
        return;
    };
    let before = extra.len();
    extra.extend(
        desktop
            .ring
            .iter()
            .filter(|slot| slot.time >= window.0 && slot.time <= window.1)
            .map(|slot| Record {
                time: slot.time,
                stream: StreamId::Desktop,
                payload: slot.bytes.to_vec(),
            }),
    );
    if extra.len() > before {
        mark(meta);
    }
}

impl Client {
    /// Desktop audio, each frame after the microphone (`<<NW-RE:rec>>` in the parent's frame).
    pub(super) fn rec_desktop(&mut self, now: Instant) {
        let opts = self.rec.desktop_opts;
        let clip_bound = self.rec.opts.keys[KEY_CLIP] != Input::Unbound;
        let Some(conn) = self.rec.conn.as_deref_mut() else {
            return;
        };
        if !opts.on && conn.desktop.is_none() {
            return;
        }
        let to_file = conn
            .recording
            .as_ref()
            .is_some_and(|recording| recording.paused_at.is_none());
        let to_ring = clip_bound && conn.buffer.is_some();
        let wanted = opts.on && (to_file || to_ring);
        let same_source = conn
            .desktop
            .as_ref()
            .is_none_or(|desktop| desktop.source == opts.source);
        if !wanted || !same_source {
            if conn.desktop.take().is_some() {
                tracing::info!("replay desktop audio off");
            }
            if !wanted {
                return;
            }
        }
        let mut desktop = match conn.desktop.take() {
            Some(desktop) => desktop,
            None => {
                // Opening never blocks (the stream opens on the capture's own thread).
                let capture = self.rec.open_desktop(opts.source);
                tracing::info!(source = opts.source.label(), "replay desktop audio on");
                Desktop::new(capture, opts.source)
            }
        };
        let Some(conn) = self.rec.conn.as_deref_mut() else {
            return;
        };
        if !to_ring {
            desktop.ring.clear();
        } else if desktop.ring.capacity() < MIC_RING_FRAMES {
            // Room for the whole ring at once, so pushing never allocates.
            let missing = MIC_RING_FRAMES.saturating_sub(desktop.ring.len());
            desktop.ring.reserve(missing);
        }
        let session_now = conn.now;
        desktop.frame(session_now, now);
        for record in desktop.records.chunks_exact(RECORD_BYTES) {
            if to_file {
                if let Some(recording) = conn.recording.as_mut() {
                    mark(&mut recording.meta);
                    let time = recording.time(session_now);
                    recording.writer.push(time, StreamId::Desktop, record);
                }
            }
            if to_ring {
                let mut bytes = [0u8; RECORD_BYTES];
                bytes.copy_from_slice(record);
                desktop.ring.push_back(MicSlot {
                    time: session_now,
                    bytes,
                });
            }
        }
        // The ring holds the longest clip there can be.
        while desktop
            .ring
            .front()
            .is_some_and(|slot| session_now.saturating_sub(slot.time) > MIC_RING_SPAN)
        {
            desktop.ring.pop_front();
        }
        conn.desktop = Some(desktop);
    }
}

impl super::RecState {
    /// Open desktop audio for `source` (a fake capture in tests).
    fn open_desktop(&mut self, source: DesktopSource) -> DesktopCapture {
        #[cfg(test)]
        if let Some(capture) = self.test_desktop.take() {
            return capture;
        }
        DesktopCapture::start(source)
    }
}
