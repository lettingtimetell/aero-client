//! Desktop audio: other programs' sound (a voice call, music, a browser) recorded as its own
//! stream, next to the game's sound events and the microphone.
//!
//! It records other people's voices, so it is opt-in: the Replay mod's **Record desktop audio**
//! is off by default.
//!
//! # The record format ([`crate::StreamId::Desktop`], container tag 4)
//!
//! Exactly the microphone's layout ([`crate::mic`]): one record is one 20 ms frame of 24 kHz mono
//! IMA ADPCM,
//!
//! ```text
//! u32 lead (LE, µs) | i16 predictor (LE) | u8 step index | u8 0 | 240 bytes of 4-bit codes
//! ```
//!
//! and the frame's first sample sits at `record.time - lead` on the replay clock;
//! [`decode_record`] decodes one. [`crate::ReplayMeta::desktop_format`] states the format, so a
//! later version can change it. Silence is left out (a gap is silence), with a far lower gate
//! than the microphone's: a digital source has no noise floor, so anything audible is kept.
//!
//! Why the microphone's codec: the point is voices (a voice call), and the microphone's codec
//! carries them well; it needs no codec library; and every decode path the microphone already
//! has (the exporter's mix, the viewer) works unchanged. About 96 kbit/s while something plays,
//! nothing while all is quiet. Music loses its stereo image and everything above 12 kHz.
//!
//! # Capture
//!
//! [`DesktopCapture::start`] opens Windows' *process loopback* (WASAPI with
//! `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`, Windows 10 version 2004 and later) on a helper
//! thread, so the game thread never waits for a device:
//!
//! * [`DesktopSource::AllExceptGame`]: every program's sound except this game's own process
//!   tree (the game's sound is recorded separately, as sound events);
//! * [`DesktopSource::Discord`]: only Discord's process tree (Discord, PTB, Canary), found by its
//!   process name and followed when Discord starts, restarts or quits.
//!
//! Where process loopback does not exist (older Windows, other systems) or the activation fails,
//! [`DesktopCapture::status`] says why and nothing is delivered; the recording goes on without
//! it. The samples arrive on one continuous 24 kHz timeline: when the device delivers nothing
//! (process loopback can go quiet while nothing plays) the gap is filled with silence, so later
//! sound lands at the right moment ([`Delivery`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use crate::mic::AdpcmState;
use crate::MicFormat;

pub use crate::mic::{decode_record, FRAME_MICROS, FRAME_SAMPLES, RECORD_BYTES, SAMPLE_RATE};

#[cfg(windows)]
mod wasapi;

/// Diagnostics: whether the process heap is intact (Windows `HeapValidate`; always `true`
/// elsewhere). The capture's COM code once corrupted it; `examples/desktop_check.rs` checks this
/// after many activations.
#[doc(hidden)]
#[must_use]
pub fn heap_is_valid() -> bool {
    #[cfg(windows)]
    {
        wasapi::heap_is_valid()
    }
    #[cfg(not(windows))]
    {
        true
    }
}

/// The format of [`crate::StreamId::Desktop`] records this module writes (the microphone's).
#[must_use]
pub fn format() -> MicFormat {
    crate::mic::format()
}

// ---- the source ------------------------------------------------------------------------------

/// Whose sound is recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DesktopSource {
    /// Every program except this game (and anything it started).
    #[default]
    AllExceptGame,
    /// Only Discord (the desktop app: Discord, PTB or Canary, with its helper processes).
    Discord,
}

impl DesktopSource {
    /// The source for a menu choice index: 0 every program except the game, 1 Discord only.
    #[must_use]
    pub fn from_choice(index: usize) -> DesktopSource {
        match index {
            1 => DesktopSource::Discord,
            _ => DesktopSource::AllExceptGame,
        }
    }

    /// A short label for logs and the viewer.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            DesktopSource::AllExceptGame => "All other programs",
            DesktopSource::Discord => "Discord",
        }
    }
}

/// What the capture is doing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DesktopStatus {
    /// Opening the stream (a moment after [`DesktopCapture::start`]).
    Starting,
    /// Recording.
    Capturing,
    /// [`DesktopSource::Discord`]: Discord is not running. Recording starts when it does.
    WaitingForDiscord,
    /// Desktop audio cannot be recorded here; the reason, for the log and the player.
    Unavailable(String),
}

// ---- the encoder -----------------------------------------------------------------------------

/// Frames quieter than this RMS (about -66 dBFS) count as silence. A digital source is exactly
/// zero when nothing plays; anything above this is a sound someone could hear.
const GATE_RMS: f32 = 16.0;
/// Frames kept after the last loud one (500 ms), so fades and word endings are not clipped.
const GATE_HANGOVER: u32 = 25;

/// Encodes 24 kHz mono frames and drops silent ones (one frame of lead-in, a 500 ms tail).
/// Every frame is encoded, kept or not, so each kept frame's codec header continues the stream
/// and decodes on its own ([`crate::mic::decode_frame`]).
#[derive(Debug, Default)]
pub struct DesktopEncoder {
    state: AdpcmState,
    open_for: u32,
    /// The previous frame, encoded, if it was dropped: sent ahead of a frame that opens the gate.
    lead_in: Option<(u64, Vec<u8>)>,
    frames: u64,
    out: Vec<u8>,
    /// Frames recorded (not dropped as silence).
    pub kept: u64,
}

impl DesktopEncoder {
    /// Encode the next frame of [`FRAME_SAMPLES`] samples. `emit(frame_index, frame)` is called
    /// for each frame to record (the lead-in and this one when sound starts, none in silence).
    pub fn push(&mut self, frame: &[i16], mut emit: impl FnMut(u64, &[u8])) {
        let index = self.frames;
        self.frames += 1;
        crate::mic::encode_frame(&mut self.state, frame, &mut self.out);
        let loud = rms(frame) >= GATE_RMS;
        if loud {
            if self.open_for == 0 {
                if let Some((lead_index, lead)) = self.lead_in.take() {
                    emit(lead_index, &lead);
                    self.kept += 1;
                }
            }
            self.open_for = GATE_HANGOVER;
        }
        if self.open_for > 0 {
            if !loud {
                self.open_for -= 1;
            }
            emit(index, &self.out);
            self.kept += 1;
            self.lead_in = None;
        } else {
            let mut lead = self.lead_in.take().map(|(_, v)| v).unwrap_or_default();
            lead.clear();
            lead.extend_from_slice(&self.out);
            self.lead_in = Some((index, lead));
        }
    }

    /// Frames pushed so far.
    #[must_use]
    pub fn frames(&self) -> u64 {
        self.frames
    }
}

fn rms(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f64 = frame.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / frame.len() as f64).sqrt() as f32
}

// ---- the capture -----------------------------------------------------------------------------

/// The longest stretch of samples kept waiting for the game thread (5 s); older ones are
/// dropped (the game thread drains every frame, so this only matters while it is stalled).
const MAX_BUFFERED: usize = SAMPLE_RATE as usize * 5;

#[derive(Debug)]
struct Shared {
    samples: Mutex<VecDeque<i16>>,
    delivered: AtomicU64,
    dropped: AtomicU64,
    stop: AtomicBool,
    status: Mutex<DesktopStatus>,
}

impl Shared {
    fn new() -> Shared {
        Shared {
            samples: Mutex::new(VecDeque::new()),
            delivered: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            status: Mutex::new(DesktopStatus::Starting),
        }
    }

    fn set_status(&self, status: DesktopStatus) {
        let mut slot = self.status.lock().unwrap_or_else(PoisonError::into_inner);
        if *slot != status {
            match &status {
                DesktopStatus::Unavailable(reason) => {
                    tracing::warn!(%reason, "replay desktop audio unavailable");
                }
                other => tracing::info!(status = ?other, "replay desktop audio"),
            }
            *slot = status;
        }
    }

    fn stopping(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Append samples (24 kHz mono), dropping the oldest beyond [`MAX_BUFFERED`].
    fn push(&self, samples: impl ExactSizeIterator<Item = i16>) {
        let count = samples.len() as u64;
        let mut queue = self.samples.lock().unwrap_or_else(PoisonError::into_inner);
        queue.extend(samples);
        let excess = queue.len().saturating_sub(MAX_BUFFERED);
        if excess > 0 {
            queue.drain(..excess);
            self.dropped.fetch_add(excess as u64, Ordering::Relaxed);
        }
        drop(queue);
        self.delivered.fetch_add(count, Ordering::Relaxed);
    }
}

/// Other programs' sound, captured on a helper thread and delivered as 24 kHz mono samples.
/// Dropping it closes the stream (the thread notices within 100 ms).
#[derive(Debug)]
pub struct DesktopCapture {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DesktopCapture {
    /// Start capturing `source`. Never blocks; a failure shows in [`DesktopCapture::status`].
    #[must_use]
    pub fn start(source: DesktopSource) -> DesktopCapture {
        let shared = Arc::new(Shared::new());
        let thread_shared = Arc::clone(&shared);
        // A panic in the capture is its failure (the status), not the
        // game's crash.
        let thread = std::thread::Builder::new()
            .name("mc-rs replay desktop audio".into())
            .spawn(move || {
                crate::contained::run_helper(
                    "desktop audio capture",
                    || run(source, &thread_shared),
                    |message| thread_shared.set_status(DesktopStatus::Unavailable(message)),
                );
            })
            .ok();
        if thread.is_none() {
            shared.set_status(DesktopStatus::Unavailable(
                "could not start the capture thread".into(),
            ));
        }
        DesktopCapture { shared, thread }
    }

    /// A capture with no device behind it, fed by [`DesktopCapture::feed`]: for tests of what
    /// is built on top. Never opens anything; its status is [`DesktopStatus::Capturing`].
    #[must_use]
    pub fn fake() -> DesktopCapture {
        let shared = Arc::new(Shared::new());
        shared.set_status(DesktopStatus::Capturing);
        DesktopCapture {
            shared,
            thread: None,
        }
    }

    /// Deliver samples (24 kHz mono) as the device would; for a [`DesktopCapture::fake`].
    pub fn feed(&self, samples: &[i16]) {
        self.shared.push(samples.iter().copied());
    }

    /// A handle that delivers samples into this capture from elsewhere (a test keeps it after
    /// handing the capture over).
    #[must_use]
    pub fn feeder(&self) -> DesktopFeeder {
        DesktopFeeder {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Move every waiting sample into `out` (appended).
    pub fn drain(&self, out: &mut Vec<i16>) {
        let mut samples = self
            .shared
            .samples
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        out.extend(samples.drain(..));
    }

    /// Samples delivered since the capture started.
    #[must_use]
    pub fn delivered(&self) -> u64 {
        self.shared.delivered.load(Ordering::Relaxed)
    }

    /// What the capture is doing.
    #[must_use]
    pub fn status(&self) -> DesktopStatus {
        self.shared
            .status
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Why nothing is recorded, if desktop audio is unavailable.
    #[must_use]
    pub fn error(&self) -> Option<String> {
        match self.status() {
            DesktopStatus::Unavailable(reason) => Some(reason),
            _ => None,
        }
    }

    /// Close the stream and wait for the helper thread (it checks every 50 ms at most).
    pub fn stop(mut self) {
        self.stop_inner(true);
    }

    fn stop_inner(&mut self, wait: bool) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            if wait {
                let _ = thread.join();
            }
        }
    }
}

impl Drop for DesktopCapture {
    /// Never waits: the game thread drops a capture when a recording ends, and the helper thread
    /// closes the stream by itself within 50 ms.
    fn drop(&mut self) {
        self.stop_inner(false);
    }
}

/// Delivers samples into a [`DesktopCapture`] as its device would ([`DesktopCapture::feeder`]).
#[derive(Debug, Clone)]
pub struct DesktopFeeder {
    shared: Arc<Shared>,
}

impl DesktopFeeder {
    /// Deliver samples (24 kHz mono).
    pub fn feed(&self, samples: &[i16]) {
        self.shared.push(samples.iter().copied());
    }
}

/// The capture thread's body.
fn run(source: DesktopSource, shared: &Arc<Shared>) {
    #[cfg(windows)]
    wasapi::run(source, shared);
    #[cfg(not(windows))]
    {
        let _ = source;
        shared.set_status(DesktopStatus::Unavailable(
            "desktop audio recording needs Windows 10 version 2004 or newer".into(),
        ));
    }
}

// ---- the timeline ----------------------------------------------------------------------------

/// How far the device's delivery may lag or jitter before a gap is filled with silence (40 ms).
const TOLERANCE: u64 = SAMPLE_RATE as u64 / 25;

/// Puts the device's packets on one continuous 24 kHz timeline that starts when the capture
/// started. Sample `n` is meant to be heard `n / 24000` s after that, give or take the device's
/// latency: the game thread places samples by counting them, so a stretch the device skipped
/// (process loopback can deliver nothing while nothing plays) must still take its time. A packet
/// that ends now but starts after the timeline's end gets silence in front of it; a timeline that
/// falls behind the clock while nothing arrives is topped up with silence, a little behind the
/// clock so a late packet still fits.
#[derive(Debug)]
pub(crate) struct Delivery {
    started: Instant,
    /// Samples delivered so far.
    pos: u64,
    /// Device samples converted to 24 kHz mono.
    converter: Converter,
    scratch: Vec<i16>,
}

impl Delivery {
    pub(crate) fn new(started: Instant, rate: u32, channels: u16) -> Delivery {
        Delivery {
            started,
            pos: 0,
            converter: Converter::new(rate, channels),
            scratch: Vec::new(),
        }
    }

    /// A new stream (another format) continues the same timeline.
    pub(crate) fn set_format(&mut self, rate: u32, channels: u16) {
        self.converter = Converter::new(rate, channels);
    }

    /// The timeline position of `now`.
    fn clock(&self, now: Instant) -> u64 {
        let micros = now.saturating_duration_since(self.started).as_micros() as u64;
        micros * u64::from(SAMPLE_RATE) / 1_000_000
    }

    /// A packet of interleaved 16-bit device samples (`None`: a packet the device marked silent,
    /// `frames` long) that arrived at `now`.
    pub(crate) fn packet(
        &mut self,
        now: Instant,
        data: Option<&[i16]>,
        frames: usize,
        out: &mut dyn FnMut(&[i16]),
    ) {
        self.scratch.clear();
        match data {
            Some(samples) => self.converter.process(samples, &mut self.scratch),
            None => self.converter.silence(frames, &mut self.scratch),
        }
        let end = self.clock(now);
        let start = end.saturating_sub(self.scratch.len() as u64);
        if start > self.pos + TOLERANCE {
            self.silence(start - self.pos, out);
        }
        self.pos += self.scratch.len() as u64;
        out(&self.scratch);
    }

    /// Nothing arrived: top the timeline up with silence if it fell well behind the clock.
    pub(crate) fn idle(&mut self, now: Instant, out: &mut dyn FnMut(&[i16])) {
        let end = self.clock(now);
        if end > self.pos + 2 * TOLERANCE {
            self.silence(end - TOLERANCE - self.pos, out);
        }
    }

    fn silence(&mut self, mut count: u64, out: &mut dyn FnMut(&[i16])) {
        const ZEROS: [i16; 960] = [0; 960];
        self.pos += count;
        while count > 0 {
            let n = count.min(ZEROS.len() as u64) as usize;
            out(&ZEROS[..n]);
            count -= n as u64;
        }
    }

    /// Samples delivered so far.
    #[cfg(test)]
    pub(crate) fn position(&self) -> u64 {
        self.pos
    }
}

/// Converts interleaved 16-bit device samples (any rate, any channel count) to 24 kHz mono:
/// a straight copy when the device already delivers that, otherwise a downmix, a one-pole
/// low-pass against aliasing and linear interpolation.
#[derive(Debug)]
struct Converter {
    channels: usize,
    passthrough: bool,
    /// Input samples per output sample.
    step: f64,
    /// Position of the next output sample, in input samples, relative to `prev`.
    phase: f64,
    prev: f32,
    lowpass: f32,
    alpha: f32,
}

impl Converter {
    fn new(rate: u32, channels: u16) -> Converter {
        let rate = rate.max(1);
        let alpha = if rate > SAMPLE_RATE {
            let dt = 1.0 / f64::from(rate);
            let rc = 1.0 / (2.0 * std::f64::consts::PI * 10_000.0);
            (dt / (rc + dt)) as f32
        } else {
            1.0
        };
        Converter {
            channels: usize::from(channels.max(1)),
            passthrough: rate == SAMPLE_RATE && channels <= 1,
            step: f64::from(rate) / f64::from(SAMPLE_RATE),
            phase: 0.0,
            prev: 0.0,
            lowpass: 0.0,
            alpha,
        }
    }

    fn process(&mut self, input: &[i16], out: &mut Vec<i16>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        for frame in input.chunks(self.channels) {
            let mono = frame.iter().map(|&s| f32::from(s)).sum::<f32>() / frame.len() as f32;
            self.push(mono, out);
        }
    }

    fn silence(&mut self, frames: usize, out: &mut Vec<i16>) {
        if self.passthrough {
            out.extend(std::iter::repeat_n(0, frames));
            return;
        }
        for _ in 0..frames {
            self.push(0.0, out);
        }
    }

    #[inline]
    fn push(&mut self, mono: f32, out: &mut Vec<i16>) {
        self.lowpass += self.alpha * (mono - self.lowpass);
        let current = self.lowpass;
        while self.phase <= 1.0 {
            let t = self.phase as f32;
            let value = self.prev + (current - self.prev) * t;
            out.push(value.clamp(-32768.0, 32767.0) as i16);
            self.phase += self.step;
        }
        self.phase -= 1.0;
        self.prev = current;
    }
}

// ---- Discord ---------------------------------------------------------------------------------

/// Executable names of the Discord desktop app, preferred in this order.
const DISCORD_EXES: [&str; 4] = [
    "discord.exe",
    "discordptb.exe",
    "discordcanary.exe",
    "discorddevelopment.exe",
];

/// The root of Discord's process tree in a process list of `(pid, parent pid, exe name)`: a
/// Discord process whose parent is not a Discord process (the app starts its helpers itself).
/// With several apps running, the stable one first, then PTB, Canary; among equals the one with
/// the most helper processes. `None` when Discord is not running.
#[must_use]
pub fn discord_root(processes: &[(u32, u32, String)]) -> Option<u32> {
    let rank = |name: &str| {
        DISCORD_EXES
            .iter()
            .position(|exe| name.eq_ignore_ascii_case(exe))
    };
    let discord: Vec<(u32, u32, usize)> = processes
        .iter()
        .filter_map(|(pid, parent, name)| rank(name).map(|r| (*pid, *parent, r)))
        .collect();
    discord
        .iter()
        .filter(|(pid, parent, _)| {
            *parent == *pid || !discord.iter().any(|(other, _, _)| other == parent)
        })
        .min_by_key(|(pid, _, rank)| {
            let helpers = discord
                .iter()
                .filter(|(_, parent, _)| parent == pid)
                .count();
            (*rank, std::cmp::Reverse(helpers))
        })
        .map(|(pid, _, _)| *pid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn tone(freq: f32, amplitude: f32, samples: usize) -> Vec<i16> {
        (0..samples)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16
            })
            .collect()
    }

    /// Digital silence is dropped; quiet sound the microphone's gate would drop (-55 dBFS) is
    /// kept, with one frame of lead-in and a 500 ms tail; every kept frame decodes to what the
    /// encoder reconstructed.
    #[test]
    fn the_gate_keeps_quiet_sound_and_drops_digital_silence() {
        let mut encoder = DesktopEncoder::default();
        let silent = vec![0i16; FRAME_SAMPLES];
        // -55 dBFS RMS: about 58.
        let quiet = tone(440.0, 82.0, FRAME_SAMPLES);
        let mut kept = Vec::new();
        for _ in 0..5 {
            encoder.push(&silent, |i, _| kept.push(i));
        }
        assert!(kept.is_empty(), "silence is not recorded");
        for _ in 0..2 {
            encoder.push(&quiet, |i, frame| {
                assert_eq!(frame.len(), crate::mic::FRAME_BYTES);
                kept.push(i);
            });
        }
        assert_eq!(
            kept,
            vec![4, 5, 6],
            "the frame before the sound is the lead-in"
        );
        kept.clear();
        for _ in 0..40 {
            encoder.push(&silent, |i, _| kept.push(i));
        }
        assert_eq!(
            kept.len(),
            GATE_HANGOVER as usize,
            "a 500 ms tail, then nothing"
        );
        // The microphone's gate (RMS 104) would have dropped that sound.
        assert!(rms(&quiet) < 104.0 && rms(&quiet) >= GATE_RMS);
    }

    /// The record layout is the microphone's: what the encoder emits, behind a lead, decodes
    /// with [`decode_record`] to a close copy of the input.
    #[test]
    fn records_decode_with_the_microphone_decoder() {
        let input = tone(700.0, 9_000.0, FRAME_SAMPLES * 6);
        let mut encoder = DesktopEncoder::default();
        let mut decoded = Vec::new();
        let mut records = 0;
        for frame in input.chunks(FRAME_SAMPLES) {
            encoder.push(frame, |_, bytes| {
                let mut record = 1234u32.to_le_bytes().to_vec();
                record.extend_from_slice(bytes);
                assert_eq!(record.len(), RECORD_BYTES);
                let lead = decode_record(&record, &mut decoded).expect("decodes");
                assert_eq!(lead, 1234);
                records += 1;
            });
        }
        assert_eq!(records, 6);
        let tail = &input[FRAME_SAMPLES..];
        let err: f64 = tail
            .iter()
            .zip(&decoded[FRAME_SAMPLES..])
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / tail.len() as f64;
        let signal: f64 =
            tail.iter().map(|&a| f64::from(a).powi(2)).sum::<f64>() / tail.len() as f64;
        let snr = 10.0 * (signal / err).log10();
        assert!(snr > 25.0, "SNR {snr:.1} dB");
        assert_eq!(format().codec, crate::mic::CODEC);
    }

    /// The timeline: packets that keep up are appended as they are; a stretch with no packets
    /// is filled with silence (a little behind the clock), and a packet after a gap gets the
    /// missing silence in front of it, so it lands at its time.
    #[test]
    fn the_timeline_fills_gaps_with_silence_and_keeps_packets_in_place() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let mut delivery = Delivery::new(t0, SAMPLE_RATE, 1);
        let mut got: Vec<i16> = Vec::new();
        // 10 ms packets of a constant, arriving on time for 200 ms.
        let packet = vec![1000i16; 240];
        for i in 1..=20u64 {
            delivery.packet(ms(i * 10), Some(&packet), 240, &mut |s| {
                got.extend_from_slice(s);
            });
        }
        assert_eq!(
            delivery.position(),
            4_800,
            "no silence while packets keep up"
        );
        // Nothing for 500 ms: idle top-ups keep the timeline 40-80 ms behind the clock (it is
        // topped up to 40 ms behind whenever it falls 80 ms behind).
        for i in 21..=70u64 {
            delivery.idle(ms(i * 10), &mut |s| got.extend_from_slice(s));
            let behind = (i * 10 * 24).saturating_sub(delivery.position());
            assert!(
                behind <= 2 * TOLERANCE,
                "{behind} samples behind at {} ms",
                i * 10
            );
        }
        let behind = 700 * 24 - delivery.position();
        assert!((TOLERANCE..=2 * TOLERANCE).contains(&behind), "{behind}");
        // A packet arriving 300 ms later still lands at its time: silence first.
        delivery.packet(ms(1_000), Some(&packet), 240, &mut |s| {
            got.extend_from_slice(s);
        });
        assert_eq!(delivery.position(), 1_000 * 24);
        assert_eq!(got.len() as u64, delivery.position());
        let first_new = got.len() - 240;
        assert!(got[first_new..].iter().all(|&s| s == 1000));
        assert!(got[4_800..first_new].iter().all(|&s| s == 0));
        // Packets the device marks silent count as time too.
        delivery.packet(ms(1_010), None, 240, &mut |s| got.extend_from_slice(s));
        assert_eq!(delivery.position(), 1_010 * 24);
    }

    /// 48 kHz stereo device samples become 24 kHz mono at the right length and level.
    #[test]
    fn the_converter_downmixes_and_halves_the_rate() {
        let mut converter = Converter::new(48_000, 2);
        let mut out = Vec::new();
        let input: Vec<i16> = (0..48_000)
            .flat_map(|i| {
                let v = (2.0 * std::f32::consts::PI * 500.0 * i as f32 / 48_000.0).sin() * 16_000.0;
                [v as i16, v as i16]
            })
            .collect();
        for chunk in input.chunks(2 * 480) {
            converter.process(chunk, &mut out);
        }
        assert!((out.len() as i64 - 24_000).abs() <= 2, "{}", out.len());
        let peak = out.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(peak > 14_500 && peak < 16_500, "peak {peak}");
        let mut same = Converter::new(24_000, 1);
        let mut copied = Vec::new();
        same.process(&[1, -2, 3], &mut copied);
        assert_eq!(
            copied,
            vec![1, -2, 3],
            "24 kHz mono passes straight through"
        );
    }

    #[test]
    fn discord_is_found_at_the_root_of_its_process_tree() {
        let list = |entries: &[(u32, u32, &str)]| -> Vec<(u32, u32, String)> {
            entries
                .iter()
                .map(|(p, q, n)| (*p, *q, (*n).to_owned()))
                .collect()
        };
        // The app (100, started by explorer 4) and its helpers.
        let procs = list(&[
            (4, 1, "explorer.exe"),
            (100, 4, "Discord.exe"),
            (101, 100, "Discord.exe"),
            (102, 100, "Discord.exe"),
            (103, 101, "Discord.exe"),
            (200, 4, "mc-client.exe"),
        ]);
        assert_eq!(discord_root(&procs), Some(100));
        // Not running.
        assert_eq!(discord_root(&list(&[(4, 1, "explorer.exe")])), None);
        // Stable and PTB at once: stable wins; PTB alone is found.
        let both = list(&[(300, 4, "DiscordPTB.exe"), (301, 300, "DiscordPTB.exe")]);
        assert_eq!(discord_root(&both), Some(300));
        let mut all = procs.clone();
        all.extend(both);
        assert_eq!(discord_root(&all), Some(100));
        // A helper whose parent exited (orphaned) and the real root: the one with helpers wins.
        let orphan = list(&[
            (500, 9_999, "discord.exe"),
            (600, 4, "Discord.exe"),
            (601, 600, "Discord.exe"),
        ]);
        assert_eq!(discord_root(&orphan), Some(600));
    }

    #[test]
    fn a_fake_capture_delivers_what_it_is_fed() {
        let capture = DesktopCapture::fake();
        assert_eq!(capture.status(), DesktopStatus::Capturing);
        let feeder = capture.feeder();
        feeder.feed(&[1, 2, 3]);
        capture.feed(&[4]);
        let mut out = Vec::new();
        capture.drain(&mut out);
        assert_eq!(out, vec![1, 2, 3, 4]);
        assert_eq!(capture.delivered(), 4);
        assert_eq!(capture.error(), None);
        capture.stop();
    }

    #[test]
    fn sources_map_from_menu_choices() {
        assert_eq!(DesktopSource::from_choice(0), DesktopSource::AllExceptGame);
        assert_eq!(DesktopSource::from_choice(1), DesktopSource::Discord);
        assert_eq!(DesktopSource::from_choice(9), DesktopSource::AllExceptGame);
    }
}
