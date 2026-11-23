//! The microphone stream: capture from the default input device, a 24 kHz mono IMA ADPCM codec,
//! and a noise gate that drops silent frames.
//!
//! # The record format ([`crate::StreamId::Mic`])
//!
//! One record is one frame of [`FRAME_SAMPLES`] samples (20 ms at [`SAMPLE_RATE`]),
//! [`RECORD_BYTES`] long:
//!
//! ```text
//! u32 lead (LE, µs) | i16 predictor (LE) | u8 step index | u8 0 | 240 bytes of 4-bit codes
//! ```
//!
//! `lead` is how long before the record's own time the frame's first sample was captured: the
//! record is pushed in the frame that received it, the audio is a little older. So the frame
//! starts at `record.time - lead` on the replay clock; consecutive frames of one stretch of
//! speech are exactly 20 ms apart there ([`decode_record`]). The codec header is the state
//! *before* the frame's first sample, so every frame decodes on its own ([`decode_frame`]);
//! codes are packed low nibble first. Frames the gate judged silent are not recorded: a gap is
//! silence. 4 bits a sample makes about 99 kbit/s while speaking and nothing while quiet.
//!
//! # Why ADPCM
//!
//! It is tiny, exact to specify and decode (exports mix it in, the replay player plays it),
//! needs no codec library, and costs nothing to encode. Opus would be smaller but is not in the
//! dependency graph; Media Foundation AAC (the mc-video) is an option for a later version.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use crate::{MicFormat, ReplayError};

/// Samples per second of the recorded stream.
pub const SAMPLE_RATE: u32 = 24_000;
/// Samples in one record (20 ms).
pub const FRAME_SAMPLES: usize = 480;
/// Bytes in one encoded frame (codec header and codes).
pub const FRAME_BYTES: usize = 4 + FRAME_SAMPLES / 2;
/// Bytes in one record: the lead, then the frame.
pub const RECORD_BYTES: usize = 4 + FRAME_BYTES;
/// One frame's length in microseconds.
pub const FRAME_MICROS: u64 = (FRAME_SAMPLES as u64) * 1_000_000 / SAMPLE_RATE as u64;
/// The codec name stored in [`MicFormat::codec`].
pub const CODEC: &str = "ima-adpcm";

/// The [`MicFormat`] this module writes.
#[must_use]
pub fn format() -> MicFormat {
    MicFormat {
        codec: CODEC.to_owned(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
        frame_samples: FRAME_SAMPLES as u32,
    }
}

const INDEX_TABLE: [i8; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

const STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// The IMA ADPCM codec state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdpcmState {
    /// The last decoded sample.
    pub predictor: i16,
    /// Index into the step table (0..=88).
    pub index: u8,
}

impl AdpcmState {
    /// Decode one 4-bit code, updating the state; returns the sample.
    #[inline]
    fn decode(&mut self, code: u8) -> i16 {
        let step = STEP_TABLE[self.index as usize];
        let mut delta = step >> 3;
        if code & 4 != 0 {
            delta += step;
        }
        if code & 2 != 0 {
            delta += step >> 1;
        }
        if code & 1 != 0 {
            delta += step >> 2;
        }
        let predictor = if code & 8 != 0 {
            i32::from(self.predictor) - delta
        } else {
            i32::from(self.predictor) + delta
        };
        self.predictor = predictor.clamp(-32768, 32767) as i16;
        self.index =
            (i32::from(self.index) + i32::from(INDEX_TABLE[code as usize])).clamp(0, 88) as u8;
        self.predictor
    }

    /// Encode one sample to its 4-bit code, updating the state exactly as the decoder will.
    #[inline]
    fn encode(&mut self, sample: i16) -> u8 {
        let step = STEP_TABLE[self.index as usize];
        let mut diff = i32::from(sample) - i32::from(self.predictor);
        let mut code = 0u8;
        if diff < 0 {
            code = 8;
            diff = -diff;
        }
        if diff >= step {
            code |= 4;
            diff -= step;
        }
        if diff >= step >> 1 {
            code |= 2;
            diff -= step >> 1;
        }
        if diff >= step >> 2 {
            code |= 1;
        }
        // The state follows the decoder's arithmetic, so encoder and decoder never drift.
        self.decode(code);
        code
    }
}

/// Encode one frame of [`FRAME_SAMPLES`] samples (fewer are padded with the last one) into
/// `out` (cleared first), continuing from `state`.
pub fn encode_frame(state: &mut AdpcmState, samples: &[i16], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&state.predictor.to_le_bytes());
    out.push(state.index);
    out.push(0);
    let last = samples.last().copied().unwrap_or(0);
    let sample = |i: usize| samples.get(i).copied().unwrap_or(last);
    for pair in 0..FRAME_SAMPLES / 2 {
        let low = state.encode(sample(pair * 2));
        let high = state.encode(sample(pair * 2 + 1));
        out.push(low | (high << 4));
    }
}

/// Decode one record into `out` (appended: [`FRAME_SAMPLES`] samples).
///
/// # Errors
///
/// [`ReplayError::Corrupt`] if the record is not one frame long or its step index is out of
/// range.
pub fn decode_frame(payload: &[u8], out: &mut Vec<i16>) -> Result<(), ReplayError> {
    if payload.len() != FRAME_BYTES || payload[2] > 88 {
        return Err(ReplayError::Corrupt("microphone frame".into()));
    }
    let mut state = AdpcmState {
        predictor: i16::from_le_bytes([payload[0], payload[1]]),
        index: payload[2],
    };
    out.reserve(FRAME_SAMPLES);
    for &byte in &payload[4..] {
        out.push(state.decode(byte & 0x0F));
        out.push(state.decode(byte >> 4));
    }
    Ok(())
}

/// Decode one [`crate::StreamId::Mic`] record into `out` (appended: [`FRAME_SAMPLES`] samples).
/// Returns its lead in microseconds: the frame's first sample is at `record.time - lead`.
///
/// # Errors
///
/// [`ReplayError::Corrupt`] if the record is not [`RECORD_BYTES`] long or its frame is damaged.
pub fn decode_record(payload: &[u8], out: &mut Vec<i16>) -> Result<u32, ReplayError> {
    if payload.len() != RECORD_BYTES {
        return Err(ReplayError::Corrupt("microphone record".into()));
    }
    decode_frame(&payload[4..], out)?;
    Ok(u32::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3],
    ]))
}

/// Frames quieter than this RMS (about -50 dBFS) count as silence.
const GATE_RMS: f32 = 104.0;
/// Frames kept after the last loud one (300 ms), so word endings are not clipped.
const GATE_HANGOVER: u32 = 15;

/// Encodes frames and drops silent ones (with one frame of lead-in and a 300 ms tail).
#[derive(Debug, Default)]
pub struct MicEncoder {
    state: AdpcmState,
    /// Frames still to keep after the last loud one.
    open_for: u32,
    /// The previous frame, encoded, if it was dropped: sent ahead of a frame that opens the gate
    /// so the start of a word is kept.
    lead_in: Option<(u64, Vec<u8>)>,
    frames: u64,
    out: Vec<u8>,
    /// Frames recorded (not dropped as silence).
    pub kept: u64,
}

impl MicEncoder {
    /// Encode the next frame (its index in the stream is counted here). `emit(frame_index,
    /// record)` is called for each frame to record: none for silence, the lead-in frame and
    /// this one when speech starts.
    pub fn push(&mut self, frame: &[i16], mut emit: impl FnMut(u64, &[u8])) {
        let index = self.frames;
        self.frames += 1;
        encode_frame(&mut self.state, frame, &mut self.out);
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

// ---- capture ----------------------------------------------------------------------------------

/// The longest stretch of samples kept waiting for the game thread (5 s); older ones are
/// dropped (the game thread drains every frame, so this only matters while it is stalled).
const MAX_BUFFERED: usize = SAMPLE_RATE as usize * 5;

#[derive(Debug, Default)]
struct MicShared {
    samples: Mutex<VecDeque<i16>>,
    /// Total samples delivered (at [`SAMPLE_RATE`]).
    delivered: AtomicU64,
    dropped: AtomicU64,
    failed: AtomicBool,
    error: Mutex<Option<String>>,
}

/// The default microphone, open, delivering 24 kHz mono samples. The device is opened and owned
/// by a helper thread (opening a WASAPI device can take a few hundred milliseconds, which must
/// not stall a frame); dropping this closes it.
#[derive(Debug)]
pub struct MicCapture {
    shared: Arc<MicShared>,
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Disconnects when the helper thread ends (its sender is dropped with the thread, panic or
    /// not), so [`MicCapture::stop`] can wait for it with a bound.
    done: Option<mpsc::Receiver<()>>,
}

impl MicCapture {
    /// Start opening the default input device. Never blocks: failures (no microphone, access
    /// denied) show up in [`MicCapture::error`] and simply deliver no samples.
    #[must_use]
    pub fn start() -> MicCapture {
        let shared = Arc::new(MicShared::default());
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let thread_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("mc-rs replay microphone".into())
            .spawn(move || {
                let _done = done_tx;
                let fail = |error: String| {
                    tracing::warn!(%error, "replay microphone unavailable");
                    thread_shared.failed.store(true, Ordering::Relaxed);
                    *thread_shared
                        .error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                };
                // A panic in a device call is the microphone's failure, not
                // the game's crash.
                crate::contained::run_helper(
                    "microphone",
                    || {
                        // The stream lives (and the device stays open) until the stop message or
                        // the handle is dropped.
                        match open_default(&thread_shared) {
                            Ok(stream) => {
                                let _ = stop_rx.recv();
                                drop(stream);
                            }
                            Err(error) => fail(error),
                        }
                    },
                    fail,
                );
            })
            .ok();
        MicCapture {
            shared,
            stop: Some(stop_tx),
            thread,
            done: Some(done_rx),
        }
    }

    /// A capture with no device behind it, fed by [`MicCapture::feed`]: for tests of what is
    /// built on top (the client's timeline). Never opens anything.
    #[must_use]
    pub fn fake() -> MicCapture {
        MicCapture {
            shared: Arc::new(MicShared::default()),
            stop: None,
            thread: None,
            done: None,
        }
    }

    /// Deliver samples (24 kHz mono) as the device would; for a [`MicCapture::fake`].
    pub fn feed(&self, samples: &[i16]) {
        self.feeder().feed(samples);
    }

    /// A handle that delivers samples into this capture from elsewhere (a test keeps it after
    /// handing the capture over).
    #[must_use]
    pub fn feeder(&self) -> MicFeeder {
        MicFeeder {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Move every waiting sample into `out` (appended).
    pub fn drain(&self, out: &mut Vec<i16>) {
        let mut samples = self
            .shared
            .samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        out.extend(samples.drain(..));
    }

    /// Samples delivered since the device opened.
    #[must_use]
    pub fn delivered(&self) -> u64 {
        self.shared.delivered.load(Ordering::Relaxed)
    }

    /// Why the microphone is not delivering, if it failed.
    #[must_use]
    pub fn error(&self) -> Option<String> {
        if !self.shared.failed.load(Ordering::Relaxed) {
            return None;
        }
        self.shared
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Close the device and wait (bounded) for the helper thread.
    pub fn stop(mut self) {
        self.stop_inner(Duration::from_secs(2));
    }

    fn stop_inner(&mut self, timeout: Duration) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            // The thread only drops the stream after the message; joining is normally quick.
            // Bounded as documented: a device call that does not return (a
            // microphone pulled or failing while it was being opened or closed) used to hold
            // the caller, the end of a recording or the game's exit, for as long. The thread is
            // then left to finish on its own.
            let finished = self.done.take().is_none_or(|done| {
                !matches!(
                    done.recv_timeout(timeout),
                    Err(mpsc::RecvTimeoutError::Timeout)
                )
            });
            if finished {
                let _ = thread.join();
            } else {
                tracing::warn!(?timeout, "the replay microphone did not close in time");
            }
        }
    }
}

impl Drop for MicCapture {
    fn drop(&mut self) {
        self.stop_inner(Duration::from_secs(2));
    }
}

/// Delivers samples into a [`MicCapture`] as its device would ([`MicCapture::feeder`]).
#[derive(Debug, Clone)]
pub struct MicFeeder {
    shared: Arc<MicShared>,
}

impl MicFeeder {
    /// Deliver samples (24 kHz mono).
    pub fn feed(&self, samples: &[i16]) {
        self.shared
            .samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(samples.iter().copied());
        self.shared
            .delivered
            .fetch_add(samples.len() as u64, Ordering::Relaxed);
    }
}

/// Converts device samples (any rate, any channel count) to 24 kHz mono.
#[derive(Debug)]
struct Resampler {
    channels: usize,
    /// Input samples per output sample.
    step: f64,
    /// Position of the next output sample, in input samples, relative to `prev`.
    phase: f64,
    /// The previous input sample (mono), for interpolation across callbacks.
    prev: f32,
    /// A one-pole low-pass ahead of the decimation, against aliasing.
    lowpass: f32,
    alpha: f32,
}

impl Resampler {
    fn new(rate: u32, channels: u16) -> Resampler {
        let step = f64::from(rate) / f64::from(SAMPLE_RATE);
        // Cutoff near 10 kHz at the input rate.
        let alpha = if rate > SAMPLE_RATE {
            let dt = 1.0 / f64::from(rate);
            let rc = 1.0 / (2.0 * std::f64::consts::PI * 10_000.0);
            (dt / (rc + dt)) as f32
        } else {
            1.0
        };
        Resampler {
            channels: usize::from(channels.max(1)),
            step,
            phase: 0.0,
            prev: 0.0,
            lowpass: 0.0,
            alpha,
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<i16>) {
        for frame in input.chunks(self.channels) {
            let mono = frame.iter().sum::<f32>() / frame.len() as f32;
            self.lowpass += self.alpha * (mono - self.lowpass);
            let current = self.lowpass;
            // Emit every output sample that falls between `prev` (at 0) and `current` (at 1).
            while self.phase <= 1.0 {
                let t = self.phase as f32;
                let value = self.prev + (current - self.prev) * t;
                out.push((value.clamp(-1.0, 1.0) * 32767.0) as i16);
                self.phase += self.step;
            }
            self.phase -= 1.0;
            self.prev = current;
        }
    }
}

fn open_default(shared: &Arc<MicShared>) -> Result<cpal::Stream, String> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "no default microphone".to_owned())?;
    let supported = device
        .default_input_config()
        .map_err(|e| format!("microphone config: {e}"))?;
    let format = supported.sample_format();
    let config = supported.config();
    tracing::info!(
        rate = config.sample_rate,
        channels = config.channels,
        ?format,
        "replay microphone opening"
    );
    let stream = match format {
        cpal::SampleFormat::F32 => build::<f32>(&device, config, shared),
        cpal::SampleFormat::I16 => build::<i16>(&device, config, shared),
        cpal::SampleFormat::U16 => build::<u16>(&device, config, shared),
        cpal::SampleFormat::I32 => build::<i32>(&device, config, shared),
        cpal::SampleFormat::U8 => build::<u8>(&device, config, shared),
        cpal::SampleFormat::F64 => build::<f64>(&device, config, shared),
        other => return Err(format!("unsupported microphone sample format {other:?}")),
    }?;
    stream
        .play()
        .map_err(|e| format!("microphone start: {e}"))?;
    Ok(stream)
}

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    shared: &Arc<MicShared>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    use cpal::traits::DeviceTrait;
    let mut resampler = Resampler::new(config.sample_rate, config.channels);
    let mut floats: Vec<f32> = Vec::new();
    let mut converted: Vec<i16> = Vec::new();
    let data_shared = Arc::clone(shared);
    let error_shared = Arc::clone(shared);
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data: &[T], _info: &cpal::InputCallbackInfo| {
                floats.clear();
                floats.extend(data.iter().map(|&s| s.to_sample::<f32>()));
                converted.clear();
                resampler.process(&floats, &mut converted);
                data_shared
                    .delivered
                    .fetch_add(converted.len() as u64, Ordering::Relaxed);
                let mut samples = data_shared
                    .samples
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                samples.extend(converted.iter().copied());
                let excess = samples.len().saturating_sub(MAX_BUFFERED);
                if excess > 0 {
                    samples.drain(..excess);
                    data_shared
                        .dropped
                        .fetch_add(excess as u64, Ordering::Relaxed);
                }
            },
            move |error| {
                tracing::warn!(%error, "replay microphone stream error");
                error_shared.failed.store(true, Ordering::Relaxed);
                *error_shared
                    .error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error.to_string());
            },
            None,
        )
        .map_err(|e| format!("microphone stream: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, amplitude: f32, samples: usize) -> Vec<i16> {
        (0..samples)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16
            })
            .collect()
    }

    /// A helper thread standing in for the device one, holding the done sender like it does.
    fn capture_over(thread_body: impl FnOnce() + Send + 'static) -> MicCapture {
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let _done = done_tx;
            thread_body();
        });
        MicCapture {
            shared: Arc::new(MicShared::default()),
            stop: None,
            thread: Some(thread),
            done: Some(done_rx),
        }
    }

    /// A thread stuck in a device call no longer holds `stop` (the end of a
    /// recording, the game's exit): it returns after its bound. The old `stop_inner` ignored the
    /// bound and joined.
    #[test]
    fn a_stuck_microphone_thread_does_not_hold_stop() {
        let (release, released) = mpsc::channel::<()>();
        let mut capture = capture_over(move || {
            let _ = released.recv();
        });
        let started = std::time::Instant::now();
        capture.stop_inner(Duration::from_millis(200));
        let took = started.elapsed();
        let _ = release.send(());
        assert!(took >= Duration::from_millis(150), "{took:?}");
        assert!(took < Duration::from_secs(2), "stop waited {took:?}");
    }

    #[test]
    fn a_finished_microphone_thread_is_still_joined_at_once() {
        let mut capture = capture_over(|| {});
        let started = std::time::Instant::now();
        capture.stop_inner(Duration::from_secs(5));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(capture.thread.is_none());
    }

    /// The decoder reproduces the encoder's own reconstruction exactly, and that reconstruction
    /// tracks a voice-band tone closely (ADPCM is lossy; the bound is its usual quality).
    #[test]
    fn a_frame_round_trips_through_the_codec() {
        let input = sine(440.0, 12_000.0, FRAME_SAMPLES * 4);
        let mut state = AdpcmState::default();
        let mut decoded = Vec::new();
        let mut payload = Vec::new();
        for frame in input.chunks(FRAME_SAMPLES) {
            encode_frame(&mut state, frame, &mut payload);
            assert_eq!(payload.len(), FRAME_BYTES);
            decode_frame(&payload, &mut decoded).expect("decodes");
        }
        assert_eq!(decoded.len(), input.len());
        // Skip the first frame while the step size adapts.
        let err: f64 = input[FRAME_SAMPLES..]
            .iter()
            .zip(&decoded[FRAME_SAMPLES..])
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / (input.len() - FRAME_SAMPLES) as f64;
        let signal: f64 = input[FRAME_SAMPLES..]
            .iter()
            .map(|&a| f64::from(a).powi(2))
            .sum::<f64>()
            / (input.len() - FRAME_SAMPLES) as f64;
        let snr_db = 10.0 * (signal / err).log10();
        assert!(snr_db > 25.0, "SNR {snr_db:.1} dB");
    }

    #[test]
    fn a_damaged_frame_is_refused() {
        let mut out = Vec::new();
        assert!(decode_frame(&[0; FRAME_BYTES - 1], &mut out).is_err());
        let mut bad = vec![0; FRAME_BYTES];
        bad[2] = 89;
        assert!(decode_frame(&bad, &mut out).is_err());
    }

    /// Silence is dropped; speech is kept with one frame of lead-in and a 300 ms tail.
    #[test]
    fn the_gate_drops_silence_and_keeps_the_edges_of_speech() {
        let mut encoder = MicEncoder::default();
        let quiet = vec![3i16; FRAME_SAMPLES];
        let loud = sine(300.0, 8_000.0, FRAME_SAMPLES);
        let mut kept = Vec::new();
        for _ in 0..10 {
            encoder.push(&quiet, |i, _| kept.push(i));
        }
        assert!(kept.is_empty(), "silence is not recorded");
        for _ in 0..3 {
            encoder.push(&loud, |i, _| kept.push(i));
        }
        assert_eq!(
            kept,
            vec![9, 10, 11, 12],
            "the frame before speech is the lead-in"
        );
        kept.clear();
        for _ in 0..40 {
            encoder.push(&quiet, |i, _| kept.push(i));
        }
        assert_eq!(
            kept.len(),
            GATE_HANGOVER as usize,
            "a 300 ms tail, then nothing"
        );
        assert_eq!(kept[0], 13);
    }

    #[test]
    fn the_resampler_makes_24_khz_mono_from_48_khz_stereo() {
        let mut resampler = Resampler::new(48_000, 2);
        let mut out = Vec::new();
        // One second of a 1 kHz tone in both channels, in uneven callback sizes.
        let input: Vec<f32> = (0..48_000)
            .flat_map(|i| {
                let v = (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 48_000.0).sin() * 0.5;
                [v, v]
            })
            .collect();
        for chunk in input.chunks(2 * 441) {
            resampler.process(chunk, &mut out);
        }
        assert!((out.len() as i64 - 24_000).abs() <= 2, "{}", out.len());
        let peak = out.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(peak > 14_000 && peak < 17_000, "peak {peak}");
    }
}
