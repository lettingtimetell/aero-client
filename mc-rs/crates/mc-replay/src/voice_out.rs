//! A small audio output for the replay viewer's recorded voices (the microphone and desktop
//! audio). The game's own sounds play through the game's audio
//! engine; these two streams are PCM, which that engine does not take, so they get an output of
//! their own on the default device, mixed by Windows with the game's.
//!
//! The device is opened and owned by a helper thread (opening can take a few hundred
//! milliseconds, which must not stall a frame), like [`crate::mic::MicCapture`]. The game thread
//! pushes interleaved stereo `f32` frames at the device's rate ([`VoiceOutput::rate`], 0 until the
//! device is open); the device takes them in order and plays silence when there are none.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex, PoisonError};

/// The most audio kept waiting (1 s at 48 kHz stereo); older frames are dropped.
const MAX_QUEUED: usize = 48_000 * 2;

#[derive(Debug, Default)]
struct Shared {
    /// Interleaved stereo frames waiting for the device.
    queue: Mutex<VecDeque<f32>>,
    rate: AtomicU32,
    failed: AtomicBool,
    error: Mutex<Option<String>>,
}

/// The default output device, open for the replay viewer's voices. Dropping it closes it.
#[derive(Debug)]
pub struct VoiceOutput {
    shared: Arc<Shared>,
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl VoiceOutput {
    /// Start opening the default output device. Never blocks; a failure shows in
    /// [`VoiceOutput::error`] and the output then takes frames and drops them.
    #[must_use]
    pub fn start() -> VoiceOutput {
        let shared = Arc::new(Shared::default());
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let thread_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("mc-rs replay voices".into())
            .spawn(move || {
                let fail = |error: String| {
                    tracing::warn!(%error, "replay voice output unavailable");
                    thread_shared.failed.store(true, Ordering::Relaxed);
                    *thread_shared
                        .error
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner) = Some(error);
                };
                // A panic in a device call is this output's failure, not the
                // game's crash.
                crate::contained::run_helper(
                    "replay voice output",
                    || match open_default(&thread_shared) {
                        Ok(stream) => {
                            let _ = stop_rx.recv();
                            drop(stream);
                        }
                        Err(error) => fail(error),
                    },
                    fail,
                );
            })
            .ok();
        VoiceOutput {
            shared,
            stop: Some(stop_tx),
            thread,
        }
    }

    /// An output with no device behind it, at `rate`: what is pushed waits in the queue (tests
    /// read it back with [`VoiceOutput::take_queued`]). Never opens anything.
    #[must_use]
    pub fn fake(rate: u32) -> VoiceOutput {
        let shared = Arc::new(Shared::default());
        shared.rate.store(rate, Ordering::Relaxed);
        VoiceOutput {
            shared,
            stop: None,
            thread: None,
        }
    }

    /// The device's sample rate; 0 until it is open (or if it failed).
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.shared.rate.load(Ordering::Relaxed)
    }

    /// Queue interleaved stereo frames to play after what is already queued.
    pub fn push(&self, stereo: &[f32]) {
        let mut queue = self
            .shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        queue.extend(stereo.iter().copied());
        let excess = queue.len().saturating_sub(MAX_QUEUED);
        if excess > 0 {
            queue.drain(..excess + excess % 2);
        }
    }

    /// Stereo frames queued and not yet played.
    #[must_use]
    pub fn queued_frames(&self) -> usize {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
            / 2
    }

    /// Drop everything queued (a seek, a pause).
    pub fn clear(&self) {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Everything queued, removed (for a [`VoiceOutput::fake`] in tests).
    #[must_use]
    pub fn take_queued(&self) -> Vec<f32> {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain(..)
            .collect()
    }

    /// Why the device is not playing, if it failed.
    #[must_use]
    pub fn error(&self) -> Option<String> {
        if !self.shared.failed.load(Ordering::Relaxed) {
            return None;
        }
        self.shared
            .error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Drop for VoiceOutput {
    /// Never waits (the game thread drops it when a replay closes): the helper thread closes the
    /// device by itself once told.
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        drop(self.thread.take());
    }
}

fn open_default(shared: &Arc<Shared>) -> Result<cpal::Stream, String> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default output device".to_owned())?;
    let supported = device
        .default_output_config()
        .map_err(|e| format!("output config: {e}"))?;
    let format = supported.sample_format();
    let config = supported.config();
    let stream = match format {
        cpal::SampleFormat::F32 => build::<f32>(&device, config, shared),
        cpal::SampleFormat::I16 => build::<i16>(&device, config, shared),
        cpal::SampleFormat::U16 => build::<u16>(&device, config, shared),
        cpal::SampleFormat::I32 => build::<i32>(&device, config, shared),
        other => return Err(format!("unsupported output sample format {other:?}")),
    }?;
    stream.play().map_err(|e| format!("output start: {e}"))?;
    shared.rate.store(config.sample_rate, Ordering::Relaxed);
    tracing::info!(
        rate = config.sample_rate,
        channels = config.channels,
        ?format,
        "replay voice output open"
    );
    Ok(stream)
}

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + cpal::FromSample<f32> + Send + 'static,
{
    use cpal::traits::DeviceTrait;
    let channels = usize::from(config.channels.max(1));
    let data_shared = Arc::clone(shared);
    device
        .build_output_stream::<T, _, _>(
            config,
            move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                let mut queue = data_shared
                    .queue
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                for frame in data.chunks_mut(channels) {
                    let (left, right) = if queue.len() >= 2 {
                        (
                            queue.pop_front().unwrap_or(0.0),
                            queue.pop_front().unwrap_or(0.0),
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    for (index, sample) in frame.iter_mut().enumerate() {
                        let value = match index {
                            0 => left,
                            1 => right,
                            _ => 0.0,
                        };
                        *sample = <T as cpal::Sample>::from_sample(value.clamp(-1.0, 1.0));
                    }
                }
            },
            |error| tracing::warn!(%error, "replay voice output stream error"),
            None,
        )
        .map_err(|e| format!("output stream: {e}"))
}
