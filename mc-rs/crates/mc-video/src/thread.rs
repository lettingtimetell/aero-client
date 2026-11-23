//! The encoder on a worker thread: frames in through a bounded queue, buffers recycled, audio
//! pulled from an [`AudioSource`] and interleaved by time, cancel at any moment.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::config::HNS_PER_SECOND;
use crate::{EncodeSummary, EncoderConfig, EncoderInfo, Mp4Writer, VideoError};

/// Audio frames per write. The audio trails the video by less than this (85 ms at 48 kHz), which
/// keeps the MP4 interleaved without ever making the sink writer wait for audio.
const AUDIO_CHUNK_FRAMES: u64 = 4096;

/// Video frames a hardware encoder may hold (handed to Media Foundation, not yet
/// encoded) before the encoder thread waits to hand it the next one. See [`EncoderPace`].
const PACE_FRAMES: u64 = 8;
/// How long the encoder may finish no frame at all while it holds its limit
/// before the wait gives up and lets it hold twice as many: an encoder that needs more frames in
/// hand before it outputs one costs this wait once per doubling, never a stall.
const PACE_STALL: Duration = Duration::from_millis(250);
/// Above this many frames the pacing stops for the file, and the sink writer's
/// own throttling is left alone (as before this pacing existed).
const PACE_MAX_FRAMES: u64 = 64;
/// How often the wait looks at the encoder.
const PACE_POLL: Duration = Duration::from_micros(500);

/// Where the soundtrack comes from: the encoder thread pulls it in step with the video.
pub trait AudioSource: Send {
    /// Fill `out` with the next interleaved 16-bit samples (the configured channel count). Past
    /// the end of the material, fill with silence.
    fn fill(&mut self, out: &mut [i16]);

    /// The same stretch of time as the last [`fill`] call for
    /// extra audio track `track` (0 = the first of [`crate::EncoderConfig::audio_tracks`]),
    /// called right after it once per extra track. Silence unless overridden.
    ///
    /// [`fill`]: AudioSource::fill
    fn fill_track(&mut self, track: usize, out: &mut [i16]) {
        let _ = track;
        out.fill(0);
    }
}

/// An [`AudioSource`] of silence.
#[derive(Debug, Clone, Copy, Default)]
pub struct SilentAudio;

impl AudioSource for SilentAudio {
    fn fill(&mut self, out: &mut [i16]) {
        out.fill(0);
    }
}

/// Counters the encoder thread updates, for a progress bar.
#[derive(Debug, Default)]
pub struct EncoderProgress {
    frames_encoded: AtomicU64,
}

impl EncoderProgress {
    /// Video frames handed to Media Foundation so far.
    #[must_use]
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded.load(Ordering::Relaxed)
    }
}

enum Message {
    Frame(Vec<u8>),
    Finish,
}

/// An [`Mp4Writer`] running on its own thread.
///
/// Dropping it without [`Self::finish`] cancels the encode and deletes the partial file.
pub struct EncoderThread {
    sender: Option<SyncSender<Message>>,
    recycled: Receiver<Vec<u8>>,
    progress: Arc<EncoderProgress>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<EncodeSummary, VideoError>>>,
    info: EncoderInfo,
    frame_bytes: usize,
    submitted: u64,
}

impl EncoderThread {
    /// Start the encoder thread and wait until the file and encoders are set up.
    ///
    /// `queue_frames` bounds how many frames may wait for the encoder (memory: that many NV12
    /// frames). With an audio track in `config`, `audio` supplies it (`None` = silence).
    ///
    /// # Errors
    /// Whatever [`Mp4Writer::create`] reports, or [`VideoError::Io`] if no thread can start.
    pub fn spawn(
        config: EncoderConfig,
        audio: Option<Box<dyn AudioSource>>,
        queue_frames: usize,
    ) -> Result<Self, VideoError> {
        config.validate()?;
        let frame_bytes = config.nv12_frame_bytes();
        let (sender, receiver) = mpsc::sync_channel::<Message>(queue_frames.max(1));
        let (recycle_tx, recycled) = mpsc::channel::<Vec<u8>>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<EncoderInfo, VideoError>>(1);
        let progress = Arc::new(EncoderProgress::default());
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = Worker {
            receiver,
            recycle: recycle_tx,
            progress: Arc::clone(&progress),
            cancel: Arc::clone(&cancel),
        };
        // A panic here reaches the owner through `join` (`VideoError::WorkerGone`), and the
        // export shows it as failed: contained, so it is not also recorded as the game's crash.
        let handle = std::thread::Builder::new()
            .name("mc-video encoder".to_owned())
            .spawn(move || {
                mc_core::panic_scope::contain_for_join(|| worker.run(config, audio, ready_tx))
            })?;
        match ready_rx.recv() {
            Ok(Ok(info)) => Ok(Self {
                sender: Some(sender),
                recycled,
                progress,
                cancel,
                handle: Some(handle),
                info,
                frame_bytes,
                submitted: 0,
            }),
            Ok(Err(error)) => {
                let _ = handle.join();
                Err(error)
            }
            Err(_) => Err(join_error(handle)),
        }
    }

    /// The encoder Media Foundation picked.
    #[must_use]
    pub fn info(&self) -> &EncoderInfo {
        &self.info
    }

    /// Progress counters, shareable with a UI thread.
    #[must_use]
    pub fn progress(&self) -> Arc<EncoderProgress> {
        Arc::clone(&self.progress)
    }

    /// Frames submitted so far.
    #[must_use]
    pub fn frames_submitted(&self) -> u64 {
        self.submitted
    }

    /// Bytes per frame.
    #[must_use]
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// A frame buffer to fill: one the encoder has finished with, or a new one.
    #[must_use]
    pub fn buffer(&self) -> Vec<u8> {
        match self.recycled.try_recv() {
            Ok(mut buffer) => {
                buffer.resize(self.frame_bytes, 0);
                buffer
            }
            Err(_) => vec![0; self.frame_bytes],
        }
    }

    /// Queue the next NV12 frame, waiting while the queue is full.
    ///
    /// # Errors
    /// [`VideoError::FrameSize`], or the encoder thread's own error if it has stopped.
    pub fn submit(&mut self, frame: Vec<u8>) -> Result<(), VideoError> {
        self.check(&frame)?;
        let sender = self.sender.as_ref().ok_or(VideoError::Cancelled)?;
        if sender.send(Message::Frame(frame)).is_err() {
            return Err(self.stopped());
        }
        self.submitted += 1;
        Ok(())
    }

    /// Queue the next NV12 frame if there is room; otherwise hand it back (`Ok(Some(frame))`)
    /// so the caller can keep its window responsive and try again.
    ///
    /// # Errors
    /// As [`Self::submit`].
    pub fn try_submit(&mut self, frame: Vec<u8>) -> Result<Option<Vec<u8>>, VideoError> {
        self.check(&frame)?;
        let sender = self.sender.as_ref().ok_or(VideoError::Cancelled)?;
        match sender.try_send(Message::Frame(frame)) {
            Ok(()) => {
                self.submitted += 1;
                Ok(None)
            }
            Err(TrySendError::Full(Message::Frame(frame))) => Ok(Some(frame)),
            Err(TrySendError::Full(Message::Finish)) => Ok(None),
            Err(TrySendError::Disconnected(_)) => Err(self.stopped()),
        }
    }

    fn check(&self, frame: &[u8]) -> Result<(), VideoError> {
        if frame.len() == self.frame_bytes {
            Ok(())
        } else {
            Err(VideoError::FrameSize {
                got: frame.len(),
                expected: self.frame_bytes,
            })
        }
    }

    /// The thread stopped early: collect its error.
    fn stopped(&mut self) -> VideoError {
        self.sender = None;
        match self.handle.take() {
            Some(handle) => match handle.join() {
                Ok(Err(error)) => error,
                Ok(Ok(_)) => VideoError::WorkerGone("finished early".to_owned()),
                Err(_) => VideoError::WorkerGone("panicked".to_owned()),
            },
            None => VideoError::WorkerGone("already stopped".to_owned()),
        }
    }

    /// Encode what is queued, write the trailing audio, finish the file and rename it into place.
    ///
    /// # Errors
    /// The encoder thread's error; the partial file is deleted.
    pub fn finish(mut self) -> Result<EncodeSummary, VideoError> {
        let sender = self.sender.take().ok_or(VideoError::Cancelled)?;
        let sent = sender.send(Message::Finish).is_ok();
        drop(sender);
        let handle = self
            .handle
            .take()
            .ok_or_else(|| VideoError::WorkerGone("already stopped".to_owned()))?;
        let result = match handle.join() {
            Ok(result) => result,
            Err(_) => Err(VideoError::WorkerGone("panicked".to_owned())),
        };
        if !sent && result.is_ok() {
            return Err(VideoError::WorkerGone(
                "stopped before finishing".to_owned(),
            ));
        }
        result
    }

    /// Stop now: queued frames are dropped and the partial file deleted. Returns once the thread
    /// has stopped (the encoder may take a moment to release the file).
    pub fn cancel(mut self) {
        self.abandon();
    }

    fn abandon(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.sender = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for EncoderThread {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.abandon();
        }
    }
}

fn join_error(handle: JoinHandle<Result<EncodeSummary, VideoError>>) -> VideoError {
    match handle.join() {
        Ok(Err(error)) => error,
        Ok(Ok(_)) => VideoError::WorkerGone("finished before starting".to_owned()),
        Err(_) => VideoError::WorkerGone("panicked while starting".to_owned()),
    }
}

struct Worker {
    receiver: Receiver<Message>,
    recycle: mpsc::Sender<Vec<u8>>,
    progress: Arc<EncoderProgress>,
    cancel: Arc<AtomicBool>,
}

impl Worker {
    fn run(
        self,
        config: EncoderConfig,
        audio: Option<Box<dyn AudioSource>>,
        ready: SyncSender<Result<EncoderInfo, VideoError>>,
    ) -> Result<EncodeSummary, VideoError> {
        let frame_rate = config.frame_rate;
        let audio_config = config.audio;
        let mut writer = match Mp4Writer::create(config) {
            Ok(writer) => {
                let _ = ready.send(Ok(writer.info().clone()));
                writer
            }
            Err(error) => {
                let _ = ready.send(Err(error));
                return Err(VideoError::WorkerGone("could not start".to_owned()));
            }
        };
        drop(ready);
        let mut audio = audio_config.map(|_| audio.unwrap_or_else(|| Box::new(SilentAudio)));
        let mut pcm = Vec::new();
        // A hardware encoder gets its frames at its own pace (`EncoderPace`).
        let mut pace = EncoderPace::new(writer.info().hardware);
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(VideoError::Cancelled);
            }
            match self.receiver.recv() {
                Ok(Message::Frame(frame)) => {
                    if !pace.wait(&writer, &self.cancel) {
                        return Err(VideoError::Cancelled);
                    }
                    if let (Some(cfg), Some(source)) = (audio_config, audio.as_mut()) {
                        // Bring the audio up to this frame's start, in whole chunks.
                        let start = frame_rate.frame_start_hns(writer.frames_written());
                        let target = hns_to_frames(start, cfg.sample_rate);
                        while writer.audio_frames_written() + AUDIO_CHUNK_FRAMES <= target {
                            write_audio(
                                &mut writer,
                                source.as_mut(),
                                cfg.channels,
                                AUDIO_CHUNK_FRAMES,
                                &mut pcm,
                            )?;
                        }
                    }
                    writer.write_frame(&frame)?;
                    self.progress.frames_encoded.fetch_add(1, Ordering::Relaxed);
                    let _ = self.recycle.send(frame);
                }
                Ok(Message::Finish) => {
                    if let (Some(cfg), Some(source)) = (audio_config, audio.as_mut()) {
                        // The audio ends exactly where the video does.
                        let end = frame_rate.frame_start_hns(writer.frames_written());
                        let target = hns_to_frames(end, cfg.sample_rate);
                        while writer.audio_frames_written() < target {
                            let n =
                                (target - writer.audio_frames_written()).min(AUDIO_CHUNK_FRAMES);
                            write_audio(&mut writer, source.as_mut(), cfg.channels, n, &mut pcm)?;
                        }
                    }
                    if self.cancel.load(Ordering::Relaxed) {
                        return Err(VideoError::Cancelled);
                    }
                    return writer.finish();
                }
                // The owner went away without finishing: abandon the file.
                Err(_) => return Err(VideoError::Cancelled),
            }
        }
    }
}

/// Hands a hardware encoder its frames one at a time,
/// at the pace it encodes them.
///
/// Media Foundation's sink writer has a throttle of its own, but a deep one with hysteresis
/// (measured with NVIDIA's encoder and Windows' software one, `examples/sink_probe.rs`): it takes
/// about 2.3 s of video without blocking, then holds `WriteSample` until about 0.3 s more has
/// been encoded (20 frames at 60 fps), then takes that much at once again. The export's render
/// thread waits on this thread through the frame queue, so each hold froze the export window,
/// several times a second (at 60 fps: 50-90 ms at 1080p, 200-260 ms at 4K). Waiting here before
/// every frame while the encoder holds [`PACE_FRAMES`] gives the same pace without the freezes,
/// and the writer holds 8 frames instead of 2.3 s of them (430 MB at 1080p60, 1.7 GB at 4K60).
/// The encoder gets the same frames with the same times in the same order: only when it is
/// handed them changes.
///
/// The sink writer's throttle stays on as the safety net. Windows' software encoder outputs in
/// large batches and nothing for its first hundred-odd frames, so it is not paced ("Processor
/// only" is exactly as before); an encoder that outputs nothing until it holds more than
/// [`PACE_FRAMES`] gets twice as many after [`PACE_STALL`], and the pacing stops for the file
/// past [`PACE_MAX_FRAMES`]. `MC_W19_R1_ENCODER_PACE=0` (read once) switches it off.
struct EncoderPace {
    /// Frames the encoder may hold; `None`: not pacing.
    limit: Option<u64>,
}

impl EncoderPace {
    fn new(hardware: bool) -> Self {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| {
            std::env::var("MC_W19_R1_ENCODER_PACE").map_or(true, |value| value.trim() != "0")
        });
        Self {
            limit: (hardware && enabled).then_some(PACE_FRAMES),
        }
    }

    /// Wait until the encoder holds fewer frames than the limit. `false`: cancelled meanwhile.
    fn wait(&mut self, writer: &Mp4Writer, cancel: &AtomicBool) -> bool {
        let Some(limit) = self.limit else {
            return true;
        };
        let Some(mut held) = writer.video_frames_in_encoder() else {
            return true;
        };
        let mut progress = Instant::now();
        while held >= limit {
            if cancel.load(Ordering::Relaxed) {
                return false;
            }
            std::thread::sleep(PACE_POLL);
            let Some(now) = writer.video_frames_in_encoder() else {
                return true;
            };
            if now < held {
                progress = Instant::now();
            } else if progress.elapsed() >= PACE_STALL {
                // This encoder outputs nothing until it holds more: let it (it cannot be waited
                // on for a frame it is never handed).
                let more = limit * 2;
                self.limit = (more <= PACE_MAX_FRAMES).then_some(more);
                tracing::debug!(
                    frames = ?self.limit,
                    "encoder pacing: the encoder holds more frames"
                );
                return true;
            }
            held = now;
        }
        true
    }
}

fn hns_to_frames(hns: i64, sample_rate: u32) -> u64 {
    (u128::from(hns.max(0) as u64) * u128::from(sample_rate) / u128::from(HNS_PER_SECOND)) as u64
}

fn write_audio(
    writer: &mut Mp4Writer,
    source: &mut dyn AudioSource,
    channels: u16,
    frames: u64,
    pcm: &mut Vec<i16>,
) -> Result<(), VideoError> {
    pcm.resize(frames as usize * usize::from(channels), 0);
    source.fill(pcm);
    writer.write_audio(pcm)?;
    // The extra tracks, for the same stretch of time.
    for track in 0..writer.extra_audio_tracks() {
        source.fill_track(track, pcm);
        writer.write_audio_track(track, pcm)?;
    }
    Ok(())
}
