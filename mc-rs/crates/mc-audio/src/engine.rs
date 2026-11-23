//! Playback orchestration: turns a [`PlayRequest`] into mixed, attenuated, categorised audio.
//!
// java: SoundManager.playSound / getNormalizedVolume / getNormalizedPitch

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use kira::backend::{Backend, DefaultBackend};
use kira::sound::static_sound::{StaticSoundData, StaticSoundSettings};
use kira::sound::streaming::StreamingSoundData;
use kira::sound::PlaybackState;
use kira::{AudioManager, AudioManagerSettings, Decibels, Frame, Tween};
use mc_assets::{ResourceLocation, ResourceManager};
use mc_math::{DVec3, JavaRandom};

use crate::attenuation::{attenuate, pan, AttenuationType, Listener};
use crate::category::{CategoryVolumes, SoundCategory};
use crate::gains::EventGains;
use crate::ogg::{decode_all, frames_from_last_page, VorbisDecoder};
use crate::output_recovery::{Look, OutputHeartbeat, OutputWatch};
use crate::sounds::{ResolvedVariant, SoundRegistry};
use crate::AudioError;

/// One request to play a sound event — the Rust analogue of vanilla's `ISound`/
/// `PositionedSound`/`PositionedSoundRecord`.
///
/// Position and attenuation are independent in vanilla's `ISound`: most world sounds are
/// positioned and linear, while GUI/music sounds use no position and no attenuation. Keeping the
/// two fields independent also represents `MovingSoundMinecartRiding`, which is positioned by the
/// base class but explicitly selects [`AttenuationType::None`].
// java: PositionedSound / PositionedSoundRecord
#[derive(Debug, Clone, PartialEq)]
pub struct PlayRequest {
    /// The registered event name, e.g. `minecraft:random.click`.
    pub event: ResourceLocation,
    /// `ISound.getVolume()` — a multiplier, not a `[0,1]` fraction; a value `> 1.0` also widens
    /// the attenuation fade distance (see [`crate::attenuation::fade_distance`]).
    pub volume: f32,
    /// `ISound.getPitch()` — a playback-rate multiplier, clamped with the chosen variant's own
    /// pitch to `[0.5, 2.0]` before reaching the backend.
    pub pitch: f32,
    /// World position, or `None` for a non-positional (ambient) sound.
    pub position: Option<DVec3>,
    /// `ISound.getAttenuationType()`, independent from whether a position is present.
    pub attenuation: AttenuationType,
    /// `ISound.canRepeat()`. Positive delays describe a new scheduled play after the current one
    /// ends; only a zero-delay repeat can use an equivalent OpenAL/Kira sample loop.
    pub repeat: bool,
    /// `ISound.getRepeatDelay()`, in client ticks.  Vanilla's `SoundManager` only schedules the
    /// repeat when this is positive, but preserves the signed Java `int` value on the sound.
    pub repeat_delay_ticks: i32,
}

impl PlayRequest {
    /// `PositionedSoundRecord.create(loc, x, y, z)`: volume `4.0`, pitch `1.0`, linear.
    pub fn positional(event: ResourceLocation, position: DVec3) -> Self {
        Self {
            event,
            volume: 4.0,
            pitch: 1.0,
            position: Some(position),
            attenuation: AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        }
    }

    /// `PositionedSoundRecord.create(loc)`: volume/pitch `1.0`, non-positional and unattenuated.
    pub fn ambient(event: ResourceLocation) -> Self {
        Self {
            event,
            volume: 1.0,
            pitch: 1.0,
            position: None,
            attenuation: AttenuationType::None,
            repeat: false,
            repeat_delay_ticks: 0,
        }
    }

    /// `PositionedSoundRecord.create(loc, pitch)`: volume `0.25`, caller pitch, unattenuated.
    pub fn ambient_with_pitch(event: ResourceLocation, pitch: f32) -> Self {
        Self {
            event,
            volume: 0.25,
            pitch,
            position: None,
            attenuation: AttenuationType::None,
            repeat: false,
            repeat_delay_ticks: 0,
        }
    }

    /// `new PositionedSoundRecord(loc, volume, pitch, x, y, z)`: caller values, linear.
    pub fn at(event: ResourceLocation, volume: f32, pitch: f32, position: DVec3) -> Self {
        Self {
            event,
            volume,
            pitch,
            position: Some(position),
            attenuation: AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        }
    }

    /// Sets the two `ISound` repeat accessors without losing a negative or zero Java delay.
    ///
    /// The engine's future `SoundManager`-style tick scheduler owns the *timing* consequence;
    /// keeping it on the request means tickable/moving sounds can now describe vanilla's
    /// `canRepeat()`/`getRepeatDelay()` contract exactly.
    pub fn with_repeat(mut self, repeat: bool, repeat_delay_ticks: i32) -> Self {
        self.repeat = repeat;
        self.repeat_delay_ticks = repeat_delay_ticks;
        self
    }
}

/// The fully-resolved numeric result of "what should play, how loud, panned which way" — every
/// number [`compute_playback`] can determine without touching a decoder or an audio backend.
/// Deliberately backend-independent so it can be unit-tested on its own.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputedPlayback {
    /// The `.ogg` resource to load and decode.
    pub file: ResourceLocation,
    /// The event's mixer category, for callers that want to label/filter playing sounds.
    pub category: SoundCategory,
    /// Final linear gain in `[0.0, 1.0]`, already folding in the sound's own volume, the chosen
    /// variant's volume, the category slider, and distance attenuation. The master fader is
    /// **not** included — [`AudioEngine`] applies that once, globally, via the mixer's main
    /// track, matching `SoundSystem.setMasterVolume`. An [`EventGains`] entry (the Sounds mod)
    /// multiplies it last, so it reaches up to [`crate::MAX_EVENT_GAIN`] for a boosted event.
    pub linear_gain: f32,
    /// Final playback rate, `[0.5, 2.0]`, mapped 1:1 to kira's `PlaybackRate` (which — like
    /// vanilla's OpenAL pitch — changes speed and pitch together).
    pub pitch: f32,
    /// Simple stereo pan, `[-1.0, 1.0]`. See [`crate::attenuation::pan`]'s doc note: this has no
    /// vanilla source to port, since real OpenAL spatialises internally.
    pub pan: f32,
    /// Whether to decode incrementally (music/records) or fully up front (short effects).
    pub streaming: bool,
    /// The original `ISound.canRepeat()` value.  Playback scheduling deliberately consumes this
    /// separately from one-shot decode/mixing.
    pub repeat: bool,
    /// The original signed `ISound.getRepeatDelay()` tick count.
    pub repeat_delay_ticks: i32,
}

/// A client-side sound whose request can change once per game tick.
///
/// This is the Rust counterpart of vanilla's `ITickableSound`: the engine calls [`Self::update`]
/// before refreshing the channel, then reads a new [`Self::request`] for volume, pitch and source
/// position. The event location should remain stable for the lifetime of a sound, just as
/// `PositionedSound.positionedSoundLocation` is final in 1.8.9; changing it ends the current
/// channel rather than silently playing the old file with the new event's metadata.
// java: ITickableSound / MovingSound
pub trait TickableSound {
    /// Advances the moving sound's state by one client tick.
    fn update(&mut self);

    /// Whether the sound should be stopped on this tick.
    // java: ITickableSound.isDonePlaying
    fn is_done_playing(&self) -> bool;

    /// The current `ISound` values to apply to the live channel.
    fn request(&self) -> PlayRequest;
}

/// `SoundManager.getNormalizedVolume`: `clamp(sound.volume * entry.volume, 0, 1) * category`.
pub fn normalized_volume(sound_volume: f32, entry_volume: f32, category_volume: f32) -> f32 {
    (sound_volume * entry_volume).clamp(0.0, 1.0) * category_volume
}

/// `SoundManager.getNormalizedPitch`: `clamp(sound.pitch * entry.pitch, 0.5, 2.0)`.
pub fn normalized_pitch(sound_pitch: f32, entry_pitch: f32) -> f32 {
    (sound_pitch * entry_pitch).clamp(0.5, 2.0)
}

/// Resolves a [`PlayRequest`] against the registry, category volumes, and listener position,
/// exactly as `SoundManager.playSound` does before it ever touches `paulscode.sound.SoundSystem`.
///
/// Returns `Ok(None)` for every case vanilla logs-and-drops rather than erroring: an unregistered
/// event still surfaces as [`AudioError::UnknownEvent`] (vanilla logs a warning there too, but it
/// indicates a real integration bug — a packet referencing a sound `sounds.json` never
/// registered — worth surfacing to the caller), while a resolved-but-silent sound (volume clamped
/// to `0.0`, or attenuated to nothing at this distance) returns `Ok(None)`.
// java: SoundManager.playSound
pub fn compute_playback(
    registry: &SoundRegistry,
    categories: &CategoryVolumes,
    listener: &Listener,
    request: &PlayRequest,
    rng: &mut JavaRandom,
) -> Result<Option<ComputedPlayback>, AudioError> {
    // Keep the distinction vanilla makes between an event that was never registered and a
    // registered event whose variant pool is empty.  The registry loader normally prevents the
    // latter, but an empty `sounds.json` entry is still a useful diagnostic boundary.
    if registry.category_of(&request.event).is_none() {
        return Err(AudioError::UnknownEvent(request.event.to_string()));
    }
    let variant = registry
        .pick_variant(&request.event, rng)
        .ok_or_else(|| AudioError::EmptyEvent(request.event.to_string()))?;
    let computed = compute_with_variant(registry, categories, listener, request, &variant, 1.0)?;
    if computed.linear_gain <= 0.0 {
        return Ok(None);
    }
    Ok(Some(computed))
}

/// Computes a channel from a variant which has already been selected.
///
/// `SoundManager` stores the chosen `SoundPoolEntry` alongside each playing `ISound`; moving the
/// listener or changing a category must not roll a new weighted variant. Keeping this helper
/// separate from [`compute_playback`] makes that distinction explicit and gives the per-tick
/// registry a stable file/volume/pitch source.
///
/// `event_gain` is the event's [`EventGains`] entry (`1.0` = vanilla, and then every number is
/// exactly vanilla's: `x * 1.0 == x`).
fn compute_with_variant(
    registry: &SoundRegistry,
    categories: &CategoryVolumes,
    listener: &Listener,
    request: &PlayRequest,
    variant: &ResolvedVariant,
    event_gain: f32,
) -> Result<ComputedPlayback, AudioError> {
    let category = registry
        .category_of(&request.event)
        .ok_or_else(|| AudioError::UnknownEvent(request.event.to_string()))?;

    Ok(compute_with_category(
        categories, listener, request, variant, category, event_gain,
    ))
}

fn compute_with_category(
    categories: &CategoryVolumes,
    listener: &Listener,
    request: &PlayRequest,
    variant: &ResolvedVariant,
    category: SoundCategory,
    event_gain: f32,
) -> ComputedPlayback {
    let category_volume = categories.effective(category);
    let base = normalized_volume(request.volume, variant.volume, category_volume);

    let (distance, pan_value) = match request.position {
        Some(pos) => {
            let distance = listener.distance_to(pos);
            (distance, pan(listener, pos))
        }
        None => (0.0, 0.0),
    };
    let attenuation = attenuate(distance, request.attenuation, request.volume);

    // The Sounds mod's per-event gain goes last: after vanilla's clamp, and not into the
    // request's volume, which would also widen the fade distance (see `crate::gains`).
    let gain = base * attenuation * event_gain;

    ComputedPlayback {
        file: variant.location.clone(),
        category,
        linear_gain: gain,
        pitch: normalized_pitch(request.pitch, variant.pitch),
        pan: pan_value,
        streaming: variant.streaming,
        repeat: request.repeat,
        repeat_delay_ticks: request.repeat_delay_ticks,
    }
}

/// Kira can own a zero-delay loop, but a positive delay must return to the scheduler between
/// iterations. Keeping this policy in one predicate prevents an accidental backend loop from
/// bypassing vanilla's delayed-repeat ticks.
fn backend_loops(computed: &ComputedPlayback) -> bool {
    computed.repeat && computed.repeat_delay_ticks == 0
}

fn static_settings(computed: &ComputedPlayback, db: Decibels) -> StaticSoundSettings {
    let settings = StaticSoundSettings::new()
        .volume(db)
        .playback_rate(computed.pitch as f64)
        .panning(computed.pan);
    if backend_loops(computed) {
        settings.loop_region(0.0..)
    } else {
        settings
    }
}

/// Converts a linear amplitude in `[0.0, 1.0]` to the decibels kira's volume settings expect —
/// the exact inverse of [`Decibels::as_amplitude`], which kira does not itself expose (only the
/// dB→amplitude direction).
pub fn amplitude_to_decibels(amplitude: f32) -> Decibels {
    // NaN is silence too: an S29 volume is a raw server float, and a NaN
    // one survives `normalized_volume`'s clamp, so it used to reach kira as NaN decibels, which is
    // NaN samples in the whole mix. Vanilla's OpenAL refuses a NaN gain outright.
    if amplitude.is_nan() || amplitude <= 0.0 {
        Decibels::SILENCE
    } else {
        Decibels(20.0 * amplitude.log10())
    }
}

/// What [`AudioEngine::play`] did with one request.
///
/// Vanilla has no equivalent — `SoundManager.playSound` either logs and returns or hands the
/// sound to `paulscode`, and nothing records which. This exists because "did the game *ask* for
/// the right sound" and "did a speaker move" are separable questions, and on a host with no
/// output device (CI, this sandbox) only the first one can be answered at all. See
/// [`AudioEngine::recent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayOutcome {
    /// Resolved and handed to the backend.
    Played,
    /// Resolved, but no backend is running ([`AudioEngine::is_silent`]). Everything except the
    /// final `kira` call ran.
    NoBackend,
    /// Resolved to silence before the backend was reached: the master/category slider is at
    /// zero, the sound attenuated to nothing at this distance, or the Sounds mod muted the event
    /// ([`EventGains`]). java: the `> 0.0F` guards in `SoundManager.playSound`.
    Inaudible,
    /// The registry has no such event, or the event has an empty variant pool. java:
    /// `SoundManager.playSound`'s `"Unable to play unknown soundEvent"` warning.
    Unresolved,
    /// The `.ogg` could not be loaded or decoded, or `kira` refused it.
    Failed,
}

/// One entry in [`AudioEngine::recent`]: a request the engine was asked to play, what
/// [`compute_playback`] made of it, and what happened next.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayRecord {
    /// The request exactly as the caller built it.
    pub request: PlayRequest,
    /// The resolved numbers, or `None` if resolution failed or produced silence.
    pub computed: Option<ComputedPlayback>,
    /// How far the request got.
    pub outcome: PlayOutcome,
}

/// How many [`PlayRecord`]s [`AudioEngine`] keeps. Small and fixed: this is a diagnostic tail,
/// not a log — a busy server can send hundreds of sound packets a second and an unbounded
/// history would be a slow leak.
pub const RECENT_PLAY_CAPACITY: usize = 32;

/// A handle to one currently-playing sound, static or streaming.
///
/// The inner Kira handle is shared with [`AudioEngine`]'s playing-sound registry. Kira handles
/// are command writers, so sharing one behind a mutex preserves the old caller-facing ability to
/// stop a returned sound while also keeping it alive long enough for the SoundManager-style tick
/// pass to observe completion and refresh spatial values.
#[derive(Debug, Clone)]
pub enum PlayingSound {
    /// A fully-buffered short effect.
    Static(Arc<Mutex<kira::sound::static_sound::StaticSoundHandle>>),
    /// An incrementally-decoded music track or record.
    Streaming(Arc<Mutex<kira::sound::streaming::StreamingSoundHandle<AudioError>>>),
}

/// Decoded frames retained before the effect cache starts over: about 96 MiB at 8 bytes a frame,
/// several times what a play session touches, so a clear here is a backstop and not a policy.
const MAX_CACHED_SOUND_FRAMES: usize = 12 * 1024 * 1024;

/// An effect longer than this (30 s at 44.1 kHz; vanilla's longest,
/// `mob.enderdragon.end`, is 18 s) is never decoded into the cache. It stays compressed in memory
/// and every play streams it: a three-minute clip would take 63 MB as frames and push the whole
/// PvP set out of the cache, against 3 MB compressed.
const ALWAYS_STREAMED_FRAMES: usize = 30 * 44_100;

/// Compressed bytes kept for always-streamed clips before new ones are read on each play instead.
const MAX_STREAMED_BYTES: usize = 64 * 1024 * 1024;

/// Compressed bytes the read-ahead keeps (every vanilla effect is 6.5 MB together).
const MAX_READ_AHEAD_BYTES: usize = 32 * 1024 * 1024;

/// Worker results taken in per drain (per tick or play), so a burst costs a tick ~0.1 ms at most.
const MAX_RESULTS_PER_DRAIN: usize = 64;

/// Short effects, decoded once.
///
/// Re-reading and re-decoding on every play costs about a millisecond of game-thread time per hit
/// or step. java: `SoundSystem` keeps a loaded sound's buffer until `unloadSoundSystem`, which is
/// why this is emptied at exactly that boundary and never earlier.
///
/// Nothing here decodes a clip on the caller's thread (the game
/// thread) any more. A file that is not decoded yet goes to [`DecodeWorker`] with its compressed
/// bytes, and until that decode lands every play of the file streams from those bytes
/// ([`Prepared::Stream`]). A three-minute effect from a resource pack used to stall the game
/// thread for about 250 ms at its first play. java: paulscode decodes on its own command thread.
/// A clip longer than [`ALWAYS_STREAMED_FRAMES`] is never decoded here at all: its compressed
/// bytes are kept and every play streams them.
#[derive(Default)]
struct DecodedSoundCache {
    sounds: HashMap<ResourceLocation, (u32, Arc<[Frame]>)>,
    frames: usize,
    /// Files queued on [`Self::worker`] and not decoded yet, with their compressed bytes, so a
    /// play before the decode lands streams from memory instead of reading the pack again.
    pending: HashMap<ResourceLocation, PendingDecode>,
    /// Files the worker could not decode, with the reason: a play fails the way an inline decode
    /// failure always did, without reading and decoding the file again.
    failed: HashMap<ResourceLocation, String>,
    /// Clips longer than [`ALWAYS_STREAMED_FRAMES`]: their compressed bytes, streamed by every
    /// play, and the total held ([`MAX_STREAMED_BYTES`]).
    streamed: HashMap<ResourceLocation, Arc<[u8]>>,
    streamed_bytes: usize,
    /// Read ahead by the worker after the prewarm, not decoded: a first play streams these from
    /// memory (and queues the decode) instead of reading the disk on the game thread. Bounded
    /// by [`MAX_READ_AHEAD_BYTES`].
    read_ahead: HashMap<ResourceLocation, Arc<[u8]>>,
    read_ahead_bytes: usize,
    /// What the worker (or a read here) found out about which files exist, not yet applied to
    /// the registry ([`AudioEngine::drain_worker`]): a pick never asks the disk itself.
    existence: Vec<(ResourceLocation, bool)>,
    /// Started on first use. [`Self::clear`] drops it, which ends the thread after its current
    /// file and discards that result, so a reload can never receive the previous pack's frames.
    worker: Option<DecodeWorker>,
}

/// A file waiting for [`DecodeWorker`].
struct PendingDecode {
    /// Its compressed bytes, once this thread has them: a play's own read, or a zip entry the
    /// prewarm read. `None` while only the worker is reading the file from disk.
    bytes: Option<Arc<[u8]>>,
    /// A play asked for it, rather than only [`AudioEngine::prewarm_static_sounds`]. Decides what
    /// happens when the decoded frames would overflow the cache (see
    /// [`DecodedSoundCache::insert`]).
    played: bool,
}

/// Where [`DecodeWorker`] gets a file's compressed bytes.
enum Source {
    /// Already read on the caller's thread.
    Bytes(Arc<[u8]>),
    /// A plain file the worker reads itself, at most `cap` bytes, so the game thread does not
    /// wait on the disk ([`mc_assets::ResourceManager::file_path`]).
    File { path: std::path::PathBuf, cap: u64 },
}

/// What [`DecodeWorker`] should make of a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Want {
    /// Decode it (a play or the prewarm asked).
    Frames,
    /// Only read it (the read-ahead): a first play then needs no disk access.
    Bytes,
    /// Only check that it exists (a streamed file: music, records).
    Exists,
}

/// What [`DecodeWorker`] made of one file.
enum Decoded {
    /// Sample rate and frames.
    Frames(u32, Arc<[Frame]>),
    /// Longer than [`ALWAYS_STREAMED_FRAMES`]: not decoded; the compressed bytes, to stream.
    Long(Arc<[u8]>),
    /// Read ahead ([`Want::Bytes`]): the compressed bytes, not decoded yet.
    Read(Arc<[u8]>),
    /// The bytes are not Ogg Vorbis (or the decoder panicked), with the reason.
    Undecodable(String),
    /// The worker could not read the file: it leaves the pools, as vanilla's load-time open
    /// would have dropped it.
    Unreadable,
    /// [`Want::Exists`]: whether the file is there.
    Exists(bool),
}

/// What [`DecodedSoundCache::prepare`] has for one play.
enum Prepared {
    /// Decoded: play it from memory as a static sound.
    Decoded(u32, Arc<[Frame]>),
    /// Not decoded yet: stream it from its compressed bytes. Its last Ogg page gave its length,
    /// which kira needs before the first frame.
    Stream(Arc<[u8]>),
}

/// One file [`DecodeWorker`] finished.
type DecodeOutcome = (ResourceLocation, Decoded);

/// One job for [`DecodeWorker`].
type DecodeJob = (ResourceLocation, Source, Want);

/// The decode thread: one per sound system, started on first use, files in the order queued.
struct DecodeWorker {
    jobs: std::sync::mpsc::Sender<DecodeJob>,
    done: std::sync::mpsc::Receiver<DecodeOutcome>,
}

impl DecodeWorker {
    /// `None` if the thread could not be started; callers then decode inline, as before.
    fn start() -> Option<Self> {
        let (jobs, queue) = std::sync::mpsc::channel::<DecodeJob>();
        let (finished, done) = std::sync::mpsc::channel::<DecodeOutcome>();
        std::thread::Builder::new()
            .name("mc-sound-decode".to_owned())
            .spawn(move || {
                // Ends when the cache drops `jobs` (reload, engine drop) or stops listening.
                while let Ok((file, source, want)) = queue.recv() {
                    let decoded = match (want, source) {
                        (Want::Exists, Source::File { path, .. }) => {
                            Decoded::Exists(path.is_file())
                        }
                        (Want::Exists, Source::Bytes(_)) => Decoded::Exists(true),
                        (want, Source::Bytes(bytes)) => classify(&file, bytes, want),
                        (want, Source::File { path, cap }) => match read_plain_file(&path, cap) {
                            Some(bytes) => classify(&file, Arc::from(bytes), want),
                            None => Decoded::Unreadable,
                        },
                    };
                    if finished.send((file, decoded)).is_err() {
                        return;
                    }
                }
            })
            .ok()?;
        Some(Self { jobs, done })
    }
}

/// Every `.ogg` under `sounds/` that a resource pack carries itself, in any namespace. The
/// launcher's asset store lists nothing (`ResourcePack::list_files`), so this is exactly what zip
/// and folder packs add or replace; a zip answers from its entry list, a folder by walking its
/// `sounds` directory.
fn files_supplied_by_packs(resources: &ResourceManager) -> HashSet<ResourceLocation> {
    let mut files = HashSet::new();
    for namespace in resources.namespaces() {
        for pack in resources.list_files_by_pack(namespace, "sounds/") {
            files.extend(
                pack.into_iter()
                    .filter(|path| path.ends_with(".ogg"))
                    .map(|path| ResourceLocation::new(namespace, path)),
            );
        }
    }
    files
}

/// What the worker makes of a file it has the bytes of.
fn classify(file: &ResourceLocation, bytes: Arc<[u8]>, want: Want) -> Decoded {
    if is_always_streamed(&bytes) {
        Decoded::Long(bytes)
    } else if want == Want::Bytes {
        Decoded::Read(bytes)
    } else {
        decode_on_worker(file, &bytes)
    }
}

/// Whether a clip is longer than [`ALWAYS_STREAMED_FRAMES`], by its last Ogg page.
fn is_always_streamed(bytes: &[u8]) -> bool {
    frames_from_last_page(bytes).is_some_and(|frames| frames > ALWAYS_STREAMED_FRAMES)
}

/// One whole-clip decode on the worker. A decoder panic on a hostile resource-pack file is
/// contained and reported like a decode error; a play of that file then fails as it always did.
fn decode_on_worker(file: &ResourceLocation, bytes: &[u8]) -> Decoded {
    let name = file.to_string();
    contain_prewarm_panic(|| {
        Some(match decode_all(bytes, &name) {
            Ok(decoded) => Decoded::Frames(decoded.sample_rate, Arc::from(decoded.to_frames())),
            Err(AudioError::Decode { reason, .. }) => Decoded::Undecodable(reason),
            Err(other) => Decoded::Undecodable(other.to_string()),
        })
    })
    .unwrap_or_else(|| Decoded::Undecodable("the Vorbis decoder panicked".to_owned()))
}

/// Reads at most `cap` bytes of a plain file, as `ResourceManager::load` would; `None` if it
/// cannot be read or is larger than that.
fn read_plain_file(path: &std::path::Path, cap: u64) -> Option<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path).ok()?;
    let size = file
        .metadata()
        .map_or(0, |metadata| metadata.len())
        .min(cap);
    let mut bytes = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
    file.take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .ok()?;
    (bytes.len() as u64 <= cap).then_some(bytes)
}

thread_local! {
    /// Set while the prewarm thread decodes one file inside [`contain_prewarm_panic`].
    static CONTAINING_PREWARM_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the current thread is decoding a prewarm sound inside a panic boundary.
///
/// The client's panic hook records a crash report for any panic on any thread. A decoder panic
/// during prewarm is contained instead: the file is skipped and the session goes on, so the hook
/// must not turn it into a crash.
#[must_use]
pub fn prewarm_panic_is_being_contained() -> bool {
    CONTAINING_PREWARM_PANIC.with(std::cell::Cell::get)
}

/// Runs one prewarm decode, turning a panic inside it into `None`. Also used around the game
/// thread's inline decode and the streaming decoder's reads (`ogg.rs`).
pub(crate) fn contain_prewarm_panic<T>(decode: impl FnOnce() -> Option<T>) -> Option<T> {
    // The previous value is restored, not cleared: the
    // decode API contains itself, so this runs nested inside the engine's own containment.
    let previous = CONTAINING_PREWARM_PANIC.with(|flag| flag.replace(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(decode));
    CONTAINING_PREWARM_PANIC.with(|flag| flag.set(previous));
    result.ok().flatten()
}

impl DecodedSoundCache {
    /// How to play `file` now: its decoded frames, or, until the worker has decoded it, its
    /// compressed bytes to stream (always, for a clip over [`ALWAYS_STREAMED_FRAMES`]). The bytes
    /// come from memory when the worker already read them (prewarm, read-ahead, a queued play);
    /// otherwise the file is read here, as before. A clip is never decoded on this thread when
    /// its last Ogg page gives its length, which every well-formed file's does. Without one (a
    /// malformed, chained or cut stream) kira cannot be told the length up front, so that file
    /// is decoded here as it always was.
    fn prepare(
        &mut self,
        file: &ResourceLocation,
        resources: &ResourceManager,
    ) -> Result<Prepared, AudioError> {
        self.drain_finished();
        if let Some((sample_rate, frames)) = self.sounds.get(file) {
            return Ok(Prepared::Decoded(*sample_rate, Arc::clone(frames)));
        }
        if let Some(bytes) = self.streamed.get(file) {
            return Ok(Prepared::Stream(Arc::clone(bytes)));
        }
        if let Some(reason) = self.failed.get(file) {
            return Err(AudioError::Decode {
                location: file.to_string(),
                reason: reason.clone(),
            });
        }
        let pending = self.pending.get_mut(file).map(|pending| {
            pending.played = true;
            pending.bytes.clone()
        });
        let queued = pending.is_some();
        let read_ahead = self.read_ahead.remove(file).inspect(|bytes| {
            self.read_ahead_bytes -= bytes.len();
        });
        let bytes: Arc<[u8]> = match pending.flatten().or(read_ahead) {
            Some(bytes) => bytes,
            None => {
                let bytes: Arc<[u8]> = match resources.load(file) {
                    Ok(bytes) => Arc::from(bytes),
                    Err(source) => {
                        // A presumed file that is not there leaves the pools at the next pick.
                        if matches!(source, mc_assets::AssetError::NotFound(_)) {
                            self.existence.push((file.clone(), false));
                        }
                        return Err(AudioError::Asset {
                            location: file.to_string(),
                            source,
                        });
                    }
                };
                // The worker is still reading it from disk: keep this read for the next play.
                if let Some(pending) = self.pending.get_mut(file) {
                    pending.bytes = Some(Arc::clone(&bytes));
                }
                bytes
            }
        };
        // `VorbisDecoder::new` reads the same last page, so a stream never falls back to its
        // counting decode on this thread.
        if is_always_streamed(&bytes) {
            self.keep_streamed(file.clone(), Arc::clone(&bytes));
            return Ok(Prepared::Stream(bytes));
        }
        if frames_from_last_page(&bytes).is_some()
            && (queued || self.queue(file, Source::Bytes(Arc::clone(&bytes)), Want::Frames, true))
        {
            return Ok(Prepared::Stream(bytes));
        }
        // A decoder panic on a hostile pack file is contained
        // here as on the worker: this play fails and the game goes on.
        let name = file.to_string();
        let decoded =
            contain_prewarm_panic(|| Some(decode_all(&bytes, &name))).unwrap_or_else(|| {
                Err(AudioError::Decode {
                    location: name.clone(),
                    reason: "the Vorbis decoder panicked".to_owned(),
                })
            })?;
        let frames: Arc<[Frame]> = Arc::from(decoded.to_frames());
        self.insert(file.clone(), decoded.sample_rate, Arc::clone(&frames), true);
        Ok(Prepared::Decoded(decoded.sample_rate, frames))
    }

    /// Hands `file` to the worker, starting it if needed. `false` if no thread could be started
    /// (the caller then decodes inline, as before).
    fn queue(&mut self, file: &ResourceLocation, source: Source, want: Want, played: bool) -> bool {
        if self.worker.is_none() {
            self.worker = DecodeWorker::start();
        }
        let Some(worker) = &self.worker else {
            return false;
        };
        let bytes = match &source {
            Source::Bytes(bytes) => Some(Arc::clone(bytes)),
            Source::File { .. } => None,
        };
        if worker.jobs.send((file.clone(), source, want)).is_err() {
            // The thread is gone; what it held is lost. Start over with the next miss.
            self.worker = None;
            self.pending.clear();
            return false;
        }
        self.pending
            .insert(file.clone(), PendingDecode { bytes, played });
        true
    }

    /// Adds one decoded file. Both decodes of a file are the same decoder's output, so a file
    /// already resident stays as it is.
    ///
    /// The frame cap is a backstop, not a policy. A file a play asked for always goes in, first
    /// starting the decoded set over if it would not fit (the old first-play rule). A file only
    /// the prewarm asked for goes in only if it fits (the old prewarm rule), so a prewarm can
    /// never empty the cache it is filling.
    fn insert(
        &mut self,
        file: ResourceLocation,
        sample_rate: u32,
        frames: Arc<[Frame]>,
        played: bool,
    ) {
        if self.sounds.contains_key(&file) {
            return;
        }
        if self.frames + frames.len() > MAX_CACHED_SOUND_FRAMES {
            if !played {
                return;
            }
            self.sounds.clear();
            self.frames = 0;
        }
        self.frames += frames.len();
        self.sounds.insert(file, (sample_rate, frames));
    }

    /// Whether this cache holds `file`'s bytes or frames, which proves the file exists: some
    /// read of it succeeded.
    fn holds(&self, file: &ResourceLocation) -> bool {
        self.sounds.contains_key(file)
            || self.streamed.contains_key(file)
            || self.read_ahead.contains_key(file)
            || self
                .pending
                .get(file)
                .is_some_and(|pending| pending.bytes.is_some())
    }

    /// Keeps a read-ahead file's compressed bytes, while they fit [`MAX_READ_AHEAD_BYTES`].
    fn keep_read_ahead(&mut self, file: ResourceLocation, bytes: Arc<[u8]>) {
        if self.sounds.contains_key(&file)
            || self.read_ahead.contains_key(&file)
            || self.read_ahead_bytes + bytes.len() > MAX_READ_AHEAD_BYTES
        {
            return;
        }
        self.read_ahead_bytes += bytes.len();
        self.read_ahead.insert(file, bytes);
    }

    /// Keeps an always-streamed clip's compressed bytes, while they fit [`MAX_STREAMED_BYTES`];
    /// past that, such clips are read again at each play.
    fn keep_streamed(&mut self, file: ResourceLocation, bytes: Arc<[u8]>) {
        if self.streamed.contains_key(&file)
            || self.streamed_bytes + bytes.len() > MAX_STREAMED_BYTES
        {
            return;
        }
        self.streamed_bytes += bytes.len();
        self.streamed.insert(file, bytes);
    }

    /// The sound system is being unloaded (resource reload): drop every decode, finished or not,
    /// and the worker with them.
    fn clear(&mut self) {
        self.sounds.clear();
        self.frames = 0;
        self.pending.clear();
        self.failed.clear();
        self.streamed.clear();
        self.streamed_bytes = 0;
        self.read_ahead.clear();
        self.read_ahead_bytes = 0;
        self.existence.clear();
        self.worker = None;
    }

    /// Moves what the worker has finished into the cache, at most [`MAX_RESULTS_PER_DRAIN`]
    /// results per call (the read-ahead can finish hundreds of files between two ticks; the rest
    /// wait for the next call). Never blocks, and is a single `is_empty` test when nothing is
    /// pending.
    fn drain_finished(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        for _ in 0..MAX_RESULTS_PER_DRAIN {
            let Some(worker) = &self.worker else {
                return;
            };
            match worker.done.try_recv() {
                Ok((file, decoded)) => {
                    let played = self
                        .pending
                        .remove(&file)
                        .is_some_and(|pending| pending.played);
                    // Every outcome but `Unreadable` read the file, so it exists (a file that
                    // does not decode is still in vanilla's pool: vanilla only opens it).
                    let exists = !matches!(decoded, Decoded::Unreadable | Decoded::Exists(false));
                    self.existence.push((file.clone(), exists));
                    match decoded {
                        Decoded::Frames(sample_rate, frames) => {
                            self.insert(file, sample_rate, frames, played);
                        }
                        Decoded::Long(bytes) => self.keep_streamed(file, bytes),
                        Decoded::Read(bytes) => self.keep_read_ahead(file, bytes),
                        Decoded::Undecodable(reason) => {
                            self.failed.insert(file, reason);
                        }
                        Decoded::Unreadable | Decoded::Exists(_) => {}
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Only a panic outside the contained decode ends the thread on its own. The
                    // files it still held load and stream again at their next play.
                    self.worker = None;
                    self.pending.clear();
                    return;
                }
            }
        }
    }
}

/// A snapshot of the effect cache ([`AudioEngine::decode_cache_stats`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeCacheStats {
    /// Effects decoded and held as frames.
    pub decoded_files: usize,
    /// Their frames together (the cap is 12 Mi).
    pub decoded_frames: usize,
    /// Files queued on the background worker and not finished.
    pub pending_files: usize,
    /// Effects read ahead and held compressed, not decoded yet.
    pub read_ahead_files: usize,
    /// Their compressed bytes together.
    pub read_ahead_bytes: usize,
    /// Clips over 30 s, held compressed and streamed by every play.
    pub streamed_files: usize,
    /// Their compressed bytes together.
    pub streamed_bytes: usize,
}

/// One channel retained by the SoundManager-style playing registry.
struct ActiveSound {
    request: PlayRequest,
    variant: ResolvedVariant,
    category: SoundCategory,
    computed: ComputedPlayback,
    sound: PlayingSound,
    tickable: Option<Box<dyn TickableSound>>,
    /// The event's [`EventGains`] entry when the channel started (`1.0` = vanilla), refreshed by
    /// [`AudioEngine::set_event_gains`]; the per-tick refresh reuses it without a lookup.
    event_gain: f32,
    /// Vanilla's `playingSoundsStopTime`: a newly-created channel is considered for cleanup only
    /// after this absolute client tick. This avoids dropping a channel while its backend command
    /// is still being accepted.
    stop_time: u64,
}

/// One delayed `ISound` play/replay, keyed by the client tick on which it becomes eligible.
#[derive(Debug)]
struct DelayedSound {
    due_tick: u64,
    request: PlayRequest,
}

impl PlayingSound {
    /// The sound's current playback state.
    pub fn state(&self) -> PlaybackState {
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .state(),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .state(),
        }
    }

    /// Stops the sound immediately, with `kira`'s default (instant) tween.
    ///
    /// java: `SoundManager.stopSound(ISound)`, which has no fade parameter — the one caller that
    /// needs it, `RenderGlobal.playRecord`, cuts a record dead when the disc is swapped. Exists
    /// so a caller does not need its own `kira` dependency just to name `Tween::default()`.
    pub fn stop_now(&mut self) {
        self.stop(Tween::default());
    }

    /// Stops the sound with the given fade-out.
    pub fn stop(&mut self, tween: Tween) {
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .stop(tween),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .stop(tween),
        }
    }

    /// Current playback position, in seconds.
    pub fn position(&self) -> f64 {
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .position(),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .position(),
        }
    }

    /// Changes the linear gain immediately, retaining the same dB conversion as initial play.
    fn set_volume(&mut self, linear_gain: f32) {
        let volume = amplitude_to_decibels(linear_gain);
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .set_volume(volume, Tween::default()),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .set_volume(volume, Tween::default()),
        }
    }

    /// Changes the playback rate immediately.
    fn set_playback_rate(&mut self, rate: f32) {
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .set_playback_rate(rate as f64, Tween::default()),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .set_playback_rate(rate as f64, Tween::default()),
        }
    }

    /// Changes the stereo panning immediately.
    fn set_panning(&mut self, value: f32) {
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .set_panning(value, Tween::default()),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .set_panning(value, Tween::default()),
        }
    }

    /// Pauses this channel immediately.
    fn pause(&mut self) {
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .pause(Tween::default()),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .pause(Tween::default()),
        }
    }

    /// Resumes this channel immediately.
    fn resume(&mut self) {
        match self {
            PlayingSound::Static(h) => h
                .lock()
                .expect("static sound handle mutex poisoned")
                .resume(Tween::default()),
            PlayingSound::Streaming(h) => h
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .resume(Tween::default()),
        }
    }

    /// Whether this is a clone of the same channel as `other`.
    fn same_channel(&self, other: &Self) -> bool {
        match (self, other) {
            (PlayingSound::Static(left), PlayingSound::Static(right)) => Arc::ptr_eq(left, right),
            (PlayingSound::Streaming(left), PlayingSound::Streaming(right)) => {
                Arc::ptr_eq(left, right)
            }
            _ => false,
        }
    }
}

/// Owns the audio backend, the sound registry, and per-category volumes; the single entry point
/// game code plays sounds through.
///
/// Generic over kira's [`Backend`] the way [`AudioManager`] itself is: production code uses the
/// default (`B = DefaultBackend`, real output via `cpal`); tests instantiate
/// `AudioEngine::<kira::backend::mock::MockBackend>` to exercise the full mixing pipeline —
/// resolution, decode, and kira's own sound-graph — without a real audio device, which this
/// sandbox does not have. See `tests/engine_playback.rs`.
///
/// If backend construction fails (no output device — expected in CI, and vanilla's own fallback:
/// `SoundManager.loadSoundSystem` catches this and zeroes the master volume rather than
/// crashing), the engine runs **silent**: [`AudioEngine::play`] still resolves and returns
/// `Ok(None)`, so every non-backend code path stays exercised.
pub struct AudioEngine<B: Backend = DefaultBackend> {
    /// Built by [`AudioEngine::silent`]: never start a backend, including on a sound reload.
    output_disabled: bool,
    manager: Option<AudioManager<B>>,
    /// The loaded event → variant table. Public so callers can reload it
    /// (`S48PacketResourcePackSend` handling — see [`crate::packet`]) or inspect it.
    pub registry: SoundRegistry,
    /// Per-category volume sliders (`options.txt`).
    pub categories: CategoryVolumes,
    rng: JavaRandom,
    /// The listener used by the latest play/tick operation. Category changes happen outside the
    /// tick call, so retaining it lets a slider refresh use the same spatial calculation.
    listener: Listener,
    /// The last [`RECENT_PLAY_CAPACITY`] requests, oldest first. See [`AudioEngine::recent`].
    recent: VecDeque<PlayRecord>,
    /// Vanilla's `playingSounds`, keyed by a stable local channel id. Every successfully-started
    /// sound remains here until the backend reports it stopped and the `+20` grace expires.
    playing_sounds: HashMap<u64, ActiveSound>,
    /// Vanilla's `categorySounds`: the reverse index lets a category slider update only its own
    /// live channels without changing the selected variant.
    category_sounds: HashMap<SoundCategory, HashSet<u64>>,
    /// Vanilla's `tickableSounds`, represented by channel ids. The actual updater is stored on
    /// the corresponding [`ActiveSound`] so it can be removed atomically with that channel.
    tickable_sounds: HashSet<u64>,
    /// Vanilla's `delayedSounds`, represented as absolute client tick deadlines.
    delayed_sounds: Vec<DelayedSound>,
    /// `SoundManager.playTime`, advanced exclusively by [`Self::update_all_sounds`].
    play_time: u64,
    /// Monotonic local replacement for vanilla's random channel names.
    next_channel_id: u64,
    /// Effects already decoded by this sound system.
    decoded_sounds: DecodedSoundCache,
    /// The Sounds mod's per-event gains ([`Self::set_event_gains`]). Empty = vanilla. Survives
    /// sound-system reloads: it is keyed by event name, not by anything the reload replaces.
    event_gains: EventGains,
    /// Every sound started while this is `Some` is
    /// copied here, with its resolved variant, for the replay recorder. `None` (the default)
    /// costs one `Option` test per play request. See [`Self::set_capture`].
    capture: Option<Vec<CapturedSound>>,
    /// While a replay plays, the game's own sound
    /// generation is muted and only [`Self::play_captured`] sounds are heard. See
    /// [`Self::set_suppressed`].
    suppressed: bool,
    /// Every stream this engine started that the mixer has not
    /// marked `Stopped` yet (pruned at each new stream). kira ends a stream's decode thread only
    /// once the mixer marks it `Stopped`, so [`Self::finish_streams`] stops and waits for these
    /// before the backend is dropped.
    streams: Vec<StreamHandle>,
    /// The output-thread panic count this engine last restarted its
    /// output for ([`crate::output_recovery`]).
    seen_output_panics: u64,
    /// When to look whether the output still renders, and reopen it
    /// once a device is back ([`Self::watch_output`]).
    output_watch: OutputWatch,
    /// Unit tests drive kira's mock backend here while [`Self::finish_streams`] waits: unlike a
    /// real backend it has no audio thread of its own.
    #[cfg(test)]
    backend_pump: Option<fn(&mut B)>,
    /// Unit tests: what the output watch sees instead of the backend and the system.
    #[cfg(test)]
    watch_test: WatchTest,
}

/// What a unit test makes the output watch see.
#[cfg(test)]
#[derive(Debug, Default)]
struct WatchTest {
    /// The output's periods since the last look, instead of asking the backend.
    periods: Option<Option<usize>>,
    /// Whether a default device is there, instead of asking the system.
    device: Option<bool>,
    /// Outputs the watch opened.
    reopened: u32,
}

/// One stream's kira handle, shared with the [`PlayingSound`] the caller got.
type StreamHandle = Arc<Mutex<kira::sound::streaming::StreamingSoundHandle<AudioError>>>;

/// How long a sound-system unload waits for the mixer to stop its streams. A real backend does
/// it within one audio period (about 10 ms); this only bounds a backend whose device is gone.
const STREAM_FINISH_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);

/// The internal buffer size the output watch hands the backend's setup
/// when it only asks whether a device is there (the mixer's default; the desktop backend does
/// not use it).
const OUTPUT_PROBE_BUFFER_FRAMES: usize = 128;

/// Whether the mixer has marked this stream `Stopped`; its decode thread then ends by itself.
fn stream_stopped(stream: &StreamHandle) -> bool {
    stream
        .lock()
        .expect("streaming sound handle mutex poisoned")
        .state()
        == PlaybackState::Stopped
}

/// One sound the engine started while [`AudioEngine::set_capture`] was on: the request exactly
/// as the game built it, the event's category, and the weighted variant the engine picked
/// (`None` when the master fader was at zero, which skips the pick, as vanilla does).
///
/// A replay stores these so playback and the video
/// export hear the same file at the same volume, pitch and position.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedSound {
    /// The request as the game built it.
    pub request: PlayRequest,
    /// The event's category, if the registry knows the event.
    pub category: Option<SoundCategory>,
    /// The variant the weighted pick chose.
    pub variant: Option<ResolvedVariant>,
}

impl<B: Backend + OutputHeartbeat> AudioEngine<B>
where
    B::Settings: Default,
    B::Error: std::fmt::Debug,
{
    /// Builds an engine with the given registry, attempting to start the real backend. `seed`
    /// drives the weighted-variant RNG ([`mc_math::JavaRandom`] — vanilla uses `java.util.Random`
    /// for this same purpose, and reusing our bit-exact port costs nothing and keeps every
    /// randomness source in the client on one implementation).
    pub fn new(registry: SoundRegistry, seed: i64) -> Self {
        let manager = Self::start_backend("audio backend unavailable; running with sound disabled");
        Self {
            manager,
            output_disabled: false,
            registry,
            categories: CategoryVolumes::new(),
            rng: JavaRandom::new(seed),
            listener: Listener::origin(),
            recent: VecDeque::with_capacity(RECENT_PLAY_CAPACITY),
            playing_sounds: HashMap::new(),
            category_sounds: HashMap::new(),
            tickable_sounds: HashSet::new(),
            delayed_sounds: Vec::new(),
            play_time: 0,
            next_channel_id: 0,
            decoded_sounds: DecodedSoundCache::default(),
            event_gains: EventGains::new(),
            capture: None,
            suppressed: false,
            streams: Vec::new(),
            seen_output_panics: crate::output_recovery::output_backend_panics(),
            output_watch: OutputWatch::new(),
            #[cfg(test)]
            backend_pump: None,
            #[cfg(test)]
            watch_test: WatchTest::default(),
        }
    }

    /// An engine that deliberately has no backend: resolution, category volumes and the
    /// [`AudioEngine::recent`] tail all behave exactly as the real thing, and every
    /// [`AudioEngine::play`] ends at [`PlayOutcome::NoBackend`].
    ///
    /// This is the same state [`AudioEngine::new`] falls into when the output device is missing,
    /// reachable on purpose — for a future `--nosound`, and so a test can pin the no-device
    /// behaviour on a machine that *does* have speakers. java: `SoundManager` after
    /// `loadSoundSystem` catches its `RuntimeException` and leaves `loaded == false`.
    pub fn silent(registry: SoundRegistry, seed: i64) -> Self {
        Self {
            manager: None,
            output_disabled: true,
            registry,
            categories: CategoryVolumes::new(),
            rng: JavaRandom::new(seed),
            listener: Listener::origin(),
            recent: VecDeque::with_capacity(RECENT_PLAY_CAPACITY),
            playing_sounds: HashMap::new(),
            category_sounds: HashMap::new(),
            tickable_sounds: HashSet::new(),
            delayed_sounds: Vec::new(),
            play_time: 0,
            next_channel_id: 0,
            decoded_sounds: DecodedSoundCache::default(),
            event_gains: EventGains::new(),
            capture: None,
            suppressed: false,
            streams: Vec::new(),
            seen_output_panics: crate::output_recovery::output_backend_panics(),
            output_watch: OutputWatch::new(),
            #[cfg(test)]
            backend_pump: None,
            #[cfg(test)]
            watch_test: WatchTest::default(),
        }
    }

    /// Whether the real backend failed to start (no output device). Sound math still runs;
    /// nothing audible happens.
    pub fn is_silent(&self) -> bool {
        self.manager.is_none()
    }

    /// Start (`true`) or stop (`false`) copying
    /// every started sound into a buffer that [`Self::drain_captured`] empties. Off by default.
    /// Turning it off drops anything not yet drained.
    pub fn set_capture(&mut self, on: bool) {
        match (on, self.capture.is_some()) {
            (true, false) => self.capture = Some(Vec::new()),
            (false, true) => self.capture = None,
            _ => {}
        }
    }

    /// Whether [`Self::set_capture`] is on.
    pub fn is_capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// Move every sound captured since the last call into `out`, oldest first.
    pub fn drain_captured(&mut self, out: &mut Vec<CapturedSound>) {
        if let Some(captured) = self.capture.as_mut() {
            out.append(captured);
        }
    }

    /// While `true`, every request the game
    /// makes (one-shots, delayed sounds, repeats, music, moving sounds) is dropped before it
    /// resolves, and only [`Self::play_captured`] is heard. Turning it on also stops everything
    /// playing, so a replay starts silent; turning it off leaves the engine as it was.
    pub fn set_suppressed(&mut self, on: bool) {
        if on && !self.suppressed {
            self.stop_all_sounds();
        }
        self.suppressed = on;
    }

    /// Whether [`Self::set_suppressed`] is on.
    pub fn is_suppressed(&self) -> bool {
        self.suppressed
    }

    /// Play a recorded sound with the variant it had when it was recorded, bypassing both the
    /// weighted pick and [`Self::set_suppressed`]. With no variant (the recording's master fader
    /// was at zero) the pick runs as usual.
    // java: SoundManager.playSound with a pre-resolved PositionedSound
    pub fn play_captured(
        &mut self,
        sound: &CapturedSound,
        listener: &Listener,
        resources: &ResourceManager,
    ) -> Result<Option<PlayingSound>, AudioError> {
        self.play_internal(
            sound.request.clone(),
            listener,
            resources,
            None,
            Some(sound.variant.clone()),
        )
    }

    /// The last [`RECENT_PLAY_CAPACITY`] play requests, oldest first.
    ///
    /// This is the only way to observe that the game asked for the right sound on a host with
    /// no output device — [`AudioEngine::play`] returns `Ok(None)` in that case and there is
    /// nothing else to look at. Hosts with speakers get the same record, so a debug overlay or
    /// a parity harness can read it without caring which case it is in.
    pub fn recent(&self) -> impl DoubleEndedIterator<Item = &PlayRecord> {
        self.recent.iter()
    }

    /// The most recent play request, or `None` if nothing has been played yet.
    pub fn last_play(&self) -> Option<&PlayRecord> {
        self.recent.back()
    }

    /// Forgets every [`PlayRecord`]. java: nothing — but `Minecraft.loadWorld` drops the whole
    /// sound state on a world change, and a diagnostic tail that survives a disconnect would
    /// describe a world that is gone.
    pub fn clear_recent(&mut self) {
        self.recent.clear();
    }

    /// Runs the complete `SoundManager.updateAllSounds` pass for one client tick.
    ///
    /// Every retained channel is checked, tickable sounds are advanced, and the selected variant
    /// is reused while gain/pitch/pan are refreshed from the current listener and category
    /// values. A stopped channel is retained until its vanilla `playingSoundsStopTime` grace
    /// deadline, then its positive-delay repeat is moved to the absolute delayed queue. Ready
    /// delayed sounds go back through [`Self::play`], so resource/decode failures are observable
    /// in the same way as an ordinary request.
    // java: SoundManager.updateAllSounds:254-303
    pub fn update_all_sounds(
        &mut self,
        listener: &Listener,
        resources: &ResourceManager,
    ) -> Vec<AudioError> {
        self.listener = *listener;
        self.play_time = self.play_time.saturating_add(1);
        // One atomic load per tick unless the output thread just died.
        let output_panics = crate::output_recovery::output_backend_panics();
        if output_panics != self.seen_output_panics {
            self.seen_output_panics = output_panics;
            self.restart_output();
        }
        // A countdown per tick; every few seconds a look at whether the
        // output still renders, and a new one once a device is back (`watch_output`).
        if !self.output_disabled && self.output_watch.due() {
            self.watch_output();
        }
        // Take in background work once per tick, so a clip's
        // compressed bytes are not held longer than needed and missing files leave the pools
        // promptly. Two `is_empty` tests when idle.
        self.drain_worker();

        let mut errors = Vec::new();
        let mut completed = Vec::new();
        let channel_ids: Vec<u64> = self.playing_sounds.keys().copied().collect();
        let categories = self.categories;
        let current_listener = self.listener;

        for channel_id in channel_ids {
            let mut remove = false;
            let mut repeat = None;
            {
                let Some(active) = self.playing_sounds.get_mut(&channel_id) else {
                    continue;
                };

                let mut refresh = true;
                if let Some(tickable) = active.tickable.as_mut() {
                    tickable.update();
                    if tickable.is_done_playing() {
                        active.sound.stop_now();
                        refresh = false;
                    } else {
                        let updated = tickable.request();
                        if updated.event != active.request.event {
                            // `PositionedSound.positionedSoundLocation` is final in vanilla;
                            // ending a malformed moving source is safer than applying the old
                            // decoded file to a new event.
                            active.sound.stop_now();
                            refresh = false;
                        } else {
                            active.request = updated;
                        }
                    }
                }

                if refresh {
                    let computed = compute_with_category(
                        &categories,
                        &current_listener,
                        &active.request,
                        &active.variant,
                        active.category,
                        active.event_gain,
                    );
                    active.computed = computed.clone();
                    active.sound.set_volume(computed.linear_gain);
                    active.sound.set_playback_rate(computed.pitch);
                    active.sound.set_panning(computed.pan);
                }

                if active.sound.state() == PlaybackState::Stopped
                    && active.stop_time <= self.play_time
                {
                    if active.request.repeat && active.request.repeat_delay_ticks > 0 {
                        repeat = Some(active.request.clone());
                    }
                    remove = true;
                }
            }

            if remove {
                self.remove_active(channel_id);
                if let Some(request) = repeat {
                    completed.push(request);
                }
            }
        }

        for request in completed {
            self.queue_completed_repeat(request);
        }

        let mut ready = Vec::new();
        self.delayed_sounds.retain(|delayed| {
            if delayed.due_tick <= self.play_time {
                ready.push(delayed.request.clone());
                false
            } else {
                true
            }
        });
        for request in ready {
            if let Err(error) = self.play(&request, &current_listener, resources) {
                errors.push(error);
            }
        }
        errors
    }

    /// Compatibility name for the first repeat-only client hook. It now runs the full retained
    /// channel/tickable/category/spatial update pass, so existing callers automatically gain the
    /// rest of `SoundManager.updateAllSounds` semantics.
    pub fn tick_repeats(
        &mut self,
        listener: &Listener,
        resources: &ResourceManager,
    ) -> Vec<AudioError> {
        self.update_all_sounds(listener, resources)
    }

    /// Mirrors `SoundManager`'s `canRepeat() && getRepeatDelay() > 0` completion branch.
    /// Kept separate from backend polling so the signed-delay boundary is testable without an
    /// audio device or an encoded fixture.
    fn queue_completed_repeat(&mut self, request: PlayRequest) {
        if request.repeat && request.repeat_delay_ticks > 0 {
            self.delayed_sounds.push(DelayedSound {
                due_tick: self
                    .play_time
                    .saturating_add(request.repeat_delay_ticks as u64),
                request,
            });
        }
    }

    fn remove_active(&mut self, channel_id: u64) -> Option<ActiveSound> {
        let active = self.playing_sounds.remove(&channel_id)?;
        if let Some(ids) = self.category_sounds.get_mut(&active.category) {
            ids.remove(&channel_id);
            if ids.is_empty() {
                self.category_sounds.remove(&active.category);
            }
        }
        self.tickable_sounds.remove(&channel_id);
        Some(active)
    }

    /// What the effect cache holds right now, for diagnostics and the
    /// `sound_stall` example. Does not take in finished background work.
    pub fn decode_cache_stats(&self) -> DecodeCacheStats {
        let cache = &self.decoded_sounds;
        DecodeCacheStats {
            decoded_files: cache.sounds.len(),
            decoded_frames: cache.frames,
            pending_files: cache.pending.len(),
            read_ahead_files: cache.read_ahead.len(),
            read_ahead_bytes: cache.read_ahead_bytes,
            streamed_files: cache.streamed.len(),
            streamed_bytes: cache.streamed_bytes,
        }
    }

    /// Takes in every decode or read the background worker has finished. Never blocks.
    /// [`Self::update_all_sounds`] and every play already do this; this is for a caller that
    /// wants to time it on its own.
    pub fn drain_finished_decodes(&mut self) {
        self.drain_worker();
    }

    /// Number of channels in vanilla's `playingSounds` registry, including stop-grace entries.
    pub fn active_sound_count(&self) -> usize {
        self.playing_sounds.len()
    }

    /// Number of retained channels that can schedule a positive-delay repeat on completion.
    /// Primarily useful to a client diagnostic overlay and deterministic parity tests.
    pub fn active_repeat_count(&self) -> usize {
        self.playing_sounds
            .values()
            .filter(|active| active.request.repeat && active.request.repeat_delay_ticks > 0)
            .count()
    }

    /// Number of retained [`TickableSound`] sources.
    pub fn tickable_sound_count(&self) -> usize {
        self.tickable_sounds.len()
    }

    /// Current `SoundManager.playTime`, in client ticks.
    pub fn current_tick(&self) -> u64 {
        self.play_time
    }

    /// Number of sound requests waiting in the Java `delayedSounds` queue.
    pub fn delayed_sound_count(&self) -> usize {
        self.delayed_sounds.len()
    }

    /// Compatibility name for the original repeat-only queue counter.
    pub fn delayed_repeat_count(&self) -> usize {
        self.delayed_sound_count()
    }

    fn record(
        &mut self,
        request: &PlayRequest,
        computed: Option<ComputedPlayback>,
        outcome: PlayOutcome,
    ) {
        if self.recent.len() == RECENT_PLAY_CAPACITY {
            self.recent.pop_front();
        }
        self.recent.push_back(PlayRecord {
            request: request.clone(),
            computed,
            outcome,
        });
    }

    /// Direct access to the backend, mainly so tests can drive [`kira::backend::mock::MockBackend`]'s
    /// `on_start_processing`/`process`.
    pub fn backend_mut(&mut self) -> Option<&mut B> {
        self.manager.as_mut().map(|m| m.backend_mut())
    }

    /// Re-parses every `sounds.json` visible through `resources` and replaces
    /// [`AudioEngine::registry`] with the result — the audio-relevant consequence of a resource
    /// pack swap (see [`crate::packet::ResourcePackSendPacket`]'s doc comment).
    ///
    /// Existing playback is torn down before the registry is rebuilt, matching vanilla's
    /// `SoundManager.reloadSoundSystem()` boundary: active channels and all pending delayed or
    /// repeating playback are removed before a post-reload request can use the new registry.
    // java: SoundHandler.onResourceManagerReload -> SoundManager.reloadSoundSystem ->
    // `unloadSoundSystem` -> `stopAllSounds`, followed by registry clear/load.
    pub fn reload_sounds(&mut self, resources: &ResourceManager) -> Result<(), AudioError> {
        // Java unloads the old sound system before clearing/rebuilding its event registry. Tear
        // down every retained channel and the backend first so active handles, tickable sources,
        // delayed plays and positive-delay repeats cannot survive long enough to use the new pack.
        self.teardown_sound_system();
        self.registry = SoundRegistry::load(resources)?;
        Ok(())
    }

    /// Replaces an already-loaded registry at the live resource-reload boundary.
    ///
    /// The client asset loader has already parsed the stacked `sounds.json` files and applied
    /// its malformed-resource fallback before this method is called. Re-parsing the same
    /// `ResourceManager` here would make a successful asset reload do the work twice and could
    /// disagree with the registry the rest of the client adopted. The sound system still needs
    /// the full Java reload boundary, however: `SoundManager.reloadSoundSystem()` unloads the
    /// active backend before it loads a fresh one, and `stopAllSounds()` clears every retained
    /// channel/index/queue before the new registry is visible.
    // java: SoundHandler.onResourceManagerReload -> SoundManager.reloadSoundSystem
    pub fn reload_registry(&mut self, registry: SoundRegistry) {
        self.teardown_sound_system();
        self.registry = registry;
    }

    /// Decode the non-streaming effects of every event `keep` selects, and every effect a
    /// resource pack supplies itself, on a background thread, so their
    /// first play does not read and decode on the game thread (HITCH-3 / AUDIO-1). Returns how
    /// many files were queued.
    ///
    /// java: paulscode's `SoundSystem` loads and decodes a sound on its own command thread, so
    /// vanilla's game thread never pays this. The decode runs on the same worker thread a first
    /// play uses, and so does the read of every plain file (resource packs
    /// are not `Send`, so a zip entry is still read here). Behind the decodes, the worker reads
    /// ahead every other effect's compressed bytes, so no first play waits on the disk.
    /// Finished work enters the cache at the next play or tick; a play that comes first streams
    /// instead (it used to decode inline). The frames are the same decoder's output either way,
    /// no variant is picked (the RNG is untouched), and a reload discards in-flight work.
    pub fn prewarm_static_sounds(
        &mut self,
        resources: &ResourceManager,
        keep: impl Fn(&ResourceLocation) -> bool,
    ) -> usize {
        mc_core::profile_scope!("audio.prewarm_read");
        let mut files = self.registry.static_files_where(&keep);
        // Also every effect a resource pack supplies itself,
        // whatever its event. Those are the files likely to be long (a song as the level-up
        // sound), so their first play in a fight is a cache hit instead of a read. Packs
        // usually override a handful of sounds; the cache cap still bounds the total.
        let from_packs = files_supplied_by_packs(resources);
        if !from_packs.is_empty() {
            let chosen: HashSet<ResourceLocation> = files.iter().cloned().collect();
            let extra = self
                .registry
                .static_files_where(|_| true)
                .into_iter()
                .filter(|file| from_packs.contains(file) && !chosen.contains(file));
            files.extend(extra);
        }
        files.retain(|file| {
            !self.decoded_sounds.sounds.contains_key(file)
                && !self.decoded_sounds.pending.contains_key(file)
                && !self.decoded_sounds.streamed.contains_key(file)
        });
        let cap = resources.max_resource_size();
        // Follow-up 2: first, have the worker check that every streamed file (music, records)
        // with a plain path exists. Picks never ask the disk; the worker's answers reach the
        // registry at the next tick or pick (`drain_worker`). Effects are checked by the reads
        // below. A zip's files need no check: its entry list is already in memory.
        for file in self.registry.streamed_files() {
            if self.decoded_sounds.pending.contains_key(&file) {
                continue;
            }
            let Some(path) = resources.file_path(&file) else {
                continue;
            };
            if !self
                .decoded_sounds
                .queue(&file, Source::File { path, cap }, Want::Exists, false)
            {
                return 0;
            }
        }
        let mut queued = 0;
        for file in files {
            // A plain file (the asset store, a folder pack) is
            // read by the worker, so start-up does not wait on the disk here: 142 vanilla files
            // took 11-16 ms warm and seconds from a cold or busy disk. A zip entry is still read
            // here (the archive is open in memory, and packs are not `Send`).
            let source = match resources.file_path(&file) {
                Some(path) => Source::File { path, cap },
                None => match resources.load(&file) {
                    Ok(bytes) => Source::Bytes(Arc::from(bytes)),
                    Err(error) => {
                        // Missing: out of the pools, as vanilla's load-time open would have
                        // left it. Unreadable: the play path reports it if it is ever used.
                        if matches!(error, mc_assets::AssetError::NotFound(_)) {
                            self.decoded_sounds.existence.push((file, false));
                        }
                        continue;
                    }
                },
            };
            if !self
                .decoded_sounds
                .queue(&file, source, Want::Frames, false)
            {
                return queued;
            }
            queued += 1;
        }
        // Then, behind those decodes, read ahead every other effect that is a plain file (the
        // asset store, folder packs; a zip's own sounds were all queued above): about 6.5 MB for
        // all of vanilla, kept compressed. A first play then never waits on the disk, which on a
        // cold or busy disk took up to seconds for one file (2.8 s seen on the USB drive).
        for file in self.registry.static_files_where(|_| true) {
            if self.decoded_sounds.holds(&file) || self.decoded_sounds.pending.contains_key(&file) {
                continue;
            }
            let Some(path) = resources.file_path(&file) else {
                continue;
            };
            let source = Source::File { path, cap };
            if !self.decoded_sounds.queue(&file, source, Want::Bytes, false) {
                break;
            }
        }
        queued
    }

    /// Rebuild the output after the output library's stream thread died
    /// ([`crate::output_recovery`]): every playing sound stops, as on any sound-system reload,
    /// and the new output opens on the device that is the default now. Decoded sounds, the
    /// category volumes and the registry are kept. An engine built silent stays silent.
    fn restart_output(&mut self) {
        if self.output_disabled || self.manager.is_none() {
            return;
        }
        tracing::warn!("the audio output stopped after a device change; restarting it");
        self.reopen_output("audio output could not be restarted; running with sound disabled");
    }

    /// Drop the output (if any) and open a new one on the device that is the default now: every
    /// playing sound stops, as on any sound-system reload; decoded sounds, the category volumes
    /// and the registry are kept.
    fn reopen_output(&mut self, failure: &'static str) {
        self.stop_all_sounds();
        // Bounded (`STREAM_FINISH_TIMEOUT`): the dead output may never mark a stream stopped.
        self.finish_streams();
        drop(self.manager.take());
        self.manager = Self::start_backend(failure);
        if self.manager.is_some() {
            let master = self.categories.get(SoundCategory::Master);
            self.set_master_volume(master);
        }
    }

    /// A look at the output, every few seconds
    /// ([`crate::output_recovery`]): when every device has gone, the output library stops its
    /// stream and never looks again, so sound did not come back with a device. If the output
    /// rendered nothing since the last look, or there is none at all (no device at start, or a
    /// restart that found none), a new one is opened as soon as a default device is there. A
    /// running output is never touched; an engine built silent never looks (see the caller).
    fn watch_output(&mut self) {
        let periods = self.output_periods();
        #[cfg(test)]
        let device_present = {
            let answer = self.watch_test.device;
            move || answer.unwrap_or_else(Self::output_device_present)
        };
        #[cfg(not(test))]
        let device_present = Self::output_device_present;
        if self.output_watch.look(periods, device_present) == Look::Reopen {
            tracing::warn!("the audio output was not running; opening it on the default device");
            #[cfg(test)]
            {
                self.watch_test.reopened += 1;
            }
            self.reopen_output("audio output could not be reopened; running with sound disabled");
        }
    }

    /// What the output rendered since the last look; `None`: there is no output.
    fn output_periods(&mut self) -> Option<Option<usize>> {
        #[cfg(test)]
        if let Some(periods) = self.watch_test.periods {
            return self.manager.as_ref().map(|_| periods);
        }
        self.manager
            .as_mut()
            .map(|manager| manager.backend_mut().periods_since_last_look())
    }

    /// Whether the system has a default output device the backend can open: the backend's own
    /// setup (it finds the device and its format and opens no stream), dropped at once.
    fn output_device_present() -> bool {
        B::setup(B::Settings::default(), OUTPUT_PROBE_BUFFER_FRAMES).is_ok()
    }

    /// Stops all retained playback and recreates the backend, matching Java's
    /// `SoundManager.unloadSoundSystem()`/`loadSoundSystem()` pair. Category sliders and the
    /// diagnostic play tail belong to the engine, not the backend, so they intentionally survive.
    fn teardown_sound_system(&mut self) {
        self.stop_all_sounds();
        self.decoded_sounds.clear();
        // Before the mixer goes, it must have stopped every stream, or each stream's decode
        // thread outlives it (see `finish_streams`).
        self.finish_streams();

        // Dropping kira's AudioManager stops its output backend and releases its mixer graph.
        // Keep this explicit rather than relying on a later Client drop: a live resource-pack
        // reload must not leave sources or backend state from the old pack alive.
        drop(self.manager.take());
        // An engine built by `silent` stays silent: a reload must not open the output device.
        self.manager = if self.output_disabled {
            None
        } else {
            Self::start_backend(
                "audio backend unavailable during sound reload; running with sound disabled",
            )
        };

        // `SoundManager.loadSoundSystem` reapplies the persisted master fader to a newly loaded
        // backend. The category values themselves remain owned by `CategoryVolumes`.
        if self.manager.is_some() {
            let master = self.categories.get(SoundCategory::Master);
            self.set_master_volume(master);
        }
    }

    /// Stops every stream this engine started and waits, at most [`STREAM_FINISH_TIMEOUT`], until
    /// the mixer has marked each one `Stopped`.
    ///
    /// kira 0.12's stream decode thread ends only when the mixer
    /// marks its stream `Stopped` (or it reaches its end); it never notices the mixer being
    /// dropped. A reload used to drop the backend right after `stop_all_sounds`, before the audio
    /// thread had applied the stops, so every stream still playing (music, a record, a long
    /// effect) left a thread behind that woke every millisecond for the rest of the process.
    /// The stop here has no fade, so a real backend applies it at its next audio period (about
    /// 10 ms) and the wait ends; the timeout only covers a backend whose device has gone, which
    /// then leaks as before. Nothing waits when no stream is playing.
    fn finish_streams(&mut self) {
        self.streams.retain(|stream| !stream_stopped(stream));
        if self.manager.is_none() || self.streams.is_empty() {
            self.streams.clear();
            return;
        }
        let at_once = Tween {
            duration: std::time::Duration::ZERO,
            ..Tween::default()
        };
        for stream in &self.streams {
            stream
                .lock()
                .expect("streaming sound handle mutex poisoned")
                .stop(at_once);
        }
        let deadline = std::time::Instant::now() + STREAM_FINISH_TIMEOUT;
        loop {
            self.pump_backend();
            self.streams.retain(|stream| !stream_stopped(stream));
            if self.streams.is_empty() {
                return;
            }
            if std::time::Instant::now() >= deadline {
                tracing::warn!(
                    streams = self.streams.len(),
                    "audio streams were not stopped before the sound system unloaded; their \
                     decode threads may stay alive"
                );
                self.streams.clear();
                return;
            }
        }
    }

    /// One step of [`Self::finish_streams`]'s wait. A real backend's audio thread does the work,
    /// so this sleeps a millisecond; unit tests drive kira's mock backend, which has no thread.
    fn pump_backend(&mut self) {
        #[cfg(test)]
        if let (Some(pump), Some(manager)) = (self.backend_pump, self.manager.as_mut()) {
            pump(manager.backend_mut());
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    fn start_backend(message: &'static str) -> Option<AudioManager<B>> {
        match AudioManager::<B>::new(AudioManagerSettings::default()) {
            Ok(manager) => Some(manager),
            Err(error) => {
                tracing::warn!(?error, "{message}");
                None
            }
        }
    }

    /// Sets the master fader (`SoundCategory::MASTER`). Mapped onto kira's main mixer track —
    /// one global multiplier — matching `SoundSystem.setMasterVolume` rather than being folded
    /// into every individual sound's gain the way the other eight categories are.
    // java: SoundManager.setSoundCategoryVolume(MASTER, ..)
    pub fn set_master_volume(&mut self, volume: f32) {
        let volume = volume.clamp(0.0, 1.0);
        self.categories.set(SoundCategory::Master, volume);
        if let Some(manager) = &mut self.manager {
            manager
                .main_track()
                .set_volume(amplitude_to_decibels(volume), Tween::default());
        }
        // java: SoundHandler.setSoundLevel -> SoundManager.stopSounds when the master bus is
        // muted. Clear the registry too, otherwise a muted channel would remain a phantom
        // `isSoundPlaying` entry until its backend naturally completed.
        if volume <= 0.0 {
            self.stop_all_sounds();
        }
    }

    /// Updates one category slider and immediately refreshes every retained channel in that
    /// category. The selected file/variant is stable; only normalized gain, pitch and panning
    /// are sent to the live Kira handles. A zeroed non-master category stops its channels just
    /// like vanilla's `SoundManager.setSoundCategoryVolume` loop.
    // java: SoundManager.setSoundCategoryVolume:435-462
    pub fn set_sound_category_volume(&mut self, category: SoundCategory, volume: f32) {
        if category == SoundCategory::Master {
            self.set_master_volume(volume);
            return;
        }

        self.categories.set(category, volume);
        let ids = self
            .category_sounds
            .get(&category)
            .cloned()
            .unwrap_or_default();
        let categories = self.categories;
        let listener = self.listener;
        for channel_id in ids {
            let Some(active) = self.playing_sounds.get_mut(&channel_id) else {
                continue;
            };
            let computed = compute_with_category(
                &categories,
                &listener,
                &active.request,
                &active.variant,
                active.category,
                active.event_gain,
            );
            active.computed = computed.clone();
            active.sound.set_volume(computed.linear_gain);
            active.sound.set_playback_rate(computed.pitch);
            active.sound.set_panning(computed.pan);
            if computed.linear_gain <= 0.0 {
                active.sound.stop_now();
            }
        }
    }

    /// The Sounds mod's per-event gains (empty = vanilla).
    pub fn event_gains(&self) -> &EventGains {
        &self.event_gains
    }

    /// Replaces the per-event gain table (the Sounds mod) and applies it to every retained
    /// channel at once, like a category slider: a changed event's live channels get their new
    /// volume without restarting, a muted one's stop. Call it when the mod's settings change,
    /// never per play; [`Self::play`] then looks each event up once.
    pub fn set_event_gains(&mut self, gains: EventGains) {
        self.event_gains = gains;
        let categories = self.categories;
        let listener = self.listener;
        for active in self.playing_sounds.values_mut() {
            let event_gain = self.event_gains.get(&active.request.event);
            if event_gain == active.event_gain {
                continue;
            }
            active.event_gain = event_gain;
            let computed = compute_with_category(
                &categories,
                &listener,
                &active.request,
                &active.variant,
                active.category,
                event_gain,
            );
            active.computed = computed.clone();
            active.sound.set_volume(computed.linear_gain);
            active.sound.set_playback_rate(computed.pitch);
            active.sound.set_panning(computed.pan);
            if computed.linear_gain <= 0.0 {
                active.sound.stop_now();
            }
        }
    }

    /// Stops one retained channel, if it belongs to this engine. The shared handle means callers
    /// may pass the value returned from [`Self::play`] without taking ownership away from the
    /// playing registry.
    pub fn stop_sound(&mut self, sound: &PlayingSound) -> bool {
        let mut stopped = false;
        for active in self.playing_sounds.values_mut() {
            if active.sound.same_channel(sound) {
                active.sound.stop_now();
                stopped = true;
            }
        }
        stopped
    }

    /// Stops and removes every retained channel and pending delayed repeat.
    /// Any [`PlayingSound`] clones held by callers are stopped as well, but the clones themselves
    /// are not invalidated; a client-owned sound map must drop those handles when its world goes
    /// away. This mirrors `SoundManager.stopAllSounds`, whose internal maps are the sole owner of
    /// the channel bookkeeping.
    // java: SoundManager.stopAllSounds
    pub fn stop_all_sounds(&mut self) {
        for active in self.playing_sounds.values_mut() {
            active.sound.stop_now();
        }
        self.playing_sounds.clear();
        self.category_sounds.clear();
        self.tickable_sounds.clear();
        self.delayed_sounds.clear();
    }

    /// Pauses every retained channel. Delayed repeats remain scheduled, matching vanilla's
    /// separation between `pauseAllSounds` and the delayed source map.
    // java: SoundManager.pauseAllSounds
    pub fn pause_all_sounds(&mut self) {
        for active in self.playing_sounds.values_mut() {
            active.sound.pause();
        }
    }

    /// Resumes every retained channel previously paused by [`Self::pause_all_sounds`].
    // java: SoundManager.resumeAllSounds
    pub fn resume_all_sounds(&mut self) {
        for active in self.playing_sounds.values_mut() {
            active.sound.resume();
        }
    }

    /// Returns whether a handle is playing, or is still represented by vanilla's stop-grace
    /// entry at the cleanup deadline.
    // java: SoundManager.isSoundPlaying
    pub fn is_sound_playing(&self, sound: &PlayingSound) -> bool {
        self.playing_sounds.values().any(|active| {
            active.sound.same_channel(sound)
                && (active.sound.state() != PlaybackState::Stopped
                    || active.stop_time <= self.play_time)
        })
    }

    /// Adds one sound to the vanilla `SoundManager.delayedSounds` queue.
    ///
    /// `SoundHandler.playDelayedSound` stores the absolute deadline
    /// `playTime + delay`, then `SoundManager.updateAllSounds` plays the request once
    /// `playTime >= deadline`.  The queue is deliberately separate from a request's repeat
    /// fields: a delayed one-shot (the `WorldClient.playSound(..., distanceDelay)` path) must not
    /// acquire a loop merely because it was deferred.  Negative delays are due immediately, as
    /// Java's signed `int` comparison makes them; zero and negative values still remain one
    /// queued entry until the next update pass, matching the fact that the Java method only
    /// enqueues and never plays inline.
    // java: SoundHandler.playDelayedSound -> SoundManager.playDelayedSound:459-462
    pub fn play_delayed_sound(&mut self, request: PlayRequest, delay_ticks: i32) {
        self.delayed_sounds.push(DelayedSound {
            due_tick: self.play_time.saturating_add(delay_ticks.max(0) as u64),
            request,
        });
    }

    /// Compatibility name for callers that predate the public
    /// [`Self::play_delayed_sound`] spelling.  Keep the implementation in the Java-shaped API so
    /// repeat scheduling and explicit distance-delay callers cannot drift apart.
    pub fn schedule_delayed_sound(&mut self, request: PlayRequest, delay_ticks: i32) {
        self.play_delayed_sound(request, delay_ticks);
    }

    /// Resolves and, if a real backend is running, plays a sound. Loads and decodes through
    /// `resources`, so this does file I/O and CPU-bound decode work — call it off the render
    /// thread for anything but the shortest effects (§11: `tokio` is network-only, so this is a
    /// plain blocking call; a future integration section can move it to a worker if profiling
    /// shows it matters).
    // java: SoundManager.playSound
    pub fn play(
        &mut self,
        request: &PlayRequest,
        listener: &Listener,
        resources: &ResourceManager,
    ) -> Result<Option<PlayingSound>, AudioError> {
        self.play_internal(request.clone(), listener, resources, None, None)
    }

    /// Plays and retains a tickable/moving source. Its initial request is captured before the
    /// first backend play; subsequent [`Self::update_all_sounds`] calls invoke its updater and
    /// refresh the retained channel in place.
    // java: SoundManager.playSound + `tickableSounds.add`
    pub fn play_tickable<T: TickableSound + 'static>(
        &mut self,
        tickable: T,
        listener: &Listener,
        resources: &ResourceManager,
    ) -> Result<Option<PlayingSound>, AudioError> {
        let request = tickable.request();
        self.play_internal(request, listener, resources, Some(Box::new(tickable)), None)
    }

    /// `forced` is `Some` only for [`Self::play_captured`]: a replayed sound, which bypasses
    /// suppression and, when it carries a variant, the weighted pick.
    fn play_internal(
        &mut self,
        request: PlayRequest,
        listener: &Listener,
        resources: &ResourceManager,
        tickable: Option<Box<dyn TickableSound>>,
        forced: Option<Option<ResolvedVariant>>,
    ) -> Result<Option<PlayingSound>, AudioError> {
        self.listener = *listener;
        // Replay playback: the game's own sounds are muted; only recorded ones play.
        if self.suppressed && forced.is_none() {
            return Ok(None);
        }
        // Every exit below leaves a `PlayRecord` behind, including the failures: see
        // `AudioEngine::recent` for why "the game asked for this" is worth recording separately
        // from "a speaker moved".
        // java: SoundManager.playSound's master-volume guard runs before registry lookup or
        // weighted selection, so a muted game neither consumes an RNG draw nor decodes an OGG.
        if self.categories.get(SoundCategory::Master) <= 0.0 {
            if let Some(capture) = self.capture.as_mut() {
                capture.push(CapturedSound {
                    category: self.registry.category_of(&request.event),
                    request: request.clone(),
                    variant: None,
                });
            }
            self.record(&request, None, PlayOutcome::Inaudible);
            return Ok(None);
        }
        let Some(category) = self.registry.category_of(&request.event) else {
            let error = AudioError::UnknownEvent(request.event.to_string());
            self.record(&request, None, PlayOutcome::Unresolved);
            return Err(error);
        };
        let picked = match forced {
            Some(Some(variant)) => Some(variant),
            _ => self.pick_existing(&request.event),
        };
        let variant = match picked {
            Some(variant) => variant,
            None => {
                let error = AudioError::EmptyEvent(request.event.to_string());
                self.record(&request, None, PlayOutcome::Unresolved);
                return Err(error);
            }
        };
        if let Some(capture) = self.capture.as_mut() {
            capture.push(CapturedSound {
                request: request.clone(),
                category: Some(category),
                variant: Some(variant.clone()),
            });
        }
        // The Sounds mod: one lookup per play, none at all while the table is empty.
        let event_gain = self.event_gains.get(&request.event);
        let computed = match compute_with_variant(
            &self.registry,
            &self.categories,
            listener,
            &request,
            &variant,
            event_gain,
        ) {
            Ok(computed) if computed.linear_gain > 0.0 => computed,
            Ok(_) => {
                self.record(&request, None, PlayOutcome::Inaudible);
                return Ok(None);
            }
            Err(error) => {
                self.record(&request, None, PlayOutcome::Unresolved);
                return Err(error);
            }
        };

        if self.manager.is_none() {
            self.record(&request, Some(computed), PlayOutcome::NoBackend);
            return Ok(None);
        }

        match self.play_resolved(&computed, resources) {
            Ok(sound) => {
                self.record(&request, Some(computed.clone()), PlayOutcome::Played);
                let channel_id = self.next_channel_id;
                self.next_channel_id = self.next_channel_id.wrapping_add(1);
                let category = computed.category;
                let has_tickable = tickable.is_some();
                self.playing_sounds.insert(
                    channel_id,
                    ActiveSound {
                        request,
                        variant,
                        category,
                        computed,
                        sound: sound.clone(),
                        tickable,
                        event_gain,
                        stop_time: self.play_time.saturating_add(20),
                    },
                );
                if category != SoundCategory::Master {
                    self.category_sounds
                        .entry(category)
                        .or_default()
                        .insert(channel_id);
                }
                if has_tickable {
                    self.tickable_sounds.insert(channel_id);
                }
                Ok(Some(sound))
            }
            Err(error) => {
                self.record(&request, Some(computed), PlayOutcome::Failed);
                // A presumed file found missing by this play leaves the pools now.
                self.apply_existence();
                Err(error)
            }
        }
    }

    /// The weighted pick, from in-memory facts only.
    ///
    /// `SoundRegistry::load` no longer opens every file whose
    /// pack lists it in memory (the asset index, a zip's entries); such a file is presumed to
    /// exist. Follow-up 2: a pick never asks the disk (a metadata call here, on the game thread,
    /// froze frames for up to 3 s on a saturated disk). The worker checks the presumed files
    /// after load (`prewarm_static_sounds`), and what it finds is applied here before the pick:
    /// a missing file leaves every pool, exactly as vanilla's load-time open would have left it.
    /// Until then (an engine without a backend never checks) a presumed file counts as present;
    /// a play that finds it missing fails like any unreadable file and drops it right away.
    // java: SoundHandler.loadSoundResource, the FILE branch's open at load time
    fn pick_existing(&mut self, event: &ResourceLocation) -> Option<ResolvedVariant> {
        self.drain_worker();
        self.registry.pick_variant(event, &mut self.rng)
    }

    /// Takes in the worker's finished work ([`DecodedSoundCache::drain_finished`]) and applies
    /// what it learned about which files exist.
    fn drain_worker(&mut self) {
        self.decoded_sounds.drain_finished();
        self.apply_existence();
    }

    /// Applies the existence facts the cache collected to the registry, in one pass: a missing
    /// file leaves every pool (vanilla's "file does not exist, cannot add it to event"), a found
    /// one is no longer presumed.
    fn apply_existence(&mut self) {
        if self.decoded_sounds.existence.is_empty() {
            return;
        }
        let facts: HashMap<ResourceLocation, bool> =
            self.decoded_sounds.existence.drain(..).collect();
        for (file, _) in facts.iter().filter(|(_, exists)| !**exists) {
            tracing::warn!(file = %file, "file does not exist, cannot add it to event");
        }
        self.registry.settle_files(|file| facts.get(file).copied());
    }

    /// The backend half of [`AudioEngine::play`]: load, decode, hand to `kira`. Split out so the
    /// bookkeeping above has one place to observe success or failure rather than five.
    ///
    /// # Panics
    ///
    /// If no backend is running. [`AudioEngine::play`] checks that before calling this.
    fn play_resolved(
        &mut self,
        computed: &ComputedPlayback,
        resources: &ResourceManager,
    ) -> Result<PlayingSound, AudioError> {
        let db = amplitude_to_decibels(computed.linear_gain);

        if computed.streaming {
            let bytes = resources
                .load(&computed.file)
                .map_err(|source| AudioError::Asset {
                    location: computed.file.to_string(),
                    source,
                })?;
            return self.play_stream(Arc::from(bytes), computed, db);
        }
        match self.decoded_sounds.prepare(&computed.file, resources)? {
            Prepared::Decoded(sample_rate, frames) => {
                let manager = self
                    .manager
                    .as_mut()
                    .expect("play() returns NoBackend before reaching here");
                let data = StaticSoundData {
                    sample_rate,
                    frames,
                    settings: static_settings(computed, db),
                    slice: None,
                };
                let handle = manager
                    .play(data)
                    .map_err(|e| AudioError::Playback(e.to_string()))?;
                Ok(PlayingSound::Static(Arc::new(Mutex::new(handle))))
            }
            // Not decoded yet: this play streams with the same volume, pitch, pan and loop the
            // static sound would have had. kira's stream thread has the first frames about
            // 0.8-1 ms after this call (Vorbis headers plus one packet, measured in the dev
            // build), the same work the old inline decode did here before the sound could start,
            // so the sound starts as it did, usually in the same mixer block; vanilla's paulscode
            // starts it only after the whole clip is decoded.
            Prepared::Stream(bytes) => self.play_stream(bytes, computed, db),
        }
    }

    /// Starts `bytes` as a kira stream with `computed`'s settings.
    fn play_stream(
        &mut self,
        bytes: Arc<[u8]>,
        computed: &ComputedPlayback,
        db: Decibels,
    ) -> Result<PlayingSound, AudioError> {
        let decoder = VorbisDecoder::new(bytes, computed.file.to_string())?;
        let manager = self
            .manager
            .as_mut()
            .expect("play() returns NoBackend before reaching here");
        let data = StreamingSoundData::from_decoder(decoder)
            .volume(db)
            .playback_rate(computed.pitch as f64)
            .panning(computed.pan);
        let data = if backend_loops(computed) {
            data.loop_region(0.0..)
        } else {
            data
        };
        let handle = manager
            .play(data)
            .map_err(|e| AudioError::Playback(e.to_string()))?;
        let handle: StreamHandle = Arc::new(Mutex::new(handle));
        // Remember it until the mixer marks it stopped (see `finish_streams`). The list holds
        // the streams playing now plus those ended since the last start: a handful.
        self.streams.retain(|stream| !stream_stopped(stream));
        self.streams.push(Arc::clone(&handle));
        Ok(PlayingSound::Streaming(handle))
    }
}

#[cfg(test)]
mod tests {
    /// A decoder panic during prewarm is contained: the file is skipped and the flag the client's
    /// panic hook consults is set only while the decode runs.
    #[test]
    fn a_panicking_prewarm_decode_is_contained_and_the_flag_resets() {
        assert!(!super::prewarm_panic_is_being_contained());
        let skipped: Option<u8> = super::contain_prewarm_panic(|| {
            assert!(super::prewarm_panic_is_being_contained());
            panic!("decoder bug");
        });
        assert_eq!(skipped, None);
        assert!(!super::prewarm_panic_is_being_contained());
        assert_eq!(super::contain_prewarm_panic(|| Some(7_u8)), Some(7));
    }

    use super::*;
    use kira::backend::mock::MockBackend;
    use mc_assets::ResourceLocation;

    fn resource_manager_with(json: &str) -> ResourceManager {
        use std::collections::HashMap;
        use std::io::{Cursor, Read};

        struct MemPack(HashMap<String, Vec<u8>>);
        impl mc_assets::ResourcePack for MemPack {
            fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
                self.0
                    .get(&l.pack_entry_name())
                    .cloned()
                    .map(|v| Box::new(Cursor::new(v)) as Box<dyn Read>)
                    .or_else(|| {
                        // SoundRegistry probes every FILE variant while loading, just as
                        // SoundHandler opens the OGG before registering its accessor.  The
                        // synthetic registry fixture claims those files in `contains` below;
                        // make the matching reader available too so these engine tests remain
                        // about playback math rather than an accidentally empty pool.
                        (l.path.starts_with("sounds/") && l.path.ends_with(".ogg"))
                            .then(|| Box::new(Cursor::new(Vec::new())) as Box<dyn Read>)
                    })
            }
            fn contains(&self, l: &ResourceLocation) -> bool {
                // `SoundRegistry::load` now drops `FILE` variants whose `.ogg` is absent, the
                // way `SoundHandler.loadSoundResource` does (`F-P2-client.audio-02-6`). These
                // fixtures synthesize matching `.ogg` readers above, so the registry has the
                // same non-empty shape as the real asset set. `a_missing_ogg_is_recorded_as_a_failure`
                // is unaffected: it uses a name that resolves in this registry but has no bytes
                // in the separate empty manager passed to playback, which is a *decode* failure,
                // not a registration one.
                self.0.contains_key(&l.pack_entry_name())
                    || (l.path.starts_with("sounds/") && l.path.ends_with(".ogg"))
            }
            fn namespaces(&self) -> Vec<String> {
                vec!["minecraft".to_owned()]
            }
            fn name(&self) -> &str {
                "mem"
            }
        }
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(MemPack(HashMap::from([(
            "assets/minecraft/sounds.json".to_owned(),
            json.as_bytes().to_vec(),
        )]))));
        rm
    }

    fn registry_with(json: &str) -> SoundRegistry {
        SoundRegistry::load(&resource_manager_with(json)).unwrap()
    }

    /// HITCH-3: the prewarm set is every non-streaming file the selected events can reach,
    /// through `event` indirection, each once — and selecting it draws nothing from the RNG.
    #[test]
    fn prewarm_set_follows_indirection_and_skips_streams() {
        let registry = registry_with(
            r#"{"random.explode": {"category": "block", "sounds": ["random/explode1", "random/explode2"]},
                "dig.stone": {"category": "block", "sounds": ["dig/stone1",
                    {"type": "event", "name": "random.explode"}, "random/explode1"]},
                "music.game": {"category": "music", "sounds": [{"name": "music/calm1", "stream": true}]}}"#,
        );
        let mut files = registry.static_files_where(|event| {
            event.path.starts_with("dig.") || event.path == "music.game"
        });
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let paths: Vec<&str> = files.iter().map(|file| file.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "sounds/dig/stone1.ogg",
                "sounds/random/explode1.ogg",
                "sounds/random/explode2.ogg"
            ]
        );
    }

    /// A cache whose worker is the test: it receives the jobs and sends the outcomes itself.
    #[allow(clippy::type_complexity)]
    fn cache_with_fake_worker() -> (
        DecodedSoundCache,
        std::sync::mpsc::Receiver<DecodeJob>,
        std::sync::mpsc::Sender<DecodeOutcome>,
    ) {
        let (jobs, queue) = std::sync::mpsc::channel();
        let (finished, done) = std::sync::mpsc::channel();
        let cache = DecodedSoundCache {
            worker: Some(DecodeWorker { jobs, done }),
            ..DecodedSoundCache::default()
        };
        (cache, queue, finished)
    }

    fn test_frames(left: f32, count: usize) -> Arc<[Frame]> {
        Arc::from(vec![Frame { left, right: -left }; count])
    }

    /// Finished background decodes are served exactly like a first-play decode, a resident
    /// decode that got there first is kept, and `clear()` (reload) cuts the worker off.
    #[test]
    fn worker_frames_are_served_and_a_clear_discards_in_flight_work() {
        let (mut cache, _queue, finished) = cache_with_fake_worker();
        let file = ResourceLocation::minecraft("sounds/random/pop.ogg");
        let empty = || Source::Bytes(Arc::from(Vec::new()));
        assert!(cache.queue(&file, empty(), Want::Frames, false));
        let frames = test_frames(0.25, 4);
        finished
            .send((file.clone(), Decoded::Frames(22_050, Arc::clone(&frames))))
            .unwrap();
        // An empty manager: a miss would fail to load, so success proves the worker served it.
        let Prepared::Decoded(sample_rate, served) = cache
            .prepare(&file, &ResourceManager::new())
            .expect("served from the worker")
        else {
            panic!("a finished decode must be served from memory");
        };
        assert_eq!(sample_rate, 22_050);
        assert!(Arc::ptr_eq(&served, &frames));
        assert_eq!(cache.frames, 4);
        assert!(cache.pending.is_empty());

        // A second decode of a resident file changes nothing.
        assert!(cache.queue(&file, empty(), Want::Frames, false));
        finished
            .send((file.clone(), Decoded::Frames(1, test_frames(1.0, 2))))
            .unwrap();
        cache.drain_finished();
        assert!(
            Arc::ptr_eq(&cache.sounds[&file].1, &frames),
            "first resident decode wins"
        );

        cache.clear();
        assert!(cache.worker.is_none());
        assert!(cache.sounds.is_empty() && cache.pending.is_empty());
        assert!(
            finished.send((file, Decoded::Frames(1, frames))).is_err(),
            "a reload must stop the old pack's decodes"
        );
    }

    /// Until its decode lands, a queued file plays from the queued bytes: nothing is read from
    /// the (empty) manager and nothing is decoded on this thread.
    #[test]
    fn a_queued_file_streams_from_its_bytes_until_the_decode_lands() {
        let Some(bytes) = vanilla_ogg("sounds/random/pop.ogg") else {
            return;
        };
        let (mut cache, queue, finished) = cache_with_fake_worker();
        let file = ResourceLocation::minecraft("sounds/random/pop.ogg");
        assert!(cache.queue(
            &file,
            Source::Bytes(Arc::clone(&bytes)),
            Want::Frames,
            false
        ));
        let Prepared::Stream(streamed) = cache
            .prepare(&file, &ResourceManager::new())
            .expect("streams from the queued bytes")
        else {
            panic!("a pending file must stream, not decode inline");
        };
        assert!(Arc::ptr_eq(&streamed, &bytes));
        assert!(
            cache.sounds.is_empty(),
            "nothing was decoded on this thread"
        );
        assert!(cache.pending[&file].played, "a play now wants it");
        let (job, _, _) = queue.try_recv().expect("the file went to the worker");
        assert_eq!(job, file);

        finished
            .send((file.clone(), Decoded::Undecodable("bad packet".to_owned())))
            .unwrap();
        let Err(AudioError::Decode { reason, .. }) = cache.prepare(&file, &ResourceManager::new())
        else {
            panic!("a failed background decode fails the play like an inline decode did");
        };
        assert_eq!(reason, "bad packet");
    }

    /// The frame cap: a decode a play asked for always goes in (starting the set over), one only
    /// the prewarm asked for goes in only if it fits.
    #[test]
    fn the_frame_cap_never_lets_the_prewarm_empty_the_cache() {
        let mut cache = DecodedSoundCache::default();
        let big = ResourceLocation::minecraft("sounds/big.ogg");
        let more = ResourceLocation::minecraft("sounds/more.ogg");
        let last = ResourceLocation::minecraft("sounds/last.ogg");
        cache.insert(
            big.clone(),
            1,
            test_frames(0.1, MAX_CACHED_SOUND_FRAMES - 4),
            true,
        );
        cache.insert(more.clone(), 1, test_frames(0.1, 8), false);
        assert!(cache.sounds.contains_key(&big) && !cache.sounds.contains_key(&more));
        cache.insert(last.clone(), 1, test_frames(0.1, 8), true);
        assert!(!cache.sounds.contains_key(&big) && cache.sounds.contains_key(&last));
        assert_eq!(cache.frames, 8);
    }

    /// The real decode thread: the frames it produces are `decode_all`'s, and a garbage file is
    /// reported, not dropped.
    #[test]
    fn the_decode_worker_matches_decode_all_and_reports_failures() {
        let Some(bytes) = vanilla_ogg("sounds/random/pop.ogg") else {
            return;
        };
        let worker = DecodeWorker::start().expect("a thread starts");
        let good = ResourceLocation::minecraft("sounds/random/pop.ogg");
        let bad = ResourceLocation::minecraft("sounds/garbage.ogg");
        let gone = ResourceLocation::minecraft("sounds/gone.ogg");
        let dir = tempfile::tempdir().unwrap();
        let on_disk = dir.path().join("pop.ogg");
        std::fs::write(&on_disk, &bytes).unwrap();
        let file = |path: std::path::PathBuf| Source::File { path, cap: 1 << 20 };
        let decode = |name: &ResourceLocation, source| (name.clone(), source, Want::Frames);
        worker
            .jobs
            .send(decode(&good, file(on_disk.clone())))
            .unwrap();
        let garbage = Source::Bytes(Arc::from(b"not an ogg".to_vec()));
        worker.jobs.send(decode(&bad, garbage)).unwrap();
        worker
            .jobs
            .send(decode(&gone, file(dir.path().join("gone.ogg"))))
            .unwrap();
        // Over the read cap: refused like `ResourceManager::load` refuses it.
        let tiny_cap = Source::File {
            path: on_disk.clone(),
            cap: 16,
        };
        worker.jobs.send(decode(&good, tiny_cap)).unwrap();
        // A read-ahead job only reads.
        worker
            .jobs
            .send((good.clone(), file(on_disk), Want::Bytes))
            .unwrap();

        let timeout = std::time::Duration::from_secs(60);
        let (file, decoded) = worker.done.recv_timeout(timeout).expect("first result");
        assert_eq!(file, good);
        let Decoded::Frames(sample_rate, frames) = decoded else {
            panic!("pop.ogg read from disk and decoded on the worker");
        };
        let expected = decode_all(&bytes, "pop").unwrap();
        assert_eq!(sample_rate, expected.sample_rate);
        assert_eq!(&*frames, expected.to_frames().as_slice());
        let (file, decoded) = worker.done.recv_timeout(timeout).expect("second result");
        assert_eq!(file, bad);
        assert!(matches!(decoded, Decoded::Undecodable(_)));
        let (file, decoded) = worker.done.recv_timeout(timeout).expect("third result");
        assert_eq!(file, gone);
        assert!(matches!(decoded, Decoded::Unreadable));
        let (_, decoded) = worker.done.recv_timeout(timeout).expect("fourth result");
        assert!(matches!(decoded, Decoded::Unreadable));
        let (_, decoded) = worker.done.recv_timeout(timeout).expect("fifth result");
        let Decoded::Read(read) = decoded else {
            panic!("a read-ahead job must not decode");
        };
        assert_eq!(&*read, &*bytes);
    }

    /// One pack serving `json` as `sounds.json` that *lists* every `.ogg` in memory (as the asset
    /// index does) but stores only `stored`, like a partial launcher download. Counts every
    /// storage call made for a sound file.
    struct ListedPack {
        json: &'static str,
        stored: Vec<&'static str>,
        storage_calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl mc_assets::ResourcePack for ListedPack {
        fn open(&self, l: &ResourceLocation) -> Option<Box<dyn std::io::Read>> {
            if l.path == "sounds.json" {
                return Some(Box::new(std::io::Cursor::new(
                    self.json.as_bytes().to_vec(),
                )));
            }
            if l.path.ends_with(".ogg") {
                self.storage_calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            self.stored
                .contains(&l.path.as_str())
                .then(|| Box::new(std::io::Cursor::new(Vec::new())) as Box<dyn std::io::Read>)
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            if l.path == "sounds.json" {
                return true;
            }
            if l.path.ends_with(".ogg") {
                self.storage_calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            self.stored.contains(&l.path.as_str())
        }
        fn lists(&self, l: &ResourceLocation) -> Option<bool> {
            l.path.ends_with(".ogg").then_some(true)
        }
        fn namespaces(&self) -> Vec<String> {
            vec!["minecraft".to_owned()]
        }
    }

    const PARTIAL_JSON: &str = r#"{"random.pop": {"category": "player", "sounds": [
        "random/gone", "random/here", {"name": "random/also", "weight": 2}]}}"#;

    fn partial_download() -> (ResourceManager, Arc<std::sync::atomic::AtomicUsize>) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(ListedPack {
            json: PARTIAL_JSON,
            stored: vec!["sounds/random/here.ogg", "sounds/random/also.ogg"],
            storage_calls: Arc::clone(&calls),
        }));
        (rm, calls)
    }

    /// A pick never touches storage on the calling
    /// thread, which is the game thread. A metadata call there froze SkyWars frames for up to
    /// 3 s while builds saturated the disk. Neither an engine without a backend (every benchmark
    /// runs with `MC_NO_AUDIO`) nor one with a backend may make one.
    #[test]
    fn a_pick_never_touches_storage_on_the_calling_thread() {
        let (rm, calls) = partial_download();
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.pop"));
        let silent = AudioEngine::<MockBackend>::silent(SoundRegistry::load(&rm).unwrap(), 7);
        let audible = AudioEngine::<MockBackend>::new(SoundRegistry::load(&rm).unwrap(), 7);
        for mut engine in [silent, audible] {
            calls.store(0, std::sync::atomic::Ordering::Relaxed);
            for _ in 0..50 {
                engine.pick_existing(&request.event);
            }
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::Relaxed),
                0,
                "a pick touched storage on the calling thread"
            );
        }
    }

    /// How often each file wins 600 picks of `random.pop`.
    fn pick_counts(engine: &mut AudioEngine<MockBackend>) -> HashMap<String, u32> {
        let event = ResourceLocation::minecraft("random.pop");
        let mut picks: HashMap<String, u32> = HashMap::new();
        for _ in 0..600 {
            let variant = engine.pick_existing(&event).expect("a file to play");
            *picks.entry(variant.location.path).or_default() += 1;
        }
        picks
    }

    /// Loading `sounds.json` opens no sound file a pack lists in
    /// memory. Follow-up 2: picks never ask the disk either. An engine without a backend
    /// (`MC_NO_AUDIO`) never checks, so a listed file that is missing stays in its pool there
    /// (nothing is audible either way). With a backend, the worker reads every effect after load;
    /// a missing one then leaves the pool and the rest keep vanilla's odds (1 : 2 here), all
    /// without a storage call on the calling thread.
    #[test]
    fn a_missing_listed_file_leaves_the_pool_once_the_worker_has_checked_it() {
        let (rm, calls) = partial_download();
        let registry = SoundRegistry::load(&rm).unwrap();
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "loading sounds.json opened or probed a listed sound file"
        );
        let mut silent = AudioEngine::<MockBackend>::silent(registry.clone(), 7);
        let picks = pick_counts(&mut silent);
        assert!(picks.contains_key("sounds/random/gone.ogg"), "{picks:?}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);

        // The same files kept as plain files in a directory, `gone.ogg` missing.
        let dir = tempfile::tempdir().unwrap();
        for name in ["sounds_random_here.ogg", "sounds_random_also.ogg"] {
            std::fs::write(dir.path().join(name), b"not decoded here").unwrap();
        }
        let mut on_disk = ResourceManager::new();
        on_disk.push_pack(Box::new(PlainFilePack {
            dir: dir.path().to_path_buf(),
            json: PARTIAL_JSON,
            storage_calls: Arc::clone(&calls),
        }));
        let mut audible = AudioEngine::<MockBackend>::new(registry, 7);
        audible.prewarm_static_sounds(&on_disk, |_| true);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !audible.decoded_sounds.pending.is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "the worker never finished"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
            audible.drain_finished_decodes();
        }
        let picks = pick_counts(&mut audible);
        assert_eq!(picks.get("sounds/random/gone.ogg"), None, "{picks:?}");
        let here = f64::from(picks["sounds/random/here.ogg"]);
        let also = f64::from(picks["sounds/random/also.ogg"]);
        assert!((1.6..2.5).contains(&(also / here)), "{picks:?}");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the checks ran on the worker, the picks on in-memory facts"
        );
    }

    /// A pack that keeps its `.ogg` files as plain files in a directory (as the asset store and
    /// folder packs do), lists every `.ogg` in memory, and counts storage calls made through it.
    struct PlainFilePack {
        dir: std::path::PathBuf,
        json: &'static str,
        storage_calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl PlainFilePack {
        fn path_of(&self, l: &ResourceLocation) -> std::path::PathBuf {
            self.dir.join(l.path.replace('/', "_"))
        }
    }

    impl mc_assets::ResourcePack for PlainFilePack {
        fn open(&self, l: &ResourceLocation) -> Option<Box<dyn std::io::Read>> {
            if l.path == "sounds.json" {
                return Some(Box::new(std::io::Cursor::new(
                    self.json.as_bytes().to_vec(),
                )));
            }
            if !l.path.ends_with(".ogg") {
                return None;
            }
            self.storage_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let file = std::fs::File::open(self.path_of(l)).ok()?;
            Some(Box::new(file))
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            if l.path == "sounds.json" {
                return true;
            }
            if !l.path.ends_with(".ogg") {
                return false;
            }
            self.storage_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.path_of(l).is_file()
        }
        fn lists(&self, l: &ResourceLocation) -> Option<bool> {
            l.path.ends_with(".ogg").then_some(true)
        }
        fn file_path(&self, l: &ResourceLocation) -> Option<std::path::PathBuf> {
            l.path.ends_with(".ogg").then(|| self.path_of(l))
        }
        fn namespaces(&self) -> Vec<String> {
            vec!["minecraft".to_owned()]
        }
    }

    /// The start-up prewarm hands plain files to the worker by
    /// path, so the calling (game) thread makes no storage call at all; the worker reads and
    /// decodes them. A listed file missing on disk comes back unreadable and is left to the first
    /// pick's check, which drops it.
    #[test]
    fn the_prewarm_reads_plain_files_on_the_worker_not_the_calling_thread() {
        let Some(bytes) = vanilla_ogg("sounds/random/pop.ogg") else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("sounds_random_pop.ogg"), &bytes).unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(PlainFilePack {
            dir: dir.path().to_path_buf(),
            json: r#"{"random.pop": {"category": "player", "sounds": ["random/pop", "random/gone"]}}"#,
            storage_calls: Arc::clone(&calls),
        }));
        let mut engine = AudioEngine::<MockBackend>::new(SoundRegistry::load(&rm).unwrap(), 7);
        assert_eq!(engine.prewarm_static_sounds(&rm, |_| true), 2);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the prewarm touched a sound file on the calling thread"
        );

        let pop = ResourceLocation::minecraft("sounds/random/pop.ogg");
        let gone = ResourceLocation::minecraft("sounds/random/gone.ogg");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !engine.decoded_sounds.pending.is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "the worker never finished"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
            engine.decoded_sounds.drain_finished();
        }
        let expected = decode_all(&bytes, "pop").unwrap().to_frames();
        assert_eq!(&*engine.decoded_sounds.sounds[&pop].1, expected.as_slice());
        assert!(!engine.decoded_sounds.sounds.contains_key(&gone));
        assert!(!engine.decoded_sounds.failed.contains_key(&gone));

        // The first pick that lands on the missing file drops it; every play is then pop.ogg.
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.pop"));
        for _ in 0..20 {
            let sound = engine
                .play(&request, &Listener::origin(), &rm)
                .unwrap()
                .expect("audible");
            assert!(
                matches!(sound, PlayingSound::Static(_)),
                "served from the cache"
            );
            let record = engine.last_play().unwrap();
            assert_eq!(record.computed.as_ref().unwrap().file, pop);
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the worker checked the files; picks and plays used what it found"
        );
        engine.stop_all_sounds();
    }

    /// A clip longer than 30 s comes back from the worker still
    /// compressed, and every play streams it from memory without reading or decoding it again.
    /// Vanilla's longest effect stays under the line, so vanilla effects are cached as before.
    #[test]
    fn a_long_clip_is_kept_compressed_and_streamed_from_memory() {
        let (Some(song), Some(dragon)) = (
            vanilla_ogg("sounds/records/13.ogg"),
            vanilla_ogg("sounds/mob/enderdragon/end.ogg"),
        ) else {
            return;
        };
        assert!(is_always_streamed(&song));
        assert!(
            !is_always_streamed(&dragon),
            "vanilla's longest effect is decoded as before"
        );

        let worker = DecodeWorker::start().expect("a thread starts");
        let file = ResourceLocation::minecraft("sounds/records/13.ogg");
        worker
            .jobs
            .send((file.clone(), Source::Bytes(Arc::clone(&song)), Want::Frames))
            .unwrap();
        let (_, decoded) = worker
            .done
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("a result");
        let Decoded::Long(kept) = decoded else {
            panic!("a three-minute clip must not be decoded into the cache");
        };
        assert!(Arc::ptr_eq(&kept, &song));

        let mut cache = DecodedSoundCache::default();
        cache.keep_streamed(file.clone(), kept);
        for _ in 0..3 {
            // An empty manager: a read would fail, so success proves it came from memory.
            let Prepared::Stream(streamed) = cache
                .prepare(&file, &ResourceManager::new())
                .expect("served from memory")
            else {
                panic!("a long clip always streams");
            };
            assert!(Arc::ptr_eq(&streamed, &song));
        }
        assert!(cache.sounds.is_empty() && cache.worker.is_none());
    }

    /// An effect a resource pack supplies itself is prewarmed
    /// whatever its event, so its first play is a cache hit. Any other effect is read ahead (not
    /// decoded) behind it, so its first play streams from memory: the calling thread makes no
    /// storage call at all, not even the existence check of a presumed file.
    #[test]
    fn effects_a_resource_pack_supplies_are_prewarmed_and_the_rest_read_ahead() {
        let Some(bytes) = vanilla_ogg("sounds/random/pop.ogg") else {
            return;
        };
        // The base: lists every .ogg in memory but no files of its own (like the asset store).
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let base_dir = tempfile::tempdir().unwrap();
        std::fs::write(base_dir.path().join("sounds_mob_base_only.ogg"), &bytes).unwrap();
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(PlainFilePack {
            dir: base_dir.path().to_path_buf(),
            json: r#"{"mob.custom": {"category": "hostile", "sounds": ["mob/custom"]},
                      "mob.base": {"category": "hostile", "sounds": ["mob/base_only"]}}"#,
            storage_calls: Arc::clone(&calls),
        }));
        // The resource pack on top, a real folder pack carrying its own mob/custom.ogg.
        let pack_dir = tempfile::tempdir().unwrap();
        let sounds = pack_dir.path().join("assets/minecraft/sounds/mob");
        std::fs::create_dir_all(&sounds).unwrap();
        std::fs::write(sounds.join("custom.ogg"), &bytes).unwrap();
        rm.push_pack(Box::new(
            mc_assets::FolderResourcePack::open(pack_dir.path()).unwrap(),
        ));

        let mut engine = AudioEngine::<MockBackend>::new(SoundRegistry::load(&rm).unwrap(), 7);
        let loaded = calls.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            engine.prewarm_static_sounds(&rm, |_| false),
            1,
            "one decode queued"
        );
        let custom = ResourceLocation::minecraft("sounds/mob/custom.ogg");
        let base_only = ResourceLocation::minecraft("sounds/mob/base_only.ogg");
        assert!(engine.decoded_sounds.pending.contains_key(&custom));
        assert!(
            engine.decoded_sounds.pending.contains_key(&base_only),
            "read ahead"
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !engine.decoded_sounds.pending.is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "the worker never finished"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
            engine.decoded_sounds.drain_finished();
        }
        assert!(engine.decoded_sounds.sounds.contains_key(&custom));
        assert!(!engine.decoded_sounds.sounds.contains_key(&base_only));
        assert!(engine.decoded_sounds.read_ahead.contains_key(&base_only));

        let custom_play = PlayRequest::ambient(ResourceLocation::minecraft("mob.custom"));
        let sound = engine
            .play(&custom_play, &Listener::origin(), &rm)
            .unwrap()
            .expect("audible");
        assert!(
            matches!(sound, PlayingSound::Static(_)),
            "first play is a cache hit"
        );

        let base_play = PlayRequest::ambient(ResourceLocation::minecraft("mob.base"));
        let sound = engine
            .play(&base_play, &Listener::origin(), &rm)
            .unwrap()
            .expect("audible");
        assert!(
            matches!(sound, PlayingSound::Streaming(_)),
            "streams the read-ahead bytes"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            loaded,
            "no storage call on the calling thread after the registry load"
        );
        // The read-ahead file is now being decoded for its later plays.
        assert!(engine.decoded_sounds.pending.contains_key(&base_only));
        engine.stop_all_sounds();
        // Let the mock mixer stop the stream, so its decode thread ends with the test.
        engine.backend_pump = Some(pump_mock);
        engine.finish_streams();
    }

    /// One audio period of kira's mock backend, which has no audio thread of its own: read the
    /// handles' commands, then mix one buffer.
    fn pump_mock(backend: &mut MockBackend) {
        backend.on_start_processing();
        backend.process();
    }

    /// An engine on kira's mock backend whose registry plays the three-minute `records/13` as
    /// the effect `test.song` (always streamed), with the pack serving it from a temp dir.
    #[allow(clippy::type_complexity)]
    fn song_engine() -> Option<(
        AudioEngine<MockBackend>,
        ResourceManager,
        SoundRegistry,
        tempfile::TempDir,
    )> {
        let song = vanilla_ogg("sounds/records/13.ogg")?;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("sounds_records_13.ogg"), &song).unwrap();
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(PlainFilePack {
            dir: dir.path().to_path_buf(),
            json: r#"{"test.song": {"category": "master", "sounds": ["records/13"]}}"#,
            storage_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }));
        let registry = SoundRegistry::load(&rm).unwrap();
        let engine = AudioEngine::<MockBackend>::new(registry.clone(), 7);
        Some((engine, rm, registry, dir))
    }

    /// kira ends a stream's decode thread only once the mixer has
    /// marked the stream `Stopped`. A reload drops the old backend, so it first lets the mixer
    /// stop every stream; before, each stream still playing at a reload left a thread behind that
    /// woke every millisecond for the rest of the process. The thread owns the stream's decoder,
    /// which shares the clip's bytes, so the bytes' reference count shows whether it ended.
    #[test]
    fn a_reload_lets_every_stream_decode_thread_end() {
        let Some((mut engine, rm, registry, _dir)) = song_engine() else {
            return;
        };
        engine.backend_pump = Some(pump_mock);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("test.song"));
        let sound = engine
            .play(&request, &Listener::origin(), &rm)
            .unwrap()
            .expect("audible");
        assert!(matches!(sound, PlayingSound::Streaming(_)));
        let file = ResourceLocation::minecraft("sounds/records/13.ogg");
        let bytes = Arc::clone(&engine.decoded_sounds.streamed[&file]);
        // Let kira's decode thread start and fill its ring, as a stream playing for a while has.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            Arc::strong_count(&bytes) > 2,
            "the stream's decoder shares the bytes"
        );
        assert_eq!(engine.streams.len(), 1);

        engine.reload_registry(registry);

        // The reload emptied the cache, so once the decode thread has ended only `bytes` is left.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while Arc::strong_count(&bytes) > 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the stream's decode thread outlived the reload"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(engine.streams.is_empty());
        assert_eq!(sound.state(), PlaybackState::Stopped);
    }

    /// The engine remembers a stream only until the mixer has stopped it, so the list stays as
    /// small as the number of streams playing.
    #[test]
    fn the_engine_tracks_only_streams_the_mixer_has_not_stopped() {
        let Some((mut engine, rm, _registry, _dir)) = song_engine() else {
            return;
        };
        let request = PlayRequest::ambient(ResourceLocation::minecraft("test.song"));
        for _ in 0..3 {
            engine.play(&request, &Listener::origin(), &rm).unwrap();
        }
        assert_eq!(engine.streams.len(), 3);
        engine.stop_all_sounds();
        pump_mock(engine.backend_mut().expect("mock backend"));
        engine.play(&request, &Listener::origin(), &rm).unwrap();
        assert_eq!(
            engine.streams.len(),
            1,
            "the three stopped streams were let go"
        );

        engine.backend_pump = Some(pump_mock);
        engine.finish_streams();
        assert!(engine.streams.is_empty());
    }

    /// A real vanilla `.ogg` from the jars directory (`MC_JARS_DIR`, or the checkout's own),
    /// read straight from the object store. `None` (and a SKIP line) when it is not there.
    fn vanilla_ogg(path: &str) -> Option<Arc<[u8]>> {
        let jars = match std::env::var_os("MC_JARS_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)?
                .join("MCP-919-main")
                .join("jars"),
        };
        let pack = mc_assets::ObjectStorePack::open(jars.join("assets"), "1.8").ok();
        let Some(pack) = pack else {
            eprintln!("SKIP: no 1.8 asset index; set MC_JARS_DIR to the jars/ directory");
            return None;
        };
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(pack));
        rm.load(&ResourceLocation::minecraft(path))
            .ok()
            .map(Arc::from)
    }

    /// Installs a decoded one-shot directly into Kira so registry/tick tests never depend on a
    /// speaker or an external OGG asset. Production playback still goes through `play_resolved`;
    /// this fixture only gives the retained-channel code a real MockBackend handle to mutate.
    fn insert_mock_channel(
        engine: &mut AudioEngine<MockBackend>,
        request: PlayRequest,
        tickable: Option<Box<dyn TickableSound>>,
    ) -> PlayingSound {
        let variant = engine
            .registry
            .pick_variant(&request.event, &mut engine.rng)
            .expect("test event has a variant");
        let event_gain = engine.event_gains.get(&request.event);
        let computed = compute_with_variant(
            &engine.registry,
            &engine.categories,
            &engine.listener,
            &request,
            &variant,
            event_gain,
        )
        .expect("test event resolves");
        assert!(computed.linear_gain > 0.0, "test channel must be audible");

        let decoded = crate::ogg::DecodedAudio {
            sample_rate: 44_100,
            channels: 1,
            samples: vec![8_192; 128],
        };
        let data = StaticSoundData {
            sample_rate: decoded.sample_rate,
            frames: Arc::from(decoded.to_frames()),
            settings: StaticSoundSettings::new(),
            slice: None,
        };
        let handle = engine
            .manager
            .as_mut()
            .expect("MockBackend starts")
            .play(data)
            .expect("Kira accepts synthetic PCM");
        let sound = PlayingSound::Static(Arc::new(Mutex::new(handle)));
        let channel_id = engine.next_channel_id;
        engine.next_channel_id = engine.next_channel_id.wrapping_add(1);
        let category = computed.category;
        let has_tickable = tickable.is_some();
        engine.playing_sounds.insert(
            channel_id,
            ActiveSound {
                request,
                variant,
                category,
                computed,
                sound: sound.clone(),
                tickable,
                event_gain,
                stop_time: engine.play_time.saturating_add(20),
            },
        );
        if category != SoundCategory::Master {
            engine
                .category_sounds
                .entry(category)
                .or_default()
                .insert(channel_id);
        }
        if has_tickable {
            engine.tickable_sounds.insert(channel_id);
        }
        sound
    }

    struct ProbeTick {
        request: PlayRequest,
        updates: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl TickableSound for ProbeTick {
        fn update(&mut self) {
            self.request.volume = 0.25;
            self.request.position = Some(DVec3::new(4.0, 0.0, 0.0));
            self.updates
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        fn is_done_playing(&self) -> bool {
            false
        }

        fn request(&self) -> PlayRequest {
            self.request.clone()
        }
    }

    #[test]
    fn amplitude_decibels_round_trip() {
        // 0.001 is excluded deliberately: `20*log10(0.001) == -60.0` exactly, which lands on
        // `Decibels::SILENCE`'s own boundary, and kira's `as_amplitude` clamps `<= SILENCE` to
        // `0.0` — so that one input's "round trip" legitimately loses precision at the edge
        // kira itself defines as silent, not because this conversion is wrong.
        for a in [0.01_f32, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let db = amplitude_to_decibels(a);
            let back = db.as_amplitude();
            assert!((back - a).abs() < 1e-4, "{a} -> {db:?} -> {back}");
        }
    }

    /// An S29 sound effect's volume is a raw server float. A NaN one passed
    /// `normalized_volume`'s clamp and the `<= 0.0` silence check, and reached kira as NaN
    /// decibels: NaN samples in the mix. It is silence now; every finite gain is unchanged.
    #[test]
    fn a_nan_gain_is_silence_not_nan_samples() {
        let nan_gain = normalized_volume(f32::NAN, 1.0, 1.0);
        assert!(
            nan_gain.is_nan(),
            "the clamp lets NaN through, as Java's does"
        );
        for amplitude in [nan_gain, f32::NAN, -f32::NAN, 0.0, -1.0] {
            let db = amplitude_to_decibels(amplitude);
            assert_eq!(db.0, Decibels::SILENCE.0, "{amplitude}");
            assert_eq!(db.as_amplitude(), 0.0);
        }
        assert_eq!(amplitude_to_decibels(1.0).0, 0.0);
        assert!((amplitude_to_decibels(0.5).0 - 20.0 * 0.5f32.log10()).abs() < 1e-6);
    }

    #[test]
    fn amplitude_at_the_silence_boundary_clamps_to_zero() {
        let db = amplitude_to_decibels(0.001);
        assert_eq!(db, Decibels::SILENCE);
        assert_eq!(db.as_amplitude(), 0.0);
    }

    #[test]
    fn zero_amplitude_is_silence() {
        assert_eq!(amplitude_to_decibels(0.0), Decibels::SILENCE);
        assert_eq!(amplitude_to_decibels(-1.0), Decibels::SILENCE);
    }

    #[test]
    fn unregistered_event_is_an_error_not_a_silent_none() {
        let registry = SoundRegistry::empty();
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("nope"));
        let err =
            compute_playback(&registry, &categories, &listener, &request, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::UnknownEvent(_)));
    }

    #[test]
    fn category_volume_zero_silences_without_error() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut categories = CategoryVolumes::new();
        categories.set(SoundCategory::Players, 0.0);
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let result =
            compute_playback(&registry, &categories, &listener, &request, &mut rng).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn far_away_positional_sound_attenuates_to_silence() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let request = PlayRequest::positional(
            ResourceLocation::minecraft("random.click"),
            DVec3::new(1000.0, 0.0, 0.0),
        );
        let result =
            compute_playback(&registry, &categories, &listener, &request, &mut rng).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn nearby_positional_sound_computes_full_gain() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let request = PlayRequest::positional(
            ResourceLocation::minecraft("random.click"),
            listener.position,
        );
        let computed = compute_playback(&registry, &categories, &listener, &request, &mut rng)
            .unwrap()
            .unwrap();
        assert_eq!(computed.linear_gain, 1.0);
        assert_eq!(computed.category, SoundCategory::Players);
        assert!(!computed.streaming);
        assert_eq!(
            computed.file.to_string(),
            "minecraft:sounds/random/click.ogg"
        );
    }

    #[test]
    fn ambient_music_ignores_distance() {
        let registry = registry_with(
            r#"{"music.menu": {"category": "music", "sounds": [{"name":"music/menu/menu1","stream":true}]}}"#,
        );
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("music.menu"));
        let computed = compute_playback(&registry, &categories, &listener, &request, &mut rng)
            .unwrap()
            .unwrap();
        assert_eq!(computed.linear_gain, 1.0);
        assert!(computed.streaming);
    }

    // ---- the play-record tail (GAP A14-01/A14-02's observation point) ----------------------

    /// The whole point of [`AudioEngine::recent`]: on a host with no output device `play`
    /// returns `Ok(None)` and there is otherwise *nothing* to assert on, which is how a client
    /// can be wired up to a silent engine and nobody notice for a whole section. The record has
    /// to carry enough to tell "asked for the right sound" from "asked for the wrong one".
    #[test]
    fn a_silent_engine_still_records_what_it_was_asked_to_play() {
        let registry = registry_with(
            r#"{"random.explode": {"category": "block", "sounds": ["random/explode"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::silent(registry, 0);
        assert!(engine.is_silent());

        let resources = ResourceManager::new();
        let listener = Listener::origin();
        let request = PlayRequest {
            event: ResourceLocation::minecraft("random.explode"),
            volume: 4.0,
            pitch: 0.75,
            position: Some(DVec3::new(1.0, 2.0, 3.0)),
            attenuation: AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        };
        // Nothing comes out and nothing errors — exactly the shape that let a whole crate sit
        // unwired without a failing test.
        assert!(engine
            .play(&request, &listener, &resources)
            .expect("a silent engine reports success")
            .is_none());

        let record = engine.last_play().expect("the attempt must be recorded");
        assert_eq!(record.outcome, PlayOutcome::NoBackend);
        assert_eq!(record.request, request);
        let computed = record.computed.as_ref().expect("resolution still ran");
        assert_eq!(computed.category, SoundCategory::Blocks);
        // `normalized_pitch(0.75, 1.0)`.
        assert!((computed.pitch - 0.75).abs() < 1e-6);
        // volume 4.0 -> fade distance 64, and the source is ~3.7 blocks out, so still loud.
        assert!(computed.linear_gain > 0.9, "{}", computed.linear_gain);
    }

    /// The capture tap sees every started sound with
    /// the variant the weighted pick chose; suppression mutes the game's own requests; and a
    /// captured sound replays with its recorded variant, not a fresh pick.
    #[test]
    fn captured_sounds_carry_their_variant_and_replay_it_under_suppression() {
        let registry = registry_with(
            r#"{"dig.stone": {"category": "block",
                "sounds": ["dig/stone1", "dig/stone2", "dig/stone3", "dig/stone4"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::silent(registry, 7);
        let resources = ResourceManager::new();
        let listener = Listener::origin();
        // Next to the listener, so every play resolves to an audible computed playback.
        let request = PlayRequest::at(
            ResourceLocation::minecraft("dig.stone"),
            1.0,
            0.8,
            DVec3::new(0.5, 0.5, 0.5),
        );

        // Off by default: nothing is copied.
        let _ = engine.play(&request, &listener, &resources);
        let mut out = Vec::new();
        engine.drain_captured(&mut out);
        assert!(out.is_empty());

        engine.set_capture(true);
        for _ in 0..16 {
            let _ = engine.play(&request, &listener, &resources);
        }
        engine.drain_captured(&mut out);
        assert_eq!(out.len(), 16);
        assert!(
            out.iter()
                .all(|sound| sound.request == request
                    && sound.category == Some(SoundCategory::Blocks))
        );
        let files: HashSet<String> = out
            .iter()
            .map(|sound| {
                sound
                    .variant
                    .as_ref()
                    .expect("picked")
                    .location
                    .path
                    .clone()
            })
            .collect();
        assert!(files.len() > 1, "the weighted pick varies: {files:?}");

        // Suppressed: the game's requests resolve nothing and leave no record or capture.
        engine.set_suppressed(true);
        let before = engine.recent().count();
        let last = engine.last_play().cloned();
        assert!(engine
            .play(&request, &listener, &resources)
            .expect("muted")
            .is_none());
        assert_eq!(engine.recent().count(), before);
        assert_eq!(engine.last_play().cloned(), last);
        let mut muted = Vec::new();
        engine.drain_captured(&mut muted);
        assert!(muted.is_empty(), "a muted request is not captured");

        // A replayed sound plays through suppression with its recorded variant, every time.
        engine.set_capture(false);
        let recorded = out[3].clone();
        for _ in 0..8 {
            let _ = engine.play_captured(&recorded, &listener, &resources);
            let played = engine.last_play().expect("replayed");
            let computed = played.computed.as_ref().expect("resolved");
            assert_eq!(
                computed.file,
                recorded.variant.as_ref().expect("picked").location,
                "the recorded variant, not a fresh pick"
            );
        }
        engine.set_suppressed(false);
        assert!(!engine.is_suppressed());
    }

    /// The failure path must be recorded too, or a resource pack missing one `.ogg` would look
    /// identical to a sound the game never asked for.
    #[test]
    fn a_missing_ogg_is_recorded_as_a_failure_not_a_silence() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        assert!(!engine.is_silent(), "MockBackend must start");
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        assert!(engine
            .play(&request, &Listener::origin(), &ResourceManager::new())
            .is_err());
        let record = engine.last_play().expect("recorded");
        assert_eq!(record.outcome, PlayOutcome::Failed);
        assert!(record.computed.is_some());
    }

    #[test]
    fn an_unknown_event_is_recorded_as_unresolved() {
        let mut engine = AudioEngine::<MockBackend>::new(SoundRegistry::empty(), 0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("nope"));
        let err = engine
            .play(&request, &Listener::origin(), &ResourceManager::new())
            .unwrap_err();
        assert!(matches!(err, AudioError::UnknownEvent(_)));

        let record = engine.last_play().expect("recorded even though it failed");
        assert_eq!(record.outcome, PlayOutcome::Unresolved);
        assert_eq!(record.computed, None);
        assert_eq!(record.request.event.path, "nope");
    }

    #[test]
    fn a_sound_silenced_by_its_category_slider_is_recorded_as_inaudible() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        engine.categories.set(SoundCategory::Players, 0.0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        assert!(engine
            .play(&request, &Listener::origin(), &ResourceManager::new())
            .expect("a zeroed slider is not an error")
            .is_none());
        let record = engine.last_play().expect("recorded");
        assert_eq!(record.outcome, PlayOutcome::Inaudible);
        // Nothing resolved, so there is no category or gain to report — the request is all
        // there is, and it is what a "why is this silent?" question needs.
        assert_eq!(record.computed, None);
        assert_eq!(record.request.event.path, "random.click");
    }

    #[test]
    fn the_record_tail_is_bounded_and_keeps_the_newest() {
        let mut engine = AudioEngine::<MockBackend>::new(SoundRegistry::empty(), 0);
        let listener = Listener::origin();
        let resources = ResourceManager::new();
        for i in 0..(RECENT_PLAY_CAPACITY * 2) {
            let request = PlayRequest::ambient(ResourceLocation::minecraft(format!("e{i}")));
            let _ = engine.play(&request, &listener, &resources);
        }
        assert_eq!(engine.recent().count(), RECENT_PLAY_CAPACITY);
        assert_eq!(
            engine.last_play().expect("non-empty").request.event.path,
            format!("e{}", RECENT_PLAY_CAPACITY * 2 - 1)
        );
        assert_eq!(
            engine
                .recent()
                .next()
                .expect("non-empty")
                .request
                .event
                .path,
            format!("e{}", RECENT_PLAY_CAPACITY)
        );

        engine.clear_recent();
        assert_eq!(engine.recent().count(), 0);
        assert_eq!(engine.last_play(), None);
    }

    #[test]
    fn louder_than_normal_sound_is_still_audible_further_out() {
        let registry = registry_with(
            r#"{"random.explode": {"category": "block", "sounds": ["random/explode"]}}"#,
        );
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let request = PlayRequest::positional(
            ResourceLocation::minecraft("random.explode"),
            DVec3::new(32.0, 0.0, 0.0),
        );
        let computed = compute_playback(&registry, &categories, &listener, &request, &mut rng)
            .unwrap()
            .unwrap();
        assert!(computed.linear_gain > 0.0);
    }

    #[test]
    fn positioned_sound_record_factories_keep_all_four_vanilla_shapes_distinct() {
        let event = ResourceLocation::minecraft("probe");
        let position = DVec3::new(1.0, 2.0, 3.0);

        let pitch_factory = PlayRequest::ambient_with_pitch(event.clone(), 0.5);
        assert_eq!(pitch_factory.volume, 0.25);
        assert_eq!(pitch_factory.pitch, 0.5);
        assert_eq!(pitch_factory.position, None);
        assert_eq!(pitch_factory.attenuation, AttenuationType::None);
        assert!(!pitch_factory.repeat);
        assert_eq!(pitch_factory.repeat_delay_ticks, 0);

        let ambient_factory = PlayRequest::ambient(event.clone());
        assert_eq!(ambient_factory.volume, 1.0);
        assert_eq!(ambient_factory.pitch, 1.0);
        assert_eq!(ambient_factory.position, None);
        assert_eq!(ambient_factory.attenuation, AttenuationType::None);
        assert!(!ambient_factory.repeat);
        assert_eq!(ambient_factory.repeat_delay_ticks, 0);

        let positional_factory = PlayRequest::positional(event.clone(), position);
        assert_eq!(positional_factory.volume, 4.0);
        assert_eq!(positional_factory.pitch, 1.0);
        assert_eq!(positional_factory.position, Some(position));
        assert_eq!(positional_factory.attenuation, AttenuationType::Linear);
        assert!(!positional_factory.repeat);
        assert_eq!(positional_factory.repeat_delay_ticks, 0);

        let constructor = PlayRequest::at(event, 0.75, 1.25, position);
        assert_eq!(constructor.volume, 0.75);
        assert_eq!(constructor.pitch, 1.25);
        assert_eq!(constructor.position, Some(position));
        assert_eq!(constructor.attenuation, AttenuationType::Linear);
        assert!(!constructor.repeat);
        assert_eq!(constructor.repeat_delay_ticks, 0);
    }

    #[test]
    fn repeat_contract_survives_resolution_with_signed_delay_intact() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);

        // `ISound.getRepeatDelay()` is an `int`, not a duration or an unsigned count.  The
        // positive value is the one `SoundManager` will later schedule after a completed play.
        let request =
            PlayRequest::ambient(ResourceLocation::minecraft("random.click")).with_repeat(true, 37);
        let computed = compute_playback(&registry, &categories, &listener, &request, &mut rng)
            .unwrap()
            .unwrap();
        assert!(computed.repeat);
        assert_eq!(computed.repeat_delay_ticks, 37);

        // A non-positive delay must remain distinguishable: vanilla does not schedule it, but
        // the ISound value itself is not clamped on the way through SoundManager.
        let request =
            PlayRequest::ambient(ResourceLocation::minecraft("random.click")).with_repeat(true, -1);
        let computed = compute_playback(&registry, &categories, &listener, &request, &mut rng)
            .unwrap()
            .unwrap();
        assert!(computed.repeat);
        assert_eq!(computed.repeat_delay_ticks, -1);
    }

    #[test]
    fn completed_positive_delay_repeat_replays_on_its_exact_client_tick() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::silent(registry, 0);
        let request =
            PlayRequest::ambient(ResourceLocation::minecraft("random.click")).with_repeat(true, 3);

        // A completion during tick zero records an absolute deadline of tick three.  The first
        // two updateAllSounds passes must leave it delayed; the third re-enters the normal play
        // funnel (silent here, so the PlayRecord is the portable proof).
        engine.queue_completed_repeat(request.clone());
        assert_eq!(engine.delayed_repeat_count(), 1);
        for expected_tick in 1..=2 {
            assert!(engine
                .tick_repeats(&Listener::origin(), &ResourceManager::new())
                .is_empty());
            assert_eq!(engine.play_time, expected_tick);
            assert_eq!(engine.delayed_repeat_count(), 1);
            assert!(engine.last_play().is_none());
        }
        assert!(engine
            .tick_repeats(&Listener::origin(), &ResourceManager::new())
            .is_empty());
        assert_eq!(engine.play_time, 3);
        assert_eq!(engine.delayed_repeat_count(), 0);
        let replay = engine
            .last_play()
            .expect("due repeat must use AudioEngine::play");
        assert_eq!(replay.request, request);
        assert_eq!(replay.outcome, PlayOutcome::NoBackend);
    }

    #[test]
    fn completed_zero_or_negative_delay_never_enters_delayed_repeat_queue() {
        let mut engine = AudioEngine::<MockBackend>::silent(SoundRegistry::empty(), 0);
        let event = ResourceLocation::minecraft("random.click");
        engine.queue_completed_repeat(PlayRequest::ambient(event.clone()).with_repeat(true, 0));
        engine.queue_completed_repeat(PlayRequest::ambient(event).with_repeat(true, -7));
        engine.queue_completed_repeat(
            PlayRequest::ambient(ResourceLocation::minecraft("random.click")).with_repeat(false, 9),
        );
        assert_eq!(engine.delayed_repeat_count(), 0);
    }

    #[test]
    fn explicit_delayed_sound_replays_at_the_absolute_java_tick_without_becoming_a_repeat() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::silent(registry, 0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));

        // SoundManager.playDelayedSound stores playTime + delay.  Starting at playTime 0,
        // delay 3 is therefore due on the third update pass, not immediately and not on tick 4.
        engine.play_delayed_sound(request.clone(), 3);
        assert_eq!(engine.current_tick(), 0);
        assert_eq!(engine.delayed_repeat_count(), 1);
        assert!(engine.last_play().is_none());
        for expected_tick in 1..=2 {
            engine.update_all_sounds(&Listener::origin(), &ResourceManager::new());
            assert_eq!(engine.current_tick(), expected_tick);
            assert_eq!(engine.delayed_repeat_count(), 1);
            assert!(engine.last_play().is_none());
        }

        engine.update_all_sounds(&Listener::origin(), &ResourceManager::new());
        let replay = engine
            .last_play()
            .expect("the delayed request must re-enter the ordinary play funnel");
        assert_eq!(engine.current_tick(), 3);
        assert_eq!(engine.delayed_repeat_count(), 0);
        assert_eq!(replay.request, request);
        assert!(!replay.request.repeat);
        assert_eq!(replay.outcome, PlayOutcome::NoBackend);
    }

    #[test]
    fn explicit_zero_and_negative_delays_wait_for_the_next_update_then_play() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::silent(registry, 0);
        let zero = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let negative = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));

        // Java enqueues even a non-positive delay; it never calls playSound inline.  Both
        // deadlines are already due, so one update pass drains both entries in insertion order.
        engine.play_delayed_sound(zero.clone(), 0);
        engine.play_delayed_sound(negative.clone(), -7);
        assert_eq!(engine.delayed_repeat_count(), 2);
        assert!(engine.last_play().is_none());

        engine.update_all_sounds(&Listener::origin(), &ResourceManager::new());
        let records: Vec<_> = engine.recent().collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].request, zero);
        assert_eq!(records[1].request, negative);
        assert_eq!(engine.delayed_repeat_count(), 0);
    }

    #[test]
    fn a_positioned_unattenuated_sound_does_not_fade_with_distance() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let mut request = PlayRequest::at(
            ResourceLocation::minecraft("random.click"),
            1.0,
            1.0,
            DVec3::new(1_000.0, 0.0, 0.0),
        );
        request.attenuation = AttenuationType::None;

        let computed = compute_playback(&registry, &categories, &listener, &request, &mut rng)
            .unwrap()
            .unwrap();
        assert_eq!(computed.linear_gain, 1.0);

        // Negative control: the same positioned source with LINEAR attenuation is inaudible.
        request.attenuation = AttenuationType::Linear;
        assert!(
            compute_playback(&registry, &categories, &listener, &request, &mut rng)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn zero_delay_repeat_uses_a_backend_loop_but_positive_delay_does_not() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let categories = CategoryVolumes::new();
        let listener = Listener::origin();
        let mut rng = JavaRandom::new(0);
        let event = ResourceLocation::minecraft("random.click");

        let zero = compute_playback(
            &registry,
            &categories,
            &listener,
            &PlayRequest::ambient(event.clone()).with_repeat(true, 0),
            &mut rng,
        )
        .unwrap()
        .unwrap();
        let delayed = compute_playback(
            &registry,
            &categories,
            &listener,
            &PlayRequest::ambient(event).with_repeat(true, 4),
            &mut rng,
        )
        .unwrap()
        .unwrap();

        assert!(backend_loops(&zero));
        assert!(!backend_loops(&delayed));
        assert!(static_settings(&zero, Decibels::IDENTITY)
            .loop_region
            .is_some());
        assert!(static_settings(&delayed, Decibels::IDENTITY)
            .loop_region
            .is_none());
    }

    #[test]
    fn update_all_sounds_ticks_and_refreshes_a_moving_channel_in_place() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        let updates = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let probe = ProbeTick {
            request: request.clone(),
            updates: updates.clone(),
        };
        let sound = insert_mock_channel(&mut engine, request, Some(Box::new(probe)));
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();

        assert_eq!(engine.active_sound_count(), 1);
        assert_eq!(engine.tickable_sound_count(), 1);
        assert!(engine.is_sound_playing(&sound));
        assert!(engine
            .update_all_sounds(&Listener::origin(), &ResourceManager::new())
            .is_empty());
        assert_eq!(updates.load(std::sync::atomic::Ordering::Relaxed), 1);
        let active = engine.playing_sounds.values().next().expect("retained");
        assert_eq!(active.request.volume, 0.25);
        assert_eq!(active.request.position, Some(DVec3::new(4.0, 0.0, 0.0)));
        assert!((active.computed.linear_gain - 0.25).abs() < 1e-6);
    }

    #[test]
    fn update_all_sounds_recomputes_positioned_attenuation_for_a_moved_listener() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        let request = PlayRequest::at(
            ResourceLocation::minecraft("random.click"),
            1.0,
            1.0,
            DVec3::new(8.0, 0.0, 0.0),
        );
        let _sound = insert_mock_channel(&mut engine, request, None);
        let initial = engine
            .playing_sounds
            .values()
            .next()
            .expect("retained")
            .computed
            .linear_gain;

        let mut moved_listener = Listener::origin();
        moved_listener.position = DVec3::new(4.0, 0.0, 0.0);
        assert!(engine
            .update_all_sounds(&moved_listener, &ResourceManager::new())
            .is_empty());
        let refreshed = engine
            .playing_sounds
            .values()
            .next()
            .expect("retained")
            .computed
            .linear_gain;
        assert!(refreshed > initial, "{initial} -> {refreshed}");
        assert!((refreshed - 0.75).abs() < 1e-6);
    }

    #[test]
    fn update_all_sounds_keeps_the_original_weighted_variant() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/first", "random/second"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 17);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let _sound = insert_mock_channel(&mut engine, request, None);
        let selected = engine
            .playing_sounds
            .values()
            .next()
            .expect("retained")
            .variant
            .clone();

        engine.update_all_sounds(&Listener::origin(), &ResourceManager::new());
        let after_tick = engine
            .playing_sounds
            .values()
            .next()
            .expect("retained")
            .variant
            .clone();
        assert_eq!(after_tick, selected);
    }

    #[test]
    fn category_slider_mutes_live_channels_without_dropping_registry_grace() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let sound = insert_mock_channel(&mut engine, request, None);
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();

        engine.set_sound_category_volume(SoundCategory::Players, 0.0);
        let active = engine.playing_sounds.values().next().expect("stop grace");
        assert_eq!(active.computed.linear_gain, 0.0);
        engine.backend_mut().expect("MockBackend starts").process();
        assert_eq!(sound.state(), PlaybackState::Stopped);
        assert!(!engine.is_sound_playing(&sound));
        assert_eq!(engine.active_sound_count(), 1);
    }

    // ---- the Sounds mod's per-event gains ----------------

    fn gains_of(entries: &[(&str, f32)]) -> EventGains {
        let mut gains = EventGains::new();
        for &(event, gain) in entries {
            gains.set(ResourceLocation::minecraft(event), gain);
        }
        gains
    }

    /// The recorded resolution of one play on a silent engine (no device, nothing decoded).
    fn played(engine: &mut AudioEngine<MockBackend>, request: &PlayRequest) -> PlayRecord {
        let _ = engine.play(request, &Listener::origin(), &ResourceManager::new());
        engine.last_play().expect("recorded").clone()
    }

    /// With an empty table every resolved number is exactly vanilla's `compute_playback`, for
    /// ambient, near, far and loud requests: the mod off changes nothing, not even rounding.
    #[test]
    fn an_empty_gain_table_leaves_every_number_vanilla() {
        let json = r#"{"random.explode": {"category": "block", "sounds": [
            {"name": "random/explode", "volume": 0.7, "pitch": 1.2}]}}"#;
        let mut engine = AudioEngine::<MockBackend>::silent(registry_with(json), 0);
        let event = ResourceLocation::minecraft("random.explode");
        let requests = [
            PlayRequest::ambient(event.clone()),
            PlayRequest::positional(event.clone(), DVec3::new(3.0, 1.0, -2.0)),
            PlayRequest::positional(event.clone(), DVec3::new(40.0, 0.0, 0.0)),
            PlayRequest::at(event.clone(), 0.3, 0.8, DVec3::new(5.0, 0.0, 5.0)),
        ];
        for request in &requests {
            let vanilla = compute_playback(
                &engine.registry,
                &engine.categories,
                &Listener::origin(),
                request,
                &mut JavaRandom::new(0),
            )
            .unwrap();
            assert_eq!(
                played(&mut engine, request).computed,
                vanilla,
                "{request:?}"
            );
        }
        // A table that changes only another event is just as invisible.
        engine.set_event_gains(gains_of(&[("random.click", 0.0)]));
        for request in &requests {
            let vanilla = compute_playback(
                &engine.registry,
                &engine.categories,
                &Listener::origin(),
                request,
                &mut JavaRandom::new(0),
            )
            .unwrap();
            assert_eq!(
                played(&mut engine, request).computed,
                vanilla,
                "{request:?}"
            );
        }
    }

    /// A gain scales the final channel gain after vanilla's clamp (so 200 % of a full-scale sound
    /// really is +6 dB) and never widens the audible range the way a request volume above 1
    /// does.
    #[test]
    fn an_event_gain_scales_after_the_clamp_and_does_not_widen_the_range() {
        let json = r#"{"random.explode": {"category": "block", "sounds": ["random/explode"]}}"#;
        let mut engine = AudioEngine::<MockBackend>::silent(registry_with(json), 0);
        let event = ResourceLocation::minecraft("random.explode");

        // Full scale and unattenuated: vanilla clamps at 1.0, the gain goes on top.
        let ambient = PlayRequest::ambient(event.clone());
        assert_eq!(
            played(&mut engine, &ambient).computed.unwrap().linear_gain,
            1.0
        );
        engine.set_event_gains(gains_of(&[("random.explode", 2.0)]));
        let boosted = played(&mut engine, &ambient).computed.unwrap().linear_gain;
        assert_eq!(boosted, 2.0);
        assert!((amplitude_to_decibels(boosted).0 - 6.0206).abs() < 1e-3);

        // Half volume, positioned: exactly vanilla's gain times the factor.
        let near = PlayRequest::at(event.clone(), 1.0, 1.0, DVec3::new(6.0, 0.0, 0.0));
        engine.set_event_gains(EventGains::new());
        let vanilla = played(&mut engine, &near).computed.unwrap().linear_gain;
        engine.set_event_gains(gains_of(&[("random.explode", 0.5)]));
        let half = played(&mut engine, &near).computed.unwrap().linear_gain;
        assert_eq!(half, vanilla * 0.5);

        // 20 blocks out a volume-1 sound has faded out (fade distance 16). 200 % keeps it
        // silent; a request volume of 2 would have carried it to 32 blocks.
        let far = PlayRequest::at(event.clone(), 1.0, 1.0, DVec3::new(20.0, 0.0, 0.0));
        engine.set_event_gains(gains_of(&[("random.explode", 2.0)]));
        assert_eq!(played(&mut engine, &far).outcome, PlayOutcome::Inaudible);
        engine.set_event_gains(EventGains::new());
        let louder = PlayRequest::at(event, 2.0, 1.0, DVec3::new(20.0, 0.0, 0.0));
        assert_eq!(played(&mut engine, &louder).outcome, PlayOutcome::NoBackend);
    }

    /// A muted event is resolved and recorded as inaudible, and never reaches the decoder: with
    /// a running backend and no `.ogg` bytes, an unmuted play fails to load while a muted one
    /// returns cleanly.
    #[test]
    fn a_muted_event_is_inaudible_and_never_decoded() {
        let json = r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#;
        let mut engine = AudioEngine::<MockBackend>::new(registry_with(json), 0);
        assert!(!engine.is_silent(), "MockBackend must start");
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let resources = ResourceManager::new();
        assert!(engine
            .play(&request, &Listener::origin(), &resources)
            .is_err());
        assert_eq!(engine.last_play().unwrap().outcome, PlayOutcome::Failed);

        engine.set_event_gains(gains_of(&[("random.click", 0.0)]));
        assert!(engine
            .play(&request, &Listener::origin(), &resources)
            .expect("a muted event is not an error")
            .is_none());
        let record = engine.last_play().unwrap();
        assert_eq!(record.outcome, PlayOutcome::Inaudible);
        assert_eq!(record.computed, None);
        assert_eq!(engine.active_sound_count(), 0);
    }

    /// Changing the table reaches channels that are already playing, like a category slider:
    /// the new gain applies at once, survives the per-tick refresh, and 0 stops the channel.
    #[test]
    fn new_gains_reach_live_channels_and_survive_the_tick_refresh() {
        let json = r#"{"random.click": {"category": "player", "sounds": ["random/click"]},
            "random.pop": {"category": "player", "sounds": ["random/pop"]}}"#;
        let mut engine = AudioEngine::<MockBackend>::new(registry_with(json), 0);
        let click = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let pop = PlayRequest::ambient(ResourceLocation::minecraft("random.pop"));
        let click_sound = insert_mock_channel(&mut engine, click.clone(), None);
        let _pop_sound = insert_mock_channel(&mut engine, pop.clone(), None);
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        let gain_of = |engine: &AudioEngine<MockBackend>, event: &str| {
            engine
                .playing_sounds
                .values()
                .find(|active| active.request.event.path == event)
                .map(|active| active.computed.linear_gain)
                .expect("retained")
        };
        assert_eq!(gain_of(&engine, "random.click"), 1.0);

        engine.set_event_gains(gains_of(&[("random.click", 0.5)]));
        assert_eq!(gain_of(&engine, "random.click"), 0.5);
        assert_eq!(
            gain_of(&engine, "random.pop"),
            1.0,
            "other events untouched"
        );
        let _ = engine.update_all_sounds(&Listener::origin(), &ResourceManager::new());
        assert_eq!(
            gain_of(&engine, "random.click"),
            0.5,
            "the tick keeps the gain"
        );
        engine.set_sound_category_volume(SoundCategory::Players, 0.5);
        assert_eq!(
            gain_of(&engine, "random.click"),
            0.25,
            "and so does a category change"
        );

        engine.set_event_gains(gains_of(&[("random.click", 0.0)]));
        assert_eq!(gain_of(&engine, "random.click"), 0.0);
        engine.backend_mut().expect("MockBackend starts").process();
        assert_eq!(click_sound.state(), PlaybackState::Stopped);
    }

    /// The table is the user's setting, not part of the sound system: a resource-pack reload
    /// (which tears every channel down and swaps the registry) keeps it.
    #[test]
    fn event_gains_survive_a_registry_reload() {
        let json = r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#;
        let mut engine = AudioEngine::<MockBackend>::silent(registry_with(json), 0);
        engine.set_event_gains(gains_of(&[("random.click", 1.5)]));
        engine.reload_registry(registry_with(json));
        assert_eq!(
            engine
                .event_gains()
                .get(&ResourceLocation::minecraft("random.click")),
            1.5
        );
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        assert_eq!(
            played(&mut engine, &request).computed.unwrap().linear_gain,
            1.5
        );
    }

    #[test]
    fn master_zero_stops_existing_channels_and_skips_new_plays() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let _sound = insert_mock_channel(&mut engine, request.clone(), None);
        assert_eq!(engine.active_sound_count(), 1);

        engine.set_master_volume(0.0);
        assert_eq!(engine.active_sound_count(), 0);
        assert!(engine
            .play(&request, &Listener::origin(), &ResourceManager::new())
            .expect("master mute is a successful skip")
            .is_none());
        assert_eq!(
            engine.last_play().expect("recorded").outcome,
            PlayOutcome::Inaudible
        );
    }

    #[test]
    fn stop_all_sounds_stops_retained_channels_and_drops_delayed_repeats() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let sound = insert_mock_channel(&mut engine, request.clone(), None);
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        engine.schedule_delayed_sound(request, 8);
        assert_eq!(engine.active_sound_count(), 1);
        assert_eq!(engine.delayed_repeat_count(), 1);

        engine.stop_all_sounds();
        assert_eq!(engine.active_sound_count(), 0);
        assert_eq!(engine.delayed_repeat_count(), 0);
        assert_eq!(engine.tickable_sound_count(), 0);
        engine.backend_mut().expect("MockBackend starts").process();
        assert_eq!(sound.state(), PlaybackState::Stopped);
    }

    /// When the output library's stream thread dies (a failed restart
    /// after a device change), the next tick stops what was playing on the dead output and opens
    /// a new one, once. Negative control: a tick without that news leaves playback alone.
    #[test]
    fn an_output_thread_panic_restarts_the_output_once() {
        let json = r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#;
        let resources = resource_manager_with(json);
        let mut engine = AudioEngine::<MockBackend>::new(registry_with(json), 0);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let _sound = insert_mock_channel(&mut engine, request.clone(), None);
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        let _ = engine.update_all_sounds(&Listener::origin(), &resources);
        assert_eq!(engine.active_sound_count(), 1, "no news: playback goes on");

        // What the panic hook's `is_output_backend_panic` does to the shared count, seen from
        // this engine only (the shared count is left alone so no other test's engine restarts).
        engine.seen_output_panics = engine.seen_output_panics.wrapping_sub(1);
        let _ = engine.update_all_sounds(&Listener::origin(), &resources);
        assert_eq!(
            engine.active_sound_count(),
            0,
            "the dead output's sounds are gone"
        );
        assert!(!engine.is_silent(), "a new output was opened");
        assert_eq!(
            engine.seen_output_panics,
            crate::output_recovery::output_backend_panics()
        );

        let _sound = insert_mock_channel(&mut engine, request, None);
        engine
            .backend_mut()
            .expect("the new output")
            .on_start_processing();
        let _ = engine.update_all_sounds(&Listener::origin(), &resources);
        assert_eq!(
            engine.active_sound_count(),
            1,
            "restarted once, not every tick"
        );
    }

    /// Every output device gone: the output renders nothing and the
    /// output library never looks again. The engine's watch leaves a running output alone,
    /// waits while there is no device, and opens a new output once the default device is back
    /// (the dead output's sounds gone, the master volume kept), once, not at every look. An engine
    /// with no output at all (no device at start, or a restart that found none) gets one the same
    /// way; an engine built silent never looks.
    #[test]
    fn a_dead_output_comes_back_once_a_device_returns() {
        use crate::output_recovery::WATCH_TICKS;
        let json = r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#;
        let resources = resource_manager_with(json);
        let ticks = |engine: &mut AudioEngine<MockBackend>, count: u32| {
            for _ in 0..count {
                let _ = engine.update_all_sounds(&Listener::origin(), &resources);
            }
        };
        let mut engine = AudioEngine::<MockBackend>::new(registry_with(json), 0);
        engine.set_master_volume(0.5);
        let request = PlayRequest::ambient(ResourceLocation::minecraft("random.click"));
        let _sound = insert_mock_channel(&mut engine, request, None);
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();

        // A running output, many looks: untouched.
        engine.watch_test.periods = Some(Some(48));
        engine.watch_test.device = Some(true);
        ticks(&mut engine, WATCH_TICKS * 3);
        assert_eq!(engine.watch_test.reopened, 0);
        assert_eq!(engine.active_sound_count(), 1);

        // Every device gone: the stream renders nothing, and there is nothing to open.
        engine.watch_test.periods = Some(Some(0));
        engine.watch_test.device = Some(false);
        ticks(&mut engine, WATCH_TICKS * 3);
        assert_eq!(
            engine.watch_test.reopened, 0,
            "no device: nothing to open yet"
        );

        // The headphones are plugged in again: within one look, a new output.
        engine.watch_test.device = Some(true);
        ticks(&mut engine, WATCH_TICKS);
        assert_eq!(
            engine.watch_test.reopened, 1,
            "the output comes back with the device"
        );
        assert!(!engine.is_silent());
        assert_eq!(
            engine.active_sound_count(),
            0,
            "the dead output's sounds are gone"
        );
        assert_eq!(engine.categories.get(SoundCategory::Master), 0.5);

        // The new output renders: no further reopening.
        engine.watch_test.periods = Some(Some(10));
        ticks(&mut engine, WATCH_TICKS * 4);
        assert_eq!(
            engine.watch_test.reopened, 1,
            "reopened once, not at every look"
        );

        // No output at all (a restart that found no device): opened once a device is there.
        drop(engine.manager.take());
        assert!(engine.is_silent());
        engine.watch_test.device = Some(false);
        ticks(&mut engine, WATCH_TICKS * 2);
        assert!(engine.is_silent());
        engine.watch_test.device = Some(true);
        ticks(&mut engine, WATCH_TICKS);
        assert!(!engine.is_silent(), "sound is back");
        assert_eq!(engine.watch_test.reopened, 2);

        // An engine built silent stays silent.
        let mut silent = AudioEngine::<MockBackend>::silent(registry_with(json), 0);
        silent.watch_test.device = Some(true);
        ticks(&mut silent, WATCH_TICKS * 3);
        assert!(silent.is_silent());
        assert_eq!(silent.watch_test.reopened, 0);
    }

    /// The real output: a running stream shows the watch its periods
    /// (otherwise the watch would reopen a working output every few seconds), and the device
    /// check finds the device. Nothing is played. Without an output device: skipped.
    #[test]
    fn a_running_real_output_shows_the_watch_its_periods() {
        let json = r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#;
        let mut engine = AudioEngine::<DefaultBackend>::new(registry_with(json), 0);
        let Some(backend) = engine.backend_mut() else {
            eprintln!("no audio output device: skipping");
            return;
        };
        let _ = backend.periods_since_last_look();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let periods = backend.periods_since_last_look();
        assert!(
            periods.is_some_and(|periods| periods > 0),
            "a running output renders periods: {periods:?}"
        );
        assert!(AudioEngine::<DefaultBackend>::output_device_present());
        assert!(AudioEngine::<MockBackend>::output_device_present());
    }

    #[test]
    fn reload_sounds_tears_down_old_playback_before_adopting_new_registry() {
        let old_registry =
            registry_with(r#"{"old.event": {"category": "player", "sounds": ["old/event"]}}"#);
        let mut engine = AudioEngine::<MockBackend>::new(old_registry, 0);
        let old_request =
            PlayRequest::ambient(ResourceLocation::minecraft("old.event")).with_repeat(true, 4);
        let old_sound = insert_mock_channel(&mut engine, old_request.clone(), None);
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        engine.play_delayed_sound(old_request, 12);

        assert_eq!(engine.active_sound_count(), 1);
        assert_eq!(engine.active_repeat_count(), 1);
        assert_eq!(engine.delayed_sound_count(), 1);

        let new_resources = resource_manager_with(
            r#"{"new.event": {"category": "block", "sounds": ["new/event"]}}"#,
        );
        engine
            .reload_sounds(&new_resources)
            .expect("resource reload succeeds");

        // The reload boundary must stop and forget every old playback source before the new
        // registry is observable. This covers active, repeating and explicitly delayed state.
        assert_eq!(engine.active_sound_count(), 0);
        assert_eq!(engine.active_repeat_count(), 0);
        assert_eq!(engine.tickable_sound_count(), 0);
        assert_eq!(engine.delayed_sound_count(), 0);
        engine.backend_mut().expect("MockBackend starts").process();
        // Dropping kira's old manager releases its backend immediately. A cloned handle may
        // retain the last backend-reported state until that backend's final processing callback,
        // but the engine no longer recognizes it as live at this boundary.
        assert!(!engine.is_sound_playing(&old_sound));

        assert!(engine
            .registry
            .category_of(&ResourceLocation::minecraft("old.event"))
            .is_none());
        let new_event = ResourceLocation::minecraft("new.event");
        assert_eq!(
            engine.registry.category_of(&new_event),
            Some(SoundCategory::Blocks)
        );
        let computed = compute_playback(
            &engine.registry,
            &engine.categories,
            &Listener::origin(),
            &PlayRequest::ambient(new_event),
            &mut engine.rng,
        )
        .expect("post-reload sound resolves")
        .expect("post-reload sound remains audible");
        assert_eq!(computed.category, SoundCategory::Blocks);
    }

    #[test]
    fn reload_registry_clears_every_old_channel_index_and_queue_before_reuse() {
        let old_registry =
            registry_with(r#"{"old.event": {"category": "player", "sounds": ["old/event"]}}"#);
        let new_registry =
            registry_with(r#"{"new.event": {"category": "block", "sounds": ["new/event"]}}"#);
        let mut engine = AudioEngine::<MockBackend>::new(old_registry, 0);
        engine.categories.set(SoundCategory::Players, 0.35);

        let old_request =
            PlayRequest::ambient(ResourceLocation::minecraft("old.event")).with_repeat(true, 4);
        let updates = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let old_sound = insert_mock_channel(
            &mut engine,
            old_request.clone(),
            Some(Box::new(ProbeTick {
                request: old_request.clone(),
                updates,
            })),
        );
        engine.play_delayed_sound(old_request, 12);

        assert_eq!(engine.active_sound_count(), 1);
        assert_eq!(engine.active_repeat_count(), 1);
        assert_eq!(engine.tickable_sound_count(), 1);
        assert_eq!(engine.delayed_sound_count(), 1);
        assert!(engine.category_sounds.contains_key(&SoundCategory::Players));

        engine.reload_registry(new_registry);

        assert_eq!(engine.active_sound_count(), 0);
        assert_eq!(engine.active_repeat_count(), 0);
        assert_eq!(engine.tickable_sound_count(), 0);
        assert_eq!(engine.delayed_sound_count(), 0);
        assert!(engine.category_sounds.is_empty());
        assert_eq!(engine.categories.get(SoundCategory::Players), 0.35);
        // A cloned handle can retain the last backend-reported state until its old manager's
        // final processing callback, but the reloaded engine no longer recognizes it as live.
        assert!(!engine.is_sound_playing(&old_sound));
        assert!(engine.manager.is_some(), "reload must reopen the backend");
        assert!(engine
            .registry
            .category_of(&ResourceLocation::minecraft("old.event"))
            .is_none());

        // The fresh backend and fresh registry must be usable immediately after the boundary.
        let new_sound = insert_mock_channel(
            &mut engine,
            PlayRequest::ambient(ResourceLocation::minecraft("new.event")),
            None,
        );
        assert!(engine.is_sound_playing(&new_sound));
        assert_eq!(engine.active_sound_count(), 1);
        assert_eq!(
            engine
                .playing_sounds
                .values()
                .next()
                .expect("new channel retained")
                .category,
            SoundCategory::Blocks
        );
    }

    #[test]
    fn stopped_positive_repeat_waits_for_stop_grace_then_enters_delayed_queue() {
        let registry = registry_with(
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        );
        let mut engine = AudioEngine::<MockBackend>::new(registry, 0);
        let request =
            PlayRequest::ambient(ResourceLocation::minecraft("random.click")).with_repeat(true, 3);
        let mut sound = insert_mock_channel(&mut engine, request, None);
        engine
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        sound.stop_now();
        engine.backend_mut().expect("MockBackend starts").process();
        assert_eq!(sound.state(), PlaybackState::Stopped);

        for expected_tick in 1..20 {
            engine.update_all_sounds(&Listener::origin(), &ResourceManager::new());
            assert_eq!(engine.current_tick(), expected_tick);
            assert_eq!(engine.active_sound_count(), 1);
            assert_eq!(engine.delayed_repeat_count(), 0);
        }
        engine.update_all_sounds(&Listener::origin(), &ResourceManager::new());
        assert_eq!(engine.current_tick(), 20);
        assert_eq!(engine.active_sound_count(), 0);
        assert_eq!(engine.delayed_repeat_count(), 1);
    }
}
