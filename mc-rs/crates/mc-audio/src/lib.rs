#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Positional sound effects and music.
//!
//! # Layout
//! - [`category`] — [`SoundCategory`] and per-category `options.txt` volumes.
//! - [`sounds`] — `sounds.json` parsing into [`SoundRegistry`], the event → weighted-variant
//!   table (with `"type": "event"` indirection resolved recursively, matching
//!   `SoundEventAccessorComposite`).
//! - [`ogg`] — Ogg Vorbis decoding via `lewton`: [`ogg::decode_all`] for short buffered effects,
//!   [`ogg::VorbisDecoder`] (implements `kira`'s streaming `Decoder`) for music and records.
//! - [`attenuation`] — [`AttenuationType`], vanilla's linear distance falloff, the listener
//!   transform, and stereo panning.
//! - [`engine`] — [`AudioEngine`], the playback orchestrator: resolves a [`PlayRequest`] against
//!   the registry/categories/listener ([`compute_playback`], fully pure and unit-testable), then
//!   decodes and hands the result to `kira` for mixing/output.
//! - [`music`] — [`MusicTicker`], vanilla's menu/game/biome track-selection timing.
//! - `gains` — [`EventGains`], the Sounds mod's per-event volume table, applied by
//!   [`AudioEngine`] on top of vanilla's maths; empty = vanilla.
//! - [`packet`] — the documented, decoded shape of `S29PacketSoundEffect` and
//!   `S48PacketResourcePackSend`; see that module's doc comment for the S06 dependency note.
//!
//! # Verification without real speaker output
//!
//! This crate was built in a sandbox with no audio output device. Every number that determines
//! *what should be audible* — weighted variant selection, category/volume/pitch normalisation,
//! distance attenuation and its rolloff distance, panning, PCM decode and its dB conversion, the
//! music ticker's timing — is a pure function with unit tests pinning exact values (see each
//! module's `tests`). [`engine`]'s integration tests additionally drive the real `kira` mixing
//! graph end-to-end (decode → `kira::AudioManager::play` → render) using
//! `kira::backend::mock::MockBackend`, which needs no device, to confirm the pipeline actually
//! accepts and plays the decoded audio, not just that the surrounding math is right. What is
//! **not** verified here is that the real `cpal` backend correctly drives this sandbox's (or any
//! given deployment's) actual output device — that needs a human with speakers.
//!
//! # Reuse, not reimplementation
//!
//! `.ogg` bytes and `sounds.json` are loaded through [`mc_assets::ResourceManager`]
//! (`ResourceManager::load`), never by reading `jars/assets/objects/` paths directly — S04
//! already resolves the asset-index/pack-stack lookup this crate would otherwise have to repeat.

mod attenuation;
mod category;
mod engine;
mod error;
mod gains;
mod music;
pub mod ogg;
// The output thread dying after a device change is not a crash.
mod output_recovery;
pub mod packet;
mod sounds;

pub use attenuation::{attenuate, fade_distance, pan, AttenuationType, Listener};
pub use category::{CategoryVolumes, SoundCategory};
pub use engine::{
    amplitude_to_decibels, compute_playback, normalized_pitch, normalized_volume,
    prewarm_panic_is_being_contained, AudioEngine, CapturedSound, ComputedPlayback,
    DecodeCacheStats, PlayOutcome, PlayRecord, PlayRequest, PlayingSound, TickableSound,
    RECENT_PLAY_CAPACITY,
};
pub use error::AudioError;
pub use gains::{EventGains, MAX_EVENT_GAIN};
pub use music::{MusicPlayback, MusicTicker, MusicType};
pub use output_recovery::{is_output_backend_panic, OutputHeartbeat};
pub use sounds::{ResolvedVariant, SoundRegistry};

// Re-exported so callers don't need a direct `kira` dependency just to name the default backend
// type or drive the mock one in tests.
pub use kira::backend::{mock::MockBackend, Backend, DefaultBackend};
