//! The crate-wide error type.

use thiserror::Error;

/// Errors returned by `mc-audio`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AudioError {
    /// A sound event name was requested that no loaded `sounds.json` registered.
    ///
    /// This is not fatal — vanilla logs a warning and drops the sound
    /// (`SoundManager.playSound`, "Unable to play unknown soundEvent") rather than treating it
    /// as an error, so callers should generally log this and continue rather than propagate it.
    #[error("unknown sound event {0:?}")]
    UnknownEvent(String),

    /// A sound event resolved (its registration exists) but its variant pool is empty, or its
    /// event-type indirection chain never bottomed out in a playable file.
    ///
    /// java: `SoundHandler.missing_sound` — the sentinel `SoundPoolEntry` vanilla substitutes.
    #[error("sound event {0:?} has no playable variant")]
    EmptyEvent(String),

    /// `sounds.json` failed validation (bad category name, non-positive weight/volume/pitch,
    /// unknown `type`, ...). Carries the resource that failed and why.
    #[error("invalid sounds.json entry in {location}: {reason}")]
    InvalidSoundList {
        /// The `sounds.json` resource location that failed to parse.
        location: String,
        /// Human-readable reason, for logs.
        reason: String,
    },

    /// Loading a resource (a `sounds.json` or an `.ogg` blob) through `mc-assets` failed.
    #[error("loading {location} failed: {source}")]
    Asset {
        /// The resource location that failed to load.
        location: String,
        /// The underlying `mc-assets` error.
        #[source]
        source: mc_assets::AssetError,
    },

    /// The bytes at a `.ogg` location did not decode as valid Ogg Vorbis.
    #[error("failed to decode {location} as Ogg Vorbis: {reason}")]
    Decode {
        /// The `.ogg` resource location that failed to decode.
        location: String,
        /// Human-readable reason, for logs.
        reason: String,
    },

    /// The audio output backend failed to initialise (no device, permissions, ...).
    ///
    /// Not fatal: [`crate::AudioEngine`] falls back to a silent/headless mode on this, mirroring
    /// vanilla's own fallback (`SoundManager.loadSoundSystem` sets the master volume to zero and
    /// carries on when `SoundSystem` construction throws).
    #[error("audio backend unavailable: {0}")]
    Backend(String),

    /// `AudioManager::play` rejected a sound (resource limit reached, or the backend rejected
    /// the sound data).
    #[error("could not start playback: {0}")]
    Playback(String),
}
