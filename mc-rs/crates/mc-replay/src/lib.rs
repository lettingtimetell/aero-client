//! Aero replay files.
//!
//! This crate knows nothing about Minecraft beyond "a clientbound packet is bytes": it is the
//! container, its background writer, its random-access reader, clip cutting and the library of
//! replays on disk. The client records into it (`mc-client/src/app/mods/tw_replay_rec.rs`), plays
//! it back (`tw_replay_play.rs`), browses it (`tw_replay_view.rs`) and exports it
//! (`tw_replay_export.rs`).
//!
//! # The contract between the recorder, the player, the browser and the export
//!
//! A change to the types below that breaks a caller must fix every caller at the same time.
//!
//! * **Time** is [`ReplayTime`]: microseconds since the recording started, read from the game
//!   thread's frame clock. Pauses are cut out, so replay time never jumps.
//! * **Records** are `(time, stream, payload)`. The payload's meaning belongs to the stream's
//!   owner: [`StreamId::Packet`] is a clientbound Play packet exactly as the client applied it
//!   (packet id varint plus body, after decompression), [`StreamId::Client`] carries the
//!   local-player, frame, tick, sound and event records, and [`StreamId::Mic`] the encoded
//!   microphone audio.
//! * **Metadata** is [`ReplayMeta`], stored as JSON in the file.
//!
//! # The file (`.aeroreplay`, container version 1, all integers little-endian)
//!
//! ```text
//! header   "AEROREPL" | u16 version | u16 flags | u32 n | n bytes meta JSON | u32 crc32(JSON)
//! blocks   44-byte block header + payload, repeated (see `format`)
//! footer   "ARIX" | u32 count | count x 40-byte index entries | u32 n | final meta JSON
//!          | u32 crc32(footer up to here) | u64 footer offset | "AEROEND1"
//! ```
//!
//! A block is a run of records compressed together with zstd. Blocks are independent, so a
//! reader can start at any of them, and each carries checksums of its header and its payload.
//! The header's metadata is the one known when recording began; the footer's is the final one
//! (duration, markers). A recording that never got its footer (the game crashed) is still read:
//! [`Reader`] scans the blocks and stops at the first one that is incomplete or damaged.
//!
//! Records inside a block: `varint time delta` (from the previous record; the first one from the
//! block's first time), `u8 tag` (0 packet, 1 client, 2 mic, 3 marker, 4 desktop audio),
//! `u8 kind` (client records only), `varint length`, then the payload.
//!
//! # What a recording made by the client holds
//!
//! 1. Optionally a **state prefix** (blocks flagged [`BlockInfo::prefix`], all at time 0): the
//!    packets a world reset does not clear, so the world can be rebuilt from step 2 (the last
//!    `JoinGame` with the player's entity id, the tab list, the scoreboard and teams...).
//! 2. A **world reset** at time 0, first record of a block flagged [`BlockInfo::keyframe`]: a
//!    `JoinGame`, or a `Respawn` into another dimension.
//! 3. Everything after, in the order the client applied or produced it.
//!
//! The part to show is [`ReplayMeta::window`]; what comes before its start is fast-applied.

// `deny` rather than `forbid`, so the one module that talks to
// the Windows audio stack (`desktop/wasapi.rs`, process loopback through COM) can opt in. Every
// other module is still free of `unsafe`.
#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

mod clip;
// Helper threads that recover from their own panic (not the game's crash).
mod contained;
pub mod desktop;
mod error;
mod format;
mod library;
pub mod mic;
mod reader;
// The viewer's records, packed in memory.
mod store;
pub mod voice_out;
mod writer;

pub use clip::{cut_file, ClipCut};
pub use error::ReplayError;
pub use library::{
    date_folder, date_folder_in, local_stamp, local_title_time, localize_title, localize_title_in,
    restamp, restamp_in, Library, LibraryEntry, Usage, BOOKMARK_LABEL, CLIPS_DIR, MAX_DEPTH,
    REPLAYS_DIR,
};
// The browser's folders.
pub use library::{folder_key, is_safe_folder, LibraryTree, MAX_TREE_FOLDERS};
pub use reader::{
    markers_in, read_info, BlockInfo, Cursor, Reader, RecordRef, ReplayInfo,
    MAX_MARKER_SCAN_RECORDS, RECORDS_MEMORY_BASE, RECORDS_MEMORY_PER_FILE_BYTE,
};
pub use store::{BlockCache, RecordStore};
pub use writer::{Finished, Finishing, Snapshot, Writer, WriterOptions, WriterStats};

/// Microseconds since the recording started, on the game thread's frame clock, with pauses cut
/// out.
pub type ReplayTime = u64;

/// The file extension of replays and clips.
pub const EXTENSION: &str = "aeroreplay";

/// The container version this build writes (and the newest it reads).
pub const FORMAT_VERSION: u32 = 1;

/// Which stream a record belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StreamId {
    /// A clientbound Play packet as the client applied it: packet id varint plus body, after
    /// decompression and decryption.
    Packet,
    /// A client-side record. The number is the record kind (local player state, frame,
    /// tick, sound, local event).
    Client(u8),
    /// Encoded microphone audio.
    Mic,
    /// A marker or other note added while recording.
    Marker,
    /// Encoded desktop audio: other programs' sound, such as a voice call (container tag 4). The microphone's record layout ([`desktop`]).
    Desktop,
}

impl StreamId {
    /// This stream's bit in a block's stream mask ([`BlockInfo::streams`]).
    #[must_use]
    pub const fn mask(self) -> u16 {
        match self {
            StreamId::Packet => 1,
            StreamId::Client(_) => 2,
            StreamId::Mic => 4,
            StreamId::Marker => 8,
            StreamId::Desktop => 16,
        }
    }
}

/// One timestamped record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// When the client applied or produced it.
    pub time: ReplayTime,
    /// Which stream it belongs to.
    pub stream: StreamId,
    /// The stream owner's bytes.
    pub payload: Vec<u8>,
}

/// A full recording or a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplayKind {
    /// A whole recording.
    #[default]
    Full,
    /// A clip. The file holds everything needed to rebuild the world at `start` (the records from
    /// the last world join on), but the viewer and exporter show only `start..end`.
    Clip {
        /// Where the clip starts.
        start: ReplayTime,
        /// Where the clip ends.
        end: ReplayTime,
    },
}

/// A named point on the timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Marker {
    /// Where it is.
    pub time: ReplayTime,
    /// What it says.
    pub label: String,
}

/// How the microphone stream is encoded (see [`mic`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MicFormat {
    /// The codec: `"ima-adpcm"` (see [`mic::encode_frame`]).
    pub codec: String,
    /// Samples per second.
    pub sample_rate: u32,
    /// Channels (1: mono).
    pub channels: u16,
    /// Samples per channel in one record.
    pub frame_samples: u32,
}

/// What a replay file says about itself.
///
/// Build one with [`ReplayMeta::default`] and field writes rather than a struct literal: fields
/// get added (the JSON reads older files with defaults for the missing ones).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplayMeta {
    /// The container version.
    pub format_version: u32,
    /// Full recording or clip.
    pub kind: ReplayKind,
    /// The title shown in the browser (editable by renaming).
    pub title: String,
    /// The server address the player typed, for example `mc.example.net`.
    pub server: String,
    /// Wall-clock start, in milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// Length of the recording.
    pub duration: ReplayTime,
    /// The recording player's name.
    pub player_name: String,
    /// The recording player's UUID, hyphenated.
    pub player_uuid: String,
    /// The protocol version the packets are in (47 for 1.8.9).
    pub protocol: i32,
    /// The client build that recorded it.
    pub client_version: String,
    /// Whether a microphone stream is present.
    pub has_mic: bool,
    /// Markers.
    pub markers: Vec<Marker>,
    /// Full recordings: where the part the viewer shows starts. 0, unless the recording was
    /// started in the middle of a world, in which case the records before it rebuild that world
    /// (taken from the in-memory session buffer). Clips use [`ReplayKind::Clip`]'s `start`; read
    /// both through [`ReplayMeta::window`].
    pub view_start: ReplayTime,
    /// Times of the world resets in the file (a `JoinGame`, or a `Respawn` into another
    /// dimension): points from which the world is rebuilt from scratch.
    pub world_resets: Vec<ReplayTime>,
    /// The microphone's encoding, when [`ReplayMeta::has_mic`].
    pub mic_format: Option<MicFormat>,
    /// Whether a desktop audio stream ([`StreamId::Desktop`]) is present.
    pub has_desktop: bool,
    /// The desktop audio's encoding, when [`ReplayMeta::has_desktop`] ([`desktop::format`]).
    pub desktop_format: Option<MicFormat>,
    /// Free-form tags. The client records store the version of their encoding as
    /// `"rb.version"`.
    pub tags: BTreeMap<String, String>,
}

impl Default for ReplayMeta {
    fn default() -> ReplayMeta {
        ReplayMeta {
            format_version: FORMAT_VERSION,
            kind: ReplayKind::Full,
            title: String::new(),
            server: String::new(),
            started_unix_ms: 0,
            duration: 0,
            player_name: String::new(),
            player_uuid: String::new(),
            protocol: 47,
            client_version: String::new(),
            has_mic: false,
            markers: Vec::new(),
            view_start: 0,
            world_resets: Vec::new(),
            mic_format: None,
            has_desktop: false,
            desktop_format: None,
            tags: BTreeMap::new(),
        }
    }
}

impl ReplayMeta {
    /// The part of the timeline the viewer and the exporter show: `(start, end)`. Records before
    /// `start` only rebuild the world.
    #[must_use]
    pub fn window(&self) -> (ReplayTime, ReplayTime) {
        match self.kind {
            ReplayKind::Full => (self.view_start.min(self.duration), self.duration),
            ReplayKind::Clip { start, end } => (start.min(end), end),
        }
    }

    /// Whether this is a clip.
    #[must_use]
    pub fn is_clip(&self) -> bool {
        matches!(self.kind, ReplayKind::Clip { .. })
    }

    /// The JSON stored in the file.
    #[must_use]
    pub fn to_json(&self) -> String {
        // A struct of strings, integers, vectors and maps with string keys cannot fail to
        // serialise.
        serde_json::to_string(self).unwrap_or_else(|_| String::from("{}"))
    }

    /// Parse the JSON stored in a file. Unknown fields are ignored and missing ones take their
    /// defaults, so newer and older files both read. Values no recording can have are brought
    /// into range ([`ReplayMeta::sanitized`]).
    ///
    /// # Errors
    ///
    /// [`ReplayError::Meta`] if the text is not a JSON object of the right shape.
    pub fn from_json(text: &str) -> Result<ReplayMeta, ReplayError> {
        serde_json::from_str::<ReplayMeta>(text)
            .map(ReplayMeta::sanitized)
            .map_err(|error| ReplayError::Meta(error.to_string()))
    }

    /// This metadata with every value a recording made by the client
    /// cannot have brought into range, so a shared file (damaged, or crafted) cannot make the
    /// browser, the viewer or the exporter allocate, loop or overflow on it. Changes nothing in
    /// a file the client wrote:
    ///
    /// * times (duration, view start, clip bounds, markers, world resets) at most
    ///   [`MAX_META_TIME`] (30 days), the start at most [`MAX_STARTED_UNIX_MS`] (year 9999);
    /// * texts at most [`MAX_META_TEXT_CHARS`] characters (a rename keeps 64), at most
    ///   [`MAX_META_MARKERS`] markers and world resets, [`MAX_META_TAGS`] tags;
    /// * a microphone or desktop audio format with an impossible sample rate, channel count or
    ///   frame size is dropped (the exporter then uses the rate the client records at).
    #[must_use]
    pub fn sanitized(mut self) -> ReplayMeta {
        let time = |t: ReplayTime| t.min(MAX_META_TIME);
        self.started_unix_ms = self.started_unix_ms.min(MAX_STARTED_UNIX_MS);
        self.duration = time(self.duration);
        self.view_start = time(self.view_start);
        if let ReplayKind::Clip { start, end } = &mut self.kind {
            *start = time(*start);
            *end = time(*end);
        }
        for text in [
            &mut self.title,
            &mut self.server,
            &mut self.player_name,
            &mut self.player_uuid,
            &mut self.client_version,
        ] {
            truncate_chars(text, MAX_META_TEXT_CHARS);
        }
        self.markers.truncate(MAX_META_MARKERS);
        for marker in &mut self.markers {
            marker.time = time(marker.time);
            truncate_chars(&mut marker.label, MAX_META_TEXT_CHARS);
        }
        self.world_resets.truncate(MAX_META_MARKERS);
        for reset in &mut self.world_resets {
            *reset = time(*reset);
        }
        if self.tags.len() > MAX_META_TAGS {
            self.tags = std::mem::take(&mut self.tags)
                .into_iter()
                .take(MAX_META_TAGS)
                .collect();
        }
        for value in self.tags.values_mut() {
            truncate_chars(value, MAX_META_TEXT_CHARS);
        }
        self.mic_format = self.mic_format.filter(MicFormat::is_plausible);
        self.desktop_format = self.desktop_format.filter(MicFormat::is_plausible);
        self
    }
}

/// The latest recording start [`ReplayMeta::sanitized`] keeps: the end of the year 9999 (what
/// dates and file times can show).
pub const MAX_STARTED_UNIX_MS: u64 = 253_402_300_799_999;
/// The longest time [`ReplayMeta::sanitized`] keeps: 30 days, in microseconds.
pub const MAX_META_TIME: ReplayTime = 30 * 24 * 3600 * 1_000_000;
/// The longest text [`ReplayMeta::sanitized`] keeps, in characters.
pub const MAX_META_TEXT_CHARS: usize = 1024;
/// The most markers (and world resets) [`ReplayMeta::sanitized`] keeps.
pub const MAX_META_MARKERS: usize = 4096;
/// The most tags [`ReplayMeta::sanitized`] keeps.
pub const MAX_META_TAGS: usize = 256;

impl MicFormat {
    /// A format a recording can have: 8 to 192 kHz, 1 to 8 channels, 1 to 65536 samples a
    /// record. The client records 24 kHz mono in 480-sample records.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        (8_000..=192_000).contains(&self.sample_rate)
            && (1..=8).contains(&self.channels)
            && (1..=65_536).contains(&self.frame_samples)
    }
}

/// Cut `text` to at most `max` characters.
fn truncate_chars(text: &mut String, max: usize) {
    if let Some((cut, _)) = text.char_indices().nth(max) {
        text.truncate(cut);
    }
}
