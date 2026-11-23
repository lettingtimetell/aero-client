//! The replay recorder's client half: the packet capture, the
//! `replay` mod's behaviour (record every game, the start/pause/stop keys, the clip key and its
//! in-memory session buffer), the microphone and the recording dot.
//!
//! # What is recorded, and when
//!
//! * **Packets.** While the mod is on and a server session is open, the network task copies every
//!   clientbound Play packet's exact bytes (`net::NetHandle::set_raw_tap`) and they travel with
//!   the decoded packet. The driver calls [`Client::rec_after_packet`] right after
//!   `Session::poll_next` applied one, so each record is stamped with the frame in which the game
//!   thread applied it, in the order it did. Mod off: no copy, no hook work (one `bool`).
//! * **the records** (local player per tick, camera per frame, sounds) through
//!   [`RecState::push_client`], on the same clock, in push order.
//! * **The microphone** (option, off by default) while a recording runs.
//!
//! # Sinks
//!
//! * **The recording** (a file in `<gameDir>/replays/`). "Record every game" starts one at each
//!   world reset (a `JoinGame`, or a `Respawn` into another dimension: a server switch on a
//!   proxy) and splits there; the start key starts one mid-world from the session buffer.
//! * **The session buffer** (in memory, compressed, from the last world reset; kept only while
//!   the start or clip key is bound): what a clip, or a recording started mid-world, is cut from.
//!
//! A file or clip begins with a *state prefix*: the packets a world reset does not clear (the
//! last `JoinGame` with the player's entity id, the tab list, the scoreboard and teams, the tab
//! header and footer, the difficulty, the server brand), kept here in a [`StickyLog`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mc_mods::mods::replay::ids;
use mc_mods::ModSettings;
use mc_protocol::play::s2c;
use mc_protocol::Packet as _;
use mc_render::input::Input;
use mc_render::InputState;
use mc_replay::mic::{MicCapture, MicEncoder};
use mc_replay::{
    Finishing, Library, Marker, Record, ReplayMeta, ReplayTime, StreamId, Writer, WriterOptions,
};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::widget::DrawList;

use super::super::{Client, Ui};
use crate::session::Session;

// Desktop audio while recording.
mod desktop;
// The Bookmark key, date folders and lobby-hop merging.
mod nw_ra;

/// The mod's catalog id.
const MOD_ID: &str = "replay";

/// An auto-started recording is split at a world reset only once it is this old: a proxy's
/// server switch is two `Respawn`s in quick succession, and that must not leave a file of 100 ms.
/// (also only when the world being left lasted [`nw_ra::SPLIT_WORLD_MIN`].)
const SPLIT_AFTER: ReplayTime = 5_000_000;
/// The session buffer's size cap (compressed). A SkyWars game is a few MB; an hour in a busy
/// lobby about 20.
const BUFFER_MAX_BYTES: u64 = 64 << 20;
/// The sticky log's cap. Tab-list and scoreboard packets of a whole minigame session stay far
/// below it; past it the oldest go first.
const STICKY_MAX_BYTES: usize = 32 << 20;
/// How long quitting waits for recordings and clips to reach the disk.
const QUIT_WAIT: Duration = Duration::from_secs(5);
/// The microphone ring behind clips holds the longest clip there can be (Clip length's maximum).
const MIC_RING_SPAN: ReplayTime = 300_000_000;
/// Its frames (20 ms each), with a little room: about 3.9 MB, reserved when the mic opens.
const MIC_RING_FRAMES: usize = 15_100;

/// One microphone record in the ring: its push time on the session clock, and its bytes.
#[derive(Debug, Clone, Copy)]
struct MicSlot {
    time: ReplayTime,
    bytes: [u8; mc_replay::mic::RECORD_BYTES],
}

/// The keys, in [`RecOptions::keys`] order (the Bookmark key).
const KEY_START: usize = 0;
const KEY_PAUSE: usize = 1;
const KEY_STOP: usize = 2;
const KEY_CLIP: usize = 3;
const KEY_BOOKMARK: usize = 4;
const KEY_COUNT: usize = 5;

/// The writer settings every recording uses.
fn writer_options() -> WriterOptions {
    WriterOptions::default()
}

// ---- what a packet means to the recorder (protocol 47) ----------------------------------------

/// What the recorder makes of one clientbound packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PacketClass {
    /// `JoinGame`: a new world, and a new scoreboard, in `dimension`.
    Join {
        /// The new world's dimension.
        dimension: i32,
    },
    /// `Respawn` into `dimension`: a world reset when it is another dimension than the current
    /// one (vanilla `handleRespawn` only builds a new world then).
    Respawn {
        /// The dimension respawned into.
        dimension: i32,
    },
    /// State a world reset does not clear.
    Sticky(Sticky),
    /// Everything else.
    Plain,
}

/// The kinds of state a world reset does not clear (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sticky {
    /// `JoinGame` itself (the player's entity id, game mode, world type).
    Join,
    /// `PlayerListItem`: the tab list, with every player's skin.
    TabList,
    /// `PlayerListHeaderFooter`: only the last one counts.
    HeaderFooter,
    /// Objectives, scores, display slots and teams (nametag colours). Cleared by a `JoinGame`.
    Scoreboard,
    /// `ServerDifficulty`: only the last one counts.
    Difficulty,
    /// The `MC|Brand` plugin message (F3's server brand): only the last one counts.
    Brand,
}

/// Classify one packet from its bytes (packet id varint, then body).
pub(crate) fn classify(raw: &[u8]) -> PacketClass {
    let mut pos = 0;
    let Some(id) = varint(raw, &mut pos) else {
        return PacketClass::Plain;
    };
    match id {
        // Entity id (int), game mode (byte), dimension (signed byte).
        s2c::JoinGame::ID => raw
            .get(pos + 5)
            .map_or(PacketClass::Plain, |&d| PacketClass::Join {
                dimension: i32::from(d as i8),
            }),
        // Dimension (int) first.
        s2c::Respawn::ID => {
            raw.get(pos..pos + 4)
                .map_or(PacketClass::Plain, |b| PacketClass::Respawn {
                    dimension: i32::from_be_bytes([b[0], b[1], b[2], b[3]]),
                })
        }
        s2c::PlayerListItem::ID => PacketClass::Sticky(Sticky::TabList),
        s2c::PlayerListHeaderFooter::ID => PacketClass::Sticky(Sticky::HeaderFooter),
        s2c::ScoreboardObjective::ID
        | s2c::UpdateScore::ID
        | s2c::DisplayScoreboard::ID
        | s2c::Teams::ID => PacketClass::Sticky(Sticky::Scoreboard),
        s2c::ServerDifficulty::ID => PacketClass::Sticky(Sticky::Difficulty),
        s2c::PluginMessage::ID => {
            let channel = varint(raw, &mut pos)
                .and_then(|len| usize::try_from(len).ok())
                .and_then(|len| raw.get(pos..pos + len));
            if channel == Some(b"MC|Brand".as_slice()) {
                PacketClass::Sticky(Sticky::Brand)
            } else {
                PacketClass::Plain
            }
        }
        _ => PacketClass::Plain,
    }
}

/// A protocol varint (up to 5 bytes) at `*pos`.
fn varint(bytes: &[u8], pos: &mut usize) -> Option<i32> {
    let mut value: u32 = 0;
    for shift in 0..5 {
        let byte = *bytes.get(*pos)?;
        *pos += 1;
        value |= u32::from(byte & 0x7F) << (7 * shift);
        if byte & 0x80 == 0 {
            return Some(value as i32);
        }
    }
    None
}

/// The packets that survive world resets, as received, compacted where a later packet replaces
/// an earlier one. The state prefix of every file and clip comes from here.
#[derive(Debug, Default)]
pub(crate) struct StickyLog {
    entries: Vec<(Sticky, Box<[u8]>)>,
    bytes: usize,
}

impl StickyLog {
    /// A `JoinGame` is about to be applied: it makes a new world and a new scoreboard, so
    /// everything but the tab list, its header and footer and the server brand is superseded.
    fn compact_for_join(&mut self) {
        self.entries.retain(|(kind, _)| {
            matches!(kind, Sticky::TabList | Sticky::HeaderFooter | Sticky::Brand)
        });
        self.recount();
    }

    /// Keep a packet. Returns how many of the first `protect` entries were dropped to make room
    /// (the caller's index into the log moves back by that much).
    fn add(&mut self, kind: Sticky, raw: Box<[u8]>, protect: usize) -> usize {
        let mut dropped_before = 0;
        if matches!(
            kind,
            Sticky::HeaderFooter | Sticky::Difficulty | Sticky::Brand | Sticky::Join
        ) {
            // Only the last one counts.
            let mut index = 0;
            self.entries.retain(|(k, _)| {
                let keep = *k != kind;
                if !keep && index < protect {
                    dropped_before += 1;
                }
                index += 1;
                keep
            });
        }
        self.bytes += raw.len();
        self.entries.push((kind, raw));
        if self.bytes > STICKY_MAX_BYTES {
            dropped_before += self.trim(protect.saturating_sub(dropped_before));
        }
        self.recount();
        dropped_before
    }

    /// Over the cap: drop the oldest tab-list and scoreboard packets down to three quarters of
    /// it (degrades gracefully: the oldest players' tab entries go first). Returns how many of the
    /// first `protect` entries went.
    fn trim(&mut self, protect: usize) -> usize {
        tracing::warn!(
            bytes = self.bytes,
            "replay state log over its cap; dropping its oldest entries"
        );
        let target = STICKY_MAX_BYTES / 4 * 3;
        let mut bytes = self.bytes;
        let mut index = 0;
        let mut dropped_before = 0;
        self.entries.retain(|(kind, raw)| {
            let drop = bytes > target && matches!(kind, Sticky::TabList | Sticky::Scoreboard);
            if drop {
                bytes -= raw.len();
                if index < protect {
                    dropped_before += 1;
                }
            }
            index += 1;
            !drop
        });
        dropped_before
    }

    fn recount(&mut self) {
        self.bytes = self.entries.iter().map(|(_, raw)| raw.len()).sum();
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    /// The first `count` entries as a state prefix (time 0).
    fn prefix(&self, count: usize) -> Vec<Record> {
        self.entries[..count.min(self.entries.len())]
            .iter()
            .map(|(_, raw)| Record {
                time: 0,
                stream: StreamId::Packet,
                payload: raw.to_vec(),
            })
            .collect()
    }
}

// ---- options ------------------------------------------------------------------------------------

/// The mod's settings, read when the settings generation moves.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RecOptions {
    auto_start: bool,
    keys: [Input; KEY_COUNT],
    indicator: bool,
    mic: bool,
    clip_length: ReplayTime,
    max_storage_bytes: u64,
}

impl Default for RecOptions {
    fn default() -> RecOptions {
        RecOptions {
            auto_start: true,
            keys: [Input::Unbound; KEY_COUNT],
            indicator: true,
            mic: false,
            clip_length: 30_000_000,
            max_storage_bytes: 20_000_000_000,
        }
    }
}

impl RecOptions {
    fn read(settings: &ModSettings) -> RecOptions {
        RecOptions {
            auto_start: settings.bool(MOD_ID, ids::AUTO_START),
            keys: [
                ids::KEY_START,
                ids::KEY_PAUSE,
                ids::KEY_STOP,
                ids::KEY_CLIP,
                ids::KEY_BOOKMARK,
            ]
            .map(|id| Input::from_lwjgl(settings.key(MOD_ID, id))),
            indicator: settings.bool(MOD_ID, ids::INDICATOR),
            mic: settings.bool(MOD_ID, ids::MIC),
            clip_length: (f64::from(settings.f32(MOD_ID, ids::CLIP_LENGTH)).clamp(5.0, 300.0) * 1e6)
                as ReplayTime,
            max_storage_bytes: (f64::from(settings.f32(MOD_ID, ids::MAX_STORAGE_GB))
                .clamp(1.0, 200.0)
                * 1e9) as u64,
        }
    }

    /// The session buffer is kept while something may ask for it: a clip, or a recording
    /// started mid-world.
    fn wants_buffer(&self) -> bool {
        self.keys[KEY_CLIP] != Input::Unbound || self.keys[KEY_START] != Input::Unbound
    }

    /// Whether anything can ever record with these settings: "Record every game", or a start or
    /// clip key. Without one, the recorder does not even tap the session.
    fn can_record(&self) -> bool {
        self.auto_start || self.wants_buffer()
    }
}

// ---- state ----------------------------------------------------------------------------------

/// One recording to a file.
#[derive(Debug)]
struct Recording {
    writer: Writer,
    path: PathBuf,
    /// The metadata the header got (title, server, player, start).
    meta: ReplayMeta,
    /// The session time that is 0 in the file.
    base: ReplayTime,
    /// Paused time cut out so far.
    paused_total: ReplayTime,
    /// The session time the current pause began.
    paused_at: Option<ReplayTime>,
    /// Started by "Record every game" (split at world resets).
    auto: bool,
    /// The session time it started.
    started_at: ReplayTime,
    world_resets: Vec<ReplayTime>,
    markers: Vec<Marker>,
    has_mic: bool,
    /// Wall-clock milliseconds when its visible part began: the time in its default title.
    title_unix_ms: u64,
}

impl Recording {
    /// The file time of session time `now` (frozen while paused).
    fn time(&self, now: ReplayTime) -> ReplayTime {
        self.paused_at
            .unwrap_or(now)
            .saturating_sub(self.base)
            .saturating_sub(self.paused_total)
    }

    /// The final metadata, at session time `now`. `game` is the game seen on the sidebar while
    /// recording; it goes into the default title ([`default_title`]).
    fn final_meta(
        &self,
        now: ReplayTime,
        tags: &BTreeMap<String, String>,
        game: Option<&GameTitle>,
    ) -> ReplayMeta {
        let mut meta = self.meta.clone();
        meta.title = default_title(game, &meta.server, self.title_unix_ms);
        meta.duration = self.time(now);
        meta.world_resets = self.world_resets.clone();
        meta.markers = self.markers.clone();
        meta.has_mic = self.has_mic;
        meta.mic_format = self.has_mic.then(mc_replay::mic::format);
        meta.tags
            .extend(tags.iter().map(|(k, v)| (k.clone(), v.clone())));
        meta
    }
}

/// The session buffer.
#[derive(Debug)]
struct Buffer {
    writer: Writer,
    /// The session time of its last world reset (its time 0 once cut).
    base: Option<ReplayTime>,
    /// Sticky-log entries from before that reset: a clip's state prefix.
    sticky_len: usize,
}

impl Buffer {
    /// `(base, sticky_len)` if it holds a world a clip or a recording can start from: one has
    /// been reset since it started, and it has not overflowed since. Read on this thread: the
    /// worker stores the reset's block before it answers any later clip or snapshot request.
    fn world(&self) -> Option<(ReplayTime, usize)> {
        let base = self.base?;
        (!self.writer.stats().overflowed).then_some((base, self.sticky_len))
    }
}

/// The microphone while a recording runs.
#[derive(Debug)]
struct Mic {
    capture: MicCapture,
    encoder: MicEncoder,
    pending: Vec<i16>,
    /// Samples received since the timeline began.
    received: u64,
    /// File time of the timeline's first sample (smoothed), µs.
    anchor: Option<f64>,
    /// The lowest anchor estimate this window (the delivery with the least delay).
    window_min: f64,
    window_started: Option<Instant>,
    /// This frame's records, back to back ([`mc_replay::mic::RECORD_BYTES`] each).
    records: Vec<u8>,
    warned: bool,
}

impl Mic {
    fn start() -> Mic {
        tracing::info!("replay microphone on");
        Mic::with_capture(MicCapture::start())
    }

    fn with_capture(capture: MicCapture) -> Mic {
        Mic {
            capture,
            encoder: MicEncoder::default(),
            pending: Vec::new(),
            received: 0,
            anchor: None,
            window_min: f64::MAX,
            window_started: None,
            records: Vec::new(),
            warned: false,
        }
    }

    /// Take what the device delivered, encode whole frames, and leave this frame's records in
    /// `self.records` (the gate drops silence). They are pushed at `file_now`; a record's first
    /// four bytes say how long before that its first sample was captured
    /// ([`mc_replay::mic::decode_record`]). No allocation once warm.
    fn frame(&mut self, file_now: ReplayTime, now: Instant) {
        self.records.clear();
        if let Some(error) = self.capture.error() {
            if !self.warned {
                tracing::warn!(%error, "replay microphone: nothing will be recorded");
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
        let rate = f64::from(mc_replay::mic::SAMPLE_RATE);
        // When sample 0 was captured, if the newest one was captured right now.
        let estimate = file_now as f64 - self.received as f64 * 1e6 / rate;
        let anchor = *self.anchor.get_or_insert(estimate);
        self.window_min = self.window_min.min(estimate);
        let started = *self.window_started.get_or_insert(now);
        if now.saturating_duration_since(started) >= Duration::from_secs(1) {
            // Drift and delivery jitter: move toward the least-delayed estimate, 2 ms a second
            // at most, so consecutive frames stay exactly 20 ms apart in the file.
            let anchor = anchor + (self.window_min - anchor).clamp(-2000.0, 2000.0);
            self.anchor = Some(anchor);
            self.window_min = f64::MAX;
            self.window_started = Some(now);
        }
        let anchor = self.anchor.unwrap_or(estimate);
        let samples = mc_replay::mic::FRAME_SAMPLES;
        let mut consumed = 0;
        while self.pending.len() - consumed >= samples {
            let frame = &self.pending[consumed..consumed + samples];
            let records = &mut self.records;
            self.encoder.push(frame, |index, bytes| {
                let at = anchor + index as f64 * samples as f64 * 1e6 / rate;
                let lead = (file_now as f64 - at).clamp(0.0, f64::from(u32::MAX)) as u32;
                records.extend_from_slice(&lead.to_le_bytes());
                records.extend_from_slice(bytes);
            });
            consumed += samples;
        }
        self.pending.drain(..consumed);
    }
}

/// One server session the recorder follows.
#[derive(Debug)]
struct Conn {
    /// The session clock's origin (a frame instant).
    started: Instant,
    /// This frame's session time.
    now: ReplayTime,
    server: String,
    player_name: String,
    player_uuid: String,
    /// The current world's dimension (from `JoinGame` / `Respawn`).
    dimension: Option<i32>,
    sticky: StickyLog,
    buffer: Option<Buffer>,
    recording: Option<Recording>,
    mic: Option<Mic>,
    /// The last 300 s of microphone records while the clip key's buffer runs: clips take
    /// theirs from here (the buffer itself keeps whole worlds; an hour of speech would fill it).
    mic_ring: std::collections::VecDeque<MicSlot>,
    /// Desktop audio while something records (`desktop.rs`).
    desktop: Option<desktop::Desktop>,
    /// The start key was pressed while no world could be rebuilt: start at the next reset.
    pending_start: bool,
    /// The game being played, from the sidebar ([`game_title`]); titles files and clips.
    game_title: Option<GameTitle>,
    /// The scoreboard revision `game_title` was read at, and when it was last checked.
    board_seen: Option<u64>,
    title_checked: Option<Instant>,
    /// The session time of the last world reset (how long the world lasted
    /// decides a split), and the Bookmark key's presses kept for clips (`nw_ra.rs`).
    world_started: ReplayTime,
    bookmarks: Vec<ReplayTime>,
}

impl Conn {
    /// Push one record into every sink at this frame's time.
    fn push(&mut self, stream: StreamId, payload: &[u8]) {
        if let Some(recording) = &mut self.recording {
            let time = recording.time(self.now);
            recording.writer.push(time, stream, payload);
        }
        if let Some(buffer) = &mut self.buffer {
            buffer.writer.push(self.now, stream, payload);
        }
    }

    /// Wall-clock milliseconds at session time `t`.
    fn unix_ms_at(&self, t: ReplayTime) -> u64 {
        unix_ms_now().saturating_sub(self.now.saturating_sub(t) / 1000)
    }
}

/// A file being finished, or a clip being written, on a writer thread.
#[derive(Debug)]
enum Job {
    Recording(Finishing),
    Clip(Finishing),
}

/// A message for the player (a toast in game).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Notice {
    pub title: String,
    pub detail: String,
}

/// The field on [`Client`] (`rec`).
#[derive(Debug, Default)]
pub(crate) struct RecState {
    seen: Option<u64>,
    enabled: bool,
    opts: RecOptions,
    keys_down: [bool; KEY_COUNT],
    keys_armed: bool,
    /// A sink is taking records: the one test the capture makes.
    capturing: bool,
    /// The tap is on for the current session: the one test [`Client::rec_after_packet`] makes.
    tapping: bool,
    generation: u64,
    blocked: bool,
    tags: BTreeMap<String, String>,
    game_dir: PathBuf,
    /// Paths recently handed to writers (see [`fresh_path`]).
    issued: Vec<PathBuf>,
    conn: Option<Box<Conn>>,
    jobs: Vec<Job>,
    notices: Vec<Notice>,
    repaired: bool,
    /// "Replay saved: <title>" and when, for a recording that finished after its session ended:
    /// shown for a few seconds on whatever screen follows (the in-world toast cannot draw there).
    title_notice: Option<(String, Instant)>,
    /// Tests: a fake microphone opened instead of the default device.
    #[cfg(test)]
    test_mic: Option<MicCapture>,
    /// The desktop audio options (`desktop.rs`).
    desktop_opts: desktop::DesktopOptions,
    /// Tests: a fake desktop audio capture opened instead of the real one.
    #[cfg(test)]
    test_desktop: Option<mc_replay::desktop::DesktopCapture>,
    /// What the last Bookmark key press did, and when: its confirmation.
    bookmark_flash: Option<(nw_ra::Flash, Instant)>,
}

impl RecState {
    // ---- the recording hook ----------------------------------------------------

    /// Whether anything takes records right now (a recording, or the session buffer).
    #[inline]
    pub(crate) fn capturing(&self) -> bool {
        self.capturing
    }

    /// Push one of the records (`StreamId::Client(kind)`), stamped with this frame's
    /// time, into every active sink. Keeps push order with the packet records.
    #[inline]
    pub(crate) fn push_client(&mut self, kind: u8, payload: &[u8]) {
        if let Some(conn) = self.conn.as_deref_mut() {
            conn.push(StreamId::Client(kind), payload);
        }
    }

    /// Whether the recording is paused (its time frozen; packets are still recorded).
    pub(crate) fn paused(&self) -> bool {
        self.conn
            .as_deref()
            .and_then(|conn| conn.recording.as_ref())
            .is_some_and(|recording| recording.paused_at.is_some())
    }

    /// Bumps whenever a sink starts (a recording, or the session buffer at a world reset):
    /// write a full-state record then.
    pub(crate) fn capture_generation(&self) -> u64 {
        self.generation
    }

    /// A tag for the metadata of files finished from now on (`"rb.version"`).
    pub(crate) fn set_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_owned(), value.to_owned());
    }

    /// While a replay is being watched nothing is recorded.
    pub(crate) fn block_for_playback(&mut self, blocked: bool) {
        self.blocked = blocked;
    }

    /// Whether a recording to a file is running.
    pub(crate) fn recording(&self) -> bool {
        self.conn
            .as_deref()
            .is_some_and(|conn| conn.recording.is_some())
    }

    /// This frame's time in the running recording (part of the hook API for playback and export).
    #[allow(dead_code)]
    pub(crate) fn time(&self) -> Option<ReplayTime> {
        let conn = self.conn.as_deref()?;
        conn.recording.as_ref().map(|r| r.time(conn.now))
    }

    // ---- inside -------------------------------------------------------------------------

    fn sync(&mut self, settings: &ModSettings) {
        let generation = settings.generation();
        if self.seen == Some(generation) {
            return;
        }
        self.seen = Some(generation);
        self.enabled = settings.enabled(MOD_ID);
        self.opts = RecOptions::read(settings);
        self.desktop_opts = desktop::DesktopOptions::read(settings);
    }

    fn library(&self) -> Library {
        Library::new(&self.game_dir)
    }

    /// Open the microphone (the default device; a fake one in tests).
    fn open_mic(&mut self) -> Mic {
        #[cfg(test)]
        if let Some(capture) = self.test_mic.take() {
            return Mic::with_capture(capture);
        }
        Mic::start()
    }

    fn notice(&mut self, title: &str, detail: impl Into<String>) {
        self.notices.push(Notice {
            title: title.to_owned(),
            detail: detail.into(),
        });
    }

    /// One packet the game thread just applied.
    fn on_packet(&mut self, raw: Box<[u8]>) {
        let Some(conn) = self.conn.as_deref_mut() else {
            return;
        };
        let class = classify(&raw);
        let reset = match class {
            PacketClass::Join { dimension } => {
                conn.dimension = Some(dimension);
                Some(true)
            }
            PacketClass::Respawn { dimension } => {
                let changed = conn.dimension != Some(dimension);
                conn.dimension = Some(dimension);
                changed.then_some(false)
            }
            PacketClass::Sticky(_) | PacketClass::Plain => None,
        };
        if let Some(join) = reset {
            self.world_reset(join);
        }
        let Some(conn) = self.conn.as_deref_mut() else {
            return;
        };
        conn.push(StreamId::Packet, &raw);
        let sticky = match class {
            PacketClass::Join { .. } => Some(Sticky::Join),
            PacketClass::Sticky(kind) => Some(kind),
            PacketClass::Respawn { .. } | PacketClass::Plain => None,
        };
        if let Some(kind) = sticky {
            let protect = conn.buffer.as_ref().map_or(0, |b| b.sticky_len);
            let dropped = conn.sticky.add(kind, raw, protect);
            if let Some(buffer) = conn.buffer.as_mut() {
                buffer.sticky_len = buffer.sticky_len.saturating_sub(dropped);
            }
        }
    }

    /// A world reset is about to be pushed: split or keyframe the recording, start one if
    /// "Record every game" (or a pending start) asks, re-arm the buffer.
    fn world_reset(&mut self, join: bool) {
        let opts = self.opts;
        let Some(conn) = self.conn.as_deref_mut() else {
            return;
        };
        let now = conn.now;
        if join {
            conn.sticky.compact_for_join();
        }
        let mut start = conn.pending_start && !self.blocked;
        // A world left after less than SPLIT_WORLD_MIN (a lobby hop) stays in
        // the file of the world that follows.
        let world_age = now.saturating_sub(conn.world_started);
        let split = conn.recording.as_ref().map(|recording| {
            recording.auto
                && now.saturating_sub(recording.started_at) >= SPLIT_AFTER
                && world_age >= nw_ra::SPLIT_WORLD_MIN
        });
        match split {
            // "Record every game": this game's file ends here, the next one starts below.
            Some(true) => {
                if let Some(recording) = conn.recording.take() {
                    conn.mic = None;
                    let meta = recording.final_meta(now, &self.tags, conn.game_title.as_ref());
                    self.jobs.push(Job::Recording(finish(recording, meta)));
                }
                start |= opts.auto_start;
            }
            // A manual recording (or an auto one a moment old) goes on into the new world.
            Some(false) => {
                if let Some(recording) = conn.recording.as_mut() {
                    let time = recording.time(now);
                    recording.writer.keyframe();
                    recording.world_resets.push(time);
                }
            }
            None => start |= opts.auto_start && !self.blocked,
        }
        // A new world, perhaps a new game: its title is read afresh (the file that just ended
        // took the old one above).
        conn.game_title = None;
        conn.board_seen = None;
        conn.world_started = now;
        if start && conn.recording.is_none() {
            conn.pending_start = false;
            let auto = opts.auto_start;
            let title_unix_ms = conn.unix_ms_at(now);
            let meta = new_meta(conn, now, title_unix_ms);
            let stem = file_stem(title_unix_ms, &conn.server);
            let path = fresh_path(
                &self.game_dir,
                &mut self.issued,
                false,
                &stem,
                title_unix_ms,
            );
            match Writer::create(&path, &meta, writer_options()) {
                Ok(mut writer) => {
                    // The state prefix, straight from the log (a memcpy each, no copies).
                    if !conn.sticky.entries.is_empty() {
                        writer.set_prefix_mode(true);
                        for (_, raw) in &conn.sticky.entries {
                            writer.push(0, StreamId::Packet, raw);
                        }
                        writer.set_prefix_mode(false);
                    }
                    writer.keyframe();
                    tracing::info!(file = %nw_ra::log_name(&path), auto, "replay recording started");
                    conn.recording = Some(Recording {
                        writer,
                        path,
                        meta,
                        base: now,
                        paused_total: 0,
                        paused_at: None,
                        auto,
                        started_at: now,
                        world_resets: vec![0],
                        markers: Vec::new(),
                        has_mic: false,
                        title_unix_ms,
                    });
                    self.generation += 1;
                }
                Err(error) => {
                    tracing::warn!(%error, "replay recording could not start");
                }
            }
        }
        if let Some(buffer) = conn.buffer.as_mut() {
            buffer.writer.keyframe();
            buffer.base = Some(now);
            buffer.sticky_len = conn.sticky.len();
            self.generation += 1;
        }
        self.refresh_capturing();
    }

    /// Recompute [`RecState::capturing`] right after a sink starts or stops, so the records of the same frame (the reset frame's ticks and camera) are taken too.
    fn refresh_capturing(&mut self) {
        self.capturing = self
            .conn
            .as_deref()
            .is_some_and(|conn| conn.recording.is_some() || conn.buffer.is_some());
    }

    /// The start key: resume a paused recording, or start one from the session buffer.
    fn start_key(&mut self) {
        if self.blocked {
            return;
        }
        let Some(conn) = self.conn.as_deref_mut() else {
            return;
        };
        if let Some(recording) = conn.recording.as_mut() {
            if let Some(at) = recording.paused_at.take() {
                recording.paused_total += conn.now.saturating_sub(at);
                self.notices.push(Notice {
                    title: "Recording resumed".into(),
                    detail: String::new(),
                });
            }
            return;
        }
        let now = conn.now;
        let ready = conn.buffer.as_ref().and_then(Buffer::world);
        let Some((base, sticky_len)) = ready else {
            conn.pending_start = true;
            self.notices.push(Notice {
                title: "Recording starts soon".into(),
                detail: "When you join the next world".into(),
            });
            return;
        };
        // The title's time is the key press, where the part to watch begins.
        let title_unix_ms = conn.unix_ms_at(now);
        let mut meta = new_meta(conn, base, title_unix_ms);
        meta.view_start = now - base;
        let stem = file_stem(title_unix_ms, &conn.server);
        let path = fresh_path(
            &self.game_dir,
            &mut self.issued,
            false,
            &stem,
            title_unix_ms,
        );
        let prefix = conn.sticky.prefix(sticky_len);
        let Some(buffer) = conn.buffer.as_mut() else {
            return;
        };
        let snapshot = buffer.writer.snapshot();
        match Writer::create_with_prelude(&path, &meta, writer_options(), prefix, snapshot) {
            Ok(writer) => {
                tracing::info!(file = %nw_ra::log_name(&path), "replay recording started mid-world");
                conn.recording = Some(Recording {
                    writer,
                    path,
                    meta,
                    base,
                    paused_total: 0,
                    paused_at: None,
                    auto: false,
                    started_at: now,
                    world_resets: vec![0],
                    markers: Vec::new(),
                    has_mic: false,
                    title_unix_ms,
                });
                self.generation += 1;
                self.notices.push(Notice {
                    title: "Recording".into(),
                    detail: String::new(),
                });
                self.refresh_capturing();
            }
            Err(error) => tracing::warn!(%error, "replay recording could not start"),
        }
    }

    /// The pause key: pause or resume the recording.
    fn pause_key(&mut self) {
        let Some(conn) = self.conn.as_deref_mut() else {
            return;
        };
        let now = conn.now;
        let Some(recording) = conn.recording.as_mut() else {
            return;
        };
        let title = match recording.paused_at.take() {
            Some(at) => {
                recording.paused_total += now.saturating_sub(at);
                "Recording resumed"
            }
            None => {
                recording.paused_at = Some(now);
                "Recording paused"
            }
        };
        self.notice(title, String::new());
    }

    /// The stop key: finish and save the recording.
    fn stop_key(&mut self) {
        let Some(conn) = self.conn.as_deref_mut() else {
            return;
        };
        conn.pending_start = false;
        if let Some(recording) = conn.recording.take() {
            conn.mic = None;
            let meta = recording.final_meta(conn.now, &self.tags, conn.game_title.as_ref());
            let seconds = meta.duration / 1_000_000;
            self.jobs.push(Job::Recording(finish(recording, meta)));
            self.notice(
                "Replay saved",
                format!("{}:{:02} recorded", seconds / 60, seconds % 60),
            );
            self.refresh_capturing();
        }
    }

    /// The clip key: the last `clip_length` seconds, from the session buffer.
    fn clip_key(&mut self) {
        if self.blocked {
            return;
        }
        let clip_length = self.opts.clip_length;
        let Some(conn) = self.conn.as_deref_mut() else {
            return;
        };
        let now = conn.now;
        let ready = conn.buffer.as_ref().and_then(Buffer::world);
        let Some((base, sticky_len)) = ready else {
            self.notices.push(Notice {
                title: "No clip yet".into(),
                detail: "Clips start when you join the next world".into(),
            });
            return;
        };
        // Titled like a recording, with the time of the key press (clips have their own tab).
        let title_unix_ms = conn.unix_ms_at(now);
        let mut meta = new_meta(conn, base, title_unix_ms);
        let prefix = conn.sticky.prefix(sticky_len);
        let stem = file_stem(title_unix_ms, &conn.server);
        let path = fresh_path(&self.game_dir, &mut self.issued, true, &stem, title_unix_ms);
        let window = (now.saturating_sub(clip_length), now);
        // The bookmarks set inside the window come along (the writer retimes).
        meta.markers = nw_ra::clip_bookmarks(conn, window);
        // The microphone's frames for the window, from its ring (merged in by the writer). A
        // frame pushed a moment after the window starts can begin just before it: take those.
        let mut mic: Vec<Record> = conn
            .mic_ring
            .iter()
            .filter(|slot| slot.time >= window.0 && slot.time <= window.1)
            .map(|slot| Record {
                time: slot.time,
                stream: StreamId::Mic,
                payload: slot.bytes.to_vec(),
            })
            .collect();
        if !mic.is_empty() {
            meta.has_mic = true;
            meta.mic_format = Some(mc_replay::mic::format());
        }
        // Desktop audio's frames for the window too.
        desktop::add_to_clip(conn, window, &mut mic, &mut meta);
        // A marker in the running recording too, so the moment is easy to find there.
        if let Some(recording) = conn.recording.as_mut() {
            let time = recording.time(now);
            recording.markers.push(Marker {
                time,
                label: "Clip".into(),
            });
            recording.writer.push(time, StreamId::Marker, b"Clip");
        }
        let Some(buffer) = conn.buffer.as_mut() else {
            return;
        };
        tracing::info!(file = %nw_ra::log_name(&path), mic_frames = mic.len(), "saving a clip");
        self.jobs.push(Job::Clip(
            buffer.writer.save_clip(path, meta, prefix, window, mic),
        ));
    }
}

/// Finish a recording on its writer thread. Once it is on disk, [`Client::rec_poll_jobs`]
/// applies the storage cap (on a helper thread: listing a folder on a slow disk must not stall a
/// frame).
fn finish(recording: Recording, meta: ReplayMeta) -> Finishing {
    tracing::info!(
        file = %nw_ra::log_name(&recording.path),
        seconds = meta.duration / 1_000_000,
        "replay recording stopped"
    );
    recording.writer.finish(meta)
}

/// The metadata a new file starts with. `zero` is the session time that becomes 0 in it;
/// `title_unix_ms` the wall-clock time its default title shows ([`default_title`]).
fn new_meta(conn: &Conn, zero: ReplayTime, title_unix_ms: u64) -> ReplayMeta {
    ReplayMeta {
        title: default_title(conn.game_title.as_ref(), &conn.server, title_unix_ms),
        server: conn.server.clone(),
        started_unix_ms: conn.unix_ms_at(zero),
        player_name: conn.player_name.clone(),
        player_uuid: conn.player_uuid.clone(),
        protocol: 47,
        client_version: format!("Aero Client {}", env!("CARGO_PKG_VERSION")),
        world_resets: vec![0],
        ..ReplayMeta::default()
    }
}

/// The game being played, read off a sidebar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GameTitle {
    /// "SkyWars", "Bed Wars", "Duels"...
    game: String,
    /// The sidebar's "Map:" line, if it has one.
    map: Option<String>,
}

/// A default title that tells recordings apart: the
/// game and its map when a sidebar shows them, otherwise the server, and when.
/// "SkyWars · Dwarf Fortress · Nov 12, 9:41 PM", "SkyWars · mc.example.net · Nov 12, 9:41 PM",
/// "mc.example.net · Nov 12, 9:41 PM". The server is in the metadata either way.
fn default_title(game: Option<&GameTitle>, server: &str, unix_ms: u64) -> String {
    let when = mc_replay::local_title_time(unix_ms);
    match game {
        Some(GameTitle {
            game,
            map: Some(map),
        }) => format!("{game} · {map} · {when}"),
        Some(GameTitle { game, map: None }) => format!("{game} · {server} · {when}"),
        None => format!("{server} · {when}"),
    }
}

/// The game being played, read off a minigame sidebar: its title ("SKYWARS", "BED WARS",
/// "DUELS"...) and its "Map:" line. `None` without a sidebar.
fn game_title(board: &crate::sync::scoreboard::Scoreboard, username: &str) -> Option<GameTitle> {
    // The visible sidebar: the player's team-colored one when there is one, else the plain one.
    let objective = board
        .player_team(username)
        .filter(|team| team.color >= 0)
        .and_then(|team| board.objective_in_display_slot(3 + team.color as usize))
        .or_else(|| board.objective_in_display_slot(crate::sync::scoreboard::SLOT_SIDEBAR))?;
    let title = plain_text(&objective.display_name);
    if title.is_empty() {
        return None;
    }
    let game = match title.to_ascii_uppercase().as_str() {
        "SKYWARS" | "SKY WARS" => "SkyWars".to_owned(),
        "BED WARS" | "BEDWARS" => "Bed Wars".to_owned(),
        "UHC CHAMPIONS" => "UHC Champions".to_owned(),
        "THE BRIDGE" => "The Bridge".to_owned(),
        // Anything else: its words with a capital first letter ("MURDER MYSTERY").
        _ => title
            .split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                chars.next().map_or_else(String::new, |first| {
                    first
                        .to_uppercase()
                        .chain(chars.flat_map(char::to_lowercase))
                        .collect()
                })
            })
            .collect::<Vec<_>>()
            .join(" "),
    };
    let map = board
        .sorted_scores(&objective.name)
        .into_iter()
        .filter(|(name, _)| !name.starts_with('#'))
        .map(|(name, _)| plain_text(&board.format_player_name(name)))
        .find_map(|line| {
            line.get(..4)
                .filter(|label| label.eq_ignore_ascii_case("map:"))
                .map(|_| line[4..].trim().to_owned())
        })
        .filter(|map| !map.is_empty());
    Some(GameTitle { game, map })
}

/// Text without `§` formatting codes or control characters, trimmed.
fn plain_text(text: &str) -> String {
    let mut chars = text.chars();
    let mut out = String::with_capacity(text.len());
    while let Some(ch) = chars.next() {
        if ch == '\u{00a7}' {
            let _ = chars.next();
        } else if !ch.is_control() {
            out.push(ch);
        }
    }
    out.trim().to_owned()
}

/// `2026-11-12 21.41.05 mc.example.net`.
fn file_stem(unix_ms: u64, server: &str) -> String {
    format!("{} {server}", mc_replay::local_stamp(unix_ms))
}

/// A new file path for `stem`, unused on disk **and** by the files this client handed to a
/// writer recently (the writers create them on their own threads, so a second clip in the same
/// second must not get the first one's name).
///
/// In the date folder of `unix_ms` (`replays\2026\11 November\13\`).
fn fresh_path(
    game_dir: &Path,
    issued: &mut Vec<PathBuf>,
    clip: bool,
    stem: &str,
    unix_ms: u64,
) -> PathBuf {
    let library = Library::new(game_dir);
    let mut path = library.unique_path_at(clip, stem, unix_ms);
    let mut n = 2;
    while issued.contains(&path) {
        path = library.unique_path_at(clip, &format!("{stem} ({n})"), unix_ms);
        n += 1;
    }
    if issued.len() >= 16 {
        issued.remove(0);
    }
    issued.push(path.clone());
    path
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

// ---- the client hooks -------------------------------------------------------------------------

impl Client {
    /// Start of the frame, before packets are applied. Follows the
    /// session (tap on/off, buffer, the recording ending with it), sets this frame's time,
    /// drives the writers, the microphone and finished jobs. Mod off and nothing pending: a
    /// generation compare and two tests.
    pub(crate) fn rec_begin_frame(&mut self) {
        let generation = self.mods.settings.generation();
        if self.rec.seen != Some(generation) {
            self.rec.sync(&self.mods.settings);
        }
        if !self.rec.enabled && self.rec.conn.is_none() && self.rec.jobs.is_empty() {
            return;
        }
        self.rec_frame(Instant::now());
    }

    fn rec_frame(&mut self, now: Instant) {
        if self.rec.game_dir.as_os_str().is_empty() {
            self.rec.game_dir = self.config.game_dir.clone();
        }
        if self.rec.enabled && !self.rec.repaired {
            // Files a crash left without a footer get one (on a helper thread).
            self.rec.repaired = true;
            let library = self.rec.library();
            let _ = std::thread::Builder::new()
                .name("mc-rs replay repair".into())
                .spawn(move || {
                    // Best effort (the next start tries again): a panic on a
                    // damaged or shared file is logged, not the game's crash.
                    let _ = mc_core::panic_scope::contain(|| {
                        library.repair_unfinished();
                        // Old flat files into their date folders (safe moves).
                        library.organize();
                    });
                });
        }
        self.rec_follow_session(now);
        let max_storage_bytes = self.rec.opts.max_storage_bytes;
        let mut full = false;
        if let Some(conn) = self.rec.conn.as_deref_mut() {
            conn.now = now.saturating_duration_since(conn.started).as_micros() as ReplayTime;
            if conn.player_name.is_empty() {
                if let Some(session) = self.session.as_ref() {
                    let local = &session.state().local;
                    conn.player_name.clone_from(&local.username);
                    conn.player_uuid = local.uuid.hyphenated().to_string();
                }
            }
            // The game's title off the sidebar while something records: read when the
            // scoreboard changed, at most once a second.
            let due = conn
                .title_checked
                .is_none_or(|at| now.saturating_duration_since(at) >= Duration::from_secs(1));
            if due && (conn.recording.is_some() || conn.buffer.is_some()) {
                conn.title_checked = Some(now);
                if let Some(session) = self.session.as_ref() {
                    let state = session.state();
                    let revision = state.scoreboard.revision();
                    if conn.board_seen != Some(revision) {
                        conn.board_seen = Some(revision);
                        if let Some(title) = game_title(&state.scoreboard, &state.local.username) {
                            conn.game_title = Some(title);
                        }
                    }
                }
            }
            // The buffer follows the keys.
            if self.rec.opts.wants_buffer() != conn.buffer.is_some() {
                conn.buffer = self.rec.opts.wants_buffer().then(|| Buffer {
                    writer: Writer::memory(writer_options(), BUFFER_MAX_BYTES),
                    base: None,
                    sticky_len: 0,
                });
            }
            if let Some(recording) = conn.recording.as_mut() {
                recording.writer.tick(now);
                if recording.writer.failed() {
                    let error = recording.writer.error().unwrap_or_default();
                    tracing::warn!(%error, "replay recording failed");
                    conn.recording = None;
                    conn.mic = None;
                    self.rec.notices.push(Notice {
                        title: "Recording failed".into(),
                        detail: error,
                    });
                }
            }
            // A server can send data as fast as the connection carries
            // it, and a recording keeps every packet (unknown plugin channels too, which show
            // nothing): one that reaches the replay storage limit is saved and stopped, so the
            // file cannot grow until the disk is full. Real recordings are a few MB per game.
            full = conn.recording.as_ref().is_some_and(|recording| {
                recording.writer.stats().stored_bytes >= max_storage_bytes
            });
            if let Some(buffer) = conn.buffer.as_mut() {
                buffer.writer.tick(now);
            }
            self.rec_mic(now);
            self.rec_desktop(now);
        }
        if full {
            tracing::warn!(
                "a replay recording reached the replay storage limit: saved and stopped"
            );
            self.rec.stop_key();
            self.rec
                .notice("Recording stopped", "It reached the replay storage limit");
        }
        self.rec.capturing = self
            .rec
            .conn
            .as_deref()
            .is_some_and(|conn| conn.recording.is_some() || conn.buffer.is_some());
        self.rec_poll_jobs();
        self.rec_show_notices();
    }

    /// Open, keep or close the recorder's view of the current session.
    fn rec_follow_session(&mut self, now: Instant) {
        let wanted = self.rec.enabled
            && !self.rec.blocked
            && (self.rec.opts.can_record() || self.rec.recording())
            && self.session.as_ref().is_some_and(|session| {
                !session.is_driven()
                    && !matches!(session.phase(), crate::session::Phase::Disconnected(_))
            });
        if self.rec.conn.is_some() {
            // A session we tapped that no longer reports the tap is another session.
            let same = self.session.as_ref().is_some_and(Session::raw_tap);
            if wanted && same {
                return;
            }
            self.rec_end_session();
        }
        if !wanted {
            return;
        }
        let Some(session) = self.session.as_ref() else {
            return;
        };
        session.set_raw_tap(true);
        self.rec.tapping = true;
        let server = self
            .current_server_address
            .clone()
            .or_else(|| {
                self.config
                    .replay
                    .as_ref()
                    .and_then(|path| path.file_stem())
                    .map(|stem| stem.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "Unknown server".to_owned());
        tracing::debug!(%server, "replay recorder following a session");
        self.rec.conn = Some(Box::new(Conn {
            started: now,
            now: 0,
            server,
            player_name: String::new(),
            player_uuid: String::new(),
            dimension: None,
            sticky: StickyLog::default(),
            buffer: None,
            recording: None,
            mic: None,
            mic_ring: std::collections::VecDeque::new(),
            desktop: None,
            pending_start: false,
            game_title: None,
            board_seen: None,
            title_checked: None,
            world_started: 0,
            bookmarks: Vec::new(),
        }));
    }

    /// The session ended (or the mod was turned off): finish the recording, drop the buffer,
    /// turn the tap off.
    fn rec_end_session(&mut self) {
        self.rec.tapping = false;
        if let Some(session) = self.session.as_mut() {
            session.set_raw_tap(false);
            drop(session.take_tapped_raw());
        }
        let Some(mut conn) = self.rec.conn.take() else {
            return;
        };
        conn.mic = None;
        if let Some(recording) = conn.recording.take() {
            let meta = recording.final_meta(conn.now, &self.rec.tags, conn.game_title.as_ref());
            self.rec.jobs.push(Job::Recording(finish(recording, meta)));
        }
        self.rec.capturing = false;
    }

    /// Right after `Session::poll_next` applied a message. One `bool` while
    /// the tap is off.
    #[inline]
    pub(crate) fn rec_after_packet(&mut self) {
        if !self.rec.tapping {
            return;
        }
        let Some(raw) = self.session.as_mut().and_then(Session::take_tapped_raw) else {
            return;
        };
        // The login and the first JoinGame can arrive in one poll (a capture-driven session):
        // take the player's name before a recording's metadata needs it.
        if let (Some(conn), Some(session)) = (self.rec.conn.as_deref_mut(), self.session.as_ref()) {
            if conn.player_name.is_empty() && !session.state().local.username.is_empty() {
                let local = &session.state().local;
                conn.player_name.clone_from(&local.username);
                conn.player_uuid = local.uuid.hyphenated().to_string();
            }
        }
        self.rec.on_packet(raw);
    }

    /// The microphone: open while a recording runs with the option on (and not paused).
    ///
    /// The device is open (the Replay mod being on is implied by the session being followed)
    /// while **Record microphone** is on and either a recording runs unpaused (its frames go
    /// into the file) or the clip key's session buffer runs (they go into a 300 s ring, so a clip
    /// has the microphone even with no recording running). It closes when neither holds, and
    /// with the session. Frames are timed on the session clock, which never pauses; the lead in
    /// each record stays valid in the file's clock (a constant offset while unpaused).
    fn rec_mic(&mut self, now: Instant) {
        let opts = self.rec.opts;
        let Some(conn) = self.rec.conn.as_deref_mut() else {
            return;
        };
        let to_file = conn
            .recording
            .as_ref()
            .is_some_and(|recording| recording.paused_at.is_none());
        let to_ring = opts.keys[KEY_CLIP] != Input::Unbound && conn.buffer.is_some();
        if !opts.mic || !(to_file || to_ring) {
            if conn.mic.take().is_some() {
                tracing::info!("replay microphone off");
            }
            conn.mic_ring.clear();
            return;
        }
        if !to_ring {
            conn.mic_ring.clear();
        }
        let mut mic = match conn.mic.take() {
            Some(mic) => mic,
            None => {
                // No borrow of the session is held across opening the device.
                let mic = self.rec.open_mic();
                let Some(conn) = self.rec.conn.as_deref_mut() else {
                    return;
                };
                // Room for the whole ring at once, so pushing never allocates.
                let missing = MIC_RING_FRAMES.saturating_sub(conn.mic_ring.len());
                conn.mic_ring.reserve(missing);
                mic
            }
        };
        let Some(conn) = self.rec.conn.as_deref_mut() else {
            return;
        };
        let session_now = conn.now;
        mic.frame(session_now, now);
        if !mic.records.is_empty() {
            for record in mic.records.chunks_exact(mc_replay::mic::RECORD_BYTES) {
                if to_file {
                    if let Some(recording) = conn.recording.as_mut() {
                        recording.has_mic = true;
                        let time = recording.time(session_now);
                        recording.writer.push(time, StreamId::Mic, record);
                    }
                }
                if to_ring {
                    let mut bytes = [0u8; mc_replay::mic::RECORD_BYTES];
                    bytes.copy_from_slice(record);
                    conn.mic_ring.push_back(MicSlot {
                        time: session_now,
                        bytes,
                    });
                }
            }
            // The ring holds the longest clip there can be.
            while conn
                .mic_ring
                .front()
                .is_some_and(|slot| session_now.saturating_sub(slot.time) > MIC_RING_SPAN)
            {
                conn.mic_ring.pop_front();
            }
        }
        conn.mic = Some(mic);
    }

    /// Jobs that finished: toasts, and the storage cap after a recording.
    fn rec_poll_jobs(&mut self) {
        if self.rec.jobs.is_empty() {
            return;
        }
        let max = self.rec.opts.max_storage_bytes;
        let mut index = 0;
        while index < self.rec.jobs.len() {
            let (done, clip) = match &mut self.rec.jobs[index] {
                Job::Recording(job) => (job.try_result().cloned(), false),
                Job::Clip(job) => (job.try_result().cloned(), true),
            };
            let Some(result) = done else {
                index += 1;
                continue;
            };
            self.rec.jobs.swap_remove(index);
            match (result, clip) {
                (Ok(finished), true) => {
                    let (start, end) = finished.meta.window();
                    let seconds = (end.saturating_sub(start) + 500_000) / 1_000_000;
                    tracing::info!(file = %nw_ra::log_name(&finished.path), seconds, "clip saved");
                    self.rec.notice(
                        "Clip saved",
                        format!("Last {seconds} s, in Replays › Clips"),
                    );
                }
                (Ok(finished), false) => {
                    tracing::info!(
                        file = %nw_ra::log_name(&finished.path),
                        bytes = finished.bytes,
                        "replay saved"
                    );
                    // Ended with its session (a disconnect): say so on the screen that follows.
                    if !self.rec_in_world() {
                        let text = format!("Replay saved: {}", finished.meta.title);
                        self.rec.title_notice = Some((text, Instant::now()));
                    }
                    let library = self.rec.library();
                    let keep = finished.path.clone();
                    let _ = std::thread::Builder::new()
                        .name("mc-rs replay storage cap".into())
                        .spawn(move || {
                            // Best effort, as the repair thread.
                            let _ = mc_core::panic_scope::contain(|| {
                                library.enforce_cap(max, Some(&keep));
                            });
                        });
                }
                (Err(error), true) => {
                    tracing::warn!(%error, "clip not saved");
                    self.rec.notice("Clip not saved", error);
                }
                (Err(error), false) => {
                    tracing::warn!(%error, "replay not saved");
                }
            }
        }
    }

    /// Show queued notices as toasts (in game only; the toast draws over a world).
    fn rec_show_notices(&mut self) {
        if self.rec.notices.is_empty() {
            return;
        }
        let now_ms = self.started.elapsed().as_millis() as i64;
        // The newest wins: a toast replaces the one showing.
        let newest = self.rec.notices.pop();
        self.rec.notices.clear();
        if let Some(notice) = newest {
            let icon = Some(mc_protocol::ItemStack {
                // A music disc: "this is a recording".
                item_id: 2256,
                count: 1,
                damage: 0,
                nbt: None,
            });
            self.achievement_toast
                .display(notice.title, notice.detail, icon, now_ms);
        }
    }

    /// The four keys, as press edges, in game only.
    pub(crate) fn rec_shell_frame(&mut self, input: &InputState) {
        if !self.rec.enabled || self.rec.opts.keys.iter().all(|&k| k == Input::Unbound) {
            return;
        }
        let in_game = matches!(self.ui, Ui::InGame) && !self.debug_f3_held;
        let armed = std::mem::replace(&mut self.rec.keys_armed, true);
        let mut pressed = [false; KEY_COUNT];
        for (i, &key) in self.rec.opts.keys.iter().enumerate() {
            let down = key != Input::Unbound && input.is_down(key);
            pressed[i] = armed && down && !self.rec.keys_down[i] && in_game;
            self.rec.keys_down[i] = down;
        }
        if pressed[KEY_START] {
            self.rec.start_key();
        }
        if pressed[KEY_PAUSE] {
            self.rec.pause_key();
        }
        if pressed[KEY_STOP] {
            self.rec.stop_key();
        }
        if pressed[KEY_CLIP] {
            self.rec.clip_key();
        }
        if pressed[KEY_BOOKMARK] {
            self.rec.bookmark_key(std::time::Instant::now());
        }
        if pressed.iter().any(|&p| p) {
            self.rec_show_notices();
        }
    }

    /// The recording dot, top right; two bars while paused. Not drawn under F3
    /// (its right column is there) or with F1 (the whole overlay is hidden).
    pub(crate) fn rec_draw_dot(&mut self, list: &mut DrawList, ctx: &ScreenCtx) {
        // The Bookmark key's confirmation (an `Option` test when there is none).
        if self.rec.bookmark_flash.is_some() {
            self.rec_draw_bookmark_flash(list, ctx);
        }
        if !self.rec.opts.indicator || self.debug.visible || !self.rec.recording() {
            return;
        }
        let (x, y) = (ctx.width - 9, 4);
        if self.rec.paused() {
            const AMBER: u32 = 0xE0FF_B300;
            list.rect(x, y, x + 2, y + 5, AMBER);
            list.rect(x + 3, y, x + 5, y + 5, AMBER);
        } else {
            // A 5x5 dot with its corners cut.
            const RED: u32 = 0xE0E5_3935;
            list.rect(x + 1, y, x + 4, y + 1, RED);
            list.rect(x, y + 1, x + 5, y + 4, RED);
            list.rect(x + 1, y + 4, x + 4, y + 5, RED);
        }
    }

    /// Whether the player is in a world (a session that has not ended).
    fn rec_in_world(&self) -> bool {
        self.session.as_ref().is_some_and(|session| {
            !matches!(session.phase(), crate::session::Phase::Disconnected(_))
        })
    }

    /// "Replay saved: <title>" for five seconds (then a short fade) at
    /// the top left of the screen that follows a disconnect, where the in-world toast cannot
    /// draw. One `Option` test the rest of the time.
    pub(crate) fn rec_draw_title_notice(&mut self, list: &mut DrawList, _ctx: &ScreenCtx) {
        let Some((_, at)) = self.rec.title_notice.as_ref() else {
            return;
        };
        if self.rec_in_world() {
            return;
        }
        const SHOW: Duration = Duration::from_secs(5);
        const FADE: Duration = Duration::from_millis(800);
        let age = at.elapsed();
        if age >= SHOW + FADE {
            self.rec.title_notice = None;
            return;
        }
        let fade = age.saturating_sub(SHOW).as_secs_f32() / FADE.as_secs_f32();
        let alpha = ((1.0 - fade) * 255.0) as u32;
        // The font treats an alpha under 4 as opaque.
        if alpha < 8 {
            return;
        }
        let color = ((alpha << 24) | 0x00E0_E0E0) as i32;
        if let Some((text, _)) = self.rec.title_notice.as_ref() {
            self.font
                .draw_string_with_shadow(list, text, 4.0, 4.0, color);
        }
    }

    /// Quitting saves the recording and waits (up to 5 s) for it and for
    /// clips still being written.
    pub(crate) fn rec_shutdown(&mut self) {
        self.rec_end_session();
        let deadline = Instant::now() + QUIT_WAIT;
        for job in self.rec.jobs.drain(..) {
            let left = deadline.saturating_duration_since(Instant::now());
            let (Job::Recording(job) | Job::Clip(job)) = job;
            if let Err(error) = job.wait(left) {
                tracing::warn!(%error, "a replay was not finished before quitting");
            }
        }
    }

    /// Whether a `--replay` capture should keep its frames' bytes so the
    /// recorder can record the session it drives (only while the replay mod is on).
    pub(crate) fn rec_wants_capture_bytes(&self) -> bool {
        self.mods.settings.enabled(MOD_ID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_protocol::PacketBuffer;

    fn bytes(packet: &mc_protocol::play::S2CPlay) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    #[test]
    fn world_resets_and_sticky_packets_are_recognised_from_their_bytes() {
        use mc_protocol::play::S2CPlay;
        let join = S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 42,
            hardcore_mode: false,
            game_mode: 0,
            dimension: -1,
            difficulty: 1,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        assert_eq!(classify(&bytes(&join)), PacketClass::Join { dimension: -1 });
        let respawn = S2CPlay::Respawn(s2c::Respawn {
            dimension: 1,
            difficulty: 2,
            game_mode: 0,
            level_type: "default".into(),
        });
        assert_eq!(
            classify(&bytes(&respawn)),
            PacketClass::Respawn { dimension: 1 }
        );
        let brand = S2CPlay::PluginMessage(s2c::PluginMessage {
            channel: "MC|Brand".into(),
            data: b"\x07Example".to_vec(),
        });
        assert_eq!(classify(&bytes(&brand)), PacketClass::Sticky(Sticky::Brand));
        let other = S2CPlay::PluginMessage(s2c::PluginMessage {
            channel: "MC|Other".into(),
            data: Vec::new(),
        });
        assert_eq!(classify(&bytes(&other)), PacketClass::Plain);
        assert_eq!(
            classify(&[0x3E, 1, 2]),
            PacketClass::Sticky(Sticky::Scoreboard)
        );
        assert_eq!(classify(&[0x38]), PacketClass::Sticky(Sticky::TabList));
        assert_eq!(classify(&[0x15, 1, 2, 3]), PacketClass::Plain);
        assert_eq!(classify(&[]), PacketClass::Plain);
        // A JoinGame cut short is not a reset.
        assert_eq!(classify(&[0x01, 0, 0]), PacketClass::Plain);
    }

    #[test]
    fn the_sticky_log_keeps_what_a_world_reset_does_not_clear() {
        let mut log = StickyLog::default();
        let raw = |b: &[u8]| -> Box<[u8]> { Box::from(b) };
        log.add(Sticky::Join, raw(b"\x01join1"), 0);
        log.add(Sticky::TabList, raw(b"\x38a"), 0);
        log.add(Sticky::Scoreboard, raw(b"\x3Eteam"), 0);
        log.add(Sticky::HeaderFooter, raw(b"\x47one"), 0);
        log.add(Sticky::HeaderFooter, raw(b"\x47two"), 0);
        log.add(Sticky::Brand, raw(b"\x3Fb"), 0);
        let payloads = |log: &StickyLog| -> Vec<Vec<u8>> {
            log.prefix(log.len())
                .into_iter()
                .map(|r| r.payload)
                .collect()
        };
        assert_eq!(
            payloads(&log),
            vec![
                b"\x01join1".to_vec(),
                b"\x38a".to_vec(),
                b"\x3Eteam".to_vec(),
                b"\x47two".to_vec(),
                b"\x3Fb".to_vec()
            ],
            "only the last header/footer counts"
        );
        // A new JoinGame: the old one and the scoreboard go; the tab list stays.
        log.compact_for_join();
        log.add(Sticky::Join, raw(b"\x01join2"), 0);
        assert_eq!(
            payloads(&log),
            vec![
                b"\x38a".to_vec(),
                b"\x47two".to_vec(),
                b"\x3Fb".to_vec(),
                b"\x01join2".to_vec()
            ]
        );
        // Replacing an entry before a protected index moves that index back.
        let dropped = log.add(Sticky::Brand, raw(b"\x3Fc"), 3);
        assert_eq!(dropped, 1, "the old brand was the third entry");
    }

    /// Default titles tell recordings apart: game and map, else game and server, else server;
    /// then the local date and time ("Nov 12, 9:41 PM").
    #[test]
    fn default_titles_name_the_game_the_map_and_when() {
        let at = 1_794_541_686_841; // 2026-11-12 around 22:48 local on the machine that wrote this
        let when = mc_replay::local_title_time(at);
        assert!(
            (when.ends_with(" AM") || when.ends_with(" PM")) && when.contains(", "),
            "{when}"
        );
        let skywars = GameTitle {
            game: "SkyWars".into(),
            map: Some("Dwarf Fortress".into()),
        };
        assert_eq!(
            default_title(Some(&skywars), "mc.example.net", at),
            format!("SkyWars · Dwarf Fortress · {when}")
        );
        let duels = GameTitle {
            game: "Duels".into(),
            map: None,
        };
        assert_eq!(
            default_title(Some(&duels), "mc.example.net", at),
            format!("Duels · mc.example.net · {when}")
        );
        assert_eq!(
            default_title(None, "play.example.net", at),
            format!("play.example.net · {when}")
        );
    }

    /// The tap adds nothing to the messages every packet travels in: its variants are boxed, so
    /// `ToGame` is no larger than `Packet(S2CPlay)` plus its tag (it grew by 16 bytes before the
    /// box: measured 168 against 152).
    #[test]
    fn the_tap_does_not_grow_the_network_message() {
        #[allow(dead_code)]
        enum Untapped {
            Packet(mc_protocol::play::S2CPlay),
            Other(std::sync::Arc<u8>),
        }
        let packet = std::mem::size_of::<mc_protocol::play::S2CPlay>();
        let message = std::mem::size_of::<crate::net::ToGame>();
        let untapped = std::mem::size_of::<Untapped>();
        println!("S2CPlay {packet} bytes, ToGame {message} bytes, without the tap {untapped}");
        assert!(message <= untapped, "{message} vs {untapped}");
    }

    /// The microphone timeline: samples arrive in uneven bursts with jittery delivery, but the
    /// records' frames start exactly 20 ms apart on the replay clock (time minus lead), and the
    /// audio decodes back to the voice-band tone that went in.
    #[test]
    fn microphone_frames_sit_exactly_20_ms_apart_on_the_replay_clock() {
        let mut mic = Mic::with_capture(MicCapture::fake());
        let rate = mc_replay::mic::SAMPLE_RATE as usize;
        let tone: Vec<i16> = (0..rate * 3)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (9000.0 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()) as i16
            })
            .collect();
        let start = Instant::now();
        let mut fed = 0usize;
        let mut starts = Vec::new();
        let mut decoded = Vec::new();
        // 3 s of frames at an uneven frame rate; the device delivers 10 ms bursts.
        let mut file_now = 1_000_000u64;
        let mut frame = 0u64;
        while fed < tone.len() {
            frame += 1;
            file_now += [3_000, 7_000, 1_000, 16_000][frame as usize % 4];
            // Deliver everything "captured" up to now, minus a jittery 0-9 ms in flight.
            let due = (((file_now - 1_000_000) as usize) * rate / 1_000_000)
                .saturating_sub((frame as usize % 10) * rate / 1000)
                .min(tone.len());
            let due = due - due % 240;
            if due > fed {
                mic.capture.feed(&tone[fed..due]);
                fed = due;
            }
            mic.frame(file_now, start + Duration::from_micros(file_now));
            for record in mic.records.chunks_exact(mc_replay::mic::RECORD_BYTES) {
                let lead = mc_replay::mic::decode_record(record, &mut decoded).expect("decodes");
                starts.push(file_now - u64::from(lead));
            }
        }
        assert!(starts.len() > 140, "{} frames", starts.len());
        for pair in starts.windows(2) {
            let gap = pair[1] as i64 - pair[0] as i64;
            assert!(
                (gap - 20_000).abs() <= 2_100,
                "consecutive frames are 20 ms apart (within the 2 ms/s drift correction): {gap}"
            );
        }
        // The first frame starts near when its first sample was captured (1 s on this clock; the
        // first delivery's delay, up to 9 ms in flight plus a 10 ms burst, is all it can be off).
        assert!(starts[0].abs_diff(1_000_000) < 25_000, "{}", starts[0]);
        let n = decoded.len().min(tone.len());
        let peak = decoded[..n]
            .iter()
            .map(|s| s.unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(
            peak > 8000 && peak < 10_000,
            "the tone survives: peak {peak}"
        );
    }
}

/// The recorder through the real client: a session driven by a capture (the SkyWars bench
/// scene), the recorder on, the files read back and every packet compared byte for byte with
/// what the client applied. Needs the vanilla jar (`MC_JARS_DIR`); no GPU.
#[cfg(test)]
mod client_tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use mc_parity::capture::{Capture, CaptureMeta, Direction, Recorder};
    use mc_protocol::play::{s2c, S2CPlay};
    use mc_protocol::{PacketBuffer, State};
    use mc_replay::{Library, Reader, ReplayKind, StreamId};

    use super::super::super::Client;
    use super::ids;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{NetHandle, ReplaySource};
    use crate::session::{JoinSettings, Session};

    // The desktop audio tests, with these helpers.
    mod nw_re_desktop;

    /// An offline client with its game dir in a temp folder.
    fn client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    fn skywars_capture() -> Option<(PathBuf, Capture)> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../optimization/scratch/skywars-scene-2026-11-05/captures")
            .join("sw-mega-doubles-mid.mccap");
        let capture = Capture::read_file(&path).ok()?;
        Some((path, capture))
    }

    /// The clientbound Play frames a capture-driven session applies, in order (the source
    /// drops KeepAlive, SetCompression and Disconnect: the network task's own packets).
    fn applied_frames(capture: &Capture) -> Vec<Vec<u8>> {
        capture
            .frames()
            .iter()
            .filter(|f| f.direction == Direction::Clientbound && f.state == State::Play)
            .filter(|f| !matches!(f.packet_id(), Some(0x00 | 0x40 | 0x46)))
            .map(|f| f.payload.clone())
            .collect()
    }

    fn packets(path: &Path) -> (Reader, Vec<(u64, Vec<u8>)>) {
        let reader = Reader::open(path).expect("the replay opens");
        let packets = reader
            .records()
            .map(|r| r.expect("record"))
            .filter(|r| r.stream == StreamId::Packet)
            .map(|r| (r.time, r.payload))
            .collect();
        (reader, packets)
    }

    /// Drive `client` from `source` frame by frame on a fake frame clock (`step` per frame),
    /// applying packets exactly as the game loop does; `each` runs after every frame.
    fn drive(
        client: &mut Client,
        frames: usize,
        step: Duration,
        mut each: impl FnMut(&mut Client, usize),
    ) {
        let start = Instant::now();
        client.rec.sync(&client.mods.settings);
        for frame in 0..frames {
            client.rec_frame(start + step * frame as u32);
            client.poll_in_place();
            each(client, frame);
        }
    }

    /// Open a session on `capture`; returns how many frames deliver all of it.
    fn open(client: &mut Client, capture: &Capture, path: &Path, step_micros: u64) -> usize {
        let source = ReplaySource::from_capture_keeping_bytes(
            capture,
            path.to_path_buf(),
            step_micros,
            true,
        )
        .expect("a replay source");
        let frames = (source.summary().span_micros / step_micros) as usize + 20;
        client.session = Some(Session::new(
            NetHandle::replay(source),
            JoinSettings::default(),
        ));
        client.current_server_address = Some("skywars.test".to_owned());
        frames
    }

    fn assert_exhausted(client: &Client) {
        assert!(
            client
                .session
                .as_ref()
                .and_then(Session::replay_source)
                .is_some_and(ReplaySource::is_exhausted),
            "the whole capture was applied"
        );
    }

    /// "Record every game" on the SkyWars scene: one file, holding every packet the client
    /// applied, byte for byte and in order; and the clip key, with its session buffer, cuts a
    /// clip holding the same world.
    #[test]
    fn every_packet_the_client_applies_is_recorded_byte_for_byte() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        let Some((path, capture)) = skywars_capture() else {
            panic!("the SkyWars bench capture is missing");
        };
        client.mods.settings.set_enabled("replay", true);
        // F12 (LWJGL 88) as the clip key: the session buffer runs.
        client.mods.settings.set_key("replay", ids::KEY_CLIP, 88);
        // 50 ms of capture per frame, 50 ms of frame clock per frame.
        let frames = open(&mut client, &capture, &path, 50_000);
        let expected = applied_frames(&capture);
        drive(&mut client, frames, Duration::from_millis(50), |_, _| {});
        assert_exhausted(&client);
        // The clip key at the end: the last 30 s. Twice in the same frame (a double tap): two
        // clips, the second under its own name.
        client.rec.clip_key();
        client.rec.clip_key();
        // Leave: the recording is finished; wait for it and the clip.
        client.session = None;
        client.rec_frame(Instant::now() + Duration::from_secs(120));
        // It ended with its session: the screen that follows says "Replay saved".
        let deadline = Instant::now() + Duration::from_secs(30);
        while client.rec.title_notice.is_none() {
            assert!(Instant::now() < deadline, "the replay was not saved");
            std::thread::sleep(Duration::from_millis(10));
            client.rec_poll_jobs();
        }
        let notice = client.rec.title_notice.as_ref().map(|(t, _)| t.clone());
        assert!(
            notice
                .as_deref()
                .is_some_and(|t| t.starts_with("Replay saved: SkyWars · Dwarf Fortress · ")),
            "{notice:?}"
        );
        let ctx = mc_ui::screen_stack::ScreenCtx::new(client.resolution(854, 480), 0.0);
        let mut list = mc_ui::widget::DrawList::new();
        client.rec_draw_title_notice(&mut list, &ctx);
        assert!(!list.vertices().is_empty(), "drawn out of a world");
        client.rec_shutdown();

        let library = Library::new(dir.path());
        let full: Vec<_> = library.list_dir(false);
        assert_eq!(full.len(), 1, "one recording: {full:?}");
        let (reader, recorded) = packets(&full[0].path);
        assert!(!reader.recovered());
        assert_eq!(
            recorded.len(),
            expected.len(),
            "every applied packet recorded, nothing else"
        );
        for (i, ((_, got), want)) in recorded.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "packet {i} differs");
        }
        // Times: one per frame, non-decreasing, spanning the scene.
        assert!(recorded.windows(2).all(|w| w[0].0 <= w[1].0));
        assert_eq!(recorded[0].0, 0, "the file starts at the JoinGame");
        let span = recorded.last().unwrap().0;
        let capture_span = (frames as u64 - 20) * 50_000;
        assert!(
            span.abs_diff(capture_span) <= 100_000,
            "the file's clock follows the frames: {span} µs for {capture_span} µs of capture"
        );
        let meta = reader.meta();
        assert_eq!(meta.server, "skywars.test");
        assert!(
            meta.title.starts_with("SkyWars · Dwarf Fortress · "),
            "titled from the sidebar, then when: {}",
            meta.title
        );
        assert_eq!(meta.world_resets, vec![0]);
        assert_eq!(meta.window().0, 0);
        assert!(meta.duration >= span);
        let bytes = std::fs::metadata(&full[0].path).unwrap().len();
        println!(
            "SkyWars scene: {} packets over {:.1} s recorded into {bytes} bytes ({:.2} MB/min)",
            recorded.len(),
            span as f64 / 1e6,
            bytes as f64 / 1e6 / (span as f64 / 60e6)
        );

        // The clip: the same world from its JoinGame, the last 30 s shown.
        let mut clips = library.list_dir(true);
        assert_eq!(clips.len(), 2, "two clips: {clips:?}");
        clips.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(
            packets(&clips[1].path).1.len(),
            expected.len(),
            "the second is whole too"
        );
        let (clip, clip_packets) = packets(&clips[0].path);
        assert!(
            clip.meta().title.starts_with("SkyWars · Dwarf Fortress · "),
            "{}",
            clip.meta().title
        );
        let ReplayKind::Clip { start, end } = clip.meta().kind else {
            panic!("a clip: {:?}", clip.meta().kind);
        };
        assert_eq!(end - start, 30_000_000, "the default clip length");
        assert_eq!(clip_packets.len(), expected.len());
        for (i, ((_, got), want)) in clip_packets.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "clip packet {i} differs");
        }
    }

    /// The start key in the middle of a world: the file begins with the buffered world (from
    /// its JoinGame), so it holds every packet; the visible part starts at the key press.
    #[test]
    fn a_recording_started_mid_world_holds_the_whole_world() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        let Some((path, capture)) = skywars_capture() else {
            panic!("the SkyWars bench capture is missing");
        };
        client.mods.settings.set_enabled("replay", true);
        client
            .mods
            .settings
            .set_bool("replay", ids::AUTO_START, false);
        client.mods.settings.set_key("replay", ids::KEY_START, 87); // F11
        let frames = open(&mut client, &capture, &path, 50_000);
        let expected = applied_frames(&capture);
        let mut pressed_at = None;
        drive(
            &mut client,
            frames,
            Duration::from_millis(50),
            |client, frame| {
                if frame == 600 {
                    assert!(!client.rec.recording(), "nothing records before the key");
                    client.rec.start_key();
                    assert!(client.rec.recording());
                    pressed_at = client.rec.conn.as_deref().map(|c| c.now);
                }
            },
        );
        assert_exhausted(&client);
        client.rec.stop_key();
        client.session = None;
        client.rec_frame(Instant::now() + Duration::from_secs(120));
        client.rec_shutdown();

        let full = Library::new(dir.path()).list_dir(false);
        assert_eq!(full.len(), 1);
        let (reader, recorded) = packets(&full[0].path);
        assert_eq!(
            recorded.len(),
            expected.len(),
            "the buffered world plus the rest"
        );
        for (i, ((_, got), want)) in recorded.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "packet {i} differs");
        }
        let view_start = reader.meta().window().0;
        let pressed_at = pressed_at.expect("pressed");
        assert!(
            view_start > 25_000_000 && view_start <= pressed_at,
            "the viewer starts at the key press: {view_start} (pressed at {pressed_at})"
        );
    }

    /// A recording whose file reaches the replay storage limit is
    /// saved and stopped (a server can send data as fast as the connection carries it); what
    /// was recorded up to then is a whole, readable replay.
    #[test]
    fn a_recording_that_reaches_the_storage_limit_is_saved_and_stopped() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        let Some((path, capture)) = skywars_capture() else {
            panic!("the SkyWars bench capture is missing");
        };
        client.mods.settings.set_enabled("replay", true);
        let frames = open(&mut client, &capture, &path, 50_000);
        let expected = applied_frames(&capture);
        let mut stopped_at = None;
        drive(
            &mut client,
            frames,
            Duration::from_millis(50),
            |client, frame| {
                // One byte (the setting itself goes down to 1 GB): the first block on disk
                // reaches it.
                client.rec.opts.max_storage_bytes = 1;
                if frame == 60 {
                    // The frame clock is fake, the writer thread is not: wait for its first
                    // block (sealed at 2 s of frame clock) to be on disk.
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while client
                        .rec
                        .conn
                        .as_deref()
                        .and_then(|conn| conn.recording.as_ref())
                        .is_some_and(|recording| recording.writer.stats().stored_bytes == 0)
                    {
                        assert!(Instant::now() < deadline, "no block reached the disk");
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
                if frame > 0 && stopped_at.is_none() && !client.rec.recording() {
                    stopped_at = Some(frame);
                }
            },
        );
        assert!(
            stopped_at.is_some_and(|frame| frame <= 61),
            "stopped at the first frame with a block on disk: {stopped_at:?}"
        );
        client.session = None;
        client.rec_frame(Instant::now() + Duration::from_secs(120));
        client.rec_shutdown();

        let full = Library::new(dir.path()).list_dir(false);
        assert_eq!(full.len(), 1, "one recording: {full:?}");
        let (reader, recorded) = packets(&full[0].path);
        assert!(!reader.recovered(), "saved with its footer");
        assert!(
            !recorded.is_empty() && recorded.len() < expected.len(),
            "the part up to the limit: {} of {} packets",
            recorded.len(),
            expected.len()
        );
        for (i, ((_, got), want)) in recorded.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "packet {i} differs");
        }
    }

    fn frame(packet: &S2CPlay) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    fn chat(text: &str) -> S2CPlay {
        S2CPlay::Chat(s2c::Chat {
            component: format!(r#"{{"text":"{text}"}}"#),
            position: 0,
        })
    }

    /// A proxy's server switch (two `Respawn`s) splits "Record every game" into two files, and
    /// the second begins with the state a world reset does not clear: the JoinGame, the last
    /// tab header/footer, the difficulty and the brand.
    #[test]
    fn a_server_switch_starts_a_new_file_with_the_state_prefix() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        client.mods.settings.set_enabled("replay", true);
        let join = S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 7,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 1,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        let header = |text: &str| {
            S2CPlay::PlayerListHeaderFooter(s2c::PlayerListHeaderFooter {
                header: format!(r#"{{"text":"{text}"}}"#),
                footer: r#"{"text":""}"#.into(),
            })
        };
        let brand = S2CPlay::PluginMessage(s2c::PluginMessage {
            channel: "MC|Brand".into(),
            data: b"\x07Example".to_vec(),
        });
        let difficulty = S2CPlay::ServerDifficulty(s2c::ServerDifficulty {
            difficulty: s2c::ServerDifficultyLevel::Normal,
        });
        let respawn = |dimension: i32| {
            S2CPlay::Respawn(s2c::Respawn {
                dimension,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })
        };
        // (seconds, packet): a lobby, then a switch at 25 s, then a game. (a world
        // left after less than 20 s stays in the next file, see `nw_ra.rs`; this lobby is long.)
        let script: Vec<(f64, S2CPlay)> = vec![
            (0.0, join.clone()),
            (0.1, header("lobby one")),
            (0.2, header("lobby two")),
            (0.3, brand.clone()),
            (0.4, difficulty.clone()),
            (1.0, chat("in the lobby")),
            (25.0, respawn(-1)),
            (25.05, respawn(0)),
            (26.0, chat("in the game")),
            (27.0, chat("still in the game")),
        ];
        let mut recorder = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "switch".into(),
            started_at_unix_millis: 0,
        });
        for (seconds, packet) in &script {
            recorder.record_at(
                Direction::Clientbound,
                State::Play,
                (seconds * 1e6) as u64,
                &frame(packet),
            );
        }
        let capture = recorder.finish();
        open(&mut client, &capture, Path::new("switch.mccap"), 25_000);
        drive(&mut client, 1200, Duration::from_millis(25), |_, _| {});
        client.session = None;
        client.rec_frame(Instant::now() + Duration::from_secs(120));
        client.rec_shutdown();

        let mut files = Library::new(dir.path()).list_dir(false);
        assert_eq!(files.len(), 2, "one file per world: {files:?}");
        // The game's file is the one that begins with a state prefix.
        files.sort_by_key(|e| Reader::open(&e.path).expect("opens").blocks()[0].prefix);
        let (_, lobby) = packets(&files[0].path);
        let lobby: Vec<_> = lobby.into_iter().map(|(_, p)| p).collect();
        assert_eq!(
            lobby,
            script[..6]
                .iter()
                .map(|(_, p)| frame(p))
                .collect::<Vec<_>>(),
            "the lobby file ends at the switch"
        );
        let (game_reader, game) = packets(&files[1].path);
        let payloads: Vec<_> = game.iter().map(|(_, p)| p.clone()).collect();
        let expected: Vec<Vec<u8>> = [
            frame(&join),
            frame(&header("lobby two")),
            frame(&brand),
            frame(&difficulty),
        ]
        .into_iter()
        .chain(script[6..].iter().map(|(_, p)| frame(p)))
        .collect();
        assert_eq!(payloads, expected, "prefix, then the new world");
        assert!(
            game[..5].iter().all(|(t, _)| *t == 0),
            "prefix and reset at 0"
        );
        assert!(game_reader.blocks()[0].prefix);
        assert!(game_reader.blocks()[1].keyframe);
        let last = game.last().unwrap().0;
        assert!(
            (1_900_000..2_100_000).contains(&last),
            "the game file's clock starts at the switch: {last}"
        );
    }

    /// Pause freezes the file's clock (packets keep coming, stamped at the pause), resume cuts
    /// the paused time out; the recording dot is red while recording, two amber bars while
    /// paused, and gone under F3.
    #[test]
    fn pausing_freezes_the_clock_and_the_dot_shows_it() {
        use mc_ui::screen_stack::ScreenCtx;
        use mc_ui::widget::DrawList;
        let Some((mut client, dir)) = client() else {
            return;
        };
        client.mods.settings.set_enabled("replay", true);
        client.mods.settings.set_key("replay", ids::KEY_PAUSE, 25); // P
        let join = S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 7,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 1,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        let mut recorder = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "pause".into(),
            started_at_unix_millis: 0,
        });
        recorder.record_at(Direction::Clientbound, State::Play, 0, &frame(&join));
        for second in 1..=6u64 {
            recorder.record_at(
                Direction::Clientbound,
                State::Play,
                second * 1_000_000,
                &frame(&chat(&format!("{second}"))),
            );
        }
        let capture = recorder.finish();
        open(&mut client, &capture, Path::new("pause.mccap"), 100_000);
        let ctx = ScreenCtx::new(client.resolution(854, 480), 0.0);
        let dot = |client: &mut Client| {
            let mut list = DrawList::new();
            client.rec_draw_dot(&mut list, &ctx);
            list.vertices().len()
        };
        let mut drawn = Vec::new();
        drive(
            &mut client,
            71,
            Duration::from_millis(100),
            |client, frame| {
                match frame {
                    15 | 35 => client.rec.pause_key(),
                    _ => {}
                }
                if frame == 10 || frame == 20 {
                    drawn.push(dot(client));
                }
                if frame == 50 {
                    client.debug.visible = true;
                    drawn.push(dot(client));
                    client.debug.visible = false;
                }
                if frame == 70 {
                    client.rec.stop_key();
                    drawn.push(dot(client));
                }
            },
        );
        client.rec_shutdown();
        assert!(drawn[0] > 0, "a dot while recording");
        assert!(
            drawn[1] > 0 && drawn[1] < drawn[0],
            "two bars while paused: {drawn:?}"
        );
        assert_eq!(drawn[2], 0, "nothing under F3");
        assert_eq!(drawn[3], 0, "nothing once stopped");

        let full = Library::new(dir.path()).list_dir(false);
        assert_eq!(full.len(), 1);
        let (reader, recorded) = packets(&full[0].path);
        let times: Vec<u64> = recorded.iter().map(|(t, _)| *t).collect();
        // JoinGame at 0; chat 1 s at 1 s; 2 s and 3 s during the pause (from 1.5 s) frozen at
        // 1.5 s; after resuming at 3.5 s (2 s cut out) 4 s, 5 s, 6 s at 2, 3, 4 s.
        assert_eq!(
            times,
            vec![0, 1_000_000, 1_500_000, 1_500_000, 2_000_000, 3_000_000, 4_000_000]
        );
        assert_eq!(reader.meta().duration, 5_000_000, "7 s minus the 2 s pause");
    }

    /// A capture of `seconds` of play: a JoinGame, then a chat line every second.
    fn chatty_capture(seconds: u64) -> Capture {
        let join = S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 7,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 1,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        let mut recorder = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "mic".into(),
            started_at_unix_millis: 0,
        });
        recorder.record_at(Direction::Clientbound, State::Play, 0, &frame(&join));
        for second in 1..=seconds {
            recorder.record_at(
                Direction::Clientbound,
                State::Play,
                second * 1_000_000,
                &frame(&chat(&format!("{second}"))),
            );
        }
        recorder.finish()
    }

    /// "Record microphone" on and the clip key bound, no recording running: the microphone
    /// opens for the clip buffer's ring, a clip carries frames covering its whole window
    /// (merged into the world in time order), and the device closes with the session.
    #[test]
    fn a_clip_carries_the_microphone_with_no_recording_running() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        client.mods.settings.set_enabled("replay", true);
        client
            .mods
            .settings
            .set_bool("replay", ids::AUTO_START, false);
        client.mods.settings.set_bool("replay", ids::MIC, true);
        client.mods.settings.set_key("replay", ids::KEY_CLIP, 88);
        let capture = mc_replay::mic::MicCapture::fake();
        let feeder = capture.feeder();
        client.rec.test_mic = Some(capture);
        // 44 chat lines: the one at 44 s is applied on frame 440, inside the 450 driven.
        open(
            &mut client,
            &chatty_capture(44),
            Path::new("mic.mccap"),
            100_000,
        );
        let rate = mc_replay::mic::SAMPLE_RATE as usize;
        let mut fed = 0usize;
        drive(
            &mut client,
            450,
            Duration::from_millis(100),
            |client, frame| {
                // 100 ms of a loud tone a frame, delivered as the device would.
                let chunk: Vec<i16> = (fed..fed + rate / 10)
                    .map(|i| {
                        let t = i as f32 / rate as f32;
                        (8000.0 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()) as i16
                    })
                    .collect();
                feeder.feed(&chunk);
                fed += chunk.len();
                if frame == 5 {
                    let conn = client.rec.conn.as_deref().expect("following the session");
                    assert!(conn.mic.is_some(), "the microphone opens for the clip ring");
                    assert!(!client.rec.recording(), "no recording runs");
                }
            },
        );
        client.rec.clip_key();
        client.session = None;
        client.rec_frame(Instant::now() + Duration::from_secs(120));
        assert!(
            client.rec.conn.is_none(),
            "the session ended, and the device closed with it"
        );
        client.rec_shutdown();

        let library = Library::new(dir.path());
        assert!(library.list_dir(false).is_empty(), "no recording was made");
        let clips = library.list_dir(true);
        assert_eq!(clips.len(), 1);
        let reader = Reader::open(&clips[0].path).expect("the clip opens");
        let meta = reader.meta();
        assert!(meta.has_mic && meta.mic_format.is_some(), "{meta:?}");
        let (start, end) = meta.window();
        assert_eq!(end - start, 30_000_000);
        let records: Vec<_> = reader.records().map(|r| r.expect("record")).collect();
        assert!(
            records.windows(2).all(|w| w[0].time <= w[1].time),
            "time order"
        );
        let chats = records
            .iter()
            .filter(|r| r.stream == StreamId::Packet)
            .count();
        assert_eq!(chats, 45, "the JoinGame and every chat line of the world");
        let mut pcm = Vec::new();
        let starts: Vec<u64> = records
            .iter()
            .filter(|r| r.stream == StreamId::Mic)
            .map(|r| {
                let lead = mc_replay::mic::decode_record(&r.payload, &mut pcm).expect("decodes");
                r.time - u64::from(lead)
            })
            .collect();
        assert!(
            starts.len() >= 1490,
            "about 50 frames a second over 30 s: {}",
            starts.len()
        );
        let first = *starts.first().expect("mic frames");
        let last = *starts.last().expect("mic frames");
        assert!(
            first <= start,
            "the microphone covers the window's start: {first} > {start}"
        );
        assert!(
            last + 40_000 >= end,
            "the microphone covers the window's end: {last} vs {end}"
        );
        assert!(
            starts.windows(2).all(|w| w[1].abs_diff(w[0]) <= 22_100),
            "continuous speech, no gaps"
        );
        let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(peak > 7000 && peak < 9000, "the tone survives: {peak}");
    }

    /// The microphone stays closed while nothing could take its frames: with the start key
    /// bound (the buffer runs) but no clip key and no recording.
    #[test]
    fn the_microphone_stays_closed_without_a_clip_key_or_recording() {
        let Some((mut client, _dir)) = client() else {
            return;
        };
        client.mods.settings.set_enabled("replay", true);
        client
            .mods
            .settings
            .set_bool("replay", ids::AUTO_START, false);
        client.mods.settings.set_bool("replay", ids::MIC, true);
        client.mods.settings.set_key("replay", ids::KEY_START, 87);
        client.rec.test_mic = Some(mc_replay::mic::MicCapture::fake());
        open(
            &mut client,
            &chatty_capture(3),
            Path::new("mic.mccap"),
            100_000,
        );
        drive(&mut client, 20, Duration::from_millis(100), |client, _| {
            let conn = client.rec.conn.as_deref().expect("following the session");
            assert!(conn.buffer.is_some(), "the start key's buffer runs");
            assert!(conn.mic.is_none(), "but the microphone is not opened");
        });
        assert!(client.rec.test_mic.is_some(), "the device was never opened");
        client.rec_shutdown();
    }

    /// The mod off: no tap, no copy, no file; the hook is one `bool`.
    #[test]
    fn with_the_mod_off_nothing_is_tapped_or_written() {
        let Some((mut client, dir)) = client() else {
            return;
        };
        let Some((path, capture)) = skywars_capture() else {
            panic!("the SkyWars bench capture is missing");
        };
        assert!(!client.mods.settings.enabled("replay"), "off by default");
        open(&mut client, &capture, &path, 1_000_000);
        drive(&mut client, 80, Duration::from_millis(50), |client, _| {
            assert!(!client.rec.tapping && !client.rec.capturing());
            assert!(!client.session.as_ref().is_some_and(Session::raw_tap));
        });
        client.rec_shutdown();
        assert!(!dir.path().join("replays").exists(), "nothing written");
    }
}
