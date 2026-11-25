//! The replay playback engine: the replay session, time control,
//! seeking, the local-player puppet, the cameras, the capture of every record that is not a
//! clientbound packet, and the `replay_*` API the viewer and the exporter use.
//!
//! # How a replay plays back
//!
//! A replay (the `.aeroreplay`) is a stream of records in the order the recording client
//! produced them: within one frame, the packets the frame applied, then the frame's ticks, then
//! the frame itself. Playback re-applies that stream **in the same order**:
//!
//! * **Packets** go into a [`NetHandle::driven`](crate::net::NetHandle::driven) session and
//!   through the normal `poll_in_place` path, so the world, entities, chat, HUD and screens come
//!   from exactly the code that built them while recording. Nothing is ever sent: the driven
//!   handle has no socket and drops whatever the game queues.
//! * **Ticks** ([`codec::TickRecord`]) re-run `Client::tick` with the tick's recorded movement
//!   input and game key bindings, so the hand, swings, eating, digging cracks and the player's
//!   own particles come from the same code too. Right after the player's own tick, its state is
//!   **overwritten with the recorded state** ([`codec::PlayerStateRecord::apply`]): with
//!   unchanged code that is a no-op (the per-tick mismatch counter stays at zero, which is the
//!   fidelity check), and with changed code the player still goes exactly where it went.
//! * **Frames** ([`codec::FrameRecord`]) give the clock ([`clock::ReplayClock`]): replay time →
//!   ticks and partial tick, plus the per-frame look and zoom. At a recorded frame's time the
//!   rendered frame is the recorded one; between frames the look is interpolated.
//! * **Sounds** ([`codec::SoundRecord`]) are played from the record with the recorded variant;
//!   the game's own sound generation is suppressed in the engine while a replay is open.
//!
//! Seeking forward fast-applies records without rendering; seeking backward rebuilds the world
//! from the start of the file and fast-applies up to the target, in time slices with progress.
//!
//! # Cost when not replaying
//!
//! Every hook tests one `Option` (playback) or one `bool` (capture) and returns.

pub(crate) mod clock;
pub(crate) mod codec;
// Menus, chat and the inventory your clicks changed
pub(crate) mod gui;
// The recorded mouse pointer, the desktop's standard arrow
#[cfg(test)]
mod nw_checks;
pub(crate) mod pointer;
// The recorded inventory against the live one
#[cfg(test)]
mod rin_checks;
// Checkpoints, so seeks are fast in long replays
pub(crate) mod checkpoint;
// Replay QA fixes: the world border's clock.
mod nw_rq;
// Replays recorded on a 1.7.10 connection (protocol 5).
mod v5;

use std::path::Path;
use std::time::{Duration, Instant};

use mc_audio::{AttenuationType, CapturedSound, PlayRequest, ResolvedVariant, SoundCategory};
use mc_math::DVec3;
use mc_physics::state::MoveInput;
use mc_protocol::connection::LoginOutcome;
use mc_protocol::play::S2CPlay;
use mc_protocol::PacketBuffer;
use mc_render::input::keys as bind;
use mc_render::{Camera, InputState, KeyBindings};
use mc_replay::{BlockCache, Record, RecordStore, ReplayMeta, ReplayTime, StreamId};

use self::clock::{ClockSample, FrameAnchor, ReplayClock};
use self::codec::{
    EventRecord, FrameRecord, KeyRecord, PlayerStateRecord, SessionRecord, SoundRecord, TickRecord,
    FRAME_MIN_INTERVAL_MICROS, GLINT_CLOCK_PERIOD_MS, KIND_EVENT, KIND_FRAME, KIND_SESSION,
    KIND_SOUND, KIND_TICK, RECORDED_BINDINGS,
};
use super::super::{Client, Ui};
use crate::net::ToGame;
use crate::session::Session;

/// The slowest and fastest playback speeds.
pub(crate) const SPEED_RANGE: (f32, f32) = (0.1, 8.0);
/// A seek that has not finished after this much work in one frame continues next frame, so the
/// window keeps drawing (and can show progress) during a long rebuild.
const SEEK_SLICE: Duration = Duration::from_millis(45);
/// Free camera: blocks per second at the default speed.
const FREE_CAMERA_SPEED: f32 = 10.0;
/// Free camera: the speed range the mouse wheel moves through.
const FREE_CAMERA_SPEED_RANGE: (f32, f32) = (0.5, 200.0);
/// Free camera: seconds for the velocity to close `1 - 1/e` of the gap to the input's.
const FREE_CAMERA_SMOOTHING_SECS: f32 = 0.12;
/// The version of the `Client(u8)` record layout this build writes (`meta.tags["rb.version"]`).
pub(crate) const RECORD_FORMAT_VERSION: &str = "1";

// ---------------------------------------------------------------------------------------------
// The lane's state on `Client`
// ---------------------------------------------------------------------------------------------

/// The field on [`Client`] (`tw_play`).
#[derive(Debug, Default)]
pub(crate) struct ReplayPlayState {
    /// `Some` while a replay is being watched. Boxed: one pointer on `Client` when idle.
    pub(crate) playback: Option<Box<Playback>>,
    /// The recording side: what this tick and frame write.
    capture: Capture,
    /// The recorded mouse pointer is not drawn over recorded screens (the
    /// fidelity harness compares replayed frames with live ones, which never contain it).
    pub(crate) pointer_hidden: bool,
    /// A display-only Aero mod menu of a replay's is in `mods.closing_menu`. Here,
    /// not in `Playback::gui`, so a checkpoint restore (which copies that) cannot lose it.
    pub(crate) menu_shown: bool,
    /// The recorded pointer, rasterised once per size, and the window's
    /// display scaling that sizes it.
    pub(crate) pointer: pointer::PointerCache,
}

/// Viewer camera modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplayCameraMode {
    /// The recorded view, exactly: the player's eyes, F5 as the player used it, the HUD.
    #[default]
    Pov,
    /// Behind the player (vanilla's F5 back view), no HUD.
    ThirdPerson,
    /// A free-flying camera: WASD + mouse, Space/Shift up/down, the wheel sets the speed.
    Free,
}

/// One recorded sound, for the exporter's offline mix.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplaySound {
    /// When the recording client started it.
    pub time: ReplayTime,
    /// The sound event, e.g. `minecraft:game.player.hurt`.
    pub event: String,
    /// The `.ogg` the recording chose (`None` if the master fader was at zero then).
    pub file: Option<String>,
    /// The event's category (`None` if unknown).
    pub category: Option<SoundCategory>,
    /// `ISound.getVolume()`.
    pub volume: f32,
    /// `ISound.getPitch()`.
    pub pitch: f32,
    /// The chosen variant's own volume multiplier.
    pub variant_volume: f32,
    /// The chosen variant's own pitch multiplier.
    pub variant_pitch: f32,
    /// World position, or `None` for a non-positional sound (GUI, music).
    pub position: Option<DVec3>,
    /// Linear distance attenuation (`false` = none).
    pub attenuation_linear: bool,
    /// Music/records: decoded as a stream.
    pub streaming: bool,
}

impl ReplaySound {
    fn from_record(time: ReplayTime, record: &SoundRecord) -> Self {
        Self {
            time,
            event: record.event.clone(),
            file: record.file.clone(),
            category: SoundCategory::ALL
                .get(usize::from(record.category))
                .copied(),
            volume: record.volume,
            pitch: record.pitch,
            variant_volume: record.variant_volume,
            variant_pitch: record.variant_pitch,
            position: record.position,
            attenuation_linear: record.attenuation_linear,
            streaming: record.streaming,
        }
    }

    /// The engine's form, for [`mc_audio::AudioEngine::play_captured`].
    pub fn to_captured(&self) -> Option<CapturedSound> {
        let event = mc_assets::ResourceLocation::parse(&self.event).ok()?;
        let variant = match &self.file {
            Some(file) => Some(ResolvedVariant {
                location: mc_assets::ResourceLocation::parse(file).ok()?,
                volume: self.variant_volume,
                pitch: self.variant_pitch,
                streaming: self.streaming,
            }),
            None => None,
        };
        Some(CapturedSound {
            request: PlayRequest {
                event,
                volume: self.volume,
                pitch: self.pitch,
                position: self.position,
                attenuation: if self.attenuation_linear {
                    AttenuationType::Linear
                } else {
                    AttenuationType::None
                },
                repeat: false,
                repeat_delay_ticks: 0,
            },
            category: self.category,
            variant,
        })
    }
}

/// A camera pose, for the exporter's audio listener.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplayCameraPose {
    /// The eye.
    pub pos: DVec3,
    /// Degrees, Minecraft convention.
    pub yaw: f32,
    /// Degrees, positive looking down.
    pub pitch: f32,
    /// The effective vertical FOV in degrees.
    pub fov_degrees: f32,
}

/// Fidelity and work counters for one playback, for the viewer's debug line and the proof.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaybackStats {
    /// Tick records re-run.
    pub ticks: u64,
    /// Ticks whose re-simulated player state differed from the recorded one (and was
    /// overwritten). Zero means the re-simulation matched the recording bit for bit.
    pub player_mismatches: u64,
    /// Ticks that had a player state to compare.
    pub player_ticks: u64,
    /// Packets applied.
    pub packets: u64,
    /// Packets skipped on purpose (keep-alive, compression, disconnect, resource-pack prompts).
    pub packets_skipped: u64,
    /// Packets that failed to decode.
    pub packet_errors: u64,
    /// Sounds played from the recording.
    pub sounds: u64,
    /// World rebuilds (backward seeks).
    pub rebuilds: u64,
    /// Outbound packets the re-run game code produced in the current world, all dropped by the
    /// replay's network handle (it has no socket).
    pub outbound_dropped: u64,
}

// ---------------------------------------------------------------------------------------------
// Recording side
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Capture {
    /// The recorder generation the last [`SessionRecord`] was written for.
    session_generation: Option<u64>,
    /// This tick's `(input allowed, key bindings)`, taken before input handling.
    tick_keys: Option<(bool, KeyRecord)>,
    /// The movement input the tick ran with (captured inside `Client::tick`).
    tick_input: Option<MoveInput>,
    /// The player's state right after its own tick.
    tick_player: Option<PlayerStateRecord>,
    /// When the last frame record was written, for [`FRAME_MIN_INTERVAL_MICROS`].
    last_frame_at: Option<Instant>,
    /// The last F1 state written, so a toggle writes an [`EventRecord`].
    hud_hidden: Option<bool>,
    /// Scratch for draining the audio engine's capture.
    sounds: Vec<CapturedSound>,
    /// The reused encode buffer: recording allocates nothing per record once it has grown.
    scratch: Vec<u8>,
    /// The glint clock the last session or frame record carried.
    last_glint: Option<u64>,
    /// The vignette value (bits) the last frame record carried.
    last_vignette: Option<u32>,
    /// Whether this recording pinned the glint clock (undone when capturing stops).
    glint_pinned: bool,
    /// An in-memory recorder used when no file recording exists: tests and the fidelity harness.
    shim: Option<ShimRecorder>,
    /// What the screen and inventory records last wrote (`gui.rs`).
    gui: gui::GuiCapture,
}

/// The in-memory recorder behind [`Client::replay_debug_start_capture`]. It stamps records
/// exactly as the writer does: one time per frame, taken at the start of the frame.
#[derive(Debug)]
struct ShimRecorder {
    started: Instant,
    frame_time: ReplayTime,
    generation: u64,
    records: Vec<Record>,
}

// ---------------------------------------------------------------------------------------------
// Playback side
// ---------------------------------------------------------------------------------------------

/// A replay being watched.
#[derive(Debug)]
pub(crate) struct Playback {
    meta: ReplayMeta,
    /// Every record, packed (no allocation per record): about a third less
    /// memory than a `Vec<Record>` (`mc_replay::RecordStore`). Opened from a file, large packet
    /// payloads stay in the file's compressed blocks (see [`mw_in_file_enabled`]).
    records: RecordStore,
    /// The block that in-file payloads are being read from.
    block_cache: BlockCache,
    clock: ReplayClock,
    /// `(client_tick, time)` of every tick record, in order.
    tick_times: Vec<(u64, ReplayTime)>,
    /// The visible window (a clip's, or the whole file after its state prefix).
    view: (ReplayTime, ReplayTime),
    /// The next record to apply.
    cursor: usize,
    /// The client tick of the last tick record applied.
    applied_ticks: u64,
    /// The replay time the world is at.
    time: ReplayTime,
    /// The clock at [`Self::time`].
    sample: Option<ClockSample>,
    playing: bool,
    speed: f32,
    last_wall: Option<Instant>,
    /// Packets pushed into the session and not yet polled.
    packets_pending: bool,
    /// The tick record being re-run, read by [`player_ticked`].
    running_tick: Option<TickRecord>,
    /// The last re-run tick's movement input and held bindings, for the Keystrokes HUD.
    last_keys: Option<(MoveInput, u32)>,
    /// The bindings re-driven each tick.
    keys: KeyBindings,
    seek: Option<SeekJob>,
    camera_mode: ReplayCameraMode,
    free: FreeCamera,
    /// The frame's camera-mode overrides, undone at the end of the frame.
    frame_overrides: Option<(u8, bool)>,
    /// The last rendered camera, for the exporter's audio listener.
    last_camera: Option<ReplayCameraPose>,
    restore: Restore,
    stats: PlaybackStats,
    /// The recorded screen as of [`Self::time`] (`gui.rs`; plain data, so a
    /// checkpoint can copy it).
    pub(crate) gui: gui::GuiPlayback,
    /// This replay's checkpoints (`checkpoint.rs`); they go with it.
    pub(crate) rc: checkpoint::Checkpoints,
}

/// A seek in progress.
#[derive(Debug, Clone, Copy)]
struct SeekJob {
    target: ReplayTime,
    /// Ticks to run from where the seek started, for the progress fraction.
    from_ticks: u64,
    to_ticks: u64,
    started: Instant,
}

/// Client state a replay changes and closing it puts back.
#[derive(Debug)]
struct Restore {
    chat: mc_ui::screens::chat::ChatLog,
    third_person_view: u8,
    hide_gui: bool,
    smooth_camera: bool,
}

/// The free camera.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FreeCamera {
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    velocity: DVec3,
    speed: f32,
    placed: bool,
}

impl Default for FreeCamera {
    fn default() -> Self {
        Self {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            velocity: DVec3::ZERO,
            speed: FREE_CAMERA_SPEED,
            placed: false,
        }
    }
}

/// What the free camera's keys say this frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct FreeInput {
    forward: f32,
    strafe: f32,
    up: f32,
    fast: bool,
}

impl FreeCamera {
    /// One frame of movement: `dt` seconds of `input`, eased toward the target velocity.
    fn step(&mut self, input: FreeInput, dt: f32) {
        let dt = dt.clamp(0.0, 0.25);
        let (sin_yaw, cos_yaw) = f64::from(self.yaw).to_radians().sin_cos();
        // Minecraft's convention: yaw 0 looks along +Z, 90 along -X.
        let forward = DVec3::new(-sin_yaw, 0.0, cos_yaw);
        let left = DVec3::new(cos_yaw, 0.0, sin_yaw);
        let mut wish = forward * f64::from(input.forward)
            + left * f64::from(input.strafe)
            + DVec3::new(0.0, f64::from(input.up), 0.0);
        let length = wish.length();
        if length > 1.0 {
            wish /= length;
        }
        let speed = f64::from(self.speed) * if input.fast { 3.0 } else { 1.0 };
        let target = wish * speed;
        let blend = 1.0 - (-f64::from(dt) / f64::from(FREE_CAMERA_SMOOTHING_SECS)).exp();
        self.velocity += (target - self.velocity) * blend;
        self.pos += self.velocity * f64::from(dt);
    }

    fn wheel(&mut self, notches: f32) {
        if notches != 0.0 {
            self.speed = (self.speed * 1.25f32.powf(notches))
                .clamp(FREE_CAMERA_SPEED_RANGE.0, FREE_CAMERA_SPEED_RANGE.1);
        }
    }
}

/// What one non-tick record asks playback to do, decoded before the record's borrow ends.
enum Work {
    Packet(Result<S2CPlay, String>),
    /// A packet of a 1.7.10 recording (`v5.rs`).
    PacketV5(Result<mc_protocol::v5::S2CPlay5, String>),
    Sound(SoundRecord),
    Session(SessionRecord),
    Event(EventRecord),
    /// The inventory your clicks changed, or the open screen (`gui.rs`).
    Gui(gui::GuiRecord),
    /// The end of a recorded frame: what the recording's frame observed happens now.
    Frame,
    Nothing,
}

/// Whether a replay opened from a file leaves its large packet payloads in the
/// file's compressed blocks (chunk data is about four fifths of a recording's record bytes and
/// compresses about tenfold) and reads each back when it is applied.
/// `MC_W21_MW_REPLAY_IN_FILE=0` holds every record in memory instead (read once).
fn mw_in_file_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MC_W21_MW_REPLAY_IN_FILE").map_or(true, |v| v != "0"))
}

/// Packet payloads this large or larger stay in the file.
const MW_IN_FILE_MIN_BYTES: usize = 1024;

/// Index a replay's records: the clock and the tick times.
fn index_records(records: &RecordStore) -> (ReplayClock, Vec<(u64, ReplayTime)>) {
    // Sized exactly: a long replay has over a million frame records, and
    // a doubling `Vec` would hold up to twice their 48 bytes each.
    let (frame_count, tick_count) =
        records
            .iter()
            .fold((0, 0), |(frames, ticks), record| match record.stream {
                StreamId::Client(KIND_FRAME) => (frames + 1, ticks),
                StreamId::Client(KIND_TICK) => (frames, ticks + 1),
                _ => (frames, ticks),
            });
    let mut frames = Vec::with_capacity(frame_count);
    let mut ticks = Vec::with_capacity(tick_count);
    // The glint clock: a session record's absolute value, then each frame record's step. The
    // vignette: the last value a frame record carried (vanilla starts it fully dark).
    let mut glint_ms = 0u64;
    let mut vignette = 1.0f32;
    for record in records.iter() {
        match record.stream {
            StreamId::Client(KIND_SESSION) => {
                if let Some(session) = SessionRecord::decode(record.payload) {
                    glint_ms = session.glint_ms % GLINT_CLOCK_PERIOD_MS;
                }
            }
            StreamId::Client(KIND_FRAME) => {
                if let Some(frame) = FrameRecord::decode(record.payload) {
                    glint_ms = (glint_ms + frame.glint_step_ms) % GLINT_CLOCK_PERIOD_MS;
                    if let Some(value) = frame.vignette {
                        vignette = value;
                    }
                    frames.push(FrameAnchor {
                        time: record.time,
                        client_tick: frame.client_tick,
                        partial: frame.partial,
                        yaw: frame.yaw,
                        pitch: frame.pitch,
                        zoom: frame.zoom,
                        flags: frame.flags,
                        glint_ms,
                        vignette,
                    });
                }
            }
            StreamId::Client(KIND_TICK) => {
                if let Some(tick) = TickRecord::decode(record.payload) {
                    ticks.push((tick.client_tick, record.time));
                }
            }
            _ => {}
        }
    }
    (ReplayClock::new(frames), ticks)
}

/// What the first tick count is that `time` needs, when the replay has no frames to clock by:
/// every tick recorded at or before `time`.
fn ticks_by_time(tick_times: &[(u64, ReplayTime)], time: ReplayTime) -> u64 {
    let after = tick_times.partition_point(|(_, at)| *at <= time);
    after.checked_sub(1).map_or(0, |index| tick_times[index].0)
}

impl Playback {
    fn new(meta: ReplayMeta, records: RecordStore, restore: Restore) -> Self {
        let (clock, tick_times) = index_records(&records);
        let last_record = records.last().map_or(0, |record| record.time);
        let (start, end) = meta.window();
        let end = if end == 0 || end > last_record.max(meta.duration) {
            last_record.max(meta.duration)
        } else {
            end
        };
        let start = start.min(end);
        Self {
            meta,
            records,
            block_cache: BlockCache::default(),
            clock,
            tick_times,
            view: (start, end),
            cursor: 0,
            applied_ticks: 0,
            time: 0,
            sample: None,
            playing: false,
            speed: 1.0,
            last_wall: None,
            packets_pending: false,
            running_tick: None,
            last_keys: None,
            keys: KeyBindings::vanilla(),
            seek: None,
            camera_mode: ReplayCameraMode::Pov,
            free: FreeCamera::default(),
            frame_overrides: None,
            last_camera: None,
            restore,
            stats: PlaybackStats::default(),
            gui: gui::GuiPlayback::default(),
            rc: checkpoint::Checkpoints::for_span(end.saturating_sub(start)),
        }
    }

    /// The tick count `time` needs.
    fn target_ticks(&self, time: ReplayTime) -> u64 {
        match self.clock.sample(time) {
            Some(sample) => sample.ticks,
            None => ticks_by_time(&self.tick_times, time),
        }
    }

    /// Where playback is: the world's time, or a running seek's target.
    fn position(&self) -> ReplayTime {
        self.seek.map_or(self.time, |seek| seek.target)
    }

    /// Set the speed, clamped to [`SPEED_RANGE`]; a non-finite value is ignored.
    fn set_speed(&mut self, speed: f32) {
        if speed.is_finite() {
            self.speed = speed.clamp(SPEED_RANGE.0, SPEED_RANGE.1);
        }
    }

    /// Move the replay time on by `wall` seconds of real time at the playback speed (nothing
    /// while paused), stopping at the end of the view.
    fn advance_wall(&mut self, wall: f64) {
        if !self.playing {
            return;
        }
        let advance = (wall * 1e6 * f64::from(self.speed)) as u64;
        self.time = self.time.saturating_add(advance).min(self.view.1);
        if self.time >= self.view.1 {
            self.playing = false;
        }
    }

    /// Where a one-tick step goes: forward, the first moment the next tick is due; backward, the
    /// first moment of the current tick when past it (as a frame step first goes back to the
    /// frame it is past), otherwise of the tick before. `None` when there is nowhere to go.
    fn step_tick_target(&self, forward: bool) -> Option<ReplayTime> {
        let from = self.position();
        let ticks = self.target_ticks(from);
        let target = if forward {
            self.clock.time_of_tick(ticks.saturating_add(1))?
        } else {
            match self.clock.time_of_tick(ticks) {
                Some(start) if start < from => start,
                _ => self
                    .clock
                    .time_of_tick(ticks.saturating_sub(1))
                    .unwrap_or(self.view.0),
            }
        };
        (target != from).then_some(target)
    }

    /// Where a one-frame step goes: forward, the first recorded frame after the current time;
    /// backward, the last recorded frame before it. `None` when there is nowhere to go.
    fn step_frame_target(&self, forward: bool) -> Option<ReplayTime> {
        let from = self.position();
        let frames = self.clock.frames();
        let index = self.clock.frame_at_or_before(from);
        if forward {
            let next = index.map_or(0, |i| i + 1);
            frames.get(next).map(|frame| frame.time)
        } else {
            index.and_then(|i| {
                if frames[i].time < from {
                    Some(frames[i].time)
                } else {
                    i.checked_sub(1).map(|j| frames[j].time)
                }
            })
        }
    }

    /// Re-drive the recorded bindings for one tick.
    fn load_keys(&mut self, record: &KeyRecord) {
        for (index, name) in RECORDED_BINDINGS.iter().enumerate() {
            let held = record.held & (1 << index) != 0;
            let presses = record
                .presses
                .iter()
                .find(|(i, _)| usize::from(*i) == index)
                .map_or(0, |(_, count)| u32::from(*count));
            if let Some(binding) = self.keys.get_mut(name) {
                binding.force_state(held, presses);
            }
        }
    }
}

/// A seed from a JoinGame packet's fields: the same in the recording and in its playback.
fn join_seed(packet: &mc_protocol::play::s2c::JoinGame) -> i64 {
    // FNV-1a over the fields, so every one of them moves the seed.
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01B3);
        }
    };
    mix(&packet.entity_id.to_le_bytes());
    mix(&[
        u8::from(packet.hardcore_mode),
        packet.game_mode,
        packet.dimension as u8,
        packet.difficulty,
        packet.max_players,
    ]);
    mix(packet.level_type.as_bytes());
    hash as i64
}

/// Decode one recorded clientbound packet: id varint, then the body.
fn decode_packet(payload: &[u8]) -> Result<S2CPlay, String> {
    let mut id = 0i32;
    let mut used = 0usize;
    for (index, byte) in payload.iter().enumerate().take(5) {
        id |= i32::from(byte & 0x7F) << (7 * index);
        if byte & 0x80 == 0 {
            used = index + 1;
            break;
        }
    }
    if used == 0 {
        return Err("no packet id".to_owned());
    }
    let mut buf = PacketBuffer::from_slice(&payload[used..]);
    S2CPlay::read(id, &mut buf).map_err(|error| error.to_string())
}

/// Whether playback applies a recorded packet. Transport packets and the ones that would leave
/// the replay (a kick, a server resource pack download) are skipped.
fn replays_packet(packet: &S2CPlay) -> bool {
    !matches!(
        packet,
        S2CPlay::KeepAlive(_)
            | S2CPlay::SetCompression(_)
            | S2CPlay::Disconnect(_)
            | S2CPlay::ResourcePackSend(_)
    )
}

// ---------------------------------------------------------------------------------------------
// The one free function the player tick calls (it holds `&mut Session`)
// ---------------------------------------------------------------------------------------------

/// Called by `Client::tick` right after the player's own tick (`Session::tick_with_held`).
/// Recording: remember the input and the resulting state. Playback: overwrite the re-simulated
/// state with the recorded one and count whether it differed.
pub(crate) fn player_ticked(play: &mut ReplayPlayState, session: &mut Session, input: &MoveInput) {
    if let Some(playback) = play.playback.as_deref_mut() {
        let recorded = playback.running_tick.as_ref().and_then(|tick| tick.player);
        if let (Some(recorded), Some(predictor)) = (recorded, session.predictor_mut()) {
            playback.stats.player_ticks += 1;
            if recorded.apply(&mut predictor.state) {
                playback.stats.player_mismatches += 1;
            }
        }
        return;
    }
    if play.capture.tick_keys.is_some() {
        play.capture.tick_input = Some(*input);
        play.capture.tick_player = session
            .predictor()
            .map(|predictor| PlayerStateRecord::capture(&predictor.state));
    }
}

// ---------------------------------------------------------------------------------------------
// Hooks the driver calls
// ---------------------------------------------------------------------------------------------

impl Client {
    // ---- the recorder adapter: The `self.rec` (or the in-memory recorder) -------------

    /// Whether a recording (or RA's clip buffer) is taking records right now. `recording()` too:
    /// a recording that "Record every game" starts on a JoinGame takes records from that very
    /// packet on, before RA's per-frame `capturing` flag catches up at the next frame.
    fn tw_rb_capturing(&self) -> bool {
        self.rec.capturing() || self.rec.recording() || self.tw_play.capture.shim.is_some()
    }

    /// Bumps whenever a sink starts, so a new [`SessionRecord`] is due.
    fn tw_rb_generation(&self) -> u64 {
        self.tw_play
            .capture
            .shim
            .as_ref()
            .map_or_else(|| self.rec.capture_generation(), |shim| shim.generation)
    }

    /// Whether the file recording is paused (time frozen).
    fn tw_rb_paused(&self) -> bool {
        self.tw_play.capture.shim.is_none() && self.rec.paused()
    }

    /// Encode one of this lane's records into the reused scratch buffer and push it, stamped
    /// with this frame's time.
    fn tw_rb_push_with(&mut self, kind: u8, encode: impl FnOnce(&mut Vec<u8>)) {
        let mut scratch = std::mem::take(&mut self.tw_play.capture.scratch);
        encode(&mut scratch);
        match self.tw_play.capture.shim.as_mut() {
            Some(shim) => shim.records.push(Record {
                time: shim.frame_time,
                stream: StreamId::Client(kind),
                payload: scratch.clone(),
            }),
            None => self.rec.push_client(kind, &scratch),
        }
        self.tw_play.capture.scratch = scratch;
    }

    /// Push one clientbound packet into the in-memory recorder (the fidelity harness only;
    /// the tap does this for real recordings).
    pub fn replay_debug_capture_packet(&mut self, payload: &[u8]) {
        if let Some(shim) = self.tw_play.capture.shim.as_mut() {
            shim.records.push(Record {
                time: shim.frame_time,
                stream: StreamId::Packet,
                payload: payload.to_vec(),
            });
        }
    }

    /// Start the in-memory recorder (tests and the fidelity harness). The recorder is
    /// the real one; this one exists so the playback engine can be proven without a file.
    pub fn replay_debug_start_capture(&mut self) {
        self.tw_play.capture.shim = Some(ShimRecorder {
            started: Instant::now(),
            frame_time: 0,
            generation: 1,
            records: Vec::new(),
        });
        self.tw_play.capture.session_generation = None;
        self.audio.set_capture(true);
    }

    /// Stop the in-memory recorder and take what it recorded.
    pub fn replay_debug_finish_capture(&mut self) -> Vec<Record> {
        self.audio.set_capture(false);
        self.tw_play.capture.glint_pinned = false;
        self.tw_play.capture.last_glint = None;
        mc_render::entity::item::pin_item_glint_clock(None);
        self.tw_play
            .capture
            .shim
            .take()
            .map(|shim| shim.records)
            .unwrap_or_default()
    }

    /// A long replay for seek measurements: `records` played `times` times back to back, each
    /// copy shifted in time and in client ticks so the result is one continuous recording (each
    /// copy starts with its own JoinGame, as a server switch does).
    pub fn replay_debug_repeat_records(records: &[Record], times: usize) -> Vec<Record> {
        let span = records.last().map_or(0, |record| record.time) + 50_000;
        // The tick counters both kinds carry, so each copy starts past the previous one's last
        // frame (a frame may sit up to one tick past the last tick record).
        let ticks = records
            .iter()
            .filter_map(|record| match record.stream {
                StreamId::Client(KIND_TICK) => {
                    TickRecord::decode(&record.payload).map(|tick| tick.client_tick)
                }
                StreamId::Client(KIND_FRAME) => {
                    FrameRecord::decode(&record.payload).map(|frame| frame.client_tick)
                }
                _ => None,
            })
            .fold((u64::MAX, 0u64), |(lo, hi), tick| {
                (lo.min(tick), hi.max(tick))
            });
        let tick_span = ticks.1.saturating_sub(ticks.0) + 1;
        let mut out = Vec::with_capacity(records.len() * times);
        for copy in 0..times as u64 {
            for record in records {
                let mut record = record.clone();
                record.time += copy * span;
                match record.stream {
                    StreamId::Client(KIND_TICK) => {
                        if let Some(mut tick) = TickRecord::decode(&record.payload) {
                            tick.client_tick += copy * tick_span;
                            record.payload = tick.encode();
                        }
                    }
                    StreamId::Client(KIND_FRAME) => {
                        if let Some(mut frame) = FrameRecord::decode(&record.payload) {
                            frame.client_tick += copy * tick_span;
                            record.payload = frame.encode();
                        }
                    }
                    _ => {}
                }
                out.push(record);
            }
        }
        out
    }

    /// The in-memory recorder's records so far.
    pub fn replay_debug_captured(&self) -> &[Record] {
        self.tw_play
            .capture
            .shim
            .as_ref()
            .map_or(&[], |shim| shim.records.as_slice())
    }

    // ---- shell hooks ------------------------------------------------------------------------

    /// `ShellApp::before_ticks`, first thing. Recording: stamp this frame's time. Playback: this
    /// is where the replay's frame happens (records applied, ticks run); returns `true` so the
    /// live poll is skipped.
    pub(crate) fn tw_rb_before_ticks(&mut self) -> bool {
        if let Some(shim) = self.tw_play.capture.shim.as_mut() {
            shim.frame_time = shim.started.elapsed().as_micros() as u64;
        }
        if self.tw_play.playback.is_none() {
            self.tw_rb_sync_audio_capture();
            return false;
        }
        self.tw_rb_playback_frame();
        true
    }

    /// The audio engine copies started sounds exactly while a recording is capturing; the glint
    /// clock goes back to the wall clock when it stops.
    fn tw_rb_sync_audio_capture(&mut self) {
        let capturing = self.tw_rb_capturing();
        if capturing != self.audio.is_capturing() {
            self.audio.set_capture(capturing);
        }
        if !capturing && self.tw_play.capture.glint_pinned {
            self.tw_play.capture.glint_pinned = false;
            self.tw_play.capture.last_glint = None;
            mc_render::entity::item::pin_item_glint_clock(None);
        }
    }

    /// `ShellApp::tick`, first thing: during playback the replay runs its own ticks.
    pub(crate) fn tw_rb_owns_ticks(&self) -> bool {
        self.tw_play.playback.is_some()
    }

    /// `ShellApp::tick`, after the wheel pump and before `handle_game_input`: write down this
    /// tick's game key bindings while recording.
    pub(crate) fn tw_rb_tick_begin(&mut self, keys: &KeyBindings) {
        if !self.tw_rb_capturing() {
            return;
        }
        self.tw_rb_sync_audio_capture();
        let mut record = KeyRecord {
            hotbar_scroll: self.pending_hotbar_scroll,
            ctrl: self.keyboard.modifiers().ctrl,
            ..KeyRecord::default()
        };
        for (index, name) in RECORDED_BINDINGS.iter().enumerate() {
            if let Some(binding) = keys.get(name) {
                if binding.is_down() {
                    record.held |= 1 << index;
                }
                let presses = binding.pending_presses();
                if presses > 0 {
                    record
                        .presses
                        .push((index as u8, presses.min(u32::from(u8::MAX)) as u8));
                }
            }
        }
        self.tw_play.capture.tick_keys = Some((self.ui.allows_user_input(), record));
    }

    /// `ShellApp::tick`, after `Client::tick`: write the tick record while recording.
    pub(crate) fn tw_rb_tick_end(&mut self) {
        let Some((input_allowed, keys)) = self.tw_play.capture.tick_keys.take() else {
            return;
        };
        let input = self.tw_play.capture.tick_input.take();
        let player = self.tw_play.capture.tick_player.take();
        if !self.tw_rb_capturing() {
            return;
        }
        let record = TickRecord {
            client_tick: self.tick,
            input_allowed,
            keys,
            input: input.unwrap_or_default(),
            third_person_view: self.third_person_view,
            held_slot: self.held_slot(),
            player,
        };
        self.tw_rb_push_with(KIND_TICK, |out| record.encode_into(out));
        // After the tick record: the reseed takes effect from the next tick on, in the recording
        // and in its playback alike.
        self.tw_rb_session_record();
        self.tw_rb_drain_sounds();
    }

    /// Write a [`SessionRecord`] when a new sink started (and reseed the particle generator, so
    /// playback can reseed it at the same point in the stream).
    fn tw_rb_session_record(&mut self) {
        let generation = self.tw_rb_generation();
        if self.tw_play.capture.session_generation == Some(generation) {
            return;
        }
        self.tw_play.capture.session_generation = Some(generation);
        self.rec.set_tag("rb.version", RECORD_FORMAT_VERSION);
        let particle_seed = self.particles.rng_mut().next_i64();
        self.particles.rng_mut().set_seed(particle_seed);
        // The frame records' glint steps count from here; the next frame record carries the
        // vignette in full.
        let glint_ms = mc_render::entity::item::item_glint_system_time_ms() % GLINT_CLOCK_PERIOD_MS;
        self.tw_play.capture.last_glint = Some(glint_ms);
        self.tw_play.capture.last_vignette = None;
        let record = SessionRecord {
            client_tick: self.tick,
            particle_seed,
            entity_seed: 0,
            third_person_view: self.third_person_view,
            fov: self.settings.fov,
            view_bobbing: self.settings.view_bobbing,
            hud_hidden: self.hide_gui,
            glint_ms,
        };
        self.tw_play.capture.hud_hidden = Some(self.hide_gui);
        self.tw_rb_push_with(KIND_SESSION, |out| record.encode_into(out));
    }

    /// Move the audio engine's captured sounds into sound records.
    fn tw_rb_drain_sounds(&mut self) {
        let mut sounds = std::mem::take(&mut self.tw_play.capture.sounds);
        self.audio.drain_captured(&mut sounds);
        if !self.tw_rb_paused() {
            for sound in sounds.drain(..) {
                let record = SoundRecord {
                    event: sound.request.event.to_string(),
                    file: sound
                        .variant
                        .as_ref()
                        .map(|variant| variant.location.to_string()),
                    category: sound
                        .category
                        .and_then(|category| SoundCategory::ALL.iter().position(|c| *c == category))
                        .map_or(u8::MAX, |index| index as u8),
                    volume: sound.request.volume,
                    pitch: sound.request.pitch,
                    variant_volume: sound.variant.as_ref().map_or(1.0, |v| v.volume),
                    variant_pitch: sound.variant.as_ref().map_or(1.0, |v| v.pitch),
                    position: sound.request.position,
                    attenuation_linear: sound.request.attenuation == AttenuationType::Linear,
                    streaming: sound.variant.as_ref().is_some_and(|v| v.streaming),
                    repeat: sound.request.repeat,
                    repeat_delay_ticks: sound.request.repeat_delay_ticks,
                };
                self.tw_rb_push_with(KIND_SOUND, |out| record.encode_into(out));
            }
        }
        sounds.clear();
        self.tw_play.capture.sounds = sounds;
    }

    /// `Client::frame`, first thing. Recording: write the frame record. Playback: put the
    /// recorded look and zoom on the player, apply the camera mode, and return the replay's
    /// partial tick (which the rest of the frame renders with).
    pub(crate) fn tw_rb_frame_begin(&mut self, partial_ticks: f32) -> f32 {
        if self.tw_play.playback.is_some() {
            return self.tw_rb_playback_frame_begin(partial_ticks);
        }
        if self.tw_rb_capturing() {
            self.tw_rb_capture_frame(partial_ticks);
        }
        partial_ticks
    }

    fn tw_rb_capture_frame(&mut self, partial_ticks: f32) {
        // One glint clock value for everything this frame draws (hand, hotbar, entities), so
        // the frame record can carry it and playback can draw the same glint.
        let glint = mc_render::entity::item::item_glint_wall_clock_ms() % GLINT_CLOCK_PERIOD_MS;
        mc_render::entity::item::pin_item_glint_clock(Some(glint));
        self.tw_play.capture.glint_pinned = true;
        self.tw_rb_session_record();
        self.tw_rb_drain_sounds();
        if self.tw_play.capture.hud_hidden != Some(self.hide_gui) {
            self.tw_play.capture.hud_hidden = Some(self.hide_gui);
            let event = EventRecord::HudHidden(self.hide_gui);
            self.tw_rb_push_with(KIND_EVENT, |out| event.encode_into(out));
        }
        if self.tw_rb_paused() {
            return;
        }
        let now = Instant::now();
        // The in-memory recorder keeps every frame: the fidelity proof compares them one by one.
        let decimate = self.tw_play.capture.shim.is_none();
        if decimate
            && self.tw_play.capture.last_frame_at.is_some_and(|last| {
                now.duration_since(last) < Duration::from_micros(FRAME_MIN_INTERVAL_MICROS)
            })
        {
            return;
        }
        self.tw_play.capture.last_frame_at = Some(now);
        let (yaw, pitch) = self
            .session
            .as_ref()
            .and_then(Session::predictor)
            .map_or((0.0, 0.0), |predictor| {
                (predictor.state.yaw, predictor.state.pitch)
            });
        let glint_step_ms = self.tw_play.capture.last_glint.map_or(0, |last| {
            (glint + GLINT_CLOCK_PERIOD_MS - last) % GLINT_CLOCK_PERIOD_MS
        });
        self.tw_play.capture.last_glint = Some(glint);
        // The vignette's persistent ease, as this frame begins (it only changes per rendered
        // frame, so playback at another frame rate has to be told).
        let vignette_now = self
            .render
            .as_ref()
            .map(|render| render.overlay.with(|phase| phase.prev_vignette_brightness));
        let vignette = vignette_now
            .filter(|value| self.tw_play.capture.last_vignette != Some(value.to_bits()));
        if let Some(value) = vignette {
            self.tw_play.capture.last_vignette = Some(value.to_bits());
        }
        let record = FrameRecord {
            client_tick: self.tick,
            partial: partial_ticks,
            yaw,
            pitch,
            zoom: self.m3.zoom.divisor,
            flags: if self.hide_gui {
                FrameRecord::FLAG_HUD_HIDDEN
            } else {
                0
            },
            glint_step_ms,
            vignette,
        };
        self.tw_rb_push_with(KIND_FRAME, |out| record.encode_into(out));
    }

    /// `Client::frame`, right after `place_camera` (and the 1.7 animations' eye): the free
    /// camera replaces the view.
    pub(crate) fn tw_rb_camera(&mut self, camera: &mut Camera) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        if playback.camera_mode == ReplayCameraMode::Free {
            if !playback.free.placed {
                playback.free.pos = camera.pos;
                playback.free.yaw = camera.yaw;
                playback.free.pitch = camera.pitch;
                playback.free.placed = true;
            }
            camera.pos = playback.free.pos;
            camera.yaw = playback.free.yaw;
            camera.pitch = playback.free.pitch;
            camera.fov_modifier = 1.0;
            camera.update();
        }
        playback.last_camera = Some(ReplayCameraPose {
            pos: camera.pos,
            yaw: camera.yaw,
            pitch: camera.pitch,
            fov_degrees: camera.effective_fov(),
        });
    }

    /// `Client::frame`, last thing: undo this frame's camera-mode overrides.
    pub(crate) fn tw_rb_frame_end(&mut self) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        let overrides = playback.frame_overrides.take();
        // The other cameras hear from where they look (POV keeps the player's ears).
        let listener = (playback.camera_mode != ReplayCameraMode::Pov)
            .then_some(playback.last_camera)
            .flatten();
        if let Some((third_person_view, hide_gui)) = overrides {
            self.third_person_view = third_person_view;
            self.hide_gui = hide_gui;
        }
        if let Some(camera) = listener {
            self.listener = mc_audio::Listener::interpolated(
                camera.pos,
                camera.pos,
                0.0,
                camera.pitch,
                camera.pitch,
                camera.yaw,
                camera.yaw,
                0.0,
            );
        }
    }

    /// `ShellApp::frame`, before mouse look: while a replay plays the mouse never turns the
    /// player. In the free camera it steers the camera, and the movement keys fly it. Returns
    /// `true` when the look delta was taken (the caller then skips its own mouse look).
    pub(crate) fn tw_rb_shell_frame(
        &mut self,
        input: &mut InputState,
        keys: &KeyBindings,
        grabbed: bool,
        dt: f32,
    ) -> bool {
        let Some(mode) = self
            .tw_play
            .playback
            .as_deref()
            .map(|playback| playback.camera_mode)
        else {
            return false;
        };
        if mode != ReplayCameraMode::Free {
            if grabbed {
                let _ = input.look_delta_raw(self.look_sensitivity(), false);
            }
            return true;
        }
        let down = |name: &str| {
            keys.get(name)
                .is_some_and(|binding| input.is_down(binding.binding()))
        };
        let axis = |positive: bool, negative: bool| f32::from(positive) - f32::from(negative);
        // The keys fly the camera only while it is steering (the cursor grabbed): otherwise they
        // belong to the viewer (Space is play/pause there), and the camera glides to a stop.
        let free_input = if grabbed {
            FreeInput {
                forward: axis(down(bind::FORWARD), down(bind::BACK)),
                strafe: axis(down(bind::LEFT), down(bind::RIGHT)),
                up: axis(down(bind::JUMP), down(bind::SNEAK)),
                fast: down(bind::SPRINT),
            }
        } else {
            FreeInput::default()
        };
        let look = if grabbed {
            Some(input.look_delta_raw(self.look_sensitivity(), false))
        } else {
            None
        };
        // The wheel arrives through `mouse_wheel` (`tw_rb_wheel`): `handle_screen_input` has
        // already drained it by the time this runs.
        let invert = self.settings.invert_mouse;
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return true;
        };
        let free = &mut playback.free;
        if let Some((dx, dy)) = look {
            let mut turn = Camera {
                yaw: free.yaw,
                pitch: free.pitch,
                ..Camera::default()
            };
            turn.turn_mouse(dx, if invert { -dy } else { dy });
            free.yaw = turn.yaw;
            free.pitch = turn.pitch;
        }
        free.step(free_input, dt);
        true
    }

    /// `mouse_wheel`, after the viewer's own wheel hook (the timeline and speed button
    /// take the wheel unless the free camera is flying). While a replay is watched the wheel sets
    /// the free camera's speed in the free camera, and never reaches the recorded player's
    /// hotbar. Returns whether it took the notch (always, while a replay is open).
    pub(crate) fn tw_rb_wheel(&mut self, delta: f32) -> bool {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return false;
        };
        if playback.camera_mode == ReplayCameraMode::Free && delta.is_finite() {
            playback.free.wheel(delta);
        }
        true
    }

    /// `show_screen`, first thing: a replay being watched never opens a gameplay screen. `None`
    /// (vanilla's `displayGuiScreen(null)`, which shows the death screen while the player is
    /// dead) and the screens the recorded game would open (death, sleep, containers, books, sign
    /// editors, chat) become the screen the replay is watched under. The mod menu, the viewer's
    /// own screens and menus pass through.
    pub(crate) fn tw_rb_redirect_screen(&self, screen: Option<Ui>) -> Option<Ui> {
        if self.tw_play.playback.is_none() {
            return screen;
        }
        match screen {
            None
            | Some(
                Ui::GameOver(_)
                | Ui::Sleep(_)
                | Ui::Container(_)
                | Ui::Creative(_)
                | Ui::Book(_)
                | Ui::EditSign(_)
                | Ui::CommandBlock(_)
                | Ui::Chat(_)
                | Ui::Demo(_)
                | Ui::ConfirmServerResourcePack(..),
            ) => Some(self.tw_rb_watching_ui()),
            other => other,
        }
    }

    /// `build_ui`: the in-game HUD's millisecond clock. While a replay is watched it is the
    /// replay's own time, so the HUD's wall-clock animations (the damaged-heart blink, the tab
    /// list, toasts) depend on where the replay is, not on how fast the frames were drawn.
    ///
    /// It is the same per-frame clock the enchantment glint uses: while recording, the wall
    /// clock pinned at the frame's start (and written into the frame record); while watching,
    /// the recorded value. So a replayed frame's HUD matches the recorded frame's exactly.
    pub(crate) fn tw_rb_hud_clock_ms(&self, now_ms: u64) -> u64 {
        if let Some(playback) = self.tw_play.playback.as_deref() {
            return playback
                .sample
                .map_or(playback.time / 1_000, |sample| sample.glint_ms);
        }
        if self.tw_play.capture.glint_pinned {
            return mc_render::entity::item::item_glint_system_time_ms();
        }
        now_ms
    }

    /// `Client::tick`'s death/sleep screen check: never during playback (the viewer is not the
    /// player, and a death screen would take the viewer's input).
    pub(crate) fn tw_rb_blocks_screens(&self) -> bool {
        self.tw_play.playback.is_some()
    }

    /// `react(JoinedGame)`, last thing: while a recording is capturing or a replay plays back,
    /// restart every client-side visual random stream (particles, torch flicker, the enderman
    /// shimmer, texture animation, each entity's own stream) from a seed derived from the join
    /// packet. The recording and its playback then draw the same particle jitter and flicker,
    /// which is what lets a replayed frame match the recorded one pixel for pixel. These streams
    /// only shape visuals; the default (entropy-seeded) behaviour is untouched when neither runs.
    pub(crate) fn tw_rb_world_joined(&mut self, packet: &mc_protocol::play::s2c::JoinGame) {
        if self.tw_play.playback.is_none() && !self.tw_rb_capturing() {
            return;
        }
        let seed = join_seed(packet);
        self.particles
            .rng_mut()
            .set_seed(seed ^ 0x5041_5254_4943_4C45);
        self.flicker_rng = mc_math::JavaRandom::new(seed ^ 0x5DEE_CE66);
        self.enderman_shimmer_rng = mc_math::JavaRandom::new(seed ^ 0x454E_4445_524D_414E);
        self.atlas_animation_rng = mc_math::JavaRandom::new(seed ^ 0x4154_4C41_534C_4956);
        if let Some(session) = self.session.as_mut() {
            session
                .state_mut()
                .entities
                .reseed_random(seed ^ 0x454E_5449_5449_4553);
        }
    }

    // ---- playback internals -----------------------------------------------------------------

    /// One frame of playback, from `before_ticks`.
    fn tw_rb_playback_frame(&mut self) {
        let now = Instant::now();
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        let wall = playback
            .last_wall
            .map_or(0.0, |last| now.duration_since(last).as_secs_f64());
        playback.last_wall = Some(now);
        if playback.seek.is_some() {
            self.tw_rb_seek_slice();
            return;
        }
        playback.advance_wall(wall);
        let time = playback.time;
        self.tw_rb_advance_to(time, true, None);
    }

    /// Apply records until the world is at `time`. With a deadline, stop early once it passes
    /// and return `false`.
    fn tw_rb_advance_to(
        &mut self,
        time: ReplayTime,
        sounds: bool,
        deadline: Option<Instant>,
    ) -> bool {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return true;
        };
        let target_ticks = playback.target_ticks(time);
        let mut checked = 0u32;
        let reached = loop {
            let Some(playback) = self.tw_play.playback.as_deref_mut() else {
                return true;
            };
            let Some(record) = playback.records.get(playback.cursor) else {
                break true;
            };
            if record.stream == StreamId::Client(KIND_TICK) {
                let Some(tick) = TickRecord::decode(record.payload) else {
                    playback.cursor += 1;
                    continue;
                };
                if tick.client_tick > target_ticks {
                    break true;
                }
                // The tick record's time, for the checkpoint taken after it.
                let tick_time = record.time;
                playback.cursor += 1;
                // The packets flushed here and the tick on its time.
                self.nw_rq_replay_clock(tick_time);
                self.tw_rb_flush_packets();
                self.tw_rb_run_tick(tick);
                self.tw_rc_after_tick(tick_time);
            } else {
                if playback.applied_ticks >= target_ticks && record.time > time {
                    break true;
                }
                let record_time = record.time;
                // A 1.7.10 recording's packets are protocol 5 (`v5.rs`).
                let protocol_5 = v5::recorded_version(playback.meta.protocol).is_1_7_10();
                let work = match record.stream {
                    StreamId::Packet if protocol_5 => Work::PacketV5(
                        playback
                            .records
                            .payload_with(playback.cursor, &mut playback.block_cache)
                            .map_err(|error| error.to_string())
                            .and_then(v5::decode_packet_v5),
                    ),
                    // A large packet's bytes are read back from the file.
                    StreamId::Packet => Work::Packet(
                        playback
                            .records
                            .payload_with(playback.cursor, &mut playback.block_cache)
                            .map_err(|error| error.to_string())
                            .and_then(decode_packet),
                    ),
                    StreamId::Client(KIND_SOUND) if sounds => {
                        SoundRecord::decode(record.payload).map_or(Work::Nothing, Work::Sound)
                    }
                    StreamId::Client(KIND_SESSION) => {
                        SessionRecord::decode(record.payload).map_or(Work::Nothing, Work::Session)
                    }
                    StreamId::Client(KIND_EVENT) => {
                        EventRecord::decode(record.payload).map_or(Work::Nothing, Work::Event)
                    }
                    StreamId::Client(KIND_FRAME) => Work::Frame,
                    StreamId::Client(kind @ (gui::KIND_INVENTORY | gui::KIND_SCREEN)) => {
                        gui::GuiRecord::decode(kind, record.payload)
                            .map_or(Work::Nothing, Work::Gui)
                    }
                    _ => Work::Nothing,
                };
                playback.cursor += 1;
                self.nw_rq_replay_clock(record_time);
                match work {
                    Work::Packet(decoded) => self.tw_rb_apply_packet(decoded),
                    Work::PacketV5(decoded) => self.tw_rb_apply_packet_v5(decoded),
                    Work::Sound(sound) => self.tw_rb_play_sound(record_time, &sound),
                    // Stream order: the packets pushed before this record are applied first (a
                    // session record's reseed must land after the JoinGame's, as recorded).
                    Work::Session(session) => {
                        self.tw_rb_flush_packets();
                        self.tw_rb_apply_session(&session);
                    }
                    Work::Event(event) => {
                        self.tw_rb_flush_packets();
                        self.tw_rb_apply_event(&event);
                    }
                    Work::Frame => {
                        self.tw_rb_flush_packets();
                        self.tw_rb_observe_recorded_frame(record_time);
                    }
                    // After the packets before it, as recorded.
                    Work::Gui(record) => {
                        self.tw_rb_flush_packets();
                        self.nw_rb_apply_gui(record);
                    }
                    Work::Nothing => {}
                }
            }
            checked += 1;
            if let Some(deadline) = deadline {
                if checked.is_multiple_of(64) && Instant::now() >= deadline {
                    break false;
                }
            }
        };
        self.tw_rb_flush_packets();
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            if reached {
                playback.time = time;
                playback.sample = playback.clock.sample(time);
            }
        }
        if reached {
            // The frame drawn at `time` reads the border at `time`.
            self.nw_rq_replay_clock(time);
        }
        reached
    }

    fn tw_rb_apply_packet(&mut self, decoded: Result<S2CPlay, String>) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        match decoded {
            Ok(packet) if replays_packet(&packet) => {
                playback.stats.packets += 1;
                playback.packets_pending = true;
                // A recorded container screen is drawn from it.
                if let S2CPlay::OpenWindow(open) = &packet {
                    playback.gui.open_window = Some(open.clone());
                }
                if let Some(session) = self.session.as_mut() {
                    session.push_driven(ToGame::Packet(packet));
                }
            }
            Ok(_) => playback.stats.packets_skipped += 1,
            Err(error) => {
                playback.stats.packet_errors += 1;
                tracing::warn!(%error, "replay: a recorded packet did not decode; skipped");
            }
        }
    }

    /// Apply every pushed packet through the live path, then close any screen a packet opened
    /// (a chest, a book, a sign editor): the viewer is watching, not playing.
    fn tw_rb_flush_packets(&mut self) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        if !std::mem::take(&mut playback.packets_pending) {
            return;
        }
        self.poll_in_place();
        // A 1.7.10 session's packets wait for a tick drain: apply them now, as recorded.
        self.drain_v5_all();
        self.tw_rb_close_game_screens();
    }

    /// The screen a replay is watched under. Return the viewer's `Ui` variant here.
    fn tw_rb_watching_ui(&self) -> Ui {
        Ui::InGame
    }

    /// Put the watching screen back over anything the recording's packets or ticks opened: the
    /// terrain-download screen, containers, books, sign editors, death and sleep screens. The
    /// viewer is watching, not playing; the mod menu and the viewer's own screens are left alone.
    fn tw_rb_close_game_screens(&mut self) {
        if matches!(
            self.ui,
            Ui::Loading
                | Ui::InGame
                | Ui::Container(_)
                | Ui::Creative(_)
                | Ui::Book(_)
                | Ui::EditSign(_)
                | Ui::CommandBlock(_)
                | Ui::Sleep(_)
                | Ui::GameOver(_)
                | Ui::ConfirmServerResourcePack(..)
                | Ui::Chat(_)
                | Ui::Demo(_)
        ) {
            self.ui = self.tw_rb_watching_ui();
        }
    }

    /// Re-run one recorded tick: the recorded bindings through the input code, the recorded
    /// movement input through `Client::tick` (which overwrites the player state, see
    /// [`player_ticked`]), then the recorded perspective and hotbar slot.
    fn tw_rb_run_tick(&mut self, tick: TickRecord) {
        // The client's tick counter is the recording's: tick-phased animations (the portal
        // wobble, block-crack expiry, HUD counters) then run in the same phase.
        self.tick = tick.client_tick.saturating_sub(1);
        // The HUD keeps its own tick counter (`GuiIngame.updateCounter`:
        // the low-health heart shake's seed, the regeneration bounce, the food jitter). It is
        // the recording's too, or a frame would show a shake that depends on how long the
        // client ran before the replay opened (an export at another frame rate, or a second
        // export of the same range, drew different hearts). `Client::tick` adds the one.
        self.hud.update_counter = tick.client_tick.saturating_sub(1) as i32;
        // The look at the start of the tick is the one the recorded input carries.
        if let Some(predictor) = self.session.as_mut().and_then(Session::predictor_mut) {
            predictor.state.yaw = tick.input.yaw;
            predictor.state.pitch = tick.input.pitch;
        }
        // The replayed tick runs as in game, whatever the viewer has open over it.
        let viewer_ui = (!matches!(self.ui, Ui::InGame | Ui::Loading))
            .then(|| std::mem::replace(&mut self.ui, Ui::InGame));
        // ..or under the inventory the recording had open then (game
        // input differs with a screen up). Closed again below with the other game screens.
        if let Some(screen) = self.nw_rb_tick_screen() {
            self.ui = screen;
        }
        self.update_hit_result();
        if let Some(session) = &mut self.session {
            session.update_controller();
        }
        if tick.input_allowed {
            let keys = self.tw_play.playback.as_deref_mut().map(|playback| {
                playback.load_keys(&tick.keys);
                std::mem::replace(&mut playback.keys, KeyBindings::vanilla())
            });
            if let Some(mut keys) = keys {
                self.pending_hotbar_scroll = tick.keys.hotbar_scroll;
                self.handle_game_input(&InputState::new(), &mut keys);
                if let Some(playback) = self.tw_play.playback.as_deref_mut() {
                    playback.keys = keys;
                }
            }
        }
        let input = tick.input;
        let client_tick = tick.client_tick;
        let third_person_view = tick.third_person_view;
        let held_slot = tick.held_slot;
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.last_keys = Some((tick.input, tick.keys.held));
            playback.running_tick = Some(tick);
        }
        Client::tick(self, &input);
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.running_tick = None;
            playback.applied_ticks = client_tick;
            playback.stats.ticks += 1;
        }
        self.third_person_view = third_person_view;
        if self.session.is_some() && self.held_slot() != held_slot {
            self.select_hotbar_slot(held_slot);
        }
        self.tw_rb_close_game_screens();
        if let Some(ui) = viewer_ui {
            self.ui = ui;
        }
    }

    /// What the recording client observed once per rendered frame, replayed at the end of each
    /// recorded frame (not per playback frame): the damaged-heart blink (`GuiIngame`'s health
    /// flash, which reads a millisecond clock). With `build_ui`'s own observation off during
    /// playback, a frame shows exactly the blink the recording showed, however it was reached.
    fn tw_rb_observe_recorded_frame(&mut self, time: ReplayTime) {
        let Some(clock_ms) = self
            .tw_play
            .playback
            .as_deref()
            .and_then(|playback| playback.clock.sample(time))
            .map(|sample| sample.glint_ms)
        else {
            return;
        };
        let hurt_resistant_time = self
            .session
            .as_ref()
            .map_or(0, |session| session.state().local.hurt_resistant_time);
        self.hud
            .observe_player_health(self.player_hud.health, hurt_resistant_time, clock_ms);
    }

    /// `build_ui`: whether the HUD's per-frame health observation is the replay's (above).
    pub(crate) fn tw_rb_drives_hud_health(&self) -> bool {
        self.tw_play.playback.is_some()
    }

    fn tw_rb_apply_session(&mut self, record: &SessionRecord) {
        self.particles.rng_mut().set_seed(record.particle_seed);
        self.hide_gui = record.hud_hidden;
    }

    fn tw_rb_apply_event(&mut self, event: &EventRecord) {
        match event {
            EventRecord::HudHidden(hidden) => self.hide_gui = *hidden,
        }
    }

    fn tw_rb_play_sound(&mut self, time: ReplayTime, record: &SoundRecord) {
        let Some(mut captured) = ReplaySound::from_record(time, record).to_captured() else {
            return;
        };
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.stats.sounds += 1;
        }
        // The viewer's game sound volume and mute: lowering
        // only, and only sounds at or below full volume (a louder one's volume also sets how far
        // it carries, which must not change).
        let gain = self.nw_re_game_gain();
        if gain <= 0.0 {
            return;
        }
        if gain < 1.0 && captured.request.volume <= 1.0 {
            captured.request.volume *= gain;
        }
        if let Err(error) =
            self.audio
                .play_captured(&captured, &self.listener, &self.assets.resources)
        {
            tracing::debug!(%error, event = %record.event, "replay: a recorded sound did not play");
        }
    }

    fn tw_rb_playback_frame_begin(&mut self, partial_ticks: f32) -> f32 {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return partial_ticks;
        };
        let Some(sample) = playback.sample else {
            return partial_ticks;
        };
        let mode = playback.camera_mode;
        if mode != ReplayCameraMode::Pov {
            playback.frame_overrides = Some((self.third_person_view, self.hide_gui));
        }
        let last_keys = playback.last_keys;
        // The Keystrokes HUD shows the recorded player's keys, not the viewer's (whose live
        // bindings `sample_mod_keys` read just before this frame).
        if let Some((input, held)) = last_keys.filter(|_| self.mods.snapshot.keystrokes) {
            self.mods.hud_data.keys = mc_ui::modhud::HudKeys {
                forward: input.forward,
                left: input.left,
                back: input.back,
                right: input.right,
                jump: input.jump,
                sneak: input.sneak,
                attack: held & 1 != 0,
                use_item: held & 2 != 0,
            };
        }
        // The recording's tick counter at this moment (frame-time code reads it).
        if playback.applied_ticks == sample.ticks {
            self.tick = sample.ticks;
        }
        if let Some(predictor) = self.session.as_mut().and_then(Session::predictor_mut) {
            predictor.state.yaw = sample.yaw;
            predictor.state.pitch = sample.pitch;
        }
        self.m3.zoom.divisor = sample.zoom;
        // Enchantment glints draw with the recorded clock (exact on recorded frames), and the
        // vignette eases on from the recorded value, so a frame after a seek, or at another
        // frame rate, starts where the recording's did.
        mc_render::entity::item::pin_item_glint_clock(Some(sample.glint_ms));
        if let Some(render) = self.render.as_ref() {
            render
                .overlay
                .with(|phase| phase.prev_vignette_brightness = sample.vignette);
        }
        if mode != ReplayCameraMode::Pov {
            self.third_person_view = 1;
            self.hide_gui = true;
        }
        sample.partial
    }

    /// Fresh driven session for the replay, logged in as the recording player.
    fn tw_rb_new_session(&mut self) {
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return;
        };
        let uuid = uuid::Uuid::parse_str(&playback.meta.player_uuid)
            .unwrap_or_else(|_| uuid::Uuid::from_u128(0));
        let username = if playback.meta.player_name.is_empty() {
            "Replay".to_owned()
        } else {
            playback.meta.player_name.clone()
        };
        // The recording's version: a 1.7.10 replay plays back through the 1.7.10 session.
        let version = v5::recorded_version(playback.meta.protocol);
        let mut session = Session::with_version(
            crate::net::NetHandle::driven(),
            self.join_settings(),
            version,
        );
        session.push_driven(ToGame::Connected(LoginOutcome {
            uuid,
            username,
            compression_threshold: None,
            encrypted: false,
        }));
        self.session = Some(session);
    }

    /// Throw the replay's world away and start again from the first record (a backward seek).
    fn tw_rb_rebuild(&mut self) {
        self.unload_world();
        // Tick counters and texture animations as at the replay's start.
        self.tw_rc_replay_start();
        // Chat lines from later in the replay must not survive going back to its start.
        self.chat.clear();
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.cursor = 0;
            playback.applied_ticks = 0;
            playback.time = 0;
            playback.sample = None;
            playback.packets_pending = false;
            playback.running_tick = None;
            playback.last_keys = None;
            playback.keys = KeyBindings::vanilla();
            playback.stats.rebuilds += 1;
        }
        // The recorded screen starts afresh with the world, and so does the
        // hearts' damage flash (its counter would otherwise sit ahead of the replayed ticks).
        self.nw_rb_drop_gui();
        self.hud.reset_health_flash();
        self.tw_rb_new_session();
        self.ui = Ui::Loading;
    }

    /// Start a seek to `target` (clamped to the file). Backward seeks rebuild from the start.
    fn tw_rb_begin_seek(&mut self, target: ReplayTime) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        let target = target.min(playback.view.1);
        let backward =
            target < playback.time || playback.target_ticks(target) < playback.applied_ticks;
        // A checkpoint at or before the target replaces the rebuild from the
        // start (and skips most of a long forward seek).
        if !self.tw_rc_restore_for_seek(target, backward) && backward {
            self.tw_rb_rebuild();
        }
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        // Progress counts ticks from where this seek starts (after a rebuild: the first recorded
        // tick, not zero).
        let first_tick = playback
            .tick_times
            .first()
            .map_or(0, |(tick, _)| tick.saturating_sub(1));
        playback.seek = Some(SeekJob {
            target,
            from_ticks: playback.applied_ticks.max(first_tick),
            to_ticks: playback.target_ticks(target),
            started: Instant::now(),
        });
    }

    /// One frame's slice of a seek.
    fn tw_rb_seek_slice(&mut self) {
        let Some(job) = self
            .tw_play
            .playback
            .as_deref()
            .and_then(|playback| playback.seek)
        else {
            return;
        };
        let deadline = Instant::now() + SEEK_SLICE;
        if self.tw_rb_advance_to(job.target, false, Some(deadline)) {
            if let Some(playback) = self.tw_play.playback.as_deref_mut() {
                playback.seek = None;
                playback.last_wall = Some(Instant::now());
                tracing::debug!(
                    target = job.target,
                    ms = job.started.elapsed().as_secs_f64() * 1e3,
                    "replay seek done"
                );
            }
        }
    }

    /// Finish any seek now (blocking): the exporter and tests need the world at the target.
    fn tw_rb_finish_seek(&mut self) {
        while let Some(job) = self
            .tw_play
            .playback
            .as_deref()
            .and_then(|playback| playback.seek)
        {
            if self.tw_rb_advance_to(job.target, false, None) {
                if let Some(playback) = self.tw_play.playback.as_deref_mut() {
                    playback.seek = None;
                    playback.last_wall = Some(Instant::now());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The API for the viewer and the exporter
// ---------------------------------------------------------------------------------------------

impl Client {
    /// Open a replay file and start watching it (paused at the start of its visible window).
    /// Only from the title screen or a menu: not while connected to a server.
    pub fn replay_open(&mut self, path: &Path) -> Result<(), String> {
        let reader = mc_replay::Reader::open(path).map_err(|error| error.to_string())?;
        let meta = reader.meta().clone();
        // Straight into the packed store, under the reader's memory budget.
        // Large packets (chunk data, mostly) stay in the file's compressed
        // blocks, which the store keeps; switched off, the file's bytes go before the playback
        // is built and every record is held.
        let records = if mw_in_file_enabled() {
            RecordStore::read_all_in_file(reader, MW_IN_FILE_MIN_BYTES)
        } else {
            let records = RecordStore::read_all(&reader);
            drop(reader);
            records
        }
        .map_err(|error| error.to_string())?;
        self.replay_open_store(meta, records)
    }

    /// Open a replay from records already in memory (tests, the fidelity harness, the clip
    /// buffer).
    pub fn replay_open_records(
        &mut self,
        meta: ReplayMeta,
        records: Vec<Record>,
    ) -> Result<(), String> {
        self.replay_open_store(meta, RecordStore::from_records(records))
    }

    /// [`Self::replay_open_records`] with the records already packed.
    fn replay_open_store(&mut self, meta: ReplayMeta, records: RecordStore) -> Result<(), String> {
        if self.tw_play.playback.is_some() {
            return Err("a replay is already open".to_owned());
        }
        if self.session.is_some() {
            return Err("leave the server before watching a replay".to_owned());
        }
        if records.is_empty() {
            return Err("the replay is empty".to_owned());
        }
        match self.machine.apply(crate::app::state::Transition::Connect {
            host: "replay".to_owned(),
            port: 0,
        }) {
            Ok(_effects) => {}
            Err(error) => return Err(error.to_string()),
        }
        let restore = Restore {
            chat: self.chat.clone(),
            third_person_view: self.third_person_view,
            hide_gui: self.hide_gui,
            smooth_camera: self.settings.smooth_camera,
        };
        self.audio.set_suppressed(true);
        self.audio.set_capture(false);
        self.tw_play.capture.shim = None;
        self.rec.block_for_playback(true);
        // The replay's chat starts empty, as the recording's did (the player's own chat comes
        // back on close).
        self.chat.clear();
        // No damage flash carried in from the game played before.
        self.hud.reset_health_flash();
        let playback = Playback::new(meta, records, restore);
        let start = playback.view.0;
        self.tw_play.playback = Some(Box::new(playback));
        self.tw_rb_new_session();
        self.ui = Ui::Loading;
        // Tick counters and texture animations as at the replay's start.
        self.tw_rc_replay_start();
        // The state prefix (a clip's lead-in, a recording started mid-world) is applied at once.
        self.tw_rb_begin_seek(start);
        Ok(())
    }

    /// Stop watching and go back to the title screen, with chat, perspective, F1, the smooth
    /// camera and sound as they were before the replay opened.
    pub fn replay_close(&mut self) {
        // No recorded screen outlives the replay.
        self.nw_rb_drop_gui();
        let Some(playback) = self.tw_play.playback.take() else {
            return;
        };
        // Leave through the ordinary path (state machine, world teardown, title screen).
        self.leave_server(false);
        if self.session.is_some() {
            self.unload_world();
        }
        let restore = playback.restore;
        self.chat = restore.chat;
        self.third_person_view = restore.third_person_view;
        self.hide_gui = restore.hide_gui;
        self.settings.smooth_camera = restore.smooth_camera;
        self.m3.zoom.divisor = 1.0;
        self.tab_list_held = false;
        self.spectator_outlines_held = false;
        self.audio.set_suppressed(false);
        self.rec.block_for_playback(false);
        mc_render::entity::item::pin_item_glint_clock(None);
    }

    /// Whether a replay is being watched.
    pub fn replay_is_active(&self) -> bool {
        self.tw_play.playback.is_some()
    }

    /// The open replay's metadata.
    pub fn replay_meta(&self) -> Option<&ReplayMeta> {
        self.tw_play
            .playback
            .as_deref()
            .map(|playback| &playback.meta)
    }

    /// The end of the file's records (µs).
    pub fn replay_duration(&self) -> ReplayTime {
        self.tw_play
            .playback
            .as_deref()
            .map_or(0, |playback| playback.view.1)
    }

    /// The part of the file the timeline shows: a clip's window, or everything after a state
    /// prefix.
    pub fn replay_view_range(&self) -> (ReplayTime, ReplayTime) {
        self.tw_play
            .playback
            .as_deref()
            .map_or((0, 0), |playback| playback.view)
    }

    /// Where playback is (µs). During a seek, the seek's target.
    pub fn replay_time(&self) -> ReplayTime {
        self.tw_play
            .playback
            .as_deref()
            .map_or(0, Playback::position)
    }

    /// Whether playback is running.
    pub fn replay_is_playing(&self) -> bool {
        self.tw_play
            .playback
            .as_deref()
            .is_some_and(|playback| playback.playing)
    }

    /// Play from the current time (from the start of the window if at its end).
    pub fn replay_play(&mut self) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        if playback.time >= playback.view.1 {
            let start = playback.view.0;
            self.tw_rb_begin_seek(start);
        }
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.playing = true;
            playback.last_wall = Some(Instant::now());
        }
    }

    /// Pause.
    pub fn replay_pause(&mut self) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.playing = false;
        }
    }

    /// The playback speed (1.0 = real time).
    pub fn replay_speed(&self) -> f32 {
        self.tw_play
            .playback
            .as_deref()
            .map_or(1.0, |playback| playback.speed)
    }

    /// Set the playback speed, clamped to [`SPEED_RANGE`].
    pub fn replay_set_speed(&mut self, speed: f32) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.set_speed(speed);
        }
    }

    /// Seek anywhere in the file (clamped to the visible window). Forward seeks fast-apply the
    /// records in between; backward seeks rebuild the world from the start. Long seeks finish
    /// over several frames: [`Self::replay_seek_progress`] reports how far along they are.
    pub fn replay_seek(&mut self, time: ReplayTime) {
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return;
        };
        let time = time.clamp(playback.view.0, playback.view.1);
        self.tw_rb_begin_seek(time);
    }

    /// `Some(0.0..=1.0)` while a seek is still running, `None` otherwise.
    pub fn replay_seek_progress(&self) -> Option<f32> {
        let playback = self.tw_play.playback.as_deref()?;
        let seek = playback.seek?;
        let total = seek.to_ticks.saturating_sub(seek.from_ticks).max(1);
        let done = playback.applied_ticks.saturating_sub(seek.from_ticks);
        Some((done as f32 / total as f32).clamp(0.0, 1.0))
    }

    /// Pause and move exactly one tick forward or back.
    pub fn replay_step_tick(&mut self, forward: bool) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        playback.playing = false;
        if let Some(time) = playback.step_tick_target(forward) {
            self.replay_seek(time);
        }
    }

    /// Pause and move to the next or previous recorded frame.
    pub fn replay_step_frame(&mut self, forward: bool) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        playback.playing = false;
        if let Some(time) = playback.step_frame_target(forward) {
            self.replay_seek(time);
        }
    }

    /// The viewer camera mode.
    pub fn replay_camera_mode(&self) -> ReplayCameraMode {
        self.tw_play
            .playback
            .as_deref()
            .map_or(ReplayCameraMode::Pov, |playback| playback.camera_mode)
    }

    /// Switch the viewer camera. Entering the free camera starts it where the view is now.
    pub fn replay_set_camera_mode(&mut self, mode: ReplayCameraMode) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            if mode == ReplayCameraMode::Free && playback.camera_mode != ReplayCameraMode::Free {
                playback.free.placed = false;
                playback.free.velocity = DVec3::ZERO;
            }
            playback.camera_mode = mode;
        }
    }

    /// Put the free camera at a pose (and stop it). The viewer's "reset camera", scripted export
    /// cameras and tests use it; the free camera's own input moves it from there.
    pub fn replay_set_free_camera(&mut self, pos: DVec3, yaw: f32, pitch: f32) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.free.pos = pos;
            playback.free.yaw = yaw;
            playback.free.pitch = pitch.clamp(-90.0, 90.0);
            playback.free.velocity = DVec3::ZERO;
            playback.free.placed = true;
        }
    }

    /// Mouse-wheel notches for the free camera's speed (up = faster), for a viewer that takes
    /// the wheel before the shell hook sees it.
    pub fn replay_free_camera_wheel(&mut self, notches: f32) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            if notches.is_finite() {
                playback.free.wheel(notches);
            }
        }
    }

    /// The free camera's pose and speed (blocks per second).
    pub fn replay_free_camera(&self) -> Option<(DVec3, f32, f32, f32)> {
        let playback = self.tw_play.playback.as_deref()?;
        let free = &playback.free;
        Some((free.pos, free.yaw, free.pitch, free.speed))
    }

    /// Whether the in-game HUD shows: as recorded in POV, never in the other cameras.
    pub fn replay_hud_visible(&self) -> bool {
        self.tw_play
            .playback
            .as_deref()
            .is_none_or(|playback| playback.camera_mode == ReplayCameraMode::Pov && !self.hide_gui)
    }

    /// For the exporter: put the world, the ticks and the camera exactly at `time`, pausing
    /// playback. Returns the partial tick to render with (also what `Client::frame` will use).
    /// Backward times rebuild the world first, so any order of times gives the same frames.
    pub fn replay_render_at(&mut self, time: ReplayTime) -> f32 {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return 0.0;
        };
        playback.playing = false;
        self.tw_rb_begin_seek(time);
        self.tw_rb_finish_seek();
        self.tw_play
            .playback
            .as_deref()
            .and_then(|playback| playback.sample)
            .map_or(0.0, |sample| sample.partial)
    }

    /// Every recorded sound in `from..to` (µs), for the exporter's offline mix.
    pub fn replay_sounds_in(&self, from: ReplayTime, to: ReplayTime) -> Vec<ReplaySound> {
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return Vec::new();
        };
        let first = playback.records.first_at_or_after(from);
        playback
            .records
            .iter_from(first)
            .take_while(|record| record.time < to)
            .filter(|record| record.stream == StreamId::Client(KIND_SOUND))
            .filter_map(|record| {
                SoundRecord::decode(record.payload)
                    .map(|sound| ReplaySound::from_record(record.time, &sound))
            })
            .collect()
    }

    /// Every recorded microphone record (the encoded frames) in `from..to` (µs), for the
    /// exporter's mix. Decode with `mc_replay::mic::decode_record`, which returns the
    /// lead: the frame's first sample is at `time - lead`.
    pub fn replay_mic_in(&self, from: ReplayTime, to: ReplayTime) -> Vec<(ReplayTime, &[u8])> {
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return Vec::new();
        };
        let first = playback.records.first_at_or_after(from);
        playback
            .records
            .iter_from(first)
            .take_while(|record| record.time < to)
            .filter(|record| record.stream == StreamId::Mic)
            .map(|record| (record.time, record.payload))
            .collect()
    }

    /// Every desktop audio record (the microphone's layout) in
    /// `from..to` (µs), for the exporter's mix and the viewer's voice playback. Decode with
    /// `mc_replay::desktop::decode_record`.
    pub fn replay_desktop_in(&self, from: ReplayTime, to: ReplayTime) -> Vec<(ReplayTime, &[u8])> {
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return Vec::new();
        };
        let first = playback.records.first_at_or_after(from);
        playback
            .records
            .iter_from(first)
            .take_while(|record| record.time < to)
            .filter(|record| record.stream == StreamId::Desktop)
            .map(|record| (record.time, record.payload))
            .collect()
    }

    /// The recorded first-person camera at `time` (eye position and look), from the tick and
    /// frame records alone: the exporter's audio listener for the POV camera.
    pub fn replay_camera_at(&self, time: ReplayTime) -> Option<ReplayCameraPose> {
        let playback = self.tw_play.playback.as_deref()?;
        let sample = playback.clock.sample(time)?;
        // The tick record for `sample.ticks` holds that tick's prev/current feet.
        let index = playback
            .tick_times
            .partition_point(|(tick, _)| *tick < sample.ticks);
        let (tick, at) = *playback.tick_times.get(index)?;
        if tick != sample.ticks {
            return None;
        }
        let record = playback
            .records
            .iter_from(playback.records.first_at_or_after(at))
            .take_while(|record| record.time == at)
            .filter(|record| record.stream == StreamId::Client(KIND_TICK))
            .filter_map(|record| TickRecord::decode(record.payload))
            .find(|record| record.client_tick == tick)?;
        let player = record.player?;
        let t = f64::from(sample.partial);
        let feet = player.prev_pos + (player.pos - player.prev_pos) * t;
        let eye_height = if player.sneaking { 1.54 } else { 1.62 };
        Some(ReplayCameraPose {
            pos: feet + DVec3::new(0.0, eye_height, 0.0),
            yaw: sample.yaw,
            pitch: sample.pitch,
            fov_degrees: self.settings.fov / sample.zoom.max(1e-3),
        })
    }

    /// The camera the last rendered replay frame used (any camera mode).
    pub fn replay_last_camera(&self) -> Option<ReplayCameraPose> {
        self.tw_play.playback.as_deref()?.last_camera
    }

    /// Whether every dirty terrain section has been meshed and uploaded. An exporter renders a
    /// frame only once this holds (with `set_replay_deterministic_mesh(true)`, each frame meshes
    /// a fixed batch synchronously), so no exported frame shows missing chunks.
    pub fn replay_terrain_settled(&self) -> bool {
        self.mesh_queue_len() == 0
            && self.mesh_queue_deferred_len() == 0
            && self.mesh_queue_in_flight_len() == 0
            && self
                .render
                .as_ref()
                .is_none_or(|render| render.terrain.borrow().pending_ready_upload_count() == 0)
    }

    /// Fidelity and work counters for the open replay.
    pub fn replay_stats(&self) -> Option<PlaybackStats> {
        let mut stats = self
            .tw_play
            .playback
            .as_deref()
            .map(|playback| playback.stats)?;
        stats.outbound_dropped = self
            .session
            .as_ref()
            .and_then(Session::driven_dropped)
            .unwrap_or(0);
        Some(stats)
    }

    /// The recorded frames' times (µs), for tests and the fidelity harness.
    pub fn replay_frame_times(&self) -> Vec<ReplayTime> {
        self.tw_play
            .playback
            .as_deref()
            .map_or_else(Vec::new, |playback| {
                playback
                    .clock
                    .frames()
                    .iter()
                    .map(|frame| frame.time)
                    .collect()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_free_camera_eases_toward_its_input_and_stops() {
        let mut camera = FreeCamera::default();
        let forward = FreeInput {
            forward: 1.0,
            ..FreeInput::default()
        };
        for _ in 0..120 {
            camera.step(forward, 1.0 / 60.0);
        }
        // Yaw 0 looks along +Z.
        assert!(camera.pos.z > 15.0, "{:?}", camera.pos);
        assert!(camera.pos.x.abs() < 1e-9 && camera.pos.y.abs() < 1e-9);
        assert!((camera.velocity.z - f64::from(FREE_CAMERA_SPEED)).abs() < 0.1);
        for _ in 0..120 {
            camera.step(FreeInput::default(), 1.0 / 60.0);
        }
        assert!(camera.velocity.length() < 1e-3, "it glides to a stop");
        camera.wheel(3.0);
        assert!(camera.speed > FREE_CAMERA_SPEED * 1.9);
        camera.wheel(-100.0);
        assert_eq!(camera.speed, FREE_CAMERA_SPEED_RANGE.0);
    }

    #[test]
    fn packets_that_would_leave_the_replay_are_skipped() {
        use mc_protocol::play::s2c;
        assert!(!replays_packet(&S2CPlay::KeepAlive(s2c::KeepAlive {
            id: 1
        })));
        assert!(!replays_packet(&S2CPlay::Disconnect(s2c::Disconnect {
            reason: "{}".to_owned()
        })));
        assert!(replays_packet(&S2CPlay::Chat(s2c::Chat {
            component: "{}".to_owned(),
            position: 0
        })));
    }

    #[test]
    fn recorded_packets_decode_from_id_and_body() {
        use mc_protocol::play::s2c;
        let packet = S2CPlay::Chat(s2c::Chat {
            component: r#"{"text":"hi"}"#.to_owned(),
            position: 1,
        });
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        let decoded = decode_packet(buf.as_slice()).expect("decodes");
        assert!(matches!(decoded, S2CPlay::Chat(chat) if chat.position == 1));
        assert!(decode_packet(&[]).is_err());
    }

    fn record(time: ReplayTime, kind: u8, payload: Vec<u8>) -> Record {
        Record {
            time,
            stream: StreamId::Client(kind),
            payload,
        }
    }

    fn tick(client_tick: u64) -> Vec<u8> {
        TickRecord {
            client_tick,
            input_allowed: true,
            keys: KeyRecord::default(),
            input: MoveInput::default(),
            third_person_view: 0,
            held_slot: 0,
            player: None,
        }
        .encode()
    }

    fn frame(client_tick: u64, partial: f32) -> Vec<u8> {
        FrameRecord {
            client_tick,
            partial,
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
            flags: 0,
            glint_step_ms: 16,
            vignette: None,
        }
        .encode()
    }

    /// A small recording: three frames per tick, packets ahead of each frame's ticks.
    fn recording() -> Vec<Record> {
        let mut out = Vec::new();
        for f in 0..30u64 {
            let time = f * 16_667;
            out.push(Record {
                time,
                stream: StreamId::Packet,
                payload: vec![0x02, 0, 0],
            });
            let virtual_ticks = time as f64 / 50_000.0;
            let ticks = virtual_ticks.floor() as u64;
            if f > 0 && ticks > (((f - 1) * 16_667) as f64 / 50_000.0).floor() as u64 {
                out.push(record(time, KIND_TICK, tick(ticks)));
            }
            out.push(record(
                time,
                KIND_FRAME,
                frame(ticks, (virtual_ticks - virtual_ticks.floor()) as f32),
            ));
        }
        out
    }

    #[test]
    fn the_index_finds_every_frame_and_tick() {
        let records = recording();
        let (clock, ticks) = index_records(&RecordStore::from_records(records));
        assert_eq!(clock.frames().len(), 30);
        assert_eq!(ticks.len(), 9, "{ticks:?}");
        // At each recorded frame the clock asks for exactly the ticks recorded before it.
        for anchor in clock.frames() {
            let sample = clock.sample(anchor.time).expect("sample");
            let recorded = ticks.iter().filter(|(_, at)| *at <= anchor.time).count() as u64;
            assert_eq!(sample.ticks, recorded);
        }
    }

    #[test]
    fn a_repeated_recording_is_one_continuous_timeline() {
        let records = recording();
        let long = Client::replay_debug_repeat_records(&records, 3);
        assert_eq!(long.len(), records.len() * 3);
        assert!(long.windows(2).all(|pair| pair[0].time <= pair[1].time));
        let (clock, ticks) = index_records(&RecordStore::from_records(long));
        assert_eq!(
            clock.frames().len(),
            90,
            "no frame dropped as going backwards"
        );
        assert!(
            ticks.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "{ticks:?}"
        );
    }

    #[test]
    fn ticks_by_time_takes_every_tick_recorded_by_then() {
        let ticks = [(10, 0), (11, 0), (12, 50_000), (13, 100_000)];
        assert_eq!(ticks_by_time(&ticks, 0), 11);
        assert_eq!(ticks_by_time(&ticks, 99_999), 12);
        assert_eq!(ticks_by_time(&ticks, 1_000_000), 13);
        assert_eq!(ticks_by_time(&[], 5), 0);
    }

    fn watching(records: Vec<Record>) -> Playback {
        Playback::new(
            ReplayMeta::default(),
            RecordStore::from_records(records),
            Restore {
                chat: mc_ui::screens::chat::ChatLog::new(),
                third_person_view: 0,
                hide_gui: false,
                smooth_camera: false,
            },
        )
    }

    #[test]
    fn time_controls_play_at_speed_pause_and_step() {
        let mut playback = watching(recording());
        let frames: Vec<ReplayTime> = playback.clock.frames().iter().map(|f| f.time).collect();
        assert_eq!(playback.view, (0, frames[29]));

        // Speed: clamped to 0.1x..8x; a non-finite value is ignored.
        playback.set_speed(100.0);
        assert_eq!(playback.speed, SPEED_RANGE.1);
        playback.set_speed(0.0);
        assert_eq!(playback.speed, SPEED_RANGE.0);
        playback.set_speed(f32::NAN);
        assert_eq!(playback.speed, SPEED_RANGE.0);
        playback.set_speed(2.0);

        // Paused, real time does not move the replay; at 2x, 125 ms of it is 250 ms of replay.
        playback.advance_wall(1.0);
        assert_eq!(playback.time, 0);
        playback.playing = true;
        playback.advance_wall(0.125);
        assert_eq!(playback.time, 250_000);
        // It stops at the end.
        playback.advance_wall(10.0);
        assert_eq!(playback.time, playback.view.1);
        assert!(!playback.playing);

        // One tick forward from mid-tick 2: the first moment tick 3 is due.
        playback.time = frames[7];
        assert_eq!(playback.target_ticks(playback.time), 2);
        assert_eq!(
            playback.step_tick_target(true),
            playback.clock.time_of_tick(3)
        );
        // Back from mid-tick: this tick's start first, then the tick before's.
        let tick_2 = playback.clock.time_of_tick(2).expect("tick 2");
        assert!(tick_2 < frames[7]);
        assert_eq!(playback.step_tick_target(false), Some(tick_2));
        playback.time = tick_2;
        assert_eq!(playback.target_ticks(tick_2), 2);
        assert_eq!(
            playback.step_tick_target(false),
            playback.clock.time_of_tick(1)
        );
        // Nowhere to go past either end.
        playback.time = frames[29];
        assert_eq!(playback.step_tick_target(true), None);
        playback.time = 0;
        assert_eq!(playback.step_tick_target(false), None);

        // One frame: forward to the next recorded frame; back to the frame the time is past, or
        // the one before when exactly on a frame.
        playback.time = frames[7];
        assert_eq!(playback.step_frame_target(true), Some(frames[8]));
        assert_eq!(playback.step_frame_target(false), Some(frames[6]));
        playback.time = frames[7] + 5;
        assert_eq!(playback.step_frame_target(true), Some(frames[8]));
        assert_eq!(playback.step_frame_target(false), Some(frames[7]));
        playback.time = frames[29];
        assert_eq!(playback.step_frame_target(true), None);
        playback.time = frames[0];
        assert_eq!(playback.step_frame_target(false), None);

        // While a seek runs, steps go from its target (what `replay_time` reports).
        playback.time = 0;
        playback.seek = Some(SeekJob {
            target: frames[7],
            from_ticks: 0,
            to_ticks: 2,
            started: Instant::now(),
        });
        assert_eq!(playback.position(), frames[7]);
        assert_eq!(playback.step_frame_target(true), Some(frames[8]));
    }
}
