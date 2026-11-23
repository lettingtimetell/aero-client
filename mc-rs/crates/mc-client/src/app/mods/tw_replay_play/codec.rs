//! The payloads of the `StreamId::Client(kind)` records.
//!
//! A replay holds the clientbound packets the client applied (the `StreamId::Packet`) plus
//! everything the packets alone cannot reproduce. Those are the records here:
//!
//! | kind | record | when | what |
//! |---|---|---|---|
//! | 1 | [`SessionRecord`] | recording start, every world join | seeds and view settings |
//! | 2 | [`TickRecord`] | after every client tick | that tick's inputs, then the player's state |
//! | 3 | [`FrameRecord`] | every rendered frame (at most one per [`FRAME_MIN_INTERVAL_MICROS`]) | look, partial tick |
//! | 4 | [`SoundRecord`] | every sound the client started | the request and the resolved variant |
//! | 5 | [`EventRecord`] | rare local events | F1, camera, markers |
//!
//! # Encoding
//!
//! Little-endian, with LEB128 varints for counters. Every payload starts with a version byte.
//! Decoders read what they know and ignore trailing bytes, and a payload that ends early decodes
//! the missing tail as defaults, so a newer client can add fields at the end of a record without
//! breaking older replays (and an older client can still play a newer replay's known fields).
//!
//! Floats are stored as their exact bits: playback must reproduce the recorded positions,
//! rotations and partial ticks bit for bit. The file's compressor takes care of the
//! many fields that do not change from one tick to the next.

use mc_math::DVec3;
use mc_physics::state::{MoveInput, PhysicsState};

/// [`SessionRecord`].
pub(crate) const KIND_SESSION: u8 = 1;
/// [`TickRecord`].
pub(crate) const KIND_TICK: u8 = 2;
/// [`FrameRecord`].
pub(crate) const KIND_FRAME: u8 = 3;
/// [`SoundRecord`].
pub(crate) const KIND_SOUND: u8 = 4;
/// [`EventRecord`].
pub(crate) const KIND_EVENT: u8 = 5;

/// Frames closer together than this are not all recorded: a player at 2000 fps would otherwise
/// write 2000 frame records a second. 2 ms keeps every frame up to 500 fps, and above that the
/// look between two kept frames is interpolated (a 2 ms gap is far below what an eye can see).
pub(crate) const FRAME_MIN_INTERVAL_MICROS: u64 = 2_000;

/// The enchantment glint's clock repeats every `lcm(3000, 4873)` ms (its two scroll periods), so
/// recording it modulo this keeps both phases exact in a few bytes.
pub(crate) const GLINT_CLOCK_PERIOD_MS: u64 = 3_000 * 4_873;

// ---------------------------------------------------------------------------------------------
// Bytes
// ---------------------------------------------------------------------------------------------

/// An append-only little-endian writer over a reused buffer, so recording allocates nothing per
/// record once the buffer has grown.
#[derive(Debug)]
pub(crate) struct Writer<'a>(&'a mut Vec<u8>);

impl<'a> Writer<'a> {
    /// Clear `out` and write into it.
    pub(crate) fn reuse(out: &'a mut Vec<u8>) -> Self {
        out.clear();
        Self(out)
    }

    pub(crate) fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    pub(crate) fn bool(&mut self, value: bool) {
        self.0.push(u8::from(value));
    }

    pub(crate) fn varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                self.0.push(byte);
                return;
            }
            self.0.push(byte | 0x80);
        }
    }

    /// A signed varint, zigzag-encoded so small negative numbers stay small.
    pub(crate) fn svarint(&mut self, value: i64) {
        self.varint(((value << 1) ^ (value >> 63)) as u64);
    }

    pub(crate) fn f32(&mut self, value: f32) {
        self.0.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn f64(&mut self, value: f64) {
        self.0.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn i64(&mut self, value: i64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn dvec3(&mut self, value: DVec3) {
        self.f64(value.x);
        self.f64(value.y);
        self.f64(value.z);
    }

    pub(crate) fn str(&mut self, value: &str) {
        self.varint(value.len() as u64);
        self.0.extend_from_slice(value.as_bytes());
    }
}

/// A forgiving reader: past the end, every read returns the type's default and
/// [`Reader::short`] turns true, which is how an older record decodes under a newer layout.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    short: bool,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            short: false,
        }
    }

    /// Whether a read ran past the end of the payload.
    pub(crate) fn short(&self) -> bool {
        self.short
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.at.checked_add(N)?;
        if end > self.bytes.len() {
            self.short = true;
            self.at = self.bytes.len();
            return None;
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.at..end]);
        self.at = end;
        Some(out)
    }

    pub(crate) fn u8(&mut self) -> u8 {
        self.take::<1>().map_or(0, |b| b[0])
    }

    pub(crate) fn bool(&mut self) -> bool {
        self.u8() != 0
    }

    pub(crate) fn varint(&mut self) -> u64 {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let Some([byte]) = self.take::<1>() else {
                return value;
            };
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
        }
        value
    }

    pub(crate) fn svarint(&mut self) -> i64 {
        let raw = self.varint();
        ((raw >> 1) as i64) ^ -((raw & 1) as i64)
    }

    pub(crate) fn f32(&mut self) -> f32 {
        self.take::<4>()
            .map_or(0.0, |b| f32::from_bits(u32::from_le_bytes(b)))
    }

    pub(crate) fn f64(&mut self) -> f64 {
        self.take::<8>()
            .map_or(0.0, |b| f64::from_bits(u64::from_le_bytes(b)))
    }

    pub(crate) fn i64(&mut self) -> i64 {
        self.take::<8>().map_or(0, i64::from_le_bytes)
    }

    pub(crate) fn dvec3(&mut self) -> DVec3 {
        let x = self.f64();
        let y = self.f64();
        let z = self.f64();
        DVec3::new(x, y, z)
    }

    pub(crate) fn str(&mut self) -> String {
        let len = self.varint() as usize;
        let end = self.at.saturating_add(len).min(self.bytes.len());
        if end - self.at < len {
            self.short = true;
        }
        let text = String::from_utf8_lossy(&self.bytes[self.at..end]).into_owned();
        self.at = end;
        text
    }
}

// ---------------------------------------------------------------------------------------------
// SESSION
// ---------------------------------------------------------------------------------------------

/// Written when a recording starts and again at every world join. Carries what playback must set
/// up before the following records mean the same thing they meant while recording.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SessionRecord {
    /// The client's tick counter when this was written.
    pub(crate) client_tick: u64,
    /// The seed the particle system's generator was re-seeded with at this point.
    pub(crate) particle_seed: i64,
    /// The seed the entity tracker's per-entity generators were re-seeded with (0 = not seeded).
    pub(crate) entity_seed: i64,
    /// `thirdPersonView` (0 first person, 1 back, 2 front).
    pub(crate) third_person_view: u8,
    /// The FOV setting, degrees.
    pub(crate) fov: f32,
    /// `viewBobbing`.
    pub(crate) view_bobbing: bool,
    /// F1: the HUD was hidden.
    pub(crate) hud_hidden: bool,
    /// The enchantment-glint clock at this point, modulo [`GLINT_CLOCK_PERIOD_MS`]; frame
    /// records carry the steps from here.
    pub(crate) glint_ms: u64,
}

impl SessionRecord {
    const VERSION: u8 = 1;

    #[cfg(test)]
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(40);
        self.encode_into(&mut out);
        out
    }

    /// [`Self::encode`] into a reused buffer (cleared first).
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        let mut w = Writer::reuse(out);
        w.u8(Self::VERSION);
        w.varint(self.client_tick);
        w.i64(self.particle_seed);
        w.i64(self.entity_seed);
        w.u8(self.third_person_view);
        w.f32(self.fov);
        w.bool(self.view_bobbing);
        w.bool(self.hud_hidden);
        w.varint(self.glint_ms);
    }

    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        if r.u8() == 0 {
            return None;
        }
        Some(Self {
            client_tick: r.varint(),
            particle_seed: r.i64(),
            entity_seed: r.i64(),
            third_person_view: r.u8(),
            fov: r.f32(),
            view_bobbing: r.bool(),
            hud_hidden: r.bool(),
            glint_ms: r.varint(),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// TICK
// ---------------------------------------------------------------------------------------------

/// The game key bindings a tick's input handling reads, in a fixed order. Their index is the bit
/// in [`KeyRecord::held`] and the id in [`KeyRecord::presses`]; append only.
pub(crate) const RECORDED_BINDINGS: [&str; 17] = [
    mc_render::input::keys::ATTACK,
    mc_render::input::keys::USE_ITEM,
    mc_render::input::keys::PICK_ITEM,
    mc_render::input::keys::DROP,
    mc_render::input::keys::PLAYER_LIST,
    mc_render::input::keys::SPECTATOR_OUTLINES,
    mc_render::input::keys::TOGGLE_PERSPECTIVE,
    mc_render::input::keys::SMOOTH_CAMERA,
    "key.hotbar.1",
    "key.hotbar.2",
    "key.hotbar.3",
    "key.hotbar.4",
    "key.hotbar.5",
    "key.hotbar.6",
    "key.hotbar.7",
    "key.hotbar.8",
    "key.hotbar.9",
];

/// The game key bindings at the start of one tick's input handling.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct KeyRecord {
    /// Bit `i` set: [`RECORDED_BINDINGS`]`[i]` was held.
    pub(crate) held: u32,
    /// `(binding index, queued presses)` for every binding with at least one queued press.
    pub(crate) presses: Vec<(u8, u8)>,
    /// Mouse-wheel notches queued for the hotbar this tick.
    pub(crate) hotbar_scroll: i32,
    /// Control was down (Ctrl+Q drops the whole stack).
    pub(crate) ctrl: bool,
}

impl KeyRecord {
    fn encode(&self, w: &mut Writer<'_>) {
        w.varint(u64::from(self.held));
        w.varint(self.presses.len() as u64);
        for (index, count) in &self.presses {
            w.u8(*index);
            w.u8(*count);
        }
        w.svarint(i64::from(self.hotbar_scroll));
        w.bool(self.ctrl);
    }

    fn decode(r: &mut Reader<'_>) -> Self {
        let held = r.varint() as u32;
        let count = (r.varint() as usize).min(RECORDED_BINDINGS.len());
        let mut presses = Vec::with_capacity(count);
        for _ in 0..count {
            let index = r.u8();
            let presses_queued = r.u8();
            presses.push((index, presses_queued));
        }
        Self {
            held,
            presses,
            hotbar_scroll: r.svarint() as i32,
            ctrl: r.bool(),
        }
    }
}

/// [`MoveInput`] as passed to `Client::tick`.
fn encode_move_input(w: &mut Writer<'_>, input: &MoveInput) {
    let bits = u8::from(input.forward)
        | u8::from(input.back) << 1
        | u8::from(input.left) << 2
        | u8::from(input.right) << 3
        | u8::from(input.jump) << 4
        | u8::from(input.sneak) << 5
        | u8::from(input.sprint) << 6
        | u8::from(input.using_item) << 7;
    w.u8(bits);
    w.f32(input.yaw);
    w.f32(input.pitch);
}

fn decode_move_input(r: &mut Reader<'_>) -> MoveInput {
    let bits = r.u8();
    MoveInput {
        forward: bits & 1 != 0,
        back: bits & 2 != 0,
        left: bits & 4 != 0,
        right: bits & 8 != 0,
        jump: bits & 16 != 0,
        sneak: bits & 32 != 0,
        sprint: bits & 64 != 0,
        using_item: bits & 128 != 0,
        yaw: r.f32(),
        pitch: r.f32(),
    }
}

/// The local player's simulated state right after its own tick: everything the camera, the hand,
/// the F5 body, footsteps and the HUD read from `PhysicsState`.
///
/// Playback writes these back over the re-simulated player every tick, so a replay recorded by
/// an older build plays back the recorded movement exactly even after the physics changed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlayerStateRecord {
    pub(crate) pos: DVec3,
    pub(crate) prev_pos: DVec3,
    pub(crate) vel: DVec3,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) fall_distance: f32,
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) jump_movement_factor: f32,
    pub(crate) prev_move_forward: f32,
    pub(crate) move_strafing: f32,
    pub(crate) move_forward: f32,
    pub(crate) distance_walked_modified: f32,
    pub(crate) prev_distance_walked_modified: f32,
    pub(crate) prev_camera_yaw: f32,
    pub(crate) camera_yaw: f32,
    pub(crate) prev_camera_pitch: f32,
    pub(crate) camera_pitch: f32,
    pub(crate) prev_render_arm_yaw: f32,
    pub(crate) render_arm_yaw: f32,
    pub(crate) prev_render_arm_pitch: f32,
    pub(crate) render_arm_pitch: f32,
    pub(crate) distance_walked_on_step_modified: f32,
    pub(crate) limb_swing: f32,
    pub(crate) limb_swing_amount: f32,
    pub(crate) prev_limb_swing_amount: f32,
    pub(crate) jump_ticks: i32,
    pub(crate) sprint_toggle_timer: i32,
    pub(crate) sprinting_ticks_left: i32,
    pub(crate) next_step_distance: i32,
    pub(crate) fly_toggle_timer: i32,
    pub(crate) portal_cooldown: i32,
    pub(crate) on_ground: bool,
    pub(crate) sprinting: bool,
    pub(crate) sneaking: bool,
    pub(crate) collided_horizontally: bool,
    pub(crate) collided_vertically: bool,
    pub(crate) in_water: bool,
    pub(crate) in_web: bool,
    pub(crate) jumping: bool,
    pub(crate) prev_sneak: bool,
    pub(crate) flying: bool,
    pub(crate) prev_jump: bool,
    pub(crate) in_portal: bool,
    pub(crate) first_update: bool,
}

impl PlayerStateRecord {
    /// Snapshot a predictor's state.
    pub(crate) fn capture(state: &PhysicsState) -> Self {
        Self {
            pos: state.pos,
            prev_pos: state.prev_pos,
            vel: state.vel,
            yaw: state.yaw,
            pitch: state.pitch,
            fall_distance: state.fall_distance,
            width: state.width,
            height: state.height,
            jump_movement_factor: state.jump_movement_factor,
            prev_move_forward: state.prev_move_forward,
            move_strafing: state.move_strafing,
            move_forward: state.move_forward,
            distance_walked_modified: state.distance_walked_modified,
            prev_distance_walked_modified: state.prev_distance_walked_modified,
            prev_camera_yaw: state.prev_camera_yaw,
            camera_yaw: state.camera_yaw,
            prev_camera_pitch: state.prev_camera_pitch,
            camera_pitch: state.camera_pitch,
            prev_render_arm_yaw: state.prev_render_arm_yaw,
            render_arm_yaw: state.render_arm_yaw,
            prev_render_arm_pitch: state.prev_render_arm_pitch,
            render_arm_pitch: state.render_arm_pitch,
            distance_walked_on_step_modified: state.distance_walked_on_step_modified,
            limb_swing: state.limb_swing,
            limb_swing_amount: state.limb_swing_amount,
            prev_limb_swing_amount: state.prev_limb_swing_amount,
            jump_ticks: state.jump_ticks,
            sprint_toggle_timer: state.sprint_toggle_timer,
            sprinting_ticks_left: state.sprinting_ticks_left,
            next_step_distance: state.next_step_distance,
            fly_toggle_timer: state.fly_toggle_timer,
            portal_cooldown: state.portal_cooldown,
            on_ground: state.on_ground,
            sprinting: state.sprinting,
            sneaking: state.sneaking,
            collided_horizontally: state.collided_horizontally,
            collided_vertically: state.collided_vertically,
            in_water: state.in_water,
            in_web: state.in_web,
            jumping: state.jumping,
            prev_sneak: state.prev_sneak,
            flying: state.flying,
            prev_jump: state.prev_jump,
            in_portal: state.in_portal,
            first_update: state.first_update,
        }
    }

    /// Write the recorded state over a predictor's. Returns whether anything differed, which is
    /// playback's per-tick fidelity check: with unchanged code the re-simulated player already
    /// matches and this is a no-op.
    ///
    /// `set_position` (not a bare `pos` write) keeps the bounding box with the feet, the same
    /// reason the scripted bench camera uses it.
    pub(crate) fn apply(&self, state: &mut PhysicsState) -> bool {
        let differed = Self::capture(state) != *self;
        if !differed {
            return false;
        }
        if state.pos != self.pos || state.width != self.width || state.height != self.height {
            state.width = self.width;
            state.height = self.height;
            state.set_position(self.pos);
        }
        state.prev_pos = self.prev_pos;
        state.vel = self.vel;
        state.yaw = self.yaw;
        state.pitch = self.pitch;
        state.fall_distance = self.fall_distance;
        state.jump_movement_factor = self.jump_movement_factor;
        state.prev_move_forward = self.prev_move_forward;
        state.move_strafing = self.move_strafing;
        state.move_forward = self.move_forward;
        state.distance_walked_modified = self.distance_walked_modified;
        state.prev_distance_walked_modified = self.prev_distance_walked_modified;
        state.prev_camera_yaw = self.prev_camera_yaw;
        state.camera_yaw = self.camera_yaw;
        state.prev_camera_pitch = self.prev_camera_pitch;
        state.camera_pitch = self.camera_pitch;
        state.prev_render_arm_yaw = self.prev_render_arm_yaw;
        state.render_arm_yaw = self.render_arm_yaw;
        state.prev_render_arm_pitch = self.prev_render_arm_pitch;
        state.render_arm_pitch = self.render_arm_pitch;
        state.distance_walked_on_step_modified = self.distance_walked_on_step_modified;
        state.limb_swing = self.limb_swing;
        state.limb_swing_amount = self.limb_swing_amount;
        state.prev_limb_swing_amount = self.prev_limb_swing_amount;
        state.jump_ticks = self.jump_ticks;
        state.sprint_toggle_timer = self.sprint_toggle_timer;
        state.sprinting_ticks_left = self.sprinting_ticks_left;
        state.next_step_distance = self.next_step_distance;
        state.fly_toggle_timer = self.fly_toggle_timer;
        state.portal_cooldown = self.portal_cooldown;
        state.on_ground = self.on_ground;
        state.sprinting = self.sprinting;
        state.sneaking = self.sneaking;
        state.collided_horizontally = self.collided_horizontally;
        state.collided_vertically = self.collided_vertically;
        state.in_water = self.in_water;
        state.in_web = self.in_web;
        state.jumping = self.jumping;
        state.prev_sneak = self.prev_sneak;
        state.flying = self.flying;
        state.prev_jump = self.prev_jump;
        state.in_portal = self.in_portal;
        state.first_update = self.first_update;
        true
    }

    fn encode(&self, w: &mut Writer<'_>) {
        w.dvec3(self.pos);
        w.dvec3(self.prev_pos);
        w.dvec3(self.vel);
        for value in [
            self.yaw,
            self.pitch,
            self.fall_distance,
            self.width,
            self.height,
            self.jump_movement_factor,
            self.prev_move_forward,
            self.move_strafing,
            self.move_forward,
            self.distance_walked_modified,
            self.prev_distance_walked_modified,
            self.prev_camera_yaw,
            self.camera_yaw,
            self.prev_camera_pitch,
            self.camera_pitch,
            self.prev_render_arm_yaw,
            self.render_arm_yaw,
            self.prev_render_arm_pitch,
            self.render_arm_pitch,
            self.distance_walked_on_step_modified,
            self.limb_swing,
            self.limb_swing_amount,
            self.prev_limb_swing_amount,
        ] {
            w.f32(value);
        }
        for value in [
            self.jump_ticks,
            self.sprint_toggle_timer,
            self.sprinting_ticks_left,
            self.next_step_distance,
            self.fly_toggle_timer,
            self.portal_cooldown,
        ] {
            w.svarint(i64::from(value));
        }
        let bits = [
            self.on_ground,
            self.sprinting,
            self.sneaking,
            self.collided_horizontally,
            self.collided_vertically,
            self.in_water,
            self.in_web,
            self.jumping,
            self.prev_sneak,
            self.flying,
            self.prev_jump,
            self.in_portal,
            self.first_update,
        ]
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, bit)| acc | (u64::from(*bit) << i));
        w.varint(bits);
    }

    fn decode(r: &mut Reader<'_>) -> Self {
        let pos = r.dvec3();
        let prev_pos = r.dvec3();
        let vel = r.dvec3();
        let mut f = [0.0f32; 23];
        for value in &mut f {
            *value = r.f32();
        }
        let mut i = [0i32; 6];
        for value in &mut i {
            *value = r.svarint() as i32;
        }
        let bits = r.varint();
        let bit = |n: u32| bits & (1 << n) != 0;
        Self {
            pos,
            prev_pos,
            vel,
            yaw: f[0],
            pitch: f[1],
            fall_distance: f[2],
            width: f[3],
            height: f[4],
            jump_movement_factor: f[5],
            prev_move_forward: f[6],
            move_strafing: f[7],
            move_forward: f[8],
            distance_walked_modified: f[9],
            prev_distance_walked_modified: f[10],
            prev_camera_yaw: f[11],
            camera_yaw: f[12],
            prev_camera_pitch: f[13],
            camera_pitch: f[14],
            prev_render_arm_yaw: f[15],
            render_arm_yaw: f[16],
            prev_render_arm_pitch: f[17],
            render_arm_pitch: f[18],
            distance_walked_on_step_modified: f[19],
            limb_swing: f[20],
            limb_swing_amount: f[21],
            prev_limb_swing_amount: f[22],
            jump_ticks: i[0],
            sprint_toggle_timer: i[1],
            sprinting_ticks_left: i[2],
            next_step_distance: i[3],
            fly_toggle_timer: i[4],
            portal_cooldown: i[5],
            on_ground: bit(0),
            sprinting: bit(1),
            sneaking: bit(2),
            collided_horizontally: bit(3),
            collided_vertically: bit(4),
            in_water: bit(5),
            in_web: bit(6),
            jumping: bit(7),
            prev_sneak: bit(8),
            flying: bit(9),
            prev_jump: bit(10),
            in_portal: bit(11),
            first_update: bit(12),
        }
    }
}

/// One client tick: what went in, and the player's state that came out.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TickRecord {
    /// The client's tick counter after this tick (`Client::tick`'s `self.tick`).
    pub(crate) client_tick: u64,
    /// Whether the screen up at the start of the tick accepted game input (`InGame`, inventory).
    pub(crate) input_allowed: bool,
    /// The game key bindings at the start of the tick.
    pub(crate) keys: KeyRecord,
    /// The movement input `Client::tick` was given.
    pub(crate) input: MoveInput,
    /// `thirdPersonView` after the tick.
    pub(crate) third_person_view: u8,
    /// The selected hotbar slot after the tick.
    pub(crate) held_slot: u8,
    /// The player's state right after its own tick, if a player existed.
    pub(crate) player: Option<PlayerStateRecord>,
}

impl TickRecord {
    const VERSION: u8 = 1;

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(220);
        self.encode_into(&mut out);
        out
    }

    /// [`Self::encode`] into a reused buffer (cleared first).
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        let mut w = Writer::reuse(out);
        w.u8(Self::VERSION);
        w.varint(self.client_tick);
        w.bool(self.input_allowed);
        self.keys.encode(&mut w);
        encode_move_input(&mut w, &self.input);
        w.u8(self.third_person_view);
        w.u8(self.held_slot);
        match &self.player {
            Some(player) => {
                w.bool(true);
                player.encode(&mut w);
            }
            None => w.bool(false),
        }
    }

    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        if r.u8() == 0 {
            return None;
        }
        let client_tick = r.varint();
        let input_allowed = r.bool();
        let keys = KeyRecord::decode(&mut r);
        let input = decode_move_input(&mut r);
        let third_person_view = r.u8();
        let held_slot = r.u8();
        let player = r.bool().then(|| PlayerStateRecord::decode(&mut r));
        if r.short() {
            return None;
        }
        Some(Self {
            client_tick,
            input_allowed,
            keys,
            input,
            third_person_view,
            held_slot,
            player,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// FRAME
// ---------------------------------------------------------------------------------------------

/// One rendered frame of the recording.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrameRecord {
    /// Client ticks run before this frame rendered.
    pub(crate) client_tick: u64,
    /// The frame's partial tick, exactly.
    pub(crate) partial: f32,
    /// The player's look after this frame's mouse movement (`rotationYaw`).
    pub(crate) yaw: f32,
    /// The player's look after this frame's mouse movement (`rotationPitch`).
    pub(crate) pitch: f32,
    /// The Zoom mod's FOV divisor this frame (1.0 = not zoomed).
    pub(crate) zoom: f32,
    /// Bit 0: F1 (HUD hidden).
    pub(crate) flags: u8,
    /// How far the enchantment-glint clock moved since the previous frame record (or since the
    /// session record), in ms modulo [`GLINT_CLOCK_PERIOD_MS`].
    pub(crate) glint_step_ms: u64,
    /// Vanilla's `GuiIngame.prevVignetteBrightness` as this frame began, when it changed since
    /// the previous frame record (it eases one percent per *rendered frame*, so playback at
    /// another frame rate must be told it). `None` = unchanged.
    pub(crate) vignette: Option<f32>,
}

impl FrameRecord {
    const VERSION: u8 = 1;
    /// [`Self::flags`]: the HUD was hidden (F1).
    pub(crate) const FLAG_HUD_HIDDEN: u8 = 1;

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24);
        self.encode_into(&mut out);
        out
    }

    /// [`Self::encode`] into a reused buffer (cleared first).
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        let mut w = Writer::reuse(out);
        w.u8(Self::VERSION);
        w.varint(self.client_tick);
        w.f32(self.partial);
        w.f32(self.yaw);
        w.f32(self.pitch);
        // The zoom divisor is 1.0 in almost every frame, and the vignette settles once the light
        // stops changing: one flag byte each instead of four.
        let zoomed = self.zoom != 1.0;
        w.u8(self.flags
            | if zoomed { 0x80 } else { 0 }
            | if self.vignette.is_some() { 0x40 } else { 0 });
        if zoomed {
            w.f32(self.zoom);
        }
        w.varint(self.glint_step_ms);
        if let Some(vignette) = self.vignette {
            w.f32(vignette);
        }
    }

    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        if r.u8() == 0 {
            return None;
        }
        let client_tick = r.varint();
        let partial = r.f32();
        let yaw = r.f32();
        let pitch = r.f32();
        let raw_flags = r.u8();
        let zoom = if raw_flags & 0x80 != 0 { r.f32() } else { 1.0 };
        if r.short() {
            return None;
        }
        // A record from before these fields existed decodes them as "no step" and "unchanged".
        let glint_step_ms = r.varint();
        let vignette = if raw_flags & 0x40 != 0 {
            let vignette = r.f32();
            if r.short() {
                return None;
            }
            Some(vignette)
        } else {
            None
        };
        Some(Self {
            client_tick,
            partial,
            yaw,
            pitch,
            zoom,
            flags: raw_flags & 0x3F,
            glint_step_ms,
            vignette,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// SOUND
// ---------------------------------------------------------------------------------------------

/// One sound the recording client started: the request exactly as built, plus the variant the
/// engine's weighted pick chose, so playback and the export mix play the same file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SoundRecord {
    /// The sound event, e.g. `minecraft:game.player.hurt`.
    pub(crate) event: String,
    /// The chosen `.ogg`, e.g. `minecraft:sounds/damage/hit1.ogg`, if a variant was picked (the
    /// master fader at zero skips the pick).
    pub(crate) file: Option<String>,
    /// `SoundCategory` as its `mc_audio` index.
    pub(crate) category: u8,
    /// `ISound.getVolume()`.
    pub(crate) volume: f32,
    /// `ISound.getPitch()`.
    pub(crate) pitch: f32,
    /// The chosen variant's own volume multiplier.
    pub(crate) variant_volume: f32,
    /// The chosen variant's own pitch multiplier.
    pub(crate) variant_pitch: f32,
    /// World position, or `None` for a non-positional sound.
    pub(crate) position: Option<DVec3>,
    /// Linear distance attenuation (else none).
    pub(crate) attenuation_linear: bool,
    /// Decoded incrementally (music, records).
    pub(crate) streaming: bool,
    /// `ISound.canRepeat()`.
    pub(crate) repeat: bool,
    /// `ISound.getRepeatDelay()`.
    pub(crate) repeat_delay_ticks: i32,
}

impl SoundRecord {
    const VERSION: u8 = 1;

    #[cfg(test)]
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(96);
        self.encode_into(&mut out);
        out
    }

    /// [`Self::encode`] into a reused buffer (cleared first).
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        let mut w = Writer::reuse(out);
        w.u8(Self::VERSION);
        w.str(&self.event);
        w.str(self.file.as_deref().unwrap_or(""));
        w.u8(self.category);
        w.f32(self.volume);
        w.f32(self.pitch);
        w.f32(self.variant_volume);
        w.f32(self.variant_pitch);
        let bits = u8::from(self.position.is_some())
            | u8::from(self.attenuation_linear) << 1
            | u8::from(self.streaming) << 2
            | u8::from(self.repeat) << 3;
        w.u8(bits);
        if let Some(position) = self.position {
            w.dvec3(position);
        }
        w.svarint(i64::from(self.repeat_delay_ticks));
    }

    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        if r.u8() == 0 {
            return None;
        }
        let event = r.str();
        let file = r.str();
        let category = r.u8();
        let volume = r.f32();
        let pitch = r.f32();
        let variant_volume = r.f32();
        let variant_pitch = r.f32();
        let bits = r.u8();
        let position = (bits & 1 != 0).then(|| r.dvec3());
        let repeat_delay_ticks = r.svarint() as i32;
        if r.short() {
            return None;
        }
        Some(Self {
            event,
            file: (!file.is_empty()).then_some(file),
            category,
            volume,
            pitch,
            variant_volume,
            variant_pitch,
            position,
            attenuation_linear: bits & 2 != 0,
            streaming: bits & 4 != 0,
            repeat: bits & 8 != 0,
            repeat_delay_ticks,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// EVENT
// ---------------------------------------------------------------------------------------------

/// A rare client-side event. Unknown event ids are skipped by playback.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EventRecord {
    /// F1 toggled: whether the HUD is hidden now.
    HudHidden(bool),
}

impl EventRecord {
    const VERSION: u8 = 1;

    #[cfg(test)]
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4);
        self.encode_into(&mut out);
        out
    }

    /// [`Self::encode`] into a reused buffer (cleared first).
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        let mut w = Writer::reuse(out);
        w.u8(Self::VERSION);
        match self {
            EventRecord::HudHidden(hidden) => {
                w.u8(1);
                w.bool(*hidden);
            }
        }
    }

    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        if r.u8() == 0 {
            return None;
        }
        match r.u8() {
            1 => Some(EventRecord::HudHidden(r.bool())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> PlayerStateRecord {
        let mut state = PhysicsState::new_player(DVec3::new(-3891.5, 74.0, 2059.5));
        state.vel = DVec3::new(0.1, -0.0784000015258789, -0.25);
        state.yaw = 123.456;
        state.pitch = -12.5;
        state.sprinting = true;
        state.on_ground = true;
        state.limb_swing = 7.25;
        state.limb_swing_amount = 0.75;
        state.camera_pitch = 0.0123;
        state.next_step_distance = 9;
        state.jump_ticks = -1;
        state.first_update = false;
        PlayerStateRecord::capture(&state)
    }

    #[test]
    fn varints_round_trip_at_the_edges() {
        let mut buf = vec![0xAA; 3];
        let mut w = Writer::reuse(&mut buf);
        for value in [
            0u64,
            1,
            127,
            128,
            16_383,
            16_384,
            u64::from(u32::MAX),
            u64::MAX,
        ] {
            w.varint(value);
        }
        for value in [
            0i64,
            -1,
            1,
            -64,
            64,
            i64::from(i32::MIN),
            i64::MAX,
            i64::MIN,
        ] {
            w.svarint(value);
        }
        let mut r = Reader::new(&buf);
        for value in [
            0u64,
            1,
            127,
            128,
            16_383,
            16_384,
            u64::from(u32::MAX),
            u64::MAX,
        ] {
            assert_eq!(r.varint(), value);
        }
        for value in [
            0i64,
            -1,
            1,
            -64,
            64,
            i64::from(i32::MIN),
            i64::MAX,
            i64::MIN,
        ] {
            assert_eq!(r.svarint(), value);
        }
        assert!(!r.short());
    }

    #[test]
    fn a_tick_record_round_trips_bit_for_bit() {
        let record = TickRecord {
            client_tick: 123_456,
            input_allowed: true,
            keys: KeyRecord {
                held: 0b1_0000_0001,
                presses: vec![(0, 3), (8, 1)],
                hotbar_scroll: -2,
                ctrl: true,
            },
            input: MoveInput {
                forward: true,
                sprint: true,
                jump: true,
                yaw: 359.99997,
                pitch: -90.0,
                ..MoveInput::default()
            },
            third_person_view: 2,
            held_slot: 7,
            player: Some(sample_state()),
        };
        let bytes = record.encode();
        assert_eq!(TickRecord::decode(&bytes), Some(record.clone()));
        // A negative zero and a NaN-free float must keep their exact bits.
        let mut negative_zero = record.clone();
        negative_zero.input.yaw = -0.0;
        let decoded = TickRecord::decode(&negative_zero.encode()).expect("decodes");
        assert_eq!(decoded.input.yaw.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn a_truncated_tick_record_is_rejected_not_misread() {
        let bytes = TickRecord {
            client_tick: 5,
            input_allowed: false,
            keys: KeyRecord::default(),
            input: MoveInput::default(),
            third_person_view: 0,
            held_slot: 0,
            player: Some(sample_state()),
        }
        .encode();
        assert!(TickRecord::decode(&bytes[..bytes.len() - 3]).is_none());
        // A future version that appends fields still decodes the known prefix.
        let mut longer = bytes.clone();
        longer.extend_from_slice(&[1, 2, 3, 4]);
        assert!(TickRecord::decode(&longer).is_some());
    }

    #[test]
    fn applying_a_state_reports_whether_it_changed_anything() {
        let recorded = sample_state();
        let mut live = PhysicsState::new_player(DVec3::ZERO);
        assert!(
            recorded.apply(&mut live),
            "a different state is overwritten"
        );
        assert_eq!(PlayerStateRecord::capture(&live), recorded);
        assert!(!recorded.apply(&mut live), "the same state is a no-op");
        // The bounding box moved with the feet (set_position), so the box test matches.
        let mut moved = PhysicsState::new_player(recorded.pos);
        moved.width = recorded.width;
        moved.height = recorded.height;
        moved.set_position(recorded.pos);
        assert_eq!(live.aabb(), moved.aabb());
    }

    #[test]
    fn frame_records_are_small_and_exact() {
        let plain = FrameRecord {
            client_tick: 1_500,
            partial: 0.123_456_79,
            yaw: -45.5,
            pitch: 12.25,
            zoom: 1.0,
            flags: 0,
            glint_step_ms: 7,
            vignette: None,
        };
        let bytes = plain.encode();
        assert!(bytes.len() <= 17, "{} bytes", bytes.len());
        assert_eq!(FrameRecord::decode(&bytes), Some(plain));
        let zoomed = FrameRecord {
            zoom: 4.0,
            flags: FrameRecord::FLAG_HUD_HIDDEN,
            vignette: Some(0.731_234_5),
            ..plain
        };
        assert_eq!(FrameRecord::decode(&zoomed.encode()), Some(zoomed));
        // A truncated vignette is corruption, not "unchanged".
        let bytes = zoomed.encode();
        assert_eq!(FrameRecord::decode(&bytes[..bytes.len() - 1]), None);
    }

    #[test]
    fn sound_and_session_and_event_records_round_trip() {
        let sound = SoundRecord {
            event: "minecraft:game.player.hurt".into(),
            file: Some("minecraft:sounds/damage/hit2.ogg".into()),
            category: 6,
            volume: 1.0,
            pitch: 0.912,
            variant_volume: 1.0,
            variant_pitch: 1.0,
            position: Some(DVec3::new(1.5, 64.0, -2.25)),
            attenuation_linear: true,
            streaming: false,
            repeat: false,
            repeat_delay_ticks: 0,
        };
        assert_eq!(SoundRecord::decode(&sound.encode()), Some(sound.clone()));
        let ambient = SoundRecord {
            file: None,
            position: None,
            attenuation_linear: false,
            ..sound
        };
        assert_eq!(SoundRecord::decode(&ambient.encode()), Some(ambient));

        let session = SessionRecord {
            client_tick: 99,
            particle_seed: -7,
            entity_seed: 0x1234_5678_9ABC,
            third_person_view: 1,
            fov: 90.0,
            view_bobbing: true,
            hud_hidden: false,
            glint_ms: GLINT_CLOCK_PERIOD_MS - 1,
        };
        assert_eq!(SessionRecord::decode(&session.encode()), Some(session));
        let event = EventRecord::HudHidden(true);
        assert_eq!(EventRecord::decode(&event.encode()), Some(event));
        assert_eq!(
            EventRecord::decode(&[1, 200]),
            None,
            "unknown events are skipped"
        );
    }
}
