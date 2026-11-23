//! Client-side prediction and server-correction reconciliation.
//!
//! The client simulates the local player itself and tells the server where it ended up
//! (`C03PacketPlayer` and friends, sent from `EntityPlayerSP.onUpdateWalkingPlayer`). When the
//! server disagrees it sends `S08PacketPlayerPosLook`, and the client must **snap to it and
//! confirm**, because until the confirmation arrives the server discards everything the client
//! says.
//!
//! # What vanilla actually does, and what this adds
//!
//! `NetHandlerPlayClient.handlePlayerPosLook` is blunt: it zeroes `motionX/Y/Z`, applies the
//! (optionally relative) position and rotation, and replies with a
//! `C06PacketPlayerPosLook` carrying the corrected values. There is no replay and no smoothing —
//! 1.8.9 rubber-bands visibly, and matching that is the point.
//!
//! [`Predictor`] keeps that behaviour as [`Predictor::apply_correction`] and adds an optional
//! **replay** path ([`Predictor::apply_correction_with_replay`]) for callers that also track the
//! tick each outbound packet belonged to. Replay re-runs the inputs the server had not yet seen
//! on top of the corrected state, which removes the visible snap-back for corrections caused by
//! latency rather than by a real disagreement. It is strictly additive: the packet the client
//! sends back is identical either way, so an anticheat sees no difference.
//!
//! # Packet cadence
//!
//! Hazard H11 is S14's, but the decision belongs here because it is a function of physics
//! state: [`Predictor::outbound`] reproduces `onUpdateWalkingPlayer`'s choice between the four
//! `C03` shapes, including the `9.0E-4` movement threshold and the forced full update every
//! twentieth tick.

use std::collections::VecDeque;

use mc_math::DVec3;
use mc_world::World;

use crate::collision::{CollisionBorder, EntityCollisionCandidate};
use crate::player::{tick_player_with_entities_and_border_view, tick_player_with_entities_view};
use crate::state::{MoveInput, PhysicsState};

/// A server position correction. java: `S08PacketPlayerPosLook`.
///
/// The `relative_*` flags are the packet's bit field: a set flag means the corresponding
/// component is a *delta* to add to the client's current value rather than an absolute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionCorrection {
    /// X, absolute or relative per `relative_x`.
    pub x: f64,
    /// Y — the **feet**, matching `PhysicsState::pos`.
    pub y: f64,
    /// Z, absolute or relative per `relative_z`.
    pub z: f64,
    /// Yaw in degrees.
    pub yaw: f32,
    /// Pitch in degrees.
    pub pitch: f32,
    /// java: flag bit `0x01`.
    pub relative_x: bool,
    /// java: flag bit `0x02`.
    pub relative_y: bool,
    /// java: flag bit `0x04`.
    pub relative_z: bool,
    /// java: flag bit `0x08`.
    pub relative_yaw: bool,
    /// java: flag bit `0x10`.
    pub relative_pitch: bool,
}

impl PositionCorrection {
    /// Decodes the packet's `flags` byte into the five booleans.
    /// java: `S08PacketPlayerPosLook.EnumFlags`.
    #[must_use]
    pub fn from_packet(x: f64, y: f64, z: f64, yaw: f32, pitch: f32, flags: u8) -> Self {
        Self {
            x,
            y,
            z,
            yaw,
            pitch,
            relative_x: flags & 0x01 != 0,
            relative_y: flags & 0x02 != 0,
            relative_z: flags & 0x04 != 0,
            relative_yaw: flags & 0x08 != 0,
            relative_pitch: flags & 0x10 != 0,
        }
    }
}

/// What the client should put on the wire this tick.
/// java: the four branches of `EntityPlayerSP.onUpdateWalkingPlayer`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outbound {
    /// `C03PacketPlayer` — nothing changed, just the ground flag.
    Ground {
        /// java: `onGround`.
        on_ground: bool,
    },
    /// `C04PacketPlayerPosition`.
    Position {
        /// Feet position.
        pos: DVec3,
        /// java: `onGround`.
        on_ground: bool,
    },
    /// `C05PacketPlayerLook`.
    Look {
        /// Degrees.
        yaw: f32,
        /// Degrees.
        pitch: f32,
        /// java: `onGround`.
        on_ground: bool,
    },
    /// `C06PacketPlayerPosLook`.
    PositionLook {
        /// Feet position.
        pos: DVec3,
        /// Degrees.
        yaw: f32,
        /// Degrees.
        pitch: f32,
        /// java: `onGround`.
        on_ground: bool,
    },
}

/// One simulated tick kept for replay.
#[derive(Debug, Clone, Copy)]
struct HistoryEntry {
    tick: u64,
    input: MoveInput,
    /// Whether this tick's `EntityPlayerSP.updateEntityActionState` was allowed to copy the raw
    /// movement input. Replays must retain the foreign-camera gate or their action fields drift.
    view_is_local: bool,
    /// State *before* the tick ran, so replaying from here reproduces it.
    before: PhysicsState,
}

/// Drives [`tick_player`], decides what to send, and reconciles server corrections.
#[derive(Debug, Clone)]
pub struct Predictor {
    /// The player's current predicted state.
    pub state: PhysicsState,
    tick: u64,
    history: VecDeque<HistoryEntry>,
    history_limit: usize,
    last_reported: DVec3,
    last_reported_yaw: f32,
    last_reported_pitch: f32,
    position_update_ticks: u32,
    /// How many corrections have been applied. Useful as a desync counter in the F3 overlay.
    pub corrections: u64,
    /// java: `Entity.outsideBorder`, retained across collision queries and ticks.
    ///
    /// The bit belongs to the predicted local entity rather than to the border packet: vanilla
    /// uses it as a one-block hysteresis latch when choosing between real blocks and stone.
    outside_border: bool,
}

impl Predictor {
    /// A predictor starting from `state`, keeping 100 ticks (5 seconds) of replay history.
    #[must_use]
    pub fn new(state: PhysicsState) -> Self {
        Self {
            state,
            tick: 0,
            history: VecDeque::new(),
            history_limit: 100,
            // java: `EntityPlayerSP.lastReportedPosX/Y/Z` and `lastReportedYaw/Pitch` are plain
            // fields with Java's default `0.0`/`0.0F` (`client/entity/EntityPlayerSP.java:64`).
            // An `f64::MAX` sentinel would look harmless — the first tick reports either way,
            // thanks to `position_update_ticks >= 20` — but it changes the *shape*: with zeros a
            // player spawning at yaw 0 has `d3 == d4 == 0`, so vanilla's first packet is a
            // `C04PacketPlayerPosition`, not the `C06` a sentinel forces. Hazard H11 is about
            // exactly that sequence, so the sentinel goes.
            last_reported: DVec3::ZERO,
            last_reported_yaw: 0.0,
            last_reported_pitch: 0.0,
            position_update_ticks: 0,
            corrections: 0,
            outside_border: false,
        }
    }

    /// The tick number the next [`Predictor::tick`] will run.
    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Simulates one tick and returns the packet the client should send.
    ///
    /// The returned [`Outbound`] carries the tick number implicitly: it is
    /// [`Predictor::current_tick`] *before* the call. Record it alongside the packet if you
    /// intend to use [`Predictor::apply_correction_with_replay`].
    pub fn tick(&mut self, w: &World, input: &MoveInput) -> Outbound {
        self.tick_with_entities(w, input, &[])
    }

    /// Simulates one tick with nearby entity collision candidates from the current client world.
    ///
    /// The candidate slice is intentionally an input of this one tick instead of Predictor state:
    /// tracker positions are mutable world data and must be sampled by the caller at the same
    /// moment it samples input. The block-only [`Predictor::tick`] wrapper passes an empty slice.
    pub fn tick_with_entities(
        &mut self,
        w: &World,
        input: &MoveInput,
        entity_collisions: &[EntityCollisionCandidate],
    ) -> Outbound {
        self.tick_with_entities_view(w, input, entity_collisions, true)
    }

    /// Simulates one tick with nearby entity boxes and the `isCurrentViewEntity()` gate.
    pub fn tick_with_entities_view(
        &mut self,
        w: &World,
        input: &MoveInput,
        entity_collisions: &[EntityCollisionCandidate],
        view_is_local: bool,
    ) -> Outbound {
        self.history.push_back(HistoryEntry {
            tick: self.tick,
            input: *input,
            view_is_local,
            before: self.state,
        });
        while self.history.len() > self.history_limit {
            self.history.pop_front();
        }

        tick_player_with_entities_view(w, &mut self.state, input, entity_collisions, view_is_local);
        self.tick += 1;
        self.outbound()
    }

    /// Simulates one tick with nearby entities and the current client world-border projection.
    ///
    /// `CollisionBorder` is sampled by the client caller for this tick. The predictor retains
    /// the vanilla `Entity.outsideBorder` latch so repeated movement queries during one sweep and
    /// subsequent ticks share the same hysteresis state.
    pub fn tick_with_entities_and_border(
        &mut self,
        w: &World,
        input: &MoveInput,
        entity_collisions: &[EntityCollisionCandidate],
        border: CollisionBorder,
    ) -> Outbound {
        self.tick_with_entities_and_border_view(w, input, entity_collisions, border, true)
    }

    /// Simulates one tick with nearby entities, a live border projection and the
    /// `isCurrentViewEntity()` gate.
    pub fn tick_with_entities_and_border_view(
        &mut self,
        w: &World,
        input: &MoveInput,
        entity_collisions: &[EntityCollisionCandidate],
        border: CollisionBorder,
        view_is_local: bool,
    ) -> Outbound {
        self.history.push_back(HistoryEntry {
            tick: self.tick,
            input: *input,
            view_is_local,
            before: self.state,
        });
        while self.history.len() > self.history_limit {
            self.history.pop_front();
        }

        tick_player_with_entities_and_border_view(
            w,
            &mut self.state,
            input,
            entity_collisions,
            border,
            &mut self.outside_border,
            view_is_local,
        );
        self.tick += 1;
        self.outbound()
    }

    /// Simulates one tick without consuming the movement-report cadence.
    ///
    /// java: `EntityPlayerSP.onUpdateWalkingPlayer` is guarded by
    /// `isCurrentViewEntity()`, but the entity's ordinary physics still runs for a spectator
    /// camera target.  A foreign render-view entity therefore advances prediction/history while
    /// leaving `lastReportedPos*`, `lastReportedYaw/Pitch` and the forced-20-tick counter alone.
    /// The caller emits only the view-independent `C0BPacketEntityAction` edges.
    pub fn tick_with_entities_and_border_unreported(
        &mut self,
        w: &World,
        input: &MoveInput,
        entity_collisions: &[EntityCollisionCandidate],
        border: CollisionBorder,
    ) {
        self.tick_with_entities_and_border_unreported_view(
            w,
            input,
            entity_collisions,
            border,
            true,
        );
    }

    /// Simulates one tick without consuming movement-report cadence, selecting the
    /// `isCurrentViewEntity()` gate independently of report suppression.
    pub fn tick_with_entities_and_border_unreported_view(
        &mut self,
        w: &World,
        input: &MoveInput,
        entity_collisions: &[EntityCollisionCandidate],
        border: CollisionBorder,
        view_is_local: bool,
    ) {
        self.history.push_back(HistoryEntry {
            tick: self.tick,
            input: *input,
            view_is_local,
            before: self.state,
        });
        while self.history.len() > self.history_limit {
            self.history.pop_front();
        }

        tick_player_with_entities_and_border_view(
            w,
            &mut self.state,
            input,
            entity_collisions,
            border,
            &mut self.outside_border,
            view_is_local,
        );
        self.tick += 1;
    }

    /// The jump and travel half of a mounted player's living update, after the caller ran
    /// [`crate::player::mounted_on_living_update_preflight`] on [`Predictor::state`].
    ///
    /// java: `Entity.updateRidden` zeroes motion and runs the full `onUpdate()`, so the rider's
    /// own `moveEntityWithHeading` still sweeps the world (border latch included) before
    /// `updateRiderPosition` replaces the position. Like the preflight, this neither records
    /// replay history nor touches the walking report cadence: a riding client sends no position.
    pub fn mounted_living_travel_with_entities_and_border(
        &mut self,
        w: &World,
        entity_collisions: &[EntityCollisionCandidate],
        border: CollisionBorder,
    ) {
        crate::player::mounted_living_travel(
            w,
            &mut self.state,
            entity_collisions,
            Some(border),
            &mut self.outside_border,
        );
    }

    /// java: `EntityPlayerSP.onUpdateWalkingPlayer` (`EntityPlayerSP.java:189`).
    ///
    /// The position threshold is on the **squared** distance and is `9.0E-4` — a hair over
    /// `0.03` blocks — and a full position update is forced every 20 ticks regardless, which is
    /// the cadence anticheats key off (hazard H11).
    pub fn outbound(&mut self) -> Outbound {
        let d0 = self.state.pos.x - self.last_reported.x;
        let d1 = self.state.pos.y - self.last_reported.y;
        let d2 = self.state.pos.z - self.last_reported.z;
        let d3 = f64::from(self.state.yaw - self.last_reported_yaw);
        let d4 = f64::from(self.state.pitch - self.last_reported_pitch);
        let moved = d0 * d0 + d1 * d1 + d2 * d2 > 9.0e-4 || self.position_update_ticks >= 20;
        let turned = d3 != 0.0 || d4 != 0.0;

        let out = match (moved, turned) {
            (true, true) => Outbound::PositionLook {
                pos: self.state.pos,
                yaw: self.state.yaw,
                pitch: self.state.pitch,
                on_ground: self.state.on_ground,
            },
            (true, false) => Outbound::Position {
                pos: self.state.pos,
                on_ground: self.state.on_ground,
            },
            (false, true) => Outbound::Look {
                yaw: self.state.yaw,
                pitch: self.state.pitch,
                on_ground: self.state.on_ground,
            },
            (false, false) => Outbound::Ground {
                on_ground: self.state.on_ground,
            },
        };

        self.position_update_ticks += 1;
        if moved {
            self.last_reported = self.state.pos;
            self.position_update_ticks = 0;
        }
        if turned {
            self.last_reported_yaw = self.state.yaw;
            self.last_reported_pitch = self.state.pitch;
        }
        out
    }

    /// Applies a server correction exactly as vanilla does, and returns the corrected state as
    /// an [`Outbound::PositionLook`] — **not** literally the `C06PacketPlayerPosLook` bytes to
    /// send back; see the `on_ground` warning below before wiring this return value to the wire.
    ///
    /// java: `NetHandlerPlayClient.handlePlayerPosLook` — motion is zeroed on each absolute
    /// position axis, preserved on each relative axis, relative components are added to the
    /// current value, and the reply carries the *corrected* values.
    ///
    /// **The reply is mandatory.** A server that does not receive it treats every subsequent
    /// movement packet as stale and keeps re-sending the correction, which is what a
    /// "teleport loop" looks like from the client side.
    ///
    /// **`on_ground` is a trap for a caller building the actual wire confirmation.** This
    /// method's return carries the real, physical `on_ground` — correct for a general
    /// [`Outbound`] consumer — but `handlePlayerPosLook:717` hard-codes the *wire* confirmation's
    /// `onGround` to `false` unconditionally, regardless of the real state. `mc-client` does not
    /// use this return value to build the confirm packet for exactly that reason; it calls its
    /// own `outbound::confirm_position` instead, which hard-codes `false` the way vanilla does.
    /// A future caller reading only this doc and not that call site would send the wrong value.
    pub fn apply_correction(&mut self, c: &PositionCorrection) -> Outbound {
        if !c.relative_x {
            self.state.vel.x = 0.0;
        }
        if !c.relative_y {
            self.state.vel.y = 0.0;
        }
        if !c.relative_z {
            self.state.vel.z = 0.0;
        }
        // java: `handlePlayerPosLook` ends in `setPositionAndRotation`, and
        // `Entity.setPositionAndRotation` calls `setPosition`, which rebuilds the bounding box.
        // Assigning `pos` alone is NOT enough: the box is authoritative and `pos` is *derived*
        // from it every tick by `resetPositionToBB`, so a stale box silently undoes the teleport
        // on the very next `tick_player`. Found on 2026-08-15 against a real
        // `S08PacketPlayerPosLook`; `correcting_the_position_also_moves_the_bounding_box` pins it.
        self.state.set_position(DVec3::new(
            if c.relative_x {
                self.state.pos.x + c.x
            } else {
                c.x
            },
            if c.relative_y {
                self.state.pos.y + c.y
            } else {
                c.y
            },
            if c.relative_z {
                self.state.pos.z + c.z
            } else {
                c.z
            },
        ));
        // java: `Entity.setRotation` stores each component with Java's `% 360.0F` remainder.
        // Rust's `%` has the same truncating-toward-zero sign behavior; `rem_euclid` would turn
        // negative corrections into positive angles and diverge from 1.8.9.
        self.state.yaw = (if c.relative_yaw {
            self.state.yaw + c.yaw
        } else {
            c.yaw
        }) % 360.0;
        self.state.pitch = (if c.relative_pitch {
            self.state.pitch + c.pitch
        } else {
            c.pitch
        }) % 360.0;
        // java: `NetHandlerPlayClient.handlePlayerPosLook` delegates to
        // `Entity.setPositionAndRotation`, which does not write `fallDistance`. A correction
        // therefore preserves an in-progress fall just as it preserves the other entity state
        // not named by the packet.
        self.history.clear();
        self.corrections += 1;

        // The confirmation is always absolute and always the full pos+look shape.
        Outbound::PositionLook {
            pos: self.state.pos,
            yaw: self.state.yaw,
            pitch: self.state.pitch,
            on_ground: self.state.on_ground,
        }
    }

    /// [`Predictor::apply_correction`], then re-simulates every tick the server had not yet
    /// seen when it sent the correction.
    ///
    /// `acknowledged_tick` is the tick number of the last movement packet the correction
    /// accounts for — with the packet-id-free 1.8 protocol that is an estimate from your own
    /// RTT, which is exactly why this is opt-in and vanilla does not do it.
    ///
    /// Returns the confirmation packet (built from the *corrected* position, before the replay,
    /// matching vanilla's wire behaviour) and the number of ticks replayed. If the history no
    /// longer reaches `acknowledged_tick`, nothing is replayed and the count is zero.
    pub fn apply_correction_with_replay(
        &mut self,
        w: &World,
        c: &PositionCorrection,
        acknowledged_tick: u64,
    ) -> (Outbound, usize) {
        let replay: Vec<(MoveInput, bool)> = self
            .history
            .iter()
            .filter(|e| e.tick > acknowledged_tick)
            .map(|e| (e.input, e.view_is_local))
            .collect();

        let confirm = self.apply_correction(c);

        for (input, view_is_local) in &replay {
            tick_player_with_entities_view(w, &mut self.state, input, &[], *view_is_local);
        }
        (confirm, replay.len())
    }

    /// [`Predictor::apply_correction_with_replay`] using a live world-border projection.
    ///
    /// This additive replay entry point keeps the same persistent outside-border latch as normal
    /// prediction. Vanilla itself does not replay after a correction; callers using this opt-in
    /// path must supply the border sampled for the replay window.
    pub fn apply_correction_with_replay_and_border(
        &mut self,
        w: &World,
        c: &PositionCorrection,
        acknowledged_tick: u64,
        border: CollisionBorder,
    ) -> (Outbound, usize) {
        let replay: Vec<(MoveInput, bool)> = self
            .history
            .iter()
            .filter(|e| e.tick > acknowledged_tick)
            .map(|e| (e.input, e.view_is_local))
            .collect();

        let confirm = self.apply_correction(c);

        for (input, view_is_local) in &replay {
            tick_player_with_entities_and_border_view(
                w,
                &mut self.state,
                input,
                &[],
                border,
                &mut self.outside_border,
                *view_is_local,
            );
        }
        (confirm, replay.len())
    }

    /// Rolls the predicted state back to the start of `tick`, discarding later history.
    ///
    /// # Errors
    ///
    /// [`crate::PhysicsError::HistoryExpired`] when that tick has already fallen out of the ring
    /// buffer, which for the default 100-tick window means the correction is more than five
    /// seconds stale. Snap to the server position instead of replaying.
    pub fn rewind_to(&mut self, tick: u64) -> Result<PhysicsState, crate::PhysicsError> {
        let entry = *self
            .history
            .iter()
            .find(|e| e.tick == tick)
            .ok_or(crate::PhysicsError::HistoryExpired(tick))?;
        self.history.retain(|e| e.tick < tick);
        self.state = entry.before;
        self.tick = tick;
        Ok(entry.before)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{block_by_name, BlockState};
    use mc_math::BlockPos;

    fn flat() -> World {
        let mut w = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        crate::scene::fill(
            &mut w,
            BlockPos::new(-128, 63, -128),
            BlockPos::new(128, 63, 128),
            stone,
        );
        w
    }

    fn standing() -> PhysicsState {
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.on_ground = true;
        s
    }

    #[test]
    fn a_standing_player_sends_the_bare_ground_packet_except_every_twentieth_tick() {
        let w = flat();
        let mut p = Predictor::new(standing());
        let input = MoveInput::default();
        // The first tick reports the *position* — `lastReported` starts at Java's zeros,
        // so a player standing at (0.5, 64, 0.5) has moved but, at yaw 0, has not turned, and
        // `onUpdateWalkingPlayer`'s `flag2 && flag3` picks `C04PacketPlayerPosition`. A
        // `f64::MAX` sentinel would make it a `C06` here, a shape vanilla never sends first.
        assert!(matches!(p.tick(&w, &input), Outbound::Position { .. }));
        let mut ground = 0;
        let mut position = 0;
        for _ in 0..60 {
            match p.tick(&w, &input) {
                Outbound::Ground { .. } => ground += 1,
                Outbound::Position { .. } => position += 1,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(position + ground, 60);
        assert_eq!(position, 2, "a forced update roughly every 20 ticks");
    }

    #[test]
    fn turning_on_the_spot_sends_a_look_packet() {
        let w = flat();
        let mut p = Predictor::new(standing());
        p.tick(&w, &MoveInput::default());
        for _ in 0..5 {
            p.tick(&w, &MoveInput::default());
        }
        let out = p.tick(
            &w,
            &MoveInput {
                yaw: 45.0,
                ..MoveInput::default()
            },
        );
        assert!(matches!(out, Outbound::Look { yaw, .. } if yaw == 45.0));
    }

    #[test]
    fn walking_sends_position_packets() {
        let w = flat();
        let mut p = Predictor::new(standing());
        let input = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        for _ in 0..10 {
            p.tick(&w, &input);
        }
        assert!(matches!(p.tick(&w, &input), Outbound::Position { .. }));
    }

    #[test]
    fn border_aware_predictor_threads_the_current_wall_into_live_movement() {
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let border = CollisionBorder::new(0.0, 0.0, 4.0, 29_999_984);
        let mut state = PhysicsState::new_player(DVec3::new(1.5, 64.0, 0.5));
        state.on_ground = true;
        state.vel.x = 1.0;
        let mut predictor = Predictor::new(state);

        predictor.tick_with_entities_and_border(&w, &MoveInput::default(), &[], border);

        // The empty world has no block at x=2. The border-aware live predictor nevertheless
        // reaches Entity.moveEntity's collision list and stops at the synthetic stone face.
        let expected_x = 2.0 - f64::from(predictor.state.width / 2.0);
        assert!((predictor.state.pos.x - expected_x).abs() < 1.0e-12);
        assert!((predictor.state.aabb().max.x - 2.0).abs() < 1.0e-12);
        assert!(!predictor.outside_border);
    }

    #[test]
    fn a_correction_snaps_zeroes_motion_and_is_confirmed() {
        let w = flat();
        let mut p = Predictor::new(standing());
        let input = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };
        for _ in 0..40 {
            p.tick(&w, &input);
        }
        assert!(p.state.pos.z > 5.0);
        assert!(p.state.vel.z > 0.0);

        let c = PositionCorrection::from_packet(0.5, 64.0, 0.5, 90.0, 10.0, 0);
        let confirm = p.apply_correction(&c);
        assert_eq!(p.state.pos, DVec3::new(0.5, 64.0, 0.5));
        assert_eq!(p.state.vel, DVec3::ZERO);
        assert_eq!(p.state.yaw, 90.0);
        assert_eq!(p.corrections, 1);
        match confirm {
            Outbound::PositionLook {
                pos, yaw, pitch, ..
            } => {
                assert_eq!(pos, DVec3::new(0.5, 64.0, 0.5));
                assert_eq!(yaw, 90.0);
                assert_eq!(pitch, 10.0);
            }
            other => panic!("a correction must be confirmed with pos+look, got {other:?}"),
        }
    }

    #[test]
    fn relative_flags_add_instead_of_replacing() {
        let w = flat();
        let mut p = Predictor::new(standing());
        p.tick(&w, &MoveInput::default());
        let c = PositionCorrection::from_packet(1.0, 0.0, -2.0, 5.0, 0.0, 0x01 | 0x04 | 0x08);
        p.apply_correction(&c);
        assert_eq!(p.state.pos.x, 1.5);
        assert_eq!(p.state.pos.y, 0.0, "absolute Y still replaces");
        assert_eq!(p.state.pos.z, -1.5);
        assert_eq!(p.state.yaw, 5.0);
    }

    #[test]
    fn correction_rotation_uses_java_signed_remainder_at_set_rotation_boundary() {
        // java: `Entity.setRotation`: both fields are assigned `value % 360.0F`, not a
        // non-negative Euclidean remainder. The large values bite a missing-modulo mutant, and
        // the negative values distinguish Java's signed remainder from `rem_euclid`.
        for (yaw, pitch, expected_yaw, expected_pitch) in [
            (725.0, -725.0, 5.0, -5.0),
            (-725.0, 725.0, -5.0, 5.0),
            (1080.0, -1080.0, 0.0, -0.0),
        ] {
            let mut predictor = Predictor::new(standing());
            let confirmation = predictor.apply_correction(&PositionCorrection::from_packet(
                0.5, 64.0, 0.5, yaw, pitch, 0,
            ));
            assert_eq!(predictor.state.yaw, expected_yaw, "yaw {yaw}");
            assert_eq!(predictor.state.pitch, expected_pitch, "pitch {pitch}");
            assert!(matches!(
                confirmation,
                Outbound::PositionLook {
                    yaw: actual_yaw,
                    pitch: actual_pitch,
                    ..
                } if actual_yaw == expected_yaw && actual_pitch == expected_pitch
            ));
        }

        // Relative flags add before the same setRotation remainder is applied.
        let mut predictor = Predictor::new(standing());
        predictor.state.yaw = 350.0;
        predictor.state.pitch = -350.0;
        predictor.apply_correction(&PositionCorrection::from_packet(
            0.0,
            0.0,
            0.0,
            20.0,
            -20.0,
            0x08 | 0x10,
        ));
        assert_eq!(predictor.state.yaw, 10.0);
        assert_eq!(predictor.state.pitch, -10.0);
    }

    #[test]
    fn correction_velocity_is_preserved_only_on_relative_axes() {
        let original = DVec3::new(1.25, -2.5, 3.75);

        for flags in 0u8..8 {
            let mut state = standing();
            state.vel = original;
            let mut predictor = Predictor::new(state);
            predictor.apply_correction(&PositionCorrection::from_packet(
                10.0, 70.0, -4.0, 0.0, 0.0, flags,
            ));

            assert_eq!(
                predictor.state.vel,
                DVec3::new(
                    if flags & 0x01 != 0 { original.x } else { 0.0 },
                    if flags & 0x02 != 0 { original.y } else { 0.0 },
                    if flags & 0x04 != 0 { original.z } else { 0.0 },
                ),
                "position flag matrix entry {flags:#05b}"
            );
        }
    }

    #[test]
    fn corrections_preserve_an_in_progress_fall_distance() {
        let original_fall_distance = 7.25;

        // Vanilla's S08 handler reaches `Entity.setPositionAndRotation` for either form of
        // correction. That method updates position/rotation only; it does not reset
        // `fallDistance`, even when the position is absolute.
        for (name, flags) in [("absolute", 0), ("relative", 0x01 | 0x02 | 0x04)] {
            let mut state = standing();
            state.fall_distance = original_fall_distance;
            let mut predictor = Predictor::new(state);

            predictor.apply_correction(&PositionCorrection::from_packet(
                10.0, 70.0, -4.0, 0.0, 0.0, flags,
            ));

            assert_eq!(
                predictor.state.fall_distance, original_fall_distance,
                "{name} S08 correction must not clear fall distance"
            );
        }
    }

    #[test]
    fn replay_reapplies_the_unacknowledged_inputs() {
        let w = flat();
        let input = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };

        // A reference run of 30 ticks from the corrected position.
        let mut reference = Predictor::new(standing());
        for _ in 0..30 {
            reference.tick(&w, &input);
        }

        // A run that gets corrected back to the start after 30 ticks, with the server having
        // seen nothing (`acknowledged_tick` before every tick in the buffer).
        let mut p = Predictor::new(standing());
        for _ in 0..30 {
            p.tick(&w, &input);
        }
        let c = PositionCorrection::from_packet(0.5, 64.0, 0.5, 0.0, 0.0, 0);
        let (_, replayed) = p.apply_correction_with_replay(&w, &c, 0);
        assert_eq!(replayed, 29, "ticks 1..=29 were unacknowledged");
        assert!(p.state.pos.z > 0.5, "the replay actually moved the player");
        // Not identical to the reference: the correction snapped the position but, exactly as
        // vanilla does, left the rest of the entity state (sprint flag, timers) alone, so the
        // replay starts already sprinting. That is a property of the design, not slop.
        assert!(p.state.sprinting);
        assert!(reference.state.pos.z > 0.5);
    }

    #[test]
    fn replay_is_a_no_op_when_the_history_is_gone() {
        let w = flat();
        let mut p = Predictor::new(standing());
        for _ in 0..10 {
            p.tick(&w, &MoveInput::default());
        }
        let c = PositionCorrection::from_packet(3.5, 64.0, 3.5, 0.0, 0.0, 0);
        let (_, replayed) = p.apply_correction_with_replay(&w, &c, 1_000);
        assert_eq!(replayed, 0);
        assert_eq!(p.state.pos, DVec3::new(3.5, 64.0, 3.5));
    }

    /// Regression: `apply_correction` used to assign `pos` without rebuilding the bounding box.
    /// Because the box is authoritative and `pos` is derived from it each tick, the stale box
    /// snapped the player straight back and every teleport was silently lost one tick later.
    /// Found on 2026-08-15 against a real `S08PacketPlayerPosLook`.
    #[test]
    fn correcting_the_position_also_moves_the_bounding_box() {
        let w = flat();
        let mut p = Predictor::new(standing());
        let target = DVec3::new(3.5, 64.0, -7.5);

        p.apply_correction(&PositionCorrection::from_packet(
            target.x, target.y, target.z, 0.0, 0.0, 0,
        ));
        assert_eq!(p.state.pos, target, "the correction itself must land");

        // The box has to have moved with it, or the next tick derives the old position back.
        let bb = p.state.aabb();
        assert!(
            ((bb.min.x + bb.max.x) / 2.0 - target.x).abs() < 1e-12
                && (bb.min.y - target.y).abs() < 1e-12
                && ((bb.min.z + bb.max.z) / 2.0 - target.z).abs() < 1e-12,
            "bounding box {bb:?} did not follow the correction to {target:?}"
        );

        // The real symptom, asserted end to end: one idle tick must not undo the teleport.
        p.tick(&w, &MoveInput::default());
        assert!(
            (p.state.pos.x - target.x).abs() < 1e-9 && (p.state.pos.z - target.z).abs() < 1e-9,
            "one tick after the correction the player was back at {:?}",
            p.state.pos
        );
    }

    #[test]
    fn rewind_restores_the_state_before_a_tick() {
        let w = flat();
        let mut p = Predictor::new(standing());
        let input = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        for _ in 0..10 {
            p.tick(&w, &input);
        }
        let at5 = p.rewind_to(5).expect("tick 5 is in history");
        assert_eq!(p.state, at5);
        assert_eq!(p.current_tick(), 5);
        assert!(
            matches!(p.rewind_to(9), Err(crate::PhysicsError::HistoryExpired(9))),
            "later history was discarded"
        );
    }
}
