//! The [`Predictor`]'s 1.7.10 entry points.
//!
//! A predictor built with [`Predictor::new_v5`] carries a [`PlayerV5`] beside its
//! [`PhysicsState`](crate::PhysicsState) and is driven only through these `*_v5` methods. The
//! 1.8.9 entry points stay as they are; the caller (the session) picks one set per connection.

use mc_math::DVec3;
use mc_world::World;

use super::{player, PlayerV5, STEP_HEIGHT};
use crate::collision::EntityCollisionCandidate;
use crate::state::{MoveInput, PhysicsState};
use crate::Predictor;

/// What `EntityClientPlayerMP.sendMotionUpdates` puts on the wire this tick, after its C0B
/// edges. java17: `client/entity/EntityClientPlayerMP.java:133-148`; its riding C06 (`:128-132`)
/// is never reached, because `onUpdate` calls the method only when not riding (`:74-82`).
///
/// `y` is the box bottom and `stance` the eye `posY`, the two values the protocol-5 C04/C06
/// carry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutboundV5 {
    /// `C03PacketPlayer`.
    Ground {
        /// java17: `onGround`.
        on_ground: bool,
    },
    /// `C04PacketPlayerPosition(posX, boundingBox.minY, posY, posZ, onGround)`.
    Position {
        /// java17: `posX`.
        x: f64,
        /// java17: `boundingBox.minY`.
        y: f64,
        /// java17: `posY`, the eye.
        stance: f64,
        /// java17: `posZ`.
        z: f64,
        /// java17: `onGround`.
        on_ground: bool,
    },
    /// `C05PacketPlayerLook(rotationYaw, rotationPitch, onGround)`.
    Look {
        /// Degrees.
        yaw: f32,
        /// Degrees.
        pitch: f32,
        /// java17: `onGround`.
        on_ground: bool,
    },
    /// `C06PacketPlayerPosLook(posX, boundingBox.minY, posY, posZ, yaw, pitch, onGround)`.
    PositionLook {
        /// java17: `posX`.
        x: f64,
        /// java17: `boundingBox.minY`.
        y: f64,
        /// java17: `posY`, the eye.
        stance: f64,
        /// java17: `posZ`.
        z: f64,
        /// Degrees.
        yaw: f32,
        /// Degrees.
        pitch: f32,
        /// java17: `onGround`.
        on_ground: bool,
    },
}

/// The `C06PacketPlayerPosLook` a 1.7.10 client answers an S08 with. java17:
/// `NetHandlerPlayClient.handlePlayerPosLook` (`client/network/NetHandlerPlayClient.java:556`):
/// the corrected `posX`, `boundingBox.minY`, `posY` and `posZ`, then the **packet's** raw yaw,
/// pitch and on-ground flag (not the entity's `% 360` rotation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectionReplyV5 {
    /// java17: `posX` after the correction.
    pub x: f64,
    /// java17: `boundingBox.minY` after the correction.
    pub y: f64,
    /// java17: `posY` after the correction, the packet's y.
    pub stance: f64,
    /// java17: `posZ` after the correction.
    pub z: f64,
    /// The packet's yaw.
    pub yaw: f32,
    /// The packet's pitch.
    pub pitch: f32,
    /// The packet's on-ground flag.
    pub on_ground: bool,
}

impl Predictor {
    /// A 1.7.10 predictor starting from `state`: step height `0.5` (java17:
    /// `EntityLivingBase.<init>`, `entity/EntityLivingBase.java:124`) and the 1.7.10 fields at
    /// rest, `posY = minY + 1.62F`.
    ///
    /// Replay history is not kept for a 1.7.10 predictor (vanilla never replays, and the 1.8.9
    /// history entries cannot hold [`PlayerV5`]).
    #[must_use]
    pub fn new_v5(mut state: PhysicsState) -> Self {
        state.step_height = STEP_HEIGHT;
        let v5 = PlayerV5::at_feet(&state);
        let mut predictor = Predictor::new(state);
        predictor.v5 = Some(v5);
        predictor
    }

    /// The 1.7.10 player fields, or `None` for a 1.8.9 predictor. [`PlayerV5::pos_y`] is the
    /// eye the 1.7.10 pick, stance and item ray start from.
    #[must_use]
    pub fn v5(&self) -> Option<&PlayerV5> {
        self.v5.as_ref()
    }

    /// One 1.7.10 walking tick, then the movement packet `sendMotionUpdates` chooses. `None`
    /// for a 1.8.9 predictor, which this leaves untouched.
    ///
    /// java17: `EntityClientPlayerMP.onUpdate`'s non-riding branch
    /// (`client/entity/EntityClientPlayerMP.java:68-84`).
    pub fn tick_v5(
        &mut self,
        w: &World,
        input: &MoveInput,
        entities: &[EntityCollisionCandidate],
    ) -> Option<OutboundV5> {
        let v5 = self.v5.as_mut()?;
        player::tick_player(w, &mut self.state, v5, input, entities);
        self.tick += 1;
        self.outbound_v5()
    }

    /// The first half of a mounted 1.7.10 tick; see
    /// [`super::player::mounted_on_living_update_preflight`]. `false` for a 1.8.9 predictor.
    pub fn mounted_preflight_v5(&mut self, w: &World, input: &MoveInput) -> bool {
        let Some(v5) = self.v5.as_mut() else {
            return false;
        };
        player::mounted_on_living_update_preflight(w, &mut self.state, v5, input);
        true
    }

    /// The second half of a mounted 1.7.10 tick; see [`super::player::mounted_living_travel`].
    /// `false` for a 1.8.9 predictor.
    pub fn mounted_travel_v5(&mut self, w: &World, entities: &[EntityCollisionCandidate]) -> bool {
        let Some(v5) = self.v5.as_mut() else {
            return false;
        };
        player::mounted_living_travel(w, &mut self.state, v5, entities);
        self.tick += 1;
        true
    }

    /// java17: `EntityClientPlayerMP.sendMotionUpdates`'s shape choice and bookkeeping
    /// (`client/entity/EntityClientPlayerMP.java:120-166`).
    ///
    /// The decision is 1.8.9's (`9.0E-4` squared distance; `ticksSinceMovePacket` incremented
    /// after its `>= 20` test, so a still player is re-reported every 21 ticks in both versions,
    /// java18: `EntityPlayerSP.onUpdateWalkingPlayer`). What differs is the payload: the box
    /// bottom and the eye stance. The C0B sprint and sneak edges precede this and are the
    /// caller's, as in 1.8.9.
    pub fn outbound_v5(&mut self) -> Option<OutboundV5> {
        let v5 = self.v5.as_mut()?;
        let s = &self.state;
        let min_y = s.aabb().min.y;
        let r = &mut v5.report;
        let var3 = s.pos.x - r.old_pos_x;
        let var5 = min_y - r.old_min_y;
        let var7 = s.pos.z - r.old_pos_z;
        let var9 = f64::from(s.yaw - r.old_yaw);
        let var11 = f64::from(s.pitch - r.old_pitch);
        let moved =
            var3 * var3 + var5 * var5 + var7 * var7 > 9.0e-4 || r.ticks_since_move_packet >= 20;
        let turned = var9 != 0.0 || var11 != 0.0;

        let out = match (moved, turned) {
            (true, true) => OutboundV5::PositionLook {
                x: s.pos.x,
                y: min_y,
                stance: v5.pos_y,
                z: s.pos.z,
                yaw: s.yaw,
                pitch: s.pitch,
                on_ground: s.on_ground,
            },
            (true, false) => OutboundV5::Position {
                x: s.pos.x,
                y: min_y,
                stance: v5.pos_y,
                z: s.pos.z,
                on_ground: s.on_ground,
            },
            (false, true) => OutboundV5::Look {
                yaw: s.yaw,
                pitch: s.pitch,
                on_ground: s.on_ground,
            },
            (false, false) => OutboundV5::Ground {
                on_ground: s.on_ground,
            },
        };

        r.ticks_since_move_packet = r.ticks_since_move_packet.wrapping_add(1);
        r.was_on_ground = s.on_ground;
        if moved {
            r.old_pos_x = s.pos.x;
            r.old_min_y = min_y;
            r.old_pos_y = v5.pos_y;
            r.old_pos_z = s.pos.z;
            r.ticks_since_move_packet = 0;
        }
        if turned {
            r.old_yaw = s.yaw;
            r.old_pitch = s.pitch;
        }
        Some(out)
    }

    /// Applies a 1.7.10 `S08PacketPlayerPosLook` to the player and returns the C06 reply.
    /// `None` for a 1.8.9 predictor, which this leaves untouched.
    ///
    /// java17: `NetHandlerPlayClient.handlePlayerPosLook`
    /// (`client/network/NetHandlerPlayClient.java:545-566`): `yOffset2 = 0`, all motion zeroed,
    /// then `Entity.setPositionAndRotation` (`entity/Entity.java:1069-1091`): `prevPos = pos =
    /// (x, y, z)` with `y` the eye, the box bottom `y - (double)yOffset + 0.0`, and the rotation
    /// stored `% 360`. There are no relative flags in protocol 5. The first S08's terrain-screen
    /// close is the caller's.
    pub fn apply_correction_v5(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        yaw: f32,
        pitch: f32,
        on_ground: bool,
    ) -> Option<CorrectionReplyV5> {
        let v5 = self.v5.as_mut()?;
        let s = &mut self.state;
        s.vel = DVec3::ZERO;
        v5.reset_for_teleport(y);
        v5.set_position(s, x, y, z);
        s.prev_pos = s.pos;
        // java17: `Entity.setRotation` (`entity/Entity.java:243-247`), Java's truncating `%`.
        s.yaw = yaw % 360.0;
        s.pitch = pitch % 360.0;
        self.corrections += 1;
        Some(CorrectionReplyV5 {
            x: s.pos.x,
            y: s.aabb().min.y,
            stance: v5.pos_y,
            z: s.pos.z,
            yaw,
            pitch,
            on_ground,
        })
    }

    /// Places the 1.7.10 player with its eye at `y`. java17: `Entity.setPosition`
    /// (`entity/Entity.java:249-257`), the call `updateRiderPosition` makes for a rider. `false`
    /// for a 1.8.9 predictor.
    pub fn set_position_v5(&mut self, x: f64, y: f64, z: f64) -> bool {
        let Some(v5) = self.v5.as_mut() else {
            return false;
        };
        v5.set_position(&mut self.state, x, y, z);
        true
    }

    /// Writes the 1.7.10 player's `yOffset` ([`PlayerV5::y_offset`]) and nothing else; the eye
    /// follows at the next placement or move. `false` for a 1.8.9 predictor.
    pub fn set_y_offset_v5(&mut self, y_offset: f32) -> bool {
        let Some(v5) = self.v5.as_mut() else {
            return false;
        };
        v5.set_y_offset(y_offset);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene;
    use mc_data::{block_by_name, BlockState};

    fn flat() -> World {
        scene::flat(BlockState::new(block_by_name("stone").unwrap(), 0), 63, 8)
    }

    #[test]
    fn a_1_8_9_predictor_ignores_the_1_7_10_entry_points() {
        let w = flat();
        let mut p = Predictor::new(PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5)));
        let before = p.state;
        assert!(p.v5().is_none());
        assert!(p.tick_v5(&w, &MoveInput::default(), &[]).is_none());
        assert!(p.outbound_v5().is_none());
        assert!(p
            .apply_correction_v5(1.0, 70.0, 1.0, 0.0, 0.0, false)
            .is_none());
        assert!(!p.set_position_v5(1.0, 70.0, 1.0));
        assert!(!p.set_y_offset_v5(super::super::DEATH_Y_OFFSET));
        assert_eq!(p.state, before);
        assert_eq!(p.state.step_height, 0.6);
    }

    #[test]
    fn a_1_7_10_predictor_steps_half_a_block() {
        let p = Predictor::new_v5(PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5)));
        assert_eq!(p.state.step_height, 0.5);
        assert_eq!(p.v5().unwrap().pos_y(), 64.0 + f64::from(1.62_f32));
    }

    /// The forced re-report: every 21 ticks for a still player, the same cadence as 1.8.9.
    #[test]
    fn a_still_player_is_re_reported_every_twenty_one_ticks() {
        let w = flat();
        let mut p = Predictor::new_v5(PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5)));
        let mut q = Predictor::new(PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5)));
        let mut v5_reports = Vec::new();
        let mut v47_reports = Vec::new();
        for tick in 0..50 {
            let out = p.tick_v5(&w, &MoveInput::default(), &[]).unwrap();
            if matches!(
                out,
                OutboundV5::Position { .. } | OutboundV5::PositionLook { .. }
            ) {
                v5_reports.push(tick);
            }
            let out = q.tick(&w, &MoveInput::default());
            if matches!(
                out,
                crate::prediction::Outbound::Position { .. }
                    | crate::prediction::Outbound::PositionLook { .. }
            ) {
                v47_reports.push(tick);
            }
        }
        // The first tick reports the spawn (its old position is the origin), then the counter.
        assert_eq!(v5_reports, vec![0, 21, 42]);
        assert_eq!(v47_reports, v5_reports);
    }

    #[test]
    fn the_stance_is_the_eye_and_sneaking_lowers_it() {
        let w = flat();
        let mut p = Predictor::new_v5(PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5)));
        for _ in 0..3 {
            p.tick_v5(&w, &MoveInput::default(), &[]);
        }
        let sneak = MoveInput {
            sneak: true,
            forward: true,
            ..MoveInput::default()
        };
        // The first sneak-walk tick moves less than the 9.0E-4 threshold (a C03); the next
        // reports the position.
        let first = p.tick_v5(&w, &sneak, &[]).unwrap();
        assert_eq!(first, OutboundV5::Ground { on_ground: true });
        let out = p.tick_v5(&w, &sneak, &[]).unwrap();
        let OutboundV5::Position { y, stance, .. } = out else {
            panic!("the second sneak-walk tick reports its position: {out:?}");
        };
        assert_eq!(y, 64.0);
        assert_eq!(stance, 65.539_999_999_105_93);
    }

    /// java17 `EntityPlayer.onDeath` ends with `yOffset = 0.1F`: the dying player keeps
    /// reporting, with its stance 0.1 above the box bottom, and an S08 puts the box 0.1 below the
    /// packet's y.
    #[test]
    fn a_dead_player_reports_a_stance_a_tenth_above_its_box() {
        let w = flat();
        let mut p = Predictor::new_v5(PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5)));
        let mut control = p.clone();
        p.tick_v5(&w, &MoveInput::default(), &[]);
        control.tick_v5(&w, &MoveInput::default(), &[]);
        assert!(p.set_y_offset_v5(super::super::DEATH_Y_OFFSET));
        let kick = DVec3::new(0.0, 0.10000000149011612, 0.0);
        p.state.vel = kick;
        control.state.vel = kick;
        let out = p.tick_v5(&w, &MoveInput::default(), &[]).unwrap();
        let OutboundV5::Position { y, stance, .. } = out else {
            panic!("the death kick moves the player: {out:?}");
        };
        assert_eq!(y, 64.0 + kick.y);
        assert_eq!(stance, y + f64::from(0.1_f32));
        // Control: the same move at the living offset reports the eye 1.62 up.
        let out = control.tick_v5(&w, &MoveInput::default(), &[]).unwrap();
        assert_eq!(
            out,
            OutboundV5::Position {
                x: 0.5,
                y,
                stance: y + f64::from(1.62_f32),
                z: 0.5,
                on_ground: false,
            }
        );

        let reply = p
            .apply_correction_v5(0.5, 70.0, 0.5, 0.0, 0.0, false)
            .unwrap();
        assert_eq!(reply.y, 70.0 - f64::from(0.1_f32));
        assert_eq!(reply.stance, 70.0);
    }

    /// Absolute, motion zeroed, `yOffset2` reset, eye y, raw rotation echoed.
    #[test]
    fn a_correction_places_the_eye_and_echoes_the_raw_rotation() {
        let w = flat();
        let mut p = Predictor::new_v5(PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5)));
        let sneak = MoveInput {
            sneak: true,
            ..MoveInput::default()
        };
        p.tick_v5(&w, &sneak, &[]);
        p.state.vel = DVec3::new(0.3, 0.2, 0.1);
        let eye = 65.0 + 1.620_000_004_768_371_6;
        let reply = p
            .apply_correction_v5(3.5, eye, -2.5, 400.0, -100.0, true)
            .unwrap();
        assert_eq!(p.state.vel, DVec3::ZERO);
        assert_eq!(p.v5().unwrap().y_size(), 0.0);
        assert_eq!(p.v5().unwrap().pos_y(), eye);
        assert_eq!(
            p.state.pos,
            DVec3::new(3.5, eye - f64::from(1.62_f32), -2.5)
        );
        assert_eq!(p.state.yaw, 40.0);
        assert_eq!(p.state.pitch, -100.0);
        assert_eq!(
            reply,
            CorrectionReplyV5 {
                x: 3.5,
                y: eye - f64::from(1.62_f32),
                stance: eye,
                z: -2.5,
                yaw: 400.0,
                pitch: -100.0,
                on_ground: true,
            }
        );
    }
}
