//! Minecraft 1.7.10 local-player movement, where it differs from 1.8.9.
//!
//! The rest of this crate transcribes 1.8.9. This module holds only what vanilla 1.7.10 does
//! differently and reuses the 1.8.9 code for everything the two versions share (`moveFlying`,
//! `jump`, ladders, liquid flow, `pushOutOfBlocks`, and every block collision shape that is
//! identical in both). A 1.7.10 [`Predictor`](crate::Predictor) is built with
//! [`Predictor::new_v5`](crate::Predictor::new_v5) and driven with its `*_v5` methods; nothing in
//! the 1.8.9 path consults this module.
//!
//! Citations: `java17:` cites vanilla 1.7.10's source, where the local player class is
//! `EntityClientPlayerMP` (extending the 1.7.10 `EntityPlayerSP`); `java18:` cites 1.8.9's.
//!
//! # The `posY` convention
//!
//! The 1.7.10 local player's `posY` is its **eye**, not its feet:
//! `posY = boundingBox.minY + yOffset - yOffset2` with `yOffset = 1.62F`, or `0.1F` once dead and
//! `0.2F` in bed (java17: `Entity.moveEntity`, `entity/Entity.java:696`; `Entity.setPosition`,
//! `:249-257`).
//! [`PhysicsState::pos`] keeps its 1.8.9 meaning (the feet, `minY`) for both versions, and
//! [`PlayerV5`] carries the 1.7.10-only fields beside it: `posY` itself, which Java updates only
//! at specific statements and therefore lags `yOffset2` within a tick, the decaying sneak offset
//! `yOffset2`, and the `sendMotionUpdates` bookkeeping.

pub mod collision;
pub mod movement;
pub mod player;
pub mod predictor;

use mc_math::DVec3;

use crate::state::PhysicsState;

pub use predictor::{CorrectionReplyV5, OutboundV5};

/// The local player's `yOffset` while alive and awake. java17: `EntityPlayer.<init>`
/// (`entity/player/EntityPlayer.java:121`), again in `preparePlayerToSpawn` (`:468`) and
/// `resetHeight` (`:849`, from `wakeUpPlayer`).
pub const PLAYER_Y_OFFSET: f32 = 1.62;

/// The `yOffset` of a dying player. java17: `EntityPlayer.onDeath`
/// (`entity/player/EntityPlayer.java:610`), after its 0.2 x 0.2 `setSize` and `setPosition`.
pub const DEATH_Y_OFFSET: f32 = 0.1;

/// The `yOffset` of a player in bed. java17: `EntityPlayer.sleepInBedAt`
/// (`entity/player/EntityPlayer.java:1238`), before the `setPosition` that places it.
pub const SLEEP_Y_OFFSET: f32 = 0.2;

/// The living-entity step height. java17: `EntityLivingBase.<init>`
/// (`entity/EntityLivingBase.java:124`) sets `0.5F`, where 1.8.9 sets `0.6F`.
pub const STEP_HEIGHT: f32 = 0.5;

/// The `yOffset2` a held sneak key raises the player to. java17: `EntityPlayerSP.onLivingUpdate`
/// (`client/entity/EntityPlayerSP.java:180-183`).
pub const SNEAK_Y_SIZE: f32 = 0.2;

/// java17: `Entity.moveEntity`'s first statement on the collision path,
/// `this.yOffset2 *= 0.4F;` (`entity/Entity.java:469`).
pub const Y_SIZE_DECAY: f32 = 0.4;

/// The step-up gate: below this `yOffset2` a player may step up even when it is not sneaking on
/// the ground. java17: `Entity.moveEntity` (`entity/Entity.java:612`).
pub const STEP_GATE_Y_SIZE: f32 = 0.05;

/// The vertical creative-flight impulse, a constant in 1.7.10. java17:
/// `EntityPlayerSP.onLivingUpdate` (`client/entity/EntityPlayerSP.java:227-238`); 1.8.9 uses
/// `flySpeed * 3.0F` instead.
pub const FLY_VERTICAL_SPEED: f64 = 0.15;

/// The 1.7.10-only entity fields of the local player, kept beside its [`PhysicsState`].
///
/// Everything here is a vanilla field with the same width: `yOffset` and `yOffset2` are `float`,
/// `posY` and `prevPosY` are `double`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerV5 {
    /// java17: `Entity.yOffset`: [`PLAYER_Y_OFFSET`], [`DEATH_Y_OFFSET`] or [`SLEEP_Y_OFFSET`].
    y_offset: f32,
    /// java17: `Entity.yOffset2`.
    y_size: f32,
    /// java17: `Entity.posY`, the eye. Written only where Java writes it (`moveEntity`,
    /// `setPosition`), so within a tick it can lag a `yOffset2` change.
    pos_y: f64,
    /// java17: `Entity.prevPosY`, taken by `Entity.onEntityUpdate` and `setPositionAndRotation`.
    prev_pos_y: f64,
    /// java17: `EntityClientPlayerMP.sendMotionUpdates`'s fields.
    pub(crate) report: MotionReportV5,
}

/// java17: the `EntityClientPlayerMP` fields `sendMotionUpdates` keeps, declared at
/// `client/entity/EntityClientPlayerMP.java:30-36` and `:39` and written at the end of the
/// method (`:150-166`). All start at Java's defaults, `0` and `false`, on a fresh player.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct MotionReportV5 {
    pub(crate) old_pos_x: f64,
    pub(crate) old_min_y: f64,
    /// java17: `oldPosY`, the stance last sent. Written, never read by the decision.
    pub(crate) old_pos_y: f64,
    pub(crate) old_pos_z: f64,
    pub(crate) old_yaw: f32,
    pub(crate) old_pitch: f32,
    pub(crate) ticks_since_move_packet: i32,
    pub(crate) was_on_ground: bool,
}

impl PlayerV5 {
    /// The 1.7.10 fields for a player whose box is `s`'s, with no sneak offset: `posY` is
    /// derived the way `Entity.moveEntity` derives it, `minY + yOffset - yOffset2`.
    #[must_use]
    pub fn at_feet(s: &PhysicsState) -> Self {
        let mut v = Self {
            y_offset: PLAYER_Y_OFFSET,
            y_size: 0.0,
            pos_y: 0.0,
            prev_pos_y: 0.0,
            report: MotionReportV5::default(),
        };
        v.derive_pos_y(s.aabb().min.y);
        v.prev_pos_y = v.pos_y;
        v
    }

    /// java17: `Entity.posY`. For the local player this is the **eye**: the origin of its pick
    /// ray, the stance of its C04/C06, and what `getPosition(1.0F)` returns as Y.
    #[must_use]
    pub fn pos_y(&self) -> f64 {
        self.pos_y
    }

    /// java17: `Entity.prevPosY`, the eye at the start of the last tick (or the last
    /// `setPositionAndRotation`). Render-time picks interpolate from it.
    #[must_use]
    pub fn prev_pos_y(&self) -> f64 {
        self.prev_pos_y
    }

    /// java17: `Entity.yOffset`: [`PLAYER_Y_OFFSET`] for the local player, unless it is dead
    /// ([`DEATH_Y_OFFSET`]) or in bed ([`SLEEP_Y_OFFSET`]).
    #[must_use]
    pub fn y_offset(&self) -> f32 {
        self.y_offset
    }

    /// Writes `Entity.yOffset` and nothing else, as Java does: `posY` keeps its value until the
    /// next `setPosition` or `moveEntity` derives one with the new offset. java17:
    /// `EntityPlayer.onDeath` ([`DEATH_Y_OFFSET`]), `sleepInBedAt` ([`SLEEP_Y_OFFSET`]),
    /// `resetHeight` and `preparePlayerToSpawn` ([`PLAYER_Y_OFFSET`]).
    pub(crate) fn set_y_offset(&mut self, y_offset: f32) {
        self.y_offset = y_offset;
    }

    /// java17: `Entity.yOffset2`, the decaying sneak offset.
    #[must_use]
    pub fn y_size(&self) -> f32 {
        self.y_size
    }

    /// java17: `EntityLivingBase.getPosition(1.0F)` (`entity/EntityLivingBase.java:1910-1923`):
    /// `(posX, posY, posZ)`, the eye position the tick-time pick starts from.
    #[must_use]
    pub fn eye_position(&self, s: &PhysicsState) -> DVec3 {
        DVec3::new(s.pos.x, self.pos_y, s.pos.z)
    }

    /// java17: `EntityLivingBase.getPosition(partialTicks)`: the current position at exactly
    /// `1.0F`, otherwise `prev + (pos - prev) * partialTicks` per axis.
    #[must_use]
    pub fn eye_position_at(&self, s: &PhysicsState, partial_ticks: f32) -> DVec3 {
        if partial_ticks == 1.0 {
            return self.eye_position(s);
        }
        let t = f64::from(partial_ticks);
        DVec3::new(
            s.prev_pos.x + (s.pos.x - s.prev_pos.x) * t,
            self.prev_pos_y + (self.pos_y - self.prev_pos_y) * t,
            s.prev_pos.z + (s.pos.z - s.prev_pos.z) * t,
        )
    }

    /// java17: `Entity.setPosition(x, y, z)` (`entity/Entity.java:249-257`) with `y` the eye:
    /// `posY = y` and the box bottom `minY = y - yOffset + yOffset2`, in that order of
    /// evaluation. The S08 handler, riding and respawn place the player through this.
    pub fn set_position(&mut self, s: &mut PhysicsState, x: f64, y: f64, z: f64) {
        let min_y = y - f64::from(self.y_offset) + f64::from(self.y_size);
        s.set_position(DVec3::new(x, min_y, z));
        self.pos_y = y;
    }

    /// Records the start-of-tick `prevPosY`. java17: `Entity.onEntityUpdate`
    /// (`entity/Entity.java:296`).
    pub(crate) fn snapshot_prev(&mut self) {
        self.prev_pos_y = self.pos_y;
    }

    /// java17: the sneak line of `EntityPlayerSP.onLivingUpdate`
    /// (`client/entity/EntityPlayerSP.java:180-183`).
    pub(crate) fn raise_for_sneak(&mut self, sneak: bool) {
        if sneak && self.y_size < SNEAK_Y_SIZE {
            self.y_size = SNEAK_Y_SIZE;
        }
    }

    /// java17: `this.yOffset2 *= 0.4F;` (`entity/Entity.java:469`), in `float`.
    pub(crate) fn decay(&mut self) {
        self.y_size *= Y_SIZE_DECAY;
    }

    /// java17: `this.posY = this.boundingBox.minY + (double)this.yOffset -
    /// (double)this.yOffset2;` (`entity/Entity.java:463`, `:696`).
    pub(crate) fn derive_pos_y(&mut self, min_y: f64) {
        self.pos_y = min_y + f64::from(self.y_offset) - f64::from(self.y_size);
    }

    /// java17: `Entity.setPositionAndRotation`'s `prevPosY = posY = y` and `yOffset2 = 0.0F`
    /// (`entity/Entity.java:1069-1076`).
    pub(crate) fn reset_for_teleport(&mut self, y: f64) {
        self.prev_pos_y = y;
        self.y_size = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_eye_is_the_box_bottom_plus_the_float_offset() {
        let s = PhysicsState::new_player(DVec3::new(8.5, 65.0, 8.5));
        let v = PlayerV5::at_feet(&s);
        // java17: `minY + (double)1.62F - (double)0.0F`, the value the oracle records.
        assert_eq!(v.pos_y(), 66.620_000_004_768_37);
        assert_eq!(v.pos_y(), 65.0 + f64::from(1.62_f32));
        assert_ne!(
            v.pos_y(),
            65.0 + 1.62,
            "a double 1.62 is a parity bug in every stance"
        );
        assert_eq!(v.eye_position(&s), DVec3::new(8.5, v.pos_y(), 8.5));
    }

    #[test]
    fn set_position_derives_the_box_bottom_from_the_eye() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        let mut v = PlayerV5::at_feet(&s);
        // The vanilla 1.7.10 server sends S08 y as `feet + 1.6200000047683716D`.
        let eye = 64.0 + 1.620_000_004_768_371_6;
        v.set_position(&mut s, 0.5, eye, 0.5);
        assert_eq!(s.aabb().min.y, eye - f64::from(1.62_f32));
        assert_eq!(
            v.pos_y(),
            eye,
            "posY keeps the given value, not a re-derived one"
        );
        assert_eq!(s.pos.y, s.aabb().min.y);
    }

    #[test]
    fn sneaking_raises_the_offset_to_a_fifth_and_moves_decay_it_by_four_tenths() {
        let s = PhysicsState::new_player(DVec3::new(0.0, 64.0, 0.0));
        let mut v = PlayerV5::at_feet(&s);
        v.raise_for_sneak(true);
        assert_eq!(v.y_size(), 0.2);
        v.decay();
        assert_eq!(f64::from(v.y_size()), 0.080_000_005_662_441_25);
        // A held key raises it back to 0.2 every tick: the steady sneaking value is 0.2 * 0.4.
        v.raise_for_sneak(true);
        assert_eq!(v.y_size(), 0.2);
        // Negative control: a released key leaves it decaying.
        v.decay();
        v.raise_for_sneak(false);
        v.decay();
        assert_eq!(v.y_size(), 0.2_f32 * 0.4 * 0.4);
        v.derive_pos_y(64.0);
        assert_eq!(
            v.pos_y(),
            64.0 + f64::from(1.62_f32) - f64::from(0.2_f32 * 0.4 * 0.4)
        );
    }

    #[test]
    fn a_new_y_offset_moves_the_eye_only_at_the_next_derivation_or_placement() {
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        let mut v = PlayerV5::at_feet(&s);
        let eye = v.pos_y();
        v.set_y_offset(DEATH_Y_OFFSET);
        assert_eq!(v.pos_y(), eye, "java17 writes the field and nothing else");
        v.derive_pos_y(64.0);
        assert_eq!(v.pos_y(), 64.100_000_001_490_12);
        // In bed: `yOffset = 0.2F`, then `setPosition(.., (float)64 + 0.9375F, ..)`.
        v.set_y_offset(SLEEP_Y_OFFSET);
        v.set_position(&mut s, 3.5, f64::from(64.0_f32 + 0.9375), 7.9);
        assert_eq!(s.aabb().min.y, 64.737_499_997_019_77);
        assert_eq!(v.pos_y(), 64.9375);
        // Control: the fixed 1.62F offset would have put the box 1.42 lower.
        v.set_y_offset(PLAYER_Y_OFFSET);
        v.set_position(&mut s, 3.5, 64.9375, 7.9);
        assert_eq!(s.aabb().min.y, 64.9375 - f64::from(1.62_f32));
    }

    #[test]
    fn the_partial_tick_position_interpolates_the_eye_except_at_one() {
        let mut s = PhysicsState::new_player(DVec3::new(0.0, 64.0, 0.0));
        let mut v = PlayerV5::at_feet(&s);
        v.snapshot_prev();
        s.prev_pos = s.pos;
        s.set_position(DVec3::new(1.0, 65.0, 0.0));
        v.derive_pos_y(65.0);
        let half = v.eye_position_at(&s, 0.5);
        assert_eq!(half.x, 0.5);
        assert_eq!(half.y, v.prev_pos_y() + (v.pos_y() - v.prev_pos_y()) * 0.5);
        assert_eq!(v.eye_position_at(&s, 1.0), v.eye_position(&s));
    }
}
