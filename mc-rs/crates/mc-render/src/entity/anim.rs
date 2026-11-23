//! Animation inputs and the model trait every entity model implements.
//!
//! java: `client/model/ModelBase.java` (`setRotationAngles`, `setLivingAnimations`),
//! `client/renderer/entity/RendererLivingEntity.java` (which computes the arguments).
//!
//! # Where the numbers come from
//!
//! `RendererLivingEntity.doRender` derives five floats per frame and hands them to the model:
//!
//! | Java local | Field | Meaning |
//! |---|---|---|
//! | `f6` | [`AnimParams::limb_swing`] | `limbSwing - limbSwingAmount * (1 - partialTicks)`, tripled for babies |
//! | `f5` | [`AnimParams::limb_swing_amount`] | interpolated `limbSwingAmount`, clamped to `1.0` |
//! | `f8` | [`AnimParams::age_in_ticks`] | `ticksExisted + partialTicks` |
//! | `f2` | [`AnimParams::net_head_yaw`] | head yaw minus body yaw, **degrees** |
//! | `f7` | [`AnimParams::head_pitch`] | interpolated pitch, **degrees** |
//!
//! [`LivingRenderState`] does that derivation, so the ordering quirks (the baby multiplier is
//! applied *before* the clamp; the clamp is on the amount, not the swing) live in one place.
//!
//! All trigonometry in a model body must go through [`mc_math::sin`]/[`mc_math::cos`], never
//! `f32::sin` — hazard H1. Animation is not movement, so a mismatch here is cosmetic rather
//! than desync, but the LUT is what vanilla's poses were authored against and the difference is
//! visible at small limb-swing amounts.

use glam::Mat4;

use super::model::Model;
use super::EntityVertex;

/// Radians per degree for source expressions that are naturally multiplicative.
///
/// **This is not bit-identical to vanilla and the doc here used to claim it was.** Every model
/// in 1.8.9 writes `angle / (180F / (float)Math.PI)` — a *division*. Multiplying by a
/// pre-rounded reciprocal is a different operation: the reciprocal is itself rounded to `f32`
/// before the multiply, so `d * DEG_TO_RAD != d / DEG_PER_RAD` for **more than half** of all
/// degree values. Swept over `-360..=360` in quarter-degree steps, 1520 of 2881 samples differ;
/// the gap is one ulp of the result, about `5e-7` radians at 360 degrees.
///
/// This constant remains for renderer transforms whose source expressions are multiplicative.
/// Model pose code instead calls [`model_radians`] and preserves vanilla's division.
pub const DEG_TO_RAD: f32 = 1.0 / DEG_PER_RAD;

/// Vanilla's divisor, spelled exactly as the models spell it: `180F / (float)Math.PI`.
///
/// java: `ModelBiped.setRotationAngles` and every one of its peers.
pub const DEG_PER_RAD: f32 = 180.0 / core::f32::consts::PI;

/// Converts a model input from degrees exactly as vanilla's `setRotationAngles` methods do.
///
/// The compiled 1.8.9 classes load the rounded `57.295776F` divisor and execute `fdiv`; keeping
/// the division here, rather than multiplying by [`DEG_TO_RAD`], preserves that rounding point.
#[inline]
pub fn model_radians(degrees: f32) -> f32 {
    degrees / DEG_PER_RAD
}

/// The client-local state consumed by `ModelDragon` and `RenderDragon`.
///
/// java: `EntityDragon.ringBuffer`, `ringBufferIndex`, `prevAnimTime` and `animTime`.  None of
/// these values are sent by the play protocol: vanilla advances them on the client from the
/// interpolated entity position and motion.  Keeping the state as an explicit render payload
/// makes the model usable without smuggling a second mutable dragon entity into the shared model
/// cache.  The live tracker/driver still owns producing one value per tracked dragon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragonRenderState {
    /// java: `EntityDragon.prevAnimTime`.
    pub prev_anim_time: f32,
    /// java: `EntityDragon.animTime`.
    pub anim_time: f32,
    /// java: `EntityDragon.ringBuffer`, in `[yaw, y, unused]` order.
    ///
    /// The source stores doubles.  Keep that width here even though the model ultimately narrows
    /// the angles to `f32`, because the ring is also read by `RenderDragon.rotateCorpse`.
    pub ring_buffer: [[f64; 3]; 64],
    /// java: `EntityDragon.ringBufferIndex`, normally `-1` until the first client tick.
    pub ring_buffer_index: i32,
    /// Whether `EntityDragon.getMovementOffsets` should use a zero partial tick.  Java does this
    /// when `getHealth() <= 0`, so the dying dragon uses the latest ring sample rather than the
    /// frame interpolation path.
    pub dead: bool,
    /// java: `EntityDragon.deathTicks`, advanced by the dragon's override of `onDeathUpdate`.
    ///
    /// Unlike ordinary living entities this is a 200-tick client-visible death sequence.  It is
    /// kept beside the ring buffer because both `ModelDragon` and `RenderDragon` consume the
    /// generation-local value, and a network id can be reused while the old corpse is fading.
    pub death_ticks: u32,
}

impl Default for DragonRenderState {
    fn default() -> Self {
        Self {
            prev_anim_time: 0.0,
            anim_time: 0.0,
            ring_buffer: [[0.0; 3]; 64],
            ring_buffer_index: -1,
            dead: false,
            death_ticks: 0,
        }
    }
}

impl DragonRenderState {
    /// A neutral state for a dragon that has no client-local tracker payload yet.
    ///
    /// The zero ring still gives the model a complete, finite body; a live producer can replace
    /// it with [`Self::tick`] state as soon as the dragon's first client tick is available.
    #[must_use]
    pub fn neutral(anim_time: f32) -> Self {
        Self {
            prev_anim_time: anim_time,
            anim_time,
            ring_buffer_index: 0,
            ..Self::default()
        }
    }

    /// Java's `EntityDragon.getMovementOffsets(index, partialTicks)`.
    ///
    /// Only the yaw component is shortest-arc wrapped.  The interpolation factor is deliberately
    /// `1 - partialTicks`, and the index arithmetic is the source's `& 63` rather than a modulo
    /// operation so the initial `-1` index has the same ring slot as Java.
    #[must_use]
    pub fn movement_offset(&self, offset: usize, partial_ticks: f32) -> [f64; 3] {
        let partial_ticks = if self.dead { 0.0 } else { partial_ticks };
        let blend = f64::from(1.0 - partial_ticks);
        let i = ((self.ring_buffer_index - offset as i32) & 63) as usize;
        let j = ((self.ring_buffer_index - offset as i32 - 1) & 63) as usize;

        let current = self.ring_buffer[i];
        let previous = self.ring_buffer[j];
        let yaw_delta = mc_math::wrap_degrees_f64(previous[0] - current[0]);
        [
            current[0] + yaw_delta * blend,
            current[1] + (previous[1] - current[1]) * blend,
            current[2] + (previous[2] - current[2]) * blend,
        ]
    }

    /// Writes one client-side ring sample using Java's first-fill and increment order.
    ///
    /// The caller should invoke this only on the living/non-NoAI branch, matching
    /// `EntityDragon.onLivingUpdate`: a dead dragon does not append samples, and NoAI pins the
    /// animation phase without touching the ring.
    pub fn record_sample(&mut self, yaw: f32, pos_y: f64) {
        if self.ring_buffer_index < 0 {
            for sample in &mut self.ring_buffer {
                sample[0] = f64::from(yaw);
                sample[1] = pos_y;
                sample[2] = 0.0;
            }
        }
        self.ring_buffer_index = (self.ring_buffer_index + 1) & 63;
        self.ring_buffer[self.ring_buffer_index as usize] = [f64::from(yaw), pos_y, 0.0];
    }

    /// Advances `prevAnimTime`/`animTime` and records the current ring sample in Java order.
    ///
    /// `motion_*` are the entity's client-side double motion fields.  `dead` corresponds to
    /// `getHealth() <= 0`; `no_ai` corresponds to `isAIDisabled()`.  This is deliberately a pure
    /// render-state helper: particle, sound, and entity-position side effects remain outside
    /// `mc-render`.
    // The parameter order mirrors RendererLivingEntity's Java model callback; bundling it would
    // make the transliterated call shape less auditable.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        yaw: f32,
        pos_y: f64,
        motion_x: f64,
        motion_y: f64,
        motion_z: f64,
        slowed: bool,
        dead: bool,
        no_ai: bool,
    ) {
        self.dead = dead;
        self.prev_anim_time = self.anim_time;
        if dead {
            return;
        }
        if no_ai {
            self.anim_time = 0.5;
            return;
        }

        // java: `0.2F / (MathHelper.sqrt_double(motionX * motionX + motionZ * motionZ)
        // * 10.0F + 1.0F)`, then `* (float)Math.pow(2.0D, motionY)`.
        let mut delta =
            0.2_f32 / (mc_math::sqrt_d(motion_x * motion_x + motion_z * motion_z) * 10.0 + 1.0);
        delta *= 2.0_f64.powf(motion_y) as f32;
        if slowed {
            delta *= 0.5;
        }
        self.anim_time += delta;
        self.record_sample(yaw, pos_y);
    }
}

/// The client-only camera sway state on `EntityOtherPlayerMP`.
///
/// java: `EntityOtherPlayerMP.onLivingUpdate` (`:112-133`) writes `prevCameraYaw`,
/// `cameraYaw` and `cameraPitch` from the entity's motion. These fields are not a movement
/// integrator: the remote player's position remains packet/interpolation driven. Vanilla's
/// `EntityRenderer.setupViewBobbing` and `LayerCape` consume the yaw value when this entity is
/// the render-view/player entity, so keep the state in the render payload instead of dropping the
/// otherwise camera-only S12 consumer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RemotePlayerCameraState {
    /// java: `EntityPlayer.prevCameraYaw`.
    pub prev_camera_yaw: f32,
    /// java: `EntityPlayer.cameraYaw`.
    pub camera_yaw: f32,
    /// java: `EntityLivingBase.prevCameraPitch`.
    pub prev_camera_pitch: f32,
    /// java: `EntityLivingBase.cameraPitch`.
    pub camera_pitch: f32,
}

impl RemotePlayerCameraState {
    /// Advances the remote player's camera fields for one client tick.
    ///
    /// `motion_*` are the live Java `motionX/Y/Z` values after packet writes. The order and
    /// narrowing points mirror `EntityOtherPlayerMP.onLivingUpdate`: horizontal motion is
    /// `MathHelper.sqrt_double` (a float result), pitch casts `Math.atan` before multiplying,
    /// and the ground/dead gates happen before the two spring updates.
    pub fn tick(
        &mut self,
        motion_x: f64,
        motion_y: f64,
        motion_z: f64,
        on_ground: bool,
        dead: bool,
    ) {
        self.prev_camera_yaw = self.camera_yaw;
        self.prev_camera_pitch = self.camera_pitch;

        let mut yaw = mc_math::sqrt_d(motion_x * motion_x + motion_z * motion_z);
        let mut pitch = (-(motion_y * 0.200_000_002_980_232_24_f64)).atan() as f32 * 15.0;

        if yaw > 0.1 {
            yaw = 0.1;
        }
        if !on_ground || dead {
            yaw = 0.0;
        }
        if on_ground || dead {
            pitch = 0.0;
        }

        self.camera_yaw += (yaw - self.camera_yaw) * 0.4;
        self.camera_pitch += (pitch - self.camera_pitch) * 0.8;
    }

    /// The frame-interpolated `(cameraYaw, cameraPitch)` pair used by vanilla camera/cape code.
    #[must_use]
    pub fn interpolated(self, partial_ticks: f32) -> (f32, f32) {
        (
            self.prev_camera_yaw + (self.camera_yaw - self.prev_camera_yaw) * partial_ticks,
            self.prev_camera_pitch + (self.camera_pitch - self.prev_camera_pitch) * partial_ticks,
        )
    }
}

/// The three world-space vectors `ModelGuardian` reads while aiming its movable eye.
///
/// java: `ModelGuardian.setRotationAngles` asks the watched target (or render-view entity) and
/// the Guardian for `getPositionEyes(0.0F)`, then asks the Guardian for `getLook(0.0F)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuardianLookAt {
    /// Guardian eye position at partial tick zero.
    pub self_eyes: [f64; 3],
    /// Watched-target eyes, or render-view eyes when there is no watched target.
    pub target_eyes: [f64; 3],
    /// Guardian look vector at partial tick zero.
    pub self_look: [f64; 3],
}

/// Per-frame entity facts that model bodies read directly off the entity in the Java.
///
/// Vanilla's `setRotationAngles` takes an `Entity` and downcasts; this crate cannot depend on a
/// live entity object, so the fields models actually consult are gathered here by whoever drives
/// the render. Everything defaults to the "idle adult standing still" case.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EntityAnimState {
    /// java: `Entity.isSneaking()`
    pub sneaking: bool,
    /// java: `Entity.isSprinting()`
    pub sprinting: bool,
    /// java: `Entity.isRiding()`
    pub riding: bool,
    /// java: `EntityLivingBase.isChild()`
    pub child: bool,
    /// java: `Entity.isInvisible()`
    pub invisible: bool,
    /// java: `EntityLivingBase.ticksExisted`
    pub ticks_existed: u32,
    /// java: `EntityCreeper.getCreeperFlashIntensity(partialTicks)`.
    ///
    /// This is client-tick fuse progress, not entity age. `RenderCreeper` consumes the same
    /// interpolated value for its non-uniform pre-render scale and white flash multiplier.
    pub creeper_flash: f32,
    /// java: `EntityEnderCrystal.innerRotation`, rolled once from the crystal's persistent
    /// `Entity.rand` stream by its constructor. Other entity kinds leave this at zero.
    pub crystal_inner_rotation: f32,
    /// Attack-swing progress in `0..1`. java: `EntityLivingBase.getSwingProgress`
    pub swing_progress: f32,
    /// Aggro/attack pose: creeper ignition, spider/enderman aggression, zombie arms out.
    /// java: various `isAggressive`/`isAttacking`/`isArmsRaised` predicates.
    pub aggressive: bool,
    /// java: `EntityZombie.isConverting` — watched index 14, used by
    /// `RenderZombie.rotateCorpse` for its small curing shake.
    pub converting: bool,
    /// A per-species interpolated float that is *not* `handleRotationFloat`: the wolf's
    /// `getInterestedAngle`, the rabbit's `func_175521_o` jump phase, the sheep's grazing head
    /// angle and the guardian's tail phase all land here, in radians.
    ///
    /// java: `EntityWolf.getInterestedAngle(partialTicks)` and friends. This knob is *not* the
    /// wolf's tail angle — that is [`EntityAnimState::wolf_tail`], which reaches the model
    /// through [`AnimParams::age_in_ticks`] because vanilla routes it through
    /// `RenderWolf.handleRotationFloat`. Writing `getTailRotation()` here instead rolls every
    /// wolf's head sideways by up to 1.54 rad, which is what mc-rs did before the
    /// `handleRotationFloat` seam existed.
    pub tail_rotation: f32,
    /// java: `EntityWolf.getTailRotation` (`EntityWolf:495`) — the angle
    /// `RenderWolf.handleRotationFloat` (`RenderWolf:24`) substitutes for `ageInTicks`, so it is
    /// what `ModelWolf` writes into `wolfTail.rotateAngleX`. Radians.
    ///
    /// Read by [`LivingRenderState::params`] when [`LivingRenderState::species`] is
    /// [`SpeciesState::Wolf`]; a model never reads it directly, exactly as no vanilla model
    /// reads `getTailRotation` directly.
    pub wolf_tail: f32,
    /// java: `EntityBat.getIsBatHanging` — DataWatcher index 16 bit 0.
    ///
    /// `ModelBat.setRotationAngles` forks its whole body on this. It is deliberately **not**
    /// [`EntityAnimState::sitting`]: `isSitting` is `EntityTameable`'s index-16 bit 0, a
    /// different index on a different class hierarchy, and a bat is not tameable.
    pub bat_hanging: bool,
    /// java: `EntityLivingBase.getArrowCountInEntity` — DataWatcher index 9.
    /// `LayerArrow.doRenderLayer` draws this many arrows stuck in the body and returns
    /// immediately when it is zero.
    pub arrow_count: i32,
    /// java: `EntitySlime`'s `prevSquishFactor + (squishFactor - prevSquishFactor) * partialTicks`
    /// — `ModelMagmaCube.setLivingAnimations` and `RenderSlime.preRenderCallback` both read it.
    ///
    /// The two samples are always equal in vanilla (`EntitySlime.onUpdate` assigns
    /// `prevSquishFactor = squishFactor` immediately after advancing it, which `javap -c` on
    /// `wb.t_()` confirms is not a decompile artifact), so this carries the single value and the
    /// interpolation is correctly a no-op.
    pub squish_factor: f32,
    /// java: `prevSquidPitch + (squidPitch - prevSquidPitch) * partialTicks`, in degrees.
    /// `RenderSquid.rotateCorpse` applies this around X before the ordinary model flip.
    pub squid_pitch: f32,
    /// java: `prevSquidYaw + (squidYaw - prevSquidYaw) * partialTicks`, in degrees.
    /// `RenderSquid.rotateCorpse` applies this around Y after [`Self::squid_pitch`].
    pub squid_yaw: f32,
    /// java: `EntityGuardian.func_175469_o(partialTicks)` — the interpolated spike extension in
    /// `0..1`, which `ModelGuardian.setRotationAngles` turns into
    /// `f1 = (1 - extension) * 0.55` and subtracts from every spine's pivot.
    ///
    /// `0.0` is vanilla's start value (`private float field_175485_bl;`, never assigned in the
    /// constructor), i.e. spines pulled in; a guardian sitting still in water eases it to `1.0`
    /// over a couple of seconds.
    pub spikes_animation: f32,
    /// java: `ModelGuardian.setRotationAngles`' target/render-view eye block.
    /// `None` is neutral for non-Guardians and for an unresolved nonzero watched target.
    pub guardian_look_at: Option<GuardianLookAt>,
    /// java: `EntityWolf.getShakeAngle`'s interpolated `timeWolfIsShaking`, before the `/ 1.8`
    /// and the per-part offset that `ModelWolf.setRotationAngles` applies. `0.0` is a dry wolf.
    pub wolf_shake_time: f32,
    /// `EntityWither`'s two local side-head yaw samples, in degrees **relative to body yaw**.
    /// Java derives their absolute yaw from watched target ids 18 and 19, then subtracts
    /// `renderYawOffset`; the centre head uses [`AnimParams::net_head_yaw`].
    pub wither_side_head_yaw: [f32; 2],
    /// `EntityWither`'s two local side-head pitch samples, in degrees.
    pub wither_side_head_pitch: [f32; 2],
    /// java: `EntityDragon`'s client-local animation/ring-buffer state, supplied to `ModelDragon`.
    ///
    /// The network metadata has no dragon-specific slots, so this remains optional until the
    /// tracker has run the dragon's client tick.  `None` is a neutral, fully-renderable dragon.
    pub dragon: Option<DragonRenderState>,
    /// java: `EntityOtherPlayerMP`'s client-only camera sway fields.
    ///
    /// This is present only for remote players. It is not a position or body-animation input;
    /// the remote player still uses packet interpolation, while a camera owner may consume the
    /// frame values through [`RemotePlayerCameraState::interpolated`].
    pub remote_camera: Option<RemotePlayerCameraState>,
    /// java: `EntityArmorStand`'s six `Rotations` slots (indices 11–16) and the display bits of
    /// its index-10 flag byte, which `ModelArmorStandArmor.setRotationAngles` reads by
    /// downcasting its `Entity` argument.
    ///
    /// `None` means "not an armour stand", and the model then keeps the `DEFAULT_*_ROTATION`
    /// pose its constructor built — which is what every stand rendered as while this had no
    /// producer at all.
    pub armor_stand: Option<super::models::object::ArmorStandPose>,
    /// java: `EntitySheep.getHeadRotationAngleX(partialTicks)` (`:339`) — the sheep's
    /// head pitch in radians, read by `ModelSheep2.setLivingAnimations` (`:78`).
    ///
    /// Not derivable inside the model: the grazing branch needs `sheepTimer`, and the
    /// resting branch needs the entity's **raw** `rotationPitch`, not the interpolated
    /// pitch the renderer hands `setRotationAngles`. Filled by
    /// [`LivingRenderState::params`] from [`SpeciesState::Sheep`].
    pub sheep_head_angle_x: f32,
    /// java: `EntitySheep.getHeadRotationPointY(partialTicks)` (`:353`) — `0..1`, which
    /// `ModelSheep2.setLivingAnimations` turns into `head.rotationPointY = 6 + it * 9`.
    pub sheep_head_point_y: f32,
    /// java: `EntitySheep.getSheared`
    pub sheared: bool,
    /// java: `EntityEnderman.getCarrying` != null
    pub carrying_block: bool,
    /// java: `EntityTameable.isSitting`
    pub sitting: bool,
    /// java: `EntityWolf.isBegging` — DataWatcher index 19.
    ///
    /// Not read by a model directly: it drives `EntityWolf.onUpdate`'s `headRotationCourse`
    /// spring ([`WolfAnim`]), whose interpolated value reaches `ModelWolf.setLivingAnimations`
    /// as [`EntityAnimState::tail_rotation`].
    pub begging: bool,
    /// Iron-golem attack timer in ticks. java: `EntityIronGolem.getAttackTimer`
    pub attack_timer: f32,
    /// java: `EntityIronGolem.holdRoseTick` (`:43`) — latched to `400` by `handleStatusUpdate`
    /// opcode 11 (`:197-200`) and counted down one per tick by `onLivingUpdate` (`:128-131`).
    /// `LayerIronGolemFlower.shouldCombineTextures` gates the poppy on `getHoldRoseTick() != 0`.
    ///
    /// This is the *only* source: the flag is never a watched value, so a client that drops
    /// opcode 11 can never draw the poppy, and one that gates on `isPlayerCreated` instead draws
    /// it permanently.
    pub hold_rose_ticks: u32,
    /// This entity's network id. java: `Entity.getEntityId()`.
    ///
    /// Read by `ModelWitch.setRotationAngles`, which derives the nose-twitch frequency from
    /// `0.01F * (entityIn.getEntityId() % 10)` precisely so that a coven of witches does not
    /// twitch in lockstep. It lived on `WitchModel` until now, which is the shared-model defect
    /// class — one `WitchModel` serves every witch — and nothing wrote it, so every witch ran at
    /// frequency zero and her nose was frozen.
    pub entity_id: i32,
    /// java: `ModelWitch.field_82900_g`, assigned by `RenderWitch.doRender` (`:25`) from
    /// `entity.getHeldItem() != null` immediately before `super.doRender` — the witch tips her
    /// nose down to drink. Per-entity, so it belongs here rather than on the shared model.
    pub witch_holding_item: bool,
    /// java: `EntityVillager.getProfession`, `EntityHorse` variant, `EntitySlime` size, ...
    pub variant: i32,
    /// java: `ModelBiped.aimedBow`
    pub aimed_bow: bool,
    /// java: `ModelBiped.heldItemRight` — 0 none, 1 item, 3 block held aloft.
    pub held_item_right: i32,
    /// java: `ModelBiped.heldItemLeft`
    pub held_item_left: i32,
}

/// `EntityChicken`'s four wing-flap fields.
///
/// java: `EntityChicken.java:26-30` (`wingRotation`, `destPos`, `field_70884_g`,
/// `field_70888_h`, `wingRotDelta`) advanced by `onLivingUpdate` (`:68-94`). **None of that is
/// `isRemote`-gated**, so a vanilla client runs it for every remote chicken it can see; the
/// server never sends any of it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChickenAnim {
    /// java: `wingRotation` — an unbounded phase accumulator fed to `sin`.
    pub wing_rotation: f32,
    /// java: `field_70888_h`, the previous tick's `wingRotation`.
    pub prev_wing_rotation: f32,
    /// java: `destPos`, clamped to `0..1` — how far "open" the flap is.
    pub dest_pos: f32,
    /// java: `field_70884_g`, the previous tick's `destPos`.
    pub prev_dest_pos: f32,
    /// java: `wingRotDelta`, initialised to `1.0F` and multiplied by `0.9` every tick.
    pub wing_rot_delta: f32,
}

impl ChickenAnim {
    /// java: `EntityChicken`'s field initialisers — `wingRotDelta = 1.0F`, everything else `0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            wing_rot_delta: 1.0,
            ..Self::default()
        }
    }

    /// java: `RenderChicken.handleRotationFloat` (`:28-33`) —
    /// `(sin(lerp(field_70888_h, wingRotation)) + 1) * lerp(field_70884_g, destPos)`.
    #[must_use]
    pub fn rotation_float(&self, partial: f32) -> f32 {
        let f = self.prev_wing_rotation + (self.wing_rotation - self.prev_wing_rotation) * partial;
        let f1 = self.prev_dest_pos + (self.dest_pos - self.prev_dest_pos) * partial;
        (mc_math::sin(f) + 1.0) * f1
    }
}

/// `EntitySquid`'s interpolated swim state.
///
/// java: `EntitySquid.java:15-31` and `onLivingUpdate` (`:138-217`). The client half runs
/// unconditionally; only the three `motion*` writes and the `squidRotation` wrap are
/// `!isRemote`-gated, and the client's wrap branch instead **pins** `squidRotation` to `2*PI`
/// and waits for status opcode 19 to reset it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SquidAnim {
    /// java: `squidRotation` — radians, pinned at `2*PI` on the client until opcode 19.
    pub rotation: f32,
    /// java: `prevSquidRotation`. Kept because vanilla keeps it; nothing reads it yet.
    pub prev_rotation: f32,
    /// java: `tentacleAngle`, in `0..=PI/4`.
    pub tentacle_angle: f32,
    /// java: `lastTentacleAngle`.
    pub last_tentacle_angle: f32,
    /// java: `squidYaw` — the roll `RenderSquid.rotateCorpse` applies about Y.
    pub yaw: f32,
    /// java: `prevSquidYaw`.
    pub prev_yaw: f32,
    /// java: `squidPitch` — degrees; relaxes to `-90` out of water.
    pub pitch: f32,
    /// java: `prevSquidPitch`.
    pub prev_pitch: f32,
    /// java: `rotationVelocity`, seeded from `new Random(1 + entityId)` in the constructor.
    pub rotation_velocity: f32,
    /// java: `field_70871_bB`, the swim-stroke envelope that drives `squidYaw`.
    pub stroke: f32,
}

impl SquidAnim {
    /// java: `EntitySquid`'s constructor — `rand.setSeed(1 + getEntityId());
    /// rotationVelocity = 1.0F / (rand.nextFloat() + 1.0F) * 0.2F;`
    #[must_use]
    pub fn new(network_id: i32) -> Self {
        let mut rand = mc_math::JavaRandom::new(i64::from(network_id) + 1);
        Self {
            rotation_velocity: 1.0 / (rand.next_f32() + 1.0) * 0.2,
            ..Self::default()
        }
    }

    /// java: `RenderSquid.handleRotationFloat` (`:39-42`) — the interpolated tentacle angle,
    /// which `ModelSquid.setRotationAngles` writes into all eight tentacles' `rotateAngleX`.
    #[must_use]
    pub fn rotation_float(&self, partial: f32) -> f32 {
        self.last_tentacle_angle + (self.tentacle_angle - self.last_tentacle_angle) * partial
    }

    /// java: `RenderSquid.rotateCorpse` (`:25-33`) — the two interpolated angles, in degrees.
    #[must_use]
    pub fn corpse_rotation(&self, partial: f32) -> (f32, f32) {
        (
            self.prev_pitch + (self.pitch - self.prev_pitch) * partial,
            self.prev_yaw + (self.yaw - self.prev_yaw) * partial,
        )
    }
}

/// `EntityWolf`'s beg head-tilt spring.
///
/// `EntityWolf`'s beg head-tilt spring and its wet-dog shake.
///
/// java: `EntityWolf.onUpdate` (`:226-278`), all of which runs on the client too — the server
/// contributes exactly one thing, `setEntityState(this, (byte)8)`, which arrives as
/// `S19PacketEntityStatus` opcode 8 and is the only way `isShaking` is ever set.
///
/// The complete `Entity.isWet()` predicate is resolved by the entity tick's weather-aware entry
/// point: `inWater || isRainingAt(feet) || isRainingAt(head)`. `World` owns sky visibility,
/// precipitation height and the dimension-selected biome; the session supplies the live rain
/// gate. This render state stores the resulting latched `is_wet` value, rather than re-running
/// world queries during frame rendering.
///
/// `Entity.isWet()` (`:1094`) is
/// `inWater || isRainingAt(feet) || isRainingAt(head)`; both rain probes are wired at tick time.
/// `World.isRainingAt` (`:3567`) additionally needs `isRaining()` — which lives on
/// `ClientState::env`, not on the entity tracker — plus `canSeeSky`, the precipitation
/// heightmap and the biome's `canRain`/`getEnableSnow`. The session supplies the live weather
/// gate to the tracker, while render frames only consume this latched state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WolfAnim {
    /// java: `isWet` — the latched field, not the `isWet()` predicate. Set while the wolf is in
    /// water and cleared when the shake finishes.
    pub is_wet: bool,
    /// java: `isShaking`. Set **only** by status opcode 8; the client never decides it itself.
    pub is_shaking: bool,
    /// java: `timeWolfIsShaking` — climbs by `0.05` a tick and resets past `2.0`.
    pub shake_time: f32,
    /// java: `prevTimeWolfIsShaking`.
    pub prev_shake_time: f32,
    /// java: `headRotationCourse`
    pub head_rotation_course: f32,
    /// java: `headRotationCourseOld`
    pub prev_head_rotation_course: f32,
}

impl WolfAnim {
    /// java: the interpolated `timeWolfIsShaking` that `EntityWolf.getShakeAngle` (`:297`)
    /// computes before dividing by `1.8`.
    #[must_use]
    pub fn shake_time(&self, partial: f32) -> f32 {
        self.prev_shake_time + (self.shake_time - self.prev_shake_time) * partial
    }

    /// java: `EntityWolf.getInterestedAngle` (`:313`) — the lerp, times `0.15 * PI`.
    #[must_use]
    pub fn interested_angle(&self, partial: f32) -> f32 {
        (self.prev_head_rotation_course
            + (self.head_rotation_course - self.prev_head_rotation_course) * partial)
            * 0.15
            * core::f32::consts::PI
    }
}

/// Everything a sheep carries across ticks: one countdown.
///
/// java: `EntitySheep.sheepTimer`. `handleStatusUpdate(10)` (`:390`) sets it to 40 and
/// `onLivingUpdate` (`:196`) decrements it toward zero **inside** `if (worldObj.isRemote)`
/// — it is a client-only animation clock with no watched value behind it, which is why
/// nothing in mc-rs was writing it and no sheep ever lowered its head.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SheepAnim {
    /// java: `EntitySheep.sheepTimer` — 40 ticks, counted down to 0.
    pub timer: i32,
}

impl SheepAnim {
    /// java: `EntitySheep.getHeadRotationPointY(float)` (`:353`).
    #[must_use]
    pub fn head_rotation_point_y(self, partial: f32) -> f32 {
        let t = self.timer as f32;
        if self.timer <= 0 {
            0.0
        } else if (4..=36).contains(&self.timer) {
            1.0
        } else if self.timer < 4 {
            (t - partial) / 4.0
        } else {
            -((t - 40.0) - partial) / 4.0
        }
    }

    /// java: `EntitySheep.getHeadRotationAngleX(float)` (`:339`), in radians.
    ///
    /// `raw_pitch` is `Entity.rotationPitch` — the **un**interpolated one. Vanilla reads
    /// the field directly here rather than the `f7` the renderer computed, so a sheep
    /// that is turning its head lags the rest of its body by up to one tick. mc-rs used
    /// `AnimParams::head_pitch` (the interpolated value), which is a real divergence even
    /// though it is a small one.
    #[must_use]
    pub fn head_rotation_angle_x(self, partial: f32, raw_pitch: f32) -> f32 {
        if self.timer > 4 && self.timer <= 36 {
            let f = ((self.timer - 4) as f32 - partial) / 32.0;
            core::f32::consts::PI / 5.0
                + (core::f32::consts::PI * 7.0 / 100.0) * mc_math::sin(f * 28.7)
        } else if self.timer > 0 {
            core::f32::consts::PI / 5.0
        } else {
            // java: `this.rotationPitch / (180F / (float)Math.PI)` — a division, so this
            // one site is spelled vanilla's way rather than through `DEG_TO_RAD`.
            raw_pitch / DEG_PER_RAD
        }
    }
}

/// `EntitySlime`'s squash-and-stretch spring, shared with `EntityMagmaCube`.
///
/// java: `EntitySlime.java:30-33` and `onUpdate` (`:115-157`). `prevSquishFactor` is deliberately
/// absent: vanilla assigns it from `squishFactor` in the same statement pair, so the two are
/// always equal by the time any renderer reads them (bytecode-checked — see
/// [`EntityAnimState::squish_factor`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlimeAnim {
    /// java: `squishAmount` — the target the spring chases; kicked to `1.0` on take-off and
    /// `-0.5` on landing, then decayed by `alterSquishAmount`.
    pub squish_amount: f32,
    /// java: `squishFactor`
    pub squish_factor: f32,
    /// java: `wasOnGround`
    pub was_on_ground: bool,
}

/// `EntityGuardian`'s client-side tail and spike animation.
///
/// java: `EntityGuardian.java:298-355` — the whole `if (this.worldObj.isRemote)` block, which
/// is the only place these five fields are written. The server never sends any of them.
///
/// # What is deliberately left out
///
/// * **`field_175480_bp`**, the ground latch for `mob.guardian.flop`. It is a client-audio
///   concern rather than render animation and is retained by `TrackedEntity` beside the
///   status-effect queue, so this `Copy` render state stays geometry-only.
/// * **`field_175479_bo`, the beam charge counter.** It is retained by `TrackedEntity` and feeds
///   both the laser quad's charge state and the particle seam's bubble-trail spacing.
/// * The two `spawnParticle` loops, which are consumed by the client particle seam after the
///   ordered entity tick.
///
/// These values stay out of this geometry-only `Copy` state so the entity/audio/particle owners
/// can preserve their own tick ordering and RNG streams.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuardianAnim {
    /// java: `field_175482_b` — an unbounded phase accumulator fed to `sin` by the tail links.
    pub tail_phase: f32,
    /// java: `field_175484_c`, the previous tick's `field_175482_b`.
    pub prev_tail_phase: f32,
    /// java: `field_175483_bk` — how much `tail_phase` advances this tick. `2.0` while
    /// beached, a spring towards `0.5` while swimming (kicked to `4.0` from a standing start)
    /// and towards `0.125` while hovering.
    pub tail_speed: f32,
    /// java: `field_175485_bl` — spike extension in `0..1`.
    pub spikes: f32,
    /// java: `field_175486_bm`, the previous tick's `field_175485_bl`.
    pub prev_spikes: f32,
    /// A private 48-bit fallback stream for standalone animation construction.
    ///
    /// The live client entity tick uses its persistent `TrackedEntity::particle_random` instead,
    /// so the fallback remains isolated from status/particle RNG ordering.
    noise: i64,
}

impl GuardianAnim {
    /// java: `EntityGuardian(World)` — the constructor consumes one `Entity.rand.nextFloat()`
    /// and assigns that same value to both `field_175482_b` and `field_175484_c`. Standalone
    /// render tests do not own the entity's random stream, so this compatibility constructor uses
    /// the pre-randomized zero value; live client construction uses [`Self::with_initial_tail`].
    #[must_use]
    pub fn new(network_id: i32) -> Self {
        Self::with_initial_tail(network_id, 0.0)
    }

    /// Construct a Guardian with the value consumed by Java's constructor `nextFloat()` draw.
    ///
    /// `tail_phase` and `prev_tail_phase` intentionally receive one identical `f32`: Java's
    /// chained assignment is `field_175484_c = field_175482_b = rand.nextFloat()`, not two draws.
    /// The live tracker supplies the value from its persistent `Entity.rand` stream.
    #[must_use]
    pub fn with_initial_tail(network_id: i32, initial_tail: f32) -> Self {
        Self {
            tail_phase: initial_tail,
            prev_tail_phase: initial_tail,
            tail_speed: 0.0,
            spikes: 0.0,
            prev_spikes: 0.0,
            noise: (i64::from(network_id) ^ 0x5_DEEC_E66D) & ((1 << 48) - 1),
        }
    }

    /// java: `java.util.Random.nextFloat()` — `next(24) / (float)(1 << 24)`.
    pub fn next_float(&mut self) -> f32 {
        self.noise = self.noise.wrapping_mul(0x5_DEEC_E66D).wrapping_add(0xB) & ((1 << 48) - 1);
        #[allow(clippy::cast_possible_truncation)]
        let bits = (self.noise >> (48 - 24)) as i32;
        bits as f32 / (1u32 << 24) as f32
    }

    /// java: `EntityGuardian.func_175471_a` (`:411`) — the interpolated tail phase, which
    /// `ModelGuardian.setRotationAngles` feeds to `sin` for all three tail links.
    #[must_use]
    pub fn tail_phase(&self, partial: f32) -> f32 {
        self.prev_tail_phase + (self.tail_phase - self.prev_tail_phase) * partial
    }

    /// java: `EntityGuardian.func_175469_o` (`:416`) — the interpolated spike extension.
    #[must_use]
    pub fn spikes(&self, partial: f32) -> f32 {
        self.prev_spikes + (self.spikes - self.prev_spikes) * partial
    }
}

/// Per-species tick state, and with it the `handleRotationFloat` override that reads it.
///
/// java: `RendererLivingEntity.handleRotationFloat` (`:454`) is virtual. The base returns
/// `(float)livingBase.ticksExisted + partialTicks` and **exactly three** renderers override it
/// in 1.8.9 — `RenderChicken:28`, `RenderSquid:39`, `RenderWolf:24` — so for a chicken, a squid
/// and a wolf the third argument of `setRotationAngles` is not an age at all. mc-rs had no hook
/// for that, which is why the chicken's wings, the squid's tentacles and the wolf's tail each
/// advanced by one radian per tick, without bound, forever.
///
/// The variants that are *not* rotation-float overrides (`Slime`, `Guardian`) are here because
/// this is where
/// per-species tick memory belongs: vanilla keeps it in the entity subclass, and mc-rs has no
/// entity subclass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SpeciesState {
    /// Everything with no per-species animation state. `handleRotationFloat` is the base's.
    #[default]
    Generic,
    /// java: `EntityChicken` / `RenderChicken`
    Chicken(ChickenAnim),
    /// java: `EntitySquid` / `RenderSquid`
    Squid(SquidAnim),
    /// java: `EntityWolf` / `RenderWolf`
    Wolf(WolfAnim),
    /// java: `EntitySheep`. No `handleRotationFloat` override; the grazing clock is the
    /// whole of its per-species state.
    Sheep(SheepAnim),
    /// java: `EntitySlime` and `EntityMagmaCube`. No `handleRotationFloat` override.
    Slime(SlimeAnim),
    /// java: `EntityGuardian` / `RenderGuardian`. No `handleRotationFloat` override either —
    /// `ModelGuardian` recovers the frame's partial ticks as `ageInTicks - ticksExisted`, which
    /// only works *because* the base implementation is still in force.
    Guardian(GuardianAnim),
}

impl SpeciesState {
    /// The chicken's state, or `None` if this is not a chicken.
    #[must_use]
    pub fn chicken(&self) -> Option<&ChickenAnim> {
        match self {
            SpeciesState::Chicken(c) => Some(c),
            _ => None,
        }
    }

    /// The sheep's state, or `None`.
    #[must_use]
    pub fn sheep(&self) -> Option<&SheepAnim> {
        match self {
            SpeciesState::Sheep(s) => Some(s),
            _ => None,
        }
    }

    /// The guardian's state, or `None`.
    #[must_use]
    pub fn guardian(&self) -> Option<&GuardianAnim> {
        match self {
            SpeciesState::Guardian(g) => Some(g),
            _ => None,
        }
    }

    /// The squid's state, or `None`.
    #[must_use]
    pub fn squid(&self) -> Option<&SquidAnim> {
        match self {
            SpeciesState::Squid(s) => Some(s),
            _ => None,
        }
    }

    /// The wolf's state, or `None`.
    #[must_use]
    pub fn wolf(&self) -> Option<&WolfAnim> {
        match self {
            SpeciesState::Wolf(w) => Some(w),
            _ => None,
        }
    }

    /// The slime's state, or `None`.
    #[must_use]
    pub fn slime(&self) -> Option<&SlimeAnim> {
        match self {
            SpeciesState::Slime(s) => Some(s),
            _ => None,
        }
    }
}

/// The six arguments of `ModelBase.setRotationAngles`, plus the entity facts models read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimParams {
    /// java: `limbSwing` (`f6`) — a distance accumulator, not an angle.
    pub limb_swing: f32,
    /// java: `limbSwingAmount` (`f5`), `0..1`.
    pub limb_swing_amount: f32,
    /// java: `ageInTicks` (`f8`) — `ticksExisted + partialTicks`, drives idle motion.
    pub age_in_ticks: f32,
    /// java: `netHeadYaw` (`f2`), **degrees**.
    pub net_head_yaw: f32,
    /// java: `headPitch` (`f7`), **degrees**.
    pub head_pitch: f32,
    /// java: `scaleFactor` — always `0.0625` for living entities.
    pub scale: f32,
    /// Partial ticks for this frame, for `setLivingAnimations`.
    pub partial_ticks: f32,
    /// Entity facts. See [`EntityAnimState`].
    pub entity: EntityAnimState,
}

impl AnimParams {
    /// A neutral pose: everything still, `scale = 0.0625`.
    #[must_use]
    pub fn still() -> Self {
        Self {
            scale: 0.0625,
            ..Self::default()
        }
    }
}

/// One entity model: geometry plus its `setRotationAngles` body.
///
/// java: `ModelBase`. Implementors live in [`crate::entity::models`].
pub trait EntityModel: Send {
    /// The part tree.
    fn model(&self) -> &Model;

    /// The part tree, mutably — animation writes angles through this.
    fn model_mut(&mut self) -> &mut Model;

    /// Poses the model. java: `ModelBase.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams);

    /// Entity-dependent pre-pass, run before [`EntityModel::set_rotation_angles`].
    ///
    /// java: `ModelBase.setLivingAnimations` — the squid, the wolf's tail and the ghast's
    /// tentacles use it.
    fn set_living_animations(&mut self, _p: &AnimParams) {}

    /// The part a held item hangs off, if this model has one.
    ///
    /// java: `ModelBiped.postRenderArm` — only `ModelBiped` and its subclasses define it, which
    /// is exactly why `LayerHeldItem` is only ever added to a `RenderBiped`. `None` is the
    /// honest answer for every other model: a creeper has no hand.
    ///
    /// Returning the *right* arm matches `heldItemRight`, the only slot 1.8.9 fills;
    /// [`crate::entity::layers::held_item_matrix`] takes the id as an argument so a caller with
    /// an off-hand item can pass the left one instead.
    fn held_item_arm(&self) -> Option<super::model::PartId> {
        None
    }

    /// Emits geometry under `base`. java: `ModelBase.render`
    ///
    /// The default draws every root part, which is what most models do. Models with their own
    /// matrix juggling — the child-scaling branch of `ModelBiped`, the slime's two passes —
    /// override it.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model().emit_all(p.scale, base, out);
    }

    /// Poses and draws in one call, the sequence `RendererLivingEntity.doRender` uses.
    fn pose_and_render(&mut self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.pose_only(p);
        self.render(p, base, out);
    }

    /// Everything [`Self::pose_and_render`] does to the shared model, without emitting.
    ///
    /// java: `RendererLivingEntity.doRender` runs `setLivingAnimations`/`setRotationAngles` on
    /// the main model *before* `renderModel`'s invisibility fork, so an invisible body that draws
    /// nothing still leaves its pose for `renderLayers` (`postRenderArm`, the head attachments).
    /// [`Self::render`] takes `&self`, so the model state after this call is exactly the state
    /// `pose_and_render` leaves behind.
    fn pose_only(&mut self, p: &AnimParams) {
        self.model_mut().swing_progress = p.entity.swing_progress;
        self.model_mut().is_riding = p.entity.riding;
        self.model_mut().is_child = p.entity.child;
        self.set_living_animations(p);
        self.set_rotation_angles(p);
    }
}

/// Interpolates between two angles the short way round, in degrees.
///
/// java: `RendererLivingEntity.interpolateRotation` — note the loop form: it normalises the
/// *delta* into `[-180, 180)` and then lerps, so a body that spins past 180 degrees in one tick
/// takes the short arc rather than unwinding.
#[must_use]
pub fn interpolate_rotation(prev: f32, cur: f32, partial: f32) -> f32 {
    let mut f = cur - prev;
    while f < -180.0 {
        f += 360.0;
    }
    while f >= 180.0 {
        f -= 360.0;
    }
    prev + partial * f
}

/// Everything `RendererLivingEntity` tracks across ticks so it can interpolate a frame.
///
/// The tick loop writes the `*_prev` fields at the top of each tick and the live fields from
/// packets; the render loop calls [`LivingRenderState::params`] with the frame's partial ticks.
/// java: the `prevRenderYawOffset`/`prevRotationYawHead`/`prevLimbSwingAmount` family on
/// `EntityLivingBase`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LivingRenderState {
    /// java: `EntityLivingBase.renderYawOffset` — body yaw, degrees.
    pub body_yaw: f32,
    /// java: `prevRenderYawOffset`
    pub prev_body_yaw: f32,
    /// The mounted vehicle's raw `renderYawOffset` sample, when this entity is riding another
    /// living entity. `RendererLivingEntity.doRender` temporarily substitutes the vehicle's body
    /// yaw for the rider's own body before clamping the head/body delta; keeping the two tick
    /// samples here lets the render path perform that branch without borrowing the vehicle during
    /// a frame.
    pub mount_body_yaw: Option<f32>,
    /// The mounted vehicle's previous raw `renderYawOffset` sample.
    pub prev_mount_body_yaw: Option<f32>,
    /// java: `EntityLivingBase.rotationYawHead` — head yaw, degrees.
    pub head_yaw: f32,
    /// java: `prevRotationYawHead`
    pub prev_head_yaw: f32,
    /// java: `Entity.rotationPitch`, degrees.
    pub pitch: f32,
    /// java: `Entity.prevRotationPitch`
    pub prev_pitch: f32,
    /// Distance accumulator driving limb swing. java: `EntityLivingBase.limbSwing`
    pub limb_swing: f32,
    /// java: `EntityLivingBase.limbSwingAmount`
    pub limb_swing_amount: f32,
    /// java: `prevLimbSwingAmount`
    pub prev_limb_swing_amount: f32,
    /// java: `EntityLivingBase.ticksExisted`
    pub ticks_existed: u32,
    /// java: `EntityLivingBase.deathTime`
    pub death_time: u32,
    /// java: `EntityLivingBase.hurtTime`
    pub hurt_time: u32,
    /// java: `EntityCreeper.lastActiveTime`.
    pub prev_creeper_time_since_ignited: f32,
    /// java: `EntityCreeper.timeSinceIgnited`, clamped to `0..=30` by its client tick.
    pub creeper_time_since_ignited: f32,
    /// java: `EntityLivingBase.swingProgress`
    pub swing_progress: f32,
    /// java: `EntityLivingBase.prevSwingProgress`
    pub prev_swing_progress: f32,
    /// Everything else models read. `child` here also drives the limb-swing tripling.
    pub entity: EntityAnimState,
    /// Per-species tick memory. java: the animation fields vanilla keeps on the `Entity`
    /// subclass — `EntityChicken.wingRotation`, `EntitySquid.tentacleAngle`,
    /// `EntityWolf.headRotationCourse`, `EntitySlime.squishFactor`.
    ///
    /// This is also what selects the `handleRotationFloat` override; see [`SpeciesState`].
    pub species: SpeciesState,
}

impl LivingRenderState {
    /// Copies every live field into its `prev_*` twin. Call once at the **top** of a tick,
    /// before anything moves.
    ///
    /// java: `Entity.onUpdate` (`:421`) and `EntityLivingBase.onUpdate` (`:380`, `:1737`) —
    /// vanilla scatters these assignments across three methods, but they all happen before the
    /// tick's movement and they are what makes the frame lerp in [`Self::params`] meaningful.
    /// The `while` loops vanilla runs afterwards to unwrap `prev*` past ±180 are *not*
    /// reproduced: [`interpolate_rotation`] normalises the delta itself, which is the same
    /// short-arc result without mutating the stored sample.
    pub fn snapshot_previous(&mut self) {
        self.prev_body_yaw = self.body_yaw;
        self.prev_mount_body_yaw = self.mount_body_yaw;
        self.prev_head_yaw = self.head_yaw;
        self.prev_pitch = self.pitch;
        self.prev_limb_swing_amount = self.limb_swing_amount;
        self.prev_swing_progress = self.swing_progress;
        self.prev_creeper_time_since_ignited = self.creeper_time_since_ignited;
    }

    /// Set or clear the raw body-yaw samples supplied by a mounted vehicle.
    ///
    /// This mirrors `RendererLivingEntity`'s `entity.isRiding() && ridingEntity instanceof
    /// EntityLivingBase` branch. The samples are kept separate from `body_yaw`: a rider can
    /// dismount or ride a boat, both of which must immediately fall back to its own body-helper
    /// state.
    pub fn set_mount_body_yaw(&mut self, samples: Option<(f32, f32)>) {
        match samples {
            Some((previous, current)) => {
                self.prev_mount_body_yaw = Some(previous);
                self.mount_body_yaw = Some(current);
            }
            None => {
                self.prev_mount_body_yaw = None;
                self.mount_body_yaw = None;
            }
        }
    }

    /// Vanilla's `getSwingProgress`: a plain lerp of the two tick samples, with the wrap-around
    /// case (`progress < 0`) folded forward by one full swing.
    ///
    /// java: `EntityLivingBase.getSwingProgress`
    #[must_use]
    pub fn swing_progress(&self, partial: f32) -> f32 {
        let mut f = self.swing_progress - self.prev_swing_progress;
        if f < 0.0 {
            f += 1.0;
        }
        self.prev_swing_progress + f * partial
    }

    /// The body yaw the entity is drawn at this frame, in degrees.
    #[must_use]
    // Preserve Java's two ordered comparisons, including its NaN behaviour, rather than using
    // `clamp` (which clippy suggests but obscures the source branch shape).
    #[allow(clippy::manual_clamp)]
    pub fn render_body_yaw(&self, partial: f32) -> f32 {
        let own_body = interpolate_rotation(self.prev_body_yaw, self.body_yaw, partial);
        if !self.entity.riding {
            return own_body;
        }
        let Some((previous_mount, current_mount)) =
            self.prev_mount_body_yaw.zip(self.mount_body_yaw)
        else {
            return own_body;
        };

        // java: RendererLivingEntity.doRender (`:105-123`) starts with the rider's interpolated
        // head, substitutes the vehicle's interpolated body, clamps the relative angle to
        // [-85, 85), then gives very large turns a 20% nudge back towards the head. Returning
        // the effective body here feeds both the outer transform and `net_head_yaw`, preserving
        // the same branch without changing the app's render ownership.
        let vehicle_body = interpolate_rotation(previous_mount, current_mount, partial);
        let head = interpolate_rotation(self.prev_head_yaw, self.head_yaw, partial);
        let mut relative = mc_math::wrap_degrees_f32(head - vehicle_body);
        if relative < -85.0 {
            relative = -85.0;
        }
        if relative >= 85.0 {
            relative = 85.0;
        }
        let mut effective_body = head - relative;
        if relative * relative > 2_500.0 {
            effective_body += relative * 0.2;
        }
        effective_body
    }

    /// Builds the model arguments for this frame.
    ///
    /// java: `RendererLivingEntity.doRender` lines 99-152, in the same order — the baby
    /// `* 3.0` on the swing happens before the `> 1.0` clamp on the amount, and the head yaw
    /// handed to the model is the head-minus-body difference, not the absolute head yaw.
    #[must_use]
    pub fn params(&self, partial: f32) -> AnimParams {
        let body = self.render_body_yaw(partial);
        let head = interpolate_rotation(self.prev_head_yaw, self.head_yaw, partial);
        // RendererLivingEntity keeps `f2 = head - vehicleBody` from before the +/-85 clamp;
        // only the outer body transform uses the later effective `f`. Preserve that apparently
        // redundant distinction so a mounted rider's model receives the raw head/vehicle delta
        // while its placement follows the clamped body.
        let net_head_yaw = if self.entity.riding {
            self.prev_mount_body_yaw
                .zip(self.mount_body_yaw)
                .map(|(previous_mount, current_mount)| {
                    head - interpolate_rotation(previous_mount, current_mount, partial)
                })
                .unwrap_or(head - body)
        } else {
            head - body
        };
        let pitch = self.prev_pitch + (self.pitch - self.prev_pitch) * partial;

        let mut amount = self.prev_limb_swing_amount
            + (self.limb_swing_amount - self.prev_limb_swing_amount) * partial;
        let mut swing = self.limb_swing - self.limb_swing_amount * (1.0 - partial);
        if self.entity.child {
            swing *= 3.0;
        }
        if amount > 1.0 {
            amount = 1.0;
        }

        let mut entity = self.entity.clone();
        entity.ticks_existed = self.ticks_existed;
        entity.swing_progress = self.swing_progress(partial);
        // java: `EntityCreeper.getCreeperFlashIntensity` divides by `fuseTime - 2` (30 - 2).
        // Do not clamp this frame value: `RenderCreeper.preRenderCallback` uses the raw value
        // for its sine pulse before clamping the fourth-power expansion term.
        entity.creeper_flash = (self.prev_creeper_time_since_ignited
            + (self.creeper_time_since_ignited - self.prev_creeper_time_since_ignited) * partial)
            / 28.0;
        // java: `EntityWolf.getInterestedAngle(partialTicks)`, read by
        // `ModelWolf.setLivingAnimations` as the head's Z roll. Interpolated here rather than in
        // the model because the model has no access to the two tick samples.
        if let SpeciesState::Wolf(w) = &self.species {
            entity.tail_rotation = w.interested_angle(partial);
            entity.wolf_shake_time = w.shake_time(partial);
        }
        // java: `RenderSlime`/`RenderMagmaCube.preRenderCallback` and
        // `ModelMagmaCube.setLivingAnimations` both read
        // `prevSquishFactor + (squishFactor - prevSquishFactor) * partialTicks`, which is
        // `squishFactor` — see `SlimeAnim`.
        if let SpeciesState::Slime(s) = &self.species {
            entity.squish_factor = s.squish_factor;
        }
        // java: `RenderSquid.rotateCorpse` reads the two interpolated entity fields directly.
        // Carry them beside the other per-frame species values so the trait-object placement
        // path does not fall back to its neutral zero-angle squid matrix.
        if let SpeciesState::Squid(s) = &self.species {
            (entity.squid_pitch, entity.squid_yaw) = s.corpse_rotation(partial);
        }
        // java: `ModelGuardian.setRotationAngles` recovers `f = ageInTicks - ticksExisted` (the
        // frame's partial ticks) and calls `func_175471_a(f)` and `func_175469_o(f)` itself.
        // Both are two-sample interpolations the model cannot do, so they are done here and
        // arrive as ordinary `EntityAnimState` fields.
        if let SpeciesState::Guardian(g) = &self.species {
            entity.tail_rotation = g.tail_phase(partial);
            entity.spikes_animation = g.spikes(partial);
        }
        // java: `ModelSheep2.setLivingAnimations` (`:77-79`) calls
        // `getHeadRotationPointY(pt)` and `getHeadRotationAngleX(pt)` on the entity. The
        // second reads the raw `rotationPitch`, which is `self.pitch` here and *not* the
        // interpolated `pitch` local above.
        if let SpeciesState::Sheep(s) = &self.species {
            entity.sheep_head_angle_x = s.head_rotation_angle_x(partial, self.pitch);
            entity.sheep_head_point_y = s.head_rotation_point_y(partial);
        }

        AnimParams {
            limb_swing: swing,
            limb_swing_amount: amount,
            age_in_ticks: self.rotation_float(partial),
            net_head_yaw,
            head_pitch: pitch,
            scale: 0.0625,
            partial_ticks: partial,
            entity,
        }
    }

    /// The third argument of `ModelBase.setRotationAngles`.
    ///
    /// java: `RendererLivingEntity.doRender` (`:130`) — `float f8 =
    /// this.handleRotationFloat(entity, partialTicks);`. The base implementation (`:454`) is
    /// `ticksExisted + partialTicks`; `RenderChicken`, `RenderSquid` and `RenderWolf` override
    /// it, and those three are the only overrides in 1.8.9.
    ///
    /// Naming it `age_in_ticks` on [`AnimParams`] is vanilla's own naming for the *parameter*
    /// (`ModelBase.setRotationAngles(..., float ageInTicks, ...)`); for three species the value
    /// is not an age, which is exactly the trap this method exists to keep in one place.
    #[must_use]
    pub fn rotation_float(&self, partial: f32) -> f32 {
        match &self.species {
            SpeciesState::Chicken(c) => c.rotation_float(partial),
            SpeciesState::Squid(s) => s.rotation_float(partial),
            // java: `RenderWolf.handleRotationFloat` returns `getTailRotation()`, which is a pure
            // function of watched state and so carries no partial-tick term at all.
            SpeciesState::Wolf(_) => self.entity.wolf_tail,
            // java: `RenderSheep` does not override `handleRotationFloat` — the sheep's own
            // per-species state is its grazing clock, not its `ageInTicks`.
            SpeciesState::Generic
            | SpeciesState::Sheep(_)
            | SpeciesState::Slime(_)
            | SpeciesState::Guardian(_) => self.ticks_existed as f32 + partial,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_player_camera_springs_follow_java_ground_and_air_gates() {
        let mut state = RemotePlayerCameraState::default();

        // java: horizontal motion is capped at 0.1, then cameraYaw eases by 0.4 while
        // grounded. Vertical motion is ignored for cameraPitch on the ground.
        state.tick(1.0, -0.5, 0.0, true, false);
        let grounded_yaw = 0.0_f32 + (0.1_f32 - 0.0_f32) * 0.4;
        assert_eq!(state.camera_yaw.to_bits(), grounded_yaw.to_bits());
        assert_eq!(state.camera_pitch.to_bits(), 0.0_f32.to_bits());

        // Airborne motion clears the yaw target but permits the vertical pitch spring. The
        // cast-before-multiply order is the one in EntityOtherPlayerMP's `(float)Math.atan(...)`.
        state.tick(0.0, -0.5, 0.0, false, false);
        let target_pitch = (0.5_f64 * 0.200_000_002_980_232_24_f64).atan() as f32 * 15.0;
        let airborne_yaw = grounded_yaw + (0.0_f32 - grounded_yaw) * 0.4;
        let airborne_pitch = 0.0_f32 + (target_pitch - 0.0_f32) * 0.8;
        assert_eq!(state.prev_camera_yaw.to_bits(), grounded_yaw.to_bits());
        assert_eq!(state.camera_yaw.to_bits(), airborne_yaw.to_bits());
        assert_eq!(state.camera_pitch.to_bits(), airborne_pitch.to_bits());

        let (yaw, pitch) = state.interpolated(0.5);
        assert_eq!(
            yaw.to_bits(),
            (grounded_yaw + (airborne_yaw - grounded_yaw) * 0.5).to_bits()
        );
        assert_eq!(
            pitch.to_bits(),
            (0.0_f32 + (airborne_pitch - 0.0_f32) * 0.5).to_bits()
        );
    }

    #[test]
    fn remote_player_camera_dead_gate_relaxes_both_springs() {
        let mut state = RemotePlayerCameraState {
            camera_yaw: 0.1,
            camera_pitch: 0.5,
            ..Default::default()
        };
        state.tick(1.0, -1.0, 0.0, true, true);
        assert_eq!(
            state.camera_yaw.to_bits(),
            (0.1_f32 + (0.0_f32 - 0.1_f32) * 0.4).to_bits()
        );
        assert_eq!(
            state.camera_pitch.to_bits(),
            (0.5_f32 + (0.0_f32 - 0.5_f32) * 0.8).to_bits()
        );
    }

    #[test]
    fn interpolate_rotation_takes_the_short_arc() {
        assert!((interpolate_rotation(30.0, 50.0, 0.5) - 40.0).abs() < 1e-5);
        // 350 -> 10 is +20 the short way, not -340.
        assert!((interpolate_rotation(350.0, 10.0, 0.5) - 360.0).abs() < 1e-3);
        // Exactly 180 is pushed to -180 by the `>=` in the Java.
        assert!((interpolate_rotation(0.0, 180.0, 1.0) + 180.0).abs() < 1e-3);
        assert!((interpolate_rotation(0.0, 179.0, 1.0) - 179.0).abs() < 1e-3);
    }

    #[test]
    fn limb_swing_is_rewound_by_the_amount() {
        // java: f6 = limbSwing - limbSwingAmount * (1 - partialTicks)
        let s = LivingRenderState {
            limb_swing: 10.0,
            limb_swing_amount: 0.4,
            prev_limb_swing_amount: 0.4,
            ..Default::default()
        };
        let p = s.params(0.0);
        assert!((p.limb_swing - 9.6).abs() < 1e-6);
        let p = s.params(1.0);
        assert!((p.limb_swing - 10.0).abs() < 1e-6);
    }

    #[test]
    fn squid_corpse_angles_reach_frame_params_with_partial_interpolation() {
        let state = LivingRenderState {
            species: SpeciesState::Squid(SquidAnim {
                prev_pitch: -30.0,
                pitch: -70.0,
                prev_yaw: 10.0,
                yaw: 50.0,
                ..Default::default()
            }),
            ..Default::default()
        };

        let params = state.params(0.25);
        assert_eq!(params.entity.squid_pitch, -40.0);
        assert_eq!(params.entity.squid_yaw, 20.0);

        let neutral = LivingRenderState::default().params(0.25);
        assert_eq!(neutral.entity.squid_pitch, 0.0);
        assert_eq!(neutral.entity.squid_yaw, 0.0);
    }

    #[test]
    fn babies_swing_three_times_as_fast_and_the_amount_still_clamps() {
        let s = LivingRenderState {
            limb_swing: 10.0,
            limb_swing_amount: 2.0,
            prev_limb_swing_amount: 2.0,
            entity: EntityAnimState {
                child: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let p = s.params(1.0);
        assert!((p.limb_swing - 30.0).abs() < 1e-6);
        assert!((p.limb_swing_amount - 1.0).abs() < 1e-6, "clamped to 1");
    }

    #[test]
    fn net_head_yaw_is_head_minus_body() {
        let s = LivingRenderState {
            body_yaw: 90.0,
            prev_body_yaw: 90.0,
            head_yaw: 120.0,
            prev_head_yaw: 120.0,
            pitch: -20.0,
            prev_pitch: -20.0,
            ..Default::default()
        };
        let p = s.params(0.5);
        assert!((p.net_head_yaw - 30.0).abs() < 1e-4);
        assert!((p.head_pitch + 20.0).abs() < 1e-4);
    }

    #[test]
    fn an_idle_sheep_uses_raw_pitch_instead_of_the_frame_interpolation() {
        let state = LivingRenderState {
            pitch: 30.0,
            prev_pitch: -10.0,
            species: SpeciesState::Sheep(SheepAnim::default()),
            ..Default::default()
        };

        let params = state.params(0.5);
        assert_eq!(
            params.head_pitch, 10.0,
            "the ordinary model pitch still lerps"
        );
        assert_eq!(
            params.entity.sheep_head_angle_x,
            30.0 / DEG_PER_RAD,
            "EntitySheep reads its current rotationPitch directly while idle"
        );
    }

    /// java: `RendererLivingEntity.doRender` clamps a mounted living rider's head to 85 degrees
    /// from the vehicle, then nudges a delta larger than 50 degrees by 20 percent. This is the
    /// render-facing half of the remote S1B mount state.
    #[test]
    fn a_living_mount_supplies_the_vehicle_body_and_clamps_the_riders_head() {
        let s = LivingRenderState {
            mount_body_yaw: Some(0.0),
            prev_mount_body_yaw: Some(0.0),
            head_yaw: 100.0,
            prev_head_yaw: 100.0,
            entity: EntityAnimState {
                riding: true,
                ..Default::default()
            },
            ..Default::default()
        };
        // 100 -> clamp the outer body to 85, then 85 * .2 is added: 15 + 17 = 32. The model
        // still receives f2=100 because vanilla computes it before the clamp and never rewrites
        // that local.
        assert!((s.render_body_yaw(0.5) - 32.0).abs() < 1e-4);
        assert!((s.params(0.5).net_head_yaw - 100.0).abs() < 1e-4);
    }

    #[test]
    fn swing_progress_wraps_forward() {
        let s = LivingRenderState {
            prev_swing_progress: 0.9,
            swing_progress: 0.1,
            ..Default::default()
        };
        // 0.1 - 0.9 = -0.8 -> +1 -> 0.2; half way is 0.9 + 0.1 = 1.0.
        assert!((s.swing_progress(0.5) - 1.0).abs() < 1e-6);
    }

    /// The whole point of the `prev_*` half: with a snapshot taken at the top of each tick, a
    /// frame drawn part-way through that tick lands between the two samples instead of on the
    /// newer one. java: `Entity.onUpdate`'s `prevPosX = posX` family.
    #[test]
    fn snapshot_previous_is_what_makes_the_frame_lerp_move() {
        let mut s = LivingRenderState {
            body_yaw: 10.0,
            head_yaw: 10.0,
            pitch: 0.0,
            limb_swing_amount: 0.0,
            swing_progress: 0.0,
            ..Default::default()
        };
        s.snapshot_previous();
        // A tick's worth of change, applied after the snapshot.
        s.body_yaw = 50.0;
        s.head_yaw = 90.0;
        s.pitch = -30.0;
        s.limb_swing_amount = 1.0;

        let p = s.params(0.5);
        assert!((p.head_pitch + 15.0).abs() < 1e-4, "pitch half way");
        assert!((p.limb_swing_amount - 0.5).abs() < 1e-6, "amount half way");
        // body 10 -> 50 and head 10 -> 90, both half way: 30 and 50, so net is 20.
        assert!((p.net_head_yaw - 20.0).abs() < 1e-4, "{}", p.net_head_yaw);
        assert!((s.render_body_yaw(0.5) - 30.0).abs() < 1e-4);

        // Without a second snapshot the next tick's frame would lerp from the same stale sample,
        // so snapshotting is idempotent only when nothing moved.
        s.snapshot_previous();
        assert_eq!(s.params(0.0).net_head_yaw, s.params(1.0).net_head_yaw);
    }

    /// java: `EntityGuardian()` seeds `field_175482_b` and `field_175484_c` with one shared
    /// `Entity.rand.nextFloat`; `func_175471_a` then interpolates from that constructor sample.
    #[test]
    fn guardian_constructor_tail_seed_reaches_interpolation_before_the_first_tick() {
        let initial_tail = 0.314_159_27_f32;
        let mut guardian = GuardianAnim::with_initial_tail(68, initial_tail);
        assert_eq!(guardian.tail_phase.to_bits(), initial_tail.to_bits());
        assert_eq!(guardian.prev_tail_phase.to_bits(), initial_tail.to_bits());
        assert_eq!(guardian.tail_phase(0.5).to_bits(), initial_tail.to_bits());

        // One beached client tick snapshots the old phase, then adds the fixed tail speed.
        guardian.prev_tail_phase = guardian.tail_phase;
        guardian.tail_speed = 2.0;
        guardian.tail_phase += guardian.tail_speed;
        assert_eq!(
            guardian.tail_phase(0.5).to_bits(),
            (initial_tail + 1.0).to_bits()
        );
    }

    /// java: `netHeadYaw / (180F / (float)Math.PI)`.
    ///
    /// The old test asserted `d * DEG_TO_RAD == d / (180.0 / PI)` for the single value `37.5`,
    /// and passed — 37.5 is one of the values where multiply-by-reciprocal and divide happen
    /// to round the same way. It is structurally incapable of failing on the divergence it
    /// appears to cover. This one proves the production helper follows the Java division while
    /// the reciprocal-multiply negative control demonstrably disagrees.
    #[test]
    fn model_degree_conversion_divides_like_java_and_rejects_the_reciprocal_mutant() {
        // The value the old test picked really does agree, which is the whole trap.
        assert_eq!(37.5_f32 * DEG_TO_RAD, 37.5_f32 / DEG_PER_RAD);

        let mut differ = 0_u32;
        let mut total = 0_u32;
        let mut worst = 0.0_f32;
        let mut step = -1440_i32; // -360.00 to 360.00 in quarter degrees
        while step <= 1440 {
            let d = step as f32 * 0.25;
            let mul = d * DEG_TO_RAD;
            let div = d / DEG_PER_RAD;
            assert_eq!(model_radians(d), div, "quarter-degree sample {step}");
            total += 1;
            if mul != div {
                differ += 1;
                worst = worst.max((mul - div).abs());
            }
            step += 1;
        }
        assert_eq!(total, 2881);
        assert_eq!(
            differ, 1520,
            "the divergence is the majority case, not an edge"
        );
        assert!(
            worst > 0.0 && worst < 1e-6,
            "one ulp, not a visible angle: {worst}"
        );
    }
}
