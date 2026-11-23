//! The simulated player's state and the raw key state that drives it.

use mc_data::BlockState;
use mc_math::{Aabb, BlockPos, DVec3};

/// The block reached by `Entity.moveEntity`'s grounded three-argument collision callback.
///
/// This is a transient per-tick signal. The application owns client-only block reactions and
/// particles; physics only reports the exact post-sweep block position and state Java selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroundContact {
    /// The sampled standing block position, after the fence/wall/gate fallback.
    pub pos: BlockPos,
    /// The state read at [`GroundContact::pos`] during this movement tick.
    pub state: BlockState,
}

/// Player bounding-box width. java: `EntityPlayer.preparePlayerToSpawn` — `setSize(0.6F, 1.8F)`.
///
/// 1.8.9 does **not** shrink the box when sneaking; only the eye height moves (`getEyeHeight`
/// subtracts `0.08F`). The 1.9+ pose system does not exist here.
pub const PLAYER_WIDTH: f32 = 0.6;

/// Player bounding-box height. java: `EntityPlayer.preparePlayerToSpawn`.
pub const PLAYER_HEIGHT: f32 = 1.8;

/// Auto-step height. java: `EntityPlayer.<init>` via `Entity.stepHeight`, set to `0.6F`.
pub const PLAYER_STEP_HEIGHT: f32 = 0.6;

/// Standing eye height. java: `EntityPlayer.getEyeHeight` — `1.62F`, minus `0.08F` sneaking.
pub const PLAYER_EYE_HEIGHT: f32 = 1.62;

/// `PlayerCapabilities.walkSpeed`, widened exactly as `setBaseValue((double) getWalkSpeed())`
/// widens it. java: `PlayerCapabilities.java:24`.
pub const WALK_SPEED: f64 = 0.1_f32 as f64;

/// The sprint attribute modifier's amount, operation 2 (multiply total).
/// java: `EntityLivingBase.sprintingSpeedBoostModifier`.
pub const SPRINT_SPEED_BOOST: f64 = 0.300_000_011_920_928_96;

/// `EntityPlayer.speedInAir` — the base air-control factor.
pub const SPEED_IN_AIR: f32 = 0.02;

/// `PlayerCapabilities.flySpeed`'s default. java: `PlayerCapabilities.java:26`.
pub const FLY_SPEED: f32 = 0.05;

/// One footstep the tick produced, for a caller that owns a sound engine.
///
/// java: the arguments `Entity.playStepSound` (`entity/Entity.java:986`) hands to
/// `playSound`, after its snow-layer substitution and its liquid exclusion have been applied.
/// The volume vanilla passes is `stepSound.getVolume() * 0.15F` and the pitch is
/// `stepSound.getFrequency()`; both live in `mc_data`'s block table, so only the *block whose
/// sound type won* has to travel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepSound {
    /// The block the entity is standing on. java: `blockpos` in `moveEntity`.
    pub pos: BlockPos,
    /// The state whose `stepSound` should be played — already substituted for
    /// `Blocks.snow_layer` when a snow layer sits on top.
    pub state: BlockState,
}

/// One client-visible consequence of crossing from air into water.
///
/// This is the deterministic part of `Entity.resetHeight`. The caller that owns the client RNG
/// expands it into one sound and two particle loops; keeping that fan-out outside physics makes
/// prediction replayable while still preserving each vanilla control value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterEntry {
    /// Centre of the particle clouds. Its Y is `floor(bb.minY) + 1`.
    pub particle_origin: DVec3,
    /// `f` passed to `playSound(getSplashSound(), f, pitch)`.
    pub sound_volume: f32,
    /// Each Java particle loop runs this many times.
    pub particle_count: u32,
    /// Lateral offset multiplier: `(nextFloat() * 2 - 1) * width`.
    pub particle_width: f32,
    /// `WATER_BUBBLE` starts with this velocity before its random Y subtraction.
    pub bubble_velocity: DVec3,
    /// Each bubble subtracts `nextFloat() * bubble_y_random_subtract` from Y velocity.
    pub bubble_y_random_subtract: f32,
    /// Every `WATER_SPLASH` receives exactly this velocity.
    pub splash_velocity: DVec3,
    /// Sound pitch base before random jitter.
    pub sound_pitch_base: f32,
    /// Pitch is `base + (nextFloat() - nextFloat()) * jitter`.
    pub sound_pitch_jitter: f32,
}

impl WaterEntry {
    /// `Entity.getSplashSound` for ordinary living entities.
    pub const SOUND_NAME: &'static str = "game.neutral.swim.splash";

    /// `EntityPlayer.getSplashSound` for the local player.  The local player overrides the
    /// ordinary entity sound names in vanilla 1.8.9; remote tracker entities continue to use
    /// [`Self::SOUND_NAME`] because they are expanded outside this player-physics crate.
    pub const PLAYER_SOUND_NAME: &'static str = "game.player.swim.splash";
}

impl StepSound {
    /// java: `Block.SoundType.getStepSound()` (`block/Block.java:1543`) — `"step." + soundName`.
    ///
    /// **`SLIME_SOUND` is the one exception in 1.8.9** (`block/Block.java:84`): it overrides
    /// `getStepSound()` to `"mob.slime.small"`. `mc_data`'s block table stores only the family
    /// name (`"slime"`), so a caller that naively formats `"step.{family}"` plays a sound that
    /// does not exist. Found by the `friction` differential fixture, which walks over a slime
    /// block and reported `step.slime` against the game's `mob.slime.small`.
    #[must_use]
    pub fn sound_name(self) -> String {
        let family = mc_data::block_info(mc_data::block_props(self.state).block).step_sound;
        if family == "slime" {
            "mob.slime.small".to_owned()
        } else {
            format!("step.{family}")
        }
    }

    /// java: `playStepSound` passes `stepSound.getVolume() * 0.15F`.
    #[must_use]
    pub fn volume(self) -> f32 {
        mc_data::block_info(mc_data::block_props(self.state).block).step_sound_volume * 0.15
    }

    /// java: `playStepSound` passes `stepSound.getFrequency()` unscaled.
    #[must_use]
    pub fn pitch(self) -> f32 {
        mc_data::block_info(mc_data::block_props(self.state).block).step_sound_pitch
    }
}

/// Raw key state for one tick, plus the absolute look angles that tick.
///
/// This is deliberately the *keyboard*, not a pre-digested movement vector: vanilla's
/// `MovementInputFromOptions.updatePlayerMoveState` turns key state into `moveStrafe`/
/// `moveForward` (including the `0.3` sneak scale), and `EntityPlayerSP.onLivingUpdate` decides
/// sprinting from the keys plus the previous tick's state. Both of those live in
/// [`crate::tick_player`], so a caller only has to forward what the user actually pressed.
///
/// `sprint` is the **sprint keybind**, not "is sprinting" — the latter is
/// [`PhysicsState::sprinting`], which the tick maintains.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MoveInput {
    /// Forward key (`W`).
    pub forward: bool,
    /// Back key (`S`).
    pub back: bool,
    /// Strafe-left key (`A`).
    pub left: bool,
    /// Strafe-right key (`D`).
    pub right: bool,
    /// Jump key (space).
    pub jump: bool,
    /// Sneak key (shift).
    pub sneak: bool,
    /// Sprint keybind (control). Double-tap-forward sprinting is handled without it.
    pub sprint: bool,
    /// java: `EntityPlayer.isUsingItem()` — eating, drinking, drawing a bow, blocking.
    ///
    /// Not a key, but it belongs here for the same reason the keys do: it is a per-tick fact the
    /// caller knows and `EntityPlayerSP.onLivingUpdate` (`:788`) applies to the *movement input*
    /// — `moveStrafe *= 0.2F; moveForward *= 0.2F; sprintToggleTimer = 0` — before anything
    /// downstream reads it. Modelling it as entity state would put the `0.2F` in the wrong place.
    pub using_item: bool,
    /// Absolute yaw in degrees for this tick. `0` faces `+Z`, `90` faces `-X`.
    pub yaw: f32,
    /// Absolute pitch in degrees for this tick.
    pub pitch: f32,
}

impl MoveInput {
    /// `MovementInputFromOptions.updatePlayerMoveState`'s `moveForward`, including the sneak
    /// scale and `EntityPlayerSP.onLivingUpdate`'s item-use scale. Forward and back cancel; the
    /// sneak multiplier is applied as a `f64` product and narrowed, exactly as vanilla writes it.
    ///
    /// `riding` is `EntityPlayer.isRiding()`; the item-use scale is skipped while riding
    /// (`EntityPlayerSP.java:788` — `if (this.isUsingItem() && !this.isRiding())`). This crate
    /// does not model riding yet, so [`MoveInput::move_forward`] passes `false`.
    #[must_use]
    pub fn move_forward_riding(&self, riding: bool) -> f32 {
        let mut f = 0.0f32;
        if self.forward {
            f += 1.0;
        }
        if self.back {
            f -= 1.0;
        }
        if self.sneak {
            f = (f64::from(f) * 0.3) as f32;
        }
        if self.using_item && !riding {
            // java: `this.movementInput.moveForward *= 0.2F;` — an f32 product, after the sneak
            // narrowing, and applied to the *input* rather than to the motion.
            f *= 0.2;
        }
        f
    }

    /// [`MoveInput::move_forward_riding`] for a player who is not riding anything.
    #[must_use]
    pub fn move_forward(&self) -> f32 {
        self.move_forward_riding(false)
    }

    /// `MovementInputFromOptions.updatePlayerMoveState`'s `moveStrafe`, plus the item-use scale.
    /// Positive is *left*.
    #[must_use]
    pub fn move_strafe_riding(&self, riding: bool) -> f32 {
        let mut f = 0.0f32;
        if self.left {
            f += 1.0;
        }
        if self.right {
            f -= 1.0;
        }
        if self.sneak {
            f = (f64::from(f) * 0.3) as f32;
        }
        if self.using_item && !riding {
            f *= 0.2;
        }
        f
    }

    /// [`MoveInput::move_strafe_riding`] for a player who is not riding anything.
    #[must_use]
    pub fn move_strafe(&self) -> f32 {
        self.move_strafe_riding(false)
    }
}

/// The local player's simulated state.
///
/// The first seven fields are S13's frozen API. The rest are additive: they are the vanilla
/// entity fields the tick genuinely needs to carry between ticks (`fallDistance`, `inWeb`,
/// `jumpTicks`, the sprint timers), and dropping them would make the simulation wrong rather
/// than merely incomplete.
///
/// `pos` is vanilla's `(posX, getEntityBoundingBox().minY, posZ)` — the **feet**, not the
/// centre. That is also what `C03PacketPlayer` puts on the wire, so no conversion is needed at
/// the network boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsState {
    /// Feet position. java: `posX`, `getEntityBoundingBox().minY`, `posZ`.
    pub pos: DVec3,
    /// Velocity. java: `motionX`/`motionY`/`motionZ`.
    pub vel: DVec3,
    /// java: `onGround`.
    pub on_ground: bool,
    /// java: `rotationYaw`, degrees.
    pub yaw: f32,
    /// java: `rotationPitch`, degrees.
    pub pitch: f32,
    /// java: `Entity.isSprinting()` (data-watcher flag 3).
    pub sprinting: bool,
    /// java: `EntityPlayerSP.isSneaking()` — the sneak key, mirrored each tick.
    pub sneaking: bool,

    // ---- additive: the rest of the vanilla entity state the tick needs ------------------
    /// java: `isCollidedHorizontally`. Cancels sprinting on the next tick.
    pub collided_horizontally: bool,
    /// java: `isCollidedVertically`.
    pub collided_vertically: bool,
    /// java: `fallDistance`, in blocks. `f32` in vanilla, so `f32` here.
    pub fall_distance: f32,
    /// java: `inWater`, maintained by `handleWaterMovement`.
    pub in_water: bool,
    /// java: `Entity.firstUpdate`. The initial water probe deliberately has no splash.
    pub first_update: bool,
    /// java: `isInWeb`. Set by `doBlockCollisions`, consumed and cleared by `moveEntity`.
    pub in_web: bool,
    /// java: `EntityLivingBase.jumpTicks` — the 10-tick jump cooldown.
    pub jump_ticks: i32,
    /// java: `EntityPlayerSP.sprintToggleTimer` — the double-tap-forward window.
    pub sprint_toggle_timer: i32,
    /// java: `EntityPlayerSP.sprintingTicksLeft` — the 600-tick sprint expiry.
    pub sprinting_ticks_left: i32,
    /// Position at the top of this tick. java: `Entity.prevPosX/Y/Z`, taken by
    /// `Entity.onEntityUpdate` (`entity/Entity.java:421`) *before* the tick moves anything.
    /// `moveEntityWithHeading`'s limb-swing tail is written against it.
    pub prev_pos: DVec3,
    /// java: `Entity.jumpMovementFactor` — the air-control factor.
    ///
    /// **This is a stored field on purpose, and it lags by one tick.** `EntityPlayer`'s
    /// `onLivingUpdate` (`entity/player/EntityPlayer.java:627`) assigns it *after*
    /// `super.onLivingUpdate()` has already run `moveEntityWithHeading`, so the value the move
    /// reads is the one the **previous** tick computed. Deriving it from the current sprint flag
    /// applies the `×1.3` sprint boost one tick early on the tick you leave the ground.
    pub jump_movement_factor: f32,
    /// Last tick's `movementInput.moveForward`, for the double-tap edge test.
    pub prev_move_forward: f32,
    /// Last tick's `movementInput.sneak`, for the double-tap edge test.
    pub prev_sneak: bool,
    /// java: `EntityLivingBase.moveStrafing`, the action-state value copied from
    /// `movementInput` by `EntityPlayerSP.updateEntityActionState` only while this player is the
    /// current render-view entity. Keeping it separate from [`MoveInput`] is what makes a
    /// foreign camera leave the local body on its previous action state.
    pub move_strafing: f32,
    /// java: `EntityLivingBase.moveForward`, with the same current-view gate as
    /// [`PhysicsState::move_strafing`].
    pub move_forward: f32,
    /// java: `EntityLivingBase.isJumping`, with the same current-view gate as
    /// [`PhysicsState::move_strafing`].
    pub jumping: bool,
    /// Bounding-box width. java: `Entity.width`.
    pub width: f32,
    /// Bounding-box height. java: `Entity.height`.
    pub height: f32,
    /// java: `Entity.stepHeight`.
    pub step_height: f32,
    /// Fall damage the client predicted this tick, or `None`. java: the argument
    /// `EntityLivingBase.fall` passes to `attackEntityFrom(DamageSource.fall, ...)`.
    ///
    /// The client only *predicts* this: the server is authoritative and sends the real health
    /// in `S06PacketUpdateHealth`. Cleared at the start of every tick.
    pub predicted_fall_damage: Option<u32>,

    // ---- additive: the server-driven capabilities `PlayerCapabilities` carries ------------
    /// java: `PlayerCapabilities.isFlying`. Set by `S39PacketPlayerAbilities`, by the
    /// double-tap-jump toggle, and forced on in spectator mode.
    pub flying: bool,
    /// java: `PlayerCapabilities.allowFlying`. Gates the toggle and stands in for hunger.
    pub allow_flying: bool,
    /// java: `PlayerCapabilities.flySpeed`, default `0.05`.
    pub fly_speed: f32,
    /// The `generic.movementSpeed` attribute value **before** the local sprint modifier.
    ///
    /// java: `EntityPlayer.onLivingUpdate:622` sets the attribute's *base* to
    /// `capabilities.getWalkSpeed()` only on the server (`!worldObj.isRemote`), so an
    /// `S20PacketEntityProperties` base is retained by the client. Its sprint modifier must be
    /// omitted from this value because [`PhysicsState::ai_move_speed`] applies local sprinting:
    /// `662A6B8D-DA3E-4C1C-8813-96EA6097278D`. Other server modifiers remain folded in. A caller
    /// without a snapshot should use
    /// [`crate::state::WALK_SPEED`], the vanilla player default.
    pub movement_speed: f64,
    /// Amount of the UUID-keyed sprint modifier currently installed on the `generic.movementSpeed`
    /// attribute map, or `None` when the map has no entry for that UUID.
    ///
    /// java: `EntityLivingBase.sprintingSpeedBoostModifier`, identified by
    /// `662A6B8D-DA3E-4C1C-8813-96EA6097278D`. **This is not the sprint flag.** Vanilla keeps
    /// the two apart and so must this:
    ///
    /// * [`PhysicsState::set_sprinting`] (`EntityLivingBase.setSprinting:1467`) writes both: it
    ///   removes the entry and re-applies the canonical one when sprinting.
    /// * `S20PacketEntityProperties` (`handleEntityProperties` → `removeAllModifiers` + the
    ///   server's list) rewrites **only this**, so a server snapshot taken before the client's
    ///   last sprint edge drops (or restores) the boost while [`PhysicsState::sprinting`] stays.
    /// * `S1CPacketEntityMetadata` for the local player rewrites **only the flag** (watched
    ///   index 0 bit 3), never this.
    ///
    /// [`PhysicsState::ai_move_speed`] reads this and never the flag: a vanilla client whose
    /// flag was cleared by a metadata echo keeps the boost, and one whose boost was removed by
    /// an attribute snapshot walks at walking speed while still reporting a sprint. A fresh
    /// `EntityPlayerSP` (join, respawn) has an empty map, hence `None`.
    pub sprint_modifier_amount: Option<f64>,
    /// java: `Entity.noClip`, which `EntityPlayer.onUpdate:268` sets to `isSpectator()`.
    pub no_clip: bool,
    /// java: `EntityPlayer.isSpectator()` — game mode 3 as the *tab list* reports it
    /// (`AbstractClientPlayer` looks the player up in `playerInfoMap`). Drives `noClip`, clears
    /// `onGround`, and excludes the player from `isOnLadder`.
    pub spectator: bool,
    /// java: `PlayerControllerMP.isSpectatorMode()` — the controller's game type (S01, S07,
    /// S2B reason 3). `EntityPlayerSP.onLivingUpdate` asks *this*, not
    /// [`PhysicsState::spectator`], for its forced-flight branch (`:825`) and its landing clear
    /// (`:904`); the two disagree whenever the S38 tab row is missing or lags the game mode.
    pub spectator_mode: bool,
    /// java: `Entity.isRiding()` (`ridingEntity != null`) for the tick being simulated.
    ///
    /// The walking tick ([`crate::tick_player`]) clears it and the mounted living update
    /// ([`crate::player::mounted_on_living_update_preflight`]) sets it, because the session forks
    /// on riding before choosing one of the two. The mounted rider's own travel reads it in the
    /// three places Java tests `ridingEntity == null`: `EntityPlayer.moveEntityWithHeading`'s
    /// flight wrapper, `Entity.moveEntity`'s walking branch, and `BlockPortal`'s `setPortal`.
    pub riding: bool,
    /// java: `EntityPlayer.isMovementBlocked()` — `getHealth() <= 0.0F || isPlayerSleeping()`.
    /// Zeroes the *input*, not the motion, so gravity and drag still run.
    pub movement_blocked: bool,
    /// java: `FoodStats.foodLevel`. Sprint needs `> 6` (or `allowFlying`).
    pub food_level: i32,
    /// java: `Potion.jump`'s signed packet amplifier, or `None`. Adds `(amplifier + 1) * 0.1F`
    /// to a jump and subtracts `amplifier + 1` from the fall-damage distance. `S1D` is read with
    /// `readByte`, so wire values `128..=255` arrive as negative Java bytes and must remain `i8`.
    pub jump_boost: Option<i8>,
    /// java: `isPotionActive(Potion.blindness)` — blocks sprint from *starting*.
    pub blindness: bool,
    /// java: `EnchantmentHelper.getDepthStriderModifier(this)` — the boots' Depth Strider level,
    /// unclamped. `0.0` for no enchantment.
    pub depth_strider: f32,
    /// java: `EntityPlayerSP.flyToggleTimer` — the double-tap-jump window.
    pub fly_toggle_timer: i32,
    /// Last tick's jump key, for the fly-toggle edge test. java: `EntityPlayerSP.onLivingUpdate`
    /// reads `movementInput.jump` *before* `updatePlayerMoveState()` overwrites it.
    pub prev_jump: bool,

    // ---- additive: the walk accumulators and the animation driver -------------------------
    /// java: `Entity.distanceWalkedModified` — the view-bobbing input.
    pub distance_walked_modified: f32,
    /// java: `Entity.prevDistanceWalkedModified`, snapshotted at the top of every tick.
    pub prev_distance_walked_modified: f32,
    /// java: `EntityPlayer.prevCameraYaw` / `cameraYaw`, the damped horizontal sway consumed by
    /// first- and third-person view bobbing.
    pub prev_camera_yaw: f32,
    /// Current damped horizontal camera sway.
    pub camera_yaw: f32,
    /// java: `EntityLivingBase.prevCameraPitch` / `cameraPitch`.
    pub prev_camera_pitch: f32,
    /// Current damped vertical camera sway.
    pub camera_pitch: f32,
    /// java: `EntityPlayerSP.prevRenderArmYaw`, last tick's [`PhysicsState::render_arm_yaw`].
    pub prev_render_arm_yaw: f32,
    /// java: `EntityPlayerSP.renderArmYaw`: a yaw that chases `rotationYaw` half-way each tick.
    /// `ItemRenderer.rotateWithPlayerRotations` turns the first-person hand by a tenth of the gap,
    /// which is the hand's lag behind a fast mouse turn. Render-only; it never reaches a packet.
    pub render_arm_yaw: f32,
    /// java: `EntityPlayerSP.prevRenderArmPitch`.
    pub prev_render_arm_pitch: f32,
    /// java: `EntityPlayerSP.renderArmPitch`, the pitch half of [`PhysicsState::render_arm_yaw`].
    pub render_arm_pitch: f32,
    /// java: `Entity.distanceWalkedOnStepModified` — the step-sound accumulator, which counts
    /// vertical movement too while on a ladder.
    pub distance_walked_on_step_modified: f32,
    /// java: `Entity.nextStepDistance`, initialised to `1`.
    pub next_step_distance: i32,
    /// java: `EntityLivingBase.limbSwing` — the walk-animation phase.
    pub limb_swing: f32,
    /// java: `EntityLivingBase.limbSwingAmount` — the walk-animation weight, `0..=1`.
    pub limb_swing_amount: f32,
    /// java: `EntityLivingBase.prevLimbSwingAmount`, for render-time interpolation.
    pub prev_limb_swing_amount: f32,

    // ---- additive: per-tick output signals, cleared at the top of every tick --------------
    /// Grounded block contact selected by `Entity.moveEntity`, if this tick produced one that
    /// has a client-side standing callback. The application drains it after physics.
    pub ground_contact: Option<GroundContact>,
    /// The footstep this tick produced, if any. java: `Entity.playStepSound`'s arguments.
    pub step_sound: Option<StepSound>,
    /// The swim sound's volume this tick, if the step crossed `nextStepDistance` under water.
    /// java: `Entity.moveEntity:880` — `sqrt(mx²·0.2 + my² + mz²·0.2) * 0.35F`, clamped to `1`.
    pub swim_sound_volume: Option<f32>,
    /// The water-entry sound/particle recipe produced this tick, if any.
    pub water_entry: Option<WaterEntry>,
    /// java: `Entity.inPortal`, set by `BlockPortal.onEntityCollidedWithBlock`. Drives
    /// `EntityPlayerSP.onLivingUpdate`'s `timeInPortal` ramp and the portal overlay.
    pub in_portal: bool,
    /// java: `Entity.timeUntilPortal`. Positive contact refreshes this to the local player's
    /// `EntityPlayer.getPortalCooldown() == 10` without setting [`PhysicsState::in_portal`].
    pub portal_cooldown: i32,
    /// Whether the player touched a cactus this tick, retained for the differential trace only.
    /// java: `BlockCactus:121` calls `attackEntityFrom(DamageSource.cactus, 1.0F)` without its
    /// own `isRemote` guard, but the client-side `EntityLivingBase.attackEntityFrom` returns
    /// before applying damage when `worldObj.isRemote`; this is not a client damage signal.
    pub touched_cactus: bool,

    /// The bounding box, and the **authoritative** copy of the entity's location.
    ///
    /// java: `Entity.boundingBox`. This is not redundant with [`PhysicsState::pos`] and cannot
    /// be derived from it: vanilla never rebuilds the box from `posX/posY/posZ` during
    /// movement. `moveEntity` offsets the box and then *derives* the position from it
    /// (`resetPositionToBB`), so `minX` and `maxX` accumulate their own rounding independently
    /// and are not exactly `posX ∓ width/2` after the first tick. Recomputing the box from
    /// `pos` each tick costs about one ULP per tick, which compounds into a real position
    /// difference over a few thousand ticks — and a server that disagrees.
    ///
    /// [`PhysicsState::set_position`] is the only thing that rebuilds it, matching
    /// `Entity.setPosition`.
    bb: Aabb,
}

impl PhysicsState {
    /// A player standing at `feet`, motionless, facing `+Z`.
    #[must_use]
    pub fn new_player(feet: DVec3) -> Self {
        let mut s = Self {
            pos: feet,
            vel: DVec3::ZERO,
            on_ground: false,
            yaw: 0.0,
            pitch: 0.0,
            sprinting: false,
            sneaking: false,
            collided_horizontally: false,
            collided_vertically: false,
            fall_distance: 0.0,
            in_water: false,
            first_update: true,
            in_web: false,
            jump_ticks: 0,
            sprint_toggle_timer: 0,
            sprinting_ticks_left: 0,
            prev_pos: feet,
            // java: `Entity.jumpMovementFactor = 0.02F` at construction.
            jump_movement_factor: SPEED_IN_AIR,
            prev_move_forward: 0.0,
            prev_sneak: false,
            move_strafing: 0.0,
            move_forward: 0.0,
            jumping: false,
            width: PLAYER_WIDTH,
            height: PLAYER_HEIGHT,
            step_height: PLAYER_STEP_HEIGHT,
            predicted_fall_damage: None,
            flying: false,
            allow_flying: false,
            fly_speed: FLY_SPEED,
            movement_speed: WALK_SPEED,
            // java: `EntityLivingBase.applyEntityAttributes` registers `generic.movementSpeed`
            // with no modifiers; the sprint entry only appears on a `setSprinting(true)` edge
            // or in an S20 snapshot. A caller that wants a sprinting state must call
            // `set_sprinting(true)`, as vanilla does, rather than flip the flag alone.
            sprint_modifier_amount: None,
            no_clip: false,
            spectator: false,
            spectator_mode: false,
            riding: false,
            movement_blocked: false,
            // java: `FoodStats.foodLevel` starts at 20, and `EntityPlayerSP`'s copy is only ever
            // lowered by `S06PacketUpdateHealth`.
            food_level: 20,
            jump_boost: None,
            blindness: false,
            depth_strider: 0.0,
            fly_toggle_timer: 0,
            prev_jump: false,
            distance_walked_modified: 0.0,
            prev_distance_walked_modified: 0.0,
            prev_camera_yaw: 0.0,
            camera_yaw: 0.0,
            prev_camera_pitch: 0.0,
            camera_pitch: 0.0,
            // java: the four `EntityPlayerSP` arm fields are never initialised, so a fresh player
            // (join, respawn) starts them at `0.0F` and the hand visibly swings to the spawn yaw.
            prev_render_arm_yaw: 0.0,
            render_arm_yaw: 0.0,
            prev_render_arm_pitch: 0.0,
            render_arm_pitch: 0.0,
            distance_walked_on_step_modified: 0.0,
            // java: `Entity.nextStepDistance = 1`, not 0.
            next_step_distance: 1,
            limb_swing: 0.0,
            limb_swing_amount: 0.0,
            prev_limb_swing_amount: 0.0,
            ground_contact: None,
            step_sound: None,
            swim_sound_volume: None,
            water_entry: None,
            in_portal: false,
            portal_cooldown: 0,
            touched_cactus: false,
            bb: Aabb::new(DVec3::ZERO, DVec3::ZERO),
        };
        s.set_position(feet);
        s
    }

    /// Places the player at `feet` and rebuilds the bounding box around it.
    /// java: `Entity.setPosition`.
    pub fn set_position(&mut self, feet: DVec3) {
        self.pos = feet;
        // java: `float f = this.width / 2.0F;` — an f32 halving, then widened.
        let f = f64::from(self.width / 2.0);
        let f1 = f64::from(self.height);
        self.bb = Aabb::new(
            DVec3::new(feet.x - f, feet.y, feet.z - f),
            DVec3::new(feet.x + f, feet.y + f1, feet.z + f),
        );
    }

    /// The world-space bounding box. java: `Entity.getEntityBoundingBox`.
    ///
    /// Vanilla computes the half-width in `f32` (`float f = this.width / 2.0F;`) and widens it,
    /// which for `width = 0.6` gives `0.30000001192092896`, not `0.3`.
    #[inline]
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        self.bb
    }

    /// Adopts `bb` as the bounding box and derives the position from it.
    /// java: `Entity.setEntityBoundingBox` followed by `Entity.resetPositionToBB`.
    pub fn set_from_aabb(&mut self, bb: &Aabb) {
        self.bb = *bb;
        self.pos = DVec3::new(
            (bb.min.x + bb.max.x) / 2.0,
            bb.min.y,
            (bb.min.z + bb.max.z) / 2.0,
        );
    }

    /// The exact `(position, bounding box)` pair. [`PhysicsState::set_position`] would rebuild
    /// the box and drop its accumulated rounding, so a caller that must put the entity back bit
    /// for bit saves and restores both.
    #[must_use]
    pub(crate) fn location(&self) -> (DVec3, Aabb) {
        (self.pos, self.bb)
    }

    /// Put back a pair taken by [`PhysicsState::location`].
    pub(crate) fn restore_location(&mut self, location: (DVec3, Aabb)) {
        (self.pos, self.bb) = location;
    }

    /// Eye position, for ray-casting and camera placement. java: `posY + getEyeHeight()`.
    #[must_use]
    pub fn eye_pos(&self) -> DVec3 {
        let mut eye = PLAYER_EYE_HEIGHT;
        if self.sneaking {
            eye -= 0.08;
        }
        DVec3::new(self.pos.x, self.pos.y + f64::from(eye), self.pos.z)
    }

    /// `EntityPlayer.getAIMoveSpeed()` (`entity/player/EntityPlayer.java:1814`):
    /// `(float) getEntityAttribute(movementSpeed).getAttributeValue()`, read at call time.
    ///
    /// java: `ModifiableAttributeInstance.computeValue` — operation-2 modifiers multiply the
    /// running total by `1.0 + amount`, in `f64`; the getter then narrows to `f32`.
    ///
    /// [`PhysicsState::movement_speed`] is the attribute value with every *server*-sent modifier
    /// except the canonical sprint UUID already folded in, and the sprint entry currently in the
    /// map, [`PhysicsState::sprint_modifier_amount`], is applied last. **The sprint flag is not
    /// consulted**: the attribute map, not `isSprinting()`, is what vanilla reads, and the two
    /// diverge whenever an S1C or S20 echo lands between local sprint edges.
    ///
    /// Java visits operation-2 modifiers in UUID-`HashSet` order (Slowness's bucket 5, then the
    /// sprint UUID's bucket 6, then Speed's bucket 13). Multiplying the sprint factor last can
    /// change the `f64` product by one ulp when Speed and Slowness are both present, but the
    /// narrowed `f32` this returns is identical for every Speed 0-7 x Slowness 0-3 combination
    /// (all of them were checked). `generic.movementSpeed` is a
    /// `RangedAttribute(0.0, 1024.0)`, and its clamp happens after every modifier.
    #[must_use]
    pub fn ai_move_speed(&self) -> f32 {
        let mut v = self.movement_speed;
        if let Some(amount) = self.sprint_modifier_amount {
            v *= 1.0 + amount;
        }
        // java: ModifiableAttributeInstance.computeValue -> RangedAttribute.clampValue.
        // `f64::clamp` preserves Java's NaN result while making the closed range explicit.
        v = v.clamp(0.0, 1024.0);
        v as f32
    }

    /// java: `EntityPlayerSP.setSprinting` (`client/entity/EntityPlayerSP.java:516`) →
    /// `EntityLivingBase.setSprinting` (`entity/EntityLivingBase.java:1467`) →
    /// `Entity.setSprinting`.
    ///
    /// Writes the watched flag, removes the sprint UUID's attribute entry if present and
    /// re-applies the canonical `0.30000001192092896` (operation 2) entry when sprinting, and
    /// restarts `sprintingTicksLeft` at `600` (or clears it). This is the **only** local path
    /// that installs the boost; an S1C flag echo must assign [`PhysicsState::sprinting`] alone.
    pub fn set_sprinting(&mut self, sprinting: bool) {
        self.sprinting = sprinting;
        self.sprint_modifier_amount = sprinting.then_some(SPRINT_SPEED_BOOST);
        self.sprinting_ticks_left = if sprinting { 600 } else { 0 };
    }

    /// The air-control factor `moveEntityWithHeading` actually uses this tick.
    ///
    /// java: `EntityPlayer.moveEntityWithHeading` (`entity/player/EntityPlayer.java:1788`)
    /// replaces `jumpMovementFactor` with `capabilities.getFlySpeed() * (isSprinting() ? 2 : 1)`
    /// for the duration of the call while flying, and restores it afterwards. That wrapper is
    /// guarded by `capabilities.isFlying && this.ridingEntity == null`, so a flying *rider*'s
    /// own travel keeps the stored factor.
    #[must_use]
    pub fn effective_jump_movement_factor(&self) -> f32 {
        if self.flying && !self.riding {
            self.fly_speed * if self.sprinting { 2.0 } else { 1.0 }
        } else {
            self.jump_movement_factor
        }
    }

    /// The value `EntityPlayer.onLivingUpdate` (`entity/player/EntityPlayer.java:627`) assigns to
    /// `jumpMovementFactor` — which the **next** tick's move will read, not this one's. See
    /// [`PhysicsState::jump_movement_factor`].
    #[must_use]
    pub fn recomputed_jump_movement_factor(&self) -> f32 {
        if self.sprinting {
            (f64::from(SPEED_IN_AIR) + f64::from(SPEED_IN_AIR) * 0.3) as f32
        } else {
            SPEED_IN_AIR
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_width_carries_the_f32_rounding_vanilla_has() {
        let s = PhysicsState::new_player(DVec3::new(0.0, 0.0, 0.0));
        // `(double)(0.6F / 2.0F)`, not `0.6 / 2.0`.
        assert_eq!(s.aabb().max.x, 0.300_000_011_920_928_96);
        assert_eq!(s.aabb().max.y, f64::from(1.8f32));
    }

    #[test]
    fn sprint_speed_matches_the_attribute_computation() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        assert_eq!(s.ai_move_speed(), 0.1f32);
        s.set_sprinting(true);
        // Not `0.1 * 1.3`: both operands are the f32 constants widened, so the product is
        // 0.13000001, not 0.13. Getting this wrong is a ~0.001 b/t speed error.
        assert_eq!(
            s.ai_move_speed(),
            (WALK_SPEED * (1.0 + SPRINT_SPEED_BOOST)) as f32
        );
        assert_eq!(s.ai_move_speed(), 0.130_000_01_f32);
        assert_eq!(
            s.recomputed_jump_movement_factor(),
            (f64::from(SPEED_IN_AIR) * 1.3) as f32
        );
        assert_eq!(s.recomputed_jump_movement_factor(), 0.025_999_999_f32);
        // The *stored* factor is the previous tick's, so it has not moved yet.
        assert_eq!(s.jump_movement_factor, SPEED_IN_AIR);
    }

    /// `capabilities.getFlySpeed()` bypasses the attribute entirely, and the sprint
    /// multiplier is an integer `2`, not the `1.3` the ground path uses.
    #[test]
    fn flight_replaces_the_air_control_factor_with_fly_speed() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        assert_eq!(s.effective_jump_movement_factor(), SPEED_IN_AIR);
        s.flying = true;
        assert_eq!(s.effective_jump_movement_factor(), 0.05f32);
        s.sprinting = true;
        assert_eq!(s.effective_jump_movement_factor(), 0.1f32);
        s.fly_speed = 0.2;
        assert_eq!(s.effective_jump_movement_factor(), 0.4f32);
    }

    /// A server-sent `generic.movementSpeed` is the attribute value the sprint modifier
    /// then multiplies, so Speed II and sprint compose the way vanilla's `computeValue` composes
    /// them.
    #[test]
    fn a_server_movement_speed_replaces_the_walk_base() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        // Speed II: an operation-2 modifier of +0.2 per level on a 0.1 base.
        s.movement_speed = WALK_SPEED * 1.4;
        assert_eq!(s.ai_move_speed(), (WALK_SPEED * 1.4) as f32);
        s.set_sprinting(true);
        assert_eq!(
            s.ai_move_speed(),
            (WALK_SPEED * 1.4 * (1.0 + SPRINT_SPEED_BOOST)) as f32
        );
    }

    #[test]
    fn sprint_modifier_presence_follows_the_uuid_attribute_replacement() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        s.movement_speed = 0.14;
        // A fresh EntityPlayerSP's attribute map has no sprint entry.
        assert_eq!(s.sprint_modifier_amount, None);
        assert_eq!(s.ai_move_speed(), 0.14f32);
        s.set_sprinting(true);
        assert_eq!(
            s.ai_move_speed(),
            (0.14 * (1.0 + SPRINT_SPEED_BOOST)) as f32,
            "setSprinting(true) installs the canonical UUID entry once"
        );
        assert_eq!(s.sprinting_ticks_left, 600);

        // After S20's removeAllModifiers omits the canonical UUID, a still-set sprint flag gets
        // no boost until a local setSprinting edge installs it again.
        s.sprint_modifier_amount = None;
        assert!(s.sprinting);
        assert_eq!(s.ai_move_speed(), 0.14f32);

        s.sprint_modifier_amount = Some(0.2);
        assert_eq!(s.ai_move_speed(), (0.14 * 1.2) as f32);
    }

    /// java: an S1C echo writes only `DataWatcher` index 0 bit 3 (`Entity.setFlag`), while
    /// `getAIMoveSpeed` reads the attribute map. The boost follows the map, never the flag.
    #[test]
    fn the_sprint_boost_follows_the_attribute_entry_not_the_watched_flag() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        s.set_sprinting(true);
        // A metadata echo clears the flag: the boost and the 600-tick countdown stay.
        s.sprinting = false;
        assert_eq!(
            s.ai_move_speed(),
            (WALK_SPEED * (1.0 + SPRINT_SPEED_BOOST)) as f32
        );
        assert_eq!(s.sprinting_ticks_left, 600);

        // A metadata echo sets the flag on a walking player: no entry, no boost.
        let mut t = PhysicsState::new_player(DVec3::ZERO);
        t.sprinting = true;
        assert_eq!(t.ai_move_speed(), 0.1f32);
        // The flag still drives the flag readers (air control, flight factor).
        assert_eq!(t.recomputed_jump_movement_factor(), 0.025_999_999_f32);

        // setSprinting(false) removes the entry and the countdown.
        s.set_sprinting(false);
        assert_eq!(s.sprint_modifier_amount, None);
        assert_eq!(s.sprinting_ticks_left, 0);
        assert_eq!(s.ai_move_speed(), 0.1f32);
    }

    #[test]
    fn movement_speed_is_clamped_after_the_sprint_modifier() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        s.movement_speed = -0.5;
        assert_eq!(s.ai_move_speed(), 0.0);

        s.movement_speed = 1023.0;
        s.set_sprinting(true);
        assert_eq!(s.ai_move_speed(), 1024.0);

        // Negative controls: in-range values are not flattened, and Java's comparison-based
        // MathHelper.clamp_double leaves NaN unchanged.
        s.movement_speed = 0.25;
        assert_eq!(
            s.ai_move_speed(),
            (0.25 * (1.0 + SPRINT_SPEED_BOOST)) as f32
        );
        s.movement_speed = f64::NAN;
        assert!(s.ai_move_speed().is_nan());
    }

    /// The `0.2F` item-use scale lands on the *input*, after the sneak narrowing.
    #[test]
    fn using_an_item_scales_the_movement_input_to_a_fifth() {
        let i = MoveInput {
            forward: true,
            using_item: true,
            ..MoveInput::default()
        };
        assert_eq!(i.move_forward(), 0.2f32);
        // java: `if (this.isUsingItem() && !this.isRiding())` — riding suppresses it.
        assert_eq!(i.move_forward_riding(true), 1.0f32);
        let i = MoveInput {
            left: true,
            sneak: true,
            using_item: true,
            ..MoveInput::default()
        };
        // 1.0 -> 0.3 (sneak, via f64) -> 0.06 (item use, in f32).
        assert_eq!(i.move_strafe(), 0.3f32 * 0.2f32);
    }

    #[test]
    fn sneak_scales_the_movement_input_by_three_tenths() {
        let i = MoveInput {
            forward: true,
            left: true,
            sneak: true,
            ..MoveInput::default()
        };
        assert_eq!(i.move_forward(), 0.3f32);
        assert_eq!(i.move_strafe(), 0.3f32);
        let i = MoveInput {
            forward: true,
            back: true,
            ..MoveInput::default()
        };
        assert_eq!(i.move_forward(), 0.0);
    }

    #[test]
    fn set_from_aabb_round_trips_a_position() {
        let s = PhysicsState::new_player(DVec3::new(3.25, 64.0, -7.5));
        let mut t = PhysicsState::new_player(DVec3::ZERO);
        t.set_from_aabb(&s.aabb());
        assert_eq!(t.pos, s.pos);
        assert_eq!(t.aabb(), s.aabb());
    }

    #[test]
    fn the_box_is_authoritative_and_survives_an_offset_that_pos_cannot_express() {
        // java: `moveEntity` offsets the box and derives `posX` from it, so a box whose centre
        // is not exactly `pos` must round-trip as itself, not be rebuilt from `pos`.
        let mut s = PhysicsState::new_player(DVec3::new(0.0, 64.0, 0.0));
        let skewed = Aabb::from_bounds(-0.3, 64.0, -0.3, 0.4, 65.8, 0.3);
        s.set_from_aabb(&skewed);
        assert_eq!(
            s.aabb(),
            skewed,
            "the box is kept verbatim, not rebuilt from pos"
        );
        assert!(
            (s.pos.x - 0.05).abs() < 1e-15,
            "pos is derived from the box ({}), not the other way round",
            s.pos.x
        );
        // `set_position` is the only thing that reshapes it.
        s.set_position(DVec3::new(0.0, 64.0, 0.0));
        assert_eq!(s.aabb().max.x, 0.300_000_011_920_928_96);
    }

    #[test]
    fn sneaking_lowers_the_eye_but_not_the_box() {
        let mut s = PhysicsState::new_player(DVec3::new(0.0, 64.0, 0.0));
        let standing = s.eye_pos().y;
        let box_before = s.aabb();
        s.sneaking = true;
        // java: `f -= 0.08F` happens in f32 before the widening.
        assert_eq!(s.eye_pos().y, 64.0 + f64::from(PLAYER_EYE_HEIGHT - 0.08));
        assert!(s.eye_pos().y < standing);
        assert_eq!(box_before, s.aabb(), "1.8.9 does not shrink the sneak box");
    }
}
