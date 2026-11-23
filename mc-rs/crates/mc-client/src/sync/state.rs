//! [`ClientState`]: everything `NetHandlerPlayClient` mutates, in one owned, non-aliased struct.
//!
//! java: this is the union of `NetHandlerPlayClient`'s fields, `WorldClient`, and the parts of
//! `EntityPlayerSP`/`PlayerControllerMP` that are pure server-driven state. Vanilla scatters
//! them across a cyclic object graph rooted at `Minecraft`; the arena-index rule says we do not, so they
//! live here and systems take `&mut ClientState`.

use mc_math::{BlockPos, DVec3};
use mc_protocol::play::s2c::GameMode;
use mc_protocol::ItemStack;
use mc_world::{entity::Metadata, World};
use uuid::Uuid;

use crate::sync::entities::EntityTracker;
use crate::sync::env::Environment;
use crate::sync::players::PlayerList;
use crate::sync::scoreboard::Scoreboard;

/// `WorldSettings.GameType.getByID` — note that an unrecognised id becomes `Survival`, **not**
/// `NotSet`, which only ever arrives as an explicit `-1` in the tab list.
/// java: `world/WorldSettings.java`
#[must_use]
pub fn game_mode_from_id(id: i32) -> GameMode {
    match id {
        -1 => GameMode::NotSet,
        1 => GameMode::Creative,
        2 => GameMode::Adventure,
        3 => GameMode::Spectator,
        _ => GameMode::Survival,
    }
}

/// The local player's abilities, as `S39PacketPlayerAbilities` sets them.
/// java: `entity/player/PlayerCapabilities.java`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Abilities {
    /// java: `PlayerCapabilities.disableDamage`
    pub invulnerable: bool,
    /// java: `PlayerCapabilities.isFlying`
    pub flying: bool,
    /// java: `PlayerCapabilities.allowFlying`
    pub allow_flying: bool,
    /// java: `PlayerCapabilities.isCreativeMode`
    pub creative_mode: bool,
    /// java: `PlayerCapabilities.allowEdit` (default `true`).  It is configured from game mode,
    /// not carried by `S39PacketPlayerAbilities`.
    pub allow_edit: bool,
    /// java: `PlayerCapabilities.flySpeed` (default `0.05`)
    pub fly_speed: f32,
    /// java: `PlayerCapabilities.walkSpeed` (default `0.1`)
    pub walk_speed: f32,
}

impl Default for Abilities {
    fn default() -> Self {
        Abilities {
            invulnerable: false,
            flying: false,
            allow_flying: false,
            creative_mode: false,
            allow_edit: true,
            fly_speed: 0.05,
            walk_speed: 0.1,
        }
    }
}

impl Abilities {
    /// Apply `WorldSettings.GameType.configurePlayerCapabilities` to this existing capability
    /// object.
    ///
    /// This deliberately preserves both speeds for every mode. Creative also deliberately
    /// preserves `flying`: vanilla's creative branch never assigns `isFlying`, whereas the
    /// spectator branch forces it on and the survival/adventure branch forces it off.
    /// java: `WorldSettings.GameType.configurePlayerCapabilities` (`WorldSettings.java:162`).
    pub(crate) fn configure_for_game_mode(&mut self, game_mode: GameMode) {
        match game_mode {
            GameMode::Creative => {
                self.allow_flying = true;
                self.creative_mode = true;
                self.invulnerable = true;
            }
            GameMode::Spectator => {
                self.allow_flying = true;
                self.creative_mode = false;
                self.invulnerable = true;
                self.flying = true;
            }
            GameMode::NotSet | GameMode::Survival | GameMode::Adventure => {
                self.allow_flying = false;
                self.creative_mode = false;
                self.invulnerable = false;
                self.flying = false;
            }
        }
        // java: `WorldSettings.GameType.configurePlayerCapabilities` assigns this after every
        // branch. Adventure includes Spectator, while NotSet follows the non-adventure path.
        self.allow_edit = !matches!(game_mode, GameMode::Adventure | GameMode::Spectator);
    }

    /// Decode `S39PacketPlayerAbilities`' bit field.
    /// java: `S39PacketPlayerAbilities.readPacketData`
    #[must_use]
    pub fn from_flags(flags: u8, fly_speed: f32, walk_speed: f32) -> Abilities {
        Abilities {
            invulnerable: flags & 0x01 != 0,
            flying: flags & 0x02 != 0,
            allow_flying: flags & 0x04 != 0,
            creative_mode: flags & 0x08 != 0,
            // S39's bitfield has no `allowEdit` bit. The S39 handler preserves the existing
            // value when it replaces the decoded wire fields; this is the fresh-capabilities
            // default for direct construction.
            allow_edit: true,
            fly_speed,
            walk_speed,
        }
    }
}

/// One clientbound entity packet that resolved to the session-owned player.
///
/// The Java handler mutates the local `EntityPlayerSP` immediately, but its inherited
/// `EntityLivingBase.setPositionAndRotation2` only stores a target for the next three local
/// ticks.  [`ClientState`] cannot borrow the physics predictor, so dispatch records the exact
/// packet edge here and [`crate::Session`] consumes it after the packet has been applied.  The
/// queue is deliberately ordered: two local movement packets in one network drain must replace
/// the target in the same order as Java's scheduled task, rather than being coalesced into one
/// synthetic correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LocalEntityUpdate {
    /// `S14`/`S15`/`S16`/`S17`/`S18` changed the local entity's interpolation target and ground bit.
    Movement,
    /// `S19PacketEntityHeadLook` wrote `rotationYawHead` without moving the body.
    HeadLook,
}

/// The local player's server-driven state.
///
/// Position is deliberately **absent**: it belongs to `mc_physics::Predictor`, which the game
/// loop owns, because the client predicts it and the server only ever corrects it. Putting a
/// second copy here would create exactly the "which one is authoritative?" ambiguity the
/// prediction design exists to avoid.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalPlayer {
    /// java: `EntityPlayerSP.entityId`, from `S01PacketJoinGame`.
    pub entity_id: i32,
    /// Whether `entity_id` came from a completed JoinGame lifecycle edge.
    ///
    /// Java accepts zero as an entity id; the Rust blank-state default also uses zero before a
    /// join. Keep the two meanings separate so a pre-join packet for id zero remains unknown,
    /// while a real joined player whose server id is zero still resolves locally.
    pub(crate) entity_id_known: bool,
    /// The local `EntityPlayerSP`'s DataWatcher snapshot.
    ///
    /// The predicted player deliberately stays out of the remote [`EntityTracker`], but vanilla
    /// still resolves local-id S1C packets through `WorldClient` and merges them into this
    /// entity's watcher. Keeping the watcher here preserves that single local-player owner
    /// without inventing a second position or entity record.
    pub metadata: Metadata,
    /// From `LoginSuccess`.
    pub uuid: Uuid,
    /// From `LoginSuccess`; may differ in case from what was typed.
    pub username: String,
    /// java: `PlayerControllerMP.currentGameType`
    pub game_mode: GameMode,
    /// java: `WorldInfo.hardcore`
    pub hardcore: bool,
    /// `-1` Nether, `0` Overworld, `1` End — but nothing on the wire enforces that.
    pub dimension: i32,
    /// `"default"`, `"flat"`, `"largeBiomes"`, `"amplified"`, …
    pub level_type: String,
    /// java: `NetHandlerPlayClient.currentServerMaxPlayers`, which sizes the tab list.
    pub max_players: u8,
    /// java: `EntityPlayerSP.setReducedDebug` — hides coordinates on the F3 screen.
    pub reduced_debug_info: bool,
    /// `0.0` means dead. java: `EntityPlayerSP.setPlayerSPHealth`
    pub health: f32,
    /// Resolved `generic.maxHealth`, including the latest S20 modifier snapshot.
    /// java: `EntityLivingBase.getMaxHealth`; the local default is 20.0.
    pub max_health: f32,
    /// `EntityPlayerSP.hasValidHealth`. The first S06 after an entity is constructed must set
    /// health quietly: treating the join/respawn snapshot as damage would incorrectly start a
    /// hurt flash and camera kick.
    pub has_valid_health: bool,
    /// `EntityLivingBase.lastDamage`, set from the size of the most recent health decrease.
    /// This is deliberately retained even though the current HUD/camera consumers live above
    /// this state layer.
    pub last_damage: f32,
    /// `EntityLivingBase.hurtResistantTime`; a health gain sets it to 10 and a damage packet to
    /// 20. Its per-tick decrement belongs to the game loop, not packet dispatch.
    pub hurt_resistant_time: i32,
    /// `EntityLivingBase.hurtTime`, set to [`Self::max_hurt_time`] on a health decrease.
    pub hurt_time: i32,
    /// `EntityLivingBase.maxHurtTime`; vanilla's normal living-entity value is 10.
    pub max_hurt_time: i32,
    /// `EntityLivingBase.deathTime`, advanced by `onDeathUpdate` while health is zero.
    ///
    /// A local player is not in the remote `EntityTracker`, so this render-visible timer has to
    /// live beside the local health and hurt timers. Vanilla removes the entity at 20 ticks;
    /// keeping the terminal value here preserves the value consumed by the death camera until a
    /// respawn replaces the player.
    pub death_time: i32,
    /// Whether the local player is currently in a bed.
    /// java: `EntityPlayer.sleeping`, entered by `S0APacketUseBed` and cleared by animation 2.
    pub sleeping: bool,
    /// Drives `GuiIngame`'s full-screen sleep fade. It rises to `100` while sleeping, then
    /// `wakeUpPlayer(false, ...)` starts the ten-tick `100..109` fade-out before it clears.
    /// java: `EntityPlayer.sleepTimer`.
    pub sleep_timer: i32,
    /// The bed head received in `S0APacketUseBed`; retained for vanilla's wake-up exit search.
    /// This is internal session state rather than a second spawn-point field.
    pub(crate) sleeping_at: Option<BlockPos>,
    /// `0..=20`. java: `FoodStats.foodLevel`
    pub food_level: i32,
    /// `0.0..=5.0`. java: `FoodStats.foodSaturationLevel`
    pub saturation: f32,
    /// `0.0..=1.0` progress across the current level. java: `EntityPlayer.experience`
    pub experience_bar: f32,
    /// java: `EntityPlayer.experienceLevel`
    pub level: i32,
    /// java: `EntityPlayer.experienceTotal`
    pub total_experience: i32,
    /// `0..=8`. java: `InventoryPlayer.currentItem`
    pub held_slot: u8,
    /// The local `EntityLivingBase` equipment view fed by `S04PacketEntityEquipment`.
    ///
    /// Java resolves the local id through `WorldClient` just like a remote id and calls
    /// `setCurrentItemOrArmor` for slots `0..=4` (held item, then boots through helmet).  The
    /// normal inventory mirror is app-owned, so this small packet-facing copy stays here for the
    /// entity render/state consumers that need the server's immediate equipment edge.
    pub equipment: [Option<ItemStack>; crate::sync::entities::EQUIPMENT_SLOTS],
    /// `Entity.serverPosX/Y/Z` for the session-owned player, in the wire's 1/32-block units.
    ///
    /// This is packet bookkeeping, not a second physics position.  `S14` adds signed deltas to
    /// these integers and `S18` replaces them before deriving a floating-point interpolation
    /// target, exactly as `NetHandlerPlayClient` does.
    pub server_pos: [i32; 3],
    /// The local entity's current packet-position shadow.  [`crate::Session`] refreshes it from
    /// the predictor at packet/tick boundaries so S18's dead-zone compares against the current
    /// Java `Entity.posX/Y/Z`, never against a stale target.
    pub(crate) packet_position: DVec3,
    /// Current body yaw/pitch shadow used by a rotation-less S14/S15 fallback.
    pub(crate) packet_yaw: f32,
    pub(crate) packet_pitch: f32,
    /// `EntityLivingBase.newPosX/newPosY/newPosZ` and target rotations for local interpolation.
    pub target_pos: DVec3,
    /// Target body yaw installed by the latest S14/S18 packet.
    pub target_yaw: f32,
    /// Target pitch installed by the latest S14/S18 packet.
    pub target_pitch: f32,
    /// `EntityLivingBase.newPosRotationIncrements`, normally three for S14/S18.
    pub interpolation_steps: i32,
    /// `Entity.onGround`, written at the end of every S14-family/S18 handler.
    pub on_ground: bool,
    /// `Entity.rotationYawHead`, written by S19 without changing body rotation.
    pub head_yaw: f32,
    /// Ordered local packet edges waiting for the session predictor/render seam.
    #[doc(hidden)]
    pub(crate) pending_entity_updates: Vec<LocalEntityUpdate>,
    /// java: `EntityPlayer.capabilities`
    pub abilities: Abilities,
    /// A velocity the server assigned to the **local** player, handed from dispatch to the
    /// session within the same packet.
    ///
    /// java: `handleEntityVelocity` (`:501`) calls `Entity.setVelocity` on
    /// whatever entity the id names, and that is how every hit moves you —
    /// `EntityLivingBase.knockBack` itself runs server-side only. The local player is
    /// deliberately absent from [`crate::sync::EntityTracker`] (its position lives in
    /// `mc_physics::Predictor`), so `apply_velocity` returned `false` and the packet was dropped
    /// for the one entity it matters most for. `sync::dispatch` stores it here because it cannot
    /// reach the predictor, and `Session::react_to` **assigns** it (vanilla does not add) right
    /// after this packet's dispatch, before the next packet: S27/S08/death/respawn and the next
    /// tick's attack all act on it in wire order, as on vanilla's `motionX/Y/Z`.
    /// On the live path this is therefore always `None` between packets.
    pub pending_velocity: Option<mc_math::DVec3>,
    /// Whether an `S08PacketPlayerPosLook` has arrived since the last (re)spawn.
    /// java: `NetHandlerPlayClient.doneLoadingTerrain`, which is what dismisses
    /// `GuiDownloadTerrain`.
    pub done_loading_terrain: bool,
}

impl Default for LocalPlayer {
    fn default() -> Self {
        LocalPlayer {
            entity_id: 0,
            entity_id_known: false,
            metadata: Metadata::new(),
            uuid: Uuid::nil(),
            username: String::new(),
            game_mode: GameMode::Survival,
            hardcore: false,
            dimension: 0,
            level_type: "default".to_owned(),
            max_players: 20,
            reduced_debug_info: false,
            health: 20.0,
            max_health: 20.0,
            has_valid_health: false,
            last_damage: 0.0,
            hurt_resistant_time: 0,
            hurt_time: 0,
            max_hurt_time: 10,
            death_time: 0,
            sleeping: false,
            sleep_timer: 0,
            sleeping_at: None,
            food_level: 20,
            saturation: 5.0,
            experience_bar: 0.0,
            level: 0,
            total_experience: 0,
            held_slot: 0,
            equipment: [const { None }; crate::sync::entities::EQUIPMENT_SLOTS],
            server_pos: [0; 3],
            packet_position: DVec3::ZERO,
            packet_yaw: 0.0,
            packet_pitch: 0.0,
            target_pos: DVec3::ZERO,
            target_yaw: 0.0,
            target_pitch: 0.0,
            interpolation_steps: 0,
            on_ground: false,
            head_yaw: 0.0,
            pending_entity_updates: Vec::new(),
            abilities: Abilities::default(),
            pending_velocity: None,
            done_loading_terrain: false,
        }
    }
}

impl LocalPlayer {
    /// Forget the network-side fields that belong to one concrete `EntityPlayerSP` generation.
    ///
    /// A JoinGame and both forms of Respawn construct a fresh Java player.  `serverPos*` starts
    /// at Java's zero default, the interpolation target has no pending increments, and no stale
    /// S04 equipment may leak into the replacement object.  The watched metadata is reset by the
    /// world lifecycle code only for JoinGame; S07's caller copies that snapshot separately.
    pub(crate) fn reset_entity_packet_state(&mut self) {
        self.equipment = [const { None }; crate::sync::entities::EQUIPMENT_SLOTS];
        self.server_pos = [0; 3];
        self.packet_position = DVec3::ZERO;
        self.packet_yaw = 0.0;
        self.packet_pitch = 0.0;
        self.target_pos = DVec3::ZERO;
        self.target_yaw = 0.0;
        self.target_pitch = 0.0;
        self.interpolation_steps = 0;
        self.on_ground = false;
        self.head_yaw = 0.0;
        self.pending_entity_updates.clear();
    }

    /// Refresh the non-authoritative current-position shadow from the predictor.
    ///
    /// Java reads the live `Entity` fields when a packet arrives.  The predictor remains the
    /// sole owner of the real position; this copy exists only so dispatch can apply fixed-point
    /// packet arithmetic and S18's dead-zone without borrowing the predictor.
    pub(crate) fn sync_entity_packet_shadow(
        &mut self,
        position: DVec3,
        yaw: f32,
        pitch: f32,
        on_ground: bool,
    ) {
        self.packet_position = position;
        self.packet_yaw = yaw;
        self.packet_pitch = pitch;
        self.on_ground = on_ground;
    }

    /// Apply one local `S04PacketEntityEquipment` slot.
    ///
    /// Valid slots are exactly `0..=4`: held item, boots, leggings, chestplate, helmet.  The
    /// dispatch seam still reports the event for malformed slots, matching its existing event
    /// policy; only the state mutation is range-checked here.
    pub(crate) fn apply_equipment_packet(&mut self, slot: i16, item: Option<ItemStack>) -> bool {
        let Ok(slot) = usize::try_from(slot) else {
            return false;
        };
        let Some(destination) = self.equipment.get_mut(slot) else {
            return false;
        };
        *destination = item;
        true
    }

    /// Apply the S14-family relative movement target to the local packet shadow.
    pub(crate) fn apply_relative_move_packet(
        &mut self,
        delta: [i8; 3],
        look: Option<(f32, f32)>,
        on_ground: bool,
    ) {
        for (server, step) in self.server_pos.iter_mut().zip(delta) {
            *server = server.wrapping_add(i32::from(step));
        }
        let (yaw, pitch) = look.unwrap_or((self.packet_yaw, self.packet_pitch));
        self.target_pos = DVec3::new(
            f64::from(self.server_pos[0]) / crate::sync::entities::FIXED_POINT_SCALE,
            f64::from(self.server_pos[1]) / crate::sync::entities::FIXED_POINT_SCALE,
            f64::from(self.server_pos[2]) / crate::sync::entities::FIXED_POINT_SCALE,
        );
        self.target_yaw = yaw;
        self.target_pitch = pitch;
        self.interpolation_steps = crate::sync::entities::INTERPOLATION_TICKS;
        self.on_ground = on_ground;
        self.pending_entity_updates
            .push(LocalEntityUpdate::Movement);
    }

    /// Apply the S18 fixed-point teleport target, including Java's current-position dead zone.
    pub(crate) fn apply_teleport_packet(
        &mut self,
        server_pos: [i32; 3],
        yaw: f32,
        pitch: f32,
        on_ground: bool,
    ) {
        self.server_pos = server_pos;
        let server = DVec3::new(
            f64::from(server_pos[0]) / crate::sync::entities::FIXED_POINT_SCALE,
            f64::from(server_pos[1]) / crate::sync::entities::FIXED_POINT_SCALE,
            f64::from(server_pos[2]) / crate::sync::entities::FIXED_POINT_SCALE,
        );
        // java: `handleEntityTeleport` keeps `entity.pos*` when all three differences are
        // strictly inside 1/32 X/Z and 1/64 Y, but still eases the packet rotation.
        self.target_pos = if (self.packet_position.x - server.x).abs() < 0.03125
            && (self.packet_position.y - server.y).abs() < 0.015_625
            && (self.packet_position.z - server.z).abs() < 0.03125
        {
            self.packet_position
        } else {
            server
        };
        self.target_yaw = yaw;
        self.target_pitch = pitch;
        self.interpolation_steps = crate::sync::entities::INTERPOLATION_TICKS;
        self.on_ground = on_ground;
        self.pending_entity_updates
            .push(LocalEntityUpdate::Movement);
    }

    /// Apply the local S19 head-only rotation edge.
    pub(crate) fn apply_head_look_packet(&mut self, head_yaw: f32) {
        self.head_yaw = head_yaw;
        self.pending_entity_updates
            .push(LocalEntityUpdate::HeadLook);
    }

    /// Drain local packet edges in wire order for [`crate::Session`].
    pub(crate) fn take_pending_entity_updates(&mut self) -> Vec<LocalEntityUpdate> {
        std::mem::take(&mut self.pending_entity_updates)
    }

    /// Apply `MathHelper.clamp_float(health, 0.0F, getMaxHealth())` without Rust's
    /// `min`/`max` NaN semantics. Java's comparison-based clamp returns the input unchanged
    /// when it is NaN, while negative finite values become exactly zero.
    /// java: `EntityLivingBase.setHealth` (`EntityLivingBase.java:855-858`) and
    /// `MathHelper.clamp_float` (`MathHelper.java:131-134`).
    #[must_use]
    fn clamp_health(health: f32, max_health: f32) -> f32 {
        if health < 0.0 {
            0.0
        } else if health > max_health {
            max_health
        } else {
            health
        }
    }

    /// Java's `EntityLivingBase.setHealth` write into DataWatcher index 6.
    fn set_health(&mut self, health: f32) {
        self.health = Self::clamp_health(health, self.max_health);
    }

    /// Apply the health component of `S06PacketUpdateHealth` through
    /// `EntityPlayerSP.setPlayerSPHealth`'s packet-side state machine.
    ///
    /// The packet's health remains authoritative. In vanilla the damage path calls
    /// `damageEntity(DamageSource.generic, old - packet_health)` after restoring the old value;
    /// without locally-modelled armour, absorption, and Resistance, that resolves to the same
    /// authoritative packet value. Keeping the intermediate branch nevertheless matters: it is
    /// what arms the local hurt timers and records `lastDamage` for the HUD/camera layer.
    /// java: `EntityPlayerSP.setPlayerSPHealth` (`1.8.9:346-375`).
    pub fn apply_server_health(&mut self, health: f32) {
        if !self.has_valid_health {
            self.set_health(health);
            self.has_valid_health = true;
            return;
        }

        let difference = self.health - health;
        if difference <= 0.0 {
            self.set_health(health);
            if difference < 0.0 {
                // java: `maxHurtResistantTime / 2`; the default is 20.
                self.hurt_resistant_time = 10;
            }
        } else {
            self.last_damage = difference;
            // java first restores the old value then calls `damageEntity(generic, difference)`.
            // This layer has no local absorption/Resistance model, so generic damage is exactly
            // the packet delta; the final setHealth call supplies Java's lower/upper clamp.
            self.set_health(self.health);
            self.hurt_resistant_time = 20;
            self.set_health(self.health - difference);
            self.hurt_time = self.max_hurt_time;
        }
    }

    /// Apply the local player's `S19PacketEntityStatus` state changes.
    ///
    /// java: `NetHandlerPlayClient.handleEntityStatus` looks the id up in `WorldClient` and
    /// calls `Entity.handleStatusUpdate` on **that** entity.  The local `EntityPlayerSP` is not
    /// stored in [`crate::sync::EntityTracker`] because its motion is owned by the predictor, so
    /// its two `EntityLivingBase` status opcodes need this parallel route instead of falling
    /// through the remote-entity map.
    ///
    /// Opcode 2 is `EntityLivingBase.handleStatusUpdate`: it sets both hurt counters to ten
    /// (and its otherwise-unmodelled limb-swing kick).  Opcode 3 calls `setHealth(0)` before
    /// `onDeath(DamageSource.generic)`; the latter has no additional local-state representation
    /// in this client, while zero health immediately feeds the movement/death consumers.
    pub fn apply_status(&mut self, opcode: i8) -> bool {
        match opcode {
            2 => {
                // java: `EntityLivingBase.handleStatusUpdate(2)` first assigns
                // `hurtResistantTime = maxHurtResistantTime`; the player default is 20.
                // This is distinct from the ten-tick visual hurt animation below.
                self.hurt_resistant_time = 20;
                self.max_hurt_time = 10;
                self.hurt_time = self.max_hurt_time;
                true
            }
            3 => {
                self.health = 0.0;
                true
            }
            _ => false,
        }
    }

    /// Advance the local player's `EntityLivingBase.onEntityUpdate` visual timers once.
    ///
    /// java: `EntityLivingBase.onEntityUpdate` (`1.8.9:385-402`) decrements `hurtTime` and
    /// `hurtResistantTime`, then invokes `onDeathUpdate` only when `getHealth() <= 0.0F`.
    /// `onDeathUpdate` increments `deathTime` and removes the entity at exactly 20; the local
    /// player remains as the render-view owner until the server's respawn packet, so its
    /// terminal render value is retained here.
    pub(crate) fn tick_living_state(&mut self) {
        if self.hurt_time > 0 {
            self.hurt_time -= 1;
        }
        if self.hurt_resistant_time > 0 {
            self.hurt_resistant_time -= 1;
        }
        if self.health <= 0.0 && self.death_time < 20 {
            self.death_time += 1;
        }
    }

    /// Begin the health pipeline for a fresh `EntityPlayerSP`.
    ///
    /// `handleRespawn` replaces the local entity even for a same-dimension respawn, so the next
    /// S06 is always its quiet first snapshot rather than a hit from the previous life. The new
    /// `EntityPlayerSP` also starts at max health before that packet arrives.
    pub fn reset_health_pipeline(&mut self) {
        // java: `EntityPlayer.preparePlayerToSpawn` restores `getMaxHealth()` and zeroes
        // `deathTime` while constructing the replacement player. The default max is 20.0.
        self.max_health = 20.0;
        self.health = self.max_health;
        self.has_valid_health = false;
        self.last_damage = 0.0;
        self.hurt_resistant_time = 0;
        self.hurt_time = 0;
        self.max_hurt_time = 10;
        self.death_time = 0;
    }
}

/// Everything the play-state dispatch mutates.
///
/// Held by the game loop and never shared with the network task — see the crate docs.
pub struct ClientState {
    /// Blocks, light, chunks, and the `hecs` ECS.
    pub world: World,
    /// Network entity id → ECS entity.
    pub entities: EntityTracker,
    /// The tab list.
    pub players: PlayerList,
    /// Objectives, scores, display slots and teams. java: `WorldClient.getScoreboard()`.
    pub scoreboard: Scoreboard,
    /// Time, weather, difficulty, border.
    pub env: Environment,
    /// The local player.
    pub local: LocalPlayer,
}

impl std::fmt::Debug for ClientState {
    /// Hand-written because `mc_world::World` has no `Debug` — a loaded world is megabytes of
    /// chunk arrays, and a log line wants the summary, not the contents.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientState")
            .field("loaded_chunks", &self.world.loaded_chunk_count())
            .field("entities", &self.entities.len())
            .field("players", &self.players.len())
            .field("objectives", &self.scoreboard.objective_count())
            .field("env", &self.env)
            .field("local", &self.local)
            .finish_non_exhaustive()
    }
}

impl Default for ClientState {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientState {
    /// A blank state, before `JoinGame`.
    #[must_use]
    pub fn new() -> ClientState {
        ClientState {
            world: World::new(),
            entities: EntityTracker::new(),
            players: PlayerList::new(),
            scoreboard: Scoreboard::new(),
            env: Environment::new(),
            local: LocalPlayer::default(),
        }
    }

    /// Throw the world away for a fresh `JoinGame`.
    ///
    /// java: `handleRespawn` (`:1056`) builds a whole new `WorldClient` when the dimension
    /// changes, and `handleJoinGame` (`:277`) always does. The tab list survives because
    /// `playerInfoMap` lives on the *handler*, not on the world — a detail that matters,
    /// because skins would otherwise be re-requested on every nether portal.
    ///
    /// Sky light is set from the dimension: hazard H6 says reading it in the Nether or the End
    /// desynchronises the whole chunk buffer, and `World.has_sky_light` is what
    /// `apply_chunk_data` consults.
    pub fn reset_world_for_join(&mut self, dimension: i32) {
        self.reset_world_for_join_with_level_type(dimension, "default");
    }

    /// Throw the world away for a fresh `JoinGame`, selecting the packet's client world type.
    ///
    /// The protocol decoder normally supplies the canonical spelling, but this boundary also
    /// normalizes direct callers exactly like Java's `WorldType.parseWorldType` fallback.
    pub fn reset_world_for_join_with_level_type(&mut self, dimension: i32, level_type: &str) {
        self.reset_world(dimension, level_type, true, true);
    }

    /// Throw the dimension-scoped world state away after `S07PacketRespawn` changes dimension.
    ///
    /// The scoreboard is server/session state, not terrain state: it remains available across a
    /// Nether/End transfer until the server sends its normal scoreboard updates.  A new
    /// `JoinGame`, on the other hand, starts a fresh server session and clears it through
    /// [`Self::reset_world_for_join`].
    pub fn reset_world_for_dimension_change(&mut self, dimension: i32) {
        self.reset_world_for_dimension_change_with_level_type(dimension, "default");
    }

    /// Replace the dimension-scoped world after a cross-dimension `Respawn`, including its
    /// packet world type. Same-dimension respawns intentionally do not call this method: Java
    /// keeps the existing `WorldClient` (and therefore its provider) in that case.
    pub fn reset_world_for_dimension_change_with_level_type(
        &mut self,
        dimension: i32,
        level_type: &str,
    ) {
        self.reset_world(dimension, level_type, false, false);
    }

    fn reset_world(
        &mut self,
        dimension: i32,
        level_type: &str,
        clear_scoreboard: bool,
        reset_local_watcher: bool,
    ) {
        // java: `Minecraft.loadWorld` constructs an EntityPlayerSP only `if (this.thePlayer ==
        // null)`, which on a connection is its first JoinGame. Sampled before the identity reset
        // below clears it. A JoinGame (`reset_local_watcher`) that finds one keeps that player.
        let player_exists = self.local.entity_id_known;
        let keeps_player = reset_local_watcher && player_exists;
        self.world = World::new();
        // The provider selection and sky-light packet shape are one dimension lifecycle edge.
        // Keeping both on the replacement World prevents the render/tint/weather readers from
        // briefly using the overworld biome source after a Nether/End Join or Respawn.
        let level_type = mc_world::world::canonical_level_type_name(level_type);
        self.world
            .set_dimension_and_level_type(dimension, level_type);
        self.entities.clear();
        if clear_scoreboard {
            self.scoreboard.clear();
        }
        // A replay's clock belongs to the replay, not to the world: the
        // border packet that follows a JoinGame in the same batch must be stamped with it.
        let replay_clock = self.env.replay_clock();
        self.env = Environment::new();
        self.env.set_replay_clock(replay_clock);
        // `WorldClient` immediately replaces `WorldInfo`'s zeroed spawn with (8, 64, 8).
        // This is observable before the server's S05 spawn-position packet, and applies to
        // every newly constructed client world (fresh JoinGame and cross-dimension Respawn).
        self.env.spawn_position = BlockPos::new(8, 64, 8);
        self.local.dimension = dimension;
        self.local.level_type = level_type.to_owned();
        // java: `NetHandlerPlayClient.doneLoadingTerrain` starts false with the connection's
        // handler and is cleared again only by a dimension-changing `handleRespawn`;
        // `handleJoinGame` never touches it, so a mid-connection JoinGame leaves it set.
        if !keeps_player {
            self.local.done_loading_terrain = false;
        }
        // A fresh JoinGame replaces the local EntityPlayerSP identity. A dimension-only
        // respawn calls this helper with `clear_scoreboard = false` and deliberately retains
        // the existing entity id/identity for S07's replacement player.
        if clear_scoreboard {
            self.local.entity_id_known = false;
        }
        // A fresh EntityPlayerSP starts with zero motion. `pending_velocity` is only the
        // dispatch-to-session hand-off of one S12 (drained by `Session::react_to` at the same
        // packet); clear it so a harness that bypasses that drain cannot leak it either.
        self.local.pending_velocity = None;
        // `serverPos*`, the S04 equipment view and the interpolation target are fields of the
        // EntityPlayerSP object, so a kept player keeps them.
        if !keeps_player {
            self.local.reset_entity_packet_state();
        }
        // `Minecraft.setDimensionAndSpawnPlayer` copies the old player's complete watcher into
        // the replacement after `loadWorld`; only a newly constructed player starts a new
        // watcher. The dimension-change caller therefore preserves this snapshot, and so does a
        // proxy's mid-connection JoinGame: `loadWorld` keeps the existing `thePlayer` (and its
        // DataWatcher, sprint bit included), which the S07 that follows copies into the
        // replacement player.
        if reset_local_watcher && !keeps_player {
            self.local.metadata = Metadata::new();
        }
        // A newly constructed EntityPlayerSP is awake; bed state belongs to the old entity, and
        // a kept player keeps its own (`preparePlayerToSpawn` does not wake it).
        if !keeps_player {
            self.local.sleeping = false;
            self.local.sleep_timer = 0;
            self.local.sleeping_at = None;
        }
    }

    /// The local player's round-trip time in milliseconds, as reported by the server.
    ///
    /// This is the only latency figure vanilla has: the client cannot measure a keep-alive round
    /// trip itself, because the server picks the id, and the client's job is only to echo it.
    /// `S38PacketPlayerListItem`'s `UPDATE_LATENCY` is where the number actually comes from, and
    /// vanilla renders it as the tab-list ping bars.
    /// java: `NetworkPlayerInfo.getResponseTime`
    #[must_use]
    pub fn latency_ms(&self) -> Option<i32> {
        self.players.get(&self.local.uuid).map(|info| info.ping_ms)
    }

    /// The world spawn point. java: `WorldInfo.getSpawn`
    #[must_use]
    pub fn spawn_position(&self) -> BlockPos {
        self.env.spawn_position
    }
}

/// Whether a dimension carries sky light (hazard H6).
///
/// java: `WorldProvider.getHasNoSky()` is `true` for `WorldProviderHell` (-1) and
/// `WorldProviderEnd` (1). Any other id is a modded/plugin dimension, and vanilla's
/// `WorldProviderSurface` is the fallback, so it does have sky.
#[must_use]
pub fn dimension_has_sky_light(dimension: i32) -> bool {
    !matches!(dimension, -1 | 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_mode_ids_match_world_settings_get_by_id() {
        assert_eq!(game_mode_from_id(-1), GameMode::NotSet);
        assert_eq!(game_mode_from_id(0), GameMode::Survival);
        assert_eq!(game_mode_from_id(1), GameMode::Creative);
        assert_eq!(game_mode_from_id(2), GameMode::Adventure);
        assert_eq!(game_mode_from_id(3), GameMode::Spectator);
        // java: getByID falls through to SURVIVAL, not to NOT_SET.
        assert_eq!(game_mode_from_id(99), GameMode::Survival);
        assert_eq!(game_mode_from_id(-7), GameMode::Survival);
    }

    #[test]
    fn ability_flags_decode_bit_by_bit() {
        assert_eq!(
            Abilities::from_flags(0x0F, 0.1, 0.2),
            Abilities {
                invulnerable: true,
                flying: true,
                allow_flying: true,
                creative_mode: true,
                allow_edit: true,
                fly_speed: 0.1,
                walk_speed: 0.2,
            }
        );
        let only_flying = Abilities::from_flags(0x02, 0.05, 0.1);
        assert!(only_flying.flying);
        assert!(!only_flying.allow_flying);
        assert!(!only_flying.invulnerable);
        assert!(!only_flying.creative_mode);
        assert!(only_flying.allow_edit, "S39 has no allowEdit flag");
    }

    /// `EntityPlayerSP.setPlayerSPHealth` distinguishes an initial S06 snapshot from a heal and
    /// a hit. This deliberately probes all three transitions, so replacing the helper with a
    /// bare `self.health = health` makes the timer/history assertions fail.
    #[test]
    fn server_health_uses_the_player_sp_first_heal_and_damage_paths() {
        let mut player = LocalPlayer::default();

        player.apply_server_health(12.0);
        assert_eq!(player.health, 12.0);
        assert!(player.has_valid_health);
        assert_eq!(player.last_damage, 0.0);
        assert_eq!(player.hurt_resistant_time, 0, "first S06 is quiet");
        assert_eq!(player.hurt_time, 0, "first S06 is quiet");

        player.apply_server_health(16.0);
        assert_eq!(player.health, 16.0);
        assert_eq!(player.last_damage, 0.0, "healing is not a hit");
        assert_eq!(player.hurt_resistant_time, 10);
        assert_eq!(player.hurt_time, 0);

        player.apply_server_health(9.5);
        assert_eq!(player.health, 9.5);
        assert_eq!(player.last_damage, 6.5);
        assert_eq!(player.hurt_resistant_time, 20);
        assert_eq!(player.max_hurt_time, 10);
        assert_eq!(player.hurt_time, 10);
    }

    #[test]
    fn server_health_never_exceeds_the_live_attribute_maximum() {
        let mut player = LocalPlayer {
            max_health: 40.0,
            ..LocalPlayer::default()
        };

        player.apply_server_health(35.0);
        assert_eq!(player.health, 35.0);
        player.max_health = 30.0;
        player.apply_server_health(35.0);
        assert_eq!(player.health, 30.0);
    }

    /// `EntityLivingBase.setHealth` delegates to Java's comparison-based
    /// `MathHelper.clamp_float`: a negative S06 is a zero-health hit, while NaN survives both
    /// comparisons and still runs through `setPlayerSPHealth`'s damage branch.
    #[test]
    fn server_health_clamps_negative_values_but_preserves_java_nan_behavior() {
        let mut player = LocalPlayer::default();
        player.apply_server_health(20.0);
        player.apply_server_health(-5.0);
        assert_eq!(
            player.health, 0.0,
            "setHealth clamps a negative value to zero"
        );
        assert_eq!(player.last_damage, 25.0);
        assert_eq!(player.hurt_resistant_time, 20);
        assert_eq!(player.hurt_time, 10);
        player.tick_living_state();
        assert_eq!(player.death_time, 1, "zero health enters onDeathUpdate");

        let mut player = LocalPlayer::default();
        player.apply_server_health(20.0);
        player.apply_server_health(f32::NAN);
        assert!(player.health.is_nan(), "Java clamp returns NaN unchanged");
        assert!(player.last_damage.is_nan(), "the Java difference is NaN");
        assert_eq!(player.hurt_resistant_time, 20);
        assert_eq!(player.hurt_time, 10);
        player.tick_living_state();
        assert_eq!(player.death_time, 0, "NaN is not <= zero in Java");
    }

    #[test]
    fn fresh_player_health_pipeline_forgets_the_previous_lifes_hurt_state() {
        let mut player = LocalPlayer::default();
        player.apply_server_health(20.0);
        player.apply_server_health(7.0);
        assert!(player.has_valid_health);
        assert_eq!(player.hurt_time, 10);

        player.reset_health_pipeline();
        assert_eq!(
            player.health, 20.0,
            "a respawned player starts at max health"
        );
        assert_eq!(player.max_health, 20.0);
        assert!(!player.has_valid_health);
        assert_eq!(player.last_damage, 0.0);
        assert_eq!(player.hurt_resistant_time, 0);
        assert_eq!(player.hurt_time, 0);
        assert_eq!(player.max_hurt_time, 10);
    }

    /// `EntityLivingBase.handleStatusUpdate(2)` is the packet-only hurt animation. It refreshes
    /// both timer families: the invulnerability window is twenty ticks while the render hurt
    /// animation is ten. A regression that only copies `performHurtAnimation` leaves the former
    /// stale and fails the first assertion.
    #[test]
    fn status_hurt_animation_refreshes_resistance_and_render_timers() {
        let mut player = LocalPlayer {
            hurt_resistant_time: 3,
            hurt_time: 1,
            max_hurt_time: 7,
            ..LocalPlayer::default()
        };

        assert!(player.apply_status(2));
        assert_eq!(player.hurt_resistant_time, 20);
        assert_eq!(player.hurt_time, 10);
        assert_eq!(player.max_hurt_time, 10);

        // The status handler ignores unrelated opcodes, leaving all local state untouched.
        assert!(!player.apply_status(1));
        assert_eq!(player.hurt_resistant_time, 20);
        assert_eq!(player.hurt_time, 10);
        assert_eq!(player.max_hurt_time, 10);
    }

    /// `EntityLivingBase.onEntityUpdate` decrements hurt timers before calling
    /// `onDeathUpdate`; a zero-health local player therefore starts at deathTime 1 on the first
    /// world tick and reaches the removal boundary at tick 20. The positive-health control keeps
    /// a stale death timer from being fabricated by an unconditional increment.
    #[test]
    fn local_death_time_advances_only_for_zero_health_and_stops_at_vanilla_boundary() {
        let mut player = LocalPlayer::default();
        player.apply_server_health(20.0);
        player.apply_server_health(0.0);
        player.hurt_time = 2;
        player.hurt_resistant_time = 2;

        player.tick_living_state();
        assert_eq!(player.death_time, 1);
        assert_eq!(player.hurt_time, 1);
        assert_eq!(player.hurt_resistant_time, 1);

        for _ in 0..19 {
            player.tick_living_state();
        }
        assert_eq!(player.death_time, 20);
        player.tick_living_state();
        assert_eq!(
            player.death_time, 20,
            "setDead at 20 leaves the render terminal value"
        );

        player.health = 20.0;
        player.tick_living_state();
        assert_eq!(
            player.death_time, 20,
            "revival does not invent a reset on the old entity"
        );

        player.reset_health_pipeline();
        assert_eq!(player.health, 20.0);
        assert_eq!(
            player.death_time, 0,
            "respawn constructs a fresh EntityPlayerSP"
        );
        player.tick_living_state();
        assert_eq!(player.death_time, 0, "a live replacement stays at zero");
    }

    /// F-P0-network.play.server-02-2: the game-mode mapper mutates an existing
    /// `PlayerCapabilities`; it does not construct one or replace its speed fields.
    #[test]
    fn game_modes_configure_existing_abilities_with_creatives_flying_exception() {
        let mut abilities = Abilities::from_flags(0x02, 0.25, 0.4);

        abilities.configure_for_game_mode(GameMode::Creative);
        assert_eq!(
            abilities,
            Abilities {
                invulnerable: true,
                flying: true,
                allow_flying: true,
                creative_mode: true,
                allow_edit: true,
                fly_speed: 0.25,
                walk_speed: 0.4,
            },
            "creative's Java branch does not assign isFlying or either speed"
        );

        abilities.configure_for_game_mode(GameMode::Adventure);
        assert!(!abilities.invulnerable);
        assert!(!abilities.flying);
        assert!(!abilities.allow_flying);
        assert!(!abilities.creative_mode);
        assert!(!abilities.allow_edit, "adventure is not allowed to edit");
        assert_eq!((abilities.fly_speed, abilities.walk_speed), (0.25, 0.4));

        abilities.configure_for_game_mode(GameMode::Spectator);
        assert!(abilities.invulnerable);
        assert!(abilities.flying);
        assert!(abilities.allow_flying);
        assert!(!abilities.creative_mode);
        assert!(!abilities.allow_edit, "spectator is an adventure game type");
    }

    /// Hazard H6: reading sky light in the Nether or the End desynchronises the chunk buffer.
    #[test]
    fn only_the_overworld_shaped_dimensions_have_sky_light() {
        assert!(dimension_has_sky_light(0));
        assert!(!dimension_has_sky_light(-1), "Nether");
        assert!(!dimension_has_sky_light(1), "End");
        assert!(dimension_has_sky_light(7), "a modded surface dimension");
    }

    #[test]
    fn reset_world_configures_sky_light_and_clears_entities_but_keeps_players() {
        use mc_protocol::play::s2c::{AddPlayerEntry, PlayerListItem};
        let mut state = ClientState::new();
        state
            .players
            .apply(&PlayerListItem::AddPlayer(vec![AddPlayerEntry {
                uuid: Uuid::from_u128(1),
                username: "Notch".into(),
                properties: vec![],
                game_mode: GameMode::Survival,
                ping: 10,
                display_name: None,
            }]))
            .unwrap();
        state.local.sleeping = true;
        state.local.sleep_timer = 73;
        state.local.sleeping_at = Some(BlockPos::new(1, 64, 1));
        state.local.pending_velocity = Some(mc_math::DVec3::new(0.5, 0.4, -0.3));

        state.reset_world_for_join(-1);
        assert!(!state.world.has_sky_light());
        assert_eq!(
            state.world.biome_source(),
            mc_world::world::BiomeSource::Nether,
            "the replacement world selects the Nether biome provider too"
        );
        assert_eq!(
            state.world.biome_at(BlockPos::new(0, 64, 0)),
            8,
            "unloaded Nether columns use the Hell biome"
        );
        assert!(state.entities.is_empty());
        assert_eq!(state.local.dimension, -1);
        assert!(!state.local.done_loading_terrain);
        assert!(!state.local.sleeping);
        assert_eq!(state.local.sleep_timer, 0);
        assert_eq!(state.local.sleeping_at, None);
        assert_eq!(state.local.pending_velocity, None);
        assert_eq!(
            state.players.len(),
            1,
            "playerInfoMap lives on the handler, not on the world"
        );

        state.reset_world_for_join(0);
        assert!(state.world.has_sky_light());
        assert_eq!(
            state.world.biome_source(),
            mc_world::world::BiomeSource::OverworldSeed0
        );
    }

    #[test]
    fn replacement_world_uses_java_world_client_spawn_until_s05() {
        let mut state = ClientState::new();
        state.env.spawn_position = BlockPos::new(-40, 70, 91);

        state.reset_world_for_join(0);
        assert_eq!(state.spawn_position(), BlockPos::new(8, 64, 8));

        state.env.spawn_position = BlockPos::new(12, 90, -3);
        state.reset_world_for_dimension_change(-1);
        assert_eq!(state.spawn_position(), BlockPos::new(8, 64, 8));
    }

    #[test]
    fn latency_comes_from_our_own_tab_list_row() {
        use mc_protocol::play::s2c::{AddPlayerEntry, PlayerListItem};
        let mut state = ClientState::new();
        state.local.uuid = Uuid::from_u128(0xABC);
        assert_eq!(state.latency_ms(), None);
        state
            .players
            .apply(&PlayerListItem::AddPlayer(vec![AddPlayerEntry {
                uuid: Uuid::from_u128(0xABC),
                username: "Me".into(),
                properties: vec![],
                game_mode: GameMode::Survival,
                ping: 73,
                display_name: None,
            }]))
            .unwrap();
        assert_eq!(state.latency_ms(), Some(73));
    }
}
