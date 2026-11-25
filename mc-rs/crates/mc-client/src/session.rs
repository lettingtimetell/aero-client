//! [`Session`]: the game loop's view of a connection.
//!
//! One object owns the [`NetHandle`], the [`ClientState`], the [`OutboundState`] and the
//! `mc_physics::Predictor`, and exposes exactly two things the frame calls:
//!
//! * [`Session::poll`] — drain whatever the network has decoded, apply it, hand back events.
//!   Non-blocking, bounded, and safe to call every frame.
//! * [`Session::tick`] — run one 20 Hz game tick: physics, entity interpolation, and the
//!   movement packet that hazard H11 cares about.
//!
//! java: the equivalent split is `Minecraft.runTick` calling
//! `NetworkManager.processReceivedPackets` (poll) and `EntityPlayerSP.onUpdate` (tick).

use mc_math::{floor_f64, math_helper, Aabb, BlockPos, DVec3, Facing, JavaRandom};
use mc_physics::player::mounted_on_living_update_preflight;
use mc_physics::prediction::Predictor;
use mc_physics::state::{MoveInput, PhysicsState, PLAYER_HEIGHT, PLAYER_WIDTH};
use mc_protocol::connection::LoginOutcome;
use mc_protocol::play::{c2s, C2SPlay};
use mc_render::entity::{EntityAnimState, LivingRenderState};
use mc_version::GameVersion;
use mc_world::{entity::MetadataValue, World};
use std::collections::{HashMap, HashSet};

use crate::app::interact;
use crate::net::{DisconnectReason, NetHandle, ToGame};
use crate::outbound::{self, OutboundState};
use crate::sync::entities::{
    running_particle_event, LocalBoatRider, LocalPigRider, LocalPlayerTarget,
    PigCarrotDurabilityEvent,
};
use crate::sync::state::LocalEntityUpdate;
use crate::sync::{apply, BorderState, ClientEvent, ClientState, TrackedKind, Watched};

// <<NW-RC:session-snapshot>> exact copies of a replay session (replay checkpoints).
mod rc_snapshot;
// 1.7.10 connections: the per-tick inbound drain and the session-owned 1.7.10 handlers.
pub(crate) mod v5;
// 1.7.10 connections: the local player's tick.
mod movement_v5;
// 1.7.10 connections: the session's own sends in their 1.7.10 form, and the screen packets'
// tick phase.
mod sends_v5;
// 1.7.10 connections: an attack on an entity the tick's packets removed after the pick.
mod attack_v5;

/// How the connection is progressing. java: roughly `NetHandlerLoginClient` →
/// `NetHandlerPlayClient` → `GuiDisconnected`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Phase {
    /// TCP connect, handshake and login are in flight. Nothing can be sent yet.
    Connecting,
    /// Login succeeded; waiting for `S01PacketJoinGame`. Vanilla shows `GuiDownloadTerrain`
    /// from `handleJoinGame` until the first `S08PacketPlayerPosLook`.
    LoggingIn,
    /// In the world.
    Playing,
    /// The connection ended, for the reason given.
    Disconnected(DisconnectReason),
}

/// The entity through which vanilla renders, picks and places the camera.
///
/// `NetHandlerPlayClient.handleCamera` stores an object reference, not just an id.  Retaining the
/// ECS handle and the session's world generation gives the Rust client the same identity safety:
/// an old camera cannot silently attach to a newly spawned entity that reuses its network id, or
/// survive a `loadWorld`/respawn world replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderViewTarget {
    /// The session-owned `EntityPlayerSP`.
    Local,
    /// A tracked entity in the current world generation.
    Entity {
        /// The server's entity id.
        network_id: i32,
        /// The exact ECS generation selected by the camera packet.
        entity: mc_world::Entity,
        /// The world generation at selection time.
        world_generation: u64,
    },
}

impl Phase {
    /// Whether packets can still be sent.
    #[must_use]
    pub fn is_live(&self) -> bool {
        !matches!(self, Phase::Disconnected(_))
    }
}

/// What the client tells the server about itself on join.
/// java: `GameSettings.sendSettingsToServer` plus the `MC|Brand` payload.
#[derive(Debug, Clone)]
pub struct JoinSettings {
    /// java: `GameSettings.language`
    pub locale: String,
    /// java: `GameSettings.renderDistanceChunks`
    pub view_distance: i8,
    /// java: `GameSettings.chatVisibility`
    pub chat_visibility: c2s::ChatVisibility,
    /// java: `GameSettings.chatColours`
    pub chat_colors: bool,
    /// The `EnumPlayerModelParts` bit field. java: `GameSettings.setModelParts`
    pub displayed_skin_parts: u8,
    /// java: `ClientBrandRetriever.getClientModName()`, which is `"vanilla"` for an unmodded
    /// client. Servers log it and some plugins branch on it. The default is vanilla's; the
    /// client always sends `Aero` instead (`app/mods/client_brand.rs`).
    pub brand: String,
    /// The difficulty id (0 peaceful .. 3 hard) a 1.7.10 connection's client settings carry.
    /// java17: `GameSettings.difficulty` in `C15PacketClientSettings`, the client's own option:
    /// 1.7.10's `handleJoinGame` does not overwrite it from the server. Unused by 1.8.9.
    pub difficulty: u8,
    /// Whether a 1.7.10 connection's client settings ask to show capes. java17:
    /// `GameSettings.showCape`, taken from the cape skin-part toggle. Unused by 1.8.9.
    pub show_cape: bool,
}

impl Default for JoinSettings {
    fn default() -> Self {
        JoinSettings {
            locale: "en_US".to_owned(),
            // Native 1.8.9 GameSettings default. Production overwrites this from options.txt,
            // but direct Session users must not silently advertise the old port default of 8.
            view_distance: 12,
            chat_visibility: c2s::ChatVisibility::Full,
            chat_colors: true,
            // All seven skin-layer bits, which is vanilla's default.
            displayed_skin_parts: 0x7F,
            brand: "vanilla".to_owned(),
            // java17 `GameSettings`: NORMAL, and capes shown.
            difficulty: 2,
            show_cape: true,
        }
    }
}

/// The server-driven movement inputs that have nowhere to live on [`ClientState`].
///
/// Everything else `mc_physics::PhysicsState` needs from the server —
/// `capabilities`, the game mode, health and the food level — is already a field of
/// [`crate::sync::LocalPlayer`] and is read straight from there each tick. These four are not,
/// because vanilla keeps them on `activePotionsMap`, the attribute map and the armour inventory
/// rather than on anything `NetHandlerPlayClient` writes wholesale.
#[derive(Debug, Clone, Default, PartialEq)]
struct MovementEffects {
    /// java: `getActivePotionEffect(Potion.jump)` — `(signed byte amplifier, remaining ticks)`.
    /// `S1DPacketEntityEffect` reads this field with `readByte`; values 128..255 therefore arrive
    /// as negative Java `byte`s and must not be clamped before the jump/fall consumers use them.
    jump_boost: Option<(i8, i32)>,
    /// java: `getActivePotionEffect(Potion.blindness)` — `(amplifier, remaining ticks)`.
    ///
    /// `None` means the server sent `S1E` to remove the effect.  In particular, zero is
    /// still present: `EntityLivingBase.updatePotionEffects` only removes a completed effect
    /// on the server, while the client retains its zero-duration entry until that packet.
    blindness: Option<(u8, i32)>,
    /// java: the `generic.movementSpeed` attribute value, from `S20PacketEntityProperties`.
    ///
    /// `None` until the server sends one, in which case `EntityPlayer.applyEntityAttributes`'
    /// registered base of `(double) 0.1F` stands — see [`Session::push_server_movement_state`]
    /// for why `capabilities.walkSpeed` is *not* this value on a client.
    movement_speed: Option<f64>,
    /// The local attribute instance's UUID-keyed modifier map after the last movement-speed
    /// snapshot. `handleEntityProperties` does `removeAllModifiers` before applying a snapshot,
    /// so this is replaced wholesale rather than merged with the previous map. Keeping the map
    /// (instead of only its folded scalar) lets the local sprint seam distinguish a server echo
    /// of the canonical sprint modifier from a snapshot that removed it.
    movement_modifiers: Option<MovementModifierStore>,
    /// The latest complete local `generic.maxHealth` snapshot. Keeping the UUID-keyed map
    /// makes replacement/removal and malformed-duplicate behavior match the movement sibling.
    max_health: Option<AttributeSnapshot>,
    /// Every S20 snapshot currently known for the local player, including names that are not
    /// registered by this vanilla client. Java replaces one attribute's complete UUID modifier
    /// set when that key arrives, but leaves unrelated keys alone; retaining the whole map lets
    /// a later client-side consumer observe that same lifecycle without throwing away modded
    /// metadata at the packet boundary.
    attributes: AttributeStore,
    /// java: `EnchantmentHelper.getDepthStriderModifier`, i.e. the boots' Depth Strider level.
    depth_strider: f32,
}

/// The part of vanilla's `ModifiableAttributeInstance` that a local movement-speed snapshot
/// needs. Java indexes modifiers by UUID and keeps a separate per-operation `HashSet`; `order`
/// preserves packet insertion order so the Java hash-bucket order can be reconstructed when a
/// fold is evaluated. The map is the identity store; the vector is only collision-chain input.
#[derive(Debug, Clone, Default, PartialEq)]
struct MovementModifierStore {
    modifiers: HashMap<uuid::Uuid, mc_protocol::play::s2c::AttributeModifierSnapshot>,
    order: Vec<uuid::Uuid>,
}

impl MovementModifierStore {
    /// Build the UUID-keyed map used by `ModifiableAttributeInstance.applyModifier`.
    ///
    /// Java throws when a UUID is already present. Returning an error here lets the packet
    /// handler reject the malformed snapshot before replacing the previous local state; a
    /// duplicate must never silently stack in the scalar resolver.
    fn from_slice(
        modifiers: &[mc_protocol::play::s2c::AttributeModifierSnapshot],
    ) -> Result<Self, AttributeResolutionError> {
        let mut store = MovementModifierStore::default();
        for modifier in modifiers {
            if store.modifiers.contains_key(&modifier.uuid) {
                return Err(AttributeResolutionError::DuplicateUuid(modifier.uuid));
            }
            store.order.push(modifier.uuid);
            store.modifiers.insert(modifier.uuid, *modifier);
        }
        Ok(store)
    }

    /// The modifiers in wire insertion order, with the UUID map as the source of truth.
    fn in_order(&self) -> Vec<mc_protocol::play::s2c::AttributeModifierSnapshot> {
        self.order
            .iter()
            .filter_map(|uuid| self.modifiers.get(uuid).copied())
            .collect()
    }

    /// The canonical local sprint amount currently in this map, if any.
    fn sprint_amount(&self) -> Option<f64> {
        self.modifiers
            .get(&SPRINT_SPEED_MODIFIER_UUID)
            .map(|modifier| modifier.amount)
    }
}

/// One complete S20 attribute snapshot with UUID identity preserved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AttributeSnapshot {
    base_value: f64,
    modifiers: MovementModifierStore,
}

impl AttributeSnapshot {
    /// Build the replacement snapshot, rejecting duplicate UUIDs before any live state changes.
    pub(crate) fn from_wire(
        base_value: f64,
        modifiers: &[mc_protocol::play::s2c::AttributeModifierSnapshot],
    ) -> Result<Self, AttributeResolutionError> {
        Ok(Self {
            base_value,
            modifiers: MovementModifierStore::from_slice(modifiers)?,
        })
    }

    /// Resolve the snapshot in Java's UUID-HashSet order.
    pub(crate) fn resolved_value(&self) -> f64 {
        resolve_attribute(self.base_value, &self.modifiers.in_order())
            .expect("the UUID-keyed snapshot was validated before storage")
    }

    /// The value without
    /// the canonical sprint modifier, as [`resolve_movement_attribute`] stores it for the
    /// predictor, but with the vanilla Speed and Slowness potion modifiers' amounts multiplied
    /// by `speed` and `slowness`. Every other modifier counts in full. At `1.0` and `1.0` it is
    /// the stored value bit for bit: the same modifiers, the same Java-order resolver, and
    /// `amount * 1.0 == amount`. It allocates, so the caller caches it.
    pub(crate) fn resolved_without_sprint_with_potion_shares(
        &self,
        speed: f64,
        slowness: f64,
    ) -> f64 {
        let modifiers = self
            .modifiers
            .in_order()
            .into_iter()
            .filter(|modifier| modifier.uuid != SPRINT_SPEED_MODIFIER_UUID)
            .map(|mut modifier| {
                if modifier.uuid == SPEED_POTION_MODIFIER_UUID {
                    modifier.amount *= speed;
                } else if modifier.uuid == SLOWNESS_POTION_MODIFIER_UUID {
                    modifier.amount *= slowness;
                }
                modifier
            })
            .collect::<Vec<_>>();
        resolve_attribute(self.base_value, &modifiers)
            .expect("the UUID-keyed snapshot was validated before storage")
    }
}

/// Generation-local S20 attribute snapshots.
///
/// `NetHandlerPlayClient.handleEntityProperties` does not replace the whole attribute map: it
/// replaces each named `ModifiableAttributeInstance` as it visits the packet and leaves every
/// other instance untouched. Keeping this store beside the owning entity (local player in
/// [`MovementEffects`], remote entity in [`crate::sync::entities::TrackedEntity`]) models that
/// distinction and, importantly, preserves unknown names for clients that know how to consume
/// them later. Vanilla's `ServersideAttributeMap` aliases each registered description (for
/// example `Movement Speed`) to its unlocalized key, so those known aliases share one slot here;
/// unknown keys are normalized only by case. The snapshot itself remains complete and is replaced
/// atomically after duplicate-UUID validation.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct AttributeStore {
    snapshots: HashMap<String, AttributeSnapshot>,
}

impl AttributeStore {
    /// Replace one named attribute, retaining the previous snapshot if the wire modifier list is
    /// malformed. The Java map is case-insensitive by both unlocalized name and description.
    pub(crate) fn apply(
        &mut self,
        property: &mc_protocol::play::s2c::AttributePropertySnapshot,
    ) -> Result<(), AttributeResolutionError> {
        let snapshot = AttributeSnapshot::from_wire(property.base_value, &property.modifiers)?;
        self.snapshots
            .insert(canonical_attribute_key(&property.key), snapshot);
        Ok(())
    }

    /// Return a snapshot by Java's case-insensitive name map.
    pub(crate) fn get(&self, key: &str) -> Option<&AttributeSnapshot> {
        self.snapshots.get(&canonical_attribute_key(key))
    }

    /// Return the first matching alias, in the order supplied by the caller.
    pub(crate) fn get_any(&self, keys: &[&str]) -> Option<&AttributeSnapshot> {
        keys.iter().find_map(|key| self.get(key))
    }
}

/// Collapse the description aliases registered by vanilla's `ServersideAttributeMap` onto the
/// same identity as their unlocalized names. Modded names have no client-side description map,
/// so retaining their lower-cased wire key is the only safe unknown-name behavior.
fn canonical_attribute_key(key: &str) -> String {
    if key.eq_ignore_ascii_case("generic.maxHealth") || key.eq_ignore_ascii_case("Max Health") {
        "generic.maxhealth".to_owned()
    } else if key.eq_ignore_ascii_case("generic.followRange")
        || key.eq_ignore_ascii_case("Follow Range")
    {
        "generic.followrange".to_owned()
    } else if key.eq_ignore_ascii_case("generic.knockbackResistance")
        || key.eq_ignore_ascii_case("Knockback Resistance")
    {
        "generic.knockbackresistance".to_owned()
    } else if key.eq_ignore_ascii_case("generic.movementSpeed")
        || key.eq_ignore_ascii_case("Movement Speed")
    {
        "generic.movementspeed".to_owned()
    } else if key.eq_ignore_ascii_case("generic.attackDamage") {
        "generic.attackdamage".to_owned()
    } else if key.eq_ignore_ascii_case("horse.jumpStrength")
        || key.eq_ignore_ascii_case("Jump Strength")
    {
        "horse.jumpstrength".to_owned()
    } else if key.eq_ignore_ascii_case("zombie.spawnReinforcements")
        || key.eq_ignore_ascii_case("Spawn Reinforcements Chance")
    {
        "zombie.spawnreinforcements".to_owned()
    } else {
        key.to_ascii_lowercase()
    }
}

/// java: `PotionEffect.combine`. `S1D` carries no ambient bit, and both effect instances created
/// by `NetHandlerPlayClient.handleEntityEffect` are non-ambient; the session needs only this
/// amplifier/duration portion for its movement predicates.
fn combine_effect<T: Copy + Ord>(stored: &mut Option<(T, i32)>, amplifier: T, duration: i32) {
    let Some((stored_amplifier, stored_duration)) = stored else {
        *stored = Some((amplifier, duration));
        return;
    };

    if amplifier > *stored_amplifier {
        *stored = Some((amplifier, duration));
    } else if amplifier == *stored_amplifier && *stored_duration < duration {
        *stored_duration = duration;
    }
}

/// A live (or recently dead) connection plus everything it drives.
#[derive(Debug)]
pub struct Session {
    handle: NetHandle,
    state: ClientState,
    outbound: OutboundState,
    predictor: Option<Predictor>,
    /// The session-owned `EntityPlayerSP`'s render samples. The local player is intentionally
    /// absent from `EntityTracker`, but F5 and the inventory preview still consume the same
    /// prev/current body, head, limb and swing values as `RendererLivingEntity`.
    local_render: LivingRenderState,
    /// `EntityLivingBase.swingProgressInt` / `isSwingInProgress` for the local player. The
    /// first-person renderer already owns its visual phase; these counters mirror the same
    /// packet/action edge for the third-person model without coupling the two render passes.
    local_swing_progress_int: i32,
    local_swinging: bool,
    local_swing_animation_end: i32,
    phase: Phase,
    settings: JoinSettings,
    /// The `C15PacketClientSettings` last sent on this connection (the join-time
    /// send or a later options save); `None` before the first. See
    /// [`Self::send_client_settings_if_changed`].
    last_client_settings: Option<c2s::ClientSettings>,
    login: Option<LoginOutcome>,
    effects: MovementEffects,
    riding: Riding,
    /// A local horse-jump release waiting for the same tick's remote entity pass. The horse is
    /// tracker-owned, so this short-lived handoff keeps its generation-specific jump impulse out
    /// of the session's local-player predictor.
    pending_horse_jump: Option<(i32, f32, f32, f32)>,
    /// One client-accepted `EntityHorse.mountTo` awaiting the matching local S1B. The ECS handle
    /// is the generation token, so despawn/reuse of the same network id cannot move the player.
    pending_horse_mount: Option<(i32, mc_world::Entity)>,
    /// Whether the shell already ran this tick's early `PlayerControllerMP.updateController`.
    /// Direct `Session::tick` callers leave this false and receive the update automatically.
    controller_updated: bool,
    /// Session-owned `EntityPlayerSP.rand` stream for the inherited running-dust producer. This
    /// is separate from tracker entity streams because the local player is not in EntityTracker.
    local_particle_random: JavaRandom,
    /// The client-side `Entity.fire` counter for the session-owned player. The server remains
    /// authoritative for damage and the watcher fire bit, but vanilla still advances this
    /// private counter on the remote client for one-tick local flame/fizz visuals.
    local_fire: i32,
    /// The previous tick's local `Entity.inWater` result. `spawnRunningParticles` runs before
    /// `handleWaterMovement`, so sampling the predictor's current value would be one tick early.
    local_running_in_water: bool,
    /// `GuiDownloadTerrain.progress` while that screen is open.
    ///
    /// Vanilla creates a fresh screen (and therefore resets this counter) on `JoinGame` and on
    /// a dimension-changing respawn, then closes it on the first player-position correction.
    terrain_download_ticks: Option<u32>,
    /// The local `EntityPlayerSP`'s forced spawn point.
    ///
    /// `NetHandlerPlayClient.handleSpawnPosition` writes both `WorldInfo.spawn` and the local
    /// player's `spawnChunk`/`spawnForced` pair. The world-scoped half lives in `ClientState`;
    /// this player-scoped half follows the session-owned player across packet dispatch and is
    /// cleared when Java constructs a replacement `EntityPlayerSP`.
    local_spawn_point: Option<BlockPos>,
    local_spawn_forced: bool,
    /// java: the local `EntityPlayerSP`'s `Entity.addedToChunk`.
    ///
    /// `World.spawnEntityInWorld` hands the player to `getChunkFromChunkCoords(x >> 4, z >> 4)`.
    /// A column the server has not sent is `ChunkProviderClient.blankChunk`, an `EmptyChunk`
    /// whose `addEntity` does nothing, so the flag stays `false`. The next
    /// `World.updateEntityWithOptionalForce` then skips `++ticksExisted` and `onUpdate()`
    /// entirely (no physics, no C03, no potion countdown) and only afterwards sets the flag,
    /// because a client's `isChunkLoaded(.., true)` is always true. So a player spawned into an
    /// unsent column (every JoinGame, every dimension-changing S07) loses exactly its first world
    /// tick; one spawned into a loaded column (a same-dimension S07 over loaded terrain) does not.
    local_added_to_chunk: bool,
    /// Java's `Minecraft.renderViewEntity`, retaining object identity across camera packets.
    view_target: RenderViewTarget,
    /// Monotonic identity for `WorldClient` replacements.  A camera target never crosses this
    /// boundary, even when the replacement world happens to reuse network ids.
    world_generation: u64,
    /// The particles mod's extra crit and sharpness emitters for hits the local player lands.
    /// Adds nothing until the client configures it.
    pub(crate) hit_particles: mc_render::particle::tuning::HitEmitters,
    /// The exact bytes of the packet [`Session::poll_next`] just applied, while
    /// the replay recorder's tap is on. Taken right after that
    /// call by [`Session::take_tapped_raw`].
    tapped_raw: Option<Box<[u8]>>,
    /// The version this connection plays, fixed when it opened ([`Session::version`]).
    version: GameVersion,
    /// 1.7.10's received-packet queue: the play messages of a 1.7.10 connection wait here for
    /// the tick's drain (`session/v5.rs`). Always empty on a 1.8.9 one.
    inbound_v5: v5::InboundV5,
    /// A 1.7.10 connection's held screen packets and send bookkeeping (`session/sends_v5.rs`).
    /// Never touched on a 1.8.9 one.
    sends_v5: sends_v5::SendsV5,
}

/// **W-RIDE.** java: `EntityPlayerSP.ridingEntity` plus the two `horseJumpPower*` fields.
///
/// `ridingEntity` is an `Entity` reference in vanilla; here it is the network id, because that is
/// what `S1BPacketEntityAttach` carries and what [`crate::sync::EntityTracker`] is keyed by. The
/// three fields are one struct because they are reset together — `handleRespawn` builds a fresh
/// `EntityPlayerSP`, so nothing on it survives a dimension change.
#[derive(Debug, Clone, Copy, Default)]
struct Riding {
    /// java: `Entity.ridingEntity`, as a network id. `None` is `isRiding() == false`.
    vehicle: Option<i32>,
    /// The exact world object resolved by the attach packet. Retaining the ECS generation keeps
    /// an id-reuse replacement from silently becoming the local player's new vehicle.
    vehicle_entity: Option<mc_world::Entity>,
    /// java: `EntityPlayerSP.horseJumpPowerCounter`.
    jump_power_counter: i32,
    /// java: `EntityPlayerSP.horseJumpPower`.
    jump_power: f32,
}

impl Session {
    const LOCAL_PARTICLE_RANDOM_SEED: i64 = 0x4C4F_4341_4C5F_524E;
    /// `EntityPlayer` replaces the base `Entity.fireResistance = 1` with `20`.
    const LOCAL_FIRE_RESISTANCE: i32 = 20;

    /// Wrap a freshly-spawned connection.
    #[must_use]
    pub fn new(handle: NetHandle, settings: JoinSettings) -> Session {
        Session {
            handle,
            state: ClientState::new(),
            outbound: OutboundState::new(),
            predictor: None,
            local_render: LivingRenderState::default(),
            local_swing_progress_int: 0,
            local_swinging: false,
            local_swing_animation_end: 6,
            phase: Phase::Connecting,
            settings,
            last_client_settings: None,
            login: None,
            effects: MovementEffects::default(),
            riding: Riding::default(),
            pending_horse_jump: None,
            pending_horse_mount: None,
            controller_updated: false,
            local_particle_random: JavaRandom::new(Self::LOCAL_PARTICLE_RANDOM_SEED),
            local_fire: 0,
            local_running_in_water: false,
            terrain_download_ticks: None,
            local_spawn_point: None,
            local_spawn_forced: false,
            // No player exists before JoinGame, whose spawn computes the real value; `true`
            // keeps a harness-injected predictor ticking from its first tick.
            local_added_to_chunk: true,
            view_target: RenderViewTarget::Local,
            world_generation: 0,
            hit_particles: mc_render::particle::tuning::HitEmitters::default(),
            tapped_raw: None,
            version: GameVersion::V1_8_9,
            inbound_v5: v5::InboundV5::default(),
            sends_v5: sends_v5::SendsV5::default(),
        }
    }

    /// Wrap a freshly-spawned connection that plays `version` ([`Session::new`] is a 1.8.9
    /// one). The version never changes for the life of the session.
    #[must_use]
    pub fn with_version(
        handle: NetHandle,
        settings: JoinSettings,
        version: GameVersion,
    ) -> Session {
        Session {
            version,
            ..Session::new(handle, settings)
        }
    }

    /// The version this connection plays: everything in a session asks here, never the
    /// setting, which may have changed since the connection opened.
    #[must_use]
    pub fn version(&self) -> GameVersion {
        self.version
    }

    fn reset_local_particle_state(&mut self) {
        self.local_particle_random = JavaRandom::new(Self::LOCAL_PARTICLE_RANDOM_SEED);
        self.local_fire = 0;
        self.local_running_in_water = false;
    }

    /// The local player's private client fire timer, if the current tick's base/move tail made
    /// it positive. The watcher bit remains the server-facing source; this fills vanilla's
    /// one-tick client prediction before that bit arrives.
    #[must_use]
    pub(crate) fn local_fire_timer(&self) -> i32 {
        self.local_fire
    }

    /// The local player's render snapshot used by F5 and the inventory preview.
    #[must_use]
    pub(crate) fn local_render_state(&self) -> &LivingRenderState {
        &self.local_render
    }

    /// Start the local third-person arm swing at the same action edge as the first-person arm.
    ///
    /// `Client::start_first_person_swing` is the one shared action seam for attacks, digging and
    /// use-item swings. Keeping this handoff here means a fast re-swing follows vanilla's
    /// `swingProgressInt >= swingTicks / 2` restart rule and never leaks into a later player
    /// generation.
    pub(crate) fn note_local_swing(&mut self, swing_ticks: i32) {
        let swing_ticks = swing_ticks.max(1);
        self.local_swing_animation_end = swing_ticks;
        if !self.local_swinging
            || self.local_swing_progress_int >= swing_ticks / 2
            || self.local_swing_progress_int < 0
        {
            self.local_swing_progress_int = -1;
            self.local_swinging = true;
        }
    }

    /// Refresh the local player's live `getArmSwingAnimationEnd()` before its tick.
    ///
    /// java: `updateArmSwingProgress` re-reads `getArmSwingAnimationEnd()` every tick, so a Haste
    /// or Mining Fatigue change mid-swing retimes the running swing at once; the value latched by
    /// [`Self::note_local_swing`] alone lagged until the next swing (H3).
    pub(crate) fn set_local_swing_animation_end(&mut self, swing_ticks: i32) {
        self.local_swing_animation_end = swing_ticks;
    }

    /// Advance the local `EntityLivingBase` arm-swing counter once per simulated player tick.
    fn tick_local_swing(&mut self) {
        let swing_ticks = self.local_swing_animation_end.max(1);
        if self.local_swinging {
            self.local_swing_progress_int += 1;
            if self.local_swing_progress_int >= swing_ticks {
                self.local_swing_progress_int = 0;
                self.local_swinging = false;
            }
        } else {
            self.local_swing_progress_int = 0;
        }
        self.local_render.swing_progress =
            self.local_swing_progress_int as f32 / swing_ticks as f32;
    }

    /// Reset the local render object whenever Java constructs a fresh `EntityPlayerSP`.
    ///
    /// This is deliberately separate from `reset_local_particle_state`: render samples, arm
    /// swing progress and body yaw are object-owned state, so a JoinGame/respawn/id lifecycle
    /// boundary must not inherit the old player's corpse or half-finished swing.
    fn reset_local_render_state(&mut self) {
        self.local_render = LivingRenderState::default();
        self.local_swing_progress_int = 0;
        self.local_swinging = false;
        self.local_swing_animation_end = 6;
        if let Some(predictor) = self.predictor.as_ref() {
            let yaw = predictor.state.yaw;
            self.local_render.body_yaw = yaw;
            self.local_render.prev_body_yaw = yaw;
            self.local_render.head_yaw = yaw;
            self.local_render.prev_head_yaw = yaw;
            self.local_render.pitch = predictor.state.pitch;
            self.local_render.prev_pitch = predictor.state.pitch;
            // java: the `EntityLivingBase` constructor's `rotationYawHead = rotationYaw` (a random
            // yaw there); the first simulated tick replaces it.
            self.state.local.head_yaw = yaw;
        }
        self.sync_local_living_render_state();
        self.local_render.snapshot_previous();
    }

    /// java: `EntityPlayer.updateEntityActionState` (`EntityPlayer.java:590`),
    /// `rotationYawHead = rotationYaw`.
    ///
    /// `EntityLivingBase.onLivingUpdate` (`:1991-2003`) reaches it only while
    /// `!isMovementBlocked()` (alive and not asleep), and only inside a simulated
    /// `EntityPlayerSP.onUpdate`, riding included. Called once per simulated tick after the
    /// player's movement: nothing in between changes the yaw or reads the head yaw. This is the
    /// yaw `getLook(1.0F)` gives the next tick's click pick ([`Session::tick_pick_ray`]).
    fn copy_local_rotation_yaw_head(&mut self) {
        if let Some(predictor) = self.predictor.as_ref() {
            if !predictor.state.movement_blocked {
                self.state.local.head_yaw = predictor.state.yaw;
            }
        }
    }

    /// Refresh the current local living samples after physics and local health timers advance.
    ///
    /// java: `EntityPlayer.updateEntityActionState` keeps `rotationYawHead == rotationYaw`,
    /// while `EntityLivingBase.onUpdate` uses its player-specific `updateDistance` body damper.
    /// The predictor owns feet/limb samples; this snapshot owns the previous/current render
    /// values and is consumed at frame rate by `LivingRenderState::params`.
    pub(crate) fn sync_local_living_render_state(&mut self) {
        let Some(predictor) = self.predictor.as_ref() else {
            return;
        };
        let physics = predictor.state;
        let local_sleeping = self.state.local.sleeping;
        let riding = self.riding.vehicle.is_some();
        let watched = Watched::new(TrackedKind::Player, &self.state.local.metadata);
        let horizontal_sq = (physics.pos.x - physics.prev_pos.x).powi(2)
            + (physics.pos.z - physics.prev_pos.z).powi(2);
        let mut candidate = self.local_render.body_yaw;
        if (horizontal_sq as f32) > 0.002_500_000_2_f32 {
            candidate = (mc_math::atan2(
                physics.pos.z - physics.prev_pos.z,
                physics.pos.x - physics.prev_pos.x,
            ) * 180.0
                / core::f64::consts::PI) as f32
                - 90.0;
        }
        if self.local_render.swing_progress > 0.0 {
            candidate = physics.yaw;
        }
        let body_yaw =
            local_player_update_distance(self.local_render.body_yaw, candidate, physics.yaw);
        // `state.local.head_yaw` (rotationYawHead) is not written here: this also runs on ticks
        // where Java's `updateEntityActionState` does not (dead, asleep, unsimulated) and from the
        // bed paths. See `copy_local_rotation_yaw_head`, which the click pick depends on.
        let local = &self.state.local;
        let render = &mut self.local_render;
        render.body_yaw = body_yaw;
        render.head_yaw = physics.yaw;
        render.pitch = physics.pitch;
        render.limb_swing = physics.limb_swing;
        render.limb_swing_amount = physics.limb_swing_amount;
        render.hurt_time = local.hurt_time.max(0) as u32;
        render.death_time = local.death_time.max(0) as u32;
        render.entity = EntityAnimState {
            entity_id: local.entity_id,
            sneaking: physics.sneaking && !local_sleeping,
            sprinting: physics.sprinting || watched.is_sprinting(),
            riding,
            invisible: watched.is_invisible(),
            ticks_existed: render.ticks_existed,
            ..EntityAnimState::default()
        };
    }

    /// Where the connection is.
    #[must_use]
    pub fn phase(&self) -> &Phase {
        &self.phase
    }

    /// The `MC|Brand` value this connection sends at each JoinGame ([`JoinSettings::brand`]).
    #[must_use]
    pub fn brand(&self) -> &str {
        &self.settings.brand
    }

    /// Whether vanilla's `GuiDownloadTerrain` is still the active screen.
    ///
    /// `NetHandlerPlayClient.doneLoadingTerrain` is the authoritative latch: a fresh join or a
    /// dimension-changing respawn opens the screen, and the first `S08PacketPlayerPosLook`
    /// closes it after the C06 confirmation has been queued.  The driver reads this level to
    /// choose the live screen without maintaining a second, drift-prone latch.
    #[must_use]
    pub fn terrain_download_active(&self) -> bool {
        self.terrain_download_ticks.is_some()
    }

    /// The `GuiDownloadTerrain.progress` value, or `None` after its first position correction.
    ///
    /// Vanilla increments this once per game tick even though the value is not drawn; exposing
    /// it keeps the visual screen and its unsolicited key-zero keep-alive on the same clock.
    #[must_use]
    pub fn terrain_download_progress(&self) -> Option<u32> {
        self.terrain_download_ticks
    }

    /// The local player's `EntityPlayer.spawnChunk`, if the server has supplied one.
    ///
    /// java: `EntityPlayer.getBedLocation`/`func_180470_cg` after
    /// `NetHandlerPlayClient.handleSpawnPosition` calls `setSpawnPoint`.
    #[must_use]
    pub fn local_spawn_point(&self) -> Option<BlockPos> {
        self.local_spawn_point
    }

    /// Whether the local player's spawn point was server-forced.
    ///
    /// S05 always invokes `setSpawnPoint(position, true)`; the separate accessor mirrors Java's
    /// `EntityPlayer.isSpawnForced` for callers that need to distinguish it from a bed spawn.
    #[must_use]
    pub fn local_spawn_forced(&self) -> bool {
        self.local_spawn_forced
    }

    /// The synced world and ECS.
    #[must_use]
    pub fn state(&self) -> &ClientState {
        &self.state
    }

    /// Mutable access, for the systems that own things S14 does not (containers, screens).
    pub fn state_mut(&mut self) -> &mut ClientState {
        &mut self.state
    }

    /// Shorthand for the block world.
    #[must_use]
    pub fn world(&self) -> &World {
        &self.state.world
    }

    /// The currently valid render-view target. Unknown ids and stale ECS generations resolve to
    /// the local player, matching Java's `handleCamera` null check (which leaves its prior
    /// target untouched) plus `loadWorld`'s explicit local-player reset.
    #[must_use]
    pub fn render_view_target(&self) -> RenderViewTarget {
        match self.view_target {
            RenderViewTarget::Entity {
                network_id,
                entity,
                world_generation,
            } if world_generation == self.world_generation
                && self
                    .state
                    .entities
                    .get_loaded(network_id)
                    .is_some_and(|tracked| tracked.entity == entity) =>
            {
                self.view_target
            }
            _ => RenderViewTarget::Local,
        }
    }

    /// Whether the local `EntityPlayerSP` is the current render-view entity.
    #[must_use]
    pub fn render_view_is_local(&self) -> bool {
        matches!(self.render_view_target(), RenderViewTarget::Local)
    }

    /// java: `EntityPlayerSP.isCurrentViewEntity()`, i.e. `mc.getRenderViewEntity() == this`.
    ///
    /// Unlike the render-side [`Self::render_view_target`], an S43 target that has since been
    /// removed (or whose id was reused) still counts as the view entity: `renderViewEntity`
    /// keeps the dead object, and only a null one falls back to `thePlayer` (`EntityRenderer`
    /// `:357`, `:1294`). Until the server's next S43, a JoinGame or a respawn, the local player
    /// therefore sends no movement report and copies no movement input (found by
    /// L4). Rendering keeps the local fallback, because the removed entity's last pose is gone.
    fn local_is_current_view_entity(&self) -> bool {
        match self.view_target {
            RenderViewTarget::Local => true,
            RenderViewTarget::Entity {
                world_generation, ..
            } => world_generation != self.world_generation,
        }
    }

    /// Select a render-view entity by its current server id.
    ///
    /// java: `NetHandlerPlayClient.handleCamera`. A missing id is intentionally a no-op, so a
    /// malformed/early packet cannot reset an already active spectator view.
    pub(crate) fn set_render_view_entity(&mut self, network_id: i32) {
        if network_id == self.state.local.entity_id {
            self.view_target = RenderViewTarget::Local;
        } else if let Some(tracked) = self.state.entities.get(network_id) {
            self.view_target = RenderViewTarget::Entity {
                network_id,
                entity: tracked.entity,
                world_generation: self.world_generation,
            };
        }
    }

    /// Position of the active render-view entity, interpolated for this frame.
    #[must_use]
    pub fn render_view_position(&self, partial_ticks: f32) -> Option<DVec3> {
        match self.render_view_target() {
            RenderViewTarget::Local => self.predictor.as_ref().map(|predictor| {
                predictor.state.prev_pos
                    + (predictor.state.pos - predictor.state.prev_pos) * f64::from(partial_ticks)
            }),
            RenderViewTarget::Entity { network_id, .. } => self
                .state
                .entities
                .get_loaded(network_id)
                .map(|tracked| tracked.render_position(&self.state.world, partial_ticks)),
        }
    }

    /// Eye position of the active render-view entity, including sleeping and live mob eye-height
    /// rules. This is the shared source for camera, pick/ray tracing and listener placement.
    #[must_use]
    pub fn render_view_eye_position(&self, partial_ticks: f32) -> Option<DVec3> {
        match self.render_view_target() {
            RenderViewTarget::Local => self.predictor.as_ref().map(|predictor| {
                let feet = predictor.state.prev_pos
                    + (predictor.state.pos - predictor.state.prev_pos) * f64::from(partial_ticks);
                if self.state.local.sleeping {
                    feet + DVec3::new(0.0, 0.2, 0.0)
                } else {
                    let eye = if predictor.state.sneaking {
                        mc_physics::state::PLAYER_EYE_HEIGHT - 0.08
                    } else {
                        mc_physics::state::PLAYER_EYE_HEIGHT
                    };
                    feet + DVec3::new(0.0, f64::from(eye), 0.0)
                }
            }),
            RenderViewTarget::Entity { network_id, .. } => self
                .state
                .entities
                .get_loaded(network_id)
                .and_then(|tracked| {
                    self.render_view_position(partial_ticks).map(|position| {
                        let eye_height = if tracked.kind == TrackedKind::Player && tracked.sleeping
                        {
                            0.2
                        } else {
                            f64::from(tracked.eye_height(&self.state.world))
                        };
                        position + DVec3::new(0.0, eye_height, 0.0)
                    })
                }),
        }
    }

    /// Interpolated raw yaw/pitch for the active view entity.
    ///
    /// java: `EntityRenderer.orientCamera` — `prevRotationPitch + (rotationPitch -
    /// prevRotationPitch) * partialTicks`, and the same for `rotationYaw`, except that an
    /// `EntityAnimal` turns the camera with `rotationYawHead`. This used to
    /// start the yaw lerp from `render.prev_body_yaw` (`prevRenderYawOffset`, the *body*, which
    /// trails the look by up to 75 degrees) and end it on the raw `rotationYaw`, so a spectating
    /// camera swung between body and look on every tick ("twitchy", item 35).
    #[must_use]
    pub fn render_view_rotation(&self, partial_ticks: f32) -> Option<(f32, f32)> {
        match self.render_view_target() {
            RenderViewTarget::Local => self
                .predictor
                .as_ref()
                .map(|predictor| (predictor.state.yaw, predictor.state.pitch)),
            RenderViewTarget::Entity { network_id, .. } => {
                self.state.entities.get_loaded(network_id).map(|tracked| {
                    let current = self
                        .state
                        .world
                        .ecs
                        .get::<&mc_world::Rotation>(tracked.entity)
                        .map(|rotation| (rotation.yaw, rotation.pitch))
                        .unwrap_or((tracked.target_yaw, tracked.target_pitch));
                    let (prev_yaw, yaw) = if spectator_camera_uses_head_yaw(tracked.kind) {
                        (tracked.render.prev_head_yaw, tracked.render.head_yaw)
                    } else {
                        (tracked.prev_rotation_yaw, current.0)
                    };
                    (
                        interpolate_view_rotation(prev_yaw, yaw, partial_ticks),
                        interpolate_view_rotation(
                            tracked.prev_rotation_pitch,
                            current.1,
                            partial_ticks,
                        ),
                    )
                })
            }
        }
    }

    /// The render-view entity's current tick rotation, java
    /// `entity.rotationYaw`/`rotationPitch`: what `EntityRenderer.orientCamera`'s eight F5
    /// clipping rays aim with. Only the view itself turns by the interpolated
    /// [`Self::render_view_rotation`] (an animal's by its head yaw); the rays take neither, so
    /// while spectating a mob or player that turned this tick the orbit is clipped along its
    /// latest rotation. For the local player the two are the same value.
    #[must_use]
    pub fn render_view_ray_rotation(&self) -> Option<(f32, f32)> {
        match self.render_view_target() {
            RenderViewTarget::Local => self
                .predictor
                .as_ref()
                .map(|predictor| (predictor.state.yaw, predictor.state.pitch)),
            RenderViewTarget::Entity { network_id, .. } => {
                self.state.entities.get_loaded(network_id).map(|tracked| {
                    self.state
                        .world
                        .ecs
                        .get::<&mc_world::Rotation>(tracked.entity)
                        .map(|rotation| (rotation.yaw, rotation.pitch))
                        .unwrap_or((tracked.target_yaw, tracked.target_pitch))
                })
            }
        }
    }

    /// The ray `EntityRenderer.getMouseOver(1.0F)` casts at the top of `Minecraft.runTick`, as
    /// `(eye, yaw, pitch)`: the render-view entity's `getPositionEyes(1.0F)` and the two angles
    /// its `getLook(1.0F)` hands to `getVectorForRotation`. Every click aims with this.
    ///
    /// * Both Java methods short-circuit `partialTicks == 1.0F` to the live fields
    ///   (`Entity.java:1461`, `:1487`), so nothing is lerped here.
    /// * `EntityLivingBase.getLook(1.0F)` (`:2171-2175`) aims with **`rotationYawHead`** and the
    ///   current `rotationPitch`. The local player's head yaw is the `rotationYaw` that
    ///   `EntityPlayer.updateEntityActionState` (`EntityPlayer.java:590`) copied during the
    ///   previous tick, so the frames of mouse motion since then turn the camera, the pitch and
    ///   the outline, but not the yaw a click aims with: that stays where the last C03 told the
    ///   server the player was looking. A plain `Entity` view target uses `rotationYaw`.
    #[must_use]
    pub fn tick_pick_ray(&self) -> Option<(DVec3, f32, f32)> {
        match self.render_view_target() {
            RenderViewTarget::Local => self.predictor.as_ref().map(|predictor| {
                let state = &predictor.state;
                // java: `EntityPlayer.getEyeHeight` - `1.62F`, `0.2F` while sleeping, then
                // `-= 0.08F` while `isSneaking()` (never while sleeping), widened from `float`.
                let eye_height = if self.state.local.sleeping {
                    0.2_f32
                } else if state.sneaking {
                    mc_physics::state::PLAYER_EYE_HEIGHT - 0.08
                } else {
                    mc_physics::state::PLAYER_EYE_HEIGHT
                };
                (
                    DVec3::new(
                        state.pos.x,
                        state.pos.y + f64::from(eye_height),
                        state.pos.z,
                    ),
                    self.state.local.head_yaw,
                    state.pitch,
                )
            }),
            RenderViewTarget::Entity { network_id, .. } => {
                let tracked = self.state.entities.get_loaded(network_id)?;
                let world = &self.state.world;
                let eye_height = if tracked.kind == TrackedKind::Player && tracked.sleeping {
                    0.2_f32
                } else {
                    tracked.eye_height(world)
                };
                let (yaw, pitch) = world
                    .ecs
                    .get::<&mc_world::Rotation>(tracked.entity)
                    .map(|rotation| (rotation.yaw, rotation.pitch))
                    .unwrap_or((tracked.target_yaw, tracked.target_pitch));
                // `render.head_yaw` is the one cell S19 and `EntityBodyHelper` both write.
                let yaw = if tracked.kind.is_living() {
                    tracked.render.head_yaw
                } else {
                    yaw
                };
                let position = tracked.position(world);
                Some((
                    DVec3::new(position.x, position.y + f64::from(eye_height), position.z),
                    yaw,
                    pitch,
                ))
            }
        }
    }

    /// Test fixture: face the local player along `yaw`/`pitch` as if it had already been facing
    /// that way for two ticks, so both picks see it: `rotationYaw`/`rotationPitch`, the
    /// `rotationYawHead` the click aims with ([`Self::tick_pick_ray`]) and its previous sample,
    /// which the outline lerps from. Poking only `predictor.state.yaw` models a mouse turn since
    /// the last tick, which vanilla's click does not see yet.
    #[cfg(test)]
    pub(crate) fn face_local_player(&mut self, yaw: f32, pitch: f32) {
        if let Some(predictor) = self.predictor.as_mut() {
            predictor.state.yaw = yaw;
            predictor.state.pitch = pitch;
        }
        self.state.local.head_yaw = yaw;
        self.local_render.head_yaw = yaw;
        self.local_render.prev_head_yaw = yaw;
    }

    /// The physics predictor, once `JoinGame` has created it.
    #[must_use]
    pub fn predictor(&self) -> Option<&Predictor> {
        self.predictor.as_ref()
    }

    /// The local `generic.movementSpeed` snapshot the last
    /// S20 left, `None` before one arrived (the `0.1` base stands then, with no potion part).
    /// Dynamic FOV's per-source sliders resolve it again with scaled potion modifiers. A direct
    /// lookup of the key both aliases fold onto keeps the per-tick caller allocation-free.
    pub(crate) fn local_movement_attribute(&self) -> Option<&AttributeSnapshot> {
        self.effects
            .attributes
            .snapshots
            .get("generic.movementspeed")
    }

    /// Mutable predictor, for a caller that needs to teleport or inspect the player.
    pub fn predictor_mut(&mut self) -> Option<&mut Predictor> {
        self.predictor.as_mut()
    }

    /// Snapshot the local player's current geometry for a client-generated entity emitter.
    pub(crate) fn local_particle_target(&self) -> Option<LocalPlayerTarget> {
        let predictor = self.predictor.as_ref()?;
        let eye_position = if self.state.local.sleeping {
            predictor.state.pos + DVec3::new(0.0, f64::from(0.2_f32), 0.0)
        } else {
            predictor.state.eye_pos()
        };
        let spectator = self
            .state
            .players
            .get(&self.state.local.uuid)
            .is_some_and(|info| info.game_mode == mc_protocol::play::s2c::GameMode::Spectator);
        Some(LocalPlayerTarget {
            position: predictor.state.pos,
            eye_position,
            spectator,
            width: predictor.state.width,
            height: predictor.state.height,
        })
    }

    /// Queue the local player's inherited `Entity.spawnRunningParticles` effect.
    ///
    /// The local player is simulated before `World.updateEntities`' remote list, so this event is
    /// placed into the tracker queue before that list's events are appended. The saved water bit
    /// deliberately comes from the prior tick; Java calls `spawnRunningParticles` before
    /// `handleWaterMovement` in `Entity.onEntityUpdate`.
    fn queue_local_running_particle(&mut self) {
        let Some(predictor) = self.predictor.as_ref() else {
            return;
        };
        let position = predictor.state.pos;
        if !player_position_is_valid(position) {
            return;
        }
        let velocity = predictor.state.vel;
        let width = predictor.state.width;
        let sprinting = predictor.state.sprinting;
        let previous_in_water = self.local_running_in_water;
        // The physics tick has now reached the post-movement water result. Save it for the next
        // invocation, after the event producer has consumed the old value.
        self.local_running_in_water = predictor.state.in_water;
        let Some(event) = running_particle_event(
            &self.state.world,
            position,
            width,
            velocity,
            sprinting,
            previous_in_water,
            &mut self.local_particle_random,
        ) else {
            return;
        };
        self.state.entities.push_particle_event(event);
    }

    /// Run the local player's client-only `Entity.moveEntity` fire/flammability tail.
    ///
    /// The server owns health and the DataWatcher fire bit, but Java still increments the private
    /// fire counter on a dry flammable block. A positive timer is enough for the first-person
    /// renderer to show the local flame one tick before the server metadata update. `Entity`'s
    /// lava tail is intentionally not reproduced here: `EntityLivingBase.onEntityUpdate` calls
    /// `extinguish()` for every remote-world living entity immediately after the inherited base
    /// tick, so a local `EntityPlayerSP` cannot retain the lava `setFire(15)` value or emit its
    /// generic wet-fire fizz in that path.
    fn advance_local_fire_tail(&mut self, raining: bool) {
        let Some((position, aabb, in_water, height)) = self.predictor.as_ref().map(|predictor| {
            (
                predictor.state.pos,
                predictor.state.aabb(),
                predictor.state.in_water,
                predictor.state.height,
            )
        }) else {
            return;
        };
        let wet = if in_water || !raining {
            in_water
        } else {
            let feet = BlockPos::from_f64(position.x, position.y, position.z);
            let head = BlockPos::from_f64(position.x, position.y + f64::from(height), position.z);
            self.state.world.is_raining_at(feet, true) || self.state.world.is_raining_at(head, true)
        };
        let flammable = self
            .state
            .world
            .is_flammable_within(&aabb.contract(0.001, 0.001, 0.001));
        if flammable {
            // `dealFireDamage(1)` is remote-world no-op. `setFire(8)` is only the Java
            // int-overflow guard; preserve its 160-tick conversion for the sentinel boundary.
            if !wet {
                self.local_fire = self.local_fire.wrapping_add(1);
                if self.local_fire == 0 {
                    self.local_fire = 8 * 20;
                }
            }
        } else if self.local_fire <= 0 {
            self.local_fire = -Self::LOCAL_FIRE_RESISTANCE;
        }
    }

    /// Drain the local player's grounded block callback produced by the latest physics tick.
    ///
    /// The signal is intentionally one-shot: the application owns client-only block reactions,
    /// while the predictor remains responsible only for selecting the exact standing block.
    pub(crate) fn take_ground_contact(&mut self) -> Option<mc_physics::state::GroundContact> {
        self.predictor.as_mut()?.state.ground_contact.take()
    }

    /// The outbound bookkeeping, for digging and placing.
    pub fn outbound_mut(&mut self) -> &mut OutboundState {
        &mut self.outbound
    }

    /// What `LoginSuccess` said, once login has finished.
    #[must_use]
    pub fn login(&self) -> Option<&LoginOutcome> {
        self.login.as_ref()
    }

    /// The capture driving this session, when it is a replay rather than a live connection.
    /// A bench harness reads its summary to state what the run loaded.
    #[must_use]
    pub fn replay_source(&self) -> Option<&crate::net::ReplaySource> {
        self.handle.replay_source()
    }

    /// Queue one message on a session watching an Aero replay
    /// ([`NetHandle::driven`]). The next [`Self::poll_next`] applies it exactly as a live packet.
    /// Returns `false` (dropping it) on any other session.
    pub(crate) fn push_driven(&mut self, message: ToGame) -> bool {
        self.handle.push_driven(message)
    }

    /// Whether this session is watching an Aero replay.
    #[must_use]
    pub fn is_driven(&self) -> bool {
        self.handle.is_driven()
    }

    /// On a session watching an Aero replay, how many packets
    /// the re-run game code produced and the handle dropped (nothing is ever sent).
    #[must_use]
    pub fn driven_dropped(&self) -> Option<u64> {
        self.handle.driven_dropped()
    }

    /// Queue a packet. Silently dropped once disconnected, which is what every vanilla
    /// `addToSendQueue` call site assumes.
    pub fn send(&mut self, packet: C2SPlay) {
        // 1.7.10: a screen's packet waits for the tick's screen phase (`session/sends_v5.rs`).
        let Some(packet) = self.hold_screen_packet(packet) else {
            return;
        };
        if self.phase.is_live() {
            let _ = self.handle.send(packet);
        }
    }

    /// Queue a protocol-5 (1.7.10) packet, built exactly as 1.7.10 sends it. The same gate as
    /// [`Session::send`] (dropped once disconnected) and the same queue, so the order of the two
    /// kinds is kept. A 1.8.9 session never sends one: it is dropped with an error log.
    pub fn send_v5(&mut self, packet: mc_protocol::v5::C2SPlay5) {
        if !self.version.is_1_7_10() {
            tracing::error!(?packet, "a 1.7.10 packet on a 1.8.9 connection; not sent");
            return;
        }
        // A screen's packet waits for the tick's screen phase (`session/sends_v5.rs`).
        let Some(packet) = self.hold_screen_packet_v5(packet) else {
            return;
        };
        if self.phase.is_live() {
            let _ = self.handle.send_v5(packet);
        }
    }

    /// java: `GameSettings.sendSettingsToServer`, called from `saveOptions`, but
    /// sent only when one of the packet's five fields differs from the last C15 this connection
    /// carried.
    ///
    /// Vanilla sends from every `saveOptions`, and `GuiScreenOptionsSounds.Button.mouseDragged`
    /// calls `saveOptions` on every rendered frame of a volume drag, so a vanilla client streams
    /// one identical C15 per frame while a sound slider is held. A repeat carries nothing the
    /// server does not already have: skipping it never sends a packet vanilla would not, and the
    /// first save after a real change (render distance, language, chat mode or colours, a skin
    /// part) still sends exactly vanilla's payload.
    ///
    /// The values are also kept for the next `JoinGame` (a proxy server switch), whose resend
    /// vanilla builds from the live `GameSettings`, not from the ones it connected with.
    ///
    /// Returns whether a packet was queued.
    pub fn send_client_settings_if_changed(&mut self, settings: &JoinSettings) -> bool {
        // 1.7.10: the same rule over 1.7.10's fields (`session/sends_v5.rs`).
        if self.version.is_1_7_10() {
            return self.send_client_settings_if_changed_v5(settings);
        }
        self.settings.locale.clone_from(&settings.locale);
        self.settings.view_distance = settings.view_distance;
        self.settings.chat_visibility = settings.chat_visibility;
        self.settings.chat_colors = settings.chat_colors;
        self.settings.displayed_skin_parts = settings.displayed_skin_parts;
        let packet = c2s::ClientSettings {
            locale: settings.locale.clone(),
            view_distance: settings.view_distance,
            chat_visibility: settings.chat_visibility,
            chat_colors: settings.chat_colors,
            displayed_skin_parts: settings.displayed_skin_parts,
        };
        if self.last_client_settings.as_ref() == Some(&packet) {
            return false;
        }
        self.last_client_settings = Some(packet.clone());
        self.send(C2SPlay::ClientSettings(packet));
        true
    }

    /// Queue several packets in order.
    pub fn send_all(&mut self, packets: impl IntoIterator<Item = C2SPlay>) {
        for packet in packets {
            self.send(packet);
        }
    }

    /// Send a chat message or command, truncated the way vanilla truncates it.
    pub fn send_chat(&mut self, message: &str) {
        self.send(outbound::chat_message(message));
    }

    /// [`Self::send_chat`] cut at `limit` UTF-16 units instead of 100: chat the player typed,
    /// which Aero always cuts at 256 (always on).
    pub fn send_chat_with_limit(&mut self, message: &str, limit: usize) {
        self.send(outbound::chat_message_with_limit(message, limit));
    }

    /// Run one `PlayerControllerMP`-style action and queue whatever packets it produced.
    ///
    /// This is the seam for digging, placing and held-item changes: the action decides *which*
    /// packets vanilla would send (including the suppression rules — see [`OutboundState`]) and
    /// this queues them in order.
    ///
    /// ```no_run
    /// # use mc_client::Session;
    /// # use mc_math::{BlockPos, Facing};
    /// # fn demo(session: &mut Session, target: BlockPos) {
    /// session.player_action(|out, packets| out.start_digging(target, Facing::Up, packets));
    /// # }
    /// ```
    pub fn player_action(&mut self, action: impl FnOnce(&mut OutboundState, &mut Vec<C2SPlay>)) {
        let mut packets = Vec::new();
        action(&mut self.outbound, &mut packets);
        self.send_all(packets);
    }

    /// Run `PlayerControllerMP.attackEntity` with the held weapon's Knockback enchantment level.
    ///
    /// The controller keeps inventory ownership in the app, so the driver supplies this already
    /// resolved level and the held stack while the session still owns packet ordering and target
    /// mutation. A successful client-side boat or minecart damage call damps the local horizontal
    /// motion and clears sprinting; the next `onUpdateWalkingPlayer` then reports that edge as
    /// `STOP_SPRINTING`.
    ///
    /// A remote player is the one target whose client-side `attackEntityFrom` succeeds and which
    /// is also an `EntityLivingBase` (`EntityOtherPlayerMP.java:34-37`), so it takes the whole
    /// successful-hit branch; see [`Self::finish_remote_player_attack`]. The return value says
    /// whether that branch reached `ItemStack.hitEntity` (`EntityPlayer.java:1410-1418`), whose
    /// inventory half the app owns. An ender dragon part also reaches it (on the dragon), so it
    /// is `true` there too, and `false` for every other target and in spectator mode.
    pub(crate) fn attack_entity_with_knockback(
        &mut self,
        entity_id: i32,
        target_kind: Option<mc_data::EntityType>,
        slot: u8,
        knockback_level: i32,
        held: Option<&mc_protocol::ItemStack>,
    ) -> bool {
        self.player_action(|out, packets| out.attack_entity(entity_id, slot, packets));
        if self.state.local.game_mode == mc_protocol::play::s2c::GameMode::Spectator {
            return false;
        }
        // java: `attackTargetEntityWithCurrentItem` runs its whole
        // local body (the target's damage hook and impulse, the 0.6 damping and
        // `setSprinting(false)`, crits, `hitEntity`) only `if (f > 0.0F || f1 > 0.0F)`
        // (`EntityPlayer.java:1325`). `f` is the local `generic.attackDamage`; `f1` is the held
        // stack's creature bonus, taken against `UNDEFINED` for every target whose client
        // `attackEntityFrom` can succeed (players, vehicles, hanging entities, crystals,
        // fireballs, dragon parts), i.e. Sharpness alone. A living mob's client hook fails
        // either way, so its creature type changes nothing visible.
        if !(self.local_attack_damage() > 0.0
            || interact::undefined_creature_damage_modifier(held) > 0.0)
        {
            return false;
        }
        // Java's local attack invokes the target's client-side damage hook immediately after
        // C02.  Keep the picker kind as a guard against id reuse, while the tracker owns the
        // Boat/Minecart impulse and LargeFireball motion+acceleration reflection. A remote player
        // has no EntityList type, so the picker kind is `None` and the tracked kind decides.
        let remote_player = target_kind.is_none()
            && self.state.entities.get(entity_id).is_some_and(|tracked| {
                interact::tracked_attack_result(tracked.kind)
                    == interact::LocalAttackResult::RemotePlayer
            });
        // java: `EntityDragonPart.attackEntityFrom` returns `EntityDragon.attackEntityFromPart`,
        // an unconditional `true` on the client too, so a part hit takes the knockback branch
        // (`motion *= 0.6`, `setSprinting(false)`) and `ItemStack.hitEntity` on the dragon
        // (`EntityPlayer.java:1397-1418`). Parts are synthetic snapshots with no network-id
        // entry, so the tracker lookup above misses them (L6).
        let dragon_part = target_kind.is_none()
            && self.state.entities.get(entity_id).is_none()
            && self.state.entities.dragon_part(entity_id).is_some();
        // java: the critical-hit flag is sampled before `attackEntityFrom` (`:1333`).
        let critical = remote_player && self.local_critical_hit();
        let (attacker_yaw, attacker_pitch) =
            self.predictor.as_ref().map_or((0.0, 0.0), |predictor| {
                (predictor.state.yaw, predictor.state.pitch)
            });
        let sprinting = self.local_watched_sprinting();
        self.state.entities.apply_local_target_attack(
            &mut self.state.world,
            entity_id,
            target_kind,
            attacker_yaw,
            attacker_pitch,
            knockback_level,
            sprinting,
        );
        if remote_player || dragon_part || interact::attack_cancels_sprint(target_kind) {
            self.cancel_sprint_after_attack(knockback_level);
        }
        if remote_player {
            self.finish_remote_player_attack(entity_id, critical, held);
        } else if interact::attack_cancels_sprint(target_kind)
            && interact::undefined_creature_damage_modifier(held) > 0.0
        {
            // java: `if (f1 > 0.0F) this.onEnchantmentCritical(targetEntity)` (`:1379-1382`)
            // runs for every target whose client `attackEntityFrom` succeeded, with `f1` taken
            // against `UNDEFINED` for a non-living one: a Sharpness hit on a boat, minecart, ender
            // crystal or large fireball shows the CRIT_MAGIC emitter too. Only players got it
            // (H3). The crit `flag` needs an `EntityLivingBase` target.
            self.state
                .entities
                .apply_crit_animation(&self.state.world, entity_id, 5);
        }
        // Particles mod: extra hit emitters. One `bool` test while it adds nothing.
        if self.hit_particles.is_active() {
            self.add_hit_particle_emitters(entity_id, held);
        }
        remote_player || dragon_part
    }

    /// The particles mod's extra crit and sharpness emitters for one hit the local player landed
    /// on a living target ([`mc_render::particle::tuning::HitEmitters::extra`]). Each is the
    /// `EntityParticleEmitter` that vanilla's own `onCriticalHit` / `onEnchantmentCritical`
    /// build, so it draws from the tracker's RNG and its particles pass the same spawn funnel as
    /// any other crit. Purely local, like those two: nothing is sent.
    fn add_hit_particle_emitters(&mut self, entity_id: i32, held: Option<&mc_protocol::ItemStack>) {
        let living = self
            .state
            .entities
            .get(entity_id)
            .is_some_and(|tracked| tracked.kind.is_living());
        if !living {
            return;
        }
        // java: `attackTargetEntityWithCurrentItem`'s crit `flag` (`:1333`) and `f1 > 0.0F`
        // (`:1379`), the conditions vanilla's own two emitters use.
        let crit = self.local_critical_hit();
        let sharp = interact::undefined_creature_damage_modifier(held) > 0.0;
        let (crits, sharps) = self.hit_particles.extra(crit, sharp);
        for _ in 0..crits {
            self.state
                .entities
                .apply_crit_animation(&self.state.world, entity_id, 4);
        }
        for _ in 0..sharps {
            self.state
                .entities
                .apply_crit_animation(&self.state.world, entity_id, 5);
        }
    }

    /// The rest of `EntityPlayer.attackTargetEntityWithCurrentItem`'s successful-hit branch for a
    /// remote-player target, after the knockback block (`EntityPlayer.java:1374-1431`).
    ///
    /// * `onCriticalHit` / `onEnchantmentCritical` are `EntityPlayerSP`'s
    ///   `EffectRenderer.emitParticleAtEntity` calls (`EntityPlayerSP.java:671-679`), the same
    ///   emitter S0B animations 4 and 5 construct, so both sources share one path. When the
    ///   server's own copy of the condition holds, `EntityPlayerMP` sends that S0B back to the
    ///   attacker too (`EntityPlayerMP.java:1075-1083`), so vanilla can show both emitters.
    /// * `applyThornEnchantments` is the tracker's; `applyArthropodEnchantments` is a no-op for an
    ///   `UNDEFINED` creature such as a player (`EnchantmentDamage.java:104-116`).
    /// * `ItemStack.hitEntity` is applied by the app, which owns the inventory.
    /// * `addExhaustion` is `!worldObj.isRemote`-gated (`EntityPlayer.java:2067-2076`);
    ///   `addStat`/`triggerAchievement` keep only independent stats (`EntityPlayerSP.java:380-389`),
    ///   which `damageDealtStat` and the overkill achievement are not; `setLastAttacker` has no
    ///   client reader; and a Fire Aspect `setFire` on the target is cleared by its own
    ///   `onEntityUpdate` (`Entity.java:482-485`) in this same tick, before any frame.
    fn finish_remote_player_attack(
        &mut self,
        entity_id: i32,
        critical: bool,
        held: Option<&mc_protocol::ItemStack>,
    ) {
        if critical {
            self.state
                .entities
                .apply_crit_animation(&self.state.world, entity_id, 4);
        }
        // java: `f1 > 0.0F` (`:1379`), with `f1` from `getModifierForCreature(held, UNDEFINED)`.
        if interact::undefined_creature_damage_modifier(held) > 0.0 {
            self.state
                .entities
                .apply_crit_animation(&self.state.world, entity_id, 5);
        }
        // java: `EnchantmentHelper.applyThornEnchantments(target, this)` (`:1391-1394`). The
        // thorns sound comes from `EntityPlayerSP.playSound`, at the local feet position.
        let attacker_id = self.state.local.entity_id;
        if let Some(position) = self.predictor.as_ref().map(|predictor| predictor.state.pos) {
            self.state
                .entities
                .apply_local_attack_thorns(entity_id, attacker_id, position);
        }
    }

    /// `EntityPlayer.attackTargetEntityWithCurrentItem`'s critical-hit flag
    /// (`EntityPlayer.java:1333`) for the local player:
    /// `fallDistance > 0 && !onGround && !isOnLadder() && !isInWater()
    /// && !isPotionActive(blindness) && ridingEntity == null`. The final
    /// `targetEntity instanceof EntityLivingBase` clause belongs to the caller.
    fn local_critical_hit(&self) -> bool {
        let Some(predictor) = self.predictor.as_ref() else {
            return false;
        };
        let state = &predictor.state;
        state.fall_distance > 0.0
            && !state.on_ground
            && !mc_physics::player::is_on_ladder(&self.state.world, state)
            && !state.in_water
            && self.effects.blindness.is_none()
            && self.riding.vehicle.is_none()
    }

    /// Run vanilla's early per-tick `PlayerControllerMP.updateController` half.
    ///
    /// The shell calls this before it drains hotbar input, matching `Minecraft.runTick`: a slot
    /// selected near the end of tick N is therefore reported at the start of tick N+1 unless a
    /// same-tick interaction calls `syncCurrentPlayItem` first. [`Session::tick`] also calls it
    /// when used directly, so headless/replay callers retain the complete controller cadence.
    pub(crate) fn update_controller(&mut self) {
        if !self.phase.is_live() || self.predictor.is_none() {
            return;
        }
        let held_slot = self.state.local.held_slot;
        let mut packets = Vec::with_capacity(1);
        self.outbound.sync_held_item(held_slot, &mut packets);
        self.send_all(packets);
        self.controller_updated = true;
    }

    /// Ask the connection task to close.
    pub fn disconnect(&mut self) {
        let _ = self.handle.disconnect();
    }

    /// Drain every inbound message that is currently ready, applying each to the world.
    ///
    /// This is the game loop path. Vanilla holds `Minecraft.scheduledTasks`' monitor and drains
    /// it to empty immediately before its tick loop (`Minecraft.java:1106-1119`), so a packet
    /// already queued for a frame must be visible to every tick in that frame. The caller is
    /// responsible for placing this once before the frame's ticks.
    ///
    /// **Never blocks.** It stops when no message is immediately available. A live connection
    /// can enqueue another packet after that check; it belongs to the following frame just as a
    /// Netty thread that missed vanilla's scheduled-task monitor does.
    pub fn poll_all(&mut self) -> Vec<ClientEvent> {
        self.poll_up_to(usize::MAX)
    }

    /// Drain up to `budget` inbound messages, applying each to the world.
    ///
    /// **Never blocks.** `budget` bounds the worst-case frame cost: a join burst can be tens of
    /// thousands of packets, and applying all of them in one frame is a multi-second stall.
    /// Whatever is left stays queued for the next call. This is useful to bounded headless
    /// callers, but is not vanilla's game-loop policy; use [`Session::poll_all`] there.
    ///
    /// A decoded packet handler failure is logged and discarded, then later queued packets still
    /// run. This is the game-thread equivalent of vanilla's handler exception reporting: packet
    /// decoding and transport failures have already terminated the network task and arrive as
    /// [`ToGame::Disconnected`], but one faulty handler must not tear down a live connection.
    pub fn poll(&mut self, budget: usize) -> Vec<ClientEvent> {
        self.poll_up_to(budget)
    }

    fn poll_up_to(&mut self, budget: usize) -> Vec<ClientEvent> {
        self.begin_poll();
        let mut events = Vec::new();
        for _ in 0..budget {
            if !self.poll_next(&mut events) {
                break;
            }
        }
        events
    }

    /// Start one frame's inbound drain. Call once per frame, before [`Session::poll_next`].
    ///
    /// A no-op on a live connection. On a capture-driven one this is the *entire* clock: one
    /// fixed step of captured time per poll, and the game loop polls exactly once per frame,
    /// so which frame a packet lands on does not depend on how fast the machine renders.
    /// See `crate::net::replay`.
    pub(crate) fn begin_poll(&mut self) {
        self.handle.advance();
    }

    /// Apply the next inbound message and append its events to `events`.
    ///
    /// Returns `false` when nothing was waiting, and also once the connection-ended message
    /// (always the last one) has been applied, after appending its event.
    ///
    /// java: `Minecraft.runGameLoop` runs `scheduledTasks` one packet at a time, and each
    /// handler's `addToSendQueue` replies reach the wire before the next handler runs. The game
    /// loop therefore applies one message and reacts to exactly its events (where client-side
    /// replies such as `C0F` are sent) before applying the next one (whose session-side reply,
    /// such as the S08 `C06`, would otherwise overtake it). [`Session::poll_all`] is
    /// [`Session::begin_poll`] followed by this until it returns `false`.
    pub(crate) fn poll_next(&mut self, events: &mut Vec<ClientEvent>) -> bool {
        let Some(message) = self.handle.try_recv() else {
            return false;
        };
        match message {
            ToGame::SocketConnected => events.push(ClientEvent::SocketConnected),
            ToGame::Connected(outcome) => {
                self.state.local.uuid = outcome.uuid;
                self.state.local.username = outcome.username.clone();
                self.login = Some(outcome);
                self.phase = Phase::LoggingIn;
            }
            ToGame::Disconnected(reason) => {
                self.phase = Phase::Disconnected(reason.clone());
                events.push(ClientEvent::Disconnected(reason));
                return false;
            }
            ToGame::Packet(packet) => self.apply_packet(packet, None, events),
            // Same queue slot, same step as `ToGame::Packet`: only the column unpack may
            // already have happened on the background thread (#15 / NET-1).
            ToGame::ChunkPacket(job) => self.apply_chunk_packet(&job, events),
            // The replay recorder's tap: the same application, and the bytes
            // kept for the recorder, which takes them right after this call.
            ToGame::RawPacket(tapped) => {
                let (packet, raw) = *tapped;
                self.tapped_raw = Some(raw);
                self.apply_packet(packet, None, events);
            }
            ToGame::RawChunkPacket(job, raw) => {
                self.tapped_raw = Some(raw);
                self.apply_chunk_packet(&job, events);
            }
            // 1.7.10: kept in order for the tick's drain (`session/v5.rs`), never applied here.
            message @ (ToGame::PacketV5(_)
            | ToGame::ChunkPacketV5(..)
            | ToGame::RawPacketV5(_)
            | ToGame::RawChunkPacketV5(..)) => return self.queue_v5(message, events),
        }
        true
    }

    /// Apply a chunk packet whose columns may have been unpacked off-thread.
    fn apply_chunk_packet(
        &mut self,
        job: &crate::sync::prebuild::PrebuildingChunkPacket,
        events: &mut Vec<ClientEvent>,
    ) {
        if let Some((packet, columns)) = job.take() {
            let prebuilt = columns.map(|columns| (job.assumed_sky(), columns));
            self.apply_packet(packet, prebuilt, events);
        }
    }

    /// The exact bytes of the packet the last [`Session::poll_next`] applied,
    /// if the replay recorder's tap was on for it.
    pub(crate) fn take_tapped_raw(&mut self) -> Option<Box<[u8]>> {
        self.tapped_raw.take()
    }

    /// Turn the replay recorder's tap on or off (see
    /// [`NetHandle::set_raw_tap`]).
    pub(crate) fn set_raw_tap(&self, on: bool) {
        self.handle.set_raw_tap(on);
    }

    /// Whether the replay recorder's tap is on.
    pub(crate) fn raw_tap(&self) -> bool {
        self.handle.raw_tap()
    }

    /// Apply one scheduled packet and react to its events — the body of the inbound drain.
    /// `prebuilt` carries a chunk packet's off-thread-unpacked columns and their sky layout.
    fn apply_packet(
        &mut self,
        packet: mc_protocol::play::S2CPlay,
        prebuilt: Option<(bool, Vec<mc_world::Chunk>)>,
        events: &mut Vec<ClientEvent>,
    ) {
        // Java's handler reads the live EntityPlayerSP position/rotation when an
        // S14/S18 packet arrives. Keep the dispatch-side fixed-point shadow aligned
        // with the predictor before applying this packet; it is bookkeeping only,
        // never a second movement authority.
        self.sync_local_entity_packet_shadow();
        let before = events.len();
        // `apply` installs S07/S2B's new mode before `react_to` sees the event.
        // Keep the old controller mode because respawn first applies it to the
        // freshly-created player, then applies S07's mode (Java :2456, :1072).
        let game_mode_before_packet = self.state.local.game_mode;
        let applied = match prebuilt {
            Some((assumed_sky, columns)) => crate::sync::dispatch::apply_prebuilt_chunks(
                &mut self.state,
                packet,
                assumed_sky,
                columns,
                events,
            ),
            None => apply(&mut self.state, packet, events),
        };
        match applied {
            Ok(()) => self.react_to(&events[..], before, game_mode_before_packet),
            Err(error) => {
                // Vanilla logs an exception thrown by a packet handler rather than
                // routing it through NetworkManager.exceptionCaught. Decode and I/O
                // errors take that latter path in `net::task` and are represented by
                // ToGame::Disconnected above, so do not conflate the two boundaries.
                tracing::error!(%error, "clientbound packet handler failed; continuing connection");
            }
        }
    }

    /// Copy the predictor's current local entity fields into the packet-only shadow.
    ///
    /// `LocalPlayer::packet_position` exists so synchronous dispatch can implement Java's
    /// `serverPos*` arithmetic and S18 dead zone without borrowing the predictor. The predictor
    /// itself remains authoritative for all rendered/collidable position state.
    fn sync_local_entity_packet_shadow(&mut self) {
        let Some((position, yaw, pitch, on_ground)) = self.predictor.as_ref().map(|predictor| {
            (
                predictor.state.pos,
                predictor.state.yaw,
                predictor.state.pitch,
                predictor.state.on_ground,
            )
        }) else {
            return;
        };
        self.state
            .local
            .sync_entity_packet_shadow(position, yaw, pitch, on_ground);
    }

    /// Apply local S14-family/S18/S19 packet edges after dispatch has mutated state.
    ///
    /// Position packets only install the target and `onGround` immediately, matching Java's
    /// handler; the three interpolation steps are consumed at the start of later local entity
    /// ticks. Head look is visible to the render snapshot immediately and does not move the body
    /// or route the local player through the remote ECS.
    fn apply_pending_local_entity_updates(&mut self) {
        let updates = self.state.local.take_pending_entity_updates();
        for update in updates {
            match update {
                LocalEntityUpdate::Movement => {
                    if let Some(predictor) = self.predictor.as_mut() {
                        predictor.state.on_ground = self.state.local.on_ground;
                    }
                }
                LocalEntityUpdate::HeadLook => {
                    self.local_render.head_yaw = self.state.local.head_yaw;
                }
            }
        }
    }

    /// The handful of packets the client must answer immediately.
    ///
    /// java: these are the `addToSendQueue` calls embedded in `handleJoinGame` and
    /// `handlePlayerPosLook`. They live here rather than in the dispatch so that
    /// [`crate::sync::apply`] stays a pure state transition and can be tested without a socket.
    fn react_to(
        &mut self,
        events: &[ClientEvent],
        from: usize,
        game_mode_before_packet: mc_protocol::play::s2c::GameMode,
    ) {
        self.apply_pending_local_entity_updates();
        // java: `handleEntityVelocity` (`NetHandlerPlayClient.java:501-509`) runs as one scheduled
        // task and `Entity.setVelocity` assigns `motionX/Y/Z` on the spot. Assign the local S12
        // here, after its own dispatch and before the next packet, so every later motion writer
        // sees it in wire order: an S27 adds to it, an S08 zeroes its absolute axes, a death
        // status or a respawn overwrites it, and the next tick's attack damps it. Parking it
        // until the tick lost a later S27 and let the pre-S12 motion survive an S08 relative
        // axis whose S12 component was exactly zero.
        if let Some(velocity) = self.state.local.pending_velocity.take() {
            if let Some(predictor) = self.predictor.as_mut() {
                predictor.state.vel = velocity;
            }
        }
        for event in events.iter().skip(from) {
            match event {
                ClientEvent::JoinedGame(_) => {
                    // Initial JoinGame follows loadWorld(null), so `thePlayer` is newly
                    // constructed with default capabilities. A mid-connection JoinGame (the
                    // proxy server-switch case) reuses the existing player: its newly-created
                    // PlayerController first configures Survival, preserving both speeds, then
                    // configures the packet mode. `Phase` still describes the pre-packet state.
                    //
                    // java: `Minecraft.loadWorld` (`:2408-2419`) constructs `thePlayer` only while
                    // it is null, i.e. on the connection's first JoinGame. A proxy's
                    // mid-connection JoinGame keeps the same EntityPlayerSP: its fields
                    // (`lastReported*`, `positionUpdateTicks`, `serverSprintState`/
                    // `serverSneakState`), watcher, attribute map and potions all survive, and
                    // only `preparePlayerToSpawn`, a fresh `MovementInputFromOptions` and the new
                    // PlayerControllerMP touch it (found by L4).
                    let reuses_player = matches!(self.phase, Phase::Playing);
                    let joined_game_mode = self.state.local.game_mode;
                    if !reuses_player {
                        self.state.local.abilities = crate::sync::Abilities::default();
                    }
                    self.state
                        .local
                        .abilities
                        .configure_for_game_mode(mc_protocol::play::s2c::GameMode::Survival);
                    self.state
                        .local
                        .abilities
                        .configure_for_game_mode(joined_game_mode);
                    self.phase = Phase::Playing;
                    // java: `Minecraft.loadWorld` clears renderViewEntity, then points it at the
                    // freshly constructed local player. The generation bump invalidates every
                    // remote ECS handle selected before this join.
                    self.reset_render_view_target();
                    // java: `handleJoinGame` -> `new GuiDownloadTerrain(this)`. The screen's
                    // `progress` field starts at zero and is advanced by `updateScreen`.
                    self.terrain_download_ticks = Some(0);
                    if reuses_player {
                        self.prepare_reused_player_to_spawn();
                        // The new PlayerControllerMP has `currentPlayerItem == 0` and no dig;
                        // the kept player's serverSprintState/serverSneakState stay.
                        self.outbound.reset_player_controller();
                    } else {
                        // A newly constructed EntityPlayerSP: neither an attribute map nor
                        // activePotionsMap, and no spawnChunk/spawnForced yet.
                        self.effects = MovementEffects::default();
                        self.local_spawn_point = None;
                        self.local_spawn_forced = false;
                        // java: the `EntityPlayer` constructor places the new player at the
                        // fresh WorldClient's spawn, `(8, 64, 8)` plus `(0.5, 1, 0.5)`, and
                        // `flipPlayer` turns it to -180 (`preparePlayerToSpawn` leaves pitch and
                        // motion zero). The S08 that follows really places the player, but a
                        // tick that runs before it reports from here, as vanilla's does (L5).
                        let spawn = self.state.spawn_position();
                        let mut joined = PhysicsState::new_player(DVec3::new(
                            f64::from(spawn.x) + 0.5,
                            f64::from(spawn.y) + 1.0,
                            f64::from(spawn.z) + 0.5,
                        ));
                        joined.yaw = -180.0;
                        self.predictor = Some(Predictor::new(joined));
                        // `loadWorld` -> `spawnEntityInWorld` into the new, empty WorldClient.
                        self.local_added_to_chunk = self.local_spawn_column_is_loaded();
                        self.reset_local_render_state();
                        self.reset_local_particle_state();
                        self.outbound = OutboundState::new();
                    }
                    // java: a first JoinGame's new player has no `ridingEntity`. A kept player
                    // keeps a reference to a vehicle of the discarded WorldClient, which the new
                    // world never ticks (so neither does the rider) until the S07 a proxy sends
                    // next replaces the player; the vehicle cannot be resolved here, so the edge
                    // is cleared in both cases. W-RIDE.
                    self.riding = Riding::default();
                    self.pending_horse_jump = None;
                    self.pending_horse_mount = None;
                    // 1.7.10: the C15 and MC|Brand in their 1.7.10 form (`session/sends_v5.rs`).
                    if self.version.is_1_7_10() {
                        self.send_join_packets_v5(!reuses_player);
                        continue;
                    }
                    // java: handleJoinGame (:290-291) — settings first, then the brand.
                    let settings = outbound::client_settings(
                        &self.settings.locale,
                        self.settings.view_distance,
                        self.settings.chat_visibility,
                        self.settings.chat_colors,
                        self.settings.displayed_skin_parts,
                    );
                    let brand = outbound::brand_message(&self.settings.brand);
                    // The join-time send is unconditional (vanilla); remember it
                    // so a later options save with nothing changed sends no duplicate.
                    if let C2SPlay::ClientSettings(sent) = &settings {
                        self.last_client_settings = Some(sent.clone());
                    }
                    self.send(settings);
                    self.send(brand);
                }
                // java: `NetHandlerPlayClient.handleCamera` resolves the id and calls
                // `Minecraft.setRenderViewEntity` only when the entity exists. Unknown ids are
                // deliberately ignored, preserving the previous target.
                ClientEvent::CameraChanged { entity_id } => {
                    self.set_render_view_entity(*entity_id);
                }
                // Entity id reuse is a new Java object. If it was the active camera, the old
                // object stays the view entity until a fresh S43 selects the replacement.
                ClientEvent::EntitySpawned { network_id, .. } => {
                    // A reused server id is a new Java object. A mounted local player keeps the
                    // old object reference, so the replacement cannot become its vehicle.
                    if self.riding.vehicle == Some(*network_id) {
                        self.clear_riding();
                    }
                    // The camera keeps its old (generation-tagged) object, exactly like
                    // `renderViewEntity`: `render_view_target` stops resolving it for drawing,
                    // and `local_is_current_view_entity` stays false until the next S43 (L5).
                }
                ClientEvent::EntityRemoved { network_id } => {
                    // World removal invalidates the object retained by ridingEntity. Clear the
                    // local edge at the packet boundary so the next tick sends walking packets.
                    // A removed S43 target, by contrast, stays the view entity (see above).
                    if self.riding.vehicle == Some(*network_id) {
                        self.clear_riding();
                    }
                }
                // java: `handleSpawnPosition` (:959-964) calls
                // `thePlayer.setSpawnPoint(position, true)` in addition to the world-info write
                // performed by `sync::dispatch`. Keep this player-scoped value on the session;
                // it must not be inferred from `ClientState::spawn_position`, because that is
                // reset when a new world/player is constructed.
                ClientEvent::SpawnPositionChanged(position) => {
                    self.local_spawn_point = Some(*position);
                    self.local_spawn_forced = true;
                }
                // java: `handleEntityEffect` (`:1502`) → `addPotionEffect`. Only
                // the two effects `moveEntityWithHeading`/`jump()`/`fall()` actually read are
                // kept here; Speed and Slowness are deliberately **absent**, because
                // `EntityLivingBase.onNewPotionEffect` (`:806`) applies their attribute modifiers
                // inside `if (!this.worldObj.isRemote)` — on a client they arrive as a separate
                // `S20PacketEntityProperties` instead, handled below.
                ClientEvent::EntityEffectAdded(effect)
                    if effect.entity_id == self.state.local.entity_id =>
                {
                    match effect.effect_id {
                        // java: `Potion.jump.id`. Read through `isPotionActive` in `jump()` and
                        // `fall()`, neither of which is `isRemote`-gated.
                        8 => combine_effect(
                            &mut self.effects.jump_boost,
                            effect.amplifier,
                            effect.duration,
                        ),
                        // java: `Potion.blindness.id` — blocks a sprint from *starting*.
                        // Keep its amplifier even though the movement predicate only needs
                        // presence: `PotionEffect.combine` uses it to decide whether a later
                        // S1D can replace the stored duration.
                        15 => combine_effect(
                            &mut self.effects.blindness,
                            u8::try_from(effect.amplifier).unwrap_or(0),
                            effect.duration,
                        ),
                        _ => {}
                    }
                }
                // java: `handleRemoveEntityEffect` (`:1606`).
                ClientEvent::EntityEffectRemoved {
                    network_id,
                    effect_id,
                } if *network_id == self.state.local.entity_id => match effect_id {
                    8 => self.effects.jump_boost = None,
                    15 => self.effects.blindness = None,
                    _ => {}
                },
                // GAP `A11-02`/`F-P0-network.play.server-08-4`. java:
                // `handleEntityProperties` (`:2051`) resolves the current entity and replaces
                // each named attribute's complete modifier set. Keep a generation-local map for
                // *all* names first, then mirror the two local consumers into the predictor and
                // health pipeline. Remote entities take the same event through EntityTracker
                // below; they must not overwrite the local player's state.
                ClientEvent::EntityProperties(properties) => {
                    let local = properties.entity_id == self.state.local.entity_id;
                    if local {
                        let mut movement_attribute_updated = false;
                        for property in &properties.properties {
                            match self.effects.attributes.apply(property) {
                                Ok(()) => {
                                    movement_attribute_updated |=
                                        canonical_attribute_key(&property.key)
                                            == "generic.movementspeed";
                                }
                                Err(error) => {
                                    tracing::error!(
                                        ?error,
                                        entity_id = properties.entity_id,
                                        attribute = %property.key,
                                        "rejected duplicate attribute modifier"
                                    );
                                }
                            }
                        }
                        if movement_attribute_updated {
                            if let Some(snapshot) = self
                                .effects
                                .attributes
                                .get_any(&["generic.movementSpeed", "Movement Speed"])
                            {
                                match resolve_movement_attribute(
                                    snapshot.base_value,
                                    &snapshot.modifiers.in_order(),
                                ) {
                                    Ok(resolution) => {
                                        // Java's removeAllModifiers + applyModifier sequence
                                        // replaces the complete UUID map for this snapshot. In
                                        // particular, a missing canonical sprint UUID removes it.
                                        let sprint_modifier_amount =
                                            resolution.modifiers.sprint_amount();
                                        self.effects.movement_speed =
                                            Some(resolution.value_without_sprint);
                                        self.effects.movement_modifiers =
                                            Some(resolution.modifiers);
                                        // The Java attribute map is mutated at this packet
                                        // boundary; do not replay the snapshot on every later
                                        // player tick. A local setSprinting edge may add the
                                        // canonical modifier again, and that local edge persists
                                        // until the next S20.
                                        if let Some(predictor) = self.predictor.as_mut() {
                                            predictor.state.sprint_modifier_amount =
                                                sprint_modifier_amount;
                                        }
                                    }
                                    Err(error) => {
                                        tracing::error!(
                                            ?error,
                                            entity_id = properties.entity_id,
                                            "rejected duplicate movement attribute modifier"
                                        );
                                    }
                                }
                            }
                        }
                        if let Some(snapshot) = self
                            .effects
                            .attributes
                            .get_any(&["generic.maxHealth", "Max Health"])
                        {
                            let max_health = snapshot.resolved_value().clamp(0.0, 1024.0) as f32;
                            self.effects.max_health = Some(snapshot.clone());
                            self.state.local.max_health = max_health;
                            self.state.local.health = self.state.local.health.min(max_health);
                        }
                    } else {
                        // `EntityTracker` owns remote snapshots, including unknown names, and
                        // validates the id's current living generation before mutating it.
                        self.state
                            .entities
                            .apply_attributes(properties.entity_id, properties);
                    }
                }
                // **W-RIDE.** java: `handleEntityAttach` (`:966`) →
                // `entity.mountEntity(getEntityByID(vehicleId))`, which for a `-1` vehicle id is
                // `mountEntity(null)` — the dismount. Only the branch for the **local** player
                // is here: another entity's mount is `sync/entities.rs`' business (a spider
                // jockey's skeleton is positioned by `updateRiderPosition`), and the leash
                // branch (`getLeash() == 1`) is a renderer concern, not an outbound one.
                //
                // `S1B` is what a 1.8.9 client learns riding from — **not** watched index 0's
                // riding bit, which the server never sets for the player's own entity (see
                // `sync/watched.rs`' note on `EntityFlags::is_riding`).
                ClientEvent::EntityAttached(attach)
                    if attach.attach_type == mc_protocol::play::s2c::AttachType::Vehicle
                        && attach.entity_id == self.state.local.entity_id =>
                {
                    // Java resolves the vehicle through World before calling mountEntity. A
                    // nonnegative id that is absent (or whose ECS generation is stale) resolves
                    // to null, which is the same dismount path as the -1 wire sentinel.
                    let vehicle = if attach.vehicle_entity_id >= 0 {
                        self.resolve_vehicle(attach.vehicle_entity_id)
                            .map(|entity| (attach.vehicle_entity_id, entity))
                    } else {
                        None
                    };
                    let horse_snap = if vehicle.is_some() {
                        self.confirm_horse_mount(attach.vehicle_entity_id)
                    } else {
                        self.pending_horse_mount = None;
                        None
                    };
                    if let Some((vehicle_id, vehicle_entity)) = vehicle {
                        self.riding.vehicle = Some(vehicle_id);
                        self.riding.vehicle_entity = Some(vehicle_entity);
                        self.state.entities.set_local_boat_rider(Some(vehicle_id));
                    } else {
                        self.clear_riding();
                    }
                    if let (Some(position), Some(predictor)) = (horse_snap, self.predictor.as_mut())
                    {
                        // java: a rider is only ever placed through `Entity.setPosition`
                        // (`updateRiderPosition`), which rebuilds the bounding box with the
                        // position; assigning `pos` alone left the box at the old spot (L4/L5).
                        // The rotation was set by the click.
                        predictor.state.set_position(position);
                    }
                }
                // java: `handleRespawn` (`:1071`) → `setDimensionAndSpawnPlayer` →
                // `respawnPlayer`, which builds a new `EntityPlayerSP`; the old one's
                // `ridingEntity` goes with it. W-RIDE.
                ClientEvent::Respawned { changed_world, .. } => {
                    // `setDimensionAndSpawnPlayer` sets the render view to the replacement local
                    // player even for a same-dimension respawn.
                    self.reset_render_view_target();
                    self.riding = Riding::default();
                    self.pending_horse_jump = None;
                    self.pending_horse_mount = None;
                    // setDimensionAndSpawnPlayer always constructs a fresh PlayerCapabilities.
                    // It configures the *controller's old mode* at Minecraft.java:2456 before
                    // handleRespawn installs S07's new mode at NetHandlerPlayClient.java:1072.
                    // Usually those modes are identical; retaining both stages is observable for
                    // spectator -> creative because Creative deliberately does not clear flying.
                    let respawn_game_mode = self.state.local.game_mode;
                    self.state.local.abilities = crate::sync::Abilities::default();
                    self.state
                        .local
                        .abilities
                        .configure_for_game_mode(game_mode_before_packet);
                    self.state
                        .local
                        .abilities
                        .configure_for_game_mode(respawn_game_mode);
                    // `setDimensionAndSpawnPlayer` constructs a fresh EntityPlayerSP even for a
                    // same-dimension death respawn. Reset that entity's complete client-local
                    // state, while leaving PlayerControllerMP's held-slot cache and active dig
                    // intact; see `reset_player_after_respawn` for the exact split.
                    self.reset_player_after_respawn();
                    if *changed_world {
                        // java: `handleRespawn` opens a *new* GuiDownloadTerrain only in the
                        // dimension-change branch, resetting that screen's progress counter.
                        self.terrain_download_ticks = Some(0);
                    }
                }
                // java: `handleUseBed` (`NetHandlerPlayClient:903`) calls `trySleep` on the
                // named entity. The local player is deliberately absent from EntityTracker, so
                // its copy of this transition lives here with the predictor.
                ClientEvent::EnteredBed {
                    network_id,
                    bed_position,
                } if *network_id == self.state.local.entity_id => {
                    self.enter_bed(*bed_position);
                }
                // The local player is session-owned; every other player remains a tracker
                // generation. Keep the remote bed transition in that generation so its
                // dimensions, bed offsets and later wake-up cannot bleed across id reuse.
                ClientEvent::EnteredBed {
                    network_id,
                    bed_position,
                } => {
                    self.state.entities.enter_bed(
                        &mut self.state.world,
                        *network_id,
                        *bed_position,
                    );
                }
                // handleChangeGameState reason 3 calls PlayerControllerMP.setGameType, which
                // configures the existing capabilities object without resetting its speeds.
                ClientEvent::GameModeChanged(mode) => {
                    self.state.local.abilities.configure_for_game_mode(*mode)
                }
                // java: `handleAnimation` type 2 (`:881`) calls
                // `wakeUpPlayer(false, false, false)`. It does not send C0B; the explicit
                // leave-bed action is the GUI button's separate responsibility.
                ClientEvent::EntityAnimation {
                    network_id,
                    animation: 2,
                } if *network_id == self.state.local.entity_id => self.wake_from_bed(),
                ClientEvent::EntityAnimation {
                    network_id,
                    animation: 2,
                } => {
                    self.state
                        .entities
                        .wake_from_bed(&mut self.state.world, *network_id);
                }
                // java: `EntityLivingBase.handleStatusUpdate(2)` also writes
                // `limbSwingAmount = 1.5F`; the state-layer call above owns the hurt timers, but
                // the predictor/render snapshot needs the animation kick before the next tick.
                ClientEvent::EntityStatus {
                    network_id,
                    opcode: 2,
                } if *network_id == self.state.local.entity_id => {
                    if let Some(predictor) = self.predictor.as_mut() {
                        predictor.state.limb_swing_amount = 1.5;
                    }
                    self.local_render.limb_swing_amount = 1.5;
                }
                // java: `EntityLivingBase.handleStatusUpdate(3)` calls
                // `EntityPlayer.onDeath(DamageSource.generic)` immediately, at this packet's
                // exact place in the inbound stream. Apply the resize and motion here rather
                // than deferring them to the next physics tick: a later S12 must overwrite the
                // death kick, while a later death status must overwrite an already-parked S12.
                ClientEvent::EntityStatus {
                    network_id,
                    opcode: 3,
                } if *network_id == self.state.local.entity_id => {
                    self.apply_local_player_death_pose();
                }
                ClientEvent::PositionCorrection(correction) => {
                    let correction = *correction;
                    // `Entity.setPositionAndRotation` snapshots prevPos to the corrected
                    // position on every S08. On the first S08 after JoinGame or a dimension
                    // change this is especially important: the fresh Rust predictor still has
                    // its constructor position, so leaving prev_pos untouched makes render
                    // interpolation briefly lerp from the origin (or the old dimension's
                    // position).
                    // An earlier local S12 has already been assigned at its own packet (top of
                    // this function), so `apply_correction`'s per-axis zeroing acts on it in wire
                    // order, exactly as `handlePlayerPosLook` acts on `motionX/Y/Z`.
                    let confirm = match self.predictor.as_mut() {
                        Some(predictor) => {
                            predictor.apply_correction(&correction);
                            predictor.state.prev_pos = predictor.state.pos;
                            // S14 shipped a `set_position` re-application here, working around
                            // `Predictor::apply_correction` assigning the public `pos` field
                            // without rebuilding `PhysicsState`'s private, authoritative
                            // bounding box — which made the next `tick_player` sweep the stale
                            // box and undo every teleport. That was fixed at the
                            // source (`apply_correction` now routes through
                            // `PhysicsState::set_position`, as vanilla's `handlePlayerPosLook` →
                            // `setPositionAndRotation` → `setPosition` does), so the workaround
                            // is gone and the regression test below now guards the real fix.

                            // java: `handlePlayerPosLook` (:717) hard-codes `onGround = false`
                            // in the confirmation, regardless of the real ground state. S13's
                            // `Predictor::apply_correction` returns the physical value instead,
                            // so the packet is built here rather than from its return value.
                            outbound::confirm_position(
                                predictor.state.pos,
                                predictor.state.yaw,
                                predictor.state.pitch,
                            )
                        }
                        None => continue,
                    };
                    self.send(confirm);
                    self.sync_local_entity_packet_shadow();
                }
                // java: the first `handlePlayerPosLook` closes GuiDownloadTerrain after sending
                // its C06 confirmation. `sync::dispatch` emits this immediately after the
                // PositionCorrection event, preserving that order here.
                ClientEvent::TerrainDownloaded => self.terrain_download_ticks = None,
                _ => {}
            }
        }
    }

    /// Read the local player's sprint edge from watched index 0, bit 3.
    ///
    /// java: `Entity.isSprinting()` delegates to `DataWatcher`'s shared flag byte.  The
    /// predictor mirrors this value for physics, but the packet edge must use the watched state
    /// that represents the local `EntityPlayerSP`, not a second client-only boolean.
    fn local_watched_sprinting(&self) -> bool {
        Watched::new(TrackedKind::Player, &self.state.local.metadata).is_sprinting()
    }

    /// Keep the local `EntityPlayerSP` watcher in step with the client-side prediction.
    ///
    /// Index 0 is a byte in vanilla's `Entity.entityInit`; preserve all unrelated flag bits when
    /// changing only bit 3.  A missing or malformed local slot has the same all-clear default as
    /// `DataWatcher.getFlag`, then becomes a valid byte after the first simulated tick.
    fn set_local_watched_sprinting(&mut self, sprinting: bool) {
        let current = match self.state.local.metadata.get(0) {
            Some(MetadataValue::Byte(flags)) => *flags,
            _ => 0,
        };
        let sprint_bit = 1_i8 << 3;
        let flags = if sprinting {
            current | sprint_bit
        } else {
            current & !sprint_bit
        };
        self.state.local.metadata.set(0, MetadataValue::Byte(flags));
    }

    /// Apply the client-local half of a successful target attack.
    ///
    /// java: `EntityPlayer.attackTargetEntityWithCurrentItem` multiplies `motionX` and
    /// `motionZ` by `0.6D` and calls `setSprinting(false)` when the target's client-side damage
    /// result succeeds with a positive sprint/Knockback increment.  The C02 packet has already
    /// been sent by [`Self::attack_entity`] before this method runs.
    /// java: `(float) getEntityAttribute(SharedMonsterAttributes
    /// .attackDamage).getAttributeValue()` for the local `EntityPlayerSP`.
    ///
    /// `EntityPlayer.applyEntityAttributes` registers it with base 1.0 and the client never
    /// applies held-item or potion modifiers itself (both are server-side), so only an S20 for the
    /// local player changes it. `RangedAttribute.clampValue` keeps the resolved value inside the
    /// shared attribute's `0..2048` (`SharedMonsterAttributes.java:20`).
    fn local_attack_damage(&self) -> f32 {
        self.effects
            .attributes
            .get("generic.attackDamage")
            .map_or(1.0, |snapshot| {
                snapshot.resolved_value().clamp(0.0, 2048.0) as f32
            })
    }

    fn cancel_sprint_after_attack(&mut self, knockback_level: i32) {
        let sprinting = self.local_watched_sprinting();
        if knockback_level + i32::from(sprinting) <= 0 {
            return;
        }
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        // java: `handleEntityVelocity` assigns the local motion when its scheduled task runs at
        // frame start, before `runTick`'s input section reaches this multiply, so a hit in the
        // knockback tick damps the S12 itself. `react_to` assigns a local S12 at its own packet
        // (L5); this fold keeps any velocity still deferred in `pending_velocity` on the same
        // side of the multiply (the 0.6 used to hit the stale motion and the deferred S12 then
        // overwrote it, so that hit took full knockback).
        if let Some(velocity) = self.state.local.pending_velocity.take() {
            predictor.state.vel = velocity;
        }
        predictor.state.vel.x *= 0.6;
        predictor.state.vel.z *= 0.6;
        // java: `EntityPlayerSP.setSprinting(false)`: the flag, the modifier and
        // `sprintingTicksLeft`.
        predictor.state.set_sprinting(false);
        self.set_local_watched_sprinting(false);
    }

    /// Whether the local player is being simulated this tick.
    ///
    /// java: the condition guarding the whole body of `EntityPlayerSP.onUpdate`
    /// (`client/entity/EntityPlayerSP.java:170`). `World.isBlockLoaded(pos, true)` reaches
    /// `ChunkProviderClient.chunkExists`, which is a hard-coded `true`, so on a multiplayer
    /// client this is only the horizontal world-validity check. An absent column is represented
    /// by `EmptyChunk`; it does not close this gate.
    ///
    /// A dead player is closed out one level up: `EntityLivingBase.onDeathUpdate` calls
    /// `setDead()` on the tick `deathTime` reaches 20 (that tick still moves and still sends its
    /// C03), and `World.updateEntities` then neither ticks nor keeps an `isDead` entity. From
    /// there until `S07PacketRespawn`/`S01PacketJoinGame` builds a new `EntityPlayerSP`, vanilla
    /// runs no player physics and sends **no** C03/C0B/C13 at all. A client that kept reporting
    /// the corpse every tick for as long as the death screen was open fed the server's
    /// fly/teleport checks a stream vanilla never sends.
    ///
    /// The world tick itself adds one more gate: `updateEntityWithOptionalForce`
    /// runs `onUpdate()` only for an entity that is `addedToChunk`, and a player spawned into an
    /// unsent column is not until that first skipped world tick has run (the session's
    /// `local_added_to_chunk`).
    #[must_use]
    pub fn player_is_simulated(&self) -> bool {
        self.state.local.death_time < LOCAL_PLAYER_REMOVED_DEATH_TIME
            && self.local_added_to_chunk
            && self
                .predictor
                .as_ref()
                .is_some_and(|predictor| player_position_is_valid(predictor.state.pos))
    }

    /// Run one 20 Hz game tick.
    ///
    /// Order is vanilla's: physics for the local player, then the movement packet, then the
    /// remote-entity interpolation step. Returns the packet that went out, so a caller (or a
    /// test) can assert on the cadence.
    ///
    /// Does nothing and returns `None` before `JoinGame` or after a disconnect.
    ///
    /// # The local-player gate ignores column residency
    ///
    /// java: `EntityPlayerSP.onUpdate` (`client/entity/EntityPlayerSP.java:168`) is
    ///
    /// ```text
    /// public void onUpdate() {
    ///     if (this.worldObj.isBlockLoaded(new BlockPos(this.posX, 0.0D, this.posZ))) {
    ///         super.onUpdate();                 // gravity, collision, everything
    ///         ...
    ///         this.onUpdateWalkingPlayer();     // the movement packet
    ///     }
    /// }
    /// ```
    ///
    /// so **both** halves of the player tick are inside one gate. However,
    /// `World.isBlockLoaded(pos)` passes `allowEmpty = true`, and
    /// `ChunkProviderClient.chunkExists` is a hard-coded `true`. The gate therefore reduces to
    /// the horizontal world-validity check; a player placed by `S08PacketPlayerPosLook` keeps
    /// ticking and sends one movement packet per tick before the spawn column arrives. Block
    /// reads use vanilla's blank `EmptyChunk` behavior, including the separate unloaded-column
    /// gravity pin implemented by `mc-physics`.
    ///
    /// Every *other* entity is ticked by `World.updateEntities` regardless of where the local
    /// player is, so the interpolation step below runs either way.
    pub fn tick(&mut self, input: &MoveInput) -> Option<C2SPlay> {
        self.tick_with_held(input, None)
    }

    /// Tick with the session-owned current hotbar stack available to a mounted pig's local AI
    /// shadow. Direct/session tests may keep using [`Self::tick`] because they do not own an
    /// inventory mirror; the live client passes the same stack it already sampled for HUD/FOV.
    pub(crate) fn tick_with_held(
        &mut self,
        input: &MoveInput,
        held: Option<&mc_protocol::ItemStack>,
    ) -> Option<C2SPlay> {
        if !self.phase.is_live() {
            return None;
        }
        mc_core::profile_scope!("session.tick");
        // Before `JoinGame` there is no player at all, which is a different thing from a player
        // whose world has not arrived; keep it a plain early return.
        self.predictor.as_mut()?.state.ground_contact = None;
        // `EntityLivingBase` snapshots its render fields before the local player moves. The
        // current samples are refreshed after physics below, giving frame interpolation the same
        // prev/current pair as remote living entities.
        self.local_render.snapshot_previous();

        // Direct callers do not have the shell's pre-input phase, so perform the update here.
        // In production the shell already did it before hotbar input; do not run it again after
        // that input or the supposedly lazy C09 moves to the wrong side of this tick's C03.
        if !self.controller_updated {
            self.update_controller();
        }
        self.controller_updated = false;

        // java: `GuiDownloadTerrain.updateScreen` (`:38-46`). This is an unsolicited keep-alive,
        // not the echo handled by `net::task`: the no-arg C00 constructor leaves `key == 0`.
        // The loading screen owns this cadence independently of player movement.
        if let Some(ticks) = &mut self.terrain_download_ticks {
            *ticks = ticks.wrapping_add(1);
            if *ticks % 20 == 0 {
                self.send(C2SPlay::KeepAlive(c2s::KeepAlive { key: 0 }));
            }
        }
        // 1.7.10: a terrain screen the first S01 opened in this tick counts from the next one
        // (`session/sends_v5.rs`).
        self.settle_fresh_terrain_screen_v5();

        // Keep the weather sample beside the local entity tick. The fire tail evaluates
        // `isWet()` after movement; remote-world living entities clear inherited lava fire in
        // `EntityLivingBase.onEntityUpdate`, so there is no local lava seed between those tails.
        let raining = self.state.env.is_raining();
        // java: `World.updateEntities` ticks a mounted vehicle before it recursively ticks its
        // rider. `EntityBoat.onUpdate` therefore reads the rider's persistent
        // `moveStrafing`/`moveForward` from the *previous* player update; the rider's
        // `EntityPlayerSP.onLivingUpdate` refreshes those fields only after the boat has already
        // consumed them. Snapshot them before `tick_player` refreshes the current raw keys, then
        // hand that snapshot to the vehicle pass below. Using `MoveInput` directly here makes a
        // client boat respond one tick early compared with Java.
        let rider_action_before_player_tick = self
            .predictor
            .as_ref()
            .map(|predictor| (predictor.state.move_strafing, predictor.state.move_forward))
            .unwrap_or((0.0, 0.0));
        let simulated = self.player_is_simulated();
        let movement_packet = if simulated {
            // java: `EntityLivingBase.updatePotionEffects`, reached from `onEntityUpdate` inside
            // `EntityPlayerSP.onUpdate`, so it shares that method's gates: an unadded (just
            // spawned into an unsent column), dead-and-removed or out-of-world player counts no
            // effect down (this used to run for every tick).
            self.tick_movement_effects();
            // A remote-world EntityPlayer clears its private fire counter in the inherited base
            // tick. The dry flammable tail below is the only local positive-fire path that
            // survives the living-entity remote extinguish.
            self.local_fire = 0;
            // EntityPlayer.onUpdate advances this before `super.onUpdate` runs physics and
            // before EntityPlayerSP sends the walking packet.
            self.local_render.ticks_existed = self.local_render.ticks_existed.saturating_add(1);
            self.tick_local_swing();
            self.tick_sleep_timer();
            let movement_packet = self.tick_player(input);
            self.copy_local_rotation_yaw_head();
            // Entity.moveEntity's fire/flammability tail runs after subclass movement.
            self.advance_local_fire_tail(raining);
            movement_packet
        } else {
            // No `onUpdate` ran, so no movement edge happened this tick. `tick_player` is what
            // clears these per-tick physics signals; without it the app would replay the last
            // simulated tick's footstep, splash or fall grunt every tick (for a dead player, for
            // the whole time the death screen is open).
            if let Some(predictor) = self.predictor.as_mut() {
                predictor.state.predicted_fall_damage = None;
                predictor.state.step_sound = None;
                predictor.state.swim_sound_volume = None;
                predictor.state.water_entry = None;
            }
            None
        };
        // java: `updateEntityWithOptionalForce`'s chunk check after the (possibly skipped)
        // update: `isChunkLoaded(.., true)` is always true on a client, so the player is
        // `addedToChunk` from here on (EmptyChunk.addEntity is a no-op, the flag is not).
        self.local_added_to_chunk = true;

        // `EntityPlayerSP` is the first loaded entity in this client architecture. Queue its
        // base running hook before the tracker walks remote entities, preserving world-list order
        // without manufacturing a network-backed local-player record. A removed (dead) or
        // out-of-world player runs no `Entity.onEntityUpdate`, so it spawns no running particle.
        if simulated {
            self.queue_local_running_particle();
        }

        // A released local horse jump is produced by `EntityPlayerSP` before the remote entity
        // tick. Hand the rider power to the exact tracked horse generation so its
        // `horse.jumpStrength` snapshot can contribute to the client-visible impulse.
        if let Some((network_id, rider_power, rider_forward, rider_yaw)) =
            self.pending_horse_jump.take()
        {
            self.state.entities.apply_local_horse_jump(
                network_id,
                rider_power,
                rider_forward,
                rider_yaw,
            );
        }

        // The local player is session-owned rather than a tracker entry.  XP orbs nevertheless
        // query `World.playerEntities`, so hand its feet/eye geometry and tab-list spectator bit
        // into the entity tick at the same boundary as the remote-player snapshot.
        let local_player_spectator = self
            .state
            .players
            .get(&self.state.local.uuid)
            .is_some_and(|info| info.game_mode == mc_protocol::play::s2c::GameMode::Spectator);
        let local_player_target = self.predictor.as_ref().map(|predictor| {
            let eye_position = if self.state.local.sleeping {
                predictor.state.pos + DVec3::new(0.0, f64::from(0.2_f32), 0.0)
            } else {
                predictor.state.eye_pos()
            };
            LocalPlayerTarget {
                position: predictor.state.pos,
                eye_position,
                spectator: local_player_spectator,
                width: predictor.state.width,
                height: predictor.state.height,
            }
        });
        let dimension = self.state.local.dimension;
        let local_boat_rider = self.riding.vehicle.map(|vehicle_id| LocalBoatRider {
            vehicle_id,
            yaw: input.yaw,
            move_strafing: rider_action_before_player_tick.0,
            move_forward: rider_action_before_player_tick.1,
        });
        let local_pig_rider = self.riding.vehicle.map(|vehicle_id| LocalPigRider {
            vehicle_id,
            yaw: input.yaw,
            move_strafing: rider_action_before_player_tick.0,
            move_forward: rider_action_before_player_tick.1,
            carrot_on_a_stick: held
                .is_some_and(|stack| stack.item_id == interact::CARROT_ON_A_STICK_ITEM_ID),
            carrot_damage: held.map_or(0, |stack| u16::try_from(stack.damage).unwrap_or(0)),
            creative: self.state.local.game_mode == mc_protocol::play::s2c::GameMode::Creative,
        });
        // Open to the end of this fn; the local rider/shadow sync after the walk is O(1).
        mc_core::profile_scope!("tracker.tick");
        // `EntityArrow`'s player-hit rule reads the world
        // scoreboard's teams; only consulted when an arrow's nearest hit is a player.
        // The local player's own box joins the arrow sweep
        // (`EntitySelectors.NOT_SPECTATING` leaves a spectator out).
        let teams = crate::sync::entities::ArrowTeamView {
            scoreboard: &self.state.scoreboard,
            local_entity_id: self.state.local.entity_id,
            local_username: &self.state.local.username,
            local_box: self
                .predictor
                .as_ref()
                .filter(|_| !local_player_spectator)
                .map(|predictor| predictor.state.aabb()),
            local_disable_damage: self.state.local.abilities.invulnerable,
        };
        self.state
            .entities
            .tick_interpolation_with_local_player_in_dimension_and_weather_with_riders_and_teams(
                &mut self.state.world,
                local_player_target,
                &self.state.players,
                dimension,
                raining,
                local_boat_rider,
                local_pig_rider,
                Some(teams),
            );
        // java: `Entity.updateRiderPosition` runs after the vehicle's `onUpdate` and writes the
        // session-owned player's position. Remote riders are handled by the tracker itself;
        // this is the local-player half that cannot live in `EntityTracker`. The pig branch only
        // follows server/S12 motion; the AI shadow above deliberately does not author movement.
        if let Some(vehicle_id) = self.riding.vehicle {
            if let Some(position) = self
                .state
                .entities
                .local_rider_position(&self.state.world, vehicle_id)
            {
                if let Some(predictor) = self.predictor.as_mut() {
                    predictor.state.set_position(position);
                }
            }
        }
        self.sync_rider_eye_v5();
        self.sync_local_entity_packet_shadow();
        self.sync_local_living_render_state();
        movement_packet
    }

    /// Drain local pig AI durability wear for the client inventory owner.
    pub(crate) fn take_pig_carrot_durability_events(&mut self) -> Vec<PigCarrotDurabilityEvent> {
        self.state.entities.take_pig_carrot_durability_events()
    }

    /// Whether a queued pig-AI wear event still belongs to the mounted pig generation that
    /// produced it. Packet dispatch can replace a reused network id before the driver's inventory
    /// drain, so never apply an old generation's wear to a new mount.
    pub(crate) fn pig_carrot_event_is_current(&self, event: PigCarrotDurabilityEvent) -> bool {
        self.riding.vehicle == Some(event.network_id)
            && self
                .state
                .entities
                .get(event.network_id)
                .is_some_and(|tracked| tracked.entity == event.entity)
    }

    /// The body of `EntityPlayerSP.onUpdate`'s guarded branch: physics, then the movement
    /// packet `onUpdateWalkingPlayer` decides on.
    /// Consume one Java `EntityLivingBase.newPosRotationIncrements` step for the local player.
    ///
    /// `S14`/`S18` call the polymorphic setter with `3`, so each local tick advances by the
    /// remaining delta divided by the remaining count. Yaw follows the shortest wrapped arc;
    /// pitch is the straight interpolation Java uses. The local predictor remains the only
    /// position/bounding-box owner, and the state shadow is updated solely for the next packet's
    /// fallback/dead-zone comparison.
    fn apply_local_entity_interpolation(&mut self) {
        let (target_pos, target_yaw, target_pitch, steps, on_ground) = {
            let local = &self.state.local;
            (
                local.target_pos,
                local.target_yaw,
                local.target_pitch,
                local.interpolation_steps,
                local.on_ground,
            )
        };
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        predictor.state.on_ground = on_ground;
        if steps <= 0 {
            return;
        }

        let denominator = f64::from(steps);
        let current = predictor.state.pos;
        let next = current + (target_pos - current) / denominator;
        // java: `EntityLivingBase.onLivingUpdate` (`:1955-1965`). `newRotationYaw/Pitch` are the
        // packet floats widened to double, the step is taken in double with
        // `wrapAngleTo180_double` and narrowed once, and `setRotation` then stores each angle
        // `% 360.0F` (L4/L5: the f32 step without the remainder could differ in the last bit).
        let yaw_step =
            mc_math::wrap_degrees_f64(f64::from(target_yaw) - f64::from(predictor.state.yaw));
        let yaw = (f64::from(predictor.state.yaw) + yaw_step / denominator) as f32 % 360.0;
        let pitch = (f64::from(predictor.state.pitch)
            + (f64::from(target_pitch) - f64::from(predictor.state.pitch)) / denominator)
            as f32
            % 360.0;
        // Java's `setPosition` rebuilds the bounding box while retaining the entity's previous
        // render sample; PhysicsState::set_position is the equivalent sole-owner operation.
        predictor.state.set_position(next);
        predictor.state.yaw = yaw;
        predictor.state.pitch = pitch;

        let local = &mut self.state.local;
        local.interpolation_steps -= 1;
        local.packet_position = next;
        local.packet_yaw = yaw;
        local.packet_pitch = pitch;
    }

    fn tick_player(&mut self, input: &MoveInput) -> Option<C2SPlay> {
        if self.version.is_1_7_10() {
            return self.tick_player_v5(input);
        }
        mc_core::profile_scope!("physics.tick_player");
        // `EntityLivingBase.onEntityUpdate` evaluates the border hazard before its
        // `onLivingUpdate` interpolation/movement body.  Sample the same projected border once
        // for both the damage probe and this tick's collision sweep; using two wall-clock reads
        // could make a fast lerp cross an edge between those adjacent Java operations.
        let border_state = self
            .state
            .env
            .border
            .dimension_view(self.state.local.dimension);
        // The wall clock live; the recording's clock in a replay.
        let border = border_state.collision_border(self.state.env.border_now_millis());
        self.apply_local_border_damage(border_state, border.diameter);

        // A destroy/reuse boundary can invalidate the retained object between packet handling
        // and this tick. Java's updateRidden observes the dead/mismatched object and clears
        // ridingEntity before the movement-packet fork; do the same before choosing C05+C0C.
        if let Some(vehicle_id) = self.riding.vehicle {
            if self.resolve_vehicle(vehicle_id) != self.riding.vehicle_entity {
                self.clear_riding();
            }
        }
        // `EntityLivingBase.onLivingUpdate` consumes S14/S18's three-tick target before local
        // physics runs. Keep this interpolation on the predictor's sole position owner; the
        // packet shadow only supplies the target and remaining-step count.
        //
        // java: that step turns `rotationYaw/rotationPitch` from this tick's look (the frame
        // loop's `setAngles` result, which the input carries) and everything after it in the
        // tick reads the turned value: the sprint-jump, `moveFlying`, the C05/C06. Physics copies
        // `input.yaw/pitch` into its state at the top of the tick, so it gets the stepped
        // rotation; feeding it the pre-step look discarded the step.
        // On the live path the input is the predictor's own rotation, so without a pending step
        // both assignments are no-ops.
        if let Some(predictor) = self.predictor.as_mut() {
            predictor.state.yaw = input.yaw;
            predictor.state.pitch = input.pitch;
        }
        self.apply_local_entity_interpolation();
        let input = &self
            .predictor
            .as_ref()
            .map_or(*input, |predictor| MoveInput {
                yaw: predictor.state.yaw,
                pitch: predictor.state.pitch,
                ..*input
            });
        self.push_server_movement_state();
        // Java still simulates EntityPlayerSP while another entity is the camera, but its
        // `onUpdateWalkingPlayer` movement-report tail is guarded by `isCurrentViewEntity()`.
        // Riding remains an earlier, unconditional branch and is handled below unchanged.
        let reports_movement = self.local_is_current_view_entity();
        // **W-RIDE.** java: `EntityPlayerSP.onUpdate` (`:174`) forks the *whole* tail on
        // `isRiding()`, and `onUpdateWalkingPlayer` is not reached at all while mounted.
        if self.riding.vehicle.is_some() {
            return self.tick_riding(input, reports_movement, border);
        }
        // `EntityPlayerSP.onLivingUpdate` sends C13 inline at each capability mutation.  Keep
        // the mutation that happens before `super.onLivingUpdate()` separate from the landing
        // mutation at its tail: looking only at the post-tick value loses an on -> off pair when
        // both sites run during this tick.  The predictor owns the actual state machine; this
        // small preflight mirrors only its two flight-toggle predicates so Session can preserve
        // the wire events without changing the frozen physics API.
        let (flying_before, flight_toggle_edge) = {
            let state = self.predictor.as_ref()?.state;
            (state.flying, flight_toggle_target(&state, input))
        };
        let watched_sprinting = self.local_watched_sprinting();
        let sleeping = self.state.local.sleeping;
        // java: `World.getCollidingBoundingBoxes` resolves the nearby loaded-entity list at the
        // same start-of-tick position that `EntityPlayerSP.onUpdate` uses for its movement.  Keep
        // the snapshot outside the predictor borrow: tracker order and ECS positions are the
        // caller-owned input to this tick, not predictor state.
        let entity_collisions = self.state.entities.local_sweep_collision_candidates(
            &self.state.world,
            self.state.local.entity_id,
            &self.state.players,
        );
        // java: World.getCollidingBoundingBoxes reads WorldBorder.getDiameter() at the query
        // instant. Keep this clock sample at the same boundary as the local movement tick so a
        // lerping S44 projection cannot drift between the caller and physics.
        let (movement, sprinting, sneaking, flying) = {
            let predictor = self.predictor.as_mut()?;
            // The local DataWatcher is the state `onUpdateWalkingPlayer` reports.  Feed it back
            // into physics before the tick so a server metadata update and a local attack use the
            // same source for both simulation and the next C0B edge.
            //
            // **The flag only**. An S1C echo is `DataWatcher.updateObject`, i.e.
            // `Entity.setFlag(3, ..)`: it does not call `setSprinting`, so the sprint attribute
            // entry and `sprintingTicksLeft` keep their values. Re-running a `setSprinting` here
            // on every flag change gave an echoed `true` the 1.3x boost vanilla does not have
            // (faster than vanilla) and stripped the boost from an echoed `false`.
            predictor.state.sprinting = watched_sprinting;
            let movement = if reports_movement {
                Some(predictor.tick_with_entities_and_border(
                    &self.state.world,
                    input,
                    &entity_collisions,
                    border,
                ))
            } else {
                predictor.tick_with_entities_and_border_unreported_view(
                    &self.state.world,
                    input,
                    &entity_collisions,
                    border,
                    reports_movement,
                );
                None
            };
            // `EntityLivingBase.setSprinting`'s attribute-entry edit happens inside physics
            // (`PhysicsState::set_sprinting`), at the same statement as the flag write.
            let sprinting = predictor.state.sprinting;
            // EntityPlayerSP.isSneaking is the movement-input flag *and* `!sleeping`. Physics
            // keeps the raw key for its previous-input edges, but the C0B state machine sees this
            // getter.
            let sneaking = predictor.state.sneaking && !sleeping;
            let flying = predictor.state.flying;
            (movement, sprinting, sneaking, flying)
        };
        // Keep the watched index canonical before handing the edge to OutboundState.  The
        // resulting STOP_SPRINTING is emitted on the next tick after an attack's C02.
        self.set_local_watched_sprinting(sprinting);
        let watched_sprinting = self.local_watched_sprinting();
        let entity_id = self.state.local.entity_id;

        // Flight edges and sendPlayerAbilities run inside super.onUpdate/onLivingUpdate. Only
        // after that returns does onUpdateWalkingPlayer emit C0B and C03..C06, so every C13
        // leads the walking burst.  Java sends unconditionally at each mutation site, hence the
        // explicit intermediate state for a same-tick toggle followed by landing clear.
        let flying_after_toggle = flight_toggle_edge.unwrap_or(flying_before);
        let landing_edge = self.predictor.as_ref().is_some_and(|predictor| {
            flying_after_toggle
                && !predictor.state.spectator_mode
                && predictor.state.on_ground
                && !flying
        });
        if let Some(target) = flight_toggle_edge {
            self.send_flight_ability_edge(target);
        }
        if landing_edge {
            self.send_flight_ability_edge(false);
        }
        // Keep the old defensive fallback for callers that mutate Predictor directly.  Under
        // the normal tick path the two mutation-site checks above cover every physics edge.
        if flying != self.state.local.abilities.flying {
            self.send_flight_ability_edge(flying);
        }

        let mut packets = Vec::with_capacity(3);
        self.outbound
            .walking_edges(entity_id, watched_sprinting, sneaking, &mut packets);
        if reports_movement {
            self.outbound.movement_only(
                movement.expect("view-local tick always produces a movement report"),
                &mut packets,
            );
        }
        let movement_packet = reports_movement.then(|| packets.last().cloned()).flatten();
        self.send_all(packets);

        // java: `EntityPlayerSP.onLivingUpdate` calls `sendPlayerAbilities()` on both edges of
        // the flight flag — the double-tap toggle (`:841`) and the "cleared on landing" one
        // (`:906`). `mc_physics::tick_player` owns both edges; the packet is ours.
        movement_packet
    }

    /// Apply the local client's prediction of `EntityLivingBase`'s world-border hazard.
    ///
    /// Vanilla evaluates the amount even on a remote world, but `EntityPlayerSP.attackEntityFrom`
    /// ultimately returns `false` through `EntityLivingBase`'s `isRemote` guard.  The multiplayer
    /// client therefore does not author a server-health packet; the live Rust session keeps the
    /// useful local health/hurt/death prediction here and lets the next authoritative S06 snapshot
    /// replace or correct it through `LocalPlayer::apply_server_health`.  No outbound packet is
    /// emitted, so a server-side border hit cannot be counted twice when that S06 arrives.
    ///
    /// The Java attack path's hurt-resistance window is retained: repeated one-point border
    /// attacks are ignored while `hurtResistantTime > maxHurtResistantTime / 2`, and only the
    /// excess over `lastDamage` is applied on a stronger hit.  The local state has no separate
    /// `maxHurtResistantTime` field; vanilla's `EntityLivingBase` default is 20.
    fn apply_local_border_damage(&mut self, border: BorderState, diameter: f64) {
        let Some((position, aabb)) = self
            .predictor
            .as_ref()
            .map(|predictor| (predictor.state.pos, predictor.state.aabb()))
        else {
            return;
        };

        // Java first checks `!WorldBorder.contains(entityBB)`, then uses the entity centre for
        // `getClosestDistance(Entity)`. A partially protruding box inside the five-block grace
        // region therefore receives no damage yet; do not collapse the two tests into one point
        // check.
        if border.contains_aabb(aabb, diameter) {
            return;
        }
        let Some(damage) = border.damage_for_point(position.x, position.z, diameter) else {
            return;
        };

        let local = &mut self.state.local;
        // Before the first S06, EntityPlayerSP has only its constructor default. Waiting for a
        // valid health snapshot avoids inventing a client hit during the join/respawn window.
        if !local.has_valid_health || local.health <= 0.0 {
            return;
        }

        let amount = damage as f32;
        if local.hurt_resistant_time > 10 {
            let previous_damage = local.last_damage;
            if amount <= previous_damage {
                return;
            }
            local.last_damage = amount;
            set_local_health(local, local.health - (amount - previous_damage));
        } else {
            local.last_damage = amount;
            local.hurt_resistant_time = 20;
            set_local_health(local, local.health - amount);
            local.hurt_time = local.max_hurt_time;
        }
    }

    /// Emit one Java `sendPlayerAbilities()` result and make the capability state match it.
    ///
    /// This is intentionally unconditional: vanilla calls the method at each mutation site,
    /// including two sites that can produce opposite values in one tick.
    fn send_flight_ability_edge(&mut self, flying: bool) {
        self.state.local.abilities.flying = flying;
        let abilities = self.state.local.abilities;
        self.send(outbound::player_abilities(&abilities));
    }

    /// The body of `EntityPlayerSP.onUpdate`'s **riding** branch.
    ///
    /// **W-RIDE.** java: `World.updateEntityWithOptionalForce` (`:1873`) calls `updateRidden()`
    /// *instead of* `onUpdate()` once `ridingEntity != null`, and `Entity.updateRidden` (`:1880`)
    /// zeroes all three motion components and then calls `onUpdate()` itself — so `onUpdate`
    /// still runs, but over a player whose motion is `0` and whose position is overwritten by
    /// `ridingEntity.updateRiderPosition()` immediately afterwards.
    ///
    /// Two consequences that are easy to get backwards:
    ///
    /// * **The walking tick must not run, but the rider's own travel must.** Its position would
    ///   be discarded by `updateRiderPosition` in vanilla; here it would be the only thing writing
    ///   the position, so keeping it would make a mounted player fall through their own boat.
    ///   [`mc_physics::player::mounted_living_travel`] runs the travel for everything else it
    ///   leaves behind (collision flags, `onGround`, `jumpTicks`, motion for the camera pitch) and
    ///   then puts the rider back on the seat it started from; the vehicle's tick writes the new
    ///   seat. The *wire* is unaffected: a riding client sends no position at all, and the `S08`
    ///   that follows any dismount re-seats it.
    /// * **`EntityPlayer.updateRidden`'s sneak-dismount is `!worldObj.isRemote`-gated**
    ///   (`entity/player/EntityPlayer.java:546-549`), so a client never dismounts *itself*. The
    ///   sneak key reaches the server as bit `1` of `C0CPacketInput` and comes back as an `S1B`
    ///   with vehicle id `-1`. Dismounting locally here would desync every ride.
    ///
    /// Returns the `C05PacketPlayerLook`, matching [`Session::tick_player`]'s contract that the
    /// return value is the movement packet this tick put on the wire.
    fn tick_riding(
        &mut self,
        input: &MoveInput,
        view_is_local: bool,
        border: mc_physics::collision::CollisionBorder,
    ) -> Option<C2SPlay> {
        // Flight toggles happen in `onLivingUpdate` before the riding packet pair. Capture the
        // predicate before borrowing the mutable predictor so each C13 edge can be emitted in
        // vanilla order, including a same-tick toggle followed by landing clear.
        let flight_toggle_edge = {
            let state = self.predictor.as_ref()?.state;
            flight_toggle_target(&state, input)
        };
        let watched_sprinting = self.local_watched_sprinting();
        // The rider's own travel sweeps the same loaded-entity list a walking tick does (the
        // vehicle's boat box included: Java's rider/vehicle exclusion compares an Entity with a
        // List and never fires). Sampled once, at the start-of-tick position, as on foot.
        let entity_collisions = self.state.entities.local_sweep_collision_candidates(
            &self.state.world,
            self.state.local.entity_id,
            &self.state.players,
        );
        let world = &self.state.world;
        let predictor = self.predictor.as_mut()?;
        // Feed the local DataWatcher sprint flag back into the state before running the same
        // SP preflight as an unmounted tick. OutboundState's server edge cache is intentionally
        // left untouched because Java freezes `serverSprintState` while riding. As on foot, an
        // S1C echo is a flag write only: the attribute entry and `sprintingTicksLeft` stay.
        predictor.state.sprinting = watched_sprinting;
        // java: the horse-jump block's `boolean flag = this.movementInput.jump`, read before
        // `updatePlayerMoveState()`: last tick's jump key whether or not that tick was mounted.
        let horse_jump_was_held = predictor.state.prev_jump;
        // java: `Entity.updateRidden` (`:1880-1891`) zeroes the motion and runs the full local
        // `onUpdate()`: the preflight is its `onEntityUpdate` part and the `EntityPlayerSP`
        // prologue (sprint, flight toggle and thrust).
        mounted_on_living_update_preflight(world, &mut predictor.state, input, view_is_local);
        let flying_before_travel = predictor.state.flying;
        // java: `EntityLivingBase.onLivingUpdate`'s jump and travel sections still run for a rider
        // (`:2005-2034`): `jumpTicks` resets while jump is up, `moveStrafing/moveForward *= 0.98F`
        // damps the fields the C0C then reads (a vanilla C0C never carries more than 0.98), and the rider's own move sets the `onGround` C05 reports and the
        // `isCollidedHorizontally` the next tick's sprint stop reads (H2). Its tail is the SP
        // landing clear, before `onUpdate` emits the riding pair.
        predictor.mounted_living_travel_with_entities_and_border(world, &entity_collisions, border);
        let landing_edge = flying_before_travel && !predictor.state.flying;
        // java: `EntityLivingBase.updateRidden` (`:2103-2108`), after `updateRiderPosition`.
        predictor.state.fall_distance = 0.0;
        // java: `EntityPlayer.updateRidden` snapshots the horizontal sway the travel's tail
        // produced and then clears only `cameraYaw`; `cameraPitch` keeps its damped value.
        predictor.state.prev_camera_yaw = predictor.state.camera_yaw;
        predictor.state.camera_yaw = 0.0;
        let on_ground = predictor.state.on_ground;
        let sprinting = predictor.state.sprinting;
        let move_strafing = predictor.state.move_strafing;
        let move_forward = predictor.state.move_forward;
        let yaw = predictor.state.yaw;
        let pitch = predictor.state.pitch;
        let flying = predictor.state.flying;
        let _ = predictor;
        self.set_local_watched_sprinting(sprinting);
        if let Some(target) = flight_toggle_edge {
            self.send_flight_ability_edge(target);
        }
        let entity_id = self.state.local.entity_id;
        // java: `onLivingUpdate`'s horse-jump block (`:861-896`) sits between the flight toggle
        // and `super.onLivingUpdate()`, so its `C0B` follows the toggle's C13 and precedes the
        // landing C13 (`:904-908`) as well as `onUpdate`'s riding pair.
        if let Some(jump) = self.charge_horse_jump(horse_jump_was_held, input.jump) {
            self.pending_horse_jump = self
                .riding
                .vehicle
                .map(|vehicle| (vehicle, jump, input.move_forward_riding(true), input.yaw));
            // java: `sendHorseJump` sends `(int)(horseJumpPower * 100.0F)` as auxData.
            self.send(C2SPlay::EntityAction(c2s::EntityAction::riding_jump(
                entity_id,
                (jump * 100.0) as i32,
            )));
        }
        if landing_edge {
            self.send_flight_ability_edge(false);
        }
        if flying != self.state.local.abilities.flying {
            self.send_flight_ability_edge(flying);
        }

        let look = C2SPlay::PlayerLook(c2s::PlayerLook {
            yaw,
            pitch,
            on_ground,
        });
        self.send(look.clone());
        // C0C reads `EntityLivingBase.moveStrafing/moveForward`, not the raw movement input.
        // That distinction is observable with a foreign camera: updateEntityActionState leaves
        // those fields untouched while the raw jump/sneak bits still update.
        self.send(outbound::steer_vehicle(
            move_strafing,
            move_forward,
            input.jump,
            input.sneak,
        ));
        Some(look)
    }

    /// One tick of `EntityPlayerSP.onLivingUpdate`'s horse jump-charge state machine
    /// (`:861-896`). Returns the power to send when this tick is the *release*.
    ///
    /// The predicate is `EntityPlayerSP.isRidingHorse()` (`:572`), which is **not**
    /// `PlayerControllerMP.isRidingHorse()` (`:617`): this one additionally requires
    /// `((EntityHorse)ridingEntity).isHorseSaddled()`, so a jump charged on an unsaddled horse
    /// (a donkey being led, say) never accumulates and never sends. The inventory key uses the
    /// controller's saddle-free version — the two are genuinely different predicates and the
    /// names hide it.
    ///
    /// `was_jumping` is Java's `flag`, `movementInput.jump` before this tick's
    /// `updatePlayerMoveState()`: the previous tick's key even when that tick was a walking one
    /// (H2: a separate copy refreshed only on ride ticks went stale across a mount).
    fn charge_horse_jump(&mut self, was_jumping: bool, jump: bool) -> Option<f32> {
        if !self.is_riding_saddled_horse() {
            // java: the `else` at `:894` — `this.horseJumpPower = 0.0F;`.
            self.riding.jump_power = 0.0;
            return None;
        }
        if self.riding.jump_power_counter < 0 {
            self.riding.jump_power_counter += 1;
            if self.riding.jump_power_counter == 0 {
                self.riding.jump_power = 0.0;
            }
        }
        if was_jumping && !jump {
            self.riding.jump_power_counter = -10;
            return Some(self.riding.jump_power);
        }
        if !was_jumping && jump {
            self.riding.jump_power_counter = 0;
            self.riding.jump_power = 0.0;
        } else if was_jumping {
            self.riding.jump_power_counter += 1;
            self.riding.jump_power = if self.riding.jump_power_counter < 10 {
                self.riding.jump_power_counter as f32 * 0.1
            } else {
                // Peaks at exactly `1.0` on tick 10 and decays back toward `0.8` after it.
                0.8 + 2.0 / (self.riding.jump_power_counter - 9) as f32 * 0.1
            };
        }
        None
    }

    /// java: `EntityPlayerSP.isRidingHorse` (`:572`) — mounted, on an `EntityHorse`, **and**
    /// `isHorseSaddled()` (watched index 16 `& 4`).
    #[must_use]
    fn is_riding_saddled_horse(&self) -> bool {
        let Some(vehicle) = self.riding.vehicle else {
            return false;
        };
        self.state
            .entities
            .with_watched(&self.state.world, vehicle, |watched| {
                watched.horse().is_some_and(|horse| horse.saddled)
            })
            .unwrap_or(false)
    }

    /// The vehicle the local player is riding, as a network id.
    /// java: `EntityPlayerSP.ridingEntity`. W-RIDE.
    #[must_use]
    pub fn riding(&self) -> Option<i32> {
        self.riding.vehicle
    }

    /// Apply the client half of `ItemCarrotOnAStick.onItemRightClick` to one exact mounted pig
    /// generation. The caller has already established the local player's mount relation; the
    /// tracker rechecks the pig type, saddle watcher and ECS generation before drawing Java's
    /// `rand.nextInt(841) + 140` boost duration.
    pub(crate) fn start_pig_boost(
        &mut self,
        network_id: i32,
        generation: mc_world::Entity,
    ) -> bool {
        self.riding.vehicle == Some(network_id)
            && self
                .state
                .entities
                .boost_pig(&self.state.world, network_id, generation)
    }

    /// Retain the generation-safe client half of `EntityHorse.mountTo` after C02 has been
    /// queued. Only the matching local-player S1B is allowed to apply the pose/seat snap.
    pub(crate) fn note_horse_mount_intent(
        &mut self,
        network_id: i32,
        generation: mc_world::Entity,
    ) {
        self.pending_horse_mount = self
            .state
            .entities
            .get(network_id)
            .filter(|tracked| {
                tracked.entity == generation
                    && tracked.kind.entity_type() == Some(mc_data::EntityType::EntityHorse)
            })
            .map(|_| (network_id, generation));
        // java: `EntityHorse.mountTo` runs on the client at the click,
        // and only its `player.mountEntity(this)` is `!isRemote`: `player.rotationYaw =
        // this.rotationYaw; player.rotationPitch = this.rotationPitch;` turn the player at once,
        // so this tick's look packet carries the horse's rotation.
        let rotation = self
            .pending_horse_mount
            .and_then(|(id, generation)| self.horse_rotation(id, generation));
        if let (Some((yaw, pitch)), Some(predictor)) = (rotation, self.predictor.as_mut()) {
            predictor.state.yaw = yaw;
            predictor.state.pitch = pitch;
        }
    }

    /// The tracked horse's `rotationYaw`/`rotationPitch`, through its world generation.
    fn horse_rotation(&self, network_id: i32, generation: mc_world::Entity) -> Option<(f32, f32)> {
        let tracked = self.state.entities.get(network_id)?;
        if tracked.entity != generation {
            return None;
        }
        Some(
            self.state
                .world
                .ecs
                .get::<&mc_world::Rotation>(generation)
                .map(|rotation| (rotation.yaw, rotation.pitch))
                .unwrap_or((tracked.target_yaw, tracked.target_pitch)),
        )
    }

    /// Resolve a local S1B vehicle through the current world generation.
    ///
    /// `EntityTracker::get` is the client world's id index, while the `Position` lookup confirms
    /// that its ECS handle still belongs to this world. This is the local equivalent of Java's
    /// `World.getEntityByID`; an absent/stale handle is `null`, not a numeric riding id.
    fn resolve_vehicle(&self, network_id: i32) -> Option<mc_world::Entity> {
        let tracked = self.state.entities.get(network_id)?;
        self.state
            .world
            .ecs
            .get::<&mc_world::Position>(tracked.entity)
            .ok()
            .map(|_| tracked.entity)
    }

    /// Clear the local `ridingEntity` edge and all client-only mount state.
    fn clear_riding(&mut self) {
        self.riding.vehicle = None;
        self.riding.vehicle_entity = None;
        self.state.entities.set_local_boat_rider(None);
        self.pending_horse_jump = None;
        self.pending_horse_mount = None;
        // java: once `isRidingHorse()` is false, `onLivingUpdate`'s `else` zeroes
        // `horseJumpPower` every tick. `horseJumpPowerCounter` is only ever written inside the
        // horse branch, so it survives the dismount (a release's `-10` keeps counting up on the
        // next horse); H2.
        self.riding.jump_power = 0.0;
    }

    /// Consume a pending click and snapshot Java's ordinary (non-rearing) horse seat.
    ///
    /// The rotation is not part of it: the click already turned the player
    /// (`note_horse_mount_intent`), and the S1B's `Entity.mountEntity` changes no rotation.
    fn confirm_horse_mount(&mut self, network_id: i32) -> Option<DVec3> {
        let (pending_id, generation) = self.pending_horse_mount.take()?;
        if pending_id != network_id {
            return None;
        }
        let tracked = self.state.entities.get(network_id)?;
        if tracked.entity != generation
            || tracked.kind.entity_type() != Some(mc_data::EntityType::EntityHorse)
        {
            return None;
        }
        let horse_position = tracked.render_position(&self.state.world, 1.0);
        let seat_y = horse_position.y + f64::from(tracked.dimensions().1) * 0.75 - 0.35;
        Some(DVec3::new(horse_position.x, seat_y, horse_position.z))
    }

    /// The live value consumed by `GuiIngame.renderHorseJumpBar`, when the local player is
    /// mounted on a saddled horse. `None` selects the ordinary XP bar.
    // java: EntityPlayerSP.getHorseJumpPower / isRidingHorse
    #[must_use]
    pub fn horse_jump_power(&self) -> Option<f32> {
        self.is_riding_saddled_horse()
            .then_some(self.riding.jump_power)
    }

    /// java: `PlayerControllerMP.isRidingHorse` (`:617`) — mounted on an `EntityHorse`, with
    /// **no** saddle test. This is the one the inventory key forks on
    /// (`Minecraft.java:2094`); [`Session::charge_horse_jump`] documents why the two differ.
    /// The mounted horse's resolved `horse.jumpStrength` value, including Java's constructor
    /// default of `0.7` before the first S20 snapshot. This is the multiplier consumed by the
    /// tracked horse's client-side jump impulse, not the rider's charge shown in the HUD.
    #[must_use]
    pub fn horse_jump_strength(&self) -> Option<f32> {
        let vehicle = self.riding.vehicle?;
        self.state
            .entities
            .get(vehicle)
            .and_then(|tracked| tracked.horse_jump_strength())
    }

    /// Whether the local player is mounted on a horse, regardless of saddle state.
    #[must_use]
    pub fn is_riding_horse(&self) -> bool {
        self.riding.vehicle.is_some_and(|vehicle| {
            self.state
                .entities
                .get(vehicle)
                .and_then(|tracked| tracked.kind.entity_type())
                == Some(mc_data::EntityType::EntityHorse)
        })
    }

    /// Ask the server to open the ridden horse's inventory.
    /// java: `EntityPlayerSP.sendHorseInventory` (`:412`). W-RIDE.
    pub fn send_horse_inventory(&mut self) {
        let entity_id = self.state.local.entity_id;
        self.send(C2SPlay::EntityAction(c2s::EntityAction::open_inventory(
            entity_id,
        )));
    }

    /// Ask the server to wake the local player from a bed.
    /// java: `GuiSleepMP.wakeFromSleep` (`:63-67`). The sleep screen owns the user action; this
    /// method keeps packet construction on Session's live wire seam, like the riding actions.
    pub fn send_stop_sleeping(&mut self) {
        let entity_id = self.state.local.entity_id;
        self.send(C2SPlay::EntityAction(c2s::EntityAction::stop_sleeping(
            entity_id,
        )));
    }

    /// Pushes everything the server has told us about the player into the predictor, immediately
    /// before the tick reads it.
    ///
    /// This is where the whole of area A11's verdict lands: the arithmetic was already right, and
    /// every one of these values was decoded, stored, and never handed to `tick_player`.
    ///
    /// **`capabilities.walkSpeed` is deliberately not among them.** `EntityPlayer.onLivingUpdate`
    /// (`entity/player/EntityPlayer.java:620-625`) pushes it into the `generic.movementSpeed`
    /// attribute inside `if (!this.worldObj.isRemote)`, so on a *client* a server-set walk speed
    /// never reaches `getAIMoveSpeed()`; the base stays the `(double) 0.1F` that
    /// `applyEntityAttributes` registered until `S20PacketEntityProperties` changes it. The
    /// `speed_attribute` differential fixture drives a real `EntityPlayer` with `walkSpeed = 0.3`
    /// and shows the trace does not move.
    pub(crate) fn push_server_movement_state(&mut self) {
        let Session {
            state,
            predictor,
            effects,
            ..
        } = self;
        let Some(predictor) = predictor.as_mut() else {
            return;
        };
        let s = &mut predictor.state;
        // AbstractClientPlayer.isSpectator resolves this player's UUID through
        // NetHandlerPlayClient.playerInfoMap. It is deliberately false when S38 has no local
        // row, even if PlayerControllerMP's current game type is Spectator.
        let local_is_spectator = state
            .players
            .get(&state.local.uuid)
            .is_some_and(|info| info.game_mode == mc_protocol::play::s2c::GameMode::Spectator);
        let local = &mut state.local;

        // java: `handlePlayerAbilities` → `thePlayer.capabilities`.
        s.flying = local.abilities.flying;
        s.allow_flying = local.abilities.allow_flying;
        s.fly_speed = local.abilities.fly_speed;
        // java: `EntityPlayer.onUpdate:268` — `noClip = isSpectator()`; EntityPlayerSP inherits
        // AbstractClientPlayer's tab-list lookup rather than consulting PlayerControllerMP.
        s.spectator = local_is_spectator;
        // java: `EntityPlayerSP.onLivingUpdate:825`/`:904` ask the controller instead:
        // `mc.playerController.isSpectatorMode()`, i.e. the S01/S07/S2B game type.
        s.spectator_mode = local.game_mode == mc_protocol::play::s2c::GameMode::Spectator;
        // java: `EntityPlayer.isMovementBlocked()` — `getHealth() <= 0.0F || isPlayerSleeping()`.
        // This zeroes movement input and jump, but deliberately does not suppress the ordinary
        // onUpdateWalkingPlayer packet cadence.
        s.movement_blocked = local.health <= 0.0 || local.sleeping;
        // java: `FoodStats.foodLevel`, from `S06PacketUpdateHealth`. `tick_player`'s sprint gate
        // is `> 6.0F || capabilities.allowFlying`.
        s.food_level = local.food_level;
        s.movement_speed = effects
            .movement_speed
            .unwrap_or(mc_physics::state::WALK_SPEED);
        // `handleEntityProperties` mutates the Java attribute map at packet-dispatch time and
        // the local `setSprinting` edge may mutate that same map later. Do not replay the last
        // S20 snapshot here: doing so erases a local sprint modifier on every ordinary tick.
        // java: the packet's signed byte remains signed through the local physics state. Do not
        // convert it through u8 (or clamp negatives): `jump()` and `fall()` both consume the
        // Java amplifier as `(amplifier + 1)`.
        s.jump_boost = effects.jump_boost.map(|(amplifier, _)| amplifier);
        s.blindness = effects.blindness.is_some();
        s.depth_strider = effects.depth_strider;

        // java: `handleEntityVelocity` (`:501`) → `Entity.setVelocity`, which
        // **assigns** rather than adds, and does it for whatever entity the id names — the local
        // player very much included, because `EntityLivingBase.knockBack` is server-side only.
        // `EntityTracker` never holds the local player, so `sync::dispatch` hands it over in
        // `pending_velocity` and `react_to` assigns it at that same packet. Nothing is left here
        // on the live path; this only serves harnesses that call `sync::apply` directly.
        if let Some(velocity) = local.pending_velocity.take() {
            s.vel = velocity;
        }
    }

    /// java: `PotionEffect.onUpdate` and `EntityLivingBase.updatePotionEffects`.  One tick is
    /// removed while a duration is positive, but an expired effect is retained at zero on a
    /// remote world.  Only the server mutates `activePotionsMap` on completion; this client
    /// waits for `S1EPacketRemoveEntityEffect` instead.
    fn tick_movement_effects(&mut self) {
        if let Some((amplifier, duration)) = self.effects.jump_boost {
            let duration = if duration > 0 { duration - 1 } else { duration };
            self.effects.jump_boost = Some((amplifier, duration));
        }
        if let Some((_, duration)) = &mut self.effects.blindness {
            if *duration > 0 {
                *duration -= 1;
            }
        }
    }

    /// `Entity.setSize` updates the box from its old minimum corner without changing `pos`.
    /// Most callers immediately follow it with `setPosition`, but the no-longer-a-bed wake edge
    /// does not; retaining that temporary box/position mismatch is observable on the next sweep.
    fn set_local_player_size(&mut self, width: f32, height: f32) {
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        let state = &mut predictor.state;
        if state.width == width && state.height == height {
            return;
        }
        let old_pos = state.pos;
        let old_box = state.aabb();
        state.width = width;
        state.height = height;
        let resized = Aabb::from_bounds(
            old_box.min.x,
            old_box.min.y,
            old_box.min.z,
            old_box.min.x + f64::from(width),
            old_box.min.y + f64::from(height),
            old_box.min.z + f64::from(width),
        );
        state.set_from_aabb(&resized);
        state.pos = old_pos;
    }

    /// Apply `EntityPlayer.onDeath`'s client-visible bounds and motion mutation.
    ///
    /// Status opcode 3 supplies `DamageSource.generic`, so the nullable-cause zero-motion arm is
    /// unreachable here. `attackedAtYaw` is zeroed by the generic client hurt status and has no
    /// other network producer; the observable angle is therefore the current player yaw.
    fn apply_local_player_death_pose(&mut self) {
        // An earlier S12 happened before onDeath and is overwritten by this motion assignment.
        self.state.local.pending_velocity = None;
        self.set_local_player_size(0.2, 0.2);
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        let state = &mut predictor.state;
        // `setSize` anchors the temporary box at its old minimum corner; onDeath immediately
        // follows it with `setPosition(posX, posY, posZ)`, rebuilding the box around the same pos.
        state.set_position(state.pos);
        let angle = state.yaw * core::f32::consts::PI / 180.0;
        state.vel = DVec3::new(
            f64::from(-math_helper::cos(angle) * 0.1),
            0.10000000149011612,
            f64::from(-math_helper::sin(angle) * 0.1),
        );
        self.death_pose_v5();
    }

    /// The client-side half of `EntityPlayer.trySleep` after the server has accepted the bed.
    fn enter_bed(&mut self, bed: BlockPos) {
        // trySleep dismounts first (`mountEntity(null)`). On a remote world this produces no
        // outbound action packet, and like any dismount it keeps the horse-jump counter.
        self.clear_riding();
        self.set_local_player_size(0.2, 0.2);

        let loaded = self.state.world.is_chunk_loaded(bed.chunk_pos());
        let (x_offset, z_offset) = if loaded {
            match bed_facing(&self.state.world, bed) {
                Facing::South => (0.5, 0.9),
                Facing::North => (0.5, 0.1),
                Facing::West => (0.1, 0.5),
                Facing::East => (0.9, 0.5),
                Facing::Down | Facing::Up => unreachable!("bed metadata is horizontal"),
            }
        } else {
            (0.5, 0.5)
        };
        // Java adds in f32, then widens the result to double. This matters at large coords.
        let position = DVec3::new(
            f64::from(bed.x as f32 + x_offset),
            f64::from(bed.y as f32 + 0.6875),
            f64::from(bed.z as f32 + z_offset),
        );
        if let Some(predictor) = self.predictor.as_mut() {
            predictor.state.set_position(position);
            predictor.state.vel = DVec3::ZERO;
        }
        self.enter_bed_v5(bed);
        self.state.local.sleeping = true;
        self.state.local.sleep_timer = 0;
        self.state.local.sleeping_at = Some(bed);
        self.sync_local_living_render_state();
    }

    /// `wakeUpPlayer(false, false, false)`, as animation type 2 invokes it on the client.
    fn wake_from_bed(&mut self) {
        self.set_local_player_size(PLAYER_WIDTH, PLAYER_HEIGHT);
        if let Some(bed) = self.state.local.sleeping_at {
            let bed_block = self.state.world.block_at(bed);
            if bed_block.block().0 == 26 {
                let exit = safe_bed_exit(&self.state.world, bed).unwrap_or_else(|| bed.up());
                let position = DVec3::new(
                    f64::from(exit.x as f32 + 0.5),
                    f64::from(exit.y as f32 + 0.1),
                    f64::from(exit.z as f32 + 0.5),
                );
                if let Some(predictor) = self.predictor.as_mut() {
                    predictor.state.set_position(position);
                }
            }
        }
        self.wake_from_bed_v5();
        self.state.local.sleeping = false;
        self.state.local.sleep_timer = 100;
        self.sync_local_living_render_state();
    }

    fn clear_sleep_state(&mut self) {
        self.set_local_player_size(PLAYER_WIDTH, PLAYER_HEIGHT);
        self.state.local.sleeping = false;
        self.state.local.sleep_timer = 0;
        self.state.local.sleeping_at = None;
    }

    /// What a mid-connection JoinGame does to the EntityPlayerSP it keeps.
    ///
    /// java: `Minecraft.loadWorld` (`:2414-2416`) calls `EntityPlayer.preparePlayerToSpawn`
    /// (`:578-584`: `setSize(0.6F, 1.8F)`, then `Entity.preparePlayerToSpawn`, full health and
    /// `deathTime = 0`, the last two in `sync::dispatch`), then installs a fresh
    /// `MovementInputFromOptions`. Everything else on the player, including the predictor's
    /// `lastReported*` and `positionUpdateTicks`, stays as it was.
    fn prepare_reused_player_to_spawn(&mut self) {
        self.set_local_player_size(PLAYER_WIDTH, PLAYER_HEIGHT);
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        let state = &mut predictor.state;
        // `Entity.preparePlayerToSpawn`: while `0 < posY < 256`, `setPosition` and climb one
        // block out of any collision in the player's `worldObj`. That is the discarded world,
        // which the client no longer has; a player standing in it is not inside its blocks, so
        // the loop ends after its first `setPosition`.
        if state.pos.y > 0.0 && state.pos.y < 256.0 {
            state.set_position(state.pos);
        }
        state.vel = DVec3::ZERO;
        state.pitch = 0.0;
        // The fresh movement input has no key down: next tick's previous-input reads (`flag`,
        // `flag1`, `flag2` in `onLivingUpdate`) and `isSneaking()` start clear.
        state.prev_jump = false;
        state.prev_sneak = false;
        state.prev_move_forward = 0.0;
        state.sneaking = false;
        self.prepare_reused_player_to_spawn_v5();
    }

    /// Rebuild the session-owned halves of Java's replacement `EntityPlayerSP`.
    ///
    /// java: `Minecraft.setDimensionAndSpawnPlayer` (`:2427-2456`) creates a new player and a
    /// new `MovementInputFromOptions`, then runs `preparePlayerToSpawn`, `flipPlayer`, and the
    /// controller capability setup. `PlayerControllerMP` itself is retained, so its
    /// `currentPlayerItem` and active-dig fields are intentionally *not* reset here.
    fn reset_player_after_respawn(&mut self) {
        self.effects = MovementEffects::default();
        self.state.local.reset_health_pipeline();
        self.state.local.food_level = 20;
        self.state.local.saturation = 5.0;
        self.state.local.experience_bar = 0.0;
        self.state.local.level = 0;
        self.state.local.total_experience = 0;
        // java: a newly constructed InventoryPlayer starts on hotbar slot zero, while the
        // retained PlayerControllerMP.currentPlayerItem still owns the server-side cache. The
        // next updateController therefore emits C09(0) when the old slot was non-zero.
        self.state.local.held_slot = 0;
        self.state.local.pending_velocity = None;
        // `setDimensionAndSpawnPlayer` constructs a fresh EntityPlayerSP.  The copied watcher
        // stays above (Java copies it explicitly), while equipment, server-position integers,
        // interpolation targets and head-only packet state belong to the replacement object.
        self.state.local.reset_entity_packet_state();
        self.clear_sleep_state();
        // `setDimensionAndSpawnPlayer` constructs a fresh EntityPlayerSP. Its spawnChunk and
        // spawnForced fields therefore start empty until a later S05 sets them again.
        self.local_spawn_point = None;
        self.local_spawn_forced = false;

        // `setInitialSpawnLocation` runs immediately before construction on every S07 path, and
        // the new EntityPlayer constructor starts at that world spawn plus (0.5, 1.0, 0.5).
        // `flipPlayer` then installs -180 degrees; the subsequent S08 may replace both values.
        let spawn = self.state.spawn_position();
        let mut state = PhysicsState::new_player(DVec3::new(
            f64::from(spawn.x) + 0.5,
            f64::from(spawn.y) + 1.0,
            f64::from(spawn.z) + 0.5,
        ));
        state.yaw = -180.0;
        // `setDimensionAndSpawnPlayer` copies the old DataWatcher into the new entity (S07
        // dispatch keeps `local.metadata`), so `isSprinting()` (watched bit 3) survives. It then
        // installs a fresh `MovementInputFromOptions`, and `EntityPlayerSP.isSneaking()` reads
        // that input, not watched bit 1: the replacement is not sneaking until its first tick
        // samples the key (`new_player`'s `false`).
        let watched = Watched::new(TrackedKind::Player, &self.state.local.metadata);
        state.sprinting = watched.is_sprinting();
        state.flying = self.state.local.abilities.flying;
        state.allow_flying = self.state.local.abilities.allow_flying;
        state.fly_speed = self.state.local.abilities.fly_speed;
        state.movement_speed = mc_physics::state::WALK_SPEED;
        // The replacement EntityPlayerSP gets a fresh attribute map (no sprint entry) even
        // though its copied watcher may still carry the sprint flag: vanilla walks at 0.1 with
        // `isSprinting()` true until a local `setSprinting` edge or an S20 changes the map.
        state.sprint_modifier_amount = None;
        self.predictor = Some(Predictor::new(state));
        // `setDimensionAndSpawnPlayer` -> `spawnEntityInWorld`: an unsent spawn column (always,
        // after a dimension change) costs the replacement its first world tick.
        self.local_added_to_chunk = self.local_spawn_column_is_loaded();
        self.reset_local_render_state();
        self.reset_local_particle_state();

        // These are EntityPlayerSP's two report-edge fields. Do not replace OutboundState: its
        // held-slot cache and PlayerControllerMP's active dig survive the player replacement.
        self.outbound.reset_local_player_reporting();
    }

    /// java: `World.spawnEntityInWorld` gives the player to
    /// `getChunkFromChunkCoords(floor(posX / 16), floor(posZ / 16))`; only a column the server
    /// has sent is a real `Chunk` whose `addEntity` sets `addedToChunk`.
    fn local_spawn_column_is_loaded(&self) -> bool {
        self.predictor.as_ref().is_some_and(|predictor| {
            let pos = predictor.state.pos;
            self.state.world.is_chunk_loaded(mc_math::ChunkPos::new(
                floor_f64(pos.x / 16.0),
                floor_f64(pos.z / 16.0),
            ))
        })
    }

    /// Replace Java's render-view object at a world/player lifecycle boundary.
    fn reset_render_view_target(&mut self) {
        self.world_generation = self.world_generation.wrapping_add(1);
        self.view_target = RenderViewTarget::Local;
    }

    fn tick_sleep_timer(&mut self) {
        let local = &mut self.state.local;
        if local.sleeping {
            local.sleep_timer = (local.sleep_timer + 1).min(100);
        } else if local.sleep_timer > 0 {
            local.sleep_timer += 1;
            if local.sleep_timer >= 110 {
                local.sleep_timer = 0;
            }
        }
    }

    /// java: `EnchantmentHelper.getDepthStriderModifier` reads
    /// `getMaxEnchantmentLevel(Enchantment.depthStrider.effectId, player.getInventory())`, and
    /// `EntityPlayer.getInventory()` is `inventory.armorInventory` — so any armour piece counts,
    /// not only the boots, and the *maximum* level wins.
    ///
    /// Window `0`'s slots `5..=8` are the four armour slots (`ContainerPlayer`'s
    /// `SlotArmor`s), which is the same range `Client::refresh_armor` reads.
    pub fn set_armor_depth_strider(&mut self, level: f32) {
        self.effects.depth_strider = level;
    }
}

/// The flight capability mutation that occurs before `super.onLivingUpdate`'s movement body.
///
/// java: `EntityPlayerSP.onLivingUpdate` (`:823-845`) checks the previous jump flag (`flag`),
/// then either forces spectator flight on or toggles it on the second jump edge.  The predictor
/// performs that same mutation, but its public return value is deliberately only the movement
/// packet, so Session mirrors this predicate to retain the corresponding C13 event.
fn flight_toggle_target(state: &PhysicsState, input: &MoveInput) -> Option<bool> {
    if !state.allow_flying {
        return None;
    }
    // java: `this.mc.playerController.isSpectatorMode()`, not the tab-list `isSpectator()`.
    if state.spectator_mode {
        return (!state.flying).then_some(true);
    }
    if !state.prev_jump && input.jump && state.fly_toggle_timer != 0 {
        Some(!state.flying)
    } else {
        None
    }
}

/// Java's comparison-based `EntityLivingBase.setHealth` clamp for the local border prediction.
/// Keep NaN unchanged (`<`/`>` are both false), unlike Rust's `min`/`max` helpers.
fn set_local_health(local: &mut crate::sync::LocalPlayer, health: f32) {
    local.health = if health < 0.0 {
        0.0
    } else if health > local.max_health {
        local.max_health
    } else {
        health
    };
}

/// UUID identity is the only equality used by Java `AttributeModifier.equals/hashCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttributeResolutionError {
    DuplicateUuid(uuid::Uuid),
}

/// Return modifier indices in the order of vanilla's UUID-keyed `HashSet` for one operation.
///
/// `ModifiableAttributeInstance.computeValue` does not iterate S20's wire vector directly:
/// `func_180375_b` copies each operation's HashSet and `computeValue` then folds that copy. The
/// UUID order helper models the Java 8 bucket/chain traversal, including the insertion order of
/// colliding entries. Keeping this as a separate helper makes the operation-0/1 controls bite the
/// same ordering mutant as operation 2.
fn modifier_indices_in_java_order(
    modifiers: &[mc_protocol::play::s2c::AttributeModifierSnapshot],
    operation: mc_protocol::play::s2c::AttributeModifierOperation,
) -> Vec<usize> {
    let indices = modifiers
        .iter()
        .enumerate()
        .filter(|(_, modifier)| modifier.operation == operation)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let ids = indices
        .iter()
        .map(|&index| modifiers[index].uuid.as_u128())
        .collect::<Vec<_>>();
    mc_physics::attribute::java_uuid_hash_set_order(&ids)
        .into_iter()
        .map(|index| indices[index])
        .collect()
}

/// java: `ModifiableAttributeInstance.computeValue` (`:166-188`).
///
/// Operation 0 and operation 1 are deliberately left-folded in Java's UUID-HashSet order. A
/// floating-point `sum()` is not equivalent: it both chooses wire order and changes the rounding
/// tree. The returned error mirrors `applyModifier`'s hard UUID duplicate rejection.
///
/// Both movement attributes and `generic.maxHealth` snapshots use this shared resolver, keeping
/// UUID replacement, duplicate rejection, iteration order, and floating-point rounding aligned.
pub(crate) fn resolve_attribute(
    base_value: f64,
    modifiers: &[mc_protocol::play::s2c::AttributeModifierSnapshot],
) -> Result<f64, AttributeResolutionError> {
    let mut seen = HashSet::with_capacity(modifiers.len());
    for modifier in modifiers {
        if !seen.insert(modifier.uuid) {
            return Err(AttributeResolutionError::DuplicateUuid(modifier.uuid));
        }
    }

    use mc_protocol::play::s2c::AttributeModifierOperation as Op;
    let mut base = base_value;
    for index in modifier_indices_in_java_order(modifiers, Op::Add) {
        base += modifiers[index].amount;
    }

    let mut total = base;
    for index in modifier_indices_in_java_order(modifiers, Op::MultiplyBase) {
        total += base * modifiers[index].amount;
    }
    for index in modifier_indices_in_java_order(modifiers, Op::Multiply) {
        total *= 1.0 + modifiers[index].amount;
    }
    Ok(total)
}

/// UUID used by `EntityLivingBase.setSprinting` for its local 30% speed multiplier.
const SPRINT_SPEED_MODIFIER_UUID: uuid::Uuid =
    uuid::Uuid::from_u128(0x662A6B8D_DA3E_4C1C_8813_96EA6097278D);

/// java: `Potion.moveSpeed`'s movement-speed modifier (`potion/Potion.java:26`), `+0.2` per
/// level, operation 2. Dynamic FOV's Speed slider.
const SPEED_POTION_MODIFIER_UUID: uuid::Uuid =
    uuid::Uuid::from_u128(0x91AEAA56_376B_4498_935B_2F7F68070635);

/// java: `Potion.moveSlowdown`'s movement-speed modifier (`potion/Potion.java:27`), `-0.15` per
/// level, operation 2. Dynamic FOV's Slowness slider.
const SLOWNESS_POTION_MODIFIER_UUID: uuid::Uuid =
    uuid::Uuid::from_u128(0x7107DE5E_7CE8_4030_940E_514C1F160890);

/// A resolved local movement-speed snapshot and its UUID-keyed modifier replacement.
#[derive(Debug, Clone, PartialEq)]
struct MovementAttributeResolution {
    /// The value after all packet modifiers except the canonical local sprint UUID.
    value_without_sprint: f64,
    /// `removeAllModifiers` + `applyModifier` replacement state for this snapshot.
    modifiers: MovementModifierStore,
}

/// Resolve a server movement-speed snapshot without applying the sprint modifier twice.
///
/// The predictor owns the local `setSprinting` branch. If S20 includes the canonical UUID, the
/// replacement map carries its packet amount; if it does not, `removeAllModifiers` has removed
/// the local entry and the physics seam suppresses the boost until a later local edge.
fn resolve_movement_attribute(
    base_value: f64,
    modifiers: &[mc_protocol::play::s2c::AttributeModifierSnapshot],
) -> Result<MovementAttributeResolution, AttributeResolutionError> {
    let store = MovementModifierStore::from_slice(modifiers)?;
    let without_sprint = store
        .in_order()
        .into_iter()
        .filter(|modifier| modifier.uuid != SPRINT_SPEED_MODIFIER_UUID)
        .collect::<Vec<_>>();
    Ok(MovementAttributeResolution {
        value_without_sprint: resolve_attribute(base_value, &without_sprint)?,
        modifiers: store,
    })
}

/// `BlockBed.FACING` is stored as the low two metadata bits in S-W-N-E order.
fn bed_facing(world: &World, bed: BlockPos) -> Facing {
    Facing::from_horizontal(i32::from(world.block_at(bed).meta() & 3))
}

/// `BlockBed.getSafeExitLocation(world, bed, 0)`.
fn safe_bed_exit(world: &World, bed: BlockPos) -> Option<BlockPos> {
    let facing = bed_facing(world, bed);
    let (dx, _, dz) = facing.offset();
    for layer in 0..=1 {
        let min_x = bed.x - dx * layer - 1;
        let min_z = bed.z - dz * layer - 1;
        for x in min_x..=min_x + 2 {
            for z in min_z..=min_z + 2 {
                let candidate = BlockPos::new(x, bed.y, z);
                if has_room_to_leave_bed(world, candidate) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn has_room_to_leave_bed(world: &World, pos: BlockPos) -> bool {
    world.does_block_have_solid_top_surface(pos.down())
        && !world.block_at(pos).props().material.is_solid()
        && !world.block_at(pos.up()).props().material.is_solid()
}

/// java: `EntityLivingBase.onDeathUpdate`'s `if (this.deathTime == 20) { ... this.setDead(); }`.
/// `LocalPlayer::tick_living_state` stops counting at this value and only a new player (respawn
/// or join, via `reset_health_pipeline`) resets it, so reaching it is vanilla's `isDead` removal.
const LOCAL_PLAYER_REMOVED_DEATH_TIME: i32 = 20;

/// `World.isBlockLoaded(new BlockPos(pos.x, 0.0, pos.z))`, for the player at `pos`.
///
/// **What vanilla's expression actually evaluates to, stated exactly rather than transcribed.**
/// `World.isBlockLoaded(pos)` delegates to `isBlockLoaded(pos, true)` — `allowEmpty = true` —
/// which is `isValid(pos) && chunkProvider.chunkExists(x >> 4, z >> 4)`. On a *client*
/// `ChunkProviderClient.chunkExists` is a hard-coded `return true`
/// (`client/multiplayer/ChunkProviderClient.java:44`), so vanilla's own gate degrades to the
/// coordinate-validity (`World.isValid`) test and nothing else. This is the hard world limit,
/// not the dynamic `WorldBorder`: 1.8.9's client really can be ticked in a column it has never
/// been sent, and the block reads come back as air because `provideChunk` hands out one shared
/// `EmptyChunk`.
///
/// Unlike `allowEmpty = false`, this predicate deliberately does not ask whether a real chunk
/// has arrived. Block reads from a missing column use the client's blank `EmptyChunk` semantics.
/// The Y coordinate is intentionally absent: the local object is an `EntityLivingBase`, whose
/// `kill()` calls `attackEntityFrom(DamageSource.outOfWorld, 4)`; that method returns immediately
/// on `WorldClient.isRemote`, so a local player below Y=-64 is not marked dead by this base hook.
/// Java constructs a `BlockPos` before calling `World.isValid`, so the horizontal coordinates
/// must use `MathHelper.floor_double`'s Java narrowing rules. In particular NaN becomes block 0,
/// while infinities narrow outside the valid range; the player's raw position is never clamped.
fn player_position_is_valid(pos: DVec3) -> bool {
    // java: World.isValid — ±30,000,000 horizontally. The Y range is not tested because
    // vanilla passes a hard-coded y = 0, which is always inside 0..256.
    let x = floor_f64(pos.x);
    let z = floor_f64(pos.z);
    (-30_000_000..30_000_000).contains(&x) && (-30_000_000..30_000_000).contains(&z)
}

/// Interpolate a vanilla yaw/pitch sample over the shortest degree arc.
fn interpolate_view_rotation(previous: f32, current: f32, partial_ticks: f32) -> f32 {
    previous + mc_math::wrap_degrees_f32(current - previous) * partial_ticks
}

/// Whether `orientCamera` turns a spectating camera with `rotationYawHead` rather than
/// `rotationYaw`: java's `if (entity instanceof EntityAnimal)`. The 1.8.9 `EntityAnimal`
/// subclasses are the pig, sheep, cow, mooshroom, chicken, horse, rabbit, wolf and ocelot.
fn spectator_camera_uses_head_yaw(kind: TrackedKind) -> bool {
    use mc_data::EntityType as E;
    matches!(
        kind.entity_type(),
        Some(
            E::Pig
                | E::Sheep
                | E::Cow
                | E::MushroomCow
                | E::Chicken
                | E::EntityHorse
                | E::Rabbit
                | E::Wolf
                | E::Ozelot
        )
    )
}

/// `EntityLivingBase.updateDistance` for the session-owned player.
///
/// Keep this copy beside the local snapshot rather than reaching into the remote tracker: the
/// two objects have independent render-yaw lifetimes, and local body state must survive neither
/// a JoinGame nor a respawn boundary.
fn local_player_update_distance(body_yaw: f32, candidate: f32, rotation_yaw: f32) -> f32 {
    let mut body = body_yaw;
    body += mc_math::wrap_degrees_f32(candidate - body) * 0.3;
    let head_delta =
        mc_math::clamp_f32(mc_math::wrap_degrees_f32(rotation_yaw - body), -75.0, 75.0);
    body = rotation_yaw - head_delta;
    if head_delta * head_delta > 2_500.0 {
        body += head_delta * 0.2;
    }
    body
}

// The border's clock is `Environment::border_now_millis`: the wall clock
// (`System.currentTimeMillis()`, the same domain the S44 stamps use) in live play, and the
// recording's clock while a replay plays.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::channel::NetHandle as _NetHandle;
    use crate::sync::entities::EntityParticleArgs;
    use mc_protocol::play::{s2c, S2CPlay};

    /// A session wired to a hand-fed channel, with no runtime and no socket.
    ///
    /// The outbound receiver is handed back so the caller keeps it alive; dropping it would
    /// make every `send` report `NotConnected`.
    fn rig() -> (
        Session,
        tokio::sync::mpsc::Sender<ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (handle, to_game, to_net) = _NetHandle::new_pair();
        (
            Session::new(handle, JoinSettings::default()),
            to_game,
            to_net,
        )
    }

    fn join_game() -> S2CPlay {
        S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 3,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        })
    }

    /// A real full-column `S21PacketChunkData` for `(chunk_x, chunk_z)` carrying one solid
    /// layer of stone at world height `floor_y` and air everywhere else.
    ///
    /// Built as bytes and sent through the ordinary packet path rather than poked into the
    /// `World` directly, because the ordering of *packets* is exactly what is under test.
    fn chunk_with_a_stone_floor(chunk_x: i32, chunk_z: i32, floor_y: i32) -> S2CPlay {
        assert!((0..256).contains(&floor_y));
        let section = (floor_y >> 4) as u16;
        let bitmask = 1u16 << section;
        // Dimension 0, so sky light is present and biomes follow (H6).
        let mut data = vec![0u8; mc_world::expected_chunk_data_len(bitmask, true, true)];
        let stone = mc_data::BlockState::new(
            mc_data::block_by_name("stone").expect("1.8.9 registers stone"),
            0,
        );
        // Blocks come first, one little-endian u16 per cell in `y << 8 | z << 4 | x` order, so
        // the layer at section-local y is a contiguous 256-cell (512-byte) run.
        let local_y = (floor_y & 15) as usize;
        for cell in 0..256usize {
            let at = (local_y * 256 + cell) * 2;
            data[at] = (stone.id & 0xFF) as u8;
            data[at + 1] = (stone.id >> 8) as u8;
        }
        S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
            chunk_x,
            chunk_z,
            ground_up_continuous: true,
            primary_bit_mask: bitmask,
            data,
        })
    }

    /// Every packet the session has queued for the network task so far.
    fn drain_outbound(
        to_net: &mut tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) -> Vec<C2SPlay> {
        let mut packets = Vec::new();
        while let Ok(command) = to_net.try_recv() {
            if let crate::net::ToNet::Send(packet) = command {
                packets.push(packet);
            }
        }
        packets
    }

    /// The four packets `onUpdateWalkingPlayer` chooses between. java: `C03PacketPlayer` and
    /// its three nested subclasses.
    fn is_movement(packet: &C2SPlay) -> bool {
        matches!(
            packet,
            C2SPlay::Player(_)
                | C2SPlay::PlayerPosition(_)
                | C2SPlay::PlayerLook(_)
                | C2SPlay::PlayerPositionAndLook(_)
        )
    }

    #[test]
    fn phase_walks_connecting_to_playing_to_disconnected() {
        let (mut session, to_game, _to_net) = rig();
        assert_eq!(*session.phase(), Phase::Connecting);

        to_game
            .try_send(ToGame::Connected(LoginOutcome {
                uuid: uuid::Uuid::from_u128(9),
                username: "Dev".into(),
                compression_threshold: None,
                encrypted: false,
            }))
            .unwrap();
        session.poll(64);
        assert_eq!(*session.phase(), Phase::LoggingIn);
        assert_eq!(session.state().local.username, "Dev");

        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        session.poll(64);
        assert_eq!(*session.phase(), Phase::Playing);
        assert!(session.predictor().is_some());

        to_game
            .try_send(ToGame::Disconnected(DisconnectReason::ClosedByClient))
            .unwrap();
        let events = session.poll(64);
        assert!(matches!(
            session.phase(),
            Phase::Disconnected(DisconnectReason::ClosedByClient)
        ));
        assert!(events
            .iter()
            .any(|e| matches!(e, ClientEvent::Disconnected(_))));
    }

    /// S05 updates both the world spawn and the session-owned player's forced spawn point. A
    /// respawn constructs a new EntityPlayerSP, so the player-scoped half must be cleared until
    /// another S05 arrives.
    #[test]
    fn spawn_position_updates_world_and_local_forced_spawn_state() {
        let (mut session, to_game, _to_net) = rig();
        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        session.poll(16);
        assert_eq!(session.local_spawn_point(), None);
        assert!(!session.local_spawn_forced());

        let spawn = BlockPos::new(-12, 73, 91);
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnPosition(s2c::SpawnPosition {
                position: spawn,
            })))
            .unwrap();
        let events = session.poll(16);
        assert_eq!(session.state().spawn_position(), spawn);
        assert_eq!(session.local_spawn_point(), Some(spawn));
        assert!(session.local_spawn_forced());
        assert!(events
            .iter()
            .any(|event| matches!(event, ClientEvent::SpawnPositionChanged(position) if *position == spawn)));

        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert_eq!(session.local_spawn_point(), None);
        assert!(!session.local_spawn_forced());
    }

    /// W-KEEPALIVE: the loading screen owns an independent one-second cadence. The controls are
    /// deliberately in the same test: tick 19 must have no key-zero keepalive, the first S08
    /// must stop the cadence, a same-dimension respawn must not restart it, and a
    /// cross-dimension one must.
    #[test]
    fn terrain_download_sends_key_zero_every_twenty_ticks_only_while_its_screen_is_open() {
        let (mut session, to_game, mut to_net) = rig();
        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        session.poll(16);
        assert_eq!(session.terrain_download_progress(), Some(0));
        let _join_packets = drain_outbound(&mut to_net);

        // An ordinary non-S08 packet is not a completion edge. The screen and its progress
        // counter must remain alive until the position correction arrives.
        to_game
            .try_send(ToGame::Packet(S2CPlay::KeepAlive(s2c::KeepAlive { id: 7 })))
            .unwrap();
        session.poll(16);
        assert_eq!(session.terrain_download_progress(), Some(0));

        for _ in 0..19 {
            session.tick(&MoveInput::default());
        }
        assert_eq!(session.terrain_download_progress(), Some(19));
        let first_nineteen = drain_outbound(&mut to_net);
        // The screen counts every tick; the player, spawned into an unsent column, skips the
        // first world tick (`addedToChunk`) and reports from the second.
        assert_eq!(first_nineteen.iter().filter(|p| is_movement(p)).count(), 18);
        assert!(
            !first_nineteen
                .iter()
                .any(|packet| matches!(packet, C2SPlay::KeepAlive(_))),
            "first key-zero keepalive is tick 20"
        );
        session.tick(&MoveInput::default());
        assert_eq!(session.terrain_download_progress(), Some(20));
        let twentieth = drain_outbound(&mut to_net);
        assert_eq!(twentieth.iter().filter(|p| is_movement(p)).count(), 1);
        assert_eq!(
            twentieth
                .iter()
                .filter(|packet| matches!(packet, C2SPlay::KeepAlive(c2s::KeepAlive { key: 0 })))
                .count(),
            1
        );

        let correction = || {
            S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x: 8.5,
                y: 70.0,
                z: 8.5,
                yaw: 0.0,
                pitch: 0.0,
                flags: 0,
            })
        };
        to_game.try_send(ToGame::Packet(correction())).unwrap();
        let events = session.poll(16);
        assert!(matches!(
            events.first(),
            Some(ClientEvent::PositionCorrection(_))
        ));
        assert!(matches!(
            events.get(1),
            Some(ClientEvent::TerrainDownloaded)
        ));
        assert_eq!(session.terrain_download_progress(), None);
        let confirmation = drain_outbound(&mut to_net);
        assert_eq!(confirmation.iter().filter(|p| is_movement(p)).count(), 1);
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        assert!(
            !drain_outbound(&mut to_net)
                .iter()
                .any(|packet| matches!(packet, C2SPlay::KeepAlive(_))),
            "the first S08 closes GuiDownloadTerrain"
        );

        // java: same-dimension `handleRespawn` does not open GuiDownloadTerrain.
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert_eq!(session.terrain_download_progress(), None);
        for _ in 0..20 {
            session.tick(&MoveInput::default());
        }
        assert!(
            !drain_outbound(&mut to_net)
                .iter()
                .any(|packet| matches!(packet, C2SPlay::KeepAlive(_))),
            "same-dimension respawn keeps the screen closed"
        );

        // A dimension change constructs a fresh screen, so its first send is twenty ticks later.
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: -1,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert_eq!(session.terrain_download_progress(), Some(0));
        for _ in 0..19 {
            session.tick(&MoveInput::default());
        }
        assert_eq!(session.terrain_download_progress(), Some(19));
        let first_nineteen = drain_outbound(&mut to_net);
        // The screen counts every tick; the player, spawned into an unsent column, skips the
        // first world tick (`addedToChunk`) and reports from the second.
        assert_eq!(first_nineteen.iter().filter(|p| is_movement(p)).count(), 18);
        assert!(
            !first_nineteen
                .iter()
                .any(|packet| matches!(packet, C2SPlay::KeepAlive(_))),
            "fresh counter resets to zero"
        );
        session.tick(&MoveInput::default());
        assert_eq!(session.terrain_download_progress(), Some(20));
        let twentieth = drain_outbound(&mut to_net);
        assert_eq!(twentieth.iter().filter(|p| is_movement(p)).count(), 1);
        assert_eq!(
            twentieth
                .iter()
                .filter(|packet| matches!(packet, C2SPlay::KeepAlive(c2s::KeepAlive { key: 0 })))
                .count(),
            1
        );
    }

    /// W-HELDITEM: `PlayerControllerMP.updateController` performs the lazy once-per-tick
    /// `syncCurrentPlayItem` before the world updates the player. This covers both local
    /// coalescing and `NetHandlerPlayClient.handleHeldItemChange`'s server-forced echo.
    #[test]
    fn held_item_changes_are_coalesced_and_server_forced_slots_echo_before_movement() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);

        // Several local mutations between controller updates collapse to the final slot. The
        // cache still contains vanilla's initial zero until the tick actually runs.
        session.state_mut().local.held_slot = 2;
        session.state_mut().local.held_slot = 7;
        assert!(
            drain_outbound(&mut to_net).is_empty(),
            "selection itself is local-only"
        );
        session.tick(&MoveInput::default());
        let first = drain_outbound(&mut to_net);
        assert!(matches!(
            first.first(),
            Some(C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 7 }))
        ));
        assert_eq!(
            first
                .iter()
                .filter(|packet| matches!(packet, C2SPlay::HeldItemChange(_)))
                .count(),
            1,
            "one controller update sends at most one C09: {first:?}"
        );
        assert!(first.iter().skip(1).any(is_movement));

        session.tick(&MoveInput::default());
        assert!(
            !drain_outbound(&mut to_net)
                .iter()
                .any(|packet| matches!(packet, C2SPlay::HeldItemChange(_))),
            "the cached slot suppresses a redundant per-tick C09"
        );

        // S09 writes inventory.currentItem but deliberately leaves the controller cache at 7.
        // The packet handler itself sends nothing; the next update echoes slot 3 before C03.
        to_game
            .try_send(ToGame::Packet(S2CPlay::HeldItemChange(
                s2c::HeldItemChange { slot: 3 },
            )))
            .unwrap();
        let events = session.poll(16);
        assert!(events
            .iter()
            .any(|event| matches!(event, ClientEvent::HeldSlotChanged(3))));
        assert_eq!(session.state().local.held_slot, 3);
        assert!(drain_outbound(&mut to_net).is_empty());

        session.tick(&MoveInput::default());
        let echoed = drain_outbound(&mut to_net);
        assert!(matches!(
            echoed.first(),
            Some(C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 3 }))
        ));
        assert!(echoed.iter().skip(1).any(is_movement));

        // Production's shell owns the earlier boundary: updateController has already run when
        // the hotbar input at the tail of runTick changes the local field. That tick's movement
        // crosses first; the following update reports only the final selection.
        session.update_controller();
        assert!(drain_outbound(&mut to_net).is_empty());
        session.state_mut().local.held_slot = 5;
        session.state_mut().local.held_slot = 8;
        session.tick(&MoveInput::default());
        let same_tick = drain_outbound(&mut to_net);
        assert!(same_tick.first().is_some_and(is_movement));
        assert!(!same_tick
            .iter()
            .any(|packet| matches!(packet, C2SPlay::HeldItemChange(_))));

        session.update_controller();
        assert!(matches!(
            drain_outbound(&mut to_net).as_slice(),
            [C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 8 })]
        ));
    }

    /// F-P0-client.entity-01-4 / F-P0-network.play.server-02-1: S07 replaces
    /// EntityPlayerSP, but not PlayerControllerMP. The first post-respawn walking update must
    /// therefore restart the position/look, movement-input, and sprint edge state while keeping
    /// the controller's active dig/cache (which then notices the fresh inventory slot 0).
    #[test]
    fn respawn_restarts_player_reporting_without_resetting_controller_state() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        let active = MoveInput {
            forward: true,
            sprint: true,
            yaw: 45.0,
            ..MoveInput::default()
        };
        session.tick(&active);
        let started = drain_outbound(&mut to_net);
        assert!(started.iter().any(|packet| matches!(
            packet,
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSprinting,
                ..
            })
        )));

        let target = BlockPos::new(9, 65, 9);
        // The controller cache and InventoryPlayer's selected slot move together in production.
        // Keeping only the cache at 4 constructs an impossible state that the new per-tick sync
        // quite correctly repairs back to local slot 0 before the movement assertion below.
        session.state_mut().local.held_slot = 4;
        session.player_action(|out, packets| {
            out.sync_held_item(4, packets);
            out.start_digging(target, Facing::Up, packets);
        });
        let _setup_packets = drain_outbound(&mut to_net);
        {
            // Poison the fields that a copy-through respawn implementation would leak. These
            // values are not touched by packet dispatch; only rebuilding PhysicsState can clear
            // them, so this is the biting carry-state mutant for the whole-player replacement.
            let state = &mut session.predictor_mut().expect("joined predictor").state;
            state.sneaking = true;
            state.sprint_toggle_timer = 6;
            state.sprinting_ticks_left = 599;
            state.fly_toggle_timer = 6;
            state.prev_jump = true;
            state.prev_sneak = true;
            state.prev_move_forward = 1.0;
            state.fall_distance = 4.0;
            state.first_update = false;
            state.in_water = true;
            state.in_web = true;
            state.distance_walked_modified = 3.0;
            state.in_portal = true;
        }
        session.state_mut().local.food_level = 4;
        session.state_mut().local.saturation = 1.0;
        session.state_mut().local.experience_bar = 0.75;
        session.state_mut().local.level = 17;
        session.state_mut().local.total_experience = 1234;
        assert!(session.predictor().unwrap().current_tick() > 0);

        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        let same_dimension_events = session.poll(16);
        assert!(same_dimension_events.iter().any(|event| matches!(
            event,
            ClientEvent::Respawned {
                changed_world: false,
                ..
            }
        )));
        assert!(!same_dimension_events
            .iter()
            .any(|event| matches!(event, ClientEvent::TerrainDownloaded)));
        assert_eq!(session.terrain_download_progress(), None);
        assert_eq!(
            session.predictor().unwrap().current_tick(),
            0,
            "the replacement EntityPlayerSP has fresh reporting history"
        );
        assert_eq!(
            session.outbound_mut().digging(),
            Some(target),
            "PlayerControllerMP survives S07"
        );
        let replacement = session.predictor().expect("replacement predictor").state;
        assert_eq!(
            replacement.pos,
            DVec3::new(8.5, 65.0, 8.5),
            "the new EntityPlayer starts at World.setInitialSpawnLocation"
        );
        assert_eq!(replacement.yaw, -180.0);
        assert!(
            replacement.sprinting,
            "S07's fresh EntityPlayerSP copies the old DataWatcher flags before reporting resets"
        );
        assert!(!replacement.sneaking);
        assert_eq!(replacement.sprint_toggle_timer, 0);
        assert_eq!(replacement.sprinting_ticks_left, 0);
        assert_eq!(replacement.fly_toggle_timer, 0);
        assert!(!replacement.prev_jump);
        assert!(!replacement.prev_sneak);
        assert_eq!(replacement.prev_move_forward, 0.0);
        assert_eq!(replacement.fall_distance, 0.0);
        assert!(replacement.first_update);
        assert_eq!(session.state().local.health, 20.0);
        assert!(!session.state().local.has_valid_health);
        assert_eq!(session.state().local.food_level, 20);
        assert_eq!(session.state().local.saturation, 5.0);
        assert_eq!(session.state().local.experience_bar, 0.0);
        assert_eq!(session.state().local.level, 0);
        assert_eq!(session.state().local.total_experience, 0);
        assert_eq!(session.state().local.held_slot, 0);

        // PlayerControllerMP.currentPlayerItem survives at 4, but the fresh InventoryPlayer
        // starts at 0, so updateController emits exactly the vanilla C09 reset.
        session.update_controller();
        assert!(matches!(
            drain_outbound(&mut to_net).as_slice(),
            [C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 0 })]
        ));

        session.tick(&active);
        let after = drain_outbound(&mut to_net);
        assert!(matches!(
            after.first(),
            Some(C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSprinting,
                ..
            }))
        ));
        assert!(
            after
                .iter()
                .any(|packet| matches!(packet, C2SPlay::PlayerPositionAndLook(_))),
            "zeroed lastReported position/look forces the full post-respawn C06"
        );

        // Dimension-changing S07 takes the same replacement path, then opens a fresh terrain
        // wait. Poison a second batch of local state so this control cannot pass by inheriting
        // the first replacement's already-clean values.
        {
            let state = &mut session
                .predictor_mut()
                .expect("replacement predictor")
                .state;
            state.sprint_toggle_timer = 5;
            state.fly_toggle_timer = 5;
            state.prev_jump = true;
            state.prev_move_forward = 1.0;
            state.in_portal = true;
        }
        session.state_mut().local.held_slot = 6;
        session.player_action(|out, packets| out.sync_held_item(6, packets));
        let _second_setup = drain_outbound(&mut to_net);
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: -1,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        let events = session.poll(16);
        assert!(events.iter().any(|event| matches!(
            event,
            ClientEvent::Respawned {
                changed_world: true,
                ..
            }
        )));
        assert_eq!(session.terrain_download_progress(), Some(0));
        let cross_dimension = session
            .predictor()
            .expect("cross-dimension predictor")
            .state;
        assert_eq!(cross_dimension.pos, DVec3::new(8.5, 65.0, 8.5));
        assert_eq!(cross_dimension.yaw, -180.0);
        assert_eq!(cross_dimension.sprint_toggle_timer, 0);
        assert_eq!(cross_dimension.fly_toggle_timer, 0);
        assert!(!cross_dimension.prev_jump);
        assert_eq!(cross_dimension.prev_move_forward, 0.0);
        assert!(!cross_dimension.in_portal);
        assert_eq!(session.state().local.held_slot, 0);
        assert_eq!(
            session.outbound_mut().digging(),
            Some(target),
            "dimension replacement keeps PlayerControllerMP's active dig"
        );
        session.update_controller();
        assert!(matches!(
            drain_outbound(&mut to_net).as_slice(),
            [C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 0 })]
        ));
    }

    /// **F-P0-util-01-5.** The sampled sprint field is the raw `keyBindSprint.isKeyDown()` value;
    /// forward and sneak are separate downstream eligibility gates in `EntityPlayerSP`.  Run all
    /// four combinations through app input sampling and a fresh Session so a filtered input field
    /// cannot hide behind the predictor's later `moveForward`/sneak checks.
    #[test]
    fn raw_sprint_sampling_survives_while_session_engagement_uses_forward_and_sneak_gates() {
        for (forward, sneak, should_sprint) in [
            (false, false, false),
            (false, true, false),
            (true, false, true),
            (true, true, false),
        ] {
            let mut bindings = mc_render::KeyBindings::vanilla();
            bindings.set_state(
                mc_render::Input::Key(winit::keyboard::KeyCode::ControlLeft),
                true,
            );
            if forward {
                bindings.set_state(mc_render::Input::Key(winit::keyboard::KeyCode::KeyW), true);
            }
            if sneak {
                bindings.set_state(
                    mc_render::Input::Key(winit::keyboard::KeyCode::ShiftLeft),
                    true,
                );
            }
            let sampled = crate::app::input::move_input(&bindings, 0.0, 0.0, false);
            assert!(
                sampled.sprint,
                "raw sprint key survives forward={forward}, sneak={sneak}"
            );
            assert_eq!(sampled.forward, forward);
            assert_eq!(sampled.sneak, sneak);

            let (mut session, _to_game, mut to_net) = joined_and_standing();
            let _join_packets = drain_outbound(&mut to_net);
            session.tick(&sampled);
            let packets = drain_outbound(&mut to_net);
            assert_eq!(
                session.predictor().unwrap().state.sprinting,
                should_sprint,
                "Session keeps eligibility out of raw MoveInput::sprint: forward={forward}, sneak={sneak}"
            );
            assert_eq!(
                packets.iter().any(|packet| matches!(
                    packet,
                    C2SPlay::EntityAction(c2s::EntityAction {
                        action: c2s::EntityActionType::StartSprinting,
                        ..
                    })
                )),
                should_sprint,
                "C0B edge follows the downstream engagement gate"
            );
        }
    }

    #[test]
    fn poll_respects_its_budget_and_leaves_the_rest_queued() {
        let (mut session, to_game, _to_net) = rig();
        for id in 0..10 {
            to_game
                .try_send(ToGame::Packet(S2CPlay::SpawnExperienceOrb(
                    s2c::SpawnExperienceOrb {
                        entity_id: id,
                        x: 0,
                        y: 0,
                        z: 0,
                        xp_value: 1,
                    },
                )))
                .unwrap();
        }
        session.poll(4);
        assert_eq!(session.state().entities.len(), 4);
        session.poll(100);
        assert_eq!(session.state().entities.len(), 10);
    }

    /// `Minecraft.runGameLoop` drains `scheduledTasks` while holding its monitor, rather than
    /// imposing the invented 1,000-packet `NetworkManager` limit the old production call site
    /// documented. The 1,001st packet is the negative control: a replacement that silently
    /// retained that limit would leave it queued.
    #[test]
    fn poll_all_drains_past_the_former_one_thousand_packet_cap() {
        let (mut session, to_game, _to_net) = rig();
        for id in 0..=1_000 {
            to_game
                .try_send(ToGame::Packet(S2CPlay::SpawnExperienceOrb(
                    s2c::SpawnExperienceOrb {
                        entity_id: id,
                        x: 0,
                        y: 0,
                        z: 0,
                        xp_value: 1,
                    },
                )))
                .unwrap();
        }

        session.poll_all();
        assert_eq!(session.state().entities.len(), 1_001);
    }

    #[test]
    fn a_failing_packet_handler_is_logged_and_later_packets_still_apply() {
        let (mut session, to_game, _to_net) = rig();
        to_game
            .try_send(ToGame::Packet(S2CPlay::Chat(s2c::Chat {
                component: "{ not json".into(),
                position: 0,
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnExperienceOrb(
                s2c::SpawnExperienceOrb {
                    entity_id: 17,
                    x: 0,
                    y: 0,
                    z: 0,
                    xp_value: 1,
                },
            )))
            .unwrap();
        let events = session.poll(16);
        assert!(session.phase().is_live());
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ClientEvent::Disconnected(_))),
            "a handler exception is not a transport disconnect"
        );
        assert_eq!(session.state().entities.len(), 1);
        assert!(events
            .iter()
            .any(|event| matches!(event, ClientEvent::EntitySpawned { network_id: 17, .. })));
    }

    #[test]
    fn a_transport_or_decode_failure_still_ends_the_session_before_later_packets() {
        let (mut session, to_game, _to_net) = rig();
        to_game
            .try_send(ToGame::Disconnected(DisconnectReason::ConnectionLost {
                detail: "malformed packet body".into(),
            }))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnExperienceOrb(
                s2c::SpawnExperienceOrb {
                    entity_id: 17,
                    x: 0,
                    y: 0,
                    z: 0,
                    xp_value: 1,
                },
            )))
            .unwrap();

        let events = session.poll(16);
        assert!(matches!(session.phase(), Phase::Disconnected(_)));
        assert!(matches!(
            events.last(),
            Some(ClientEvent::Disconnected(
                DisconnectReason::ConnectionLost { .. }
            ))
        ));
        assert_eq!(session.state().entities.len(), 0);
    }

    #[test]
    fn a_pre_join_play_disconnect_keeps_its_origin_through_phase_and_event() {
        let (mut session, to_game, _to_net) = rig();
        let reason = DisconnectReason::ConnectionLost {
            detail: "EOF before JoinGame".into(),
        }
        .with_play_origin();
        to_game
            .try_send(ToGame::Disconnected(reason.clone()))
            .expect("disconnect channel");

        let events = session.poll(16);
        assert_eq!(session.phase(), &Phase::Disconnected(reason.clone()));
        assert!(events
            .iter()
            .any(|event| matches!(event, ClientEvent::Disconnected(got) if got == &reason)));
        assert_eq!(reason.title_key(), "disconnect.lost");
    }

    /// Regression test for the `apply_correction` bounding-box bug documented in
    /// [`Session::react_to`], now guarding S13's own fix rather than a call-site workaround.
    /// Without it the player is back at the origin one tick after every teleport, which is
    /// invisible in any test that only checks the packet the client sends back.
    #[test]
    fn a_position_correction_survives_the_next_physics_tick() {
        let (mut session, to_game, _to_net) = rig();
        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        session.poll(16);
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: 40.5,
                    y: 71.0,
                    z: -39.5,
                    yaw: 180.0,
                    pitch: 12.5,
                    flags: 0,
                },
            )))
            .unwrap();
        // `40 >> 4 == 2`, `-40 >> 4 == -3`: give the corrected position a real floor so this
        // test exercises ordinary loaded-column gravity rather than EmptyChunk's sink-rate pin.
        to_game
            .try_send(ToGame::Packet(chunk_with_a_stone_floor(2, -3, 64)))
            .unwrap();
        session.poll(16);

        let placed = session.predictor().unwrap().state.pos;
        assert_eq!(placed, mc_math::DVec3::new(40.5, 71.0, -39.5));
        assert_eq!(
            session.predictor().unwrap().state.prev_pos,
            placed,
            "the first S08 must snapshot prev_pos after applying the terrain correction"
        );
        assert_eq!(
            session.render_view_position(0.5),
            Some(placed),
            "first-terrain interpolation must not lerp from the constructor position"
        );

        // The first world tick skips the player spawned at JoinGame into an unsent column
        // (`addedToChunk`). Then two ticks: vanilla's first simulated tick only
        // accumulates `motionY -= 0.08`, so the fall does not show up until the second.
        session.tick(&MoveInput::default());
        assert_eq!(session.predictor().unwrap().state.pos, placed);
        session.tick(&MoveInput::default());
        session.tick(&MoveInput::default());
        let after = session.predictor().unwrap().state.pos;
        assert_eq!(after.x, 40.5, "X must not snap back");
        assert_eq!(after.z, -39.5, "Z must not snap back");
        assert!(
            after.y < 71.0 && after.y > 70.0,
            "expected a short fall from 71.0, got {}",
            after.y
        );

        // Java's Entity.setPositionAndRotation snapshots prevPos for every S08, not only the
        // terrain-completion packet. This second correction is the negative control for the
        // old first-terrain-only implementation: its render sample must not retain the prior
        // tick's position after a later rubber-band.
        let second = mc_math::DVec3::new(-12.25, 88.0, 17.75);
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: second.x,
                    y: second.y,
                    z: second.z,
                    yaw: -45.0,
                    pitch: -8.0,
                    flags: 0,
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.predictor().unwrap().state.pos, second);
        assert_eq!(session.predictor().unwrap().state.prev_pos, second);
        assert_eq!(session.render_view_position(0.5), Some(second));
    }

    /// F-P0-network.play.client-01-3 / F-P1-client.multiplayer-01-1: on a multiplayer client
    /// `ChunkProviderClient.chunkExists` is always true, so missing-column residency cannot stop
    /// either `EntityPlayerSP.onUpdate` or its final `onUpdateWalkingPlayer` call. The loaded
    /// control makes sure the fix did not merely invert the old residency predicate.
    #[test]
    fn unloaded_and_loaded_columns_emit_one_movement_packet_per_tick() {
        const TICKS: usize = 25;

        for column_loaded in [false, true] {
            let (mut session, to_game, mut to_net) = rig();
            to_game.try_send(ToGame::Packet(join_game())).unwrap();
            to_game
                .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                    s2c::PlayerPositionLook {
                        x: 8.5,
                        y: 70.0,
                        z: 8.5,
                        yaw: 0.0,
                        pitch: 0.0,
                        flags: 0,
                    },
                )))
                .unwrap();
            if column_loaded {
                to_game
                    .try_send(ToGame::Packet(chunk_with_a_stone_floor(0, 0, 64)))
                    .unwrap();
            }
            session.poll(64);

            assert_eq!(
                session
                    .world()
                    .is_chunk_loaded(BlockPos::new(8, 0, 8).chunk_pos()),
                column_loaded
            );
            // The first world tick skips the player spawned at JoinGame (`addedToChunk`), whatever arrived since; after it the column no longer matters.
            assert!(!session.player_is_simulated());
            assert_eq!(session.tick(&MoveInput::default()), None);
            assert!(
                session.player_is_simulated(),
                "allowEmpty=true makes chunk residency irrelevant"
            );

            // Ignore JoinGame traffic and S08's mandatory C06 confirmation.
            let _handshake = drain_outbound(&mut to_net);
            for tick in 0..TICKS {
                let packet = session.tick(&MoveInput::default());
                assert!(
                    packet.as_ref().is_some_and(is_movement),
                    "column_loaded={column_loaded}, tick {tick} did not produce its movement packet"
                );
            }

            let packets = drain_outbound(&mut to_net);
            assert_eq!(
                packets.iter().filter(|packet| is_movement(packet)).count(),
                TICKS,
                "column_loaded={column_loaded}: vanilla emits exactly one C03/C04/C05/C06 per tick"
            );
        }
    }

    #[test]
    fn local_player_tick_gate_is_only_the_horizontal_world_validity_check() {
        assert!(player_position_is_valid(DVec3::new(
            -30_000_000.0,
            f64::NAN,
            29_999_999.999
        )));
        // `EntityLivingBase.kill()` is remote-world no-op, so the inherited Y < -64 check does
        // not close EntityPlayerSP's horizontal `isBlockLoaded` gate.
        assert!(player_position_is_valid(DVec3::new(0.0, -65.0, 0.0)));
        assert!(!player_position_is_valid(DVec3::new(
            30_000_000.0,
            0.0,
            0.0
        )));
        assert!(player_position_is_valid(DVec3::new(f64::NAN, 0.0, 0.0)));
        assert!(!player_position_is_valid(DVec3::new(
            f64::INFINITY,
            0.0,
            0.0
        )));
        assert!(!player_position_is_valid(DVec3::new(
            f64::NEG_INFINITY,
            0.0,
            0.0
        )));
        assert!(player_position_is_valid(DVec3::new(
            29_999_999.999,
            0.0,
            -29_999_999.999
        )));
        assert!(!player_position_is_valid(DVec3::new(
            30_000_000.001,
            0.0,
            -30_000_000.001
        )));
    }

    #[test]
    fn local_player_tick_wires_tracker_boat_collision_into_predictor() {
        let (mut session, _to_game, mut to_net) = rig();
        session.phase = Phase::Playing;
        session.state.local.entity_id = 3;
        session.predictor = Some(Predictor::new(PhysicsState::new_player(DVec3::new(
            0.0, 64.0, 0.0,
        ))));
        {
            let predictor = session.predictor.as_mut().expect("test predictor");
            // Keep the player airborne so the 0.6-high boat cannot be stepped over; this
            // isolates the entity candidate from the separate vanilla step-up branch.
            predictor.state.on_ground = false;
            predictor.state.vel = DVec3::new(0.0, 0.0, 3.0);
        }

        session.state.entities.spawn(
            &mut session.state.world,
            42,
            TrackedKind::Type(mc_data::EntityType::Boat),
            None,
            [0, 64 * 32, 2 * 32],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::entity::Metadata::new(),
        );

        let candidates = session.state.entities.collision_candidates(
            &session.state.world,
            session.state.local.entity_id,
            &session.state.players,
        );
        assert_eq!(
            candidates.len(),
            1,
            "the test boat reaches the session query"
        );
        assert_eq!(candidates[0].collision_box, Some(candidates[0].query_box));
        assert_eq!(candidates[0].query_box.min.z, 1.25);
        assert_eq!(candidates[0].query_box.min.y, 64.0);

        let before = session.predictor.as_ref().expect("test predictor").state;
        let mut block_only = before;
        mc_physics::tick_player(&session.state.world, &mut block_only, &MoveInput::default());
        session.tick_player(&MoveInput::default());
        let with_tracker = session.predictor.as_ref().expect("test predictor").state;

        assert_eq!(
            with_tracker.aabb().max.z,
            1.25,
            "the boat's own AABB clips the local player's forward sweep"
        );
        assert!(
            block_only.aabb().max.z > 2.0,
            "the empty-candidate control reaches the requested displacement"
        );
        let _ = drain_outbound(&mut to_net);
    }

    /// A session that has joined, been placed at `(8.5, 70, 8.5)` and been sent chunk `(0, 0)`
    /// with a stone floor at `y = 64` — i.e. one that will actually simulate the player.
    fn joined_and_standing() -> (
        Session,
        tokio::sync::mpsc::Sender<ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (mut session, to_game, to_net) = rig();
        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: 8.5,
                    y: 70.0,
                    z: 8.5,
                    yaw: 0.0,
                    pitch: 0.0,
                    flags: 0,
                },
            )))
            .unwrap();
        // Four columns of floor, not one: these tests walk for hundreds of ticks and need real
        // collision rather than the client provider's blank EmptyChunk fallback.
        for cx in 0..2 {
            for cz in 0..2 {
                to_game
                    .try_send(ToGame::Packet(chunk_with_a_stone_floor(cx, cz, 64)))
                    .unwrap();
                session.poll(8);
            }
        }
        session.poll(64);
        // java: the player was spawned at JoinGame into an unsent column, so the first world
        // tick skips it (`addedToChunk`); the columns arriving later change nothing.
        assert!(!session.player_is_simulated());
        session.tick(&MoveInput::default());
        assert!(session.player_is_simulated());
        (session, to_game, to_net)
    }

    /// The yaw every click aims with is `rotationYawHead`, which
    /// `EntityPlayer.updateEntityActionState` (`EntityPlayer.java:590`) copies from `rotationYaw`
    /// once per tick, and only while `!isMovementBlocked()` (`EntityLivingBase.java:1991-2003`).
    /// A mouse turn between ticks moves the look but not the pick's yaw; the pitch is current.
    /// Negative control: the old per-tick render sync copied the yaw on a dead player's tick too.
    #[test]
    fn the_click_pick_yaw_is_the_head_yaw_copied_only_by_unblocked_ticks() {
        let (mut session, _to_game, _to_net) = joined_and_standing();
        let input = |yaw: f32, pitch: f32| MoveInput {
            yaw,
            pitch,
            ..MoveInput::default()
        };
        session.tick(&input(30.0, 10.0));
        assert_eq!(session.state().local.head_yaw, 30.0);

        // A frame of mouse motion after that tick turns the predictor, not the head.
        {
            let predictor = session.predictor_mut().expect("joined");
            predictor.state.yaw = 50.0;
            predictor.state.pitch = -20.0;
        }
        let (eye, yaw, pitch) = session.tick_pick_ray().expect("a local pick ray");
        assert_eq!(
            yaw, 30.0,
            "getLook(1.0F) aims with rotationYawHead, last tick's yaw"
        );
        assert_eq!(pitch, -20.0, "getLook(1.0F) uses the current rotationPitch");
        let feet = session.predictor().expect("joined").state.pos;
        assert_eq!(
            eye,
            DVec3::new(feet.x, feet.y + f64::from(1.62_f32), feet.z),
            "getPositionEyes(1.0F) reads posX/posY/posZ, with no partial-tick lerp"
        );

        // The next tick reports the turned yaw and copies it into the head.
        session.tick(&input(50.0, -20.0));
        assert_eq!(session.state().local.head_yaw, 50.0);

        // Dead: `isMovementBlocked()`. The tick itself still runs, its AI branch does not.
        session.state_mut().local.health = 0.0;
        session.tick(&input(70.0, 0.0));
        assert_eq!(session.predictor().expect("joined").state.yaw, 70.0);
        assert_eq!(
            session.state().local.head_yaw,
            50.0,
            "a movement-blocked tick must not copy rotationYaw into rotationYawHead"
        );
        assert_eq!(session.tick_pick_ray().expect("still local").1, 50.0);
    }

    /// java: `EntityLivingBase.onDeathUpdate` calls `setDead()` on the tick `deathTime` reaches
    /// 20, and `World.updateEntities` then stops ticking the removed `EntityPlayerSP`. A dead
    /// client therefore sends exactly twenty movement packets after the zero-health S06 (the
    /// last one on the `setDead` tick itself) and then none at all, however long the death
    /// screen stays open, until S07 builds a new player that reports from its first tick. The
    /// live driver advances `deathTime` right after `Session::tick`; this test does the same.
    #[test]
    fn a_dead_player_reports_twenty_ticks_then_goes_silent_until_respawn() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        session.tick(&MoveInput::default());
        let _ = drain_outbound(&mut to_net);

        for health in [20.0, 0.0] {
            to_game
                .try_send(ToGame::Packet(S2CPlay::UpdateHealth(s2c::UpdateHealth {
                    health,
                    food_level: 20,
                    saturation_level: 5.0,
                })))
                .unwrap();
        }
        session.poll(16);
        assert_eq!(session.state().local.health, 0.0);

        let mut reporting_ticks = Vec::new();
        for tick in 0..60 {
            session.tick(&MoveInput::default());
            session.state_mut().local.tick_living_state();
            if drain_outbound(&mut to_net).iter().any(is_movement) {
                reporting_ticks.push(tick);
            }
        }
        assert_eq!(
            reporting_ticks,
            (0..20).collect::<Vec<_>>(),
            "a dying player reports on deathTime 1..=20 and is then removed from the world"
        );
        assert!(!session.player_is_simulated());

        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert!(
            session.player_is_simulated(),
            "S07 builds a live replacement player"
        );
        session.tick(&MoveInput::default());
        assert!(
            drain_outbound(&mut to_net).iter().any(is_movement),
            "the replacement player reports again from its first tick"
        );
    }

    #[test]
    fn local_world_border_damage_uses_strict_buffer_boundary_and_minimum_one() {
        let (mut session, _to_game, mut to_net) = joined_and_standing();
        let _ = drain_outbound(&mut to_net);
        session.state_mut().env.border = BorderState {
            old_diameter: 10.0,
            new_diameter: 10.0,
            ..BorderState::default()
        };
        session.state_mut().local.has_valid_health = true;
        session.state_mut().local.health = 20.0;

        // The east edge is x=5 and the default five-block buffer makes x=10 the exact no-damage
        // boundary. The player's AABB is outside, so this proves the negative-distance branch is
        // gated by `closest + buffer < 0`, not merely by `contains(AABB) == false`.
        session
            .predictor
            .as_mut()
            .expect("joined predictor")
            .state
            .set_position(DVec3::new(10.0, 70.0, 0.5));
        session.tick(&MoveInput::default());
        assert_eq!(session.state().local.health, 20.0);

        // One epsilon farther out is still less than one point of raw damage, so Java's
        // `Math.max(1, floor(...))` supplies the minimum one-health hit.
        session
            .predictor
            .as_mut()
            .expect("joined predictor")
            .state
            .set_position(DVec3::new(10.000_001, 70.0, 0.5));
        session.tick(&MoveInput::default());
        assert_eq!(session.state().local.health, 19.0);
        assert_eq!(session.state().local.last_damage, 1.0);
        assert_eq!(session.state().local.hurt_resistant_time, 20);
        assert_eq!(session.state().local.hurt_time, 10);
    }

    #[test]
    fn local_world_border_damage_respects_hurt_resistance_death_and_s06_correction() {
        let (mut session, _to_game, mut to_net) = joined_and_standing();
        let _ = drain_outbound(&mut to_net);
        session.state_mut().env.border = BorderState {
            old_diameter: 10.0,
            new_diameter: 10.0,
            ..BorderState::default()
        };
        session.state_mut().local.has_valid_health = true;
        session.state_mut().local.health = 20.0;
        session
            .predictor
            .as_mut()
            .expect("joined predictor")
            .state
            .set_position(DVec3::new(11.0, 70.0, 0.5));

        session.tick(&MoveInput::default());
        assert_eq!(session.state().local.health, 19.0);
        // The same one-point attack is ignored while Java's 20-tick hurt-resistance timer is
        // above half. `Session::tick` intentionally leaves this living tail to the driver's
        // post-tick boundary, so advance it explicitly here.
        session.state_mut().local.tick_living_state();
        session.tick(&MoveInput::default());
        assert_eq!(session.state().local.health, 19.0);

        // S06 is authoritative and repairs a client-only prediction without generating a local
        // packet. A server that says the player is back at 20 therefore clears the visual hit.
        session.state_mut().local.apply_server_health(20.0);
        assert_eq!(session.state().local.health, 20.0);
        assert_eq!(session.state().local.hurt_resistant_time, 10);

        // A sufficiently deep point computes floor(20 * 0.2) == 20, reaches zero, and the
        // driver's following living tail starts (but does not skip) the death timer.
        session
            .predictor
            .as_mut()
            .expect("joined predictor")
            .state
            .set_position(DVec3::new(110.0, 70.0, 0.5));
        session.state_mut().local.hurt_resistant_time = 0;
        session.tick(&MoveInput::default());
        assert_eq!(session.state().local.health, 0.0);
        session.state_mut().local.tick_living_state();
        assert_eq!(session.state().local.death_time, 1);
        assert_eq!(session.state().local.hurt_time, 9);
    }

    /// Local S04/S14/S18/S19 packets must cross the live dispatch/session seam even though the
    /// local player has no remote ECS generation.  This deliberately checks the packet target
    /// before and after a real tick, then puts the camera on a remote generation to ensure the
    /// local predictor still receives the target while movement reporting is view-gated.
    #[test]
    fn local_entity_packets_reconcile_through_session_without_entering_remote_tracker() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _ = drain_outbound(&mut to_net);
        let local_id = session.state().local.entity_id;

        // S14 adds to Entity.serverPosX/Y/Z, not to the floating-point position.  Seed the
        // packet-only integers to the live S08 position just as a server-originated local stream
        // would; the predictor remains the sole position authority.
        session.state_mut().local.server_pos = [8 * 32 + 16, 70 * 32, 8 * 32 + 16];
        let helmet = mc_protocol::ItemStack {
            item_id: 310,
            count: 1,
            damage: 0,
            nbt: None,
        };
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityEquipment(
                s2c::EntityEquipment {
                    entity_id: local_id,
                    equipment_slot: 4,
                    item: Some(helmet.clone()),
                },
            )))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityRelMove(s2c::EntityRelMove {
                entity_id: local_id,
                delta_x: 16,
                delta_y: 0,
                delta_z: -8,
                on_ground: true,
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityHeadLook(
                s2c::EntityHeadLook {
                    entity_id: local_id,
                    yaw: mc_protocol::Angle::from_degrees(-90.0),
                },
            )))
            .unwrap();
        let events = session.poll(16);
        assert!(events.iter().any(|event| matches!(
            event,
            ClientEvent::EntityEquipment {
                network_id,
                slot: 4,
                ..
            } if *network_id == local_id
        )));
        assert_eq!(session.state().local.equipment[4], Some(helmet));
        assert_eq!(session.state().local.server_pos, [288, 2240, 264]);
        assert_eq!(
            session.state().local.target_pos,
            DVec3::new(9.0, 70.0, 8.25)
        );
        assert_eq!(session.state().local.interpolation_steps, 3);
        assert!(session.state().local.on_ground);
        assert_eq!(session.state().local.head_yaw, -90.0);
        assert_eq!(
            session.local_render_state().head_yaw,
            -90.0,
            "S19 is visible immediately, before the next EntityPlayerSP tick resets head-to-body"
        );
        assert!(
            session.state().entities.get(local_id).is_none(),
            "the local packet path must not manufacture a remote ECS generation"
        );

        let before_tick = session.predictor().unwrap().state.pos;
        session.tick(&MoveInput::default());
        let after_tick = session.predictor().unwrap().state.pos;
        assert_eq!(session.state().local.interpolation_steps, 2);
        assert!(
            (after_tick.x - 9.0).abs() < (before_tick.x - 9.0).abs(),
            "the first Java three-tick interpolation step moves toward S14's target"
        );
        assert!(
            (after_tick.z - 8.25).abs() < (before_tick.z - 8.25).abs(),
            "the first Java three-tick interpolation step moves Z toward S14's target"
        );

        // S18 replaces the integer shadow and installs a fresh three-tick target.  A following
        // tick consumes one step on the same predictor rather than teleporting the local player.
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityTeleport(
                s2c::EntityTeleport {
                    entity_id: local_id,
                    x: 10 * 32,
                    y: 70 * 32,
                    z: 9 * 32,
                    yaw: mc_protocol::Angle::from_degrees(90.0),
                    pitch: mc_protocol::Angle::from_degrees(12.0),
                    on_ground: false,
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.server_pos, [320, 2240, 288]);
        assert_eq!(
            session.state().local.target_pos,
            DVec3::new(10.0, 70.0, 9.0)
        );
        assert!(!session.state().local.on_ground);
        let before_teleport_tick = session.predictor().unwrap().state.pos;
        session.tick(&MoveInput::default());
        let after_teleport_tick = session.predictor().unwrap().state.pos;
        assert_eq!(session.state().local.interpolation_steps, 2);
        assert!((after_teleport_tick.x - 10.0).abs() < (before_teleport_tick.x - 10.0).abs());
        assert!((after_teleport_tick.z - 9.0).abs() < (before_teleport_tick.z - 9.0).abs());
        let _ = drain_outbound(&mut to_net);

        // A remote target still goes through EntityTracker, while an unknown target remains a
        // no-op.  Neither packet can overwrite the local target just installed above.
        let remote_id = 42;
        spawn_test_pig(&mut session, remote_id);
        let local_before_remote = session.state().local.clone();
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityRelMove(s2c::EntityRelMove {
                entity_id: remote_id,
                delta_x: 3,
                delta_y: 0,
                delta_z: 0,
                on_ground: true,
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityTeleport(
                s2c::EntityTeleport {
                    entity_id: 404,
                    x: 77 * 32,
                    y: 88 * 32,
                    z: 99 * 32,
                    yaw: mc_protocol::Angle::from_degrees(20.0),
                    pitch: mc_protocol::Angle::from_degrees(-10.0),
                    on_ground: true,
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(
            session.state().entities.get(remote_id).unwrap().server_pos[0],
            8 * 32 + 3
        );
        assert!(session.state().entities.get(remote_id).unwrap().on_ground);
        assert_eq!(session.state().local, local_before_remote);

        // A foreign camera suppresses the local walking packet, but does not suppress local
        // EntityPlayerSP interpolation or the local S19 render edge.
        session.set_render_view_entity(remote_id);
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityLook(s2c::EntityLook {
                entity_id: local_id,
                yaw: mc_protocol::Angle::from_degrees(135.0),
                pitch: mc_protocol::Angle::from_degrees(-8.0),
                on_ground: true,
            })))
            .unwrap();
        session.poll(16);
        assert!((session.state().local.target_yaw - 135.0).abs() < 1.0);
        session.tick(&MoveInput::default());
        let foreign_camera_packets = drain_outbound(&mut to_net);
        assert!(
            !foreign_camera_packets.iter().any(is_movement),
            "foreign camera gates only local walking reports, not local packet reconciliation"
        );
    }

    #[test]
    fn local_entity_lifecycle_tracks_first_update_fire_and_remote_void_semantics() {
        let (mut session, _to_game, _to_net) = joined_and_standing();
        let position = DVec3::new(8.5, 70.0, 8.5);
        let fire = mc_data::BlockState::new(mc_data::block_by_name("fire").expect("fire"), 0);
        session.state.world.set_block(BlockPos::new(8, 70, 8), fire);
        {
            let predictor = session.predictor.as_mut().expect("joined predictor");
            predictor.state.set_position(position);
            predictor.state.in_water = false;
            predictor.state.first_update = true;
        }

        // The local base tick clears the private counter, while a dry flammable body increments
        // it once after movement. This is the one-tick flame source before S1C metadata catches
        // up, and the sentinel is the player's Entity.fireResistance (20), not zero.
        session.local_fire = 0;
        session.advance_local_fire_tail(false);
        assert_eq!(session.local_fire, 1);
        assert!(session
            .state
            .entities
            .take_status_effect_events()
            .is_empty());

        // A wet local body does not increment the private counter. The inherited Java
        // `EntityLivingBase.onEntityUpdate` already extinguishes any remote-world lava seed
        // before this tail, so no local `random.fizz` event is fabricated here.
        session.local_fire = 0;
        session.predictor.as_mut().unwrap().state.in_water = true;
        session.advance_local_fire_tail(false);
        assert_eq!(session.local_fire, 0);

        // `EntityPlayerSP` still clears firstUpdate on the mounted/non-mounted base path, while
        // its EntityLivingBase kill() is a remote-world damage no-op: Y below -64 does not close
        // the horizontal simulation gate or suppress the next movement packet.
        session
            .state
            .world
            .set_block(BlockPos::new(8, 70, 8), mc_data::BlockState::AIR);
        {
            let predictor = session.predictor.as_mut().unwrap();
            predictor.state.in_water = false;
            predictor.state.first_update = true;
            predictor.state.set_position(DVec3::new(8.5, -65.0, 8.5));
        }
        assert!(session.player_is_simulated());
        assert!(session.tick(&MoveInput::default()).is_some());
        assert!(!session.predictor.as_ref().unwrap().state.first_update);
    }

    /// The local `EntityPlayerSP` shares the base running-dust recipe with remote entities but
    /// keeps its own persistent `Entity.rand` and previous-water latch in `Session`. The gate is
    /// only `isSprinting() && !isInWater()` (`Entity.java:1165-1171`): no chance draw, and no
    /// motion threshold, so a sprinting player emits on every call even while standing still.
    #[test]
    fn local_sprint_running_dust_enters_the_ordered_entity_particle_queue() {
        let (mut session, _to_game, _to_net) = joined_and_standing();
        let position = DVec3::new(8.5, 70.0, 8.5);
        let velocity = DVec3::new(0.1, 0.0, -0.2);
        let underfoot = BlockPos::new(8, 69, 8);
        let stone = mc_data::BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
        session.state.world.set_block(underfoot, stone);
        session.local_particle_random = JavaRandom::new(0x10CA1);
        let mut oracle = JavaRandom::new(0x10CA1);
        {
            let predictor = session.predictor.as_mut().unwrap();
            predictor.state.pos = position;
            predictor.state.vel = velocity;
            predictor.state.sprinting = true;
            predictor.state.in_water = false;
        }
        let width = f64::from(session.predictor.as_ref().unwrap().state.width);
        for velocity in [velocity, DVec3::ZERO] {
            session.predictor.as_mut().unwrap().state.vel = velocity;
            session.queue_local_running_particle();
            let events = session.state.entities.take_particle_events();
            assert_eq!(
                events,
                vec![crate::sync::entities::EntityParticleEvent {
                    particle_id: mc_render::particle::packet::BLOCK_CRACK,
                    position: DVec3::new(
                        position.x + (f64::from(oracle.next_f32()) - 0.5) * width,
                        position.y + 0.1,
                        position.z + (f64::from(oracle.next_f32()) - 0.5) * width,
                    ),
                    velocity: DVec3::new(-velocity.x * 4.0, 1.5, -velocity.z * 4.0),
                    args: EntityParticleArgs::block_state(stone.to_java_state_id()),
                }],
                "velocity {velocity:?}"
            );
            assert_eq!(session.local_particle_random.raw_seed(), oracle.raw_seed());
        }

        // The inherited hook samples the previous water result before its offset draws.
        session.local_running_in_water = true;
        let before = session.local_particle_random.raw_seed();
        session.queue_local_running_particle();
        assert!(session.state.entities.take_particle_events().is_empty());
        assert_eq!(session.local_particle_random.raw_seed(), before);
    }

    #[test]
    fn camera_changed_is_generation_safe_and_unknown_ids_keep_prior_view() {
        let (mut session, _to_game, _to_net) = joined_and_standing();
        let remote = {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                77,
                TrackedKind::Type(mc_data::EntityType::Pig),
                None,
                [0, 64 * 32, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            )
        };
        session.set_render_view_entity(77);
        assert!(matches!(
            session.render_view_target(),
            RenderViewTarget::Entity {
                network_id: 77,
                entity,
                ..
            } if entity == remote
        ));

        // Java's null lookup is a no-op: a bad S43 cannot snap the camera back to the player.
        session.set_render_view_entity(9999);
        assert!(matches!(
            session.render_view_target(),
            RenderViewTarget::Entity { network_id: 77, .. }
        ));

        // Reusing the id creates a new ECS object. The old object reference must not survive.
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                77,
                TrackedKind::Type(mc_data::EntityType::Pig),
                None,
                [0, 64 * 32, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        assert_eq!(session.render_view_target(), RenderViewTarget::Local);
    }

    #[test]
    fn foreign_camera_still_reports_c0b_edges_but_suppresses_movement_packets() {
        let (mut session, _to_game, mut to_net) = joined_and_standing();
        let _ = drain_outbound(&mut to_net);
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                77,
                TrackedKind::Type(mc_data::EntityType::Pig),
                None,
                [0, 64 * 32, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        session.set_render_view_entity(77);
        session.tick(&MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        });
        let packets = drain_outbound(&mut to_net);
        assert!(packets.iter().any(|packet| matches!(
            packet,
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSprinting,
                ..
            })
        )));
        assert!(!packets.iter().any(is_movement));
    }

    /// `EntityPlayerSP.updateEntityActionState` is current-view gated, while the rest of
    /// `onLivingUpdate` still travels the local body and evaluates the flight toggle. This is
    /// the session-level seam: a remote camera must not replace the body's persistent action
    /// fields or apply view-local vertical thrust, but it must still move and emit C13 edges.
    #[test]
    fn foreign_camera_travels_old_actions_and_toggles_flight_without_thrust() {
        let (mut session, _to_game, mut to_net) = joined_and_standing();
        let _ = drain_outbound(&mut to_net);
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                77,
                TrackedKind::Type(mc_data::EntityType::Pig),
                None,
                [0, 64 * 32, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        session.set_render_view_entity(77);
        {
            let predictor = session.predictor_mut().expect("joined predictor");
            predictor.state.move_strafing = 0.25;
            predictor.state.move_forward = 1.0;
            predictor.state.jumping = false;
            predictor.state.prev_move_forward = 0.0;
            predictor.state.prev_jump = false;
            predictor.state.vel = DVec3::ZERO;
            predictor.state.yaw = 0.0;
        }
        let before = session.predictor().expect("joined predictor").state;

        // The new key is backward + jump, but Java's foreign-camera body keeps the previous
        // EntityLivingBase action fields and nevertheless runs travel with those old values.
        session.tick(&MoveInput {
            back: true,
            jump: true,
            ..MoveInput::default()
        });
        let after_travel = session.predictor().expect("joined predictor").state;
        // Not replaced by the foreign tick's keys, but damped in place by travel's
        // `moveStrafing *= 0.98F; moveForward *= 0.98F`.
        assert_eq!(after_travel.move_strafing, before.move_strafing * 0.98);
        assert_eq!(after_travel.move_forward, before.move_forward * 0.98);
        assert!(!after_travel.jumping);
        assert!(
            after_travel.pos.z > before.pos.z,
            "foreign camera still runs travel with the prior forward action: {before:?} -> {after_travel:?}"
        );
        assert_eq!(after_travel.prev_move_forward, -1.0);
        assert!(after_travel.prev_jump, "raw input edges still refresh");
        assert!(!drain_outbound(&mut to_net).iter().any(is_movement));

        // The second jump edge toggles capabilities even though the view is foreign; only the
        // local-view thrust branch is gated, so a zero vertical velocity must stay zero.
        session.state_mut().local.abilities = crate::sync::Abilities::from_flags(0x04, 0.05, 0.1);
        {
            let predictor = session.predictor_mut().expect("joined predictor");
            predictor.state.allow_flying = true;
            predictor.state.flying = false;
            predictor.state.fly_toggle_timer = 1;
            predictor.state.prev_jump = false;
            predictor.state.vel.y = 0.0;
            predictor.state.on_ground = false;
        }
        session.tick(&MoveInput {
            jump: true,
            ..MoveInput::default()
        });
        let after_flight = session.predictor().expect("joined predictor").state;
        assert!(after_flight.flying);
        assert_eq!(after_flight.vel.y, 0.0);
        let flight_packets = drain_outbound(&mut to_net);
        assert!(matches!(
            flight_packets.as_slice(),
            [C2SPlay::PlayerAbilities(c2s::PlayerAbilities {
                flying: true,
                ..
            })]
        ));
    }

    /// F-P1-client-01-12 / F-P2-client.gui-18-1/-2: the two clientbound packets own the
    /// local sleep lifecycle. This test goes through dispatch rather than calling the helpers,
    /// so it also proves the local-id guards and the driver-readable state seam.
    #[test]
    fn local_bed_and_wake_packets_drive_pose_timer_input_block_and_fade_out() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        let bed = BlockPos::new(8, 65, 8);
        // East is horizontal metadata 3. Block-state packets use `block << 4 | meta`.
        to_game
            .try_send(ToGame::Packet(S2CPlay::BlockChange(s2c::BlockChange {
                position: bed,
                block_state_id: (26 << 4) | 3,
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::UseBed(s2c::UseBed {
                player_id: 3,
                bed_position: bed,
            })))
            .unwrap();
        session.poll(16);

        assert!(session.state().local.sleeping);
        assert_eq!(session.state().local.sleep_timer, 0);
        let sleeping = session.predictor().unwrap().state;
        assert_eq!(sleeping.width, 0.2);
        assert_eq!(sleeping.height, 0.2);
        assert_eq!(sleeping.pos, DVec3::new(8.9f32 as f64, 65.6875, 8.5));
        assert_eq!(sleeping.vel, DVec3::ZERO);

        // A packet naming somebody else cannot wake us. This guard catches the tempting but
        // incorrect `animation == 2`-only match arm.
        to_game
            .try_send(ToGame::Packet(S2CPlay::Animation(s2c::Animation {
                entity_id: 99,
                animation_type: 2,
            })))
            .unwrap();
        session.poll(16);
        assert!(session.state().local.sleeping);

        let held_keys = MoveInput {
            forward: true,
            jump: true,
            sneak: true,
            ..MoveInput::default()
        };
        for _ in 0..100 {
            session.tick(&held_keys);
        }
        assert_eq!(session.state().local.sleep_timer, 100);
        let after = session.predictor().unwrap().state;
        assert_eq!(after.pos.x, sleeping.pos.x, "sleep blocks forward input");
        assert_eq!(after.pos.z, sleeping.pos.z, "sleep blocks strafe input");

        let sleep_packets = drain_outbound(&mut to_net);
        assert!(
            sleep_packets.iter().any(is_movement),
            "sleep does not suppress onUpdateWalkingPlayer"
        );
        assert!(
            sleep_packets
                .iter()
                .any(|packet| matches!(packet, C2SPlay::Player(_))),
            "a stable sleeper still sends vanilla's twenty-tick C03 heartbeat"
        );
        assert!(
            !sleep_packets.iter().any(|packet| matches!(
                packet,
                C2SPlay::EntityAction(c2s::EntityAction {
                    action: c2s::EntityActionType::StartSneaking,
                    ..
                })
            )),
            "EntityPlayerSP.isSneaking is false while sleeping"
        );

        to_game
            .try_send(ToGame::Packet(S2CPlay::Animation(s2c::Animation {
                entity_id: 3,
                animation_type: 2,
            })))
            .unwrap();
        session.poll(16);
        assert!(!session.state().local.sleeping);
        assert_eq!(session.state().local.sleep_timer, 100);
        let awake = session.predictor().unwrap().state;
        assert_eq!(awake.width, PLAYER_WIDTH);
        assert_eq!(awake.height, PLAYER_HEIGHT);
        // East-facing getSafeExitLocation starts at x/z 7/7; the stone floor makes it valid.
        assert_eq!(awake.pos, DVec3::new(7.5, 65.1f32 as f64, 7.5));

        for expected in 101..=109 {
            session.tick(&MoveInput::default());
            assert_eq!(session.state().local.sleep_timer, expected);
        }
        session.tick(&MoveInput::default());
        assert_eq!(session.state().local.sleep_timer, 0);
    }

    /// `S0APacketUseBed` names the entity that accepted the bed. A remote player's packet must
    /// remain in the event stream for remote consumers, but it cannot mutate this session's
    /// local `EntityPlayerSP` pose or sleeping predicate.
    #[test]
    fn foreign_use_bed_packet_cannot_enter_local_sleep() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        let before = session.predictor().expect("joined predictor").state;

        to_game
            .try_send(ToGame::Packet(S2CPlay::UseBed(s2c::UseBed {
                player_id: 99,
                bed_position: BlockPos::new(8, 65, 8),
            })))
            .unwrap();
        session.poll(16);

        assert!(!session.state().local.sleeping);
        assert_eq!(session.state().local.sleep_timer, 0);
        let after = session.predictor().expect("joined predictor").state;
        assert_eq!(after.pos, before.pos, "foreign S0A moved the local player");
        assert_eq!(after.width, before.width);
        assert_eq!(after.height, before.height);
    }

    /// **F-P0-client.entity-01-10.** `EntityPlayerSP.isSneaking()` is the raw sneak key **and**
    /// `!sleeping`, while `EntityPlayer.isMovementBlocked()` includes sleeping in addition to
    /// dead-player blocking. This drives both predicates through the local S0A/S23 state seam:
    /// an already-sneaking player reports STOP while entering bed, cannot move with held keys,
    /// ignores another entity's wake animation, and resumes both movement and START_SNEAKING
    /// after the local wake animation.
    #[test]
    fn sleep_gates_sneak_and_movement_but_awake_input_still_acts() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);

        // Establish the server-side sneak edge before the bed transition. The later sleeping
        // tick must therefore emit STOP_SNEAKING, rather than merely avoiding START_SNEAKING.
        session.tick(&MoveInput {
            sneak: true,
            ..MoveInput::default()
        });
        let awake_sneak = drain_outbound(&mut to_net);
        assert!(awake_sneak.iter().any(|packet| matches!(
            packet,
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSneaking,
                aux_data: 0,
                ..
            })
        )));

        let bed = BlockPos::new(8, 65, 8);
        to_game
            .try_send(ToGame::Packet(S2CPlay::BlockChange(s2c::BlockChange {
                position: bed,
                block_state_id: (26 << 4) | 3,
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::UseBed(s2c::UseBed {
                player_id: 3,
                bed_position: bed,
            })))
            .unwrap();
        session.poll(16);
        let sleeping_pos = session.predictor().expect("joined predictor").state.pos;

        let held = MoveInput {
            forward: true,
            jump: true,
            sneak: true,
            ..MoveInput::default()
        };
        session.tick(&held);
        let sleeping_packets = drain_outbound(&mut to_net);
        assert!(sleeping_packets.iter().any(|packet| matches!(
            packet,
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StopSneaking,
                aux_data: 0,
                ..
            })
        )));
        assert!(
            !sleeping_packets.iter().any(|packet| matches!(
                packet,
                C2SPlay::EntityAction(c2s::EntityAction {
                    action: c2s::EntityActionType::StartSneaking,
                    ..
                })
            )),
            "sleeping isSneaking is false even while the key is held: {sleeping_packets:?}"
        );
        assert_eq!(
            session.predictor().expect("joined predictor").state.pos,
            sleeping_pos,
            "isMovementBlocked includes sleeping and suppresses held movement"
        );
        assert!(
            session
                .predictor()
                .expect("joined predictor")
                .state
                .movement_blocked
        );

        // A wake animation naming another entity must not clear the local sleep predicate.
        to_game
            .try_send(ToGame::Packet(S2CPlay::Animation(s2c::Animation {
                entity_id: 99,
                animation_type: 2,
            })))
            .unwrap();
        session.poll(16);
        assert!(session.state().local.sleeping);

        // The local animation clears sleeping and restores ordinary movement input. The next
        // tick proves both that movement resumes and that the held sneak key starts again.
        to_game
            .try_send(ToGame::Packet(S2CPlay::Animation(s2c::Animation {
                entity_id: 3,
                animation_type: 2,
            })))
            .unwrap();
        session.poll(16);
        let awake_pos = session.predictor().expect("joined predictor").state.pos;
        session.tick(&held);
        let awake_packets = drain_outbound(&mut to_net);
        assert!(awake_packets.iter().any(|packet| matches!(
            packet,
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSneaking,
                aux_data: 0,
                ..
            })
        )));
        let moved = session.predictor().expect("joined predictor").state;
        assert!(!moved.movement_blocked);
        assert!(
            moved.pos.z > awake_pos.z,
            "awake forward input resumes movement: before={awake_pos:?}, after={:?}",
            moved.pos
        );
    }

    #[test]
    fn bed_metadata_facing_is_vanillas_south_west_north_east_order() {
        let mut world = World::new();
        let bed = BlockPos::new(1, 2, 3);
        world.load_empty_chunk(bed.chunk_pos());
        for (meta, expected) in [
            (0, Facing::South),
            (1, Facing::West),
            (2, Facing::North),
            (3, Facing::East),
        ] {
            world.set_block(bed, mc_data::BlockState::from_raw((26 << 4) | meta));
            assert_eq!(bed_facing(&world, bed), expected, "metadata {meta}");
        }
    }

    /// `BlockBed.getSafeExitLocation` scans the 3x3 ring nearest the bed first, then the
    /// facing-offset second ring. Every horizontal facing gets a candidate exclusive to its
    /// second ring here, proving the low-two-bit orientation is used to move the search window.
    #[test]
    fn safe_bed_exit_scans_both_orientation_rings_in_java_order() {
        let bed = BlockPos::new(8, 64, 8);
        let cases = [
            (0_u8, BlockPos::new(7, 64, 6)),  // SOUTH: second ring is toward -Z
            (1_u8, BlockPos::new(10, 64, 7)), // WEST: second ring is toward +X
            (2_u8, BlockPos::new(7, 64, 10)), // NORTH: second ring is toward +Z
            (3_u8, BlockPos::new(6, 64, 7)),  // EAST: second ring is toward -X
        ];
        for (meta, expected) in cases {
            let mut world = World::new();
            world.load_empty_chunk(bed.chunk_pos());
            world.set_block(bed, mc_data::BlockState::new(mc_data::BlockId(26), meta));
            world.set_block(
                expected.down(),
                mc_data::BlockState::new(mc_data::BlockId(1), 0),
            );
            assert_eq!(
                safe_bed_exit(&world, bed),
                Some(expected),
                "bed metadata {meta} must reach its oriented second ring"
            );
        }
    }

    /// `BlockBed.hasRoomForPlayer` requires a solid-top floor, non-solid feet, and non-solid
    /// headroom. The exit search skips an occupied first candidate, and reports `None` when every
    /// candidate is blocked; the wake caller then owns the Java `bed.up()` fallback.
    #[test]
    fn safe_bed_exit_requires_floor_feet_and_headroom_and_skips_occupied_candidates() {
        let bed = BlockPos::new(8, 64, 8);
        let first = BlockPos::new(7, 64, 7);
        let second = BlockPos::new(7, 64, 8);
        let stone = mc_data::BlockState::new(mc_data::BlockId(1), 0);
        let glass = mc_data::BlockState::new(mc_data::BlockId(20), 0);

        let mut valid = World::new();
        valid.load_empty_chunk(bed.chunk_pos());
        valid.set_block(bed, mc_data::BlockState::new(mc_data::BlockId(26), 0));
        valid.set_block(first.down(), stone);
        assert!(has_room_to_leave_bed(&valid, first));

        let mut no_floor = World::new();
        no_floor.load_empty_chunk(bed.chunk_pos());
        no_floor.set_block(bed, mc_data::BlockState::new(mc_data::BlockId(26), 0));
        no_floor.set_block(first.down(), glass);
        assert!(!has_room_to_leave_bed(&no_floor, first));

        let mut occupied = World::new();
        occupied.load_empty_chunk(bed.chunk_pos());
        occupied.set_block(bed, mc_data::BlockState::new(mc_data::BlockId(26), 0));
        occupied.set_block(first.down(), stone);
        occupied.set_block(first, stone);
        occupied.set_block(second.down(), stone);
        assert_eq!(safe_bed_exit(&occupied, bed), Some(second));

        let mut head_blocked = World::new();
        head_blocked.load_empty_chunk(bed.chunk_pos());
        head_blocked.set_block(bed, mc_data::BlockState::new(mc_data::BlockId(26), 0));
        head_blocked.set_block(first.down(), stone);
        head_blocked.set_block(first.up(), stone);
        assert!(!has_room_to_leave_bed(&head_blocked, first));
        assert_eq!(safe_bed_exit(&head_blocked, bed), None);
    }

    /// F-P0-client.network-03-5: the local player's spectator predicate is its own S38 tab-list
    /// row, not PlayerControllerMP's game mode. The null lookup explicitly means false.
    #[test]
    fn local_spectator_physics_follows_the_local_tab_row_and_not_controller_mode() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        let local_uuid = uuid::Uuid::from_u128(7);
        let other_uuid = uuid::Uuid::from_u128(8);
        session.state_mut().local.uuid = local_uuid;

        let change_controller_mode = |value| {
            S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: s2c::GameStateReason::ChangeGameMode,
                value,
            })
        };
        let add_player = |uuid, username: &str, game_mode| {
            S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid,
                username: username.into(),
                properties: Vec::new(),
                game_mode,
                ping: 0,
                display_name: None,
            }]))
        };

        // Controller says Spectator, but no local NetworkPlayerInfo exists: Java returns false.
        to_game
            .try_send(ToGame::Packet(change_controller_mode(3.0)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.game_mode, s2c::GameMode::Spectator);
        session.push_server_movement_state();
        assert!(!session.predictor().unwrap().state.spectator);
        // EntityPlayerSP's forced-flight/landing branches ask the controller.
        assert!(session.predictor().unwrap().state.spectator_mode);

        // An unrelated spectator row cannot make the local player a spectator.
        to_game
            .try_send(ToGame::Packet(add_player(
                other_uuid,
                "Other",
                s2c::GameMode::Spectator,
            )))
            .unwrap();
        // Nor can merely having a local row whose own mode is Survival.
        to_game
            .try_send(ToGame::Packet(add_player(
                local_uuid,
                "Local",
                s2c::GameMode::Survival,
            )))
            .unwrap();
        session.poll(16);
        session.push_server_movement_state();
        assert!(!session.predictor().unwrap().state.spectator);

        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerListItem(
                s2c::PlayerListItem::UpdateGameMode(vec![s2c::UpdateGameModeEntry {
                    uuid: local_uuid,
                    game_mode: s2c::GameMode::Spectator,
                }]),
            )))
            .unwrap();
        // The inverse mismatch proves the source is the row, not an accidental AND with the
        // controller: local row Spectator remains true when controller mode becomes Survival.
        to_game
            .try_send(ToGame::Packet(change_controller_mode(0.0)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.game_mode, s2c::GameMode::Survival);
        session.push_server_movement_state();
        assert!(session.predictor().unwrap().state.spectator);
        assert!(!session.predictor().unwrap().state.spectator_mode);

        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerListItem(
                s2c::PlayerListItem::RemovePlayer(vec![local_uuid]),
            )))
            .unwrap();
        session.poll(16);
        session.push_server_movement_state();
        assert!(!session.predictor().unwrap().state.spectator);
    }

    /// F-P0-network.play.client-05-1, F-P0-network.play.server-14-1, and
    /// F-P1-entity.player-05-1 are the same defect: flight toggles inside `super.onUpdate`,
    /// before `onUpdateWalkingPlayer` sends its sprint/sneak edges and movement packet. A tick
    /// that performs both transitions must therefore put C13 first on the wire.
    #[test]
    fn a_flight_edge_leads_the_same_ticks_sprint_edge_and_movement_packet() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerAbilities(
                s2c::PlayerAbilities {
                    // Allow flight, but begin on foot.
                    flags: 0x04,
                    fly_speed: 0.05,
                    walk_speed: 0.1,
                },
            )))
            .unwrap();
        session.poll(16);

        // Merely having flight permission is not an edge. The ordinary tick still emits its
        // walking packet, but vanilla does not resend C13 until `isFlying` actually changes.
        let no_edge_returned = session.tick(&MoveInput::default());
        let no_edge_packets = drain_outbound(&mut to_net);
        assert_eq!(
            no_edge_packets.len(),
            1,
            "a stable flight flag emits only movement: {no_edge_packets:?}"
        );
        assert!(is_movement(&no_edge_packets[0]));
        assert_eq!(no_edge_returned.as_ref(), Some(&no_edge_packets[0]));

        // First jump press arms the seven-tick double-tap window. Release creates the next
        // rising edge; neither setup tick is part of the ordering assertion.
        session.tick(&MoveInput {
            jump: true,
            ..MoveInput::default()
        });
        let _ = drain_outbound(&mut to_net);
        session.tick(&MoveInput::default());
        let _ = drain_outbound(&mut to_net);

        let returned = session.tick(&MoveInput {
            jump: true,
            forward: true,
            sprint: true,
            ..MoveInput::default()
        });
        let packets = drain_outbound(&mut to_net);
        assert_eq!(
            packets.len(),
            3,
            "C13, start-sprint C0B, movement: {packets:?}"
        );
        assert!(
            matches!(
                &packets[0],
                C2SPlay::PlayerAbilities(c2s::PlayerAbilities { flying: true, .. })
            ),
            "sendPlayerAbilities runs in onLivingUpdate: {packets:?}"
        );
        assert!(
            matches!(
                &packets[1],
                C2SPlay::EntityAction(c2s::EntityAction {
                    action: c2s::EntityActionType::StartSprinting,
                    ..
                })
            ),
            "onUpdateWalkingPlayer's C0B follows C13: {packets:?}"
        );
        assert!(
            is_movement(&packets[2]),
            "walking packet is last: {packets:?}"
        );
        assert_eq!(returned.as_ref(), Some(&packets[2]));
    }

    /// The toggle and landing-clear calls are separate Java mutation sites.  A blocked player
    /// still runs the flight toggle before `isMovementBlocked()` zeroes the jump input; placing
    /// the player just above the test floor with a downward motion keeps the sweep grounded so
    /// both sites are reached in one `onLivingUpdate`.
    #[test]
    fn same_tick_flight_toggle_and_landing_clear_emit_each_c13_before_movement() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerAbilities(
                s2c::PlayerAbilities {
                    flags: 0x04,
                    fly_speed: 0.05,
                    walk_speed: 0.1,
                },
            )))
            .unwrap();
        session.poll(16);

        // `isMovementBlocked()` is evaluated after the flight toggle and its vertical impulse.
        // Start just above the floor and give the player enough downward motion that the impulse
        // still lands during this sweep; this makes the tail's landing clear reachable.
        session.state_mut().local.health = 0.0;
        let predictor = session.predictor_mut().expect("joined predictor");
        predictor.state.set_position(DVec3::new(8.5, 65.1, 8.5));
        predictor.state.on_ground = true;
        predictor.state.flying = false;
        predictor.state.fly_toggle_timer = 1;
        predictor.state.prev_jump = false;
        predictor.state.vel.y = -0.3;

        let returned = session.tick(&MoveInput {
            jump: true,
            ..MoveInput::default()
        });
        let packets = drain_outbound(&mut to_net);
        assert_eq!(packets.len(), 3, "two C13 edges plus movement: {packets:?}");
        assert!(matches!(
            &packets[0],
            C2SPlay::PlayerAbilities(c2s::PlayerAbilities { flying: true, .. })
        ));
        assert!(matches!(
            &packets[1],
            C2SPlay::PlayerAbilities(c2s::PlayerAbilities { flying: false, .. })
        ));
        assert!(
            is_movement(&packets[2]),
            "walking packet is last: {packets:?}"
        );
        assert_eq!(returned.as_ref(), Some(&packets[2]));
        assert!(!session.state().local.abilities.flying);
        assert!(!session.predictor().expect("joined predictor").state.flying);
    }

    /// A stable flying state is not an ability edge.  This negative control keeps the
    /// same-tick regression from being "fixed" by sending C13 on every movement tick.
    #[test]
    fn stable_flight_tick_does_not_resend_player_abilities() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerAbilities(
                s2c::PlayerAbilities {
                    // Allow flight and begin already flying; neither value changes this tick.
                    flags: 0x0E,
                    fly_speed: 0.05,
                    walk_speed: 0.1,
                },
            )))
            .unwrap();
        session.poll(16);

        let returned = session.tick(&MoveInput::default());
        let packets = drain_outbound(&mut to_net);
        assert_eq!(
            packets.len(),
            1,
            "stable flight emits only movement: {packets:?}"
        );
        assert!(is_movement(&packets[0]));
        assert_eq!(returned.as_ref(), Some(&packets[0]));
        assert!(session.state().local.abilities.flying);
        assert!(session.predictor().expect("joined predictor").state.flying);
    }

    /// **W-SPRINTSTATE / F-P1-entity.item-07-2.** `PlayerControllerMP.attackEntity` puts C02 on
    /// the wire before the client-local minecart damage result clears sprint.  The resulting
    /// C0B STOP therefore leads the next walking packet, rather than being sent before C02.
    #[test]
    fn a_minecart_attack_damps_motion_and_reports_stop_on_the_following_tick() {
        use mc_data::EntityType;

        let (mut session, _to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        let local_id = session.state().local.entity_id;
        // Pretend the prior walking tick already reported START_SPRINTING. The local watcher is
        // the state read by EntityPlayer.attackTargetEntityWithCurrentItem, while OutboundState's
        // cache is the server-side edge that must observe the later STOP.
        session.player_action(|out, packets| {
            out.walking_update(
                local_id,
                mc_physics::prediction::Outbound::Ground { on_ground: true },
                true,
                false,
                packets,
            );
        });
        let _start_packets = drain_outbound(&mut to_net);
        session.set_local_watched_sprinting(true);
        let velocity = DVec3::new(0.75, 0.125, -0.5);
        session.predictor_mut().expect("joined predictor").state.vel = velocity;
        session
            .predictor_mut()
            .expect("joined predictor")
            .state
            .sprinting = true;

        let hit_entity = session.attack_entity_with_knockback(
            91,
            Some(EntityType::MinecartRideable),
            0,
            0,
            None,
        );
        assert!(!hit_entity, "a minecart is not an EntityLivingBase");
        let attack_packets = drain_outbound(&mut to_net);
        assert_eq!(
            attack_packets.len(),
            1,
            "slot 0 needs no C09: {attack_packets:?}"
        );
        assert!(matches!(
            attack_packets[0],
            C2SPlay::UseEntity(c2s::UseEntity {
                entity_id: 91,
                action: c2s::UseEntityAction::Attack,
            })
        ));
        let after_attack = session.predictor().expect("joined predictor").state;
        assert!(!after_attack.sprinting);
        assert_eq!(after_attack.sprinting_ticks_left, 0);
        assert_eq!(after_attack.vel.x, velocity.x * 0.6);
        assert_eq!(after_attack.vel.z, velocity.z * 0.6);
        assert_eq!(after_attack.vel.y, velocity.y);

        // The attack's C02 was already drained above. The next tick is the first opportunity for
        // EntityPlayerSP.onUpdateWalkingPlayer to report the locally-cleared watcher bit.
        session.tick(&MoveInput::default());
        let next_tick = drain_outbound(&mut to_net);
        assert!(
            matches!(
                next_tick.first(),
                Some(C2SPlay::EntityAction(c2s::EntityAction {
                    entity_id,
                    action: c2s::EntityActionType::StopSprinting,
                    ..
                })) if *entity_id == local_id
            ),
            "STOP_SPRINTING must follow (not precede) the attack C02: {next_tick:?}"
        );
        assert!(
            next_tick.get(1).is_some_and(is_movement),
            "the ordinary walking packet follows the STOP edge: {next_tick:?}"
        );
    }

    /// java: `EntityPlayer.attackTargetEntityWithCurrentItem` runs
    /// its local body only `if (f > 0.0F || f1 > 0.0F)`. After an S20 that sets the local
    /// `generic.attackDamage` base to 0 (or below, which `RangedAttribute` clamps to 0), a
    /// sprinting hit on a minecart sends its C02 and nothing else happens locally: no 0.6
    /// damping, no `setSprinting(false)`. Sharpness I (`f1 = 1.25`) or a positive attribute
    /// restores the hit. Negative control: before the gate every case below damped and stopped.
    #[test]
    fn a_zero_attack_damage_attribute_skips_the_local_attack_body() {
        use mc_data::EntityType;

        for (base, sharpness, hit) in [
            (0.0, false, false),
            (-3.0, false, false),
            (0.0, true, true),
            (1.0, false, true),
        ] {
            let (mut session, to_game, mut to_net) = joined_and_standing();
            let local_id = session.state().local.entity_id;
            to_game
                .try_send(ToGame::Packet(S2CPlay::EntityProperties(
                    s2c::EntityProperties {
                        entity_id: local_id,
                        properties: vec![s2c::AttributePropertySnapshot {
                            key: "generic.attackDamage".into(),
                            base_value: base,
                            modifiers: Vec::new(),
                        }],
                    },
                )))
                .unwrap();
            session.poll(8);
            let _ = drain_outbound(&mut to_net);
            session.set_local_watched_sprinting(true);
            let velocity = DVec3::new(0.75, 0.125, -0.5);
            {
                let state = &mut session.predictor_mut().expect("joined predictor").state;
                state.vel = velocity;
                state.sprinting = true;
            }
            let held = sharpness.then(|| enchanted_stack("diamond_sword", &[(16, 1)]));
            session.attack_entity_with_knockback(
                91,
                Some(EntityType::MinecartRideable),
                0,
                0,
                held.as_ref(),
            );
            let sent = drain_outbound(&mut to_net);
            assert_eq!(sent.len(), 1, "base {base}: the C02 alone, got {sent:?}");
            let after = session.predictor().expect("joined predictor").state;
            let label = format!("base {base}, sharpness {sharpness}");
            assert_eq!(after.sprinting, !hit, "{label}");
            let expected_x = if hit { velocity.x * 0.6 } else { velocity.x };
            assert_eq!(after.vel.x, expected_x, "{label}");
        }
    }

    /// An `ench`-list stack for the attack tests, with `{id, lvl}` shorts in list order.
    fn enchanted_stack(name: &str, enchantments: &[(i16, i16)]) -> mc_protocol::ItemStack {
        let list = enchantments
            .iter()
            .map(|&(id, level)| {
                let mut entry = mc_nbt::NbtCompound::new();
                entry.insert("id", mc_nbt::Nbt::Short(id));
                entry.insert("lvl", mc_nbt::Nbt::Short(level));
                mc_nbt::Nbt::Compound(entry)
            })
            .collect();
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("ench", mc_nbt::Nbt::list(list));
        mc_protocol::ItemStack {
            item_id: mc_data::item_by_name(name).expect("known item").0 as i16,
            count: 1,
            damage: 0,
            nbt: Some(nbt),
        }
    }

    /// A joined session whose tracker uses a seeded emitter stream, with one remote player
    /// (network id 61) wearing a Thorns III chestplate, and the previous walking tick having
    /// reported START_SPRINTING for the sprinting, airborne local player.
    fn session_facing_a_thorns_wearer() -> (
        Session,
        tokio::sync::mpsc::Sender<ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        let local_id = session.state().local.entity_id;
        {
            let state = session.state_mut();
            state.entities = crate::sync::EntityTracker::with_random_seed(0x5EED);
            state.entities.spawn(
                &mut state.world,
                61,
                TrackedKind::Player,
                Some(uuid::Uuid::from_u128(61)),
                [8 * 32, 70 * 32, 10 * 32],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
            assert!(state.entities.apply_equipment(
                61,
                3,
                Some(enchanted_stack("diamond_chestplate", &[(7, 3)]))
            ));
        }
        session.player_action(|out, packets| {
            out.walking_update(
                local_id,
                mc_physics::prediction::Outbound::Ground { on_ground: true },
                true,
                false,
                packets,
            );
        });
        let _start_packets = drain_outbound(&mut to_net);
        session.set_local_watched_sprinting(true);
        let state = &mut session.predictor_mut().expect("joined predictor").state;
        state.vel = DVec3::new(0.75, -0.25, -0.5);
        state.sprinting = true;
        state.yaw = 0.0;
        // `fallDistance > 0 && !onGround`: the falling-critical condition.
        state.fall_distance = 1.5;
        state.on_ground = false;
        (session, to_game, to_net)
    }

    /// java: `EntityOtherPlayerMP.attackEntityFrom` returns true (`EntityOtherPlayerMP.java:34-37`),
    /// so a local hit on a remote player runs the whole successful branch of
    /// `EntityPlayer.attackTargetEntityWithCurrentItem` (`:1355-1431`) after C02: the target's
    /// `addVelocity`, the local `motion *= 0.6` and `setSprinting(false)`, the CRIT then CRIT_MAGIC
    /// emitters, and the thorns hook. The STOP_SPRINTING edge then leads the next walking packet.
    #[test]
    fn a_remote_player_hit_runs_the_successful_attack_branch_after_c02() {
        let (mut session, _to_game, mut to_net) = session_facing_a_thorns_wearer();
        let local_id = session.state().local.entity_id;
        let thorns_hit_seed = (0_i64..)
            .find(|&seed| JavaRandom::new(seed).next_f32() < 0.45)
            .expect("a Thorns III hit");
        *session
            .state_mut()
            .entities
            .particle_random_mut(61)
            .unwrap() = JavaRandom::new(thorns_hit_seed);
        let local_position = session.predictor().unwrap().state.pos;
        let sharpness = enchanted_stack("diamond_sword", &[(16, 1)]);

        assert!(
            session.attack_entity_with_knockback(61, None, 0, 0, Some(&sharpness)),
            "a remote player is an EntityLivingBase, so ItemStack.hitEntity runs"
        );
        let attack_packets = drain_outbound(&mut to_net);
        assert!(
            matches!(
                attack_packets.as_slice(),
                [C2SPlay::UseEntity(c2s::UseEntity {
                    entity_id: 61,
                    action: c2s::UseEntityAction::Attack,
                })]
            ),
            "C02 is the only packet of the attack call: {attack_packets:?}"
        );

        // `addVelocity(-sin(0) * 1 * 0.5, 0.1, cos(0) * 1 * 0.5)` onto the zero S12 motion.
        let remote = session.state().entities.get(61).unwrap().entity;
        assert_eq!(
            session
                .state()
                .world
                .ecs
                .get::<&mc_world::Velocity>(remote)
                .unwrap()
                .0,
            DVec3::new(0.0, 0.1, 0.5)
        );
        let after_attack = session.predictor().unwrap().state;
        assert!(!after_attack.sprinting);
        assert_eq!(after_attack.vel.x, 0.75 * 0.6);
        assert_eq!(after_attack.vel.y, -0.25);
        assert_eq!(after_attack.vel.z, -0.5 * 0.6);

        // `onCriticalHit` then `onEnchantmentCritical`: each emitter's constructor update.
        let particles = session.state_mut().entities.take_particle_events();
        let crits = particles
            .iter()
            .take_while(|event| event.particle_id == mc_render::particle::packet::CRIT)
            .count();
        assert!(crits > 0, "the CRIT emitter's first update: {particles:?}");
        assert!(particles.len() > crits, "the CRIT_MAGIC emitter follows");
        assert!(particles[crits..]
            .iter()
            .all(|event| event.particle_id == mc_render::particle::packet::CRIT_MAGIC));

        // Thorns III hit: the request is the local `EntityPlayerSP.playSound` at its feet, and
        // the chestplate copy takes `damageItem(3)`.
        assert_eq!(
            session.state_mut().entities.take_status_effect_events(),
            vec![crate::sync::entities::EntityStatusEffectEvent::Sound(
                crate::sync::entities::EntityStatusSoundEvent {
                    network_id: local_id,
                    source_entity: None,
                    name: "damage.thorns",
                    volume: 0.5,
                    pitch: 1.0,
                    first_random: 0.0,
                    second_random: 0.0,
                    position: Some(local_position),
                    far_name: None,
                }
            )]
        );
        let chestplate = &session.state().entities.get(61).unwrap().equipment[3];
        assert_eq!(chestplate.as_ref().map(|stack| stack.damage), Some(3));
        // The roll, then the discarded `1 + nextInt(4)`; no Unbreaking means no armour draws.
        let mut expected_stream = JavaRandom::new(thorns_hit_seed);
        expected_stream.next_f32();
        expected_stream.next_i32_bound(4);
        assert_eq!(
            session
                .state_mut()
                .entities
                .particle_random_mut(61)
                .unwrap()
                .raw_seed(),
            expected_stream.raw_seed()
        );

        session.tick(&MoveInput::default());
        let next_tick = drain_outbound(&mut to_net);
        assert!(
            matches!(
                next_tick.first(),
                Some(C2SPlay::EntityAction(c2s::EntityAction {
                    entity_id,
                    action: c2s::EntityActionType::StopSprinting,
                    ..
                })) if *entity_id == local_id
            ),
            "STOP_SPRINTING follows the attack C02 on the next walking update: {next_tick:?}"
        );
        assert!(next_tick.get(1).is_some_and(is_movement));
    }

    /// java: `attackTargetEntityWithCurrentItem` runs
    /// `if (f1 > 0.0F) this.onEnchantmentCritical(targetEntity)` (`:1379-1382`) for every target
    /// whose client `attackEntityFrom` succeeded, `f1` being taken against `UNDEFINED` for a
    /// target that is not an `EntityLivingBase`. `EntityBoat.attackEntityFrom` returns true on a
    /// client, so a Sharpness sword on a boat shows the CRIT_MAGIC emitter; the crit `flag` needs
    /// a living target, so no CRIT. A zombie's client `attackEntityFrom` is false: nothing.
    #[test]
    fn a_sharpness_hit_on_a_boat_shows_the_enchantment_critical_emitter() {
        use mc_data::EntityType;

        let (mut session, _to_game, mut to_net) = joined_and_standing();
        let _join_packets = drain_outbound(&mut to_net);
        {
            let state = session.state_mut();
            state.entities = crate::sync::EntityTracker::with_random_seed(0x5EED);
            for (id, kind) in [(91, EntityType::Boat), (92, EntityType::Zombie)] {
                state.entities.spawn(
                    &mut state.world,
                    id,
                    TrackedKind::Type(kind),
                    None,
                    [8 * 32, 70 * 32, 10 * 32],
                    0.0,
                    0.0,
                    DVec3::ZERO,
                    mc_world::Metadata::new(),
                );
            }
        }
        // Airborne and falling, so a living target *would* satisfy the crit flag.
        {
            let state = &mut session.predictor_mut().expect("joined predictor").state;
            state.fall_distance = 1.5;
            state.on_ground = false;
        }
        let sharpness = enchanted_stack("diamond_sword", &[(16, 2)]);
        let plain = enchanted_stack("diamond_sword", &[(19, 1)]);

        let particles = |session: &mut Session| session.state_mut().entities.take_particle_events();
        session.attack_entity_with_knockback(91, Some(EntityType::Boat), 0, 0, Some(&sharpness));
        let boat = particles(&mut session);
        assert!(
            !boat.is_empty(),
            "CRIT_MAGIC emitter's first update on the boat"
        );
        assert!(boat
            .iter()
            .all(|event| event.particle_id == mc_render::particle::packet::CRIT_MAGIC));

        session.attack_entity_with_knockback(91, Some(EntityType::Boat), 0, 0, Some(&plain));
        assert!(
            particles(&mut session).is_empty(),
            "no Sharpness, no f1, no emitter"
        );

        session.attack_entity_with_knockback(92, Some(EntityType::Zombie), 0, 0, Some(&sharpness));
        assert!(
            particles(&mut session).is_empty(),
            "a remote zombie's attackEntityFrom is false on a client"
        );
        let _ = drain_outbound(&mut to_net);
    }

    /// java: `updateArmSwingProgress` evaluates
    /// `getArmSwingAnimationEnd()` on every tick, so Haste III (end 3) arriving mid-swing
    /// shortens the running swing at once: `-1 -> 0/6 -> 1/6 -> 2/3 -> 0`. The latched end gave
    /// `2/6, 3/6, ...` until the next swing.
    #[test]
    fn the_local_third_person_swing_reads_the_live_animation_end_every_tick() {
        let (mut session, _to_game, _to_net) = rig();
        session.set_local_swing_animation_end(6);
        session.note_local_swing(6);
        session.tick_local_swing();
        assert_eq!(session.local_render.swing_progress, 0.0);
        session.tick_local_swing();
        assert_eq!(session.local_render.swing_progress, 1.0 / 6.0);
        session.set_local_swing_animation_end(3);
        session.tick_local_swing();
        assert_eq!(session.local_render.swing_progress, 2.0 / 3.0);
        session.tick_local_swing();
        assert_eq!(
            session.local_render.swing_progress, 0.0,
            "3 >= 3 ends the swing"
        );
        assert!(!session.local_swinging);
    }

    /// **A sprint hit in the knockback tick.** java: an S12 for the
    /// local player is a scheduled task that *assigns* `motionX/Y/Z` at frame start; the same
    /// frame's `runTick` input section then reaches `attackTargetEntityWithCurrentItem`, whose
    /// `motionX/Z *= 0.6D` damps that knockback (`EntityPlayer.java:1355-1362`). The session
    /// used to park a local S12 until `tick_player`, after the input section, so the hit
    /// multiplied the stale motion and the parked S12 then overwrote it: full knockback, which
    /// an anticheat's velocity model does not predict. Differential: a session given the S12 motion
    /// directly (vanilla's order) and the live packet path must match through the attack and
    /// the following tick, packets included.
    #[test]
    fn a_sprint_hit_in_the_knockback_tick_damps_the_parked_s12_like_vanilla() {
        // 8000 units per block per tick (`handleEntityVelocity`).
        let s12_motion = DVec3::new(0.5, 0.4, -0.3);
        let (mut vanilla, _vanilla_game, mut vanilla_net) = session_facing_a_thorns_wearer();
        vanilla.predictor_mut().expect("joined predictor").state.vel = s12_motion;
        let (mut live, live_game, mut live_net) = session_facing_a_thorns_wearer();
        live_game
            .try_send(ToGame::Packet(S2CPlay::EntityVelocity(
                s2c::EntityVelocity {
                    entity_id: live.state().local.entity_id,
                    motion_x: 4000,
                    motion_y: 3200,
                    motion_z: -2400,
                },
            )))
            .unwrap();
        live.poll(16);
        // L5: `react_to` now assigns a local S12 at its own packet (vanilla's order), so the
        // knockback is already the live motion when the input section's attack runs.
        assert_eq!(
            live.predictor().expect("joined predictor").state.vel,
            s12_motion
        );

        for session in [&mut vanilla, &mut live] {
            assert!(session.attack_entity_with_knockback(61, None, 0, 0, None));
        }
        let live_after = live.predictor().expect("joined predictor").state;
        assert_eq!(
            live_after.vel,
            DVec3::new(s12_motion.x * 0.6, s12_motion.y, s12_motion.z * 0.6),
            "the 0.6 lands on the knockback"
        );
        assert_eq!(
            live.state().local.pending_velocity,
            None,
            "consumed by the hit"
        );
        assert_eq!(
            live_after.vel,
            vanilla.predictor().expect("joined predictor").state.vel
        );
        assert_eq!(
            drain_outbound(&mut live_net),
            drain_outbound(&mut vanilla_net)
        );

        live.tick(&MoveInput::default());
        vanilla.tick(&MoveInput::default());
        let (live_state, vanilla_state) = (
            live.predictor().expect("joined predictor").state,
            vanilla.predictor().expect("joined predictor").state,
        );
        assert_eq!(live_state.pos, vanilla_state.pos);
        assert_eq!(live_state.vel, vanilla_state.vel);
        assert_eq!(live_state.sprinting, vanilla_state.sprinting);
        assert_eq!(
            drain_outbound(&mut live_net),
            drain_outbound(&mut vanilla_net)
        );
    }

    /// **An ender dragon part is a client-successful target.** java:
    /// `EntityDragonPart.attackEntityFrom` returns `attackEntityFromPart`'s unconditional `true`
    /// (`EntityDragon.java:562-583`), so a sprint hit on a part damps the motion by 0.6 and
    /// clears sprint, and `ItemStack.hitEntity` runs on the dragon. The picker hands the session
    /// a part id with no tracked kind; a zombie (EntityLivingBase, remote `false`) is the control.
    #[test]
    fn a_sprint_hit_on_an_ender_dragon_part_damps_motion_and_clears_sprint() {
        use mc_data::EntityType;

        let (mut session, _to_game, mut to_net) = session_facing_a_thorns_wearer();
        {
            let state = session.state_mut();
            for (id, kind) in [(81, EntityType::EnderDragon), (91, EntityType::Zombie)] {
                state.entities.spawn(
                    &mut state.world,
                    id,
                    TrackedKind::Type(kind),
                    None,
                    [8 * 32, 70 * 32, 14 * 32],
                    0.0,
                    0.0,
                    DVec3::ZERO,
                    mc_world::Metadata::new(),
                );
            }
            assert!(
                state.entities.dragon_part(82).is_some(),
                "the head part is pickable"
            );
        }

        assert!(
            !session.attack_entity_with_knockback(91, Some(EntityType::Zombie), 0, 0, None),
            "a mob hit reaches no hitEntity on the client"
        );
        let after_mob = session.predictor().expect("joined predictor").state;
        assert!(after_mob.sprinting && session.local_watched_sprinting());
        assert_eq!(
            after_mob.vel,
            DVec3::new(0.75, -0.25, -0.5),
            "no 0.6 for a mob"
        );

        assert!(
            session.attack_entity_with_knockback(82, None, 0, 0, None),
            "the part's hit reaches hitEntity on the dragon"
        );
        let after_part = session.predictor().expect("joined predictor").state;
        assert!(!after_part.sprinting && !session.local_watched_sprinting());
        assert_eq!(after_part.vel, DVec3::new(0.75 * 0.6, -0.25, -0.5 * 0.6));
        assert!(drain_outbound(&mut to_net).iter().any(|packet| matches!(
            packet,
            C2SPlay::UseEntity(c2s::UseEntity {
                entity_id: 82,
                action: c2s::UseEntityAction::Attack,
            })
        )));
    }

    /// **A sprint hit with the sprint key held.** java: the attack's
    /// `setSprinting(false)` runs in the input section, then the same tick's
    /// `EntityPlayerSP.onLivingUpdate` (`:813`) sees `!isSprinting() && moveForward >= 0.8F &&
    /// keyBindSprint.isKeyDown()` and sprints again before `onUpdateWalkingPlayer`, whose
    /// `isSprinting() != serverSprintState` test then finds nothing to report. The hit costs
    /// exactly the 0.6 and sends no C0B; a STOP/START pair here is a pattern anticheats flag.
    #[test]
    fn a_sprint_hit_with_the_sprint_key_held_resprints_in_the_same_tick_without_c0b() {
        let (mut session, _to_game, mut to_net) = session_facing_a_thorns_wearer();
        assert!(session.attack_entity_with_knockback(61, None, 0, 0, None));
        let _attack_packets = drain_outbound(&mut to_net);
        assert!(
            !session.local_watched_sprinting(),
            "the hit clears the flag"
        );

        session.tick(&MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        });
        let sent = drain_outbound(&mut to_net);
        assert!(
            !sent
                .iter()
                .any(|packet| matches!(packet, C2SPlay::EntityAction(_))),
            "re-sprinted before onUpdateWalkingPlayer, so no C0B: {sent:?}"
        );
        assert!(session.local_watched_sprinting());
        assert!(
            session
                .predictor()
                .expect("joined predictor")
                .state
                .sprinting
        );
    }

    /// The negative gates of the same branch: on the ground there is no critical emitter, a
    /// plain sword has no enchantment-critical emitter, a non-sprinting hit without Knockback
    /// adds no target impulse, and a thorns miss requests no sound but still costs one point.
    #[test]
    fn a_grounded_plain_remote_player_hit_skips_the_emitters_and_the_impulse() {
        let (mut session, _to_game, mut to_net) = session_facing_a_thorns_wearer();
        session.set_local_watched_sprinting(false);
        {
            let state = &mut session.predictor_mut().unwrap().state;
            state.sprinting = false;
            state.on_ground = true;
        }
        let thorns_miss_seed = (0_i64..)
            .find(|&seed| JavaRandom::new(seed).next_f32() >= 0.45)
            .expect("a Thorns III miss");
        *session
            .state_mut()
            .entities
            .particle_random_mut(61)
            .unwrap() = JavaRandom::new(thorns_miss_seed);

        assert!(session.attack_entity_with_knockback(
            61,
            None,
            0,
            0,
            Some(&enchanted_stack("diamond_sword", &[]))
        ));
        assert_eq!(drain_outbound(&mut to_net).len(), 1);
        let remote = session.state().entities.get(61).unwrap().entity;
        assert_eq!(
            session
                .state()
                .world
                .ecs
                .get::<&mc_world::Velocity>(remote)
                .unwrap()
                .0,
            DVec3::ZERO
        );
        assert_eq!(
            session.predictor().unwrap().state.vel,
            DVec3::new(0.75, -0.25, -0.5),
            "i == 0 skips the whole knockback block, including the local damping"
        );
        assert!(session
            .state_mut()
            .entities
            .take_particle_events()
            .is_empty());
        assert!(session
            .state_mut()
            .entities
            .take_status_effect_events()
            .is_empty());
        let mut expected_stream = JavaRandom::new(thorns_miss_seed);
        expected_stream.next_f32();
        assert_eq!(
            session
                .state_mut()
                .entities
                .particle_random_mut(61)
                .unwrap()
                .raw_seed(),
            expected_stream.raw_seed(),
            "a miss draws only the roll"
        );
        let chestplate = &session.state().entities.get(61).unwrap().equipment[3];
        assert_eq!(chestplate.as_ref().map(|stack| stack.damage), Some(1));
    }

    /// Particles mod: configured hit emitters are whole extra `EntityParticleEmitter`s
    /// right after vanilla's own (`n - 1` with the vanilla condition, `n` with "always" and no
    /// condition), for living targets only. Unconfigured or at vanilla values the hit is
    /// vanilla's, event for event.
    #[test]
    fn m6_hit_particles_add_whole_emitters_after_vanillas_own() {
        use mc_render::particle::packet::{CRIT, CRIT_MAGIC};
        use mc_render::particle::tuning::HitConfig;

        /// The seeded session facing remote player 61, plus a zombie (62) and a boat (63).
        fn setup(critical: bool) -> (Session, impl Sized) {
            let (mut session, to_game, to_net) = session_facing_a_thorns_wearer();
            let state = session.state_mut();
            for (id, kind) in [
                (62, mc_data::EntityType::Zombie),
                (63, mc_data::EntityType::Boat),
            ] {
                state.entities.spawn(
                    &mut state.world,
                    id,
                    TrackedKind::Type(kind),
                    None,
                    [9 * 32, 70 * 32, 10 * 32],
                    0.0,
                    0.0,
                    DVec3::ZERO,
                    mc_world::Metadata::new(),
                );
            }
            if !critical {
                let predictor = &mut session.predictor_mut().expect("joined predictor").state;
                predictor.fall_distance = 0.0;
                predictor.on_ground = true;
            }
            (session, (to_game, to_net))
        }
        fn attack(session: &mut Session, target: i32, held: Option<&mc_protocol::ItemStack>) {
            let kind = match target {
                61 => None,
                62 => Some(mc_data::EntityType::Zombie),
                _ => Some(mc_data::EntityType::Boat),
            };
            session.attack_entity_with_knockback(target, kind, 0, 0, held);
        }
        // One attack with `config` installed: its particle events.
        let modded = |config: HitConfig, critical: bool, target: i32, held| {
            let (mut session, _channels) = setup(critical);
            session.hit_particles.set_config(config);
            attack(&mut session, target, held);
            session.state_mut().entities.take_particle_events()
        };
        // Vanilla's attack on the same seed, then `crits` and `sharps` emitters by hand.
        let reference = |critical: bool, target: i32, held, crits: u32, sharps: u32| {
            let (mut session, _channels) = setup(critical);
            attack(&mut session, target, held);
            let state = session.state_mut();
            for _ in 0..crits {
                state.entities.apply_crit_animation(&state.world, target, 4);
            }
            for _ in 0..sharps {
                state.entities.apply_crit_animation(&state.world, target, 5);
            }
            state.entities.take_particle_events()
        };
        let config = |crit: f32, sharp: f32, always_crit: bool, always_sharp: bool| HitConfig {
            crit,
            sharp,
            always_crit,
            always_sharp,
        };
        let sword = enchanted_stack("diamond_sword", &[(16, 1)]);
        let sharp = Some(&sword);

        // Vanilla: a sharpness crit on a player shows one emitter of each locally.
        let vanilla = reference(true, 61, sharp, 0, 0);
        assert!(vanilla.iter().any(|event| event.particle_id == CRIT));
        assert!(vanilla.iter().any(|event| event.particle_id == CRIT_MAGIC));
        assert_eq!(modded(HitConfig::VANILLA, true, 61, sharp), vanilla);
        assert_eq!(
            modded(config(1.0, 1.0, false, false), true, 61, sharp),
            vanilla
        );
        // 3x: two more of each, after vanilla's.
        assert_eq!(
            modded(config(3.0, 3.0, false, false), true, 61, sharp),
            reference(true, 61, sharp, 2, 2)
        );
        // The mod's defaults on a plain, non-critical hit: vanilla shows nothing; "always
        // sharpness" adds two emitters, crits (no condition, no "always") none.
        assert!(reference(false, 61, None, 0, 0).is_empty());
        assert_eq!(
            modded(config(2.0, 2.0, false, true), false, 61, None),
            reference(false, 61, None, 0, 2)
        );
        // "Always crits" at 2x; sharpness at 0 adds nothing even with "always".
        assert_eq!(
            modded(config(2.0, 0.0, true, true), false, 61, None),
            reference(false, 61, None, 2, 0)
        );
        // A living mob: vanilla shows nothing locally (only the server's S0B echo); a crit at 2x
        // adds one emitter, "always sharpness" two.
        assert!(reference(true, 62, None, 0, 0).is_empty());
        assert_eq!(
            modded(config(2.0, 2.0, false, true), true, 62, None),
            reference(true, 62, None, 1, 2)
        );
        // Not living: nothing.
        assert_eq!(
            modded(config(3.0, 3.0, true, true), true, 63, None),
            reference(true, 63, None, 0, 0)
        );
    }

    /// `PlayerControllerMP.attackEntity` sends C02 in every mode but skips
    /// `attackTargetEntityWithCurrentItem` for a spectator, and a mob target keeps the remote
    /// `EntityLivingBase.attackEntityFrom` `false`: neither reaches the local half.
    #[test]
    fn spectator_and_mob_hits_keep_skipping_the_local_attack_half() {
        let (mut session, _to_game, mut to_net) = session_facing_a_thorns_wearer();
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                62,
                TrackedKind::Type(mc_data::EntityType::Zombie),
                None,
                [9 * 32, 70 * 32, 10 * 32],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        let before_stream = session
            .state_mut()
            .entities
            .particle_random_mut(61)
            .unwrap()
            .raw_seed();
        let sharpness = enchanted_stack("diamond_sword", &[(16, 5), (19, 2)]);

        assert!(!session.attack_entity_with_knockback(
            62,
            Some(mc_data::EntityType::Zombie),
            0,
            2,
            Some(&sharpness)
        ));
        session.state_mut().local.game_mode = mc_protocol::play::s2c::GameMode::Spectator;
        assert!(!session.attack_entity_with_knockback(61, None, 0, 2, Some(&sharpness)));

        let packets = drain_outbound(&mut to_net);
        assert_eq!(packets.len(), 2, "both clicks still send C02: {packets:?}");
        let predictor = session.predictor().unwrap().state;
        assert!(predictor.sprinting);
        assert_eq!(predictor.vel, DVec3::new(0.75, -0.25, -0.5));
        assert!(session.local_watched_sprinting());
        for id in [61, 62] {
            let entity = session.state().entities.get(id).unwrap().entity;
            assert_eq!(
                session
                    .state()
                    .world
                    .ecs
                    .get::<&mc_world::Velocity>(entity)
                    .unwrap()
                    .0,
                DVec3::ZERO
            );
        }
        assert!(session
            .state_mut()
            .entities
            .take_particle_events()
            .is_empty());
        assert!(session
            .state_mut()
            .entities
            .take_status_effect_events()
            .is_empty());
        assert_eq!(
            session
                .state_mut()
                .entities
                .particle_random_mut(61)
                .unwrap()
                .raw_seed(),
            before_stream
        );
    }

    /// The area's P0. `handlePlayerAbilities` reaches the predictor's flying
    /// state, and `EntityPlayer.moveEntityWithHeading` then applies its client-side hover path.
    #[test]
    fn server_set_flight_reaches_the_predictor_and_stops_the_fall() {
        let (mut session, to_game, _to_net) = joined_and_standing();

        // Six ticks of ordinary gravity from y = 70.
        for _ in 0..6 {
            session.tick(&MoveInput::default());
        }
        let falling = session.predictor().unwrap().state;
        assert!(falling.vel.y < -0.2, "falling: {}", falling.vel.y);
        assert!(!falling.flying);

        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerAbilities(
                s2c::PlayerAbilities {
                    // invulnerable | flying | allow_flying | creative
                    flags: 0x0F,
                    fly_speed: 0.05,
                    walk_speed: 0.1,
                },
            )))
            .unwrap();
        session.poll(16);
        assert!(session.state().local.abilities.flying);

        for _ in 0..20 {
            session.tick(&MoveInput::default());
        }
        let hovering = session.predictor().unwrap().state;
        assert!(hovering.flying, "the capability reached the predictor");
        // `motionY = previous * 0.6` decays to zero; gravity would have run away to -3.92.
        assert!(
            hovering.vel.y.abs() < 1e-3,
            "a flying player hovers, got vy = {}",
            hovering.vel.y
        );
        // `motionY = previous * 0.6` bleeds the existing downward speed off over a handful of
        // ticks rather than cutting it dead, so about a block of the fall survives the toggle.
        // Twenty ticks of real gravity from here would have been more than fifteen.
        assert!(
            hovering.pos.y > falling.pos.y - 1.5,
            "and stopped falling: {} then {}",
            falling.pos.y,
            hovering.pos.y
        );
    }

    /// F-P0-network.play.server-02-2 / F-P1-entity.player-05-8: game modes configure
    /// `PlayerCapabilities` at three different lifecycle seams. Initial join and respawn create
    /// defaults, a proxy's second JoinGame reuses the player, and reason 3 mutates in place.
    #[test]
    fn join_respawn_and_mode_change_follow_vanillas_distinct_ability_lifecycles() {
        let (mut session, to_game, _to_net) = rig();
        session.state_mut().local.abilities = crate::sync::Abilities::from_flags(0x0F, 0.25, 0.4);

        let mut creative_join = join_game();
        let S2CPlay::JoinGame(join) = &mut creative_join else {
            unreachable!()
        };
        join.game_mode = 1;
        to_game
            .try_send(ToGame::Packet(creative_join.clone()))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.game_mode, s2c::GameMode::Creative);
        assert_eq!(
            session.state().local.abilities,
            crate::sync::Abilities {
                invulnerable: true,
                flying: false,
                allow_flying: true,
                creative_mode: true,
                allow_edit: true,
                fly_speed: 0.05,
                walk_speed: 0.1,
            },
            "an initial JoinGame starts with a fresh PlayerCapabilities"
        );

        // A second JoinGame on the same connection does not take loadWorld(null)'s player-null
        // path. The new controller configures Survival and then the packet's Creative mode on
        // the existing object: flying clears, but S39's two speeds survive.
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerAbilities(
                s2c::PlayerAbilities {
                    flags: 0x0F,
                    fly_speed: 0.2,
                    walk_speed: 0.3,
                },
            )))
            .unwrap();
        session.poll(16);
        to_game.try_send(ToGame::Packet(creative_join)).unwrap();
        session.poll(16);
        let abilities = session.state().local.abilities;
        assert!(
            !abilities.flying,
            "the new controller's Survival pass clears it"
        );
        assert!(abilities.invulnerable && abilities.allow_flying && abilities.creative_mode);
        assert_eq!((abilities.fly_speed, abilities.walk_speed), (0.2, 0.3));

        let change_mode = |value| {
            S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: s2c::GameStateReason::ChangeGameMode,
                value,
            })
        };
        to_game.try_send(ToGame::Packet(change_mode(3.0))).unwrap();
        session.poll(16);
        assert!(
            session.state().local.abilities.flying,
            "spectator forces flight"
        );
        assert!(
            !session.state().local.abilities.allow_edit,
            "the reason-3 game-mode lifecycle also installs Spectator's no-edit capability"
        );
        to_game.try_send(ToGame::Packet(change_mode(1.0))).unwrap();
        session.poll(16);
        let abilities = session.state().local.abilities;
        assert!(abilities.flying, "creative does not clear existing flight");
        assert!(abilities.allow_edit, "Creative restores editing");
        assert_eq!((abilities.fly_speed, abilities.walk_speed), (0.2, 0.3));
        to_game.try_send(ToGame::Packet(change_mode(2.0))).unwrap();
        session.poll(16);
        let abilities = session.state().local.abilities;
        assert!(!abilities.invulnerable);
        assert!(!abilities.flying);
        assert!(!abilities.allow_flying);
        assert!(!abilities.creative_mode);
        assert!(!abilities.allow_edit, "Adventure disables editing");
        assert_eq!((abilities.fly_speed, abilities.walk_speed), (0.2, 0.3));

        // The fresh respawn player is configured with the controller's old mode first, then
        // S07's new mode. Spectator -> Creative therefore retains the first pass's flight bit,
        // while still resetting S39's speeds to PlayerCapabilities defaults.
        to_game.try_send(ToGame::Packet(change_mode(3.0))).unwrap();
        session.poll(16);
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 1,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        let abilities = session.state().local.abilities;
        assert!(
            abilities.flying,
            "old Spectator config runs before new Creative config"
        );
        assert!(abilities.invulnerable && abilities.allow_flying && abilities.creative_mode);
        assert!(
            abilities.allow_edit,
            "S07's final Creative pass restores editing"
        );
        assert_eq!((abilities.fly_speed, abilities.walk_speed), (0.05, 0.1));

        // The headline defect: a flying Creative player respawns as Creative. Both passes leave
        // the fresh default flight bit false, rather than leaking S39's true bit across lives.
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerAbilities(
                s2c::PlayerAbilities {
                    flags: 0x0F,
                    fly_speed: 0.4,
                    walk_speed: 0.5,
                },
            )))
            .unwrap();
        session.poll(16);
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 1,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        let abilities = session.state().local.abilities;
        assert!(!abilities.flying);
        assert_eq!((abilities.fly_speed, abilities.walk_speed), (0.05, 0.1));
    }

    /// java: `handleEntityVelocity` (`:501`) calls `Entity.setVelocity` on
    /// **whatever entity the id names**. `EntityTracker` deliberately never holds the local
    /// player, so `apply_velocity` took its `return false` path and every knockback in the game
    /// was discarded — you got hit in PvP and did not move.
    #[test]
    fn a_velocity_packet_for_the_local_player_moves_the_predictor() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        assert!(session.predictor().unwrap().state.on_ground, "landed");
        let before = session.predictor().unwrap().state.pos;

        // 8000 units per block per tick, as `handleEntityVelocity` divides.
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityVelocity(
                s2c::EntityVelocity {
                    entity_id: session.state().local.entity_id,
                    motion_x: 4000,
                    motion_y: 3200,
                    motion_z: -2400,
                },
            )))
            .unwrap();
        session.poll(16);
        // java: `Entity.setVelocity` assigns while the packet is handled, not at the next tick.
        assert_eq!(
            session.predictor().unwrap().state.vel,
            DVec3::new(0.5, 0.4, -0.3),
            "the local player's velocity is assigned at its packet rather than dropped"
        );
        assert_eq!(
            session.state().local.pending_velocity,
            None,
            "consumed by the packet's own react_to"
        );

        session.tick(&MoveInput::default());
        let state = session.predictor().unwrap().state;
        assert!(state.pos.x > before.x + 0.4, "knocked +X: {}", state.pos.x);
        assert!(state.pos.z < before.z - 0.2, "knocked -Z: {}", state.pos.z);
        assert!(state.pos.y > before.y, "and upward: {}", state.pos.y);

        // A velocity packet for somebody else must not touch us. Let the knockback above bleed
        // off first, or this would be measuring the leftover momentum.
        for _ in 0..80 {
            session.tick(&MoveInput::default());
        }
        let before = session.predictor().unwrap().state.pos;
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityVelocity(
                s2c::EntityVelocity {
                    entity_id: session.state().local.entity_id + 1,
                    motion_x: 8000,
                    motion_y: 0,
                    motion_z: 0,
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.pending_velocity, None);
        session.tick(&MoveInput::default());
        assert!(
            (session.predictor().unwrap().state.pos.x - before.x).abs() < 1e-9,
            "a velocity packet for another entity moved us"
        );
    }

    /// F-P1-entity.player-01-13: local status 3 runs `EntityPlayer.onDeath` at the packet
    /// boundary. The two S12 orderings are the biting control: deferring the death mutation to
    /// `tick` would make both batches collapse to the same final velocity.
    #[test]
    fn local_player_death_resizes_kicks_once_and_preserves_same_batch_velocity_order() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        session.predictor_mut().unwrap().state.yaw = 90.0;
        let velocity = || {
            S2CPlay::EntityVelocity(s2c::EntityVelocity {
                entity_id,
                motion_x: 4000,
                motion_y: 3200,
                motion_z: -2400,
            })
        };
        let death = || {
            S2CPlay::EntityStatus(s2c::EntityStatus {
                entity_id,
                opcode: 3,
            })
        };

        // Earlier S12 is overwritten by the later onDeath call.
        to_game.try_send(ToGame::Packet(velocity())).unwrap();
        to_game.try_send(ToGame::Packet(death())).unwrap();
        session.poll(16);
        let state = session.predictor().unwrap().state;
        let angle = 90.0_f32 * core::f32::consts::PI / 180.0;
        let death_velocity = DVec3::new(
            f64::from(-math_helper::cos(angle) * 0.1),
            0.10000000149011612,
            f64::from(-math_helper::sin(angle) * 0.1),
        );
        assert_eq!((state.width, state.height), (0.2, 0.2));
        assert_eq!(state.vel, death_velocity);
        assert_eq!(session.state().local.pending_velocity, None);

        // A later S12 overwrites the death kick at its own packet; the death pose is not
        // replayed merely because health stays at zero.
        to_game.try_send(ToGame::Packet(death())).unwrap();
        to_game.try_send(ToGame::Packet(velocity())).unwrap();
        session.poll(16);
        assert_eq!(session.state().local.pending_velocity, None);
        assert_eq!(
            session.predictor().unwrap().state.vel,
            DVec3::new(0.5, 0.4, -0.3)
        );
        session.push_server_movement_state();
        assert_eq!(
            session.predictor().unwrap().state.vel,
            DVec3::new(0.5, 0.4, -0.3)
        );

        // Respawn constructs a normal-sized replacement player and discards the corpse motion.
        session.reset_player_after_respawn();
        let reset = session.predictor().unwrap().state;
        assert_eq!((reset.width, reset.height), (PLAYER_WIDTH, PLAYER_HEIGHT));
        assert_eq!(reset.vel, DVec3::ZERO);
    }

    /// F-P0-network.play.server-05-2: S12 is deferred for the local predictor, but later packets
    /// in the same poll batch must still supersede it in wire order exactly as direct entity
    /// mutation does in Java.
    #[test]
    fn a_later_position_correction_or_respawn_cancels_deferred_local_velocity() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        assert!(session.predictor().unwrap().state.on_ground, "landed");
        let entity_id = session.state().local.entity_id;
        let velocity = || {
            S2CPlay::EntityVelocity(s2c::EntityVelocity {
                entity_id,
                motion_x: 4000,
                motion_y: 3200,
                motion_z: -2400,
            })
        };

        // Absolute S08 clears all three components after S12 assigned them in vanilla.
        to_game.try_send(ToGame::Packet(velocity())).unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: 8.5,
                    y: 65.0,
                    z: 8.5,
                    yaw: 0.0,
                    pitch: 0.0,
                    flags: 0,
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.pending_velocity, None);
        assert_eq!(session.predictor().unwrap().state.vel, DVec3::ZERO);
        session.tick(&MoveInput::default());
        assert_eq!(
            session.predictor().unwrap().state.pos.x,
            8.5,
            "the cancelled +X knockback must not leak into the next C03 position"
        );
        assert_eq!(session.predictor().unwrap().state.pos.z, 8.5);

        // With only X relative, Java preserves X motion and clears absolute Y/Z. This is the
        // asymmetric negative control: clearing the whole option would pass the first half.
        to_game.try_send(ToGame::Packet(velocity())).unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: 0.25,
                    y: 65.0,
                    z: 8.5,
                    yaw: 0.0,
                    pitch: 0.0,
                    flags: 0x01,
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.pending_velocity, None);
        assert_eq!(
            session.predictor().unwrap().state.vel,
            DVec3::new(0.5, 0.0, 0.0)
        );
        let corrected_x = session.predictor().unwrap().state.pos.x;
        session.tick(&MoveInput::default());
        assert!(
            session.predictor().unwrap().state.pos.x > corrected_x + 0.4,
            "relative X preserves the S12 component"
        );
        assert_eq!(session.predictor().unwrap().state.pos.z, 8.5);

        // Same-dimension S07 does not call ClientState::reset_world, but it still creates a
        // fresh EntityPlayerSP and therefore must discard the S12 motion.
        to_game.try_send(ToGame::Packet(velocity())).unwrap();
        session.poll(16);
        assert_eq!(
            session.predictor().unwrap().state.vel,
            DVec3::new(0.5, 0.4, -0.3)
        );
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.pending_velocity, None);
        assert_eq!(session.predictor().unwrap().state.vel, DVec3::ZERO);
    }

    /// java: `handleEntityVelocity` assigns all three motion components
    /// at its packet, and a later `handlePlayerPosLook` with a *relative* X leaves `motionX`
    /// exactly as the S12 set it, here `0.0`. Negative control for the old parking: once the
    /// S08 zeroed the parked copy's absolute Y/Z it compared equal to zero and was dropped, so
    /// the motion from *before* the S12 survived on the relative axis.
    #[test]
    fn a_relative_s08_axis_keeps_the_s12_component_even_when_it_is_zero() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        let entity_id = session.state().local.entity_id;
        // Motion left over from before the S12; vanilla's assignment replaces all of it.
        session.predictor_mut().unwrap().state.vel = DVec3::new(0.3, 0.0, 0.0);
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityVelocity(
                s2c::EntityVelocity {
                    entity_id,
                    motion_x: 0,
                    motion_y: 3200,
                    motion_z: -2400,
                },
            )))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: 0.0,
                    y: 65.0,
                    z: 8.5,
                    yaw: 0.0,
                    pitch: 0.0,
                    flags: 0x01,
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(
            session.predictor().unwrap().state.vel,
            DVec3::ZERO,
            "S12 set motionX to 0 and the relative X keeps it; absolute Y/Z are zeroed"
        );
    }

    /// java: a proxy server switch is a mid-connection S01 followed by an
    /// S07 or two S07s, depending on the proxy. `Minecraft.loadWorld` builds a player only while
    /// `thePlayer == null`, so the S01 keeps the existing EntityPlayerSP and its DataWatcher;
    /// `setDimensionAndSpawnPlayer` then copies that watcher, sprint bit included, into the
    /// replacement, whose attribute map has no sprint entry, whose `serverSprintState` is false
    /// and whose fresh `MovementInputFromOptions` is not sneaking. With W held the first tick
    /// keeps sprinting and reports START_SPRINTING. Negative control: wiping the watcher at the
    /// S01 lost the bit, so a player without the sprint key stopped and sent no C0B.
    #[test]
    fn a_proxy_switch_keeps_the_watched_sprint_flag_into_the_respawned_player() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        // The server's echo of a sneaking, running player: index 0 with bits 1 and 3.
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityMetadata(
                s2c::EntityMetadata {
                    entity_id,
                    metadata: vec![mc_protocol::MetadataEntry {
                        index: 0,
                        value: mc_protocol::MetadataValue::Byte(0x0A),
                    }],
                },
            )))
            .unwrap();
        session.poll(16);

        // A proxy's server switch: S01 into a placeholder dimension, then S07 into the real one.
        let mut switch_join = join_game();
        if let S2CPlay::JoinGame(join) = &mut switch_join {
            join.dimension = -1;
        }
        to_game.try_send(ToGame::Packet(switch_join)).unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert_eq!(
            session.state().local.metadata.get(0),
            Some(&MetadataValue::Byte(0x0A)),
            "S01 keeps thePlayer's watcher and S07 copies it"
        );
        let replacement = session.predictor().unwrap().state;
        assert!(
            replacement.sprinting,
            "isSprinting() is the copied watched bit"
        );
        assert!(
            !replacement.sneaking,
            "isSneaking() reads the fresh movement input, not watched bit 1"
        );
        assert_eq!(
            replacement.sprint_modifier_amount, None,
            "fresh attribute map"
        );
        assert_eq!(
            replacement.ai_move_speed(),
            mc_physics::state::WALK_SPEED as f32,
            "sprint flag without the attribute entry walks"
        );
        let _ = drain_outbound(&mut to_net);

        // The dimension-changing S07 spawned the replacement into an empty world: its first
        // world tick is skipped (`addedToChunk`), and the edge rides the second.
        let forward = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        session.tick(&forward);
        assert_eq!(drain_outbound(&mut to_net), Vec::new());
        session.tick(&forward);
        let sent = drain_outbound(&mut to_net);
        let start = sent.iter().position(|packet| {
            matches!(
                packet,
                C2SPlay::EntityAction(c2s::EntityAction {
                    action: c2s::EntityActionType::StartSprinting,
                    ..
                })
            )
        });
        let movement = sent.iter().position(is_movement);
        assert!(
            start.is_some() && start < movement,
            "the fresh serverSprintState reports the kept sprint first: {sent:?}"
        );
    }

    /// java: `loadWorld` constructs the connection's first EntityPlayerSP
    /// at the new WorldClient's spawn, `(8, 64, 8)` + `(0.5, 1, 0.5)`, and `flipPlayer` turns it
    /// to -180, so a tick that runs before the first S08 reports a C06 from there: moved from the
    /// zero `lastReportedPos*` and turned from the zero `lastReportedYaw`. Negative control: the
    /// old origin start did not move (the unloaded-column pin holds `posY == 0`), so it sent C05.
    #[test]
    fn a_tick_before_the_first_s08_reports_from_the_constructor_spawn() {
        let (mut session, to_game, mut to_net) = rig();
        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        session.poll(16);
        let joined = session.predictor().unwrap().state;
        assert_eq!(joined.pos, DVec3::new(8.5, 65.0, 8.5));
        assert_eq!((joined.yaw, joined.pitch), (-180.0, 0.0));
        let _ = drain_outbound(&mut to_net);

        // The live driver feeds the predictor's own rotation back as the tick's look.
        let look = MoveInput {
            yaw: joined.yaw,
            pitch: joined.pitch,
            ..MoveInput::default()
        };
        // The first world tick skips the player spawned into the unsent column
        // (`addedToChunk`); the report comes from the second.
        session.tick(&look);
        assert_eq!(drain_outbound(&mut to_net), Vec::new());
        assert_eq!(session.predictor().unwrap().state.pos, joined.pos);
        session.tick(&look);
        let sent = drain_outbound(&mut to_net);
        assert!(
            matches!(
                sent.last(),
                Some(C2SPlay::PlayerPositionAndLook(c2s::PlayerPositionAndLook {
                    yaw,
                    on_ground: false,
                    ..
                })) if *yaw == -180.0
            ),
            "{sent:?}"
        );
    }

    /// Found by the end-to-end Java oracle (`app/h1_movement_oracle.rs`). java:
    /// `World.spawnEntityInWorld` gives a new player to `getChunkFromChunkCoords`, which for an
    /// unsent column is `ChunkProviderClient.blankChunk`; `EmptyChunk.addEntity` is a no-op, so
    /// `addedToChunk` stays false and the next `updateEntityWithOptionalForce` skips
    /// `onUpdate()` (no physics, no C03, no effect countdown) before setting it. A JoinGame and a
    /// dimension-changing S07 always spawn into an empty world; a same-dimension S07 whose spawn
    /// column (8, 8) is loaded does not lose its first tick. Negative control: the old session
    /// reported (and fell) on the JoinGame's first tick.
    #[test]
    fn a_player_spawned_into_an_unsent_column_skips_exactly_its_first_world_tick() {
        let respawn = |dimension| {
            S2CPlay::Respawn(s2c::Respawn {
                dimension,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })
        };
        let (mut session, to_game, mut to_net) = rig();
        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityEffect(s2c::EntityEffect {
                entity_id: 3,
                effect_id: 8,
                amplifier: 1,
                duration: 100,
                show_particles: false,
            })))
            .unwrap();
        session.poll(16);
        let _ = drain_outbound(&mut to_net);
        let spawned = session.predictor().unwrap().state;
        // The live driver feeds the predictor's own rotation back as the tick's look.
        let tick = |session: &mut Session| {
            let state = session.predictor().unwrap().state;
            session.tick(&MoveInput {
                yaw: state.yaw,
                pitch: state.pitch,
                ..MoveInput::default()
            });
        };

        tick(&mut session);
        assert_eq!(
            drain_outbound(&mut to_net),
            Vec::new(),
            "the skipped tick reports nothing"
        );
        assert_eq!(session.predictor().unwrap().state.pos, spawned.pos);
        assert_eq!(session.predictor().unwrap().state.vel, spawned.vel);
        assert_eq!(
            session.effects.jump_boost,
            Some((1, 100)),
            "no countdown either"
        );

        tick(&mut session);
        let sent = drain_outbound(&mut to_net);
        assert!(
            matches!(sent.last(), Some(C2SPlay::PlayerPositionAndLook(_))),
            "the second tick reports from the constructor spawn: {sent:?}"
        );
        assert_eq!(session.effects.jump_boost, Some((1, 99)));

        // Load the default spawn column (8, 8) -> chunk (0, 0), then respawn in the same
        // dimension: the replacement is added to a real chunk and ticks at once.
        to_game
            .try_send(ToGame::Packet(chunk_with_a_stone_floor(0, 0, 60)))
            .unwrap();
        to_game.try_send(ToGame::Packet(respawn(0))).unwrap();
        session.poll(16);
        let _ = drain_outbound(&mut to_net);
        assert!(session.player_is_simulated());
        tick(&mut session);
        let sent = drain_outbound(&mut to_net);
        assert!(
            matches!(sent.last(), Some(C2SPlay::PlayerPositionAndLook(_))),
            "a same-dimension respawn over a loaded column ticks at once: {sent:?}"
        );

        // A dimension change builds an empty world: skipped again.
        to_game.try_send(ToGame::Packet(respawn(-1))).unwrap();
        session.poll(16);
        let _ = drain_outbound(&mut to_net);
        assert!(!session.player_is_simulated());
        tick(&mut session);
        assert_eq!(drain_outbound(&mut to_net), Vec::new());
        assert!(session.player_is_simulated());
    }

    /// java: `Minecraft.loadWorld` (`:2408-2419`) keeps
    /// `thePlayer` on a mid-connection JoinGame. `preparePlayerToSpawn` zeroes motion and pitch
    /// and restores full health, the movement input is fresh, and the new PlayerControllerMP has
    /// `currentPlayerItem == 0` (so the kept slot is re-synced), but `serverSprintState`,
    /// `lastReported*`, the watcher, the attribute map and the potions are the same player's.
    /// Negative control: rebuilding the player re-reported START_SPRINTING on the next tick and
    /// dropped the jump boost.
    #[test]
    fn a_mid_connection_join_game_keeps_the_player_and_prepares_it_to_spawn() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        let entity_id = session.state().local.entity_id;
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityEffect(s2c::EntityEffect {
                entity_id,
                effect_id: 8,
                amplifier: 1,
                duration: 600,
                show_particles: true,
            })))
            .unwrap();
        session.poll(16);
        session.state_mut().local.held_slot = 3;
        session.update_controller();
        let sprint = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };
        session.tick(&sprint);
        let before = session.predictor().unwrap().state;
        assert!(before.sprinting, "the setup sprints and has reported it");
        let _ = drain_outbound(&mut to_net);

        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        session.poll(16);
        let kept = session.predictor().unwrap().state;
        assert_eq!(
            (kept.pos, kept.yaw),
            (before.pos, before.yaw),
            "same player, same place"
        );
        assert_eq!(
            (kept.vel, kept.pitch),
            (DVec3::ZERO, 0.0),
            "preparePlayerToSpawn"
        );
        assert!(kept.sprinting, "the kept watcher still reports a sprint");
        assert_eq!(kept.aabb(), {
            let mut rebuilt = kept;
            rebuilt.set_position(kept.pos);
            rebuilt.aabb()
        });
        let _ = drain_outbound(&mut to_net);

        session.tick(&sprint);
        let sent = drain_outbound(&mut to_net);
        assert!(
            !sent.iter().any(|packet| matches!(
                packet,
                C2SPlay::EntityAction(c2s::EntityAction {
                    action: c2s::EntityActionType::StartSprinting,
                    ..
                })
            )),
            "serverSprintState survives the JoinGame: {sent:?}"
        );
        assert!(
            matches!(
                sent.first(),
                Some(C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 3 }))
            ),
            "the new controller re-syncs the kept slot: {sent:?}"
        );
        assert_eq!(
            session.predictor().unwrap().state.jump_boost,
            Some(1),
            "activePotionsMap survives"
        );
    }

    /// java: `renderViewEntity` keeps a removed S43 target
    /// (only a null one falls back to `thePlayer`), so `isCurrentViewEntity()` stays false and
    /// `onUpdateWalkingPlayer` sends no C03 until the server's next S43. Negative control: the
    /// old fallback to the local player resumed the movement reports at the S13.
    #[test]
    fn a_removed_camera_target_keeps_suppressing_movement_reports_until_the_next_s43() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                77,
                TrackedKind::Type(mc_data::EntityType::Pig),
                None,
                [0, 64 * 32, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        let send = |packet: S2CPlay| to_game.try_send(ToGame::Packet(packet)).unwrap();
        send(S2CPlay::Camera(s2c::Camera { entity_id: 77 }));
        send(S2CPlay::DestroyEntities(s2c::DestroyEntities {
            entity_ids: vec![77],
        }));
        session.poll(16);
        let _ = drain_outbound(&mut to_net);
        session.tick(&MoveInput::default());
        assert!(
            !drain_outbound(&mut to_net).iter().any(is_movement),
            "the removed camera entity is still the view entity"
        );
        assert!(
            session.render_view_is_local(),
            "drawing falls back to the local player"
        );

        send(S2CPlay::Camera(s2c::Camera {
            entity_id: session.state().local.entity_id,
        }));
        session.poll(16);
        session.tick(&MoveInput::default());
        assert!(
            drain_outbound(&mut to_net).iter().any(is_movement),
            "an S43 back to the player resumes the reports"
        );
    }

    /// java: `EntityLivingBase.onLivingUpdate`
    /// (`:1955-1965`) steps a local S14/S18 target in double precision with
    /// `wrapAngleTo180_double`, narrows once, and `setRotation` keeps the yaw `% 360.0F`.
    /// Negative control: the f32 step without the remainder left the yaw past 360.
    #[test]
    fn local_interpolation_steps_rotation_in_double_and_wraps_it_like_set_rotation() {
        let (mut session, _to_game, _to_net) = joined_and_standing();
        let pos = session.predictor().unwrap().state.pos;
        session.predictor_mut().unwrap().state.yaw = 350.0;
        session.predictor_mut().unwrap().state.pitch = 10.0;
        {
            let local = &mut session.state_mut().local;
            local.target_pos = pos;
            local.target_yaw = 14.0625;
            local.target_pitch = -20.0;
            local.interpolation_steps = 3;
        }
        // The same expression vanilla evaluates, step by step.
        let java = |yaw: f32, pitch: f32, steps: f64| {
            let d3 = mc_math::wrap_degrees_f64(14.0625_f64 - f64::from(yaw));
            let yaw = (f64::from(yaw) + d3 / steps) as f32;
            let pitch = (f64::from(pitch) + (-20.0_f64 - f64::from(pitch)) / steps) as f32;
            (yaw % 360.0, pitch % 360.0)
        };
        let (mut yaw, mut pitch) = (350.0_f32, 10.0_f32);
        for steps in [3.0, 2.0] {
            session.apply_local_entity_interpolation();
            (yaw, pitch) = java(yaw, pitch, steps);
            let state = session.predictor().unwrap().state;
            assert_eq!((state.yaw, state.pitch), (yaw, pitch));
        }
        assert!(
            (0.0..360.0).contains(&yaw),
            "the second step crosses 360 and setRotation wraps it: {yaw}"
        );
    }

    /// The local S14/S18 step turns `rotationYaw/rotationPitch` inside
    /// the tick (`EntityLivingBase.onLivingUpdate:1955-1965`), and vanilla reads the turned value
    /// for the rest of it: the walking report carries it and it stays the player's rotation.
    /// Physics used to copy the tick's pre-step look over it. Negative control: the report and
    /// the next tick's rotation were the input's 10/0.
    #[test]
    fn a_local_interpolation_step_survives_the_tick_it_runs_in() {
        let (mut session, _to_game, mut to_net) = joined_and_standing();
        let _ = drain_outbound(&mut to_net);
        let pos = session.predictor().unwrap().state.pos;
        session.predictor_mut().unwrap().state.yaw = 10.0;
        session.predictor_mut().unwrap().state.pitch = 0.0;
        {
            let local = &mut session.state_mut().local;
            local.target_pos = pos;
            local.target_yaw = 40.0;
            local.target_pitch = 30.0;
            local.interpolation_steps = 3;
        }
        // The driver samples the player's own rotation as the tick's look (`local_look`).
        let look = MoveInput {
            yaw: 10.0,
            pitch: 0.0,
            ..MoveInput::default()
        };
        let before = session.predictor().unwrap().state;
        session.tick(&look);
        let want_yaw = (10.0_f64 + mc_math::wrap_degrees_f64(40.0 - 10.0) / 3.0) as f32 % 360.0;
        let want_pitch = (0.0_f64 + 30.0 / 3.0) as f32 % 360.0;
        let state = session.predictor().unwrap().state;
        assert_eq!((state.yaw, state.pitch), (want_yaw, want_pitch));
        // Negative control: the old wiring stepped the predictor and then ticked physics with the
        // pre-step look, which copies 10/0 straight back over the step.
        let mut old = before;
        old.yaw = want_yaw;
        old.pitch = want_pitch;
        mc_physics::tick_player(&session.state().world, &mut old, &look);
        assert_eq!((old.yaw, old.pitch), (10.0, 0.0));
        assert_eq!(session.state().local.interpolation_steps, 2);
        let reported = drain_outbound(&mut to_net)
            .into_iter()
            .find_map(|packet| match packet {
                C2SPlay::PlayerLook(look) => Some((look.yaw, look.pitch)),
                C2SPlay::PlayerPositionAndLook(look) => Some((look.yaw, look.pitch)),
                _ => None,
            });
        assert_eq!(
            reported,
            Some((want_yaw, want_pitch)),
            "the C05/C06 carries the step"
        );
    }

    /// F-P1-entity-10-3 / F-P1-entity.ai.attributes-01-5 / -02-4: both the registered
    /// unlocalized name and RangedAttribute description resolve through LowerStringMap.
    #[test]
    fn movement_speed_attribute_lookup_is_case_insensitive_and_accepts_its_description() {
        use mc_protocol::play::s2c::{AttributePropertySnapshot, EntityProperties};

        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        let snapshot = |key: &str, base_value| {
            S2CPlay::EntityProperties(EntityProperties {
                entity_id,
                properties: vec![AttributePropertySnapshot {
                    key: key.into(),
                    base_value,
                    modifiers: Vec::new(),
                }],
            })
        };

        to_game
            .try_send(ToGame::Packet(snapshot("GENERIC.MOVEMENTSPEED", 0.23)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.effects.movement_speed, Some(0.23));

        // Case-insensitivity must not turn into a loose substring/"generic.*" match.
        to_game
            .try_send(ToGame::Packet(snapshot("GENERIC.MAXHEALTH", 0.9)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.effects.movement_speed, Some(0.23));

        to_game
            .try_send(ToGame::Packet(snapshot("mOvEmEnT sPeEd", 0.31)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.effects.movement_speed, Some(0.31));
        session.push_server_movement_state();
        assert_eq!(session.predictor().unwrap().state.movement_speed, 0.31);
    }

    /// Unknown names stay in the generation-local store so a client extension can consume them,
    /// while later packets still apply normally. The store deliberately does not collapse the
    /// packet to the small set of vanilla aliases.
    #[test]
    fn an_unknown_s20_attribute_is_retained_and_later_packets_still_apply() {
        use mc_protocol::play::s2c::{AttributePropertySnapshot, EntityProperties};

        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        let snapshot = |key: &str, base_value| {
            S2CPlay::EntityProperties(EntityProperties {
                entity_id,
                properties: vec![AttributePropertySnapshot {
                    key: key.into(),
                    base_value,
                    modifiers: Vec::new(),
                }],
            })
        };

        to_game
            .try_send(ToGame::Packet(snapshot("unknown.parityProbe", 99.0)))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(snapshot("generic.movementSpeed", 0.42)))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnExperienceOrb(
                s2c::SpawnExperienceOrb {
                    entity_id: 91,
                    x: 0,
                    y: 0,
                    z: 0,
                    xp_value: 1,
                },
            )))
            .unwrap();

        let events = session.poll(16);
        assert!(
            session.phase().is_live(),
            "a handler error is not a disconnect"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            ClientEvent::EntityProperties(properties)
                if properties.properties.iter().any(|property| property.key == "unknown.parityProbe")
        )));
        assert_eq!(
            session
                .effects
                .attributes
                .get("unknown.parityProbe")
                .unwrap()
                .resolved_value(),
            99.0
        );
        assert_eq!(session.effects.movement_speed, Some(0.42));
        assert!(session.state().entities.get(91).is_some());
    }

    /// java: `EntityPlayer.onLivingUpdate:620-625` pushes `capabilities.walkSpeed`
    /// into the `generic.movementSpeed` attribute inside `if (!this.worldObj.isRemote)`, so on a
    /// **client** a server-set walk speed changes nothing and `S20PacketEntityProperties` is the
    /// only thing that does. Both halves are asserted, because the audit's fix sketch had it the
    /// other way round.
    #[test]
    fn movement_speed_comes_from_the_attribute_packet_and_not_from_walk_speed() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation, AttributeModifierSnapshot, AttributePropertySnapshot,
            EntityProperties,
        };
        let (mut session, to_game, _to_net) = joined_and_standing();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        let walk = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        for _ in 0..60 {
            session.tick(&walk);
        }
        let baseline = session.predictor().unwrap().state.ai_move_speed();
        assert_eq!(baseline, 0.1f32, "the registered attribute base");

        // A server-set walk speed of 0.3 — decoded, stored, and correctly inert.
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerAbilities(
                s2c::PlayerAbilities {
                    flags: 0,
                    fly_speed: 0.05,
                    walk_speed: 0.3,
                },
            )))
            .unwrap();
        session.poll(16);
        session.tick(&walk);
        assert_eq!(
            session.state().local.abilities.walk_speed,
            0.3,
            "it is still decoded and stored"
        );
        assert_eq!(
            session.predictor().unwrap().state.ai_move_speed(),
            baseline,
            "but a client never routes walkSpeed into the movementSpeed attribute"
        );

        // Speed II: base 0.1 with a `+0.4` operation-2 modifier.
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityProperties(
                EntityProperties {
                    entity_id: session.state().local.entity_id,
                    properties: vec![AttributePropertySnapshot {
                        key: "generic.movementSpeed".into(),
                        base_value: 0.1,
                        modifiers: vec![AttributeModifierSnapshot {
                            uuid: uuid::Uuid::from_u128(7),
                            amount: 0.4,
                            operation: AttributeModifierOperation::Multiply,
                        }],
                    }],
                },
            )))
            .unwrap();
        session.poll(16);
        session.tick(&walk);
        assert_eq!(
            session.predictor().unwrap().state.ai_move_speed(),
            (0.1f32 as f64 * 1.4) as f32,
            "Speed II reaches movement through the attribute packet"
        );
    }

    /// F-P0-network.play.server-08-3, plus the Session-owned half of F-P1-potion-01-3:
    /// replacing EntityPlayerSP drops its attribute map and active potion map. The server does
    /// not need to send removal packets for state belonging to the discarded entity.
    #[test]
    fn respawn_and_fresh_join_clear_local_movement_attributes_and_effects() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation, AttributeModifierSnapshot, AttributePropertySnapshot,
            EntityProperties,
        };
        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        let speed_packet = || {
            S2CPlay::EntityProperties(EntityProperties {
                entity_id,
                properties: vec![AttributePropertySnapshot {
                    key: "generic.movementSpeed".into(),
                    base_value: 0.1,
                    modifiers: vec![AttributeModifierSnapshot {
                        uuid: uuid::Uuid::from_u128(7),
                        amount: 0.4,
                        operation: AttributeModifierOperation::Multiply,
                    }],
                }],
            })
        };
        for (effect_id, amplifier) in [(8, 2), (15, 0)] {
            to_game
                .try_send(ToGame::Packet(S2CPlay::EntityEffect(s2c::EntityEffect {
                    entity_id,
                    effect_id,
                    amplifier,
                    duration: 3_600,
                    show_particles: false,
                })))
                .unwrap();
        }
        to_game.try_send(ToGame::Packet(speed_packet())).unwrap();
        session.poll(16);
        session.set_armor_depth_strider(3.0);
        assert_eq!(session.effects.jump_boost, Some((2, 3_600)));
        assert_eq!(session.effects.blindness, Some((0, 3_600)));
        assert!(session.effects.movement_speed.is_some());
        assert_eq!(session.effects.depth_strider, 3.0);

        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert_eq!(session.effects, MovementEffects::default());
        session.push_server_movement_state();
        let state = session.predictor().unwrap().state;
        assert_eq!(state.movement_speed, mc_physics::state::WALK_SPEED);
        assert_eq!(state.jump_boost, None);
        assert!(!state.blindness);
        assert_eq!(state.depth_strider, 0.0);

        // A new-life packet must still apply; a test that simply stopped consuming S20/S1D
        // would otherwise satisfy every reset assertion above.
        to_game.try_send(ToGame::Packet(speed_packet())).unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityEffect(s2c::EntityEffect {
                entity_id,
                effect_id: 8,
                amplifier: 1,
                duration: 200,
                show_particles: false,
            })))
            .unwrap();
        session.poll(16);
        session.push_server_movement_state();
        assert!(session.predictor().unwrap().state.movement_speed > 0.1);
        assert_eq!(session.predictor().unwrap().state.jump_boost, Some(1));

        // A proxy's mid-connection JoinGame does *not* build a fresh player: `loadWorld` keeps
        // `thePlayer` (and with it the attribute map and activePotionsMap) while it is non-null
        // (found by L4). The S07 a proxy sends next is what replaces it.
        let before_join = session.effects.clone();
        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        session.poll(16);
        assert_eq!(session.effects, before_join);
        to_game
            .try_send(ToGame::Packet(S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })))
            .unwrap();
        session.poll(16);
        assert_eq!(session.effects, MovementEffects::default());
    }

    /// F-P1-potion-01-4: a remote-world client floors a completed effect at zero but cannot
    /// remove it itself.  The explicit S1E packet, rather than the second local tick, is what
    /// makes `isPotionActive` false.
    #[test]
    fn local_potion_effects_remain_active_at_zero_until_remove_packet() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        for (effect_id, amplifier) in [(8, 2), (15, 0)] {
            to_game
                .try_send(ToGame::Packet(S2CPlay::EntityEffect(s2c::EntityEffect {
                    entity_id,
                    effect_id,
                    amplifier,
                    duration: 1,
                    show_particles: false,
                })))
                .unwrap();
        }
        session.poll(16);

        session.tick_movement_effects();
        assert_eq!(session.effects.jump_boost, Some((2, 0)));
        assert_eq!(session.effects.blindness, Some((0, 0)));
        session.push_server_movement_state();
        assert_eq!(session.predictor().unwrap().state.jump_boost, Some(2));
        assert!(session.predictor().unwrap().state.blindness);

        // A further client tick neither underflows nor removes the zero-duration entries.
        session.tick_movement_effects();
        assert_eq!(session.effects.jump_boost, Some((2, 0)));
        assert_eq!(session.effects.blindness, Some((0, 0)));

        for effect_id in [8, 15] {
            to_game
                .try_send(ToGame::Packet(S2CPlay::RemoveEntityEffect(
                    s2c::RemoveEntityEffect {
                        entity_id,
                        effect_id,
                    },
                )))
                .unwrap();
        }
        session.poll(16);
        session.push_server_movement_state();
        assert_eq!(session.effects.jump_boost, None);
        assert_eq!(session.effects.blindness, None);
        assert_eq!(session.predictor().unwrap().state.jump_boost, None);
        assert!(!session.predictor().unwrap().state.blindness);
    }

    /// F-P1-potion-01-7: `handleEntityEffect` calls `addPotionEffect`, which combines an
    /// existing effect instead of replacing it. The branches distinguish stronger/shorter,
    /// equal/longer, and weaker/longer packets; wholesale assignment fails the latter two.
    #[test]
    fn repeated_entity_effect_packets_combine_like_potion_effect() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        let send = |effect_id, amplifier, duration| {
            to_game
                .try_send(ToGame::Packet(S2CPlay::EntityEffect(s2c::EntityEffect {
                    entity_id,
                    effect_id,
                    amplifier,
                    duration,
                    show_particles: false,
                })))
                .unwrap();
        };

        send(8, 1, 600);
        session.poll(16);
        // Higher strength replaces even with a shorter duration.
        send(8, 2, 20);
        session.poll(16);
        assert_eq!(session.effects.jump_boost, Some((2, 20)));

        // Equal strength only takes a longer duration.
        send(8, 2, 19);
        session.poll(16);
        assert_eq!(session.effects.jump_boost, Some((2, 20)));
        send(8, 2, 700);
        session.poll(16);
        assert_eq!(session.effects.jump_boost, Some((2, 700)));

        // A weaker effect is ignored regardless of its duration. This proves blindness must
        // retain its otherwise-unused amplifier to use the same merge rule.
        send(8, 1, 10_000);
        send(15, 2, 20);
        session.poll(16);
        assert_eq!(session.effects.jump_boost, Some((2, 700)));
        assert_eq!(session.effects.blindness, Some((2, 20)));
        send(15, 1, 10_000);
        session.poll(16);
        assert_eq!(session.effects.blindness, Some((2, 20)));
    }

    /// `S1DPacketEntityEffect` carries the amplifier as a signed byte. Preserve the exact Java
    /// values at the session boundary for ordinary `127` and the wire values `128`/`255`, which
    /// arrive as `-128`/`-1`; a zero-effect control is removed by the explicit S1E packet rather
    /// than being silently converted into an ordinary level-I boost.
    #[test]
    fn jump_boost_keeps_signed_wire_amplifiers_at_the_local_player_boundary() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        for amplifier in [127_i8, -128_i8, -1_i8] {
            to_game
                .try_send(ToGame::Packet(S2CPlay::EntityEffect(s2c::EntityEffect {
                    entity_id,
                    effect_id: 8,
                    amplifier,
                    duration: 200,
                    show_particles: false,
                })))
                .unwrap();
            session.poll(16);
            assert_eq!(session.effects.jump_boost, Some((amplifier, 200)));
            session.push_server_movement_state();
            assert_eq!(
                session.predictor().unwrap().state.jump_boost,
                Some(amplifier),
                "signed amplifier {amplifier} must not clamp to zero"
            );
            session.effects.jump_boost = None;
        }

        to_game
            .try_send(ToGame::Packet(S2CPlay::RemoveEntityEffect(
                s2c::RemoveEntityEffect {
                    entity_id,
                    effect_id: 8,
                },
            )))
            .unwrap();
        session.poll(16);
        session.push_server_movement_state();
        assert_eq!(session.effects.jump_boost, None);
        assert_eq!(session.predictor().unwrap().state.jump_boost, None);
    }

    #[test]
    fn the_canonical_sprint_modifier_is_not_folded_into_server_movement_speed() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation as Op, AttributeModifierSnapshot as Modifier,
        };
        let modifiers = [
            Modifier {
                uuid: SPRINT_SPEED_MODIFIER_UUID,
                amount: 0.3,
                operation: Op::Multiply,
            },
            Modifier {
                uuid: uuid::Uuid::from_u128(7),
                amount: 0.4,
                operation: Op::Multiply,
            },
        ];
        assert!(
            (resolve_movement_attribute(0.1, &modifiers)
                .unwrap()
                .value_without_sprint
                - 0.14)
                .abs()
                < 1.0e-12
        );
        assert!(
            (resolve_attribute(0.1, &modifiers).unwrap() - 0.182).abs() < 1.0e-12,
            "negative control: retaining the S20 sprint UUID would multiply by 1.3 twice"
        );
    }

    #[test]
    fn operation_two_uses_java_uuid_hash_set_order_not_s20_wire_order() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation as Op, AttributeModifierSnapshot as Modifier,
        };

        // The wire puts UUID 18 before UUID 1.  Java's initial 16-bucket
        // HashSet visits UUID 1 first, then its bucket-2 chain (18, 2).  The
        // amounts deliberately make those otherwise-commutative products land
        // on adjacent f64 values when evaluated in the wrong order.
        let wire = [
            Modifier {
                uuid: uuid::Uuid::from_u128(18),
                amount: 4.004_807_362_210_215,
                operation: Op::Multiply,
            },
            Modifier {
                uuid: uuid::Uuid::from_u128(1),
                amount: 1.625_183_354_820_274_8,
                operation: Op::Multiply,
            },
            Modifier {
                uuid: uuid::Uuid::from_u128(2),
                amount: 3.549_961_541_408_507,
                operation: Op::Multiply,
            },
        ];
        let java =
            ((0.1_f64 * 2.625_183_354_820_274_8) * 5.004_807_362_210_215) * 4.549_961_541_408_507;
        let wire_order =
            ((0.1_f64 * 5.004_807_362_210_215) * 2.625_183_354_820_274_8) * 4.549_961_541_408_507;
        let reordered = [wire[1], wire[0], wire[2]];

        assert_eq!(
            resolve_attribute(0.1, &wire).unwrap().to_bits(),
            java.to_bits()
        );
        assert_eq!(
            resolve_attribute(0.1, &wire).unwrap().to_bits(),
            resolve_attribute(0.1, &reordered).unwrap().to_bits(),
            "S20 serializes a HashSet in arbitrary order"
        );
        assert_ne!(
            java.to_bits(),
            wire_order.to_bits(),
            "rounding mutant must bite"
        );
    }

    #[test]
    fn operation_zero_and_one_use_java_uuid_hash_set_order_and_sequential_adds() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation as Op, AttributeModifierSnapshot as Modifier,
        };

        // UUID 18 is serialized before UUID 1, but Java's initial HashSet buckets visit UUID 1
        // first (`java_uuid_hash_set_order` returns [1, 0, 2]). The magnitudes make a wire-order
        // fold differ by one f64 bit, so replacing the old `sum()` cannot hide behind tolerance.
        let add_wire = [
            Modifier {
                uuid: uuid::Uuid::from_u128(18),
                amount: -100_000_000.0,
                operation: Op::Add,
            },
            Modifier {
                uuid: uuid::Uuid::from_u128(1),
                amount: 5.139_593_396_247_941,
                operation: Op::Add,
            },
            Modifier {
                uuid: uuid::Uuid::from_u128(2),
                amount: -0.000_001,
                operation: Op::Add,
            },
        ];
        let add_java = ((0.1_f64 + 5.139_593_396_247_941) - 100_000_000.0) - 0.000_001;
        let add_wire_order = ((0.1_f64 - 100_000_000.0) + 5.139_593_396_247_941) - 0.000_001;
        let add_reordered = [add_wire[1], add_wire[0], add_wire[2]];
        assert_eq!(
            resolve_attribute(0.1, &add_wire).unwrap().to_bits(),
            add_java.to_bits()
        );
        assert_eq!(
            resolve_attribute(0.1, &add_wire).unwrap().to_bits(),
            resolve_attribute(0.1, &add_reordered).unwrap().to_bits()
        );
        assert_ne!(
            add_java.to_bits(),
            add_wire_order.to_bits(),
            "operation-0 wire-order mutant must bite"
        );

        let multiply_base_wire = [
            Modifier {
                uuid: uuid::Uuid::from_u128(18),
                amount: -1_000.0,
                operation: Op::MultiplyBase,
            },
            Modifier {
                uuid: uuid::Uuid::from_u128(1),
                amount: 0.000_01,
                operation: Op::MultiplyBase,
            },
            Modifier {
                uuid: uuid::Uuid::from_u128(2),
                amount: 9.659_309_795_323_193,
                operation: Op::MultiplyBase,
            },
        ];
        let multiply_base_java =
            ((0.1_f64 + 0.1 * 0.000_01) + 0.1 * -1_000.0) + 0.1 * 9.659_309_795_323_193;
        let multiply_base_wire_order =
            ((0.1_f64 + 0.1 * -1_000.0) + 0.1 * 0.000_01) + 0.1 * 9.659_309_795_323_193;
        let multiply_base_reordered = [
            multiply_base_wire[1],
            multiply_base_wire[0],
            multiply_base_wire[2],
        ];
        assert_eq!(
            resolve_attribute(0.1, &multiply_base_wire)
                .unwrap()
                .to_bits(),
            multiply_base_java.to_bits()
        );
        assert_eq!(
            resolve_attribute(0.1, &multiply_base_wire)
                .unwrap()
                .to_bits(),
            resolve_attribute(0.1, &multiply_base_reordered)
                .unwrap()
                .to_bits()
        );
        assert_ne!(
            multiply_base_java.to_bits(),
            multiply_base_wire_order.to_bits(),
            "operation-1 wire-order mutant must bite"
        );
    }

    #[test]
    fn movement_modifier_uuid_duplicates_are_rejected_before_replacing_the_snapshot() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation as Op, AttributeModifierSnapshot as Modifier,
            AttributePropertySnapshot, EntityProperties,
        };

        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        let modifier = |amount| Modifier {
            uuid: uuid::Uuid::from_u128(7),
            amount,
            operation: Op::Multiply,
        };
        let packet = |base_value, modifiers| {
            S2CPlay::EntityProperties(EntityProperties {
                entity_id,
                properties: vec![AttributePropertySnapshot {
                    key: "generic.movementSpeed".into(),
                    base_value,
                    modifiers,
                }],
            })
        };

        to_game
            .try_send(ToGame::Packet(packet(0.1, vec![modifier(0.4)])))
            .unwrap();
        session.poll(16);
        let before = session.effects.clone();
        assert_eq!(
            before
                .movement_modifiers
                .as_ref()
                .expect("valid snapshot installs a UUID map")
                .modifiers
                .len(),
            1
        );

        // Java's second applyModifier throws because equality is UUID-only. A scalar resolver
        // that blindly folds the vector would multiply by 1.4 * 1.5 and mutate this state.
        to_game
            .try_send(ToGame::Packet(packet(
                0.1,
                vec![modifier(0.4), modifier(0.5)],
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(
            session.effects, before,
            "a duplicate UUID must not stack or partially replace the prior map"
        );
        assert!(matches!(
            resolve_attribute(0.1, &[modifier(0.4), modifier(0.5)]),
            Err(AttributeResolutionError::DuplicateUuid(uuid)) if uuid == uuid::Uuid::from_u128(7)
        ));
    }

    #[test]
    fn local_max_health_snapshots_replace_reject_duplicates_clamp_and_bound_current_health() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation as Op, AttributeModifierSnapshot as Modifier,
            AttributePropertySnapshot, EntityProperties,
        };

        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        let modifier = |amount| Modifier {
            uuid: uuid::Uuid::from_u128(7),
            amount,
            operation: Op::Multiply,
        };
        let packet = |base_value, modifiers| {
            S2CPlay::EntityProperties(EntityProperties {
                entity_id,
                properties: vec![AttributePropertySnapshot {
                    key: "generic.maxHealth".into(),
                    base_value,
                    modifiers,
                }],
            })
        };

        to_game
            .try_send(ToGame::Packet(packet(20.0, vec![modifier(1.0)])))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.max_health, 40.0);
        assert_eq!(session.state().local.health, 20.0);
        let first = session
            .effects
            .max_health
            .clone()
            .expect("valid max-health snapshot installs its UUID map");

        session.state.local.apply_server_health(35.0);
        assert_eq!(session.state().local.health, 35.0);

        // removeAllModifiers + applyModifier replaces the complete map. Omitting UUID 7 removes
        // its multiplier and lowering max health immediately bounds the living value.
        to_game
            .try_send(ToGame::Packet(packet(30.0, Vec::new())))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.max_health, 30.0);
        assert_eq!(session.state().local.health, 30.0);
        assert_ne!(session.effects.max_health.as_ref(), Some(&first));
        let before_bad = session.effects.clone();

        // UUID equality is the rejection boundary. A flat-vector mutant would accept both and
        // multiply the scalar twice, partially replacing the last good snapshot.
        to_game
            .try_send(ToGame::Packet(packet(
                20.0,
                vec![modifier(0.5), modifier(0.5)],
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.effects, before_bad);
        assert_eq!(session.state().local.max_health, 30.0);
        assert_eq!(session.state().local.health, 30.0);

        to_game
            .try_send(ToGame::Packet(packet(2_048.0, Vec::new())))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.max_health, 1_024.0);
        to_game
            .try_send(ToGame::Packet(packet(-1.0, Vec::new())))
            .unwrap();
        session.poll(16);
        assert_eq!(session.state().local.max_health, 0.0);
        assert_eq!(session.state().local.health, 0.0);
    }

    #[test]
    fn movement_modifier_snapshot_replaces_the_map_and_can_remove_local_sprint() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation as Op, AttributeModifierSnapshot as Modifier,
            AttributePropertySnapshot, EntityProperties,
        };

        let (mut session, to_game, _to_net) = joined_and_standing();
        let entity_id = session.state().local.entity_id;
        let canonical = Modifier {
            uuid: SPRINT_SPEED_MODIFIER_UUID,
            amount: mc_physics::state::SPRINT_SPEED_BOOST,
            operation: Op::Multiply,
        };
        let server_modifier = |amount| Modifier {
            uuid: uuid::Uuid::from_u128(7),
            amount,
            operation: Op::Multiply,
        };
        let packet = |base_value, modifiers| {
            S2CPlay::EntityProperties(EntityProperties {
                entity_id,
                properties: vec![AttributePropertySnapshot {
                    key: "generic.movementSpeed".into(),
                    base_value,
                    modifiers,
                }],
            })
        };

        to_game
            .try_send(ToGame::Packet(packet(
                0.1,
                vec![canonical, server_modifier(0.4)],
            )))
            .unwrap();
        session.poll(16);
        let first = session
            .effects
            .movement_modifiers
            .as_ref()
            .expect("first snapshot installs a UUID map");
        assert_eq!(first.modifiers.len(), 2);
        assert_eq!(
            first.sprint_amount(),
            Some(mc_physics::state::SPRINT_SPEED_BOOST)
        );

        // A later S20 is removeAllModifiers + applyModifier, not an additive merge. Omitting the
        // canonical UUID therefore removes the local sprint entry even while the bool is true.
        to_game
            .try_send(ToGame::Packet(packet(0.1, vec![server_modifier(0.4)])))
            .unwrap();
        session.poll(16);
        let second = session
            .effects
            .movement_modifiers
            .as_ref()
            .expect("replacement snapshot remains stored");
        assert_eq!(second.modifiers.len(), 1);
        assert_eq!(second.sprint_amount(), None);

        let predictor = session.predictor_mut().expect("joined predictor");
        predictor.state.sprinting = true;
        predictor.state.sprinting_ticks_left = 100;
        session.push_server_movement_state();
        let state = session.predictor().unwrap().state;
        assert_eq!(state.sprint_modifier_amount, None);
        assert_eq!(state.ai_move_speed(), (0.1_f64 * 1.4) as f32);

        // A later local setSprinting edge re-adds the canonical modifier in Java. A following
        // ordinary tick must retain that edge until another S20 replaces the attribute map;
        // replaying the stale server snapshot here would incorrectly erase it again.
        session
            .predictor_mut()
            .expect("joined predictor")
            .state
            .sprint_modifier_amount = Some(mc_physics::state::SPRINT_SPEED_BOOST);
        // An unrelated S20 must not replay the old movement snapshot either. Java mutates only
        // the named max-health attribute, so this local sprint edge remains installed.
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityProperties(
                EntityProperties {
                    entity_id,
                    properties: vec![AttributePropertySnapshot {
                        key: "generic.maxHealth".into(),
                        base_value: 20.0,
                        modifiers: Vec::new(),
                    }],
                },
            )))
            .unwrap();
        session.poll(16);
        session.push_server_movement_state();
        let state = session.predictor().unwrap().state;
        assert_eq!(
            state.sprint_modifier_amount,
            Some(mc_physics::state::SPRINT_SPEED_BOOST)
        );
        assert_eq!(state.ai_move_speed(), (0.1_f64 * 1.4 * 1.3) as f32);
    }

    #[test]
    fn local_sprint_tick_keeps_its_java_attribute_edge_until_movement_replacement() {
        use mc_protocol::play::s2c::{AttributePropertySnapshot, EntityProperties};

        let (mut session, to_game, _to_net) = joined_and_standing();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        let entity_id = session.state().local.entity_id;
        let movement = |modifiers| {
            S2CPlay::EntityProperties(EntityProperties {
                entity_id,
                properties: vec![AttributePropertySnapshot {
                    key: "generic.movementSpeed".into(),
                    base_value: 0.1,
                    modifiers,
                }],
            })
        };

        // Java's initial S20 can contain an empty movement map. The next local sprint edge is
        // applied by EntityLivingBase.setSprinting during the actual Session::tick.
        to_game
            .try_send(ToGame::Packet(movement(Vec::new())))
            .unwrap();
        session.poll(16);
        let sprint = MoveInput {
            forward: true,
            sprint: true,
            yaw: 0.0,
            ..MoveInput::default()
        };
        let before = session.predictor().unwrap().state.pos;
        session.tick(&sprint);
        let after = session.predictor().unwrap().state;
        assert_eq!(
            after.sprint_modifier_amount,
            Some(mc_physics::state::SPRINT_SPEED_BOOST)
        );
        assert_eq!(after.ai_move_speed(), (0.1_f64 * 1.3) as f32);
        assert!(
            after.pos.z > before.z,
            "forward sprint tick must move the player"
        );
        session.tick(&sprint);
        assert_eq!(
            session.predictor().unwrap().state.sprint_modifier_amount,
            Some(mc_physics::state::SPRINT_SPEED_BOOST),
            "an ordinary tick must not replay the old empty server snapshot"
        );

        // An unrelated S20 mutates only its named Java attribute and must preserve that local
        // modifier. A later movement S20 without the canonical UUID then removes it.
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityProperties(
                EntityProperties {
                    entity_id,
                    properties: vec![AttributePropertySnapshot {
                        key: "generic.maxHealth".into(),
                        base_value: 20.0,
                        modifiers: Vec::new(),
                    }],
                },
            )))
            .unwrap();
        session.poll(16);
        session.tick(&sprint);
        assert_eq!(
            session.predictor().unwrap().state.sprint_modifier_amount,
            Some(mc_physics::state::SPRINT_SPEED_BOOST)
        );

        to_game
            .try_send(ToGame::Packet(movement(Vec::new())))
            .unwrap();
        session.poll(16);
        assert_eq!(
            session.predictor().unwrap().state.sprint_modifier_amount,
            None
        );
    }

    /// java: `handleEntityMetadata` for the local player is
    /// `DataWatcher.updateWatchedObjectsFromList`, i.e. `Entity.setFlag(3, ..)` for the sprint
    /// bit. It never calls `setSprinting`, so the `movementSpeed` sprint entry and
    /// `sprintingTicksLeft` keep their values while `isSprinting()` (and with it the C0B edge,
    /// the sprint-jump kick and the next tick's air control) follows the echo. Oracle evidence:
    /// `mc-physics/tests/parity.rs` `sprint_flag_echo`/`sprint_expiry`. The pre-wave seam re-ran
    /// a whole `setSprinting` on every echo: an echoed `true` sprinted a walker at 1.3x (faster
    /// than vanilla) and an echoed `false` stripped the boost from a double-tap sprinter.
    #[test]
    fn a_sprint_flag_echo_moves_the_flag_but_not_the_attribute_entry() {
        use mc_physics::state::{SPRINT_SPEED_BOOST, WALK_SPEED};
        let echo = |entity_id, flags| {
            ToGame::Packet(S2CPlay::EntityMetadata(s2c::EntityMetadata {
                entity_id,
                metadata: vec![mc_protocol::MetadataEntry {
                    index: 0,
                    value: mc_protocol::MetadataValue::Byte(flags),
                }],
            }))
        };
        let sent_action = |sent: &[C2SPlay], action: c2s::EntityActionType| {
            sent.iter().any(|packet| {
                matches!(packet, C2SPlay::EntityAction(c2s::EntityAction { action: a, .. })
                    if *a == action)
            })
        };
        let forward = MoveInput {
            forward: true,
            ..MoveInput::default()
        };

        // A double-tap sprinter (no sprint key) receives a stale `false` echo.
        let (mut session, to_game, mut to_net) = joined_and_standing();
        for _ in 0..20 {
            session.tick(&MoveInput::default());
        }
        session.tick(&forward);
        session.tick(&MoveInput::default());
        for _ in 0..6 {
            session.tick(&forward);
        }
        let before = session.predictor().unwrap().state;
        assert!(before.sprinting, "double-tap W engaged the sprint");
        assert_eq!(before.sprint_modifier_amount, Some(SPRINT_SPEED_BOOST));
        let _ = drain_outbound(&mut to_net);
        let entity_id = session.state().local.entity_id;
        to_game.try_send(echo(entity_id, 0)).unwrap();
        session.poll(16);
        session.tick(&forward);
        let after = session.predictor().unwrap().state;
        assert!(
            !after.sprinting,
            "W still held: the double-tap gate cannot re-engage"
        );
        assert_eq!(
            after.sprint_modifier_amount,
            Some(SPRINT_SPEED_BOOST),
            "the echo leaves the attribute entry"
        );
        assert_eq!(
            after.ai_move_speed(),
            (WALK_SPEED * (1.0 + SPRINT_SPEED_BOOST)) as f32,
            "vanilla keeps the sprint speed under a cleared flag"
        );
        assert_eq!(
            after.sprinting_ticks_left,
            before.sprinting_ticks_left - 1,
            "the 600-tick countdown is not reset by the echo"
        );
        let sent = drain_outbound(&mut to_net);
        assert!(
            sent_action(&sent, c2s::EntityActionType::StopSprinting),
            "isSprinting() is false again, so the edge reports STOP: {sent:?}"
        );

        // A walker receives a stale `true` echo.
        let (mut session, to_game, mut to_net) = joined_and_standing();
        for _ in 0..30 {
            session.tick(&forward);
        }
        let _ = drain_outbound(&mut to_net);
        assert!(!session.predictor().unwrap().state.sprinting);
        let entity_id = session.state().local.entity_id;
        to_game.try_send(echo(entity_id, 0x08)).unwrap();
        session.poll(16);
        session.tick(&forward);
        let after = session.predictor().unwrap().state;
        assert!(
            after.sprinting,
            "nothing stops a forward walker's echoed flag"
        );
        assert_eq!(
            after.sprint_modifier_amount, None,
            "no setSprinting, no entry"
        );
        assert_eq!(after.sprinting_ticks_left, 0);
        assert_eq!(
            after.ai_move_speed(),
            WALK_SPEED as f32,
            "an echoed flag alone must not grant the 1.3x boost"
        );
        let sent = drain_outbound(&mut to_net);
        assert!(
            sent_action(&sent, c2s::EntityActionType::StartSprinting),
            "isSprinting() differs from serverSprintState, so START is reported: {sent:?}"
        );
    }

    #[test]
    fn movement_attributes_for_a_non_local_entity_do_not_replace_local_state() {
        use mc_protocol::play::s2c::{
            AttributeModifierOperation as Op, AttributeModifierSnapshot as Modifier,
            AttributePropertySnapshot, EntityProperties,
        };

        let (mut session, to_game, _to_net) = joined_and_standing();
        let local_id = session.state().local.entity_id;
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityProperties(
                EntityProperties {
                    // Java applies this to any tracked EntityLivingBase. The local Session seam
                    // intentionally has no remote attribute store; this control ensures a remote
                    // snapshot cannot leak into the local predictor while that work is queued.
                    entity_id: local_id + 1,
                    properties: vec![AttributePropertySnapshot {
                        key: "generic.movementSpeed".into(),
                        base_value: 0.9,
                        modifiers: vec![Modifier {
                            uuid: uuid::Uuid::from_u128(99),
                            amount: 0.4,
                            operation: Op::Multiply,
                        }],
                    }],
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.effects.movement_speed, None);
        assert_eq!(session.effects.movement_modifiers, None);
    }

    #[test]
    fn tick_is_a_no_op_before_join_and_after_disconnect() {
        let (mut session, to_game, _to_net) = rig();
        assert!(session.tick(&MoveInput::default()).is_none());

        to_game.try_send(ToGame::Packet(join_game())).unwrap();
        // Give the live control real collision; chunk residency does not gate its tick.
        to_game
            .try_send(ToGame::Packet(chunk_with_a_stone_floor(0, 0, 64)))
            .unwrap();
        session.poll(16);
        // The player was spawned at JoinGame, before the column arrived: the first world tick
        // skips it (`addedToChunk`), the second reports.
        assert!(session.tick(&MoveInput::default()).is_none());
        assert!(session.tick(&MoveInput::default()).is_some());

        to_game
            .try_send(ToGame::Disconnected(DisconnectReason::ClosedByClient))
            .unwrap();
        session.poll(16);
        assert!(session.tick(&MoveInput::default()).is_none());
    }

    // ---- W-RIDE ---------------------------------------------------------------------------

    /// [`joined_and_standing`] with the join handshake already drained off the wire, so a test
    /// sees only what its own ticks produced. `join_game()`'s entity id is `3`.
    fn mountable_session() -> (
        Session,
        tokio::sync::mpsc::Sender<ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (session, to_game, mut to_net) = joined_and_standing();
        let _ = drain_outbound(&mut to_net);
        (session, to_game, to_net)
    }

    /// The three non-walking C0B producers must remain reachable from the live Session seam.
    /// The driver applies the vanilla predicates (sleep screen / horse mount) before invoking
    /// these helpers; this test focuses on the packet identity and payload at that boundary.
    #[test]
    fn session_emits_sleep_and_horse_entity_actions_with_the_local_id() {
        let (mut session, _to_game, mut to_net) = mountable_session();
        session.send_stop_sleeping();
        session.send_horse_inventory();

        let sent = drain_outbound(&mut to_net);
        assert!(matches!(
            sent.as_slice(),
            [
                C2SPlay::EntityAction(c2s::EntityAction {
                    entity_id: 3,
                    action: c2s::EntityActionType::StopSleeping,
                    aux_data: 0,
                }),
                C2SPlay::EntityAction(c2s::EntityAction {
                    entity_id: 3,
                    action: c2s::EntityActionType::OpenInventory,
                    aux_data: 0,
                }),
            ]
        ));
    }

    /// `S1BPacketEntityAttach` mounting `rider` on `vehicle` (`-1` dismounts).
    fn attach(rider: i32, vehicle: i32, attach_type: s2c::AttachType) -> S2CPlay {
        S2CPlay::EntityAttach(s2c::EntityAttach {
            entity_id: rider,
            vehicle_entity_id: vehicle,
            attach_type,
        })
    }

    /// Install a real world-backed vehicle for local S1B tests. The session-owned player is not
    /// in `EntityTracker`, so the attach must resolve this generation before it can mount.
    fn spawn_test_pig(session: &mut Session, network_id: i32) -> mc_world::Entity {
        let state = session.state_mut();
        state.entities.spawn(
            &mut state.world,
            network_id,
            TrackedKind::Type(mc_data::EntityType::Pig),
            None,
            [8 * 32, 65 * 32, 8 * 32],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::Metadata::new(),
        )
    }

    fn spawn_test_boat(session: &mut Session, network_id: i32) -> mc_world::Entity {
        let state = session.state_mut();
        state.entities.spawn(
            &mut state.world,
            network_id,
            TrackedKind::Type(mc_data::EntityType::Boat),
            None,
            [8 * 32, 69 * 32, 8 * 32],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::Metadata::new(),
        )
    }

    fn steers(packets: &[C2SPlay]) -> Vec<&mc_protocol::play::c2s::SteerVehicle> {
        packets
            .iter()
            .filter_map(|packet| match packet {
                C2SPlay::SteerVehicle(inner) => Some(inner),
                _ => None,
            })
            .collect()
    }

    /// **W-RIDE vehicle-input ordering.** java updates a loaded vehicle before its mounted
    /// rider. The boat therefore consumes the rider's persistent movement fields from the
    /// previous player tick; only after that does `EntityPlayerSP.onLivingUpdate` refresh them
    /// from `MovementInput`. A current-raw-key handoff would make the client boat move one tick
    /// before Java and is especially visible when a mount is received between ticks.
    #[test]
    fn an_occupied_boat_consumes_the_previous_player_movement_fields() {
        let (mut session, to_game, mut to_net) = mountable_session();
        spawn_test_boat(&mut session, 42);
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), Some(42));

        // The first mounted tick sees the old zeroed moveForward in Java. The raw forward key
        // is refreshed by the rider only after the boat's update has consumed that old field.
        let forward = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        session.tick(&forward);
        let _ = drain_outbound(&mut to_net);
        let first_velocity = session
            .state()
            .world
            .ecs
            .get::<&mc_world::Velocity>(session.state().entities.get(42).unwrap().entity)
            .unwrap()
            .0;
        assert_eq!(
            first_velocity.z, 0.0,
            "boat must not consume current raw input early"
        );

        // The second vehicle tick consumes the moveForward that the first rider tick installed.
        session.tick(&forward);
        let _ = drain_outbound(&mut to_net);
        let second_velocity = session
            .state()
            .world
            .ecs
            .get::<&mc_world::Velocity>(session.state().entities.get(42).unwrap().entity)
            .unwrap()
            .0;
        assert!(
            second_velocity.z > 0.0,
            "boat must consume the previous mounted player movement fields"
        );
    }

    /// **The packet-family fork.** java: `EntityPlayerSP.onUpdate` (`:174-183`) sends
    /// `C05PacketPlayerLook` + `C0CPacketInput` while `isRiding()` and calls
    /// `onUpdateWalkingPlayer` **only** in the `else`. `S1BPacketEntityAttach` was decoded to a
    /// `ClientEvent` nobody read, so a mounted mc-rs client kept sending the walking set and
    /// never sent a single `C0C` — `F-P0-client.entity-01-1`, `F-P0-network.play-01-2`,
    /// `-02-1`, `F-P0-network.play.client-01-5`, `-03-1`, `F-P0-network.play.server-07-1`,
    /// `F-P0-util-01-1`, `F-P1-entity.item-02-1`, `-07-1`, `F-P2-client.animstate-01-3`,
    /// `F-P0-client.network-02-6`.
    ///
    /// Both directions are asserted with the *same* input on the *same* session: the walking
    /// set before the attach is the negative control, and it comes back after the dismount. A
    /// test that only checked "a `C0C` appears" would pass with the walking packets still going
    /// out alongside it, which is the divergence a server actually sees.
    #[test]
    fn a_mounted_client_swaps_the_walking_packet_set_for_look_plus_steer() {
        let (mut session, to_game, mut to_net) = mountable_session();
        spawn_test_pig(&mut session, 42);
        let walk = MoveInput {
            forward: true,
            ..MoveInput::default()
        };

        session.tick(&walk);
        let on_foot = drain_outbound(&mut to_net);
        assert!(
            on_foot.iter().any(is_movement),
            "control: an unmounted tick sends the walking set — {on_foot:?}"
        );
        assert!(
            steers(&on_foot).is_empty(),
            "control: `C0CPacketInput` is a riding-only packet — {on_foot:?}"
        );

        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), Some(42));
        {
            let state = &mut session.predictor.as_mut().expect("predictor").state;
            state.prev_camera_yaw = 0.125;
            state.camera_yaw = 0.25;
            state.first_update = true;
        }

        session.tick(&walk);
        let state = &session.predictor.as_ref().expect("predictor").state;
        assert!(
            !state.first_update,
            "mounted updateRidden still runs Entity.onUpdate"
        );
        assert_eq!(
            state.prev_camera_yaw, 0.15,
            "mounted `super.onLivingUpdate` damps cameraYaw before EntityPlayer snapshots it"
        );
        assert_eq!(
            state.camera_yaw, 0.0,
            "mounted updateRidden clears sway so dismount cannot flash a stale bob"
        );
        let mounted = drain_outbound(&mut to_net);
        assert_eq!(
            steers(&mounted).len(),
            1,
            "one `C0CPacketInput` per mounted tick — {mounted:?}"
        );
        assert_eq!(
            mounted.iter().filter(|p| is_movement(p)).count(),
            1,
            "only the `C05PacketPlayerLook` — {mounted:?}"
        );
        assert!(
            matches!(mounted[0], C2SPlay::PlayerLook(_)),
            "java: the look leads the pair — {mounted:?}"
        );
        assert!(
            !mounted
                .iter()
                .any(|p| matches!(p, C2SPlay::EntityAction(_))),
            "java: `onUpdateWalkingPlayer` never runs, so the sprint/sneak edges are frozen \
             for the whole ride — {mounted:?}"
        );

        to_game
            .try_send(ToGame::Packet(attach(3, -1, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), None, "vehicle id -1 is the dismount");

        session.tick(&walk);
        let on_foot_again = drain_outbound(&mut to_net);
        assert!(
            on_foot_again.iter().any(is_movement) && steers(&on_foot_again).is_empty(),
            "the walking set comes back after the dismount — {on_foot_again:?}"
        );
    }

    #[test]
    fn an_unresolved_positive_vehicle_id_is_a_dismount_not_a_phantom_mount() {
        let (mut session, to_game, mut to_net) = mountable_session();
        spawn_test_pig(&mut session, 42);
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        session.tick(&MoveInput {
            forward: true,
            ..MoveInput::default()
        });
        let mounted = drain_outbound(&mut to_net);
        assert_eq!(
            steers(&mounted).len(),
            1,
            "valid target mounts: {mounted:?}"
        );

        // Java's World lookup returns null for an unknown nonnegative id, and mountEntity(null)
        // clears the existing ridingEntity. The next tick must therefore return to walking.
        to_game
            .try_send(ToGame::Packet(attach(3, 404, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), None);
        assert_eq!(session.riding.vehicle_entity, None);

        session.tick(&MoveInput {
            forward: true,
            ..MoveInput::default()
        });
        let on_foot = drain_outbound(&mut to_net);
        assert!(
            on_foot.iter().any(is_movement),
            "walking resumes: {on_foot:?}"
        );
        assert!(steers(&on_foot).is_empty(), "no phantom C0C: {on_foot:?}");
    }

    #[test]
    fn a_removed_or_replaced_vehicle_generation_cannot_keep_the_local_player_mounted() {
        let (mut session, to_game, mut to_net) = mountable_session();
        let old_generation = spawn_test_pig(&mut session, 42);
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), Some(42));
        assert_eq!(session.riding.vehicle_entity, Some(old_generation));

        // Reusing the network id creates a new Java object. A direct tracker replacement models
        // that boundary without routing a second S0F through the dispatch tests.
        let replacement = spawn_test_pig(&mut session, 42);
        assert_ne!(replacement, old_generation);
        session.tick(&MoveInput::default());
        assert_eq!(session.riding(), None);
        assert!(steers(&drain_outbound(&mut to_net)).is_empty());

        spawn_test_pig(&mut session, 43);
        to_game
            .try_send(ToGame::Packet(attach(3, 43, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), Some(43));
        {
            let state = session.state_mut();
            assert!(state.entities.despawn(&mut state.world, 43));
        }
        session.tick(&MoveInput::default());
        assert_eq!(session.riding(), None);
        assert!(steers(&drain_outbound(&mut to_net)).is_empty());
    }

    #[test]
    fn the_explicit_negative_one_vehicle_id_clears_the_local_mount() {
        let (mut session, to_game, _to_net) = mountable_session();
        spawn_test_pig(&mut session, 42);
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), Some(42));

        to_game
            .try_send(ToGame::Packet(attach(3, -1, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), None);
        assert_eq!(session.riding.vehicle_entity, None);
    }

    /// The mounted branch still executes `onLivingUpdate` before `C05+C0C`. A capability edge
    /// from that preflight therefore leads the mounted packet pair, while the raw jump bit still
    /// reaches `C0C`.
    #[test]
    fn mounted_preflight_orders_flight_edge_before_look_and_steer() {
        let (mut session, to_game, mut to_net) = mountable_session();
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                42,
                TrackedKind::Type(mc_data::EntityType::Pig),
                None,
                [0, 64 * 32, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        session.set_render_view_entity(42);
        session.state_mut().local.abilities = crate::sync::Abilities::from_flags(0x04, 0.05, 0.1);
        {
            let predictor = session.predictor_mut().expect("joined predictor");
            predictor.state.move_strafing = 0.35;
            predictor.state.move_forward = 0.65;
            predictor.state.jumping = false;
            predictor.state.allow_flying = true;
            predictor.state.flying = false;
            predictor.state.fly_toggle_timer = 1;
            predictor.state.prev_jump = false;
            predictor.state.on_ground = false;
        }

        session.tick(&MoveInput {
            forward: true,
            jump: true,
            yaw: 27.0,
            pitch: -11.0,
            ..MoveInput::default()
        });
        let packets = drain_outbound(&mut to_net);
        assert_eq!(packets.len(), 3, "C13, C05, C0C: {packets:?}");
        assert!(matches!(
            packets[0],
            C2SPlay::PlayerAbilities(c2s::PlayerAbilities { flying: true, .. })
        ));
        assert!(matches!(
            packets[1],
            C2SPlay::PlayerLook(c2s::PlayerLook {
                yaw: 27.0,
                pitch: -11.0,
                ..
            })
        ));
        assert!(matches!(packets[2], C2SPlay::SteerVehicle(_)));
        let steer = steers(&packets);
        assert_eq!(steer.len(), 1);
        // java: a foreign camera leaves the action fields un-refreshed, but the travel section's
        // `*= 0.98F` still damps them before the C0C reads them.
        assert_eq!(steer[0].forward, 0.65_f32 * 0.98);
        assert_eq!(steer[0].sideways, 0.35_f32 * 0.98);
        assert_ne!(steer[0].flags & 1, 0, "the raw jump key still reaches C0C");
        // The foreign camera gets no view-local thrust (`+0.15` would leave this positive). The
        // rider's own travel still runs (H2): `EntityPlayer`'s flight wrapper is
        // `ridingEntity == null`-gated, so even this freshly flying rider takes the plain
        // living travel and its gravity step.
        assert_eq!(
            session.predictor().expect("joined predictor").state.vel.y,
            (0.0 - mc_physics::player::GRAVITY) * mc_physics::player::DRAG_Y
        );
    }

    /// **The riding pair's two `onLivingUpdate` products.** java:
    /// `EntityPlayerSP.onUpdate` builds `C05PacketPlayerLook(.., this.onGround)` and
    /// `C0CPacketInput(this.moveStrafing, this.moveForward, ..)` *after* `super.onUpdate()`,
    /// whose `EntityLivingBase.onLivingUpdate` damps both action fields by `0.98F` and then runs
    /// the rider's own `moveEntity` over the motion `updateRidden` zeroed. So a vanilla rider
    /// holding W+A sends `0.98`/`0.98` (never `1.0`, a value only a non-vanilla client puts in
    /// a C0C) and reports `onGround == false` even when it mounted from solid ground. The
    /// previous port copied the raw input (`1.0`) and the stale walking `onGround` (`true`).
    #[test]
    fn a_vanilla_rider_steers_with_damped_fields_and_reports_its_own_move_off_the_ground() {
        let (mut session, to_game, mut to_net) = mountable_session();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        let _ = drain_outbound(&mut to_net);
        assert!(
            session
                .predictor()
                .expect("joined predictor")
                .state
                .on_ground,
            "control: the player stands on the stone floor before mounting"
        );
        spawn_test_pig(&mut session, 42);
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), Some(42));

        session.tick(&MoveInput {
            forward: true,
            left: true,
            ..MoveInput::default()
        });
        let packets = drain_outbound(&mut to_net);
        assert!(
            matches!(
                packets.as_slice(),
                [
                    C2SPlay::PlayerLook(c2s::PlayerLook {
                        on_ground: false,
                        ..
                    }),
                    C2SPlay::SteerVehicle(_)
                ]
            ),
            "C05 with the rider move's onGround, then C0C: {packets:?}"
        );
        let steer = steers(&packets);
        assert_eq!(steer[0].forward, 0.98_f32);
        assert_eq!(steer[0].sideways, 0.98_f32);
        assert_eq!(steer[0].flags, 0);

        // MovementInputFromOptions' sneak scale is a double product narrowed to float; the
        // travel damping then multiplies that float.
        session.tick(&MoveInput {
            back: true,
            right: true,
            sneak: true,
            ..MoveInput::default()
        });
        let packets = drain_outbound(&mut to_net);
        let steer = steers(&packets);
        // `(float)((double)-1.0F * 0.3D)`: the product is exactly -0.3D.
        let sneak_scaled = -0.3_f64 as f32;
        assert_eq!(steer[0].forward, sneak_scaled * 0.98);
        assert_eq!(steer[0].sideways, sneak_scaled * 0.98);
        assert_eq!(steer[0].flags, 0b10, "the raw sneak bit");
        assert!(steer
            .iter()
            .all(|steer| steer.forward.abs() <= 0.98 && steer.sideways.abs() <= 0.98));
    }

    /// `Entity.updateRidden` runs the rider's full `onUpdate`, so
    /// its own travel recomputes `isCollidedHorizontally` every ride tick. Walking into the
    /// vehicle left the flag set: vanilla's first ride tick still stops the sprint on it, the
    /// travel clears it, and from the second tick a held sprint key keeps the rider sprinting.
    /// The first walking tick after the dismount then reports START_SPRINTING before its
    /// movement packet. Negative control: the stale flag stopped the sprint on every ride tick
    /// (no travel ran), so the rider never sprinted and START came a tick late.
    #[test]
    fn a_rider_sprints_once_the_stale_walking_collision_clears() {
        use mc_protocol::play::c2s::EntityActionType;

        let (mut session, to_game, mut to_net) = mountable_session();
        for _ in 0..40 {
            session.tick(&MoveInput::default());
        }
        // The walking tick that bumped into the vehicle.
        session
            .predictor_mut()
            .expect("joined predictor")
            .state
            .collided_horizontally = true;
        spawn_test_pig(&mut session, 42);
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), Some(42));
        let _ = drain_outbound(&mut to_net);

        let run = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };
        session.tick(&run);
        let state = session.predictor().expect("joined predictor").state;
        assert!(
            !state.sprinting,
            "ride tick 1 still reads the walking collision"
        );
        assert!(
            !state.collided_horizontally,
            "the rider's own travel found no wall"
        );
        session.tick(&run);
        assert!(
            session
                .predictor()
                .expect("joined predictor")
                .state
                .sprinting,
            "ride tick 2 keeps the sprint the key started"
        );
        let riding = drain_outbound(&mut to_net);
        assert!(
            !riding
                .iter()
                .any(|packet| matches!(packet, C2SPlay::EntityAction(_))),
            "serverSprintState is frozen while riding: {riding:?}"
        );

        to_game
            .try_send(ToGame::Packet(attach(3, -1, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.riding(), None);
        session.tick(&run);
        let walking = drain_outbound(&mut to_net);
        assert!(
            matches!(
                walking.as_slice(),
                [
                    C2SPlay::EntityAction(c2s::EntityAction {
                        action: EntityActionType::StartSprinting,
                        ..
                    }),
                    movement,
                ] if is_movement(movement)
            ),
            "the first walking tick reports the ride's sprint: {walking:?}"
        );
    }

    /// The one requested-`y < 0` ride tick: a view-local creative flyer descending with sneak.
    /// java: that `moveEntity` sweeps the box down by the flight thrust first, so `onGround` is
    /// whether the sweep was cut short, and the landing tail then clears flight with a C13 ahead
    /// of the riding pair. The same descent with clearance reports `false` and keeps flying.
    #[test]
    fn a_descending_flying_rider_reports_the_vertical_sweeps_ground_like_vanilla() {
        for (feet_y, lands) in [(65.1, true), (66.0, false)] {
            let (mut session, to_game, mut to_net) = mountable_session();
            spawn_test_pig(&mut session, 42);
            to_game
                .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
                .unwrap();
            session.poll(16);
            session.state_mut().local.abilities =
                crate::sync::Abilities::from_flags(0x02 | 0x04, 0.05, 0.1);
            session
                .predictor_mut()
                .expect("joined predictor")
                .state
                .set_position(DVec3::new(8.5, feet_y, 8.5));
            let _ = drain_outbound(&mut to_net);

            session.tick(&MoveInput {
                sneak: true,
                ..MoveInput::default()
            });
            let packets = drain_outbound(&mut to_net);
            let look_on_ground = packets.iter().find_map(|packet| match packet {
                C2SPlay::PlayerLook(look) => Some(look.on_ground),
                _ => None,
            });
            assert_eq!(look_on_ground, Some(lands), "feet {feet_y}: {packets:?}");
            let landed_c13 = packets.iter().any(|packet| {
                matches!(
                    packet,
                    C2SPlay::PlayerAbilities(c2s::PlayerAbilities { flying: false, .. })
                )
            });
            assert_eq!(landed_c13, lands, "feet {feet_y}: {packets:?}");
            if lands {
                assert!(matches!(packets[0], C2SPlay::PlayerAbilities(_)));
            }
        }
    }

    /// **The horse-jump C0B leads the landing C13.** java:
    /// `EntityPlayerSP.onLivingUpdate` sends `sendHorseJump` (`:861-896`) before
    /// `super.onLivingUpdate()` and the landing `sendPlayerAbilities` (`:904-908`) after it, so a
    /// release tick that also lands a flying rider writes C0B, C13, C05, C0C in that order. The
    /// previous port sent both C13s first.
    #[test]
    fn a_horse_jump_released_on_a_landing_tick_precedes_the_landing_c13() {
        use mc_protocol::metadata::{MetadataEntry, MetadataValue};

        let (mut session, to_game, mut to_net) = mountable_session();
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 42,
                entity_type: 100,
                x: 8 * 32,
                y: 65 * 32,
                z: 8 * 32,
                yaw: mc_protocol::Angle(0),
                pitch: mc_protocol::Angle(0),
                head_pitch: mc_protocol::Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityMetadata(
                s2c::EntityMetadata {
                    entity_id: 42,
                    metadata: vec![MetadataEntry {
                        index: 16,
                        value: MetadataValue::Int(4),
                    }],
                },
            )))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert!(session.is_riding_saddled_horse());
        session.state_mut().local.abilities =
            crate::sync::Abilities::from_flags(0x02 | 0x04, 0.05, 0.1);
        let held = MoveInput {
            jump: true,
            ..MoveInput::default()
        };
        for _ in 0..5 {
            session.tick(&held);
        }
        let _ = drain_outbound(&mut to_net);
        // Release the jump while descending with sneak 0.1 above the floor: the rider's own
        // move lands this tick.
        session
            .predictor_mut()
            .expect("joined predictor")
            .state
            .set_position(DVec3::new(8.5, 65.1, 8.5));
        session.tick(&MoveInput {
            sneak: true,
            ..MoveInput::default()
        });
        let packets = drain_outbound(&mut to_net);
        assert!(
            matches!(
                packets.as_slice(),
                [
                    C2SPlay::EntityAction(c2s::EntityAction {
                        action: c2s::EntityActionType::RidingJump,
                        ..
                    }),
                    C2SPlay::PlayerAbilities(c2s::PlayerAbilities { flying: false, .. }),
                    C2SPlay::PlayerLook(c2s::PlayerLook {
                        on_ground: true,
                        ..
                    }),
                    C2SPlay::SteerVehicle(_),
                ]
            ),
            "{packets:?}"
        );
    }

    /// **An S08 leaves `onUpdateWalkingPlayer`'s bookkeeping alone.** java:
    /// `handlePlayerPosLook` moves the player and sends its own C06 (`onGround == false`), but
    /// `lastReportedPos*`, `lastReportedYaw/Pitch` and `positionUpdateTicks` are private to
    /// `EntityPlayerSP.onUpdateWalkingPlayer` and untouched. So the forced 20-tick report keeps
    /// its phase across a teleport, and the tick after a far teleport re-reports the position
    /// against the *last C03-family report* even though the player has not moved since the S08
    /// (absolute Y zeroed the motion, so that tick's move is zero).
    #[test]
    fn a_teleport_leaves_the_walking_report_cadence_alone() {
        let (mut session, to_game, mut to_net) = joined_and_standing();
        let idle = MoveInput::default();
        for _ in 0..60 {
            session.tick(&idle);
        }
        // Find a forced report: a standing player sends one C04 every twenty-one ticks.
        let _ = drain_outbound(&mut to_net);
        let mut found = false;
        for _ in 0..25 {
            session.tick(&idle);
            if matches!(
                drain_outbound(&mut to_net).as_slice(),
                [C2SPlay::PlayerPosition(_)]
            ) {
                found = true;
                break;
            }
        }
        assert!(found, "a standing player sends a forced C04");
        let shape = |packets: Vec<C2SPlay>| match packets.as_slice() {
            [C2SPlay::Player(_)] => "C03",
            [C2SPlay::PlayerPosition(_)] => "C04",
            other => panic!("one walking packet per tick, got {other:?}"),
        };
        for _ in 0..10 {
            session.tick(&idle);
            assert_eq!(shape(drain_outbound(&mut to_net)), "C03");
        }
        // A fully relative, zero-offset S08: nothing moves, the confirmation still goes out.
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    yaw: 0.0,
                    pitch: 0.0,
                    flags: 0x1F,
                },
            )))
            .unwrap();
        session.poll(16);
        assert!(matches!(
            drain_outbound(&mut to_net).as_slice(),
            [C2SPlay::PlayerPositionAndLook(c2s::PlayerPositionAndLook {
                on_ground: false,
                ..
            })]
        ));
        // Ticks 11-20 after the forced report stay bare; tick 21 is forced, as without the S08.
        for _ in 0..10 {
            session.tick(&idle);
            assert_eq!(shape(drain_outbound(&mut to_net)), "C03");
        }
        session.tick(&idle);
        assert_eq!(
            shape(drain_outbound(&mut to_net)),
            "C04",
            "the phase survived"
        );

        // A far absolute teleport with the same look: the next tick does not move (motion was
        // zeroed), yet it reports the new position because the last report is the old one.
        to_game
            .try_send(ToGame::Packet(S2CPlay::PlayerPositionLook(
                s2c::PlayerPositionLook {
                    x: 8.5,
                    y: 80.0,
                    z: 8.5,
                    yaw: 0.0,
                    pitch: 0.0,
                    flags: 0,
                },
            )))
            .unwrap();
        session.poll(16);
        let _ = drain_outbound(&mut to_net);
        session.tick(&idle);
        match drain_outbound(&mut to_net).as_slice() {
            [C2SPlay::PlayerPosition(report)] => {
                assert_eq!((report.x, report.y, report.z), (8.5, 80.0, 8.5));
                assert!(!report.on_ground);
            }
            other => panic!("the teleported position is re-reported: {other:?}"),
        }
    }

    /// java: `EntityHorse.mountTo` turns the player to the horse on the
    /// client at the click; the S1B (`Entity.mountEntity`) only seats the player and leaves any
    /// later turning alone.
    #[test]
    fn a_horse_mount_click_turns_the_player_and_the_confirmation_snaps_the_seat() {
        let (mut session, to_game, _to_net) = mountable_session();
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 42,
                entity_type: 100,
                x: 8 * 32,
                y: 65 * 32,
                z: 9 * 32,
                yaw: mc_protocol::Angle(64),
                pitch: mc_protocol::Angle(32),
                head_pitch: mc_protocol::Angle(64),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            })))
            .unwrap();
        session.poll(16);
        let generation = session.state.entities.get(42).expect("horse").entity;
        {
            let predictor = session.predictor_mut().expect("joined predictor");
            predictor.state.pos = DVec3::new(1.0, 2.0, 3.0);
            predictor.state.yaw = 5.0;
            predictor.state.pitch = 6.0;
        }
        session.note_horse_mount_intent(42, generation);
        // The click itself turns the player to the horse (yaw byte 64 = 90, pitch 32 = 45).
        {
            let clicked = &session.predictor().unwrap().state;
            assert_eq!((clicked.yaw, clicked.pitch), (90.0, 45.0));
            assert_eq!(clicked.pos, DVec3::new(1.0, 2.0, 3.0));
        }
        // The player turns a little before the server's answer arrives.
        {
            let predictor = session.predictor_mut().expect("joined predictor");
            predictor.state.yaw = 93.0;
            predictor.state.pitch = 44.0;
        }

        // Neither another rider nor the leash packet family may consume or apply our pending
        // client mount. The matching local vehicle packet that follows is the sole edge.
        to_game
            .try_send(ToGame::Packet(attach(99, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Leash)))
            .unwrap();
        session.poll(16);
        assert_eq!(
            session.predictor().unwrap().state.pos,
            DVec3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(session.riding(), None);

        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        let mounted = &session.predictor().unwrap().state;
        assert_eq!(mounted.pos.x, 8.0);
        assert!((mounted.pos.y - 65.85).abs() < 1.0e-6, "{:?}", mounted.pos);
        assert_eq!(mounted.pos.z, 9.0);
        assert_eq!(
            (mounted.yaw, mounted.pitch),
            (93.0, 44.0),
            "the S1B keeps the player's own look"
        );
        assert_eq!(session.riding(), Some(42));

        let snapped = mounted.pos;
        to_game
            .try_send(ToGame::Packet(attach(3, -1, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);
        assert_eq!(session.predictor().unwrap().state.pos, snapped);
        assert_eq!(session.riding(), None);

        let production = include_str!("app/driver.rs")
            // Driver has cfg(test) helpers in its production section; use the actual test
            // module boundary instead of truncating at the first helper constant.
            .split_once("#[cfg(test)]\nmod tests {")
            .expect("driver has tests")
            .0;
        assert!(production.contains("interact::horse_mount_intent("));
        assert!(production.contains("session.note_horse_mount_intent("));
    }

    /// **W-RIDE.** `C0CPacketInput`'s payload: the two floats are the values *after*
    /// `MovementInputFromOptions.updatePlayerMoveState` (so the sneak `0.3` scale is in them,
    /// and the item-use `0.2` scale is **not**, per `EntityPlayerSP.java:788`'s
    /// `isUsingItem() && !isRiding()`), and the byte is `jump | sneak << 1`
    /// (`C0CPacketInput.writePacketData:44-60`).
    ///
    /// `using_item` is the control that bites: with the riding exemption dropped, sneaking
    /// forward while eating would put `0.06` on the wire instead of `0.3`.
    #[test]
    fn the_steer_packet_carries_the_scaled_inputs_and_the_two_flag_bits() {
        let (mut session, to_game, mut to_net) = mountable_session();
        spawn_test_pig(&mut session, 42);
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);

        session.tick(&MoveInput {
            forward: true,
            left: true,
            jump: true,
            sneak: true,
            using_item: true,
            ..MoveInput::default()
        });
        let sent = drain_outbound(&mut to_net);
        let steer = steers(&sent);
        assert_eq!(steer.len(), 1, "{sent:?}");
        // java: the C0C reads the action fields after the travel section's `*= 0.98F`,
        // so every value is the movement input times 0.98.
        assert_eq!(
            steer[0].sideways,
            0.3_f32 * 0.98,
            "strafe-left, sneak-scaled, damped"
        );
        assert_eq!(
            steer[0].forward,
            0.3_f32 * 0.98,
            "forward, sneak-scaled, not item-scaled, damped"
        );
        assert_eq!(steer[0].flags, 0b11, "bit 0 jump, bit 1 sneak");

        session.tick(&MoveInput {
            back: true,
            right: true,
            ..MoveInput::default()
        });
        let sent = drain_outbound(&mut to_net);
        let steer = steers(&sent);
        assert_eq!(steer[0].sideways, -0.98, "strafe-right is negative");
        assert_eq!(steer[0].forward, -0.98, "back is negative");
        assert_eq!(steer[0].flags, 0, "neither bit");

        // The two bits asymmetrically, because `jump | sneak << 1` and `sneak | jump << 1`
        // agree on every input where the two keys match — which both cases above are.
        session.tick(&MoveInput {
            jump: true,
            ..MoveInput::default()
        });
        let sent = drain_outbound(&mut to_net);
        assert_eq!(steers(&sent)[0].flags, 0b01, "jump alone is bit 0");
        session.tick(&MoveInput {
            sneak: true,
            ..MoveInput::default()
        });
        let sent = drain_outbound(&mut to_net);
        assert_eq!(steers(&sent)[0].flags, 0b10, "sneak alone is bit 1");
    }

    /// **W-RIDE.** The two guards on the attach arm. `handleEntityAttach` (`:976`) only treats
    /// the packet as the local player's mount when `packetIn.getEntityId()` is
    /// `thePlayer.getEntityId()`, and the whole vehicle branch is gated on `getLeash() == 0` —
    /// a leash attach with our own id (a lead tied to a fence while we hold it) must not put
    /// the client into riding mode and silence its movement packets.
    #[test]
    fn a_leash_or_another_entitys_mount_does_not_put_us_in_riding_mode() {
        let (mut session, to_game, _to_net) = mountable_session();

        to_game
            .try_send(ToGame::Packet(attach(99, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Leash)))
            .unwrap();
        session.poll(16);

        assert_eq!(session.riding(), None);
        assert!(!session.is_riding_horse());
    }

    /// **W-RIDE.** `EntityPlayerSP.isRidingHorse` (`:572`) and
    /// `PlayerControllerMP.isRidingHorse` (`:617`) are **different predicates** with the same
    /// name: only the first requires `isHorseSaddled()`. The inventory key uses the controller's
    /// one, the jump charge uses the entity's one.
    ///
    /// The horse here is spawned unsaddled and then saddled by a metadata update, so the
    /// saddle-free predicate answering `true` in both states is what separates them.
    #[test]
    fn the_two_is_riding_horse_predicates_differ_by_the_saddle() {
        use mc_protocol::metadata::{MetadataEntry, MetadataValue};

        let (mut session, to_game, _to_net) = mountable_session();
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 42,
                // java: `EntityList` id 100 is `EntityHorse`.
                entity_type: 100,
                x: 8 * 32,
                y: 65 * 32,
                z: 8 * 32,
                yaw: mc_protocol::Angle(0),
                pitch: mc_protocol::Angle(0),
                head_pitch: mc_protocol::Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);

        assert!(
            session.is_riding_horse(),
            "the controller's predicate has no saddle test"
        );
        assert!(
            !session.is_riding_saddled_horse(),
            "index 16 is absent, so `isHorseSaddled` is false"
        );

        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityMetadata(
                s2c::EntityMetadata {
                    entity_id: 42,
                    // java: `isHorseSaddled` is `getHorseWatchableBoolean(4)` on index 16.
                    metadata: vec![MetadataEntry {
                        index: 16,
                        value: MetadataValue::Int(4),
                    }],
                },
            )))
            .unwrap();
        session.poll(16);

        assert!(session.is_riding_horse());
        assert!(
            session.is_riding_saddled_horse(),
            "the saddle bit arrived, so the entity's predicate flips and the controller's \
             does not move"
        );
    }

    /// **W-RIDE.** `EntityPlayerSP.onLivingUpdate`'s horse-jump charge (`:861-896`) and
    /// `sendHorseJump` (`:407`).
    ///
    /// Three things the shape of the Java hides and this pins:
    ///
    /// * the charge is keyed on the **previous** tick's jump key (`boolean flag =
    ///   this.movementInput.jump`, read before `updatePlayerMoveState()`), so the release is
    ///   `was_jumping && !jump` and the packet goes out on the tick the key comes *up*;
    /// * the power peaks at exactly `1.0` on the tenth charge tick
    ///   (`0.8F + 2.0F / (10 - 9) * 0.1F`) and then **decays** back toward `0.8` — holding
    ///   longer is worse;
    /// * the payload is `(int)(power * 100.0F)`, a truncation.
    ///
    /// The control: the same key pattern on an **unsaddled** horse sends nothing at all,
    /// because the charge block is gated on `EntityPlayerSP.isRidingHorse()`.
    #[test]
    fn a_released_horse_jump_sends_its_charged_power_and_an_unsaddled_horse_sends_none() {
        use mc_protocol::metadata::{MetadataEntry, MetadataValue};

        let jumps = |session: &mut Session,
                     to_net: &mut tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
                     charge_ticks: usize| {
            let held = MoveInput {
                jump: true,
                ..MoveInput::default()
            };
            for _ in 0..charge_ticks {
                session.tick(&held);
            }
            session.tick(&MoveInput::default());
            drain_outbound(to_net)
                .into_iter()
                .filter_map(|packet| match packet {
                    C2SPlay::EntityAction(action)
                        if action.action
                            == mc_protocol::play::c2s::EntityActionType::RidingJump =>
                    {
                        Some(action.aux_data)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        };

        let (mut session, to_game, mut to_net) = mountable_session();
        to_game
            .try_send(ToGame::Packet(S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 42,
                entity_type: 100,
                x: 8 * 32,
                y: 65 * 32,
                z: 8 * 32,
                yaw: mc_protocol::Angle(0),
                pitch: mc_protocol::Angle(0),
                head_pitch: mc_protocol::Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            })))
            .unwrap();
        to_game
            .try_send(ToGame::Packet(attach(3, 42, s2c::AttachType::Vehicle)))
            .unwrap();
        session.poll(16);

        assert_eq!(
            jumps(&mut session, &mut to_net, 6),
            Vec::<i32>::new(),
            "an unsaddled horse never charges, so nothing is released"
        );
        assert_eq!(
            session.horse_jump_power(),
            None,
            "an unsaddled mount keeps the XP bar selected",
        );

        to_game
            .try_send(ToGame::Packet(S2CPlay::EntityMetadata(
                s2c::EntityMetadata {
                    entity_id: 42,
                    metadata: vec![MetadataEntry {
                        index: 16,
                        value: MetadataValue::Int(4),
                    }],
                },
            )))
            .unwrap();
        session.poll(16);
        assert_eq!(session.horse_jump_power(), Some(0.0));

        // Six ticks of hold: the first sets `previous_jump`, the next five charge, so the
        // counter reaches 5 and the power is `5 * 0.1F`.
        assert_eq!(
            jumps(&mut session, &mut to_net, 6),
            vec![50],
            "(int)(0.5F * 100.0F)"
        );
        assert_eq!(session.horse_jump_power(), Some(0.5));
        // A full charge: eleven ticks of hold puts the counter at 10, which is the single tick
        // where the second branch evaluates to exactly `1.0`.
        assert_eq!(
            jumps(&mut session, &mut to_net, 11),
            vec![100],
            "0.8F + 2.0F / (10 - 9) * 0.1F"
        );
        // One more tick of hold and it has already decayed.
        assert_eq!(
            jumps(&mut session, &mut to_net, 12),
            vec![90],
            "0.8F + 2.0F / (11 - 9) * 0.1F — holding past the peak is worse"
        );
    }

    /// java: `EntityRenderer.orientCamera` turns a spectating camera with
    /// `prevRotationYaw + (rotationYaw - prevRotationYaw) * partialTicks`. It used to start from
    /// the previous *body* yaw (`prevRenderYawOffset`), which trails the look, so the camera swung
    /// between the two on every tick.
    #[test]
    fn a_spectating_camera_turns_with_the_raw_rotation_not_the_body() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                77,
                TrackedKind::Player,
                None,
                [8 * 32, 65 * 32, 12 * 32],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        session.set_render_view_entity(77);
        assert!(!session.render_view_is_local());
        // Turn the player 120 degrees: the look lerps over three ticks, the body lags behind.
        for _ in 0..3 {
            to_game
                .try_send(ToGame::Packet(S2CPlay::EntityLook(s2c::EntityLook {
                    entity_id: 77,
                    yaw: mc_protocol::Angle::from_degrees(120.0),
                    pitch: mc_protocol::Angle::from_degrees(30.0),
                    on_ground: true,
                })))
                .unwrap();
            session.poll(16);
            session.tick(&MoveInput::default());
        }
        let tracked = session.state().entities.get(77).expect("spawned");
        let current = session
            .state()
            .world
            .ecs
            .get::<&mc_world::Rotation>(tracked.entity)
            .map(|rotation| (rotation.yaw, rotation.pitch))
            .expect("a rotation component");
        let (prev_yaw, prev_pitch) = (tracked.prev_rotation_yaw, tracked.prev_rotation_pitch);
        assert!(
            (tracked.render.prev_body_yaw - prev_yaw).abs() > 5.0,
            "precondition: the body trails the look ({} vs {prev_yaw})",
            tracked.render.prev_body_yaw
        );
        let at = |partial: f32| session.render_view_rotation(partial).expect("a view");
        let start = at(0.0);
        let end = at(1.0);
        assert!(
            (start.0 - prev_yaw).abs() < 1e-4,
            "{start:?} vs prev {prev_yaw}"
        );
        assert!(
            (start.1 - prev_pitch).abs() < 1e-4,
            "{start:?} vs prev {prev_pitch}"
        );
        assert!((end.0 - current.0).abs() < 1e-4, "{end:?} vs {current:?}");
        assert!((end.1 - current.1).abs() < 1e-4, "{end:?} vs {current:?}");
    }

    /// java: `orientCamera`'s eight F5 clipping rays aim with the view
    /// entity's current `rotationYaw`/`rotationPitch` (`f1`/`f2`); only the view turns by the
    /// interpolated pair. Mid-turn the two differ, so near a wall the orbit is clipped along the
    /// current rotation. Previously, the rays took the interpolated one.
    #[test]
    fn spectated_f5_clipping_rays_use_the_current_rotation() {
        let (mut session, to_game, _to_net) = joined_and_standing();
        {
            let state = session.state_mut();
            state.entities.spawn(
                &mut state.world,
                78,
                TrackedKind::Player,
                None,
                [8 * 32, 65 * 32, 12 * 32],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::Metadata::new(),
            );
        }
        session.set_render_view_entity(78);
        for _ in 0..2 {
            to_game
                .try_send(ToGame::Packet(S2CPlay::EntityLook(s2c::EntityLook {
                    entity_id: 78,
                    yaw: mc_protocol::Angle::from_degrees(120.0),
                    pitch: mc_protocol::Angle::from_degrees(30.0),
                    on_ground: true,
                })))
                .unwrap();
            session.poll(16);
            session.tick(&MoveInput::default());
        }
        let tracked = session.state().entities.get(78).expect("spawned");
        let current = session
            .state()
            .world
            .ecs
            .get::<&mc_world::Rotation>(tracked.entity)
            .map(|rotation| (rotation.yaw, rotation.pitch))
            .expect("a rotation component");
        let rays = session.render_view_ray_rotation().expect("a view");
        assert_eq!(rays, current, "the rays aim with the current tick rotation");
        let mid_tick = session.render_view_rotation(0.5).expect("a view");
        assert!(
            (mid_tick.0 - rays.0).abs() > 1.0,
            "negative control: mid-turn, the view's interpolated yaw differs: {mid_tick:?} vs {rays:?}"
        );
    }

    /// java: `orientCamera`'s `if (entity instanceof EntityAnimal)` turns with
    /// `rotationYawHead` instead.
    #[test]
    fn a_spectated_animal_turns_the_camera_with_its_head() {
        assert!(spectator_camera_uses_head_yaw(TrackedKind::Type(
            mc_data::EntityType::Pig
        )));
        assert!(spectator_camera_uses_head_yaw(TrackedKind::Type(
            mc_data::EntityType::Wolf
        )));
        assert!(!spectator_camera_uses_head_yaw(TrackedKind::Player));
        assert!(!spectator_camera_uses_head_yaw(TrackedKind::Type(
            mc_data::EntityType::Villager
        )));
        assert!(!spectator_camera_uses_head_yaw(TrackedKind::Type(
            mc_data::EntityType::Zombie
        )));
    }
}
