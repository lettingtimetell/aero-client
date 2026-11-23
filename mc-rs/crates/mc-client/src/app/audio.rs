//! The client's sound half: the `AudioEngine` the game actually plays through, the listener the
//! camera keeps up to date, and the `ClientEvent` handlers that turn server sound packets into
//! `PlayRequest`s.
//!
//! java: `Minecraft`'s `mcSoundHandler`/`mcMusicTicker` fields plus the `SoundManager.playSound`
//! funnel every sound in the game passes through.
//!
//! Split out of [`super::driver`] so this seam has a file of its own; `Client`'s audio methods
//! live here as a second `impl Client` block.
//!
//! # What this module is for
//!
//! `mc-audio` was finished, tested and then never called: `AudioEngine::new` had zero call sites
//! outside the crate's own tests, `SoundEffectPacket::into_request` had none at all, and
//! `ClientEvent::SoundEffect` was produced by `sync::dispatch` and matched by nothing, so the
//! game was silent end to end. This file is that
//! missing seam and nothing else — every piece of sound *behaviour* still lives in `mc-audio`.
//!
//! # Verifying sound without listening to it
//!
//! `AudioEngine::new` runs silent when no output device exists
//! ([`mc_audio::AudioEngine::is_silent`]), and every `play` then returns `Ok(None)` — which is
//! what makes constructing it unconditionally safe, and also why "silent" and "unwired" look
//! identical from the outside. That is precisely how this gap survived a whole section being
//! ledgered complete at 100%. The tests below therefore assert on
//! [`mc_audio::AudioEngine::recent`] — the request the game *formed*: its event name, volume,
//! pitch, position, and the category the real `sounds.json` resolved it to — and never on
//! audible output, so they mean the same thing on a machine with speakers and one without.
//!
//! (On the machine this was written on `cpal` *does* accept a stream — see the "audio backend:"
//! line `the_client_owns_an_engine_holding_the_real_sound_registry` prints. That means the
//! decode and mixer path really runs here; it does **not** mean anyone has confirmed by ear
//! that the result sounds like Minecraft. `mc-audio`'s crate docs make the same distinction.)

use mc_audio::{
    AudioEngine, AudioError, Backend, Listener, MusicPlayback, MusicType, PlayRequest,
    PlayingSound, SoundCategory, TickableSound,
};
use mc_math::DVec3;
use mc_physics::state::PhysicsState;
use mc_protocol::play::s2c;
use mc_render::settings::GameSettings;
use std::sync::{Arc, Mutex};

use crate::app::assets::GameAssets;
use crate::app::driver::{Client, Ui};
use crate::session::Session;
use crate::sync::entities::{GuardianAttackEvent, GuardianSoundState};
#[cfg(test)]
use crate::sync::TrackedKind;

/// The GUI click. java: `GuiButton.playPressSound` — `PositionedSoundRecord.create(new
/// ResourceLocation("gui.button.press"), 1.0F)`.
///
/// That `1.0F` is the **pitch**: it selects `create(ResourceLocation, float pitch)`
/// (`PositionedSoundRecord.java:7-10`), whose volume is a hardcoded `0.25F` with
/// `AttenuationType.NONE`. The other overload — `create(ResourceLocation)`, volume `1.0F` — takes
/// no float at all, so there is no reading of this call that yields volume 1.0.
pub const BUTTON_PRESS: &str = "gui.button.press";

/// java: `PositionedSoundRecord.create(ResourceLocation, float)`'s hardcoded volume.
/// `F-P2-client.audio-02-1`, `F-P2-client.audio-01-10`.
pub const BUTTON_PRESS_VOLUME: f32 = 0.25;

/// The exact `EntityPickupFX` sound pitch formula, kept pure so its float order is testable.
///
/// java: `((Random.nextFloat() - Random.nextFloat()) * 0.7F + 1.0F) * 2.0F`.
#[must_use]
pub(crate) fn item_collected_pitch(first: f32, second: f32) -> f32 {
    ((first - second) * 0.7 + 1.0) * 2.0
}

/// Java's `instanceof EntityXPOrb` sound type branch.
#[must_use]
pub(crate) fn item_collected_sound(xp_orb: bool) -> &'static str {
    if xp_orb {
        "random.orb"
    } else {
        "random.pop"
    }
}

/// Seed for the engine's weighted-variant RNG.
///
/// Vanilla uses a time-seeded `java.util.Random` (`SoundEventAccessorComposite.rnd`); which
/// variant of `step.grass` you hear has no parity requirement. A fixed seed is used here for the
/// same reason [`Client`]'s torch flicker uses one: headless runs and the replay-determinism test
/// must be reproducible, and `Client::new` has no wall-clock source to hand it anyway.
///
/// **The value matters, and the old one was the worst possible choice.** It was
/// `0x0005_DEEC_E66D` — exactly `java.util.Random`'s LCG multiplier — and `JavaRandom::new`
/// applies Java's `(seed ^ MULTIPLIER) & MASK` scramble, so the internal state started at **0**.
/// The first `next(31)` is then `0xB >>> 17 == 0`, i.e. the first `nextInt(bound)` is `0` for
/// every bound, and the first few multi-variant sounds of every session replayed the pool's first
/// entry: four identical `dig.stone` hits from a cold client. `F-P2-client.audio-02-7`.
///
/// `1` is chosen for the same reason `0` would have been fine and `0x5DEECE66D` was not: it is
/// arbitrary, documented, and not a fixed point of the scramble.
const VARIANT_SEED: i64 = 1;

/// Seed for the independent `Minecraft.mcMusicTicker` random stream.
///
/// Vanilla creates this `Random` separately from the sound-event variant picker. A fixed seed
/// keeps headless/replay runs deterministic while preserving the fact that a menu delay draw
/// must not consume the event-picker stream (and vice versa).
pub(crate) const MUSIC_TICKER_SEED: i64 = 1;

/// java: `EntityPlayer.getHurtSound` (`entity/player/EntityPlayer.java:747-750`).
const LOCAL_PLAYER_HURT_SOUND: &str = "game.player.hurt";
/// java: `EntityPlayer.getDeathSound` (`entity/player/EntityPlayer.java:755-758`).
const LOCAL_PLAYER_DEATH_SOUND: &str = "game.player.die";

/// java: `GuardianSound`'s registered event (`GuardianSound.java:13`).
const GUARDIAN_ATTACK_SOUND: &str = "mob.guardian.attack";

/// java: `MovingSoundMinecart`'s registered event (`MovingSoundMinecart.java:14`).
const MINECART_BASE_SOUND: &str = "minecart.base";
/// java: `MovingSoundMinecartRiding`'s registered event.
const MINECART_INSIDE_SOUND: &str = "minecart.inside";

/// Remove and immediately stop the record retained for one jukebox position.
///
/// Kept outside `Client` so the exact production ownership operation can be exercised with
/// `MockBackend`; a silent/default backend cannot produce a retained handle on headless hosts.
fn stop_record_at(
    record_sounds: &mut std::collections::HashMap<mc_math::BlockPos, PlayingSound>,
    pos: mc_math::BlockPos,
) {
    if let Some(mut playing) = record_sounds.remove(&pos) {
        playing.stop_now();
    }
}

/// Stop the backend registry and discard every position-owned record handle for a world swap.
fn stop_world_audio<B: Backend + mc_audio::OutputHeartbeat>(
    audio: &mut AudioEngine<B>,
    record_sounds: &mut std::collections::HashMap<mc_math::BlockPos, PlayingSound>,
) where
    B::Settings: Default,
    B::Error: std::fmt::Debug,
{
    audio.stop_all_sounds();
    record_sounds.clear();
}

/// Apply one `RenderGlobal.playRecord` audio/map transition against any backend.
///
/// Production uses the default backend; tests use `MockBackend` so deleting the production
/// replacement/eject call cannot hide behind a headless engine that returns no handles.
fn drive_record_effect_audio<B: Backend + mc_audio::OutputHeartbeat>(
    audio: &mut AudioEngine<B>,
    record_sounds: &mut std::collections::HashMap<mc_math::BlockPos, PlayingSound>,
    listener: &Listener,
    resources: &mc_assets::ResourceManager,
    pos: mc_math::BlockPos,
    data: i32,
) -> Result<Option<&'static str>, AudioError>
where
    B::Settings: Default,
    B::Error: std::fmt::Debug,
{
    stop_record_at(record_sounds, pos);
    let Some(record) = record_name_for_item(data) else {
        return Ok(None);
    };
    if audio.categories.get(SoundCategory::Master) <= 0.0 {
        return Ok(Some(record));
    }
    let request = PlayRequest::positional(
        mc_assets::ResourceLocation::minecraft(format!("records.{record}")),
        DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z)),
    );
    if let Some(handle) = audio.play(&request, listener, resources)? {
        record_sounds.insert(pos, handle);
    }
    Ok(Some(record))
}

/// The mutable part of one live Guardian attack sound.
///
/// The entity tracker owns the authoritative snapshot.  This copy is shared with the retained
/// [`GuardianSound`] because `AudioEngine::update_all_sounds` may own the tickable after the
/// application-side owner has advanced to the next client tick.  A lock is sufficient here: the
/// game thread is the only writer and the audio backend only reads it during its update pass.
#[derive(Debug, Clone, Copy)]
struct GuardianSoundValues {
    position: DVec3,
    ticks: u32,
    max_ticks: u32,
    active: bool,
}

impl Default for GuardianSoundValues {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            ticks: 0,
            max_ticks: 80,
            active: true,
        }
    }
}

/// One vanilla `GuardianSound`/`MovingSound` retained by the client.
///
/// The `event` and entity generation make updates generation-safe.  In particular, a status
/// packet queued for an entity id that is subsequently reused cannot update the replacement
/// Guardian's channel; [`EntityTracker::guardian_sound_state`] returns `None` for that mismatch.
#[derive(Debug)]
pub(crate) struct GuardianSoundOwner {
    event: GuardianAttackEvent,
    state: Arc<Mutex<GuardianSoundValues>>,
}

impl GuardianSoundOwner {
    fn new(event: GuardianAttackEvent) -> Self {
        Self {
            event,
            state: Arc::new(Mutex::new(GuardianSoundValues::default())),
        }
    }

    /// Refresh the sound's entity-backed fields, or make it stop when its entity disappeared.
    fn refresh(&self, snapshot: Option<GuardianSoundState>) {
        let mut state = self
            .state
            .lock()
            .expect("GuardianSound state lock poisoned");
        match snapshot {
            Some(snapshot) => {
                state.position = snapshot.position;
                state.ticks = snapshot.ticks;
                state.max_ticks = snapshot.max_ticks.max(1);
                state.active = snapshot.active;
            }
            None => state.active = false,
        }
    }

    fn is_done(&self) -> bool {
        !self
            .state
            .lock()
            .expect("GuardianSound state lock poisoned")
            .active
    }

    fn tickable(&self) -> GuardianSound {
        GuardianSound {
            state: Arc::clone(&self.state),
            first_update: true,
        }
    }
}

/// A direct Rust counterpart of vanilla's `GuardianSound extends MovingSound`.
///
/// Vanilla plays the object once with `PositionedSound`'s constructor defaults (origin, volume
/// 1, pitch 1), then `SoundManager.updateAllSounds` calls `update()` before the first refresh.
/// `first_update` preserves that observable start edge while the shared state supplies every
/// later position/progress sample.
#[derive(Debug)]
struct GuardianSound {
    state: Arc<Mutex<GuardianSoundValues>>,
    first_update: bool,
}

impl GuardianSound {
    fn initial_request() -> PlayRequest {
        PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(GUARDIAN_ATTACK_SOUND),
            volume: 1.0,
            pitch: 1.0,
            position: Some(DVec3::ZERO),
            attenuation: mc_audio::AttenuationType::None,
            repeat: true,
            repeat_delay_ticks: 0,
        }
    }

    fn live_request(state: GuardianSoundValues) -> PlayRequest {
        // java: `func_175477_p(0.0F)` is a float division.  The tracker has already clamped the
        // counter to this elder/ordinary maximum, but keeping the denominator defensive makes a
        // malformed synthetic snapshot deterministic rather than producing NaN.
        let f = state.ticks as f32 / state.max_ticks.max(1) as f32;
        PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(GUARDIAN_ATTACK_SOUND),
            volume: f * f,
            pitch: 0.7 + 0.5 * f,
            position: Some(state.position),
            attenuation: mc_audio::AttenuationType::None,
            repeat: true,
            repeat_delay_ticks: 0,
        }
    }
}

impl TickableSound for GuardianSound {
    fn update(&mut self) {
        // `Client::tick_guardian_sounds` refreshed the shared entity snapshot immediately before
        // the engine's update pass.  The flag is the one piece of state vanilla changes inside
        // `GuardianSound.update`: after this call, requests use the live fields rather than the
        // constructor defaults.
        self.first_update = false;
    }

    fn is_done_playing(&self) -> bool {
        !self
            .state
            .lock()
            .expect("GuardianSound state lock poisoned")
            .active
    }

    fn request(&self) -> PlayRequest {
        if self.first_update {
            Self::initial_request()
        } else {
            Self::live_request(
                *self
                    .state
                    .lock()
                    .expect("GuardianSound state lock poisoned"),
            )
        }
    }
}

/// The mutable part of one live `MovingSoundMinecart` source.
///
/// Vanilla keeps the sound object pointed at one concrete `EntityMinecart`; the entity tracker
/// supplies the generation-safe equivalent here.  `distance` looks vestigial because 1.8.9's
/// `MovingSoundMinecart.update` writes it but never reads it, but retaining that write is useful
/// when comparing a tick trace against the Java object and prevents a future reader from
/// accidentally deriving volume from the wrong field.
#[derive(Debug, Clone, Copy)]
struct MinecartSoundValues {
    position: DVec3,
    motion: DVec3,
    distance: f32,
    active: bool,
}

#[derive(Debug, Clone, Copy)]
struct MinecartSoundSnapshot {
    network_id: i32,
    entity: mc_world::Entity,
    position: DVec3,
    motion: DVec3,
}

impl Default for MinecartSoundValues {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            motion: DVec3::ZERO,
            distance: 0.0,
            active: true,
        }
    }
}

/// One vanilla `MovingSoundMinecart` retained by the client.
#[derive(Debug)]
pub(crate) struct MinecartSoundOwner {
    /// The network id plus ECS handle identifies one concrete minecart generation.  A reused id
    /// must never refresh the old channel with the replacement cart's position.
    pub(crate) network_id: i32,
    pub(crate) entity: mc_world::Entity,
    state: Arc<Mutex<MinecartSoundValues>>,
}

impl MinecartSoundOwner {
    fn new(network_id: i32, entity: mc_world::Entity) -> Self {
        Self {
            network_id,
            entity,
            state: Arc::new(Mutex::new(MinecartSoundValues::default())),
        }
    }

    /// Refresh the exact fields `MovingSoundMinecart.update` reads, or stop on despawn/id reuse.
    fn refresh(&self, snapshot: Option<(DVec3, DVec3)>) {
        let mut state = self
            .state
            .lock()
            .expect("MinecartSound state lock poisoned");
        let Some((position, motion)) = snapshot else {
            state.active = false;
            return;
        };
        let speed = ((motion.x * motion.x + motion.z * motion.z).sqrt()) as f32;
        if speed >= 0.01 {
            state.distance = (state.distance + 0.0025).clamp(0.0, 1.0);
        } else {
            state.distance = 0.0;
        }
        state.position = position;
        state.motion = motion;
        state.active = true;
    }

    fn is_done(&self) -> bool {
        !self
            .state
            .lock()
            .expect("MinecartSound state lock poisoned")
            .active
    }

    fn tickable(&self) -> MinecartSound {
        MinecartSound {
            state: Arc::clone(&self.state),
            first_update: true,
        }
    }
}

/// A direct Rust counterpart of vanilla's `MovingSoundMinecart`.
#[derive(Debug)]
struct MinecartSound {
    state: Arc<Mutex<MinecartSoundValues>>,
    first_update: bool,
}

impl MinecartSound {
    fn initial_request() -> PlayRequest {
        PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(MINECART_BASE_SOUND),
            volume: 1.0,
            pitch: 1.0,
            position: Some(DVec3::ZERO),
            attenuation: mc_audio::AttenuationType::Linear,
            repeat: true,
            repeat_delay_ticks: 0,
        }
    }

    fn live_request(state: MinecartSoundValues) -> PlayRequest {
        // java: `MathHelper.sqrt_double` returns a float before the cart's double motion is
        // widened by the volume expression.  The pitch is never assigned by
        // `MovingSoundMinecart.update`, so it remains the constructor's exact 1.0F.
        let speed =
            ((state.motion.x * state.motion.x + state.motion.z * state.motion.z).sqrt()) as f32;
        let volume = if speed >= 0.01 {
            speed.clamp(0.0, 0.5) * 0.7
        } else {
            0.0
        };
        PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(MINECART_BASE_SOUND),
            volume,
            pitch: 1.0,
            position: Some(state.position),
            attenuation: mc_audio::AttenuationType::Linear,
            repeat: true,
            repeat_delay_ticks: 0,
        }
    }
}

impl TickableSound for MinecartSound {
    fn update(&mut self) {
        self.first_update = false;
    }

    fn is_done_playing(&self) -> bool {
        !self
            .state
            .lock()
            .expect("MinecartSound state lock poisoned")
            .active
    }

    fn request(&self) -> PlayRequest {
        if self.first_update {
            Self::initial_request()
        } else {
            Self::live_request(
                *self
                    .state
                    .lock()
                    .expect("MinecartSound state lock poisoned"),
            )
        }
    }
}

/// One vanilla `MovingSoundMinecartRiding` retained for the session-owned player.
///
/// This is the unattenuated `minecart.inside` channel, distinct from the world-positioned
/// `minecart.base` source that every cart owns. Its active edge is the local S1B riding relation,
/// not merely the presence of a cart near the player.
#[derive(Debug)]
pub(crate) struct MinecartRidingSoundOwner {
    /// Network id plus ECS generation of the mounted cart.
    pub(crate) network_id: i32,
    pub(crate) entity: mc_world::Entity,
    state: Arc<Mutex<MinecartSoundValues>>,
}

impl MinecartRidingSoundOwner {
    fn new(network_id: i32, entity: mc_world::Entity) -> Self {
        Self {
            network_id,
            entity,
            state: Arc::new(Mutex::new(MinecartSoundValues::default())),
        }
    }

    fn refresh(&self, snapshot: Option<(DVec3, DVec3)>) {
        let mut state = self
            .state
            .lock()
            .expect("MinecartRidingSound state lock poisoned");
        let Some((position, motion)) = snapshot else {
            state.active = false;
            return;
        };
        state.position = position;
        state.motion = motion;
        state.active = true;
    }

    fn is_done(&self) -> bool {
        !self
            .state
            .lock()
            .expect("MinecartRidingSound state lock poisoned")
            .active
    }

    fn tickable(&self) -> MinecartRidingSound {
        MinecartRidingSound {
            state: Arc::clone(&self.state),
            first_update: true,
        }
    }
}

/// A direct Rust counterpart of vanilla's `MovingSoundMinecartRiding`.
#[derive(Debug)]
struct MinecartRidingSound {
    state: Arc<Mutex<MinecartSoundValues>>,
    first_update: bool,
}

impl MinecartRidingSound {
    fn initial_request() -> PlayRequest {
        PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(MINECART_INSIDE_SOUND),
            volume: 1.0,
            pitch: 1.0,
            position: None,
            attenuation: mc_audio::AttenuationType::None,
            repeat: true,
            repeat_delay_ticks: 0,
        }
    }

    fn live_request(state: MinecartSoundValues) -> PlayRequest {
        let speed =
            ((state.motion.x * state.motion.x + state.motion.z * state.motion.z).sqrt()) as f32;
        let volume = if speed >= 0.01 {
            speed.clamp(0.0, 1.0) * 0.75
        } else {
            0.0
        };
        PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(MINECART_INSIDE_SOUND),
            volume,
            pitch: 1.0,
            position: None,
            attenuation: mc_audio::AttenuationType::None,
            repeat: true,
            repeat_delay_ticks: 0,
        }
    }
}

impl TickableSound for MinecartRidingSound {
    fn update(&mut self) {
        self.first_update = false;
    }

    fn is_done_playing(&self) -> bool {
        !self
            .state
            .lock()
            .expect("MinecartRidingSound state lock poisoned")
            .active
    }

    fn request(&self) -> PlayRequest {
        if self.first_update {
            Self::initial_request()
        } else {
            Self::live_request(
                *self
                    .state
                    .lock()
                    .expect("MinecartRidingSound state lock poisoned"),
            )
        }
    }
}

/// Builds the engine the client owns, with the registry the asset load already produced and the
/// nine volume sliders `options.txt` already carries.
///
/// Constructed **unconditionally**, with no headless branch: `AudioEngine::new` catches a
/// missing output device itself and runs silent, which is vanilla's own behaviour
/// (`SoundManager.loadSoundSystem` catches the `RuntimeException` and leaves `loaded == false`).
/// A `Client` that only builds audio "when a device exists" is a `Client` whose sound path is
/// untested everywhere it is built, which is the state this replaces.
///
/// Set to anything but empty or `0` for a run that shares a desktop with someone, such as a test
/// suite or an image check: the client never opens the output device. Without it, every `Client`
/// plays through the speakers at the default full volume. Sounds still resolve and record exactly
/// as they do on a machine with no output device.
pub const NO_AUDIO_ENV: &str = "MC_NO_AUDIO";

fn no_audio_requested() -> bool {
    std::env::var_os(NO_AUDIO_ENV).is_some_and(|value| !value.is_empty() && value != "0")
}

/// java: `Minecraft.startGame:506` — `mcSoundHandler = new SoundHandler(...)`, registered as a
/// resource-reload listener, before the main menu is shown.
pub(crate) fn build_engine(assets: &GameAssets, settings: &GameSettings) -> AudioEngine {
    // The registry is cloned rather than moved: `GameAssets` keeps its copy because
    // `assets.rs` is what a resource-pack reload rebuilds, and `refresh_sound_registry`
    // copies the new one across afterwards. It is a `HashMap` of a few hundred entries.
    let mut engine = if no_audio_requested() {
        AudioEngine::silent(assets.sounds.clone(), VARIANT_SEED)
    } else {
        AudioEngine::new(assets.sounds.clone(), VARIANT_SEED)
    };
    apply_volumes(&mut engine, settings);
    if engine.is_silent() {
        tracing::info!(
            "no audio output device; the sound path will resolve and drop every sound (see \
             AudioEngine::is_silent)"
        );
    } else {
        engine.prewarm_static_sounds(&assets.resources, is_prewarmed_sound_event);
    }
    engine
}

/// Event-name prefixes whose effects are decoded on a background thread at load and after every
/// pack reload (HITCH-3): what a PvP game plays in its first seconds — hits and hurt, bows, TNT
/// and explosions, bridging (`dig`/`step` per material), generator pickups, countdown notes,
/// villager shops, fire and liquids. The engine also decodes every effect a resource pack supplies
/// and reads ahead the rest; a first play of anything not decoded yet streams instead of decoding
/// on the game thread (`mc_audio::AudioEngine::prewarm_static_sounds`).
const PREWARMED_SOUND_EVENTS: &[&str] = &[
    "random.",
    "dig.",
    "step.",
    "game.player.",
    "game.tnt.",
    "game.potion.",
    "note.",
    "mob.villager.",
    "fire.",
    "liquid.",
];

fn is_prewarmed_sound_event(event: &mc_assets::ResourceLocation) -> bool {
    event.namespace == "minecraft"
        && PREWARMED_SOUND_EVENTS
            .iter()
            .any(|prefix| event.path.starts_with(prefix))
}

/// Copies `options.txt`'s nine `soundCategory_*` values into the engine.
///
/// There used to be two rival owners of these nine numbers —
/// `GameSettings::sound_levels` (which the client loads and saves) and
/// `mc_audio::CategoryVolumes` (which the engine reads, and which nothing populated). The
/// `mc-audio` side's `options.txt` parser and writer are gone; `GameSettings` owns the file
/// format and this function is the one place the values cross. Vanilla has one owner too:
/// `SoundManager.getSoundCategoryVolume` reads `GameSettings.getSoundLevel` directly.
fn apply_volumes(engine: &mut AudioEngine, settings: &GameSettings) {
    for category in SoundCategory::ALL {
        engine
            .categories
            .set(category, settings.sound_level(category.category_name()));
    }
    // Last, and through the setter: `Master` is the one category the engine does *not* fold
    // into each sound's gain — it is the mixer's main track, matching
    // `SoundSystem.setMasterVolume`. The loop above only recorded the slider value.
    engine.set_master_volume(settings.sound_level(SoundCategory::Master.category_name()));
}

/// The listener transform for a player in this state, as
/// `SoundManager.setListener(player, partialTicks)` builds it.
///
/// The previous position and yaw/pitch are supplied separately because the root client owns the
/// correction-aware camera snapshot; keeping them together prevents listener interpolation from
/// crossing a server teleport.
///
/// The eye height is taken from [`PhysicsState::eye_pos`] rather than a constant because vanilla
/// reads `player.getEyeHeight()`, which is 0.08 lower while sneaking.
#[cfg(test)]
fn listener_for(state: &PhysicsState, partial_ticks: f32) -> Listener {
    listener_for_with_previous(state, None, None, partial_ticks)
}

/// Builds the listener with an optional previous-tick feet position and `(yaw, pitch)` pair.
///
/// `SoundManager.setListener` interpolates all three coordinates before adding the current eye
/// height, then linearly interpolates both angles before deriving its at/up vectors.
fn listener_for_with_previous(
    state: &PhysicsState,
    previous_position: Option<DVec3>,
    previous_rotation: Option<(f32, f32)>,
    partial_ticks: f32,
) -> Listener {
    let eye_height = state.eye_pos().y - state.pos.y;
    let previous_position = previous_position.unwrap_or(state.pos);
    let (previous_yaw, previous_pitch) = previous_rotation.unwrap_or((state.yaw, state.pitch));
    Listener::interpolated(
        previous_position,
        state.pos,
        eye_height,
        previous_pitch,
        state.pitch,
        previous_yaw,
        state.yaw,
        partial_ticks,
    )
}

/// Turns a decoded `S29PacketSoundEffect` into the request the engine plays.
///
/// The fixed-point and namespace conversions are `mc-audio`'s
/// ([`mc_audio::packet::SoundEffectPacket`]) — the wire carries `(int)(x * 8.0)` per axis and
/// `(int)(pitch * 63.0)` in one byte, and no `minecraft:` prefix on the name. This function is
/// only the field copy the two crates' doc comments each predicted would be needed, since
/// `mc-audio` deliberately does not depend on `mc-protocol`.
///
/// java: `NetHandlerPlayClient.handleSoundEffect` (`:1695`) → `WorldClient.playSound` (`:439`).
fn sound_request(packet: &s2c::SoundEffect) -> Result<PlayRequest, AudioError> {
    mc_audio::packet::SoundEffectPacket::from_wire_fields(
        packet.sound_name.clone(),
        packet.x_fixed,
        packet.y_fixed,
        packet.z_fixed,
        packet.volume,
        packet.pitch_quantized,
    )
    .into_request()
}

/// The small adapter between `MusicTicker`'s vanilla-shaped handle API and the client's retained
/// `AudioEngine` channels.
///
/// `MusicTicker` deliberately does not know about resources or a backend. Keeping that policy in
/// this module lets the ticker's tested timing stay in `mc-audio`, while music uses the exact same
/// registry/category/decode path as every other client sound. A missing output device still calls
/// `AudioEngine::play` and records a resolved `NoBackend` attempt; it simply has no
/// `PlayingSound` to retain, so the next ticker update observes completion just as
/// `SoundManager.isSoundPlaying` would after a failed sound-system load.
struct ClientMusicPlayback<'a> {
    engine: &'a mut AudioEngine,
    listener: Listener,
    resources: &'a mc_assets::ResourceManager,
    current: &'a mut Option<(u64, PlayingSound)>,
    next_handle: &'a mut u64,
}

impl MusicPlayback for ClientMusicPlayback<'_> {
    fn play(&mut self, event: &mc_assets::ResourceLocation) -> u64 {
        let handle = *self.next_handle;
        *self.next_handle = self.next_handle.wrapping_add(1);
        let request = PlayRequest::ambient(event.clone());
        match self.engine.play(&request, &self.listener, self.resources) {
            Ok(Some(sound)) => *self.current = Some((handle, sound)),
            Ok(None) => *self.current = None,
            Err(error) => {
                *self.current = None;
                tracing::warn!(%error, event = %request.event, "music failed to play");
            }
        }
        handle
    }

    fn is_playing(&self, handle: u64) -> bool {
        self.current.as_ref().is_some_and(|(current, sound)| {
            *current == handle && self.engine.is_sound_playing(sound)
        })
    }

    fn stop(&mut self, handle: u64) {
        let matches = self
            .current
            .as_ref()
            .is_some_and(|(current, _)| *current == handle);
        if !matches {
            return;
        }
        if let Some((_, sound)) = self.current.take() {
            let _ = self.engine.stop_sound(&sound);
        }
    }
}

/// The game-state portion of vanilla's `Minecraft.getAmbientMusicType`.
///
/// Keeping the selector pure makes the priority ordering independently testable: Nether and End
/// must win over creative, and End-boss must win over ordinary End. `has_player` represents
/// vanilla's `thePlayer != null` guard; `end_boss` is the render/status owner's answer to
/// `BossStatus.bossName != null && statusBarTime > 0`.
fn ambient_music_type_for(
    has_player: bool,
    dimension: i32,
    creative_mode: bool,
    allow_flying: bool,
    end_boss: bool,
) -> MusicType {
    if !has_player {
        MusicType::Menu
    } else if dimension == -1 {
        MusicType::Nether
    } else if dimension == 1 {
        if end_boss {
            MusicType::EndBoss
        } else {
            MusicType::End
        }
    } else if creative_mode && allow_flying {
        MusicType::Creative
    } else {
        MusicType::Game
    }
}

/// Whether vanilla's render-owned `BossStatus` currently exposes a live boss bar.
///
/// Entity lifetime is deliberately not part of this predicate: the renderer refreshes the
/// status for every accepted boss frame and the HUD expires it independently after 100 rendered
/// frames. Music therefore follows the same last-writer/expiry state as the visible bar.
fn has_active_boss_status(hud: &mc_ui::screens::hud::Hud) -> bool {
    hud.boss_name().is_some() && hud.boss_status_bar_time() > 0
}

/// Whether `event` is being reported for the first time this session.
///
/// Process-wide rather than a `Client` field so the per-packet path needs no new state threaded
/// through the client; bounded so a server inventing names cannot grow it without limit (past
/// the bound every report is a repeat, i.e. `debug`).
fn first_sound_warning(event: &str) -> bool {
    use std::sync::{Mutex, OnceLock, PoisonError};
    const MAX_DISTINCT: usize = 256;
    static SEEN: OnceLock<Mutex<mc_core::FastSet<String>>> = OnceLock::new();
    let mut seen = SEEN
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if seen.len() >= MAX_DISTINCT || seen.contains(event) {
        return false;
    }
    seen.insert(event.to_owned())
}

impl Client {
    /// Refresh and start the client-local Guardian attack sounds.
    ///
    /// java: `NetHandlerPlayClient.handleEntityStatus` starts a `GuardianSound` immediately for
    /// opcode 21, while `SoundManager.updateAllSounds` updates every retained moving source once
    /// per client tick.  Entity status edges are drained here, after `Session::tick` has advanced
    /// positions and the Guardian wind-up counter, but before the engine's own update pass.  New
    /// sounds still retain `PositionedSound`'s constructor defaults for their first play; the
    /// shared snapshot is visible to the first subsequent `update()` exactly as in vanilla.
    fn tick_guardian_sounds(&mut self) {
        let snapshots = self
            .session
            .as_ref()
            .map(|session| {
                let state = session.state();
                self.guardian_sounds
                    .iter()
                    .map(|owner| {
                        state.entities.guardian_sound_state(
                            &state.world,
                            owner.event.network_id,
                            owner.event.entity,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![None; self.guardian_sounds.len()]);
        for (owner, snapshot) in self.guardian_sounds.iter().zip(snapshots) {
            owner.refresh(snapshot);
        }
        // The engine's retained tickable still owns the shared Arc for one final update pass when
        // a real backend is active.  Dropping this application owner prevents stale id/generation
        // entries from being considered on the next client tick; `GuardianSound::is_done_playing`
        // performs the actual channel stop in the engine.
        self.guardian_sounds.retain(|owner| !owner.is_done());

        let events = self
            .session
            .as_mut()
            .map(|session| session.state_mut().entities.take_guardian_attack_events())
            .unwrap_or_default();
        for event in events {
            // java: SoundManager's master-volume guard runs before the sound registry lookup and
            // before a tickable enters its retained set.  Keep this status path consistent with
            // Client::play_sound's ordinary one-shot funnel.
            if self.audio.categories.get(SoundCategory::Master) <= 0.0 {
                continue;
            }

            let snapshot = self.session.as_ref().and_then(|session| {
                let state = session.state();
                state
                    .entities
                    .guardian_sound_state(&state.world, event.network_id, event.entity)
            });
            let owner = GuardianSoundOwner::new(event);
            let tickable = owner.tickable();
            match self
                .audio
                .play_tickable(tickable, &self.listener, &self.assets.resources)
            {
                Ok(Some(_)) => {
                    // `play_tickable` has already captured the constructor-default request.  The
                    // state update is deliberately after that call so the retained tickable's
                    // next engine update observes the post-world-tick entity values.
                    owner.refresh(snapshot);
                    if snapshot.is_some_and(|snapshot| snapshot.active) {
                        self.guardian_sounds.push(owner);
                    }
                }
                Ok(None) => {
                    // A muted category or no output device produces a resolved, non-retained
                    // attempt.  There is no live backend owner to refresh in that case.
                }
                Err(AudioError::UnknownEvent(event_name)) => {
                    tracing::warn!(
                        event = %event_name,
                        "unable to play Guardian attack sound"
                    );
                }
                Err(AudioError::EmptyEvent(event_name)) => {
                    tracing::warn!(
                        event = %event_name,
                        "Guardian attack sound has no variants"
                    );
                }
                Err(error) => {
                    tracing::warn!(%error, "Guardian attack sound failed to play");
                }
            }
        }
    }

    /// Refresh and start client-local `MovingSoundMinecart` sources.
    ///
    /// java: `WorldClient.spawnEntityInWorld` calls `SoundHandler.playSound(new
    /// MovingSoundMinecart((EntityMinecart)entity))`, and `SoundManager.updateAllSounds` then
    /// drives that source every client tick.  The tracker exposes the concrete ECS generation,
    /// so a destroy/reuse packet cannot move an old channel onto a replacement cart.  The
    /// snapshot is gathered before mutating owners because the sound engine retains its own
    /// tickable object between this application pass and the update pass below.
    fn tick_minecart_sounds(&mut self) {
        let snapshots = self
            .session
            .as_ref()
            .map(|session| {
                let state = session.state();
                state
                    .entities
                    .iter()
                    .filter(|(_, tracked)| tracked.kind.is_minecart())
                    .map(|(network_id, tracked)| MinecartSoundSnapshot {
                        network_id,
                        entity: tracked.entity,
                        position: state
                            .world
                            .ecs
                            .get::<&mc_world::entity::Position>(tracked.entity)
                            .map(|position| position.0)
                            .unwrap_or(tracked.target_pos),
                        motion: state
                            .world
                            .ecs
                            .get::<&mc_world::entity::Velocity>(tracked.entity)
                            .map(|velocity| velocity.0)
                            .unwrap_or(DVec3::ZERO),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        for owner in &self.minecart_sounds {
            let snapshot = snapshots
                .iter()
                .find(|snapshot| {
                    snapshot.network_id == owner.network_id && snapshot.entity == owner.entity
                })
                .map(|snapshot| (snapshot.position, snapshot.motion));
            owner.refresh(snapshot);
        }
        self.minecart_sounds.retain(|owner| !owner.is_done());

        for snapshot in snapshots {
            if self.minecart_sounds.iter().any(|owner| {
                owner.network_id == snapshot.network_id && owner.entity == snapshot.entity
            }) {
                continue;
            }
            // Java starts one source for every concrete minecart subtype, including chest,
            // furnace, TNT, hopper, spawner and command carts; `minecart.base` is shared.
            let owner = MinecartSoundOwner::new(snapshot.network_id, snapshot.entity);
            let tickable = owner.tickable();
            match self
                .audio
                .play_tickable(tickable, &self.listener, &self.assets.resources)
            {
                Ok(Some(_)) => {
                    owner.refresh(Some((snapshot.position, snapshot.motion)));
                    self.minecart_sounds.push(owner);
                }
                Ok(None) => {
                    // Muted/silent engines intentionally do not retain a backend channel.
                }
                Err(AudioError::UnknownEvent(event_name)) => {
                    tracing::warn!(
                        event = %event_name,
                        entity_id = snapshot.network_id,
                        "unable to play minecart rolling sound"
                    );
                }
                Err(AudioError::EmptyEvent(event_name)) => {
                    tracing::warn!(
                        event = %event_name,
                        entity_id = snapshot.network_id,
                        "minecart rolling sound has no variants"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        entity_id = snapshot.network_id,
                        "minecart rolling sound failed to play"
                    );
                }
            }
        }
    }

    /// Refresh the local player's `MovingSoundMinecartRiding` channel.
    ///
    /// java: `EntityPlayerSP.mountEntity` immediately queues `minecart.inside` for a minecart
    /// mount; its `MovingSoundMinecartRiding.update` keeps the loop alive only while the same
    /// player-to-cart relation remains. The session exposes that relation, while this owner
    /// retains the cart generation so id reuse cannot revive stale cabin audio.
    fn tick_minecart_riding_sound(&mut self) {
        let snapshot = self.session.as_ref().and_then(|session| {
            let network_id = session.riding()?;
            let state = session.state();
            let tracked = state.entities.get(network_id)?;
            tracked.kind.is_minecart().then(|| MinecartSoundSnapshot {
                network_id,
                entity: tracked.entity,
                position: state
                    .world
                    .ecs
                    .get::<&mc_world::entity::Position>(tracked.entity)
                    .map(|position| position.0)
                    .unwrap_or(tracked.target_pos),
                motion: state
                    .world
                    .ecs
                    .get::<&mc_world::entity::Velocity>(tracked.entity)
                    .map(|velocity| velocity.0)
                    .unwrap_or(DVec3::ZERO),
            })
        });

        if let Some(owner) = self.minecart_riding_sound.as_ref() {
            // A network id can be reused after a despawn.  Keep the cabin loop bound to
            // both the id and the ECS generation, just like the rolling cart loop above;
            // otherwise switching mounts would retarget the old sound instead of ending it.
            let same_cart = snapshot.as_ref().is_some_and(|snapshot| {
                snapshot.network_id == owner.network_id && snapshot.entity == owner.entity
            });
            owner.refresh(if same_cart {
                snapshot
                    .as_ref()
                    .map(|snapshot| (snapshot.position, snapshot.motion))
            } else {
                None
            });
            if owner.is_done() {
                self.minecart_riding_sound = None;
            }
        }

        let Some(snapshot) = snapshot else {
            return;
        };
        if self.minecart_riding_sound.is_some() {
            return;
        }

        let owner = MinecartRidingSoundOwner::new(snapshot.network_id, snapshot.entity);
        let tickable = owner.tickable();
        match self
            .audio
            .play_tickable(tickable, &self.listener, &self.assets.resources)
        {
            Ok(Some(_)) => {
                owner.refresh(Some((snapshot.position, snapshot.motion)));
                self.minecart_riding_sound = Some(owner);
            }
            Ok(None) => {
                // Muted/silent engines intentionally do not retain a backend channel.
            }
            Err(AudioError::UnknownEvent(event_name)) => {
                tracing::warn!(
                    event = %event_name,
                    entity_id = snapshot.network_id,
                    "unable to play minecart riding sound"
                );
            }
            Err(AudioError::EmptyEvent(event_name)) => {
                tracing::warn!(
                    event = %event_name,
                    entity_id = snapshot.network_id,
                    "minecart riding sound has no variants"
                );
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    entity_id = snapshot.network_id,
                    "minecart riding sound failed to play"
                );
            }
        }
    }

    /// Select the ambient track using the same nested priority as
    /// `Minecraft.getAmbientMusicType`.
    fn ambient_music_type(&self) -> MusicType {
        let Some(session) = self.session.as_ref() else {
            return MusicType::Menu;
        };
        let state = session.state();
        let has_player = session.predictor().is_some();
        // `BossStatus` is render-owned in vanilla: music follows the same last-writer status and
        // 100-rendered-frame expiry as the visible bar, not the lifetime of any tracked dragon.
        let end_boss = state.local.dimension == 1 && has_active_boss_status(&self.hud);
        ambient_music_type_for(
            has_player,
            state.local.dimension,
            state.local.abilities.creative_mode,
            state.local.abilities.allow_flying,
            end_boss,
        )
    }

    /// Advance the ticker once at the `Minecraft.runTick` audio boundary.
    ///
    /// `GuiWinGame.updateScreen` is a paused-screen exception in 1.8.9: it stops the ordinary
    /// track and force-starts `MusicType.CREDITS`, then updates the sound handler itself. The
    /// shell still drives the client tick while this screen is visible, so we make that forced
    /// start idempotent here and deliberately skip `MusicTicker::update` until the screen closes.
    pub(crate) fn tick_music(&mut self) {
        self.tick_guardian_sounds();
        self.tick_minecart_sounds();
        self.tick_minecart_riding_sound();
        let credits = matches!(&self.ui, Ui::GameOver(screen) if screen.is_win_game());
        let start_credits = credits && !self.credits_music_started;
        self.credits_music_started = credits;

        let wanted = (!credits).then(|| self.ambient_music_type());
        let mut playback = ClientMusicPlayback {
            engine: &mut self.audio,
            listener: self.listener,
            resources: &self.assets.resources,
            current: &mut self.music_playback,
            next_handle: &mut self.next_music_handle,
        };
        if credits {
            if start_credits {
                self.music_ticker.stop(&mut playback);
                self.music_ticker.start(MusicType::Credits, &mut playback);
            }
        } else if let Some(wanted) = wanted {
            self.music_ticker.update(wanted, &mut playback);
        }
    }

    /// Stop every retained sound as `Minecraft.loadWorld` enters its world-swap seam.
    ///
    /// Vanilla calls `SoundHandler.stopSounds()` for both `loadWorld(newWorld)` and
    /// `loadWorld(null)`. That clears `SoundManager`'s channel/delayed/tickable registries, but
    /// it deliberately does **not** call `MusicTicker.func_181557_a`: the ticker keeps its
    /// logical `currentMusic` until the next update observes that `isSoundPlaying` is false and
    /// performs the ordinary type-change/cooldown draws. Preserve that timing here by dropping
    /// only the adapter's retained backend handle, not resetting [`MusicTicker`].
    ///
    /// The record map is the client's `RenderGlobal.mapSoundPositions` equivalent. Its handles
    /// are clones of the same engine channels, so it must be dropped with the engine registry or
    /// a later world could retain a stopped channel from the previous one.
    // java: Minecraft.loadWorld -> SoundHandler.stopSounds -> SoundManager.stopAllSounds
    pub(crate) fn stop_audio_for_world_change(&mut self) {
        stop_world_audio(&mut self.audio, &mut self.record_sounds);
        self.music_playback = None;
        self.guardian_sounds.clear();
        self.minecart_sounds.clear();
        self.minecart_riding_sound = None;
        // `Minecraft.loadWorld` clears EffectRenderer alongside the world. Pickup children are
        // render-only effects whose source ids are no longer valid after this boundary.
        self.pickup_effects.clear();
        self.audio.clear_recent();
    }

    /// Play one sound. **The** funnel: every sound in the game goes through here, exactly as
    /// every sound in vanilla goes through `SoundManager.playSound`.
    ///
    /// java: `SoundManager.playSound` (`client/audio/SoundManager.java:335`).
    pub(crate) fn play_sound(&mut self, request: PlayRequest) {
        // java: `if (this.sndSystem.getMasterVolume() <= 0.0F) { logger.debug("Skipped playing
        // soundEvent: {}, master volume was zero"); }` — checked before resolution, so a muted
        // game does not decode an `.ogg` per sound it is about to throw away. The engine
        // applies master on the mixer's main track, so without this guard the whole decode
        // would still run.
        if self.audio.categories.get(SoundCategory::Master) <= 0.0 {
            tracing::debug!(event = %request.event, "skipped: master volume is zero");
            return;
        }
        match self
            .audio
            .play(&request, &self.listener, &self.assets.resources)
        {
            Ok(_) => {}
            // java: `logger.warn("Unable to play unknown soundEvent: {}", ...)` — a server may
            // legitimately name a sound this client's resource packs do not carry, so this is
            // warn-and-drop, never an error the caller has to handle.
            //
            // A server whose resource pack the player declined names the same
            // custom events over and over, so only the first sighting of each name is a `warn`;
            // repeats drop below the default `info` filter instead of formatting a console line
            // per packet. Vanilla logs every one; this changes the log, not the game.
            Err(AudioError::UnknownEvent(event)) => {
                if first_sound_warning(&event) {
                    tracing::warn!(%event, "unable to play unknown soundEvent (repeats logged at debug)");
                } else {
                    tracing::debug!(%event, "unable to play unknown soundEvent");
                }
            }
            // java: `"Unable to play empty soundEvent: {}"`, one line below the above.
            Err(AudioError::EmptyEvent(event)) => {
                if first_sound_warning(&event) {
                    tracing::warn!(%event, "unable to play empty soundEvent (repeats logged at debug)");
                } else {
                    tracing::debug!(%event, "unable to play empty soundEvent");
                }
            }
            // Vanilla has no equivalent (paulscode swallows load failures internally). A
            // resource pack missing the `.ogg` an event names is the realistic cause.
            Err(error) => {
                tracing::warn!(%error, event = %request.event, "sound failed to play");
            }
        }
    }

    /// java: `handleSoundEffect` (`NetHandlerPlayClient.java:1695`).
    ///
    /// This is what `ClientEvent::SoundEffect` was missing. Vanilla
    /// routes it through `WorldClient.playSound(..., distanceDelay: false)` — note the literal
    /// `false` at `NetHandlerPlayClient.java:1698`, which is why the `distanceDelay` queue
    /// is *not* on this path; the only vanilla caller that passes `true` is
    /// `EntityFirework`, a client-side particle.
    pub(crate) fn react_to_sound_effect(&mut self, packet: &s2c::SoundEffect) {
        match sound_request(packet) {
            Ok(request) => self.play_sound(request),
            // Vanilla's `new ResourceLocation(soundName)` is total. Retain this defensive
            // diagnostic because `sound_request` keeps its fallible API for future conversion
            // failures, but arbitrary server-provided names take the `Ok` path above.
            Err(error) => {
                tracing::warn!(%error, name = %packet.sound_name, "malformed sound name");
            }
        }
    }

    /// java: `GuiButton.playPressSound(SoundHandler)`.
    ///
    /// Non-positional (`AttenuationType.NONE`) at pitch `1.0` and volume
    /// [`BUTTON_PRESS_VOLUME`] — see that constant for why the `1.0F` in the Java call is the
    /// pitch and not the volume. This used to play at `1.0`, four times too loud
    /// (`F-P2-client.audio-02-1`).
    pub(crate) fn play_button_press(&mut self) {
        self.play_sound(PlayRequest::ambient_with_pitch(
            mc_assets::ResourceLocation::minecraft(BUTTON_PRESS),
            1.0,
        ));
    }

    /// java: `RenderGlobal.playSound(String, double, double, double, float, float)` — the client's
    /// only `IWorldAccess.playSound` sink, and an **empty body** in 1.8.9 (the jar's
    /// `bfr.a(String,DDDFF)V` is a bare `return`).
    ///
    /// Every `World.playSoundEffect` and `World.playSoundAtEntity` call made on a `WorldClient`
    /// ends here, because `WorldClient` overrides neither; so does `Entity.playSound` for every
    /// entity except the local `EntityPlayerSP`. The audible copy of such a sound is the server's
    /// `S29PacketSoundEffect`: `WorldManager.playSound` sends it to every nearby player,
    /// **including** the one who placed the block, ignited the fire or picked the item up. A
    /// client that also played its own copy made each of those sounds twice.
    ///
    /// The port keeps the call so each site still reads like the Java and still evaluates the
    /// Java argument list (pitch operands consume their `Random` before the call). Do not route
    /// this to [`Self::play_sound`]; a sound the client really owns goes through
    /// `WorldClient.playSound`, `EntityPlayerSP.playSound` or `SoundHandler.playSound` instead.
    pub(crate) fn render_global_play_sound(_request: PlayRequest) {}

    /// The sound half of `NetHandlerPlayClient.handleCollectItem`, which is silent on a client.
    ///
    /// java: `clientWorldController.playSoundAtEntity(entity, entity instanceof EntityXPOrb ?
    /// "random.orb" : "random.pop", 0.2F, pitch)` with the [`item_collected_pitch`] operands
    /// drawn from `avRandomizer` (`NetHandlerPlayClient.java:832-839`). `playSoundAtEntity`
    /// reaches [`Self::render_global_play_sound`]. The pop the collector hears is the server's
    /// S29 from `EntityItem.onCollideWithPlayer` / `EntityXPOrb.onCollideWithPlayer`; the two
    /// `avRandomizer` operands are still drawn here, exactly as Java evaluates them.
    pub(crate) fn play_item_collected_sound(&mut self, xp_orb: bool, source: DVec3) {
        let event = item_collected_sound(xp_orb);
        let first = self.pickup_rng.next_f32();
        let second = self.pickup_rng.next_f32();
        Self::render_global_play_sound(PlayRequest::at(
            mc_assets::ResourceLocation::minecraft(event),
            0.2,
            item_collected_pitch(first, second),
            source,
        ));
    }

    /// The sound half of `EntityLivingBase.handleStatusUpdate` opcodes 2 (hurt) and 3 (death)
    /// for the **local** player.
    ///
    /// java: `EntityLivingBase.java:1356-1384` calls `this.playSound(getHurtSound() /
    /// getDeathSound(), getSoundVolume(), (rand.nextFloat() - rand.nextFloat()) * 0.2F + 1.0F)`.
    /// On the local player that is `EntityPlayerSP.playSound` (`EntityPlayerSP.java:557-560`),
    /// which plays straight into `WorldClient.playSound(posX, posY, posZ, ..., false)` at the
    /// feet. This is the only copy of their own hurt sound the player ever hears: the server's
    /// `EntityLivingBase.attackEntityFrom` plays it through `EntityPlayer.playSound` →
    /// `playSoundToNearExcept(this, ...)`, which leaves the hurt player out, and sends S19
    /// opcode 2 to the player as well as to every tracker. Remote entities are different: their
    /// status sound reaches the empty `RenderGlobal.playSound` (see
    /// `consume_path_a_status_sound_random`) and their audible copy is the server's S29.
    ///
    /// `WorldClient.getEntityByID` returns `thePlayer` for its own id even after a dying player
    /// was removed from the world, so there is no liveness gate here.
    pub(crate) fn play_local_living_status_sound(&mut self, opcode: i8) {
        let name = match opcode {
            2 => LOCAL_PLAYER_HURT_SOUND,
            3 => LOCAL_PLAYER_DEATH_SOUND,
            _ => return,
        };
        let Some(feet) = self
            .session
            .as_ref()
            .and_then(Session::predictor)
            .map(|predictor| predictor.state.pos)
        else {
            return;
        };
        // java: `EntityLivingBase.getSoundVolume()` is `1.0F`; `EntityPlayer` keeps it. Rust
        // evaluates the left operand first, as Java does.
        let rng = &mut self.local_status_sound_rng;
        let pitch = (rng.next_f32() - rng.next_f32()) * 0.2 + 1.0;
        self.play_sound_with_distance_delay(feet, name, 1.0, pitch, false);
    }

    /// GuiOptions button 8675309's custom press sound.
    ///
    /// Unlike an ordinary `GuiButton`, vanilla creates a fresh `Random`, chooses uniformly from
    /// ANIMALS/BLOCKS/MOBS/PLAYERS/WEATHER, and plays the event at pitch 0.5. The seed stream is
    /// private to this button, so the choice cannot perturb music or weighted sound variants.
    pub(crate) fn play_super_secret_settings_sound(&mut self) {
        let mut random = mc_math::JavaRandom::new(self.super_secret_sound_seed);
        self.super_secret_sound_seed = self.super_secret_sound_seed.wrapping_add(1);
        let event = self.assets.sounds.random_event_from_categories(
            &[
                SoundCategory::Animals,
                SoundCategory::Blocks,
                SoundCategory::Mobs,
                SoundCategory::Players,
                SoundCategory::Weather,
            ],
            &mut random,
        );
        if let Some(event) = event {
            self.play_sound(PlayRequest::ambient_with_pitch(event, 0.5));
        }
    }

    /// Move the listener to the player's ears. Called once a frame from [`Client::frame`], which
    /// is where java calls it: `Minecraft.runGameLoop:1127`, after the tick loop and before the
    /// render.
    ///
    /// No session means no player; the listener keeps its last value (the origin, for a client
    /// that has never joined), matching `setListener`'s own `player != null` guard.
    pub(crate) fn update_listener(&mut self, partial_ticks: f32) {
        // Copied out before the assignment: `PhysicsState` is `Copy`, and holding the borrow on
        // `self.session()` across the write to `self.listener` would not type-check.
        let Some((state, corrections)) = self
            .session()
            .and_then(Session::predictor)
            .map(|predictor| (predictor.state, predictor.corrections))
        else {
            return;
        };
        // `PhysicsState::prev_pos` is captured at the top of the simulation tick, before motion
        // runs, and is therefore the same previous feet pose that vanilla stores on the player.
        // It also becomes the corrected position on the first tick after a teleport, so this
        // avoids pulling the listener across a server correction.
        let previous_position = Some(state.prev_pos);
        let previous_rotation = self
            .camera_prev_rotation
            .filter(|_| self.camera_prev_corrections == corrections);
        self.listener =
            listener_for_with_previous(&state, previous_position, previous_rotation, partial_ticks);
    }

    /// Push `options.txt`'s volume sliders into the engine. Call after anything that can change
    /// them — the options screen, or a `GameSettings` reload.
    ///
    /// java: `GameSettings.setSoundLevel` calls `mc.getSoundHandler().setSoundLevel(category,
    /// volume)` on every change, which is the same push in the other direction.
    pub(crate) fn sync_sound_volumes(&mut self) {
        apply_volumes(&mut self.audio, &self.settings);
    }

    /// Adopt the sound registry the current pack stack produces.
    ///
    /// java: `SoundHandler` is an `IResourceManagerReloadListener`, so `refreshResources` and
    /// the end of `startGame`'s deferred load both re-derive it. `GameAssets` has already
    /// re-parsed every `sounds.json`, so this adopts rather than re-reads. The engine still runs
    /// the full `SoundManager.reloadSoundSystem` boundary before the swap: active/backend state,
    /// delayed repeats, tickables, and category indexes must not survive into the new pack.
    pub(crate) fn refresh_sound_registry(&mut self) {
        self.audio.reload_registry(self.assets.sounds.clone());
        // The reload emptied the decode cache; warm the PvP set again off the game thread so
        // the next fight does not pay every first decode inline (HITCH-3 / AUDIO-6).
        if !self.audio.is_silent() {
            self.audio
                .prewarm_static_sounds(&self.assets.resources, is_prewarmed_sound_event);
        }
        // These are client-owned clones/owners of channels that the engine just tore down. Drop
        // them as part of the same reload boundary so a stopped pre-reload owner cannot block a
        // fresh event from being registered or reused after the next tick.
        self.music_playback = None;
        self.record_sounds.clear();
        self.guardian_sounds.clear();
        self.minecart_sounds.clear();
        self.minecart_riding_sound = None;
        // A pack swap can also rewrite `options.txt` (`GameAssets::reload_packs` takes
        // `&mut settings`), so the sliders come across on the same beat.
        self.sync_sound_volumes();
    }

    // -------------------------------------------------------------------------------------
    // S28 Effect: the `playAuxSFX` / `broadcastSound` / `playRecord` channel
    // -------------------------------------------------------------------------------------

    /// java: `NetHandlerPlayClient.handleEffect` (`:1441`).
    ///
    /// The whole `S28PacketEffect` channel was decoded by `sync::dispatch` and matched by
    /// nothing (`F-P1-world-03-1`, `F-P0-network.play.server-10-2`, `F-P2-client.renderer-11-1`,
    /// `F-P0-network.play-01-8`). Unlike a missing `playSound` fan-out, **there is no server
    /// packet behind this one**: `RenderGlobal.playAuxSFX`, `broadcastSound` and `playRecord`
    /// are all fully implemented client bodies (see the table —
    /// only `playSound`, `playSoundToNearExcept`, `onEntityAdded` and `onEntityRemoved` are
    /// empty), so every one of these 25 effects was simply absent.
    pub(crate) fn react_to_effect(&mut self, packet: &s2c::Effect) {
        if packet.disable_relative_volume {
            self.broadcast_sound(packet.effect_id, packet.position, packet.data);
        } else {
            self.play_aux_sfx(packet.effect_id, packet.position, packet.data);
        }
    }

    /// java: `WorldClient.playSoundAtPos` (`:431`) → `WorldClient.playSound` (`:439`) with
    /// `distanceDelay = false`, which is what every `playAuxSFX` case passes.
    ///
    /// The `+ 0.5` on each axis is `playSoundAtPos`'s, not this port's: the sound is emitted from
    /// the block's centre, and `new PositionedSoundRecord(loc, volume, pitch, x, y, z)` is the
    /// `LINEAR`-attenuation constructor, which is [`PlayRequest`]'s `position: Some(_)`.
    fn play_sound_at_pos(&mut self, pos: mc_math::BlockPos, name: &str, volume: f32, pitch: f32) {
        self.play_sound_at_pos_with_distance_delay(pos, name, volume, pitch, false);
    }

    /// Plays a block-centred sound through the `WorldClient.playSound` distance-delay branch.
    ///
    /// Vanilla measures the delay from the render-view entity's **feet**, not the sound manager's
    /// interpolated eye listener.  A source beyond ten blocks is queued at
    /// `(int)(sqrt(distanceSq) / 40 * 20)` ticks; a source at or inside that boundary goes through
    /// `SoundManager.playSound` immediately.  Keeping this switch at the client seam lets future
    /// client-generated effects (notably `EntityFirework.StarterFX`) use the exact path without
    /// teaching `mc-audio` about entities or worlds.
    ///
    /// java: `WorldClient.playSoundAtPos`/`playSound` (`client/multiplayer/WorldClient.java:431-452`).
    pub(crate) fn play_sound_at_pos_with_distance_delay(
        &mut self,
        pos: mc_math::BlockPos,
        name: &str,
        volume: f32,
        pitch: f32,
        distance_delay: bool,
    ) {
        let source = DVec3::new(
            f64::from(pos.x) + 0.5,
            f64::from(pos.y) + 0.5,
            f64::from(pos.z) + 0.5,
        );
        self.play_sound_with_distance_delay(source, name, volume, pitch, distance_delay);
    }

    /// Plays an exact-coordinate sound through `WorldClient.playSound`'s optional distance-delay
    /// branch. Block effects use [`Self::play_sound_at_pos_with_distance_delay`] above, while
    /// client-created effects such as `EntityFirework.StarterFX` already carry a direct source
    /// position and must not acquire a half-block offset.
    ///
    /// java: `WorldClient.playSound(double, double, double, ..., distanceDelay)`
    pub(crate) fn play_sound_with_distance_delay(
        &mut self,
        source: DVec3,
        name: &str,
        volume: f32,
        pitch: f32,
        distance_delay: bool,
    ) {
        let request = PlayRequest::at(
            mc_assets::ResourceLocation::minecraft(name),
            volume,
            pitch,
            source,
        );
        if distance_delay {
            let view = self
                .session
                .as_ref()
                .and_then(|session| session.predictor())
                .map(|predictor| predictor.state.pos);
            if let Some(delay_ticks) = Self::distance_delay_ticks(view, source) {
                // Do not run the master-volume guard here: Java enqueues first and checks the
                // master fader only when the delayed request reaches `playSound`.
                self.audio.play_delayed_sound(request, delay_ticks);
                return;
            }
        }
        self.play_sound(request);
    }

    /// Java's `WorldClient.playSound` delay calculation, including its strict ten-block gate.
    /// `None` means either no render-view entity exists or the source is near enough for an
    /// immediate play.  The optional view position keeps this pure boundary easy to test while
    /// retaining the client lifecycle guard at the call site.
    fn distance_delay_ticks(view: Option<DVec3>, source: DVec3) -> Option<i32> {
        let view = view?;
        let distance_squared = (view - source).length_squared();
        if distance_squared > 100.0 {
            Some((distance_squared.sqrt() / 40.0 * 20.0) as i32)
        } else {
            None
        }
    }

    /// `this.theWorld.rand.nextFloat() * 0.1F + 0.9F` — the door/anvil pitch wobble.
    ///
    /// **Which RNG.** Vanilla draws from `World.rand`, a `Random` seeded from the wall clock and
    /// shared with world ticking; this draws from [`ParticleSystem`]'s stream, which is the only
    /// `JavaRandom` `Client` has that is not the torch flicker's. Neither is reproducible in
    /// vanilla and no parity fixture can observe either, so sharing one stream is a deliberate
    /// simplification, not an approximation of something knowable.
    fn sfx_pitch_wobble(&mut self) -> f32 {
        self.particles.rng_mut().next_f32() * 0.1 + 0.9
    }

    /// `(random.nextFloat() - random.nextFloat()) * 0.2F + 1.0F` — the mob-sound pitch spread
    /// eleven of the `playAuxSFX` cases share.
    fn sfx_pitch_spread(&mut self) -> f32 {
        let rng = self.particles.rng_mut();
        (rng.next_f32() - rng.next_f32()) * 0.2 + 1.0
    }

    /// java: `RenderGlobal.playAuxSFX(EntityPlayer, int, BlockPos, int)` (`:2160`) — the
    /// 25-case client-side level-event table.
    pub(crate) fn play_aux_sfx(&mut self, sfx: i32, pos: mc_math::BlockPos, data: i32) {
        match sfx {
            // ---- the plain positional sounds ----------------------------------------------
            1000 => self.play_sound_at_pos(pos, "random.click", 1.0, 1.0),
            1001 => self.play_sound_at_pos(pos, "random.click", 1.0, 1.2),
            1002 => self.play_sound_at_pos(pos, "random.bow", 1.0, 1.2),
            1003 => {
                let pitch = self.sfx_pitch_wobble();
                self.play_sound_at_pos(pos, "random.door_open", 1.0, pitch);
            }
            1004 => {
                // java: `2.6F + (rand.nextFloat() - rand.nextFloat()) * 0.8F` — a wider spread
                // than `sfx_pitch_spread`'s, and around 2.6 rather than 1.0.
                let rng = self.particles.rng_mut();
                let pitch = 2.6 + (rng.next_f32() - rng.next_f32()) * 0.8;
                self.play_sound_at_pos(pos, "random.fizz", 0.5, pitch);
            }
            1005 => self.play_record_effect(pos, data),
            1006 => {
                let pitch = self.sfx_pitch_wobble();
                self.play_sound_at_pos(pos, "random.door_close", 1.0, pitch);
            }
            1007 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.ghast.charge", 10.0, pitch);
            }
            1008 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.ghast.fireball", 10.0, pitch);
            }
            1009 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.ghast.fireball", 2.0, pitch);
            }
            1010 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.zombie.wood", 2.0, pitch);
            }
            1011 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.zombie.metal", 2.0, pitch);
            }
            1012 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.zombie.woodbreak", 2.0, pitch);
            }
            // 1013 and 1018 are deliberately absent: they are `broadcastSound`'s two cases and
            // arrive with `serverWide = true`, so `handleEffect` routes them the other way.
            1014 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.wither.shoot", 2.0, pitch);
            }
            1015 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.bat.takeoff", 0.05, pitch);
            }
            1016 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.zombie.infect", 2.0, pitch);
            }
            1017 => {
                let pitch = self.sfx_pitch_spread();
                self.play_sound_at_pos(pos, "mob.zombie.unfect", 2.0, pitch);
            }
            1020 => {
                let pitch = self.sfx_pitch_wobble();
                self.play_sound_at_pos(pos, "random.anvil_break", 1.0, pitch);
            }
            1021 => {
                let pitch = self.sfx_pitch_wobble();
                self.play_sound_at_pos(pos, "random.anvil_use", 1.0, pitch);
            }
            1022 => {
                let pitch = self.sfx_pitch_wobble();
                self.play_sound_at_pos(pos, "random.anvil_land", 0.3, pitch);
            }

            // ---- the particle-only cases ---------------------------------------------------
            2000 => self.sfx_dispenser_smoke(pos, data),
            2001 => self.sfx_block_break(pos, data),
            2002 => self.sfx_potion_smash(pos, data),
            2003 => self.sfx_eye_of_ender(pos),
            2004 => self.sfx_mob_spawner_flames(pos),
            2005 => self.sfx_bonemeal(pos, data),

            // java: the `switch` has no `default`, so an unknown id is silently ignored.
            _ => {}
        }
    }

    /// java: `RenderGlobal.broadcastSound` (`:2123`) — the two "everyone in the dimension hears
    /// this" events, which vanilla plays **two blocks from the listener in the direction of the
    /// source** rather than at the source, so distance attenuation cannot mute them.
    ///
    /// Note the missing `break` in vanilla's `case 1013: case 1018:` block: it falls through to
    /// an empty `default`, which is the same as breaking.
    pub(crate) fn broadcast_sound(&mut self, sound_id: i32, pos: mc_math::BlockPos, _data: i32) {
        let (name, volume) = match sound_id {
            1013 => ("mob.wither.spawn", 1.0),
            1018 => ("mob.enderdragon.end", 5.0),
            _ => return,
        };
        // java: `this.mc.getRenderViewEntity().posX/posY/posZ` — the player's feet. No view
        // entity means no world, which is when `broadcastSound`'s own null check returns.
        let Some(view) = self
            .session
            .as_ref()
            .and_then(|s| s.predictor())
            .map(|p| p.state.pos)
        else {
            return;
        };
        // java: the raw block coordinate, with no `+ 0.5` — unlike `playSoundAtPos`.
        let delta = DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z)) - view;
        let distance = delta.length();
        let at = if distance > 0.0 {
            view + delta / distance * 2.0
        } else {
            view
        };
        self.play_sound(PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(name),
            volume,
            pitch: 1.0,
            position: Some(at),
            attenuation: mc_audio::AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        });
    }

    /// java: `playAuxSFX` case 1005 → `World.playRecord` → `RenderGlobal.playRecord` (`:1997`).
    ///
    /// Three separate things, all of which were missing: stop whatever this jukebox was already
    /// playing, show the "Now playing" line in the HUD, and start the new track.
    /// `F-P1-item-11-10`.
    fn play_record_effect(&mut self, pos: mc_math::BlockPos, data: i32) {
        // java: `mapSoundPositions.get(pos)` — one playing record per jukebox, keyed by position
        // and *not* by track, so ejecting a disc (`data` naming a non-record) stops it too.
        if let Some(record) = record_name_for_item(data) {
            // java: `ItemRecord.getRecordNameLocal` is `item.record.<name>.desc`, and
            // `GuiIngame.setRecordPlayingMessage` (`:1113-1116`) wraps it in
            // `I18n.format("record.nowPlaying", name)`: "Now playing: C418 - 13".
            let key = format!("item.record.{record}.desc");
            let language = &self.assets.language;
            let message = language.format("record.nowPlaying", &[language.tr(&key)]);
            self.hud.set_record_playing(message, true);
        }
        if let Err(error) = drive_record_effect_audio(
            &mut self.audio,
            &mut self.record_sounds,
            &self.listener,
            &self.assets.resources,
            pos,
            data,
        ) {
            tracing::warn!(%error, "record sound failed to play");
        }
        // java: `Item.getItemById(data) instanceof ItemRecord` — anything else means "stop", and
        // reaches `playRecord(pos, null)`.
        // java: `ingameGUI.setRecordPlayingMessage(itemrecord.getRecordNameLocal())`, which is
        // `translateToLocal("item.record." + recordName + ".desc")`.
        // java: `PositionedSoundRecord.create(loc, (float)x, (float)y, (float)z)` — volume
        // **4.0**, pitch 1.0, LINEAR, at the raw block coordinate with no `+ 0.5`.
    }

    /// java: `playAuxSFX` case 2000 — the dispenser's ten smoke puffs, aimed along the facing
    /// `data` encodes as `(data % 3 - 1, data / 3 % 3 - 1)` in X and Z.
    ///
    /// Keep this arithmetic in a small helper because Java's integer remainder and division
    /// truncate toward zero.  The Euclidean operations that are useful for coordinate wrapping
    /// are observably different for negative effect metadata.
    #[must_use]
    fn dispenser_smoke_direction(data: i32) -> (i32, i32) {
        (data % 3 - 1, data / 3 % 3 - 1)
    }

    fn sfx_dispenser_smoke(&mut self, pos: mc_math::BlockPos, data: i32) {
        let (l, i) = Self::dispenser_smoke_direction(data);
        let l = f64::from(l);
        let i = f64::from(i);
        let cx = f64::from(pos.x) + l * 0.6 + 0.5;
        let cy = f64::from(pos.y) + 0.5;
        let cz = f64::from(pos.z) + i * 0.6 + 0.5;
        for _ in 0..10 {
            let (at, speed) = {
                let rng = self.particles.rng_mut();
                let d20 = rng.next_f64() * 0.2 + 0.01;
                let x = cx + l * 0.01 + (rng.next_f64() - 0.5) * i * 0.5;
                let y = cy + (rng.next_f64() - 0.5) * 0.5;
                let z = cz + i * 0.01 + (rng.next_f64() - 0.5) * l * 0.5;
                let vx = l * d20 + rng.next_gaussian() * 0.01;
                let vy = -0.03 + rng.next_gaussian() * 0.01;
                let vz = i * d20 + rng.next_gaussian() * 0.01;
                (DVec3::new(x, y, z), DVec3::new(vx, vy, vz))
            };
            self.spawn_effect_particle(particle_ids::SMOKE_NORMAL, at, speed, &[]);
        }
    }

    /// java: `playAuxSFX` case 2001 — **another player** breaking a block: its `dig.*` sound and
    /// its 64-particle debris burst. `data` packs `Block.getStateId`: id in the low 12 bits,
    /// metadata in bits 12..20. `F-P1-block-09-5`, `F-P2-client.renderer-11-2`.
    pub(crate) fn sfx_block_break(&mut self, pos: mc_math::BlockPos, data: i32) {
        let block = mc_data::BlockId(u16::try_from(data & 4095).unwrap_or(0));
        let state = mc_data::BlockState::new(block, ((data >> 12) & 255) as u8);
        if mc_data::block_props(state).material != mc_data::Material::Air {
            let info = mc_data::block_info(block);
            self.play_sound(PlayRequest {
                event: mc_assets::ResourceLocation::minecraft(info.break_sound),
                // java: `(stepSound.getVolume() + 1.0F) / 2.0F` and
                // `stepSound.getFrequency() * 0.8F`.
                volume: (info.step_sound_volume + 1.0) / 2.0,
                pitch: info.step_sound_pitch * 0.8,
                position: Some(DVec3::new(
                    f64::from(pos.x) + 0.5,
                    f64::from(pos.y) + 0.5,
                    f64::from(pos.z) + 0.5,
                )),
                attenuation: mc_audio::AttenuationType::Linear,
                repeat: false,
                repeat_delay_ticks: 0,
            });
        }
        self.add_block_destroy_effects(pos, state);
    }

    /// java: `playAuxSFX` case 2002 — a splash potion shattering. `F-P1-entity.projectile-04-4`.
    fn sfx_potion_smash(&mut self, pos: mc_math::BlockPos, data: i32) {
        let base = DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));
        // java: eight `ITEM_CRACK` shards carrying `[potionitem id, data]`.
        for _ in 0..8 {
            let speed = {
                let rng = self.particles.rng_mut();
                DVec3::new(
                    rng.next_gaussian() * 0.15,
                    rng.next_f64() * 0.2,
                    rng.next_gaussian() * 0.15,
                )
            };
            self.spawn_effect_particle(
                particle_ids::ITEM_CRACK,
                base,
                speed,
                &[POTION_ITEM_ID, data],
            );
        }
        // java: `Items.potionitem.getColorFromDamage(data)` -> `PotionHelper.getLiquidColor`
        // and `isEffectInstant(data)`. The metadata decoder is shared with item/HUD consumers,
        // so malformed high bits and effect ordering retain the same total vanilla semantics.
        let (particle_id, color) = potion_smash_visuals(data);
        for _ in 0..100 {
            let (at, speed, radius, brightness) = {
                let rng = self.particles.rng_mut();
                let d22 = rng.next_f64() * 4.0;
                let d23 = rng.next_f64() * std::f64::consts::PI * 2.0;
                let d24 = d23.cos() * d22;
                let d9 = 0.01 + rng.next_f64() * 0.5;
                let d11 = d23.sin() * d22;
                let f3 = 0.75 + rng.next_f32() * 0.25;
                (
                    DVec3::new(base.x + d24 * 0.1, base.y + 0.3, base.z + d11 * 0.1),
                    DVec3::new(d24, d9, d11),
                    d22,
                    f3,
                )
            };
            self.spawn_effect_particle_with(
                particle_id,
                at,
                speed,
                &[],
                |particle: &mut mc_render::particle::Particle| {
                    // java: `entityfx.setRBGColorF(f * f3, f1 * f3, f2 * f3)` then
                    // `entityfx.multiplyVelocity((float)d22)`.
                    particle.color[0] = color[0] * brightness;
                    particle.color[1] = color[1] * brightness;
                    particle.color[2] = color[2] * brightness;
                    particle.motion *= radius;
                },
            );
        }
        let pitch = self.sfx_pitch_wobble();
        self.play_sound_at_pos(pos, "game.potion.smash", 1.0, pitch);
    }

    /// java: `playAuxSFX` case 2003 — an eye of ender shattering: eight item shards plus two
    /// counter-rotating rings of forty `PORTAL` particles.
    fn sfx_eye_of_ender(&mut self, pos: mc_math::BlockPos) {
        let cx = f64::from(pos.x) + 0.5;
        let cy = f64::from(pos.y);
        let cz = f64::from(pos.z) + 0.5;
        let base = DVec3::new(cx, cy, cz);
        for _ in 0..8 {
            let speed = {
                let rng = self.particles.rng_mut();
                DVec3::new(
                    rng.next_gaussian() * 0.15,
                    rng.next_f64() * 0.2,
                    rng.next_gaussian() * 0.15,
                )
            };
            self.spawn_effect_particle(particle_ids::ITEM_CRACK, base, speed, &[ENDER_EYE_ITEM_ID]);
        }
        // java: `for (double d18 = 0.0D; d18 < (Math.PI * 2D); d18 += 0.15707963267948966D)`.
        let mut angle = 0.0f64;
        while angle < std::f64::consts::PI * 2.0 {
            let (cos, sin) = (angle.cos(), angle.sin());
            let at = DVec3::new(cx + cos * 5.0, cy - 0.4, cz + sin * 5.0);
            self.spawn_effect_particle(
                particle_ids::PORTAL,
                at,
                DVec3::new(cos * -5.0, 0.0, sin * -5.0),
                &[],
            );
            self.spawn_effect_particle(
                particle_ids::PORTAL,
                at,
                DVec3::new(cos * -7.0, 0.0, sin * -7.0),
                &[],
            );
            angle += 0.157_079_632_679_489_66;
        }
    }

    /// java: `playAuxSFX` case 2004 — the twenty smoke-and-flame pairs a mob spawner throws when
    /// it is broken or forced. `F-P1-tileentity-02-1`.
    fn sfx_mob_spawner_flames(&mut self, pos: mc_math::BlockPos) {
        for _ in 0..20 {
            let at = {
                let rng = self.particles.rng_mut();
                DVec3::new(
                    f64::from(pos.x) + 0.5 + (f64::from(rng.next_f32()) - 0.5) * 2.0,
                    f64::from(pos.y) + 0.5 + (f64::from(rng.next_f32()) - 0.5) * 2.0,
                    f64::from(pos.z) + 0.5 + (f64::from(rng.next_f32()) - 0.5) * 2.0,
                )
            };
            self.spawn_effect_particle(particle_ids::SMOKE_NORMAL, at, DVec3::ZERO, &[]);
            self.spawn_effect_particle(particle_ids::FLAME, at, DVec3::ZERO, &[]);
        }
    }

    /// java: `playAuxSFX` case 2005 → `ItemDye.spawnBonemealParticles` (`:130`).
    /// `F-P1-item-06-10`.
    fn sfx_bonemeal(&mut self, pos: mc_math::BlockPos, data: i32) {
        // java: `if (amount == 0) amount = 15;`
        let amount = if data == 0 { 15 } else { data };
        let Some(state) = self
            .session
            .as_ref()
            .map(|session| session.state().world.block_at(pos))
        else {
            return;
        };
        let props = mc_data::block_props(state);
        if props.material == mc_data::Material::Air {
            return;
        }
        // java: `block.setBlockBoundsBasedOnState(worldIn, pos)` then `getBlockBoundsMaxY()`.
        let max_y = props.bounds.max_y;
        for _ in 0..amount {
            let (at, speed) = {
                let rng = self.particles.rng_mut();
                let vx = rng.next_gaussian() * 0.02;
                let vy = rng.next_gaussian() * 0.02;
                let vz = rng.next_gaussian() * 0.02;
                let at = DVec3::new(
                    f64::from(pos.x as f32 + rng.next_f32()),
                    f64::from(pos.y) + f64::from(rng.next_f32()) * max_y,
                    f64::from(pos.z as f32 + rng.next_f32()),
                );
                (at, DVec3::new(vx, vy, vz))
            };
            self.spawn_effect_particle(particle_ids::VILLAGER_HAPPY, at, speed, &[]);
        }
    }
}

/// The `EnumParticleTypes` ids `playAuxSFX` names, re-exported from `mc-render`'s table so this
/// file names them the way the Java does rather than by number.
mod particle_ids {
    pub use mc_render::particle::packet::{
        FLAME, ITEM_CRACK, PORTAL, SMOKE_NORMAL, SPELL, SPELL_INSTANT, VILLAGER_HAPPY,
    };
}

/// Resolve the two metadata-dependent parts of `RenderGlobal.playAuxSFX` case 2002.
///
/// Java uses the brewed liquid colour and selects `SPELL_INSTANT` when any decoded effect is
/// instant. Keeping this pure makes the exact metadata decision independently testable while
/// [`Client::sfx_potion_smash`] remains the live particle/audio producer.
fn potion_smash_visuals(data: i32) -> (i32, [f32; 3]) {
    let packed = mc_data::potion_liquid_color_from_damage(data);
    let color = [
        ((packed >> 16) & 255) as f32 / 255.0,
        ((packed >> 8) & 255) as f32 / 255.0,
        (packed & 255) as f32 / 255.0,
    ];
    let instant = mc_data::potion_effects_from_damage(data)
        .iter()
        .any(|effect| mc_data::potion(effect.id).is_some_and(|potion| potion.instant));
    let particle_id = if instant {
        particle_ids::SPELL_INSTANT
    } else {
        particle_ids::SPELL
    };
    (particle_id, color)
}

/// java: `Item.getIdFromItem(Items.potionitem)`.
const POTION_ITEM_ID: i32 = 373;
/// java: `Item.getIdFromItem(Items.ender_eye)`.
const ENDER_EYE_ITEM_ID: i32 = 381;

// family name no other `SoundType` uses (`ladder`, `anvil`, `slime`) — but `soundTypeGlass`'s
// java: `SLIME_SOUND.getBreakSound()` (`Block.java:85-88`) — `mob.slime.big`, where
// `getStepSound()` is `mob.slime.small`.

// Every block registered with `Block.soundTypeGlass` in 1.8.9, by numeric id.
// java: `Block.registerBlocks` — `glass`, `ice`, `glowstone`, `portal`, `stained_glass`,
// `glass_pane`, `end_portal_frame`, `redstone_lamp`, `lit_redstone_lamp`, `stained_glass_pane`,
// `sea_lantern`, `packed_ice`.

/// `ItemRecord.recordName` for an item id, or `None` if the id is not a music disc.
///
/// java: `Item.getItemById(data) instanceof ItemRecord`, then `((ItemRecord)…).recordName`.
/// `mc-data` carries `ItemKind::Record`, and every record's registry name is
/// `minecraft:record_<recordName>` — `ItemRecord`'s constructor argument and its registry id are
/// the same string in `Items.registerItems`.
fn record_name_for_item(item_id: i32) -> Option<&'static str> {
    // Keep this dispatch in lockstep with `ItemRecord.addInformation`'s tooltip line.
    let id = mc_data::ItemId(u16::try_from(item_id).ok()?);
    mc_data::record_info(id).map(|record| record.record_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::sync::ClientEvent;
    use mc_audio::{PlayOutcome, SoundRegistry};
    use mc_math::DVec3;
    use mc_world::entity::Metadata;
    use std::collections::HashMap;
    use std::io::{Cursor, Read};

    #[test]
    fn item_collected_pitch_uses_java_float_formula_and_type_seam() {
        assert_eq!(item_collected_pitch(0.0, 0.0), 2.0);
        assert_eq!(item_collected_pitch(0.0, 1.0), 0.6);
        assert_eq!(item_collected_pitch(1.0, 0.0), 3.4);
        assert!(
            (0.6..=3.4).contains(&item_collected_pitch(0.999, 0.001)),
            "EntityPickupFX sound pitch remains in Java's random range"
        );
        assert_eq!(item_collected_sound(true), "random.orb");
        assert_eq!(item_collected_sound(false), "random.pop");

        let audio = include_str!("audio.rs");
        assert!(audio.contains("instanceof EntityXPOrb"));
        assert!(audio.contains("play_item_collected_sound"));
    }

    #[test]
    fn world_client_distance_delay_uses_a_strict_ten_block_gate_and_java_floor() {
        let view = DVec3::ZERO;
        assert_eq!(
            Client::distance_delay_ticks(Some(view), DVec3::new(10.0, 0.0, 0.0)),
            None,
            "distanceSq == 100 is immediate in WorldClient"
        );
        assert_eq!(
            Client::distance_delay_ticks(Some(view), DVec3::new(10.01, 0.0, 0.0)),
            Some(5),
            "a just-far-enough source queues floor(distance / 2) ticks"
        );
        assert_eq!(
            Client::distance_delay_ticks(Some(view), DVec3::new(20.0, 0.0, 0.0)),
            Some(10),
            "the 20-block negative control must not be treated as immediate"
        );
        assert_eq!(
            Client::distance_delay_ticks(None, DVec3::new(100.0, 0.0, 0.0)),
            None,
            "without a render-view entity there is no distance-delay decision"
        );
    }

    /// `EntityZombie.handleStatusUpdate(16)` calls the client `WorldClient.playSound` sink
    /// directly, so it must reach the real registry from the entity-status queue. This exercises
    /// the live `Client::tick` drain, rather than only checking that `EntityTracker` queued the
    /// event. The entity position is deliberately nonzero to pin Java's +0.5 coordinate offset.
    #[test]
    fn zombie_conversion_status_reaches_the_live_audio_sink_at_entity_center() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, to_game, _to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, crate::session::JoinSettings::default());
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".to_owned(),
                    reduced_debug_info: false,
                }),
            ))
            .expect("the synthetic join must reach Session::poll");
        let _ = session.poll(16);
        {
            let state = session.state_mut();
            let source_entity = state.entities.spawn(
                &mut state.world,
                9,
                TrackedKind::Type(mc_data::EntityType::Zombie),
                None,
                [7 * 32, 64 * 32, -3 * 32],
                0.0,
                0.0,
                DVec3::ZERO,
                Metadata::new(),
            );
            assert!(state.entities.apply_status_in_world(&state.world, 9, 16));
            let replacement_entity = state.entities.spawn(
                &mut state.world,
                9,
                TrackedKind::Type(mc_data::EntityType::Zombie),
                None,
                [-11 * 32, 70 * 32, 13 * 32],
                0.0,
                0.0,
                DVec3::ZERO,
                Metadata::new(),
            );
            assert_ne!(source_entity, replacement_entity);
        }
        client.session = Some(session);
        client.audio.clear_recent();

        client.tick(&mc_physics::state::MoveInput::default());

        let record = client
            .audio
            .last_play()
            .expect("opcode 16 must reach the live audio engine");
        assert_eq!(record.request.event.path, "mob.zombie.remedy");
        assert_eq!(record.request.position, Some(DVec3::new(7.5, 64.5, -2.5)));
        assert_ne!(record.request.position, Some(DVec3::new(-10.5, 70.5, 13.5)));
        assert!(record.request.volume >= 1.0 && record.request.volume < 2.0);
        assert!(record.request.pitch >= 0.3 && record.request.pitch < 1.0);
        // `AudioEngine` still records the request when the host is headless or the optional audio
        // object is unavailable; the event/path/position above are the client-local sink contract.
    }

    /// `EntityFirework.StarterFX` is the one current client callback that passes
    /// `distanceDelay = true` to `WorldClient.playSound`. This crosses the client adapter rather
    /// than only testing its arithmetic: the exact ten-block source plays immediately, while the
    /// twenty-block source stays queued for Java's ten-tick floor and then resolves as a one-shot
    /// with its direct (non-block-centred) position.
    #[test]
    fn firework_direct_distance_delay_queues_far_source_but_not_boundary_source() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, to_game, _to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, crate::session::JoinSettings::default());
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".to_owned(),
                    reduced_debug_info: false,
                }),
            ))
            .expect("the synthetic join must reach Session::poll");
        session.poll(16);
        // The listener distances below are measured from the origin. JoinGame now builds
        // vanilla's first player at spawn + (0.5, 1, 0.5), so pin the origin pose.
        if let Some(predictor) = session.predictor_mut() {
            predictor.state.set_position(DVec3::ZERO);
            predictor.state.prev_pos = DVec3::ZERO;
            predictor.state.yaw = 0.0;
            predictor.state.pitch = 0.0;
        }
        client.session = Some(session);
        client.audio.clear_recent();

        let boundary = DVec3::new(10.0, 0.0, 0.0);
        client.play_sound_with_distance_delay(boundary, "fireworks.blast", 20.0, 1.0, true);
        assert_eq!(
            client.audio.delayed_sound_count(),
            0,
            "distanceSq == 100 is the immediate WorldClient boundary"
        );
        let boundary_record = client
            .audio
            .last_play()
            .expect("the boundary sound must play inline");
        assert_eq!(boundary_record.request.event.path, "fireworks.blast");
        assert_eq!(boundary_record.request.position, Some(boundary));

        let far = DVec3::new(20.0, 0.0, 0.0);
        client.play_sound_with_distance_delay(far, "fireworks.blast_far", 20.0, 1.0, true);
        assert_eq!(
            client.audio.delayed_sound_count(),
            1,
            "a distant StarterFX sound must enter delayedSounds"
        );
        assert_eq!(
            client.audio.last_play().unwrap().request.event.path,
            "fireworks.blast",
            "queueing must not resolve or overwrite the immediately-played boundary sound"
        );

        for _ in 0..9 {
            assert!(client
                .audio
                .tick_repeats(&client.listener, &client.assets.resources)
                .is_empty());
        }
        assert_eq!(
            client.audio.delayed_sound_count(),
            1,
            "the ten-tick deadline is not nine ticks"
        );
        assert!(client
            .audio
            .tick_repeats(&client.listener, &client.assets.resources)
            .is_empty());
        assert_eq!(client.audio.delayed_sound_count(), 0);
        let delayed_record = client
            .audio
            .last_play()
            .expect("the far sound must resolve at its Java deadline");
        assert_eq!(delayed_record.request.event.path, "fireworks.blast_far");
        assert_eq!(
            delayed_record.request.position,
            Some(far),
            "the direct-coordinate firework caller must not gain a block-centre offset"
        );
    }

    /// A client over the real 1.8.9 assets, or `None` if this checkout has no jar.
    ///
    /// The real registry is the point: a synthetic one would prove the plumbing and nothing
    /// about whether `random.explode` is a `block` sound in the file the game actually ships.
    fn real_client() -> Option<(tempfile::TempDir, Client)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((dir, client))
    }

    /// A tiny registry/resource stack containing the ambient events used by ticker tests and the
    /// Guardian attack/flop events used by the no-device sound tests.
    /// The OGG bytes are intentionally empty: `AudioEngine::silent` resolves the event and
    /// records the request without touching a decoder or an output device.
    fn music_test_resources() -> (mc_assets::ResourceManager, SoundRegistry) {
        struct MemPack(HashMap<String, Vec<u8>>);

        impl mc_assets::ResourcePack for MemPack {
            fn open(&self, location: &mc_assets::ResourceLocation) -> Option<Box<dyn Read>> {
                self.0
                    .get(&location.pack_entry_name())
                    .cloned()
                    .map(|bytes| Box::new(Cursor::new(bytes)) as Box<dyn Read>)
            }

            fn contains(&self, location: &mc_assets::ResourceLocation) -> bool {
                self.0.contains_key(&location.pack_entry_name())
            }

            fn namespaces(&self) -> Vec<String> {
                vec!["minecraft".to_owned()]
            }
        }

        let sounds = br#"{
            "music.menu": {
                "category": "music",
                "sounds": [{"name": "music/menu", "stream": true}]
            },
            "music.game.end.credits": {
                "category": "music",
                "sounds": [{"name": "music/game/end/credits", "stream": true}]
            },
            "mob.guardian.attack": {
                "category": "hostile",
                "sounds": ["mob/guardian/attack_loop"]
            },
            "mob.guardian.flop": {
                "category": "hostile",
                "sounds": ["mob/guardian/flop"]
            }
        }"#;
        let mut resources = mc_assets::ResourceManager::new();
        resources.push_pack(Box::new(MemPack(HashMap::from([
            ("assets/minecraft/sounds.json".to_owned(), sounds.to_vec()),
            (
                "assets/minecraft/sounds/music/menu.ogg".to_owned(),
                Vec::new(),
            ),
            (
                "assets/minecraft/sounds/music/game/end/credits.ogg".to_owned(),
                Vec::new(),
            ),
            (
                "assets/minecraft/sounds/mob/guardian/attack_loop.ogg".to_owned(),
                Vec::new(),
            ),
            (
                "assets/minecraft/sounds/mob/guardian/flop.ogg".to_owned(),
                Vec::new(),
            ),
        ]))));
        let registry = SoundRegistry::load(&resources).expect("test sounds.json must parse");
        (resources, registry)
    }

    /// java: `GuardianSound.update`'s exact charge curve and `MovingSound` repeat fields, driven
    /// through `AudioEngine::silent` so the assertions never need a speaker or a decoder.
    #[test]
    fn guardian_sound_repeats_and_refreshes_exact_charge_values_without_a_device() {
        let (resources, registry) = music_test_resources();
        let mut engine = AudioEngine::<mc_audio::DefaultBackend>::silent(registry, VARIANT_SEED);
        let state = Arc::new(Mutex::new(GuardianSoundValues {
            position: DVec3::new(3.0, 64.0, -2.0),
            ticks: 40,
            max_ticks: 80,
            active: true,
        }));
        let initial = GuardianSound {
            state: Arc::clone(&state),
            first_update: true,
        };
        assert_eq!(
            initial.request(),
            PlayRequest {
                event: mc_assets::ResourceLocation::minecraft(GUARDIAN_ATTACK_SOUND),
                volume: 1.0,
                pitch: 1.0,
                position: Some(DVec3::ZERO),
                attenuation: mc_audio::AttenuationType::None,
                repeat: true,
                repeat_delay_ticks: 0,
            },
            "PositionedSound defaults are used on the immediate status-21 play"
        );
        engine
            .play_tickable(initial, &Listener::origin(), &resources)
            .expect("the silent engine still resolves the Guardian event");
        let record = engine.last_play().expect("status sound must be recorded");
        assert_eq!(record.outcome, PlayOutcome::NoBackend);
        assert_eq!(record.request.event.path, GUARDIAN_ATTACK_SOUND);
        assert!(record.request.repeat);
        assert_eq!(record.request.repeat_delay_ticks, 0);
        assert_eq!(
            record
                .computed
                .as_ref()
                .expect("hostile event resolves")
                .category,
            SoundCategory::Mobs
        );

        let mut updated = GuardianSound {
            state: Arc::clone(&state),
            first_update: true,
        };
        updated.update();
        assert!(!updated.is_done_playing());
        let request = updated.request();
        assert_eq!(request.position, Some(DVec3::new(3.0, 64.0, -2.0)));
        assert_eq!(request.volume, 0.25, "(40 / 80)^2");
        assert_eq!(request.pitch, 0.95, "0.7 + 0.5 * (40 / 80)");
        assert_eq!(request.attenuation, mc_audio::AttenuationType::None);

        {
            let mut values = state.lock().expect("test state lock");
            values.position = DVec3::new(-7.0, 65.5, 11.0);
            values.ticks = 30;
            values.max_ticks = 60;
        }
        let elder_request = updated.request();
        assert_eq!(elder_request.position, Some(DVec3::new(-7.0, 65.5, 11.0)));
        assert_eq!(elder_request.volume, 0.25, "(30 / 60)^2");
        assert_eq!(elder_request.pitch, 0.95, "0.7 + 0.5 * (30 / 60)");

        state.lock().expect("test state lock").active = false;
        assert!(
            updated.is_done_playing(),
            "target loss stops the moving source"
        );
    }

    /// `WorldClient.playSound(x, y, z, "mob.guardian.flop", 1, 1, false)` becomes one ordinary
    /// positional request. Resolve it through the silent engine so this verifies the registry,
    /// category, exact coordinates, and no-device outcome without requiring speakers.
    #[test]
    fn guardian_flop_sound_resolves_as_an_exact_positional_request_without_a_device() {
        let (resources, registry) = music_test_resources();
        let mut engine = AudioEngine::<mc_audio::DefaultBackend>::silent(registry, VARIANT_SEED);
        let request = PlayRequest::at(
            mc_assets::ResourceLocation::minecraft("mob.guardian.flop"),
            1.0,
            1.0,
            DVec3::new(-3.25, 1.5, 2.75),
        );
        engine
            .play(&request, &Listener::origin(), &resources)
            .expect("the silent engine still resolves the Guardian flop event");
        let record = engine.last_play().expect("flop sound must be recorded");
        assert_eq!(record.outcome, PlayOutcome::NoBackend);
        assert_eq!(record.request.event.path, "mob.guardian.flop");
        assert_eq!(record.request.position, request.position);
        assert_eq!(record.request.volume.to_bits(), 1.0_f32.to_bits());
        assert_eq!(record.request.pitch.to_bits(), 1.0_f32.to_bits());
        assert_eq!(
            record
                .computed
                .as_ref()
                .expect("hostile event resolves")
                .category,
            SoundCategory::Mobs
        );
    }

    /// java: `MovingSoundMinecart.update` uses horizontal motion only, keeps pitch at the
    /// constructor's `1.0F`, and fades a stationary cart to zero without stopping its loop.
    #[test]
    fn minecart_sound_keeps_lifecycle_and_rolling_volume_exact() {
        let mut sound = MinecartSound {
            state: Arc::new(Mutex::new(MinecartSoundValues {
                position: DVec3::new(2.0, 5.0, -3.0),
                motion: DVec3::new(0.3, 99.0, 0.4),
                distance: 0.0025,
                active: true,
            })),
            first_update: true,
        };
        assert_eq!(sound.request(), MinecartSound::initial_request());
        sound.update();
        let rolling = sound.request();
        assert_eq!(rolling.position, Some(DVec3::new(2.0, 5.0, -3.0)));
        assert_eq!(rolling.volume, 0.35, "sqrt(0.3^2 + 0.4^2) * 0.7");
        assert_eq!(rolling.pitch, 1.0);
        assert_eq!(rolling.attenuation, mc_audio::AttenuationType::Linear);
        assert!(rolling.repeat);

        {
            let mut values = sound.state.lock().expect("test state");
            values.motion = DVec3::new(0.0, -20.0, 0.0);
        }
        assert_eq!(
            sound.request().volume,
            0.0,
            "stationary carts are silent but retained"
        );
        assert!(!sound.is_done_playing());
        sound.state.lock().expect("test state").active = false;
        assert!(
            sound.is_done_playing(),
            "despawn/death ends the moving source"
        );
    }

    #[test]
    fn minecart_riding_sound_is_unattenuated_and_uses_the_one_block_cap() {
        let state = Arc::new(Mutex::new(MinecartSoundValues {
            position: DVec3::new(8.0, 4.0, 2.0),
            motion: DVec3::new(2.0, -10.0, 0.0),
            distance: 0.0,
            active: true,
        }));
        let mut sound = MinecartRidingSound {
            state,
            first_update: true,
        };
        assert_eq!(sound.request().event.path, MINECART_INSIDE_SOUND);
        assert_eq!(sound.request().position, None);
        assert_eq!(sound.request().attenuation, mc_audio::AttenuationType::None);
        sound.update();
        assert_eq!(
            sound.request().volume,
            0.75,
            "speed is capped at one block/tick"
        );
        assert_eq!(sound.request().pitch, 1.0);
        assert!(sound.request().repeat);
    }

    #[test]
    fn ambient_music_selector_matches_vanilla_priority() {
        assert_eq!(
            ambient_music_type_for(false, 0, true, true, true),
            MusicType::Menu,
            "without thePlayer vanilla stays on menu music"
        );
        assert_eq!(
            ambient_music_type_for(true, -1, true, true, true),
            MusicType::Nether,
            "Nether wins before creative and boss flags"
        );
        assert_eq!(
            ambient_music_type_for(true, 1, true, true, false),
            MusicType::End,
            "End wins before creative"
        );
        assert_eq!(
            ambient_music_type_for(true, 1, false, false, true),
            MusicType::EndBoss
        );
        assert_eq!(
            ambient_music_type_for(true, 0, true, true, false),
            MusicType::Creative
        );
        assert_eq!(
            ambient_music_type_for(true, 0, true, false, false),
            MusicType::Game,
            "creative without allowFlying is ordinary game music"
        );
        assert_eq!(
            ambient_music_type_for(true, 0, false, true, false),
            MusicType::Game
        );
    }

    /// F-P1-entity.boss-01-4: the live selector must consume the render-owned, expiring
    /// BossStatus bridge. A tracked dragon can outlive that status and must not keep boss music
    /// selected after the bar disappears.
    #[test]
    fn end_boss_music_gate_follows_active_boss_status_not_tracked_dragons() {
        let mut hud = mc_ui::screens::hud::Hud::new();
        assert!(!has_active_boss_status(&hud));

        hud.set_boss_status(200.0, 200.0, "Dragon", false);
        assert!(has_active_boss_status(&hud));

        hud.clear_boss_status();
        assert!(
            !has_active_boss_status(&hud),
            "an expired or cleared render status must release EndBoss music"
        );

        let production = include_str!("audio.rs")
            .split("mod tests {")
            .next()
            .expect("production audio module");
        let selector = production
            .split("fn ambient_music_type(&self) -> MusicType")
            .nth(1)
            .expect("live ambient selector")
            .split("fn tick_music")
            .next()
            .expect("selector body");
        assert!(
            selector.contains(
                "let end_boss = state.local.dimension == 1 && has_active_boss_status(&self.hud);"
            ),
            "the live selector must use the tested BossStatus predicate"
        );
        assert!(
            !selector.contains("EntityType::EnderDragon"),
            "tracked-dragon lifetime is not BossStatus lifetime"
        );
    }

    #[test]
    fn music_adapter_records_menu_and_forced_credits_without_audio_device() {
        let (resources, registry) = music_test_resources();
        let mut engine = AudioEngine::silent(registry, VARIANT_SEED);
        let mut ticker = mc_audio::MusicTicker::new(MUSIC_TICKER_SEED);
        let listener = Listener::origin();
        let mut current = None;
        let mut next_handle = 0;

        // MusicTicker starts with 100 and compares before decrementing, so menu music is
        // requested on tick 101. The adapter still resolves it through the engine in no-device
        // mode and leaves a diagnostic record even though there is no retained channel.
        for _ in 0..100 {
            let mut playback = ClientMusicPlayback {
                engine: &mut engine,
                listener,
                resources: &resources,
                current: &mut current,
                next_handle: &mut next_handle,
            };
            ticker.update(MusicType::Menu, &mut playback);
        }
        assert!(
            engine.last_play().is_none(),
            "the 101st tick is the first play"
        );
        {
            let mut playback = ClientMusicPlayback {
                engine: &mut engine,
                listener,
                resources: &resources,
                current: &mut current,
                next_handle: &mut next_handle,
            };
            ticker.update(MusicType::Menu, &mut playback);
        }
        let record = engine
            .last_play()
            .expect("menu request must reach the engine");
        assert_eq!(record.request.event.path, "music.menu");
        assert_eq!(record.request.volume, 1.0);
        assert_eq!(record.outcome, PlayOutcome::NoBackend);
        assert!(ticker.is_playing());
        assert!(
            current.is_none(),
            "no-device playback has no live backend handle"
        );

        // GuiWinGame's direct `func_181558_a(CREDITS)` path must also use the same adapter and
        // must not wait for the ordinary ticker delay.
        {
            let mut playback = ClientMusicPlayback {
                engine: &mut engine,
                listener,
                resources: &resources,
                current: &mut current,
                next_handle: &mut next_handle,
            };
            ticker.stop(&mut playback);
            ticker.start(MusicType::Credits, &mut playback);
        }
        let credits = engine
            .last_play()
            .expect("credits force-start must reach the engine");
        assert_eq!(credits.request.event.path, "music.game.end.credits");
        assert_eq!(credits.outcome, PlayOutcome::NoBackend);
    }

    #[test]
    fn client_tick_reaches_menu_music_ticker_without_audio_device() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.audio = AudioEngine::silent(client.assets.sounds.clone(), VARIANT_SEED);
        client.audio.clear_recent();

        for _ in 0..100 {
            client.tick(&mc_physics::state::MoveInput::default());
            assert!(
                client.audio.last_play().is_none(),
                "menu music must retain the vanilla 100-tick initial delay"
            );
        }
        client.tick(&mc_physics::state::MoveInput::default());
        let record = client
            .audio
            .last_play()
            .expect("Client::tick must drive mcMusicTicker on tick 101");
        assert_eq!(record.request.event.path, "music.menu");
        assert_eq!(record.request.volume, 1.0);
        assert_eq!(record.outcome, PlayOutcome::NoBackend);
        assert!(client.music_ticker.is_playing());
    }

    #[test]
    fn world_audio_reset_clears_retained_handles_without_resetting_ticker_without_device() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.audio = AudioEngine::silent(client.assets.sounds.clone(), VARIANT_SEED);
        client.audio.clear_recent();

        for _ in 0..101 {
            client.tick(&mc_physics::state::MoveInput::default());
        }
        assert!(client.music_ticker.is_playing());
        assert_eq!(client.audio.active_sound_count(), 0);
        assert_eq!(client.audio.recent().count(), 1);

        // `loadWorld` stops SoundManager channels but leaves MusicTicker.currentMusic intact;
        // this is why the next tick, rather than this reset, owns the cooldown draws.
        client.stop_audio_for_world_change();
        assert!(client.music_ticker.is_playing());
        assert!(client.music_playback.is_none());
        assert_eq!(client.audio.active_sound_count(), 0);
        assert_eq!(client.audio.delayed_repeat_count(), 0);
        assert!(client.audio.recent().next().is_none());
    }

    #[test]
    fn live_sound_registry_refresh_uses_reload_boundary_and_drops_client_handles() {
        let production = include_str!("audio.rs");
        let refresh = production
            .split("pub(crate) fn refresh_sound_registry(&mut self)")
            .nth(1)
            .expect("live sound refresh method");
        let refresh = refresh
            .split("// -------------------------------------------------------------------------------------")
            .next()
            .expect("refresh method body");

        assert!(
            refresh.contains("self.audio.reload_registry(self.assets.sounds.clone());"),
            "live pack reload must use the engine teardown/reopen boundary"
        );
        for retained in [
            "self.music_playback = None;",
            "self.record_sounds.clear();",
            "self.guardian_sounds.clear();",
            "self.minecart_sounds.clear();",
            "self.minecart_riding_sound = None;",
        ] {
            assert!(
                refresh.contains(retained),
                "live pack reload must drop {retained}"
            );
        }
        assert!(
            !refresh.contains("self.audio.registry = self.assets.sounds.clone();"),
            "live pack reload must not bypass engine teardown"
        );
    }

    // ---- The engine exists at all --------------------------------------------------

    /// The finding this closes was "`AudioEngine::new` has zero call sites outside `mc-audio`".
    /// This is the call site; it also pins the registry actually reaching the engine, because a
    /// client that built the engine from an *empty* registry would be just as silent and would
    /// pass any test that only checked the field existed.
    #[test]
    fn the_client_owns_an_engine_holding_the_real_sound_registry() {
        let Some((_dir, client)) = real_client() else {
            return;
        };
        assert!(
            client.assets.sounds.len() > 100,
            "the real sounds.json has hundreds of events; got {}",
            client.assets.sounds.len()
        );
        assert_eq!(
            client.audio.registry.len(),
            client.assets.sounds.len(),
            "the engine must hold the registry the asset load produced, not an empty one"
        );
        // Not an assertion — it depends on the machine. Printed so a run's own output says
        // which half of the claim it actually exercised, rather than the report having to.
        eprintln!(
            "audio backend: {}",
            if client.audio.is_silent() {
                "absent (no output device; every play resolves and stops there)"
            } else {
                "present (kira/cpal accepted a stream)"
            }
        );
    }

    // ---- The server's sound packets reach the engine --------------

    /// The end-to-end wire, driven through `Client::react` exactly as `Client::poll` drives it.
    /// Before this, `ClientEvent::SoundEffect` was created by `sync::dispatch` and matched
    /// nowhere in the workspace.
    #[test]
    fn a_server_sound_packet_reaches_the_engine_with_the_right_numbers() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        // java: `new S29PacketSoundEffect("random.explode", 12.5, 64.0, -30.25, 4.0, 1.0)`
        // encodes 100/512/-242 (× 8) and pitch byte 63 (× 63).
        let packet = s2c::SoundEffect {
            sound_name: "random.explode".to_owned(),
            x_fixed: 100,
            y_fixed: 512,
            z_fixed: -242,
            volume: 4.0,
            pitch_quantized: 63,
        };
        // Stand where the explosion is. Without this the listener is still at the origin (no
        // session has ever placed it), which is 72 blocks away — past even a volume-4.0 sound's
        // 64-block fade distance — and the sound would correctly resolve to nothing, proving
        // only that silence is silent.
        client.listener = listener_for(
            &PhysicsState::new_player(DVec3::new(12.5, 64.0 - 1.62, -30.25)),
            1.0,
        );
        client.react(&ClientEvent::SoundEffect(Box::new(packet)));

        let record = client
            .audio
            .last_play()
            .expect("the packet must have produced a play request");
        assert_eq!(record.request.event.namespace, "minecraft");
        assert_eq!(record.request.event.path, "random.explode");
        assert_eq!(record.request.volume, 4.0);
        assert!((record.request.pitch - 1.0).abs() < 1e-6);
        assert_eq!(
            record.request.position,
            Some(DVec3::new(12.5, 64.0, -30.25)),
            "the wire's × 8 fixed point must be divided back down"
        );

        // Resolution against the *real* registry: `random.explode` is a `block` sound in
        // vanilla's `sounds.json`, and its `.ogg` is one of four weighted variants.
        let computed = record
            .computed
            .as_ref()
            .expect("a real registry must resolve a real event");
        assert_eq!(computed.category, SoundCategory::Blocks);
        assert!(computed.file.path.starts_with("sounds/random/explode"));
        assert!(!computed.streaming, "a short effect is not streamed");
    }

    /// java: `"Unable to play unknown soundEvent: {}"` — a server naming a sound this client's
    /// packs do not carry must be dropped, not propagated and not fatal.
    #[test]
    fn an_unknown_sound_name_is_warned_about_and_dropped() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.react(&ClientEvent::SoundEffect(Box::new(s2c::SoundEffect {
            sound_name: "not.a.real.sound.event".to_owned(),
            x_fixed: 0,
            y_fixed: 0,
            z_fixed: 0,
            volume: 1.0,
            pitch_quantized: 63,
        })));
        let record = client.audio.last_play().expect("still recorded");
        assert_eq!(record.outcome, PlayOutcome::Unresolved);
        assert_eq!(record.request.event.path, "not.a.real.sound.event");
    }

    /// Vanilla's `ResourceLocation` accepts even an empty name. It reaches the live sound
    /// lookup as `minecraft:""`, which is unresolved rather than discarded as malformed.
    #[test]
    fn an_empty_sound_name_reaches_the_unresolved_live_lookup() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.react(&ClientEvent::SoundEffect(Box::new(s2c::SoundEffect {
            sound_name: String::new(),
            x_fixed: 0,
            y_fixed: 0,
            z_fixed: 0,
            volume: 1.0,
            pitch_quantized: 63,
        })));
        let record = client
            .audio
            .last_play()
            .expect("the permissive request reaches audio");
        assert_eq!(record.outcome, PlayOutcome::Unresolved);
        assert_eq!(
            record.request.event,
            mc_assets::ResourceLocation::minecraft("")
        );
    }

    /// The conversion itself, without the asset load — so it runs in every checkout, jar or not.
    #[test]
    fn wire_fields_convert_the_way_the_java_getters_do() {
        // java: `S29PacketSoundEffect.getX()` is `(double)((float)this.posX / 8.0F)`, and
        // `getPitch()` is `(float)this.pitch / 63.0F`.
        let request = sound_request(&s2c::SoundEffect {
            sound_name: "mob.zombie.say".to_owned(),
            x_fixed: -1,
            y_fixed: 3,
            z_fixed: 0,
            volume: 0.5,
            pitch_quantized: 32,
        })
        .expect("a bare name is valid");
        assert_eq!(
            request.event.namespace, "minecraft",
            "1.8.9 sends no prefix"
        );
        assert_eq!(request.position, Some(DVec3::new(-0.125, 0.375, 0.0)));
        assert_eq!(request.volume, 0.5);
        assert!((request.pitch - 32.0 / 63.0).abs() < 1e-6);
    }

    // ---- One owner of the nine sliders ---------------------------------------------

    /// The failure this guards against is the one the finding predicted verbatim: "the sliders
    /// will appear to work in the options screen and do nothing". Both directions are checked —
    /// a category the player turned down, and the master fader, which the engine treats
    /// differently (mixer track, not per-sound gain).
    #[test]
    fn options_txt_sliders_reach_the_engine() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.settings.sound_levels.insert("block".into(), 0.25);
        client.settings.sound_levels.insert("music".into(), 0.0);
        client.settings.sound_levels.insert("master".into(), 0.5);
        client.sync_sound_volumes();

        assert_eq!(client.audio.categories.get(SoundCategory::Blocks), 0.25);
        assert_eq!(client.audio.categories.get(SoundCategory::Music), 0.0);
        assert_eq!(client.audio.categories.get(SoundCategory::Master), 0.5);
        // java: `SoundManager.getSoundCategoryVolume` never applies MASTER per-sound.
        assert_eq!(
            client.audio.categories.effective(SoundCategory::Master),
            1.0
        );

        // And the whole way through: a quarter-volume block category must show up in the
        // resolved gain of a block sound played at the listener.
        client.react(&ClientEvent::SoundEffect(Box::new(s2c::SoundEffect {
            sound_name: "random.explode".to_owned(),
            x_fixed: 0,
            y_fixed: 0,
            z_fixed: 0,
            volume: 1.0,
            pitch_quantized: 63,
        })));
        let gain = client
            .audio
            .last_play()
            .and_then(|r| r.computed.as_ref())
            .expect("resolved")
            .linear_gain;
        assert!((gain - 0.25).abs() < 1e-6, "expected 0.25, got {gain}");
    }

    /// java: `SoundManager.playSound`'s first branch — master volume zero skips before the
    /// registry lookup, so a muted game does no Vorbis decoding at all.
    #[test]
    fn a_zero_master_volume_skips_before_the_engine_is_asked() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.settings.sound_levels.insert("master".into(), 0.0);
        client.sync_sound_volumes();
        client.react(&ClientEvent::SoundEffect(Box::new(s2c::SoundEffect {
            sound_name: "random.explode".to_owned(),
            x_fixed: 0,
            y_fixed: 0,
            z_fixed: 0,
            volume: 1.0,
            pitch_quantized: 63,
        })));
        assert!(client.audio.last_play().is_none());
    }

    // ---- the listener ----------------------------------------------------------------------

    /// java: `SoundManager.setListener` — the eye is `posY + getEyeHeight()`, and `at`/`up` come
    /// out of the same six-term expansion `mc_audio::Listener::interpolated` ports. The client's
    /// job is only to feed it the right player state, which is what this pins.
    #[test]
    fn the_listener_sits_at_the_players_eyes_facing_where_they_look() {
        let mut state = PhysicsState::new_player(DVec3::new(10.0, 64.0, -5.0));
        state.yaw = 0.0;
        state.pitch = 0.0;

        let listener = listener_for(&state, 1.0);
        assert_eq!(listener.position.x, 10.0);
        assert_eq!(listener.position.z, -5.0);
        // java: `EntityPlayer.getEyeHeight()` is 1.62 standing.
        assert!(
            (listener.position.y - (64.0 + 1.62)).abs() < 1e-6,
            "{}",
            listener.position.y
        );
        // yaw 0 faces +Z (south) in Minecraft.
        assert!(listener.at.z > 0.99, "{:?}", listener.at);

        // java: `getEyeHeight()` returns 0.08 less while sneaking, and `setListener` reads it
        // every frame rather than caching it.
        state.sneaking = true;
        let crouched = listener_for(&state, 1.0);
        assert!(
            (crouched.position.y - (64.0 + 1.62 - 0.08)).abs() < 1e-6,
            "{}",
            crouched.position.y
        );
    }

    #[test]
    fn the_listener_position_interpolates_from_the_tick_snapshot() {
        let mut state = PhysicsState::new_player(DVec3::new(10.0, 64.0, -5.0));
        state.yaw = 0.0;
        state.pitch = 0.0;
        let previous = DVec3::new(2.0, 60.0, -1.0);

        let listener = listener_for_with_previous(&state, Some(previous), None, 0.5);
        assert_eq!(
            listener.position,
            DVec3::new(6.0, 62.0 + f64::from(1.62_f32), -3.0)
        );
        assert_eq!(
            listener_for_with_previous(&state, Some(previous), None, 1.0).position,
            state.eye_pos(),
            "partial_ticks=1 reaches the current eye"
        );

        // Negative control: the old same-pose call ignores the snapshot, so this assertion bites
        // if update_listener regresses to `listener_for(&state, partial_ticks)`.
        assert_ne!(listener_for(&state, 0.5).position, listener.position);
    }

    #[test]
    fn the_listener_orientation_interpolates_from_the_tick_snapshot() {
        let mut state = PhysicsState::new_player(DVec3::new(10.0, 64.0, -5.0));
        state.yaw = 90.0;
        state.pitch = 30.0;

        let midpoint = listener_for_with_previous(&state, Some(state.pos), Some((0.0, 0.0)), 0.5);
        let expected = Listener::interpolated(
            state.pos,
            state.pos,
            state.eye_pos().y - state.pos.y,
            0.0,
            30.0,
            0.0,
            90.0,
            0.5,
        );
        assert_eq!(midpoint.at, expected.at);
        assert_eq!(midpoint.up, expected.up);

        let current_only = listener_for_with_previous(&state, Some(state.pos), None, 0.5);
        assert_ne!(
            midpoint.at, current_only.at,
            "current/current rotation must not masquerade as a midpoint"
        );
        assert_eq!(
            listener_for_with_previous(
                &state,
                Some(state.pos),
                Some((state.yaw, state.pitch)),
                0.5,
            ),
            current_only,
            "a stationary rotation remains exact"
        );
    }

    /// Distance attenuation is `mc-audio`'s, but it is measured from *this* listener, so a
    /// client that never moved the listener would play every sound as if the player stood at
    /// the world origin. That is what this checks: same packet, two listener positions, two
    /// different gains.
    #[test]
    fn moving_the_listener_changes_what_a_positional_sound_resolves_to() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let packet = s2c::SoundEffect {
            // 8 blocks out on X: half of the 16-block fade distance at volume 1.0.
            sound_name: "random.explode".to_owned(),
            x_fixed: 64,
            y_fixed: 0,
            z_fixed: 0,
            volume: 1.0,
            pitch_quantized: 63,
        };

        client.react(&ClientEvent::SoundEffect(Box::new(packet.clone())));
        let far = client
            .audio
            .last_play()
            .and_then(|r| r.computed.as_ref())
            .expect("resolved")
            .linear_gain;
        assert!((far - 0.5).abs() < 1e-6, "expected 0.5, got {far}");

        // The eye, not the feet: `listener_for` adds `getEyeHeight()`, so put the feet 1.62
        // below the sound to land the ears exactly on it.
        let state = PhysicsState::new_player(DVec3::new(8.0, -1.62, 0.0));
        client.listener = listener_for(&state, 1.0);
        client.react(&ClientEvent::SoundEffect(Box::new(packet)));
        let near = client
            .audio
            .last_play()
            .and_then(|r| r.computed.as_ref())
            .expect("resolved")
            .linear_gain;
        assert!((near - 1.0).abs() < 1e-6, "expected 1.0, got {near}");
    }

    // ---- The GUI click -------------------------------------------------------------

    /// java: `GuiButton.playPressSound` — `PositionedSoundRecord.create(loc, 1.0F)`, which is
    /// the **pitch** overload: volume `0.25`, `AttenuationType.NONE`.
    ///
    /// This assertion used to read `1.0`, mistaking the pitch argument for the volume — the
    /// "one test" of `F-P2-client.audio-01-10`, and the reason `F-P2-client.audio-02-1` survived.
    #[test]
    fn the_button_click_is_the_sound_vanilla_plays() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.play_button_press();
        let record = client.audio.last_play().expect("recorded");
        assert_eq!(record.request.event.path, BUTTON_PRESS);
        assert_eq!(record.request.position, None, "GUI sounds do not attenuate");
        assert_eq!(record.request.volume, BUTTON_PRESS_VOLUME);
        assert_eq!(record.request.pitch, 1.0);
        assert_eq!(
            record
                .computed
                .as_ref()
                .expect("gui.button.press is a real vanilla event")
                .category,
            SoundCategory::Master,
            "vanilla files gui.button.press under the default category"
        );
    }

    /// F-P2-client.audio-02-11: GuiOptions button 8675309 replaces the ordinary click with one
    /// eligible category event at the `PositionedSoundRecord.create(location, 0.5F)` pitch.
    #[test]
    fn the_super_secret_action_plays_one_eligible_half_pitch_sound() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.audio.clear_recent();
        client.handle_action(mc_ui::screens::Action::PressSuperSecretSettings);

        let plays: Vec<_> = client.audio.recent().collect();
        assert_eq!(
            plays.len(),
            1,
            "the custom press replaces the default GUI click"
        );
        let play = plays[0];
        assert_ne!(play.request.event.path, BUTTON_PRESS);
        assert_eq!(play.request.position, None);
        assert_eq!(play.request.volume, BUTTON_PRESS_VOLUME);
        assert_eq!(play.request.pitch, 0.5);
        let category = play
            .computed
            .as_ref()
            .expect("a registry-selected event resolves")
            .category;
        assert!(matches!(
            category,
            SoundCategory::Animals
                | SoundCategory::Blocks
                | SoundCategory::Mobs
                | SoundCategory::Players
                | SoundCategory::Weather
        ));
    }

    /// The wire, not the handler. `play_button_press` existing and being correct is exactly the
    /// kind of thing this whole wave is repairing — the question that matters is whether a real
    /// click reaches it. This drives the same path `tests/input_dispatch.rs` uses to prove the
    /// main menu's buttons are clickable at all: real `InputState`, real cursor mapping, real
    /// `Client::handle_screen_input`.
    ///
    /// Needs a GPU because the screen has to be laid out (`Client::frame`) before it can be
    /// hit-tested.
    #[test]
    fn clicking_a_real_menu_button_plays_the_click() {
        // java: `Minecraft`'s default window. A 64x64 framebuffer is too small for the main
        // menu's buttons to land where `GuiMainMenu.initGui` puts them.
        const WIDTH: u32 = 854;
        const HEIGHT: u32 = 480;
        let Ok(gpu) = mc_render::Gpu::headless(
            WIDTH,
            HEIGHT,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) else {
            eprintln!("skip: no wgpu adapter available on this machine");
            return;
        };
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        // vanilla's button at vanilla's coordinates: the vanilla title screen
        // (the Aero one, on by default, has its own click-sound test in `app/mods/aero_title.rs`).
        client.mods.settings.set_enabled("aero_title", false);
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, WIDTH, HEIGHT);
        let mut camera = crate::app::driver::menu_camera();
        client.frame(&gpu, &mut camera, 0.0);

        // java: `GuiMainMenu.initGui` — `j = height / 4 + 48`, Multiplayer at `j + 24`, 200×20.
        let resolution = client.resolution(WIDTH, HEIGHT);
        let gx = resolution.scaled_width() / 2;
        let gy = resolution.scaled_height() / 4 + 48 + 24 + 10;
        let px = f64::from(gx) * f64::from(WIDTH) / f64::from(resolution.scaled_width());
        let py = f64::from(gy) * f64::from(HEIGHT) / f64::from(resolution.scaled_height());

        // A click on nothing must stay silent, or the assertion below proves only that *some*
        // click makes a noise.
        let mut input = mc_render::InputState::new();
        input.on_cursor_moved(2.0, 2.0);
        input.on_mouse_button(winit::event::MouseButton::Left, true);
        client.handle_screen_input(&mut input, WIDTH, HEIGHT);
        assert!(
            client.audio.last_play().is_none(),
            "a click that hits no button must not play gui.button.press"
        );

        input.on_mouse_button(winit::event::MouseButton::Left, false);
        client.handle_screen_input(&mut input, WIDTH, HEIGHT);
        input.on_cursor_moved(px, py);
        input.on_mouse_button(winit::event::MouseButton::Left, true);
        client.handle_screen_input(&mut input, WIDTH, HEIGHT);

        assert_eq!(
            client.ui.name(),
            "multiplayer",
            "the click must have landed on Multiplayer for this to test anything"
        );
        assert_eq!(
            client
                .audio
                .last_play()
                .expect("pressing a button must play a sound")
                .request
                .event
                .path,
            BUTTON_PRESS
        );
        eprintln!("DIAG: assertions passed, test body complete");
        drop(client);
        eprintln!("DIAG: client dropped");
        drop(gpu);
        eprintln!("DIAG: gpu dropped");
    }

    /// Regression: a [`Client`] that has rendered a frame must still be droppable.
    ///
    /// [`Client::frame`] lazily starts the skin-download worker. While that
    /// worker was a `tokio` **blocking** task parked in a channel `recv()`, dropping the client
    /// deadlocked: tokio does not cancel blocking tasks at shutdown, it waits for them, and
    /// `Client` drops its `runtime` field before the sender whose drop would end the loop.
    ///
    /// This is written with an explicit timeout because the failure mode is a *hang*, not a
    /// panic — the four lib tests it originally broke all printed their assertions and passed,
    /// then wedged the whole suite on teardown, which is invisible to an ordinary `#[test]`.
    /// A bounded wait turns "never returns" into a failing assertion.
    #[test]
    fn a_client_that_has_rendered_a_frame_still_drops() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let Ok(gpu) = mc_render::Gpu::headless(
                854,
                480,
                mc_render::GpuOptions {
                    vsync: false,
                    sample_count: 1,
                    low_power: false,
                },
            ) else {
                let _ = tx.send("skip: no wgpu adapter");
                return;
            };
            let Some((_dir, mut client)) = real_client() else {
                let _ = tx.send("skip: no vanilla jar");
                return;
            };
            let mut phases = mc_render::PhaseList::new();
            client.install(&gpu, &mut phases, 854, 480);
            let mut camera = crate::app::driver::menu_camera();
            client.frame(&gpu, &mut camera, 0.0);
            drop(client);
            let _ = tx.send("dropped");
        });
        let outcome = rx
            .recv_timeout(std::time::Duration::from_secs(120))
            .expect("dropping a Client that has rendered a frame must not hang");
        eprintln!("{outcome}");
    }

    // ---- S28: the playAuxSFX / broadcastSound / playRecord channel -------------------------

    /// Stands the listener at the block so nothing attenuates to silence, and drives one
    /// `S28PacketEffect` through `Client::react` exactly as `Client::poll` would.
    fn fire_effect(client: &mut Client, id: i32, pos: mc_math::BlockPos, data: i32) {
        client.listener = listener_for(
            &PhysicsState::new_player(DVec3::new(
                f64::from(pos.x) + 0.5,
                f64::from(pos.y) + 0.5 - 1.62,
                f64::from(pos.z) + 0.5,
            )),
            1.0,
        );
        client.react(&ClientEvent::Effect(s2c::Effect {
            effect_id: id,
            position: pos,
            data,
            disable_relative_volume: false,
        }));
    }

    /// The headline of `F-P1-world-03-1` / `F-P0-network.play.server-10-2` / `F-P1-block-09-5`:
    /// `ClientEvent::Effect` was matched by nothing, so a door opening made no sound.
    ///
    /// What it would take for this to pass while the code is wrong: `last_play` would have to be
    /// left over from an earlier sound in the same `Client`. It cannot be — `clear_recent` runs
    /// first, and the event name and volume are both asserted.
    #[test]
    fn a_door_effect_packet_plays_random_door_open_at_the_block_centre() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.audio.clear_recent();
        let pos = mc_math::BlockPos::new(10, 64, -3);
        fire_effect(&mut client, 1003, pos, 0);

        let record = client
            .audio
            .last_play()
            .expect("effect 1003 must have formed a play request");
        assert_eq!(record.request.event.path, "random.door_open");
        assert_eq!(record.request.volume, 1.0);
        // java: `playSoundAtPos` adds 0.5 on every axis.
        assert_eq!(
            record.request.position,
            Some(DVec3::new(10.5, 64.5, -2.5)),
            "playSoundAtPos emits from the block centre"
        );
        // java: `this.theWorld.rand.nextFloat() * 0.1F + 0.9F`.
        assert!(
            (0.9..1.0).contains(&record.request.pitch),
            "pitch {} outside rand*0.1+0.9",
            record.request.pitch
        );
        // The registry has to actually know the event, or the fix is a name-only guess.
        assert!(
            record.computed.is_some(),
            "random.door_open must resolve against the real sounds.json"
        );
    }

    /// Every plain-sound case of the table, checked as one sweep: name, volume, and — for the
    /// eleven that share it — that the pitch landed inside `(r - r) * 0.2 + 1.0`.
    ///
    /// The negative control is `1019`, an id vanilla's `switch` has no case for: it must produce
    /// nothing, so a `play_aux_sfx` that played some default for everything would fail here.
    #[test]
    fn every_positional_aux_sfx_case_resolves_against_the_real_sound_registry() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let pos = mc_math::BlockPos::new(0, 70, 0);
        let cases: &[(i32, &str, f32)] = &[
            (1000, "random.click", 1.0),
            (1001, "random.click", 1.0),
            (1002, "random.bow", 1.0),
            (1003, "random.door_open", 1.0),
            (1004, "random.fizz", 0.5),
            (1006, "random.door_close", 1.0),
            (1007, "mob.ghast.charge", 10.0),
            (1008, "mob.ghast.fireball", 10.0),
            (1009, "mob.ghast.fireball", 2.0),
            (1010, "mob.zombie.wood", 2.0),
            (1011, "mob.zombie.metal", 2.0),
            (1012, "mob.zombie.woodbreak", 2.0),
            (1014, "mob.wither.shoot", 2.0),
            (1015, "mob.bat.takeoff", 0.05),
            (1016, "mob.zombie.infect", 2.0),
            (1017, "mob.zombie.unfect", 2.0),
            (1020, "random.anvil_break", 1.0),
            (1021, "random.anvil_use", 1.0),
            (1022, "random.anvil_land", 0.3),
        ];
        for &(id, name, volume) in cases {
            client.audio.clear_recent();
            fire_effect(&mut client, id, pos, 0);
            let record = client
                .audio
                .last_play()
                .unwrap_or_else(|| panic!("effect {id} produced no play request"));
            assert_eq!(record.request.event.path, name, "effect {id}");
            assert_eq!(record.request.volume, volume, "effect {id} volume");
            assert!(
                record.computed.is_some(),
                "effect {id} named {name}, which the real sounds.json does not carry"
            );
        }

        // Negative control: no `case 1019:` exists in `RenderGlobal.playAuxSFX`.
        client.audio.clear_recent();
        fire_effect(&mut client, 1019, pos, 0);
        assert!(
            client.audio.last_play().is_none(),
            "an id with no case must be silently ignored"
        );
    }

    /// `F-P2-client.renderer-11-2` / `F-P1-block-09-5`: effect 2001, another player breaking a
    /// block — its `dig.*` sound and its 64-particle burst.
    ///
    /// Also pins the two `Block.SoundType` numbers the sound is derived from, which is what
    /// separates this from "some sound played": vanilla uses `(volume + 1) / 2` and
    /// `frequency * 0.8`, so stone comes out at `1.0` and `0.8` rather than `1.0`/`1.0`.
    #[test]
    fn effect_2001_plays_the_block_break_sound_and_spawns_64_debris_particles() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.audio.clear_recent();
        let before = client.particles.len();
        let stone = mc_data::block_by_name("stone").expect("stone");
        let state = mc_data::BlockState::new(stone, 0);
        fire_effect(
            &mut client,
            2001,
            mc_math::BlockPos::new(4, 65, 4),
            state.to_java_state_id(),
        );

        let record = client.audio.last_play().expect("2001 must play a sound");
        assert_eq!(record.request.event.path, "dig.stone");
        assert_eq!(record.request.volume, 1.0, "(1.0 + 1.0) / 2.0");
        assert!(
            (record.request.pitch - 0.8).abs() < 1e-6,
            "frequency * 0.8, got {}",
            record.request.pitch
        );
        assert!(record.computed.is_some(), "dig.stone must resolve");

        assert_eq!(
            client.particles.len() - before,
            0,
            "no session yet, so there is no world to spawn into — vanilla returns here too"
        );

        // Now with a world behind it. `add_block_destroy_effects` needs only
        // `Client::session`'s world, not a predictor, so a bare offline session is enough.
        let (_to_game, _to_net) = attach_offline_session(&mut client);
        let before = client.particles.len();
        client.particles.clear();
        fire_effect(
            &mut client,
            2001,
            mc_math::BlockPos::new(4, 65, 4),
            state.to_java_state_id(),
        );
        assert_eq!(
            client.particles.len(),
            64,
            "addBlockDestroyEffects is 4x4x4 (was {before} before the world existed)"
        );
    }

    /// Gives `client` a socket-less [`Session`] holding a blank world. The two channel ends come
    /// back because dropping them makes every `send` report `NotConnected`.
    #[allow(clippy::type_complexity)]
    fn attach_offline_session(
        client: &mut Client,
    ) -> (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session =
            crate::session::Session::new(handle, crate::session::JoinSettings::default());
        // `blank_world` is all air, so put one real block under the effects that read the world.
        let mut world = mc_render::terrain::fixture::blank_world(1);
        mc_render::terrain::fixture::place(&mut world, mc_math::BlockPos::new(3, 10, 3), "grass");
        session.state_mut().world = world;
        client.session = Some(session);
        (to_game, to_net)
    }

    /// The bonemeal case, 2005 — the one `playAuxSFX` entry whose particle count comes from the
    /// packet (`if (amount == 0) amount = 15`). `F-P1-item-06-10`.
    #[test]
    fn effect_2005_spawns_the_bonemeal_particles_with_vanillas_zero_means_fifteen_rule() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (_g, _n) = attach_offline_session(&mut client);
        // `attach_offline_session` places one grass block here; everything else is air.
        let pos = mc_math::BlockPos::new(3, 10, 3);
        for (data, expected) in [(0, 15), (7, 7), (30, 30)] {
            client.particles.clear();
            fire_effect(&mut client, 2005, pos, data);
            assert_eq!(
                client.particles.len(),
                expected,
                "data {data} must produce {expected} VILLAGER_HAPPY particles"
            );
        }
        // Negative control: over air there is no block to bonemeal, and java returns early.
        client.particles.clear();
        fire_effect(&mut client, 2005, mc_math::BlockPos::new(3, 200, 3), 0);
        assert_eq!(client.particles.len(), 0, "air must produce nothing");
    }

    /// The three purely-particle cases with fixed counts, driven end to end through `react`.
    /// java: `playAuxSFX` 2000 (ten dispenser puffs), 2003 (eight shards + 2*40 portal motes),
    /// 2004 (twenty smoke/flame pairs).
    #[test]
    fn the_fixed_count_particle_effects_spawn_exactly_vanillas_counts() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (_g, _n) = attach_offline_session(&mut client);
        let pos = mc_math::BlockPos::new(3, 10, 3);

        client.particles.clear();
        fire_effect(&mut client, 2000, pos, 0);
        assert_eq!(client.particles.len(), 10, "2000: ten SMOKE_NORMAL");

        // 2003 spawns 8 ITEM_CRACK, which need the ender-eye icon and are dropped without a
        // bake, plus 2 PORTAL per 0.157 rad step over 2pi = 40 steps.
        client.particles.clear();
        fire_effect(&mut client, 2003, pos, 0);
        let portals = client
            .particles
            .particles()
            .iter()
            .filter(|p| matches!(p.kind, mc_render::particle::ParticleKind::Portal { .. }))
            .count();
        assert_eq!(portals, 80, "2003: two counter-rotating rings of forty");

        client.particles.clear();
        fire_effect(&mut client, 2004, pos, 0);
        assert_eq!(
            client.particles.len(),
            40,
            "2004: twenty smoke + twenty flame"
        );
    }

    /// `RenderGlobal.playAuxSFX` case 2000 decodes the dispenser facing with Java `int` math.
    /// In particular, `%` and `/` truncate toward zero; Euclidean operations would rotate
    /// negative metadata into a different pair of axes and make every puff origin/velocity
    /// wrong.
    #[test]
    fn effect_2000_dispenser_direction_keeps_java_signed_remainder_and_division() {
        let normal = [
            (0, (-1, -1)),
            (1, (0, -1)),
            (2, (1, -1)),
            (3, (-1, 0)),
            (5, (1, 0)),
            (8, (1, 1)),
        ];
        for (data, expected) in normal {
            assert_eq!(
                Client::dispenser_smoke_direction(data),
                expected,
                "data {data}"
            );
        }

        // Negative controls distinguish Java's truncation from rem_euclid/div_euclid:
        // -1 / 3 is 0 in Java/Rust signed integer arithmetic, while floor division is -1.
        for (data, expected) in [(-1, (-2, -1)), (-2, (-3, -1)), (-4, (-2, -2))] {
            assert_eq!(
                Client::dispenser_smoke_direction(data),
                expected,
                "negative data {data}"
            );
        }
    }

    /// java: `playAuxSFX` 2002 resolves both the brewed liquid colour and the spell factory from
    /// the packet's potion metadata. Healing is instant; speed is not. The live particle pass
    /// must preserve those choices through its per-particle brightness jitter.
    #[test]
    fn effect_2002_uses_brewed_color_and_instant_kind_then_plays_the_smash_sound() {
        const SPLASH_HEAL: i32 = 24_581;
        const SPLASH_SPEED: i32 = 24_578;
        let cases = [
            (
                SPLASH_HEAL,
                particle_ids::SPELL_INSTANT,
                [248.0 / 255.0, 36.0 / 255.0, 35.0 / 255.0],
                144,
            ),
            (
                SPLASH_SPEED,
                particle_ids::SPELL,
                [124.0 / 255.0, 175.0 / 255.0, 198.0 / 255.0],
                128,
            ),
        ];
        for (data, expected_kind, expected_color, _) in cases {
            assert_eq!(
                potion_smash_visuals(data),
                (expected_kind, expected_color),
                "metadata {data} must resolve its exact vanilla colour and spell family"
            );
        }

        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (_g, _n) = attach_offline_session(&mut client);
        let pos = mc_math::BlockPos::new(3, 10, 3);
        for (data, _expected_kind, expected_color, expected_base_tex) in cases {
            client.particles.clear();
            client.audio.clear_recent();
            fire_effect(&mut client, 2002, pos, data);

            let spells: Vec<_> = client
                .particles
                .particles()
                .iter()
                .filter(|particle| {
                    matches!(
                        particle.kind,
                        mc_render::particle::ParticleKind::Spell { .. }
                    )
                })
                .collect();
            assert_eq!(spells.len(), 100, "2002 must emit one hundred swirls");
            assert!(spells.iter().all(|particle| {
                matches!(
                    particle.kind,
                    mc_render::particle::ParticleKind::Spell { base_tex }
                        if base_tex == expected_base_tex
                )
            }));
            for particle in spells {
                let brightness = particle.color[0] / expected_color[0];
                for (channel, expected_channel) in expected_color.iter().enumerate() {
                    assert!(
                        (particle.color[channel] - expected_channel * brightness).abs() < 1.0e-6,
                        "metadata {data} swirl colour must retain the brewed RGB ratio"
                    );
                }
            }
            assert_eq!(
                client
                    .particles
                    .particles()
                    .iter()
                    .filter(|particle| matches!(
                        particle.kind,
                        mc_render::particle::ParticleKind::Breaking
                    ))
                    .count(),
                8,
                "2002 must emit eight potion-item shards"
            );
            assert_eq!(
                client
                    .audio
                    .last_play()
                    .expect("potion smash sound")
                    .request
                    .event
                    .path,
                "game.potion.smash"
            );
        }

        client.particles.clear();
        client.audio.clear_recent();
        fire_effect(&mut client, 1019, pos, 0);
        assert!(client.particles.particles().is_empty());
        assert!(client.audio.last_play().is_none());
    }

    /// Extracted virtual `Block.SoundType.getBreakSound()` values resolve in the shipped sound
    /// registry. Glass is the negative control: it shares stone's step family but breaks as
    /// `dig.glass`.
    #[test]
    fn break_sound_overrides_match_block_java_and_resolve_in_the_registry() {
        let Some((_dir, client)) = real_client() else {
            return;
        };
        let cases: &[(&str, &str)] = &[
            ("stone", "dig.stone"),
            ("planks", "dig.wood"),
            ("dirt", "dig.gravel"),
            // java: `soundTypeGlass.getBreakSound()` — `soundName` is "stone", so only the block
            // id distinguishes it.
            ("glass", "dig.glass"),
            ("ice", "dig.glass"),
            ("glowstone", "dig.glass"),
            // java: `soundTypeLadder.getBreakSound()` — "dig.wood", not "dig.ladder".
            ("ladder", "dig.wood"),
            // java: `soundTypeAnvil.getBreakSound()` — "dig.stone", not "dig.anvil".
            ("anvil", "dig.stone"),
            // java: `SLIME_SOUND.getBreakSound()` — "mob.slime.big", not "dig.slime".
            ("slime", "mob.slime.big"),
        ];
        for &(block, expected) in cases {
            let id = mc_data::block_by_name(block).unwrap_or_else(|| panic!("{block} exists"));
            let got = mc_data::block_info(id).break_sound;
            assert_eq!(got, expected, "{block}");
            assert!(
                client
                    .assets
                    .sounds
                    .contains(&mc_assets::ResourceLocation::minecraft(got)),
                "{got} is not an event in the real sounds.json"
            );
        }
    }

    /// `F-P1-item-11-10`: the jukebox. Three things, all previously absent — the track, the
    /// "Now playing" HUD line, and the handle that lets the next disc stop this one.
    #[test]
    fn effect_1005_starts_a_record_and_shows_the_now_playing_line() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.audio.clear_recent();
        let pos = mc_math::BlockPos::new(2, 66, 2);
        let disc = mc_data::item_by_name("minecraft:record_13").expect("record_13");
        fire_effect(&mut client, 1005, pos, i32::from(disc.0));

        let record = client.audio.last_play().expect("1005 must play a record");
        assert_eq!(record.request.event.path, "records.13");
        // java: `PositionedSoundRecord.create(loc, (float)x, (float)y, (float)z)` — volume 4.0,
        // and the raw block coordinate with no `+ 0.5`.
        assert_eq!(record.request.volume, 4.0);
        assert_eq!(record.request.position, Some(DVec3::new(2.0, 66.0, 2.0)));
        let computed = record
            .computed
            .as_ref()
            .expect("records.13 must resolve against the real sounds.json");
        assert!(
            computed.streaming,
            "a music disc is a streaming sound, not a buffered effect"
        );
        // java: `ingameGUI.setRecordPlayingMessage(itemrecord.getRecordNameLocal())`, which is
        // `setRecordPlaying(I18n.format("record.nowPlaying", name), true)`.
        assert_eq!(client.hud.record_playing, "Now playing: C418 - 13");
        assert!(client.hud.record_is_playing);
        assert_eq!(client.hud.record_ticks, 60);

        // java: `playAuxSFX` 1005 with a non-record `data` reaches `playRecord(pos, null)`,
        // which stops the track and shows nothing.
        client.audio.clear_recent();
        client.hud.record_playing = String::new();
        fire_effect(&mut client, 1005, pos, 1); // stone, not a disc
        assert!(
            client.audio.last_play().is_none(),
            "ejecting a disc must not start a sound"
        );
        assert!(client.hud.record_playing.is_empty());
        assert!(!client.record_sounds.contains_key(&pos));

        for data in [0, i32::MAX] {
            client.audio.clear_recent();
            fire_effect(&mut client, 1005, pos, data);
            assert!(
                client.audio.last_play().is_none(),
                "non-record item id {data} must be an eject-only event"
            );
            assert!(!client.record_sounds.contains_key(&pos));
        }
    }

    /// `F-P1-block-19-3`: exercise the exact position-map helpers with Kira's CPU-only mock
    /// backend. A silent engine cannot return handles, so request-history assertions alone could
    /// never prove that replacement/eject/world reset actually stop retained streaming tracks.
    #[test]
    fn jukebox_replacement_eject_and_world_reset_stop_the_retained_backend_handle() {
        use mc_audio::MockBackend;

        let Some((_dir, client)) = real_client() else {
            return;
        };
        let pos = mc_math::BlockPos::new(2, 2, 2);
        let disc = |name: &str| {
            i32::from(
                mc_data::item_by_name(&format!("minecraft:record_{name}"))
                    .unwrap_or_else(|| panic!("record_{name}"))
                    .0,
            )
        };
        let mut audio = AudioEngine::<MockBackend>::new(client.assets.sounds.clone(), VARIANT_SEED);
        let mut retained = HashMap::new();

        assert_eq!(
            drive_record_effect_audio(
                &mut audio,
                &mut retained,
                &client.listener,
                &client.assets.resources,
                pos,
                disc("13"),
            )
            .expect("records.13 decodes"),
            Some("13")
        );
        let first = retained.get(&pos).expect("first handle retained").clone();
        audio
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();

        // Same-position replacement cuts the old stream before retaining the new one.
        assert_eq!(
            drive_record_effect_audio(
                &mut audio,
                &mut retained,
                &client.listener,
                &client.assets.resources,
                pos,
                disc("cat"),
            )
            .expect("records.cat decodes"),
            Some("cat")
        );
        let second = retained.get(&pos).expect("replacement retained").clone();
        audio
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        audio.backend_mut().expect("MockBackend starts").process();
        assert!(!audio.is_sound_playing(&first));
        assert!(audio.is_sound_playing(&second));
        assert_eq!(retained.len(), 1);

        // Data 0, an ordinary item and an unknown id all stop with no replacement.
        assert_eq!(
            drive_record_effect_audio(
                &mut audio,
                &mut retained,
                &client.listener,
                &client.assets.resources,
                pos,
                0,
            )
            .expect("data-zero eject"),
            None
        );
        audio
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        audio.backend_mut().expect("MockBackend starts").process();
        assert!(!audio.is_sound_playing(&second));
        assert!(retained.is_empty());

        for invalid in [1, i32::MAX] {
            drive_record_effect_audio(
                &mut audio,
                &mut retained,
                &client.listener,
                &client.assets.resources,
                pos,
                disc("blocks"),
            )
            .expect("record before invalid eject");
            let old = retained.get(&pos).expect("record retained").clone();
            assert_eq!(
                drive_record_effect_audio(
                    &mut audio,
                    &mut retained,
                    &client.listener,
                    &client.assets.resources,
                    pos,
                    invalid,
                )
                .expect("invalid eject"),
                None
            );
            audio
                .backend_mut()
                .expect("MockBackend starts")
                .on_start_processing();
            audio.backend_mut().expect("MockBackend starts").process();
            assert!(!audio.is_sound_playing(&old));
            assert!(retained.is_empty());
        }

        drive_record_effect_audio(
            &mut audio,
            &mut retained,
            &client.listener,
            &client.assets.resources,
            pos,
            disc("blocks"),
        )
        .expect("world-owned record");
        let third = retained.get(&pos).expect("world handle retained").clone();
        stop_world_audio(&mut audio, &mut retained);
        audio
            .backend_mut()
            .expect("MockBackend starts")
            .on_start_processing();
        audio.backend_mut().expect("MockBackend starts").process();
        assert!(!audio.is_sound_playing(&third));
        assert!(retained.is_empty());
        assert_eq!(audio.active_sound_count(), 0);

        let source = include_str!("audio.rs");
        let play = source
            .split("fn play_record_effect(")
            .nth(1)
            .and_then(|tail| tail.split("fn sfx_dispenser_smoke(").next())
            .expect("play_record_effect source slice");
        assert!(play.contains("drive_record_effect_audio("));
        let reset = source
            .split("fn stop_audio_for_world_change(")
            .nth(1)
            .and_then(|tail| tail.split("pub(crate) fn play_sound(").next())
            .expect("world-audio reset source slice");
        assert!(reset.contains("stop_world_audio("));
    }

    /// `F-P0-network.play.server-10-2`'s other half: `serverWide` routes to `broadcastSound`,
    /// which plays two blocks from the *listener* toward the source so no distance can mute it.
    #[test]
    fn a_server_wide_effect_plays_two_blocks_from_the_listener() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let Some(feet) = client
            .session
            .as_ref()
            .and_then(|s| s.predictor())
            .map(|p| p.state.pos)
        else {
            eprintln!(
                "skip: no session, so broadcastSound has no view entity (vanilla returns too)"
            );
            return;
        };
        client.audio.clear_recent();
        client.react(&ClientEvent::Effect(s2c::Effect {
            effect_id: 1013,
            position: mc_math::BlockPos::new(feet.x as i32 + 400, feet.y as i32, feet.z as i32),
            data: 0,
            disable_relative_volume: true,
        }));
        let record = client.audio.last_play().expect("1013 must play");
        assert_eq!(record.request.event.path, "mob.wither.spawn");
        let at = record.request.position.expect("positional");
        assert!(
            (at - feet).length() < 2.001,
            "vanilla moves the source to 2 blocks from the listener, got {}",
            (at - feet).length()
        );
    }

    /// `F-P2-client.audio-02-1`: `PositionedSoundRecord.create(loc, pitch)` is the **0.25**
    /// volume overload. The button click was four times too loud.
    #[test]
    fn the_gui_click_plays_at_a_quarter_volume() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        client.audio.clear_recent();
        client.play_button_press();
        let record = client.audio.last_play().expect("a click must be requested");
        assert_eq!(record.request.event.path, BUTTON_PRESS);
        assert_eq!(record.request.volume, 0.25);
        assert_eq!(record.request.pitch, 1.0);
        assert_eq!(record.request.position, None, "AttenuationType.NONE");
    }

    /// `record_name_for_item` against the whole disc range plus two negative controls.
    #[test]
    fn only_the_twelve_music_discs_resolve_to_a_record_name() {
        let expected = [
            (2256, "13"),
            (2257, "cat"),
            (2258, "blocks"),
            (2259, "chirp"),
            (2260, "far"),
            (2261, "mall"),
            (2262, "mellohi"),
            (2263, "stal"),
            (2264, "strad"),
            (2265, "ward"),
            (2266, "11"),
            (2267, "wait"),
        ];
        for (id, name) in expected {
            assert_eq!(record_name_for_item(id), Some(name), "item {id}");
        }
        assert_eq!(record_name_for_item(1), None, "stone is not a disc");
        assert_eq!(record_name_for_item(2255), None, "just below the range");
        assert_eq!(record_name_for_item(2268), None, "just above the range");
        assert_eq!(record_name_for_item(-1), None);
    }
}
