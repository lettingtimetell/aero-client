//! Replay checkpoints, so a seek anywhere in a long replay is
//! fast.
//!
//! # How it works
//!
//! Before this, a backward seek threw the replayed world away and re-applied every record from
//! the start of the file (the rebuild): about ten seconds at the end of an hour-long replay
//! in a dev build. Now, while a replay moves forward (watching it, or a seek fast-applying
//! records), a **checkpoint** is taken every interval of replay time (the replay's length over
//! [`CHECKPOINTS_PER_PASS`], at least [`MIN_INTERVAL`]), right after a tick record was re-run:
//!
//! * the whole [`Session`] (world, entities, local player, inventory, tab list, scoreboard,
//!   teams...), copied exactly: [`Session::rc_try_clone`], with the `hecs` arena copied entity
//!   by entity under the same handles ([`mc_world::World::try_clone_with`]);
//! * every client-side field the replayed packets and ticks change ([`ClientSim`]: chat, HUD,
//!   particles and their random stream, the other random streams, the tick counters, the block
//!   cracks, the tile index, the texture animation counters...);
//! * the playback cursor (the next record, the ticks applied, the re-driven key bindings).
//!
//! A seek backward, or a long one forward, restores the last checkpoint at or before the
//! target and fast-applies the records from there, through exactly the loop a seek from the
//! start runs. The restore tears the old world down as a rebuild does (`unload_world`: meshes,
//! render phases, sounds), installs copies of the checkpoint's state (the checkpoint stays
//! intact for the next seek), marks every chunk column for meshing and gives the culling walk
//! the restored world's columns (a rebuild's chunk packets mark them one by one).
//!
//! # Exactness
//!
//! A frame reached through a checkpoint must be byte-identical to the one reached by playing
//! from the start. The copy is complete by construction where it can be: [`Session`]'s copy
//! names every field (a new field stops it compiling), and a world whose entity arena holds a
//! component type the cloner does not list is never copied (checkpoints then switch off for that
//! replay and seeks rebuild as before). The client-side fields are a list ([`ClientSim`]); the
//! determinism test (`rc_seek_determinism` in `tests/`) compares frames reached both ways.
//!
//! A few vanilla values count *drawn* frames rather than ticks or packets: the boss bar
//! (`BossStatus`, refreshed each frame the boss is drawn and counted down each drawn frame) and
//! the wither sky tint it requests. A seek draws no frames on its way, so a rebuild reaches its
//! target with them cleared; a restore clears them too (they come back with the next frame that
//! draws the boss), which makes a restore and a rebuild land on the same frame exactly.
//! Continuous playback keeps vanilla's linger.
//!
//! Texture animations (water, lava, fire, portals, clocks) count ticks on their own. They now
//! restart when a replay opens or rebuilds, and a restore puts back the checkpoint's counters
//! and redraws the animated sprites, so their phase depends only on the replay time.
//!
//! # Memory
//!
//! Chunk columns are shared with the live world and between checkpoints (copy-on-write through
//! `Arc::make_mut`), so a checkpoint holds its own copy only of what changed since. At most
//! [`MAX_CHECKPOINTS`] are kept: when the list is full every other one goes (the first stays)
//! and the interval doubles, so they stay spread over the whole replay. They are dropped with
//! the replay when the viewer closes it.
//!
//! # Cost when not replaying, and while playing
//!
//! Nothing without a replay (they live in the playback). While a replay plays, one comparison
//! per re-run tick, and one capture per interval of replay time (a few milliseconds: the copy of
//! the entity arena and tracker, the particles and the client fields; chunk columns are only
//! reference-counted).

use std::collections::HashMap;
use std::time::Instant;

use mc_physics::state::MoveInput;
use mc_render::KeyBindings;
use mc_replay::ReplayTime;
use mc_world::EcsCloner;

use super::super::super::*;
use crate::session::Session;

/// The shortest replay time between two checkpoints (µs).
pub(crate) const MIN_INTERVAL: ReplayTime = 10_000_000;
/// One pass over a replay takes about this many checkpoints: the interval is the replay's
/// length over this (at least [`MIN_INTERVAL`]), so a seek fast-applies at most that much
/// replay time while a long replay is not copied every ten seconds.
const CHECKPOINTS_PER_PASS: u64 = 48;
/// The most checkpoints one replay keeps. Over it, every other one goes and the interval
/// doubles.
pub(crate) const MAX_CHECKPOINTS: usize = 64;
/// The memory budget of one replay's checkpoints ([`Checkpoint::estimate`] summed). Over it,
/// every other one goes and the interval doubles, as over the count cap. The estimate came to
/// about 80% of the measured working-set cost on the hour-long test replay (105.7 MB estimated,
/// +132 MB measured, for 24 checkpoints), so this keeps the real cost near 500 MB at most.
pub(crate) const MAX_BYTES: usize = 400 << 20;
/// A forward seek restores a checkpoint only when that skips at least this much replay time.
const FORWARD_SKIP: ReplayTime = 15_000_000;
/// The seed `Client::new` gives the local player's water-entry splash
/// stream (`Client::water_entry_rng`), which `Client::tw_rc_replay_start` restarts.
const WATER_ENTRY_RNG_SEED: i64 = 0x0057_4154_4552;

/// Whether checkpoints are on. `MC_RC_CHECKPOINTS=0` switches them off (the old rebuild path),
/// for A/B timing and as a safety valve. Read once.
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MC_RC_CHECKPOINTS").ok().as_deref() != Some("0"))
}

/// Every component type the client puts in the entity arena. A component of a type missing
/// here makes [`mc_world::World::try_clone_with`] refuse the copy (checkpoints then switch
/// off), so a new component can never be dropped from a checkpoint silently.
fn component_cloner() -> EcsCloner {
    EcsCloner::with_world_components()
        .with::<crate::app::entrender::ObjectData>()
        .with::<crate::app::entrender::XpValue>()
        .with::<crate::app::entrender::Hanging>()
        .with::<crate::sync::entities::RemotePlayer>()
        .with::<crate::sync::entities::ArrowShake>()
        .with::<crate::sync::entities::ArrowFlight>()
}

/// Declares [`ClientSim`] with one line per field, so capture and restore can never disagree.
macro_rules! client_sim {
    ($($field:ident: $ty:ty,)*) => {
        /// The client-side state the replayed records change outside the [`Session`]: what
        /// `unload_world` resets, the random streams a tick draws from, and the tick counters.
        #[derive(Clone)]
        pub(crate) struct ClientSim {
            $($field: $ty,)*
        }

        impl ClientSim {
            fn capture(client: &Client) -> ClientSim {
                ClientSim {
                    $($field: client.$field.clone(),)*
                }
            }

            fn restore(self, client: &mut Client) {
                $(client.$field = self.$field;)*
            }
        }
    };
}

client_sim! {
    chat: ChatLog,
    hud: Hud,
    spectator_hud: SpectatorHud,
    player_hud: PlayerHud,
    tab_list: TabListOverlay,
    particles: ParticleSystem,
    hit: HitResult,
    render_hit: HitResult,
    controller: PlayerController,
    player_container: Container,
    pending_local_item_drops: Vec<mc_protocol::ItemStack>,
    third_person_view: u8,
    third_person_obstructed: bool,
    hide_gui: bool,
    pending_hotbar_scroll: i32,
    tick: u64,
    rain_sound_counter: i32,
    local_player_ticks_existed: u32,
    active_effects: HashMap<i8, TrackedEffect>,
    active_effect_order: JavaEffectMapOrder,
    last_lightning_bolt: u8,
    lightning_spawn_nonce: u64,
    torch_flicker: TorchFlicker,
    fog_light_smoothing: FogLightSmoothing,
    boss_color_modifier: BossColorModifier,
    boss_color_modifier_requested: bool,
    wither_heads: HashMap<i32, WitherHeadState>,
    wither_max_health: HashMap<i32, f32>,
    max_health_attributes: HashMap<i32, crate::session::AttributeSnapshot>,
    flicker_rng: mc_math::JavaRandom,
    enderman_shimmer_rng: mc_math::JavaRandom,
    camera_prev_pos: Option<DVec3>,
    camera_prev_rotation: Option<(f32, f32)>,
    camera_prev_corrections: u64,
    damaged_blocks: HashMap<i32, DamagedBlock>,
    fov_modifier_hand: f32,
    fov_modifier_hand_prev: f32,
    statistics: HashMap<String, i32>,
    achievement_toast: stats::AchievementToast,
    achievement_statistics_received: bool,
    tile_index: crate::app::tiles::TileIndex,
    time_in_portal: f32,
    prev_time_in_portal: f32,
    portal_rng: mc_math::JavaRandom,
    water_entry_rng: mc_math::JavaRandom,
    respiration_rng: mc_math::JavaRandom,
    item_use_rng: mc_math::JavaRandom,
    item_use_math_rng: mc_math::JavaRandom,
    bow_break_world_rng: mc_math::JavaRandom,
    bow_break_entity_rng: mc_math::JavaRandom,
    local_status_sound_rng: mc_math::JavaRandom,
    first_person_latch: crate::app::first_person_equip::EquipLatch,
    furnace_minecart_predictions: HashMap<i32, FurnaceMinecartClientState>,
    command_minecart_track_output: HashMap<i32, bool>,
    spawner_rng: mc_math::JavaRandom,
    barrier_rng: mc_math::JavaRandom,
    ambient_callback_rng: mc_math::JavaRandom,
    light_check_rng: mc_math::JavaRandom,
    atlas_animator: Option<mc_assets::AtlasAnimator>,
    atlas_animation_generation: u64,
    atlas_animation_rng: mc_math::JavaRandom,
    pickup_rng: mc_math::JavaRandom,
    pickup_effects: Vec<crate::app::entrender::PickupEffect>,
}

/// State the replayed ticks drive inside render phases, outside both the
/// [`Session`] and [`ClientSim`]. `unload_world` clears it, so
/// a restore without it differed from a rebuild: the held item vanished from the hand until the
/// player switched items (the model is latched only at an equip swap, and the restored equip
/// latch already matched), and a lightning bolt alive at the checkpoint was lost. `None`
/// without a renderer (headless runs have neither).
#[derive(Clone)]
struct RenderTick {
    first_person: mc_render::pipeline::FirstPersonHeld,
    lightning: mc_render::sky::weather::LightningSnapshot,
}

impl RenderTick {
    fn capture(client: &Client) -> Option<RenderTick> {
        let render = client.render.as_ref()?;
        Some(RenderTick {
            first_person: render.first_person.with(|phase| phase.held_snapshot()),
            lightning: render.weather.with(|phase| phase.lightning_snapshot()),
        })
    }

    fn restore(self, client: &Client) {
        if let Some(render) = client.render.as_ref() {
            render
                .first_person
                .with(|phase| phase.restore_held(self.first_person));
            render
                .weather
                .with(|phase| phase.restore_lightning(self.lightning));
        }
    }
}

/// One checkpoint: the replayed client right after a tick record.
struct Checkpoint {
    /// The replay time of the tick record it was taken after.
    time: ReplayTime,
    /// The next record to apply.
    cursor: usize,
    /// The client tick of that tick record.
    applied_ticks: u64,
    last_keys: Option<(MoveInput, u32)>,
    keys: KeyBindings,
    /// The recorded screen as of this point.
    gui: super::gui::GuiPlayback,
    session: Session,
    sim: ClientSim,
    /// The render phases' tick state.
    render: Option<RenderTick>,
    /// Every player's cape chase history (kept beside the skin caches).
    capes: crate::app::appearance::CapeSnapshot,
    /// About how much memory it holds beyond the checkpoint before it ([`Checkpoint::estimate`]).
    bytes: usize,
}

impl Checkpoint {
    /// About how many bytes this checkpoint holds that `previous` (the one before it) does not
    /// share: chunk columns and sections that are not the same `Arc`, the tile entities, the
    /// entities (arena and tracker), the particles and the chat. Heap payloads (sign text, NBT,
    /// chat components) are not walked: a budget estimate, not an exact count.
    fn estimate(&self, previous: Option<&Checkpoint>) -> usize {
        let state = self.session.state();
        let world = state
            .world
            .bytes_not_shared_with(previous.map(|previous| &previous.session.state().world));
        let entities =
            state.entities.len() * std::mem::size_of::<crate::sync::entities::TrackedEntity>();
        let particles =
            self.sim.particles.len() * std::mem::size_of::<mc_render::particle::Particle>();
        let chat = self.sim.chat.lines().len() * 256;
        world + entities + particles + chat
    }
}

/// Counters for the bench, the tests and the viewer's debug line.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CheckpointStats {
    /// Checkpoints taken.
    pub taken: u64,
    /// Seeks that restored one.
    pub restored: u64,
    /// Times the list was thinned (every other one dropped, the interval doubled).
    pub thinned: u64,
    /// Checkpoints kept now.
    pub kept: usize,
    /// Their memory estimate now (MB): what they hold beyond each other and the live world.
    pub bytes_mb: f64,
    /// The interval now (seconds of replay time).
    pub interval_s: f64,
    /// The slowest capture so far (ms).
    pub capture_ms_max: f64,
    /// The slowest copy of the session alone (world, entities, player) so far (ms).
    pub capture_session_ms_max: f64,
    /// All captures together (ms).
    pub capture_ms_total: f64,
    /// The last restore's own cost (ms), before the records after it are applied.
    pub restore_ms_last: f64,
    /// Why checkpoints switched off for this replay, if they did.
    pub off: Option<&'static str>,
}

/// The checkpoints of the replay being watched (a field of the playback, so they go with it).
pub(crate) struct Checkpoints {
    list: Vec<Checkpoint>,
    /// The visible length of the replay, which sets the starting interval.
    span: ReplayTime,
    interval: ReplayTime,
    next_at: ReplayTime,
    cloner: EcsCloner,
    stats: CheckpointStats,
    /// Test control only ([`Client::replay_debug_break_checkpoints`]): restores keep the live
    /// client-side fields instead of the checkpoint's, as a forgotten field list would.
    break_restores: bool,
    /// Test control only ([`Client::replay_debug_skip_render_restore`]):
    /// restores leave the render phases as `unload_world` does, as they did before this change.
    skip_render_restore: bool,
}

impl std::fmt::Debug for Checkpoints {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Checkpoints")
            .field("kept", &self.list.len())
            .field("interval", &self.interval)
            .field("next_at", &self.next_at)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl Default for Checkpoints {
    fn default() -> Self {
        Checkpoints::for_span(0)
    }
}

impl Checkpoints {
    /// The checkpoints of a replay whose visible part is `span` long.
    pub(crate) fn for_span(span: ReplayTime) -> Self {
        let interval = (span / CHECKPOINTS_PER_PASS).max(MIN_INTERVAL);
        Checkpoints {
            list: Vec::new(),
            span,
            interval,
            next_at: 0,
            cloner: component_cloner(),
            stats: CheckpointStats {
                off: (!enabled()).then_some("MC_RC_CHECKPOINTS=0"),
                interval_s: interval as f64 / 1e6,
                ..CheckpointStats::default()
            },
            break_restores: false,
            skip_render_restore: false,
        }
    }

    /// The last checkpoint a seek to `target_ticks` can start from: taken after a tick no later
    /// than the target's (every record before it is then one the seek from the start applies
    /// too). `None` when there is none.
    fn best_for(&self, target_ticks: u64) -> Option<usize> {
        let after = self
            .list
            .partition_point(|checkpoint| checkpoint.applied_ticks <= target_ticks);
        after.checked_sub(1)
    }

    /// Whether a capture is due after a tick record at `time`, and none sits close by already
    /// (a pass over ground an earlier pass covered keeps the earlier checkpoints).
    fn due(&mut self, time: ReplayTime) -> bool {
        if self.stats.off.is_some() || time < self.next_at {
            return false;
        }
        let half = self.interval / 2;
        let near = self
            .list
            .iter()
            .filter(|checkpoint| checkpoint.time.abs_diff(time) < half)
            .map(|checkpoint| checkpoint.time)
            .max();
        if let Some(existing) = near {
            self.next_at = existing.max(time) + half;
            return false;
        }
        true
    }

    /// Insert in cursor order, then thin when over the cap.
    /// Insert in cursor order, then thin while over the count cap or the memory budget.
    fn insert(&mut self, checkpoint: Checkpoint) {
        self.next_at = checkpoint.time + self.interval;
        let at = self
            .list
            .partition_point(|existing| existing.cursor < checkpoint.cursor);
        self.list.insert(at, checkpoint);
        // Its own estimate, and its successor's (whose predecessor it now is).
        for index in [at, at + 1] {
            if index < self.list.len() {
                let bytes = self.list[index].estimate(index.checked_sub(1).map(|i| &self.list[i]));
                self.list[index].bytes = bytes;
            }
        }
        self.stats.taken += 1;
        while self.list.len() > 2
            && (self.list.len() > MAX_CHECKPOINTS || self.total_bytes() > MAX_BYTES)
        {
            // Keep the first and every other one after it; the gaps double, so does the
            // interval.
            let mut index = 0usize;
            self.list.retain(|_| {
                let keep = index.is_multiple_of(2);
                index += 1;
                keep
            });
            for index in 0..self.list.len() {
                let bytes = self.list[index].estimate(index.checked_sub(1).map(|i| &self.list[i]));
                self.list[index].bytes = bytes;
            }
            self.interval = self.interval.saturating_mul(2);
            self.stats.thinned += 1;
        }
        self.stats.kept = self.list.len();
        self.stats.bytes_mb = self.total_bytes() as f64 / 1e6;
        self.stats.interval_s = self.interval as f64 / 1e6;
    }

    /// The memory estimate of every checkpoint kept.
    fn total_bytes(&self) -> usize {
        self.list.iter().map(|checkpoint| checkpoint.bytes).sum()
    }
}

/// What a restore installs (copies; the checkpoint itself stays for later seeks).
struct Restored {
    time: ReplayTime,
    cursor: usize,
    applied_ticks: u64,
    last_keys: Option<(MoveInput, u32)>,
    keys: KeyBindings,
    gui: super::gui::GuiPlayback,
    session: Session,
    sim: ClientSim,
    render: Option<RenderTick>,
    capes: crate::app::appearance::CapeSnapshot,
}

impl Client {
    /// Right after `tw_rb_advance_to` re-ran the tick record at `tick_time`: take a checkpoint
    /// when one is due.
    pub(crate) fn tw_rc_after_tick(&mut self, tick_time: ReplayTime) {
        let due = self
            .tw_play
            .playback
            .as_deref_mut()
            .is_some_and(|playback| !playback.packets_pending && playback.rc.due(tick_time));
        if due {
            self.tw_rc_capture(tick_time);
        }
    }

    fn tw_rc_capture(&mut self, tick_time: ReplayTime) {
        let started = Instant::now();
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return;
        };
        let session = self
            .session
            .as_ref()
            .and_then(|session| session.rc_try_clone(&playback.rc.cloner));
        let session_ms = started.elapsed().as_secs_f64() * 1e3;
        let Some(session) = session else {
            if let Some(playback) = self.tw_play.playback.as_deref_mut() {
                tracing::warn!(
                    "replay checkpoints off: the replayed world could not be copied (an entity \
                     component the checkpoint cloner does not list); seeks rebuild from the start"
                );
                playback.rc.stats.off = Some("the world could not be copied");
            }
            return;
        };
        let checkpoint = Checkpoint {
            time: tick_time,
            cursor: playback.cursor,
            applied_ticks: playback.applied_ticks,
            last_keys: playback.last_keys,
            keys: playback.keys.clone(),
            gui: playback.gui.clone(),
            session,
            sim: ClientSim::capture(self),
            render: RenderTick::capture(self),
            capes: self.appearance.cape_snapshot(),
            bytes: 0,
        };
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.rc.insert(checkpoint);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            playback.rc.stats.capture_ms_total += ms;
            playback.rc.stats.capture_ms_max = playback.rc.stats.capture_ms_max.max(ms);
            playback.rc.stats.capture_session_ms_max =
                playback.rc.stats.capture_session_ms_max.max(session_ms);
        }
    }

    /// From `tw_rb_begin_seek`: restore the best checkpoint for a seek to `target` when that
    /// saves work: always for a backward seek (instead of a rebuild from the start), and for a
    /// forward one when it skips at least [`FORWARD_SKIP`]. Returns whether it restored one;
    /// the seek then fast-applies the records from the checkpoint to the target.
    pub(crate) fn tw_rc_restore_for_seek(&mut self, target: ReplayTime, backward: bool) -> bool {
        if !enabled() {
            return false;
        }
        let Some(playback) = self.tw_play.playback.as_deref() else {
            return false;
        };
        let target_ticks = playback.target_ticks(target);
        let Some(index) = playback.rc.best_for(target_ticks) else {
            return false;
        };
        let checkpoint = &playback.rc.list[index];
        let worth_it = if backward {
            true
        } else {
            checkpoint.applied_ticks > playback.applied_ticks
                && checkpoint.cursor > playback.cursor
                && checkpoint.time >= playback.time.saturating_add(FORWARD_SKIP)
        };
        if !worth_it {
            return false;
        }
        let started = Instant::now();
        let Some(session) = checkpoint.session.rc_try_clone(&playback.rc.cloner) else {
            return false;
        };
        let sim = if playback.rc.break_restores {
            ClientSim::capture(self)
        } else {
            checkpoint.sim.clone()
        };
        // The broken control (a forgotten field list) leaves the render phases as
        // `unload_world` does, which is what restores did before this change.
        let render = if playback.rc.break_restores || playback.rc.skip_render_restore {
            None
        } else {
            checkpoint.render.clone()
        };
        // The broken control keeps the live capes, like its other fields.
        let capes = if playback.rc.break_restores {
            self.appearance.cape_snapshot()
        } else {
            checkpoint.capes.clone()
        };
        let restored = Restored {
            time: checkpoint.time,
            cursor: checkpoint.cursor,
            applied_ticks: checkpoint.applied_ticks,
            last_keys: checkpoint.last_keys,
            keys: checkpoint.keys.clone(),
            gui: checkpoint.gui.clone(),
            session,
            sim,
            render,
            capes,
        };
        self.tw_rc_install(restored);
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.rc.stats.restored += 1;
            playback.rc.stats.restore_ms_last = started.elapsed().as_secs_f64() * 1e3;
            // The next capture after the restored one, on the same grid.
            playback.rc.next_at = playback.rc.list[index].time + playback.rc.interval;
        }
        true
    }

    /// Tear the current world down as a rebuild does, then install a checkpoint's copies.
    fn tw_rc_install(&mut self, restored: Restored) {
        // The viewer's own settings survive the restore (the copy holds the ones from when it
        // was taken): particle density and the particles mod's "Use particle physics".
        let density = self.particles.density();
        let physics = self.particles.physics();
        self.unload_world();
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.cursor = restored.cursor;
            playback.applied_ticks = restored.applied_ticks;
            // The world is at the checkpoint's tick; the seek moves `time` and the clock sample
            // to its target once it gets there.
            playback.time = restored.time;
            playback.sample = None;
            playback.packets_pending = false;
            playback.running_tick = None;
            playback.last_keys = restored.last_keys;
            playback.keys = restored.keys;
            playback.gui = restored.gui;
        }
        let generation = mc_assets::model::installed_generation();
        let animator_fits = restored.sim.atlas_animation_generation == generation;
        restored.sim.restore(self);
        // After `unload_world` reset them.
        self.appearance.restore_capes(restored.capes);
        // After `unload_world` cleared them: the hand's latched model and
        // equip/swing state, and live lightning. A resource-pack change since the checkpoint
        // (the atlas generation moved) makes its latched model stale: the hand re-equips from
        // scratch instead (an empty latch), as after a world load.
        if let Some(render) = restored.render {
            if animator_fits {
                render.restore(self);
            } else {
                if let Some(phases) = self.render.as_ref() {
                    phases
                        .weather
                        .with(|phase| phase.restore_lightning(render.lightning));
                }
                self.first_person_latch = crate::app::first_person_equip::EquipLatch::default();
            }
        }
        self.tw_rc_forget_drawn_frames();
        self.particles.set_density(density);
        self.particles.set_physics(physics);
        self.session = Some(restored.session);
        // Every chunk column was meshed for the world just torn down: mesh the restored one.
        if let Some(session) = self.session.as_mut() {
            session.state_mut().world.mark_every_column_dirty();
        }
        // The culling walk's world-column set. `unload_world`
        // emptied it, and the restored world arrives whole, without the chunk events that mark
        // each column loaded as a rebuild's packets do. Left empty, every column of the restored
        // world counted as absent (settled air), so until its sections were meshed the walk saw
        // through all of them and drew what a rebuild keeps behind its not-yet-meshed sections:
        // entities through walls, and a hidden boss whose boss bar then showed.
        self.rehydrate_authoritative_world_columns();
        self.sidebar_cache = None;
        if animator_fits {
            self.tw_rc_redraw_animations();
        } else {
            // A resource-pack change since the checkpoint: the next tick builds a new animator.
            self.atlas_animator = None;
            self.atlas_animation_generation = 0;
        }
        self.tw_rb_close_game_screens();
    }

    /// Clear what vanilla keeps from *drawn* frames rather than from ticks or packets: the boss
    /// bar (`BossStatus`: set each frame the boss is drawn, counted down each drawn frame) and
    /// the wither sky tint it requests. A seek's fast-forward draws no frames, so a rebuild
    /// reaches its target with these cleared (`unload_world`); a checkpoint taken while frames
    /// were drawn would otherwise carry them, and a frame after a restore would differ from the
    /// same frame after a rebuild. They come back with the next drawn frame that shows the boss.
    fn tw_rc_forget_drawn_frames(&mut self) {
        self.hud.clear_boss_status();
        self.boss_color_modifier = BossColorModifier::default();
        self.boss_color_modifier_requested = false;
    }

    /// The determinism test's reference: what a jump does to drawn-frame state (see
    /// [`Self::tw_rc_forget_drawn_frames`]), applied after a plain forward seek.
    pub fn replay_debug_forget_drawn_frames(&mut self) {
        if self.tw_play.playback.is_some() {
            self.tw_rc_forget_drawn_frames();
        }
    }

    /// Put every animated sprite's current picture (by the animator's counters) into the atlas
    /// and queue its upload.
    fn tw_rc_redraw_animations(&mut self) {
        let Some(animator) = self.atlas_animator.as_ref() else {
            return;
        };
        let uploads = animator.write_current_frames(&mut self.assets.atlas);
        mc_assets::coalesce_animation_uploads(&mut self.atlas_animation_uploads, uploads);
    }

    /// What a replay's first record must find, whatever the client did before: called by the `replay_open` and `tw_rb_rebuild`, so opening a replay, a rebuild from its start and a
    /// restore all reach the same state.
    ///
    /// * The client's and the HUD's tick counters are the recording's from its first tick
    ///   record on, but the packets before that record (the join, its chat lines, block cracks)
    ///   stamp whatever the counters held: the previous position after a rebuild, the client's
    ///   own history after an open. They start at the recording's value just before its first
    ///   tick, as the recording client had them.
    /// * Texture animations count ticks on their own, so their phase would depend on how long
    ///   the client ran before. They restart here (a restore puts back the checkpoint's
    ///   counters and redraws the sprites).
    /// * The local player's water-entry splash stream (`water_entry_rng`, vanilla's
    ///   `EntityPlayerSP.rand` in `Entity.resetHeight`) restarts too. `unload_world` and the
    ///   join packet restart its sibling streams but never this one, so a rebuild used to draw
    ///   the splash offsets from wherever the earlier passes had left it: the splash particles
    ///   landed elsewhere than on the first pass and than through a checkpoint (which restores
    ///   the stream), and could change the particle RNG's draws from there on.
    pub(crate) fn tw_rc_replay_start(&mut self) {
        let before_first_tick = self.tw_play.playback.as_deref().and_then(|playback| {
            playback
                .tick_times
                .first()
                .map(|(tick, _)| tick.saturating_sub(1))
        });
        if let Some(tick) = before_first_tick {
            self.tick = tick;
            self.hud.update_counter = tick as i32;
        }
        self.water_entry_rng = mc_math::JavaRandom::new(WATER_ENTRY_RNG_SEED);
        if !self.assets.ready || !self.assets.models_installed {
            return;
        }
        self.atlas_animator = mc_assets::BakedAssets::installed_atlas_animator();
        self.atlas_animation_generation = mc_assets::model::installed_generation();
        self.atlas_animation_uploads.clear();
        self.tw_rc_redraw_animations();
    }

    /// The first-person phase's tick-driven state (equip, swing, use, the
    /// latched model), for tests: a jump must land on the hand that playing there shows. `None`
    /// without a renderer.
    pub fn replay_debug_first_person(&self) -> Option<mc_render::pipeline::FirstPersonHeld> {
        self.render
            .as_ref()
            .map(|render| render.first_person.with(|phase| phase.held_snapshot()))
    }

    /// The open replay's checkpoint counters.
    pub fn replay_checkpoint_stats(&self) -> Option<CheckpointStats> {
        self.tw_play
            .playback
            .as_deref()
            .map(|playback| playback.rc.stats)
    }

    /// Drop every checkpoint of the open replay (tests and the bench: time the path without
    /// them, or force a rebuild).
    pub fn replay_drop_checkpoints(&mut self) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            let off = playback.rc.stats.off;
            playback.rc = Checkpoints::for_span(playback.rc.span);
            playback.rc.stats.off = off;
        }
    }

    /// The determinism test's negative control: from now on a restore keeps the live
    /// client-side fields (chat, HUD, particles, random streams...) instead of the checkpoint's,
    /// which is what a forgotten field list would do. The test must then see a difference.
    pub fn replay_debug_break_checkpoints(&mut self, broken: bool) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.rc.break_restores = broken;
        }
    }

    /// The hand test's negative control: from now on a restore leaves the
    /// render phases' tick state (the hand, lightning) as `unload_world` leaves it, which is
    /// what restores did before this change. Everything else is restored as usual.
    pub fn replay_debug_skip_render_restore(&mut self, skip: bool) {
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.rc.skip_render_restore = skip;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cloner_lists_every_client_component() {
        let cloner = component_cloner();
        assert_eq!(
            cloner.len(),
            EcsCloner::with_world_components().len() + 6,
            "a component added to the cloner twice, or one dropped"
        );
    }

    /// A real client on the real assets, or `None` (skip) without the vanilla jar.
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = crate::app::config::ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: crate::app::config::LoginMode::Offline("Dev".to_owned()),
            ..crate::app::config::ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// Opening a replay and rebuilding it from its start both begin
    /// the local player's water-entry splash stream at `Client::new`'s seed, whatever drew from
    /// it before. A rebuild used to continue it from where the earlier passes had left it, so the
    /// splash particles landed elsewhere than on the first pass and a watched-then-restored frame
    /// differed from the rebuilt one (`rc_seek_determinism`'s watch test on the hour-long
    /// replay). Without the restart both checks fail.
    #[test]
    fn opening_and_rebuilding_a_replay_restart_the_water_entry_stream() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let fresh = mc_math::JavaRandom::new(WATER_ENTRY_RNG_SEED).raw_seed();
        // Play before the replay opens draws from the stream.
        for _ in 0..7 {
            let _ = client.water_entry_rng.next_f32();
        }
        assert_ne!(client.water_entry_rng.raw_seed(), fresh);
        let records: Vec<mc_replay::Record> = (0..10u64)
            .map(|i| mc_replay::Record {
                time: i * 100_000,
                stream: mc_replay::StreamId::Packet,
                payload: vec![0x02, 0, 0],
            })
            .collect();
        let meta = mc_replay::ReplayMeta {
            duration: 1_000_000,
            ..mc_replay::ReplayMeta::default()
        };
        client.replay_open_records(meta, records).expect("opens");
        assert_eq!(
            client.water_entry_rng.raw_seed(),
            fresh,
            "the open restarts it"
        );

        // A pass over the replay draws from it (the player entering water), then a seek back
        // with no checkpoint left rebuilds from the start.
        client.replay_render_at(900_000);
        for _ in 0..5 {
            let _ = client.water_entry_rng.next_f32();
        }
        client.replay_drop_checkpoints();
        let rebuilds = client.replay_stats().expect("open").rebuilds;
        client.replay_render_at(100_000);
        assert!(
            client.replay_stats().expect("open").rebuilds > rebuilds,
            "the seek back rebuilt the world"
        );
        assert_eq!(
            client.water_entry_rng.raw_seed(),
            fresh,
            "the rebuild restarts it"
        );
        client.replay_close();
    }

    /// A checkpoint restore hands the culling walk the restored
    /// world's columns, as a rebuild's chunk events do one by one. `unload_world` empties that
    /// set, and restores used to leave it empty: every column of the restored world then counted
    /// as absent (settled air), so until its sections were meshed the walk saw through all of
    /// them, and the frames right after a jump drew what a rebuild keeps behind sections that are
    /// not meshed yet (on a recording of a server lobby: a hidden boss, whose boss bar showed).
    #[test]
    fn a_restore_keeps_the_restored_worlds_unmeshed_sections_opaque_to_the_walk() {
        use mc_math::{ChunkPos, ChunkSectionPos};
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let gpu = match mc_render::Gpu::headless(64, 64, mc_render::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: no headless GPU ({error})");
                return;
            }
        };
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        // A replay's world of 7 x 7 loaded columns, none meshed yet, copied the way a checkpoint
        // holds it.
        let mut session = Session::new(
            crate::net::NetHandle::driven(),
            crate::session::JoinSettings::default(),
        );
        for x in -3..=3 {
            for z in -3..=3 {
                session
                    .state_mut()
                    .world
                    .load_empty_chunk(ChunkPos::new(x, z));
            }
        }
        let copy = session
            .rc_try_clone(&component_cloner())
            .expect("the session copies");
        client.session = Some(session);
        let restored = Restored {
            time: 0,
            cursor: 0,
            applied_ticks: 0,
            last_keys: None,
            keys: KeyBindings::default(),
            gui: super::super::gui::GuiPlayback::default(),
            session: copy,
            sim: ClientSim::capture(&client),
            render: None,
            capes: client.appearance.cape_snapshot(),
        };
        client.tw_rc_install(restored);

        // From the middle column, looking along +z: the camera's section and the next one are
        // reached; the one after that only through the next one, a loaded column's section that
        // is not meshed yet, which stops the walk (java: `CompiledChunk.DUMMY`).
        let mut camera = mc_render::Camera {
            pos: DVec3::new(8.0, 72.0, 8.0),
            yaw: 0.0,
            aspect: 1.0,
            ..mc_render::Camera::default()
        };
        camera.update();
        let next = ChunkSectionPos::new(0, 4, 1);
        let beyond = ChunkSectionPos::new(0, 4, 2);
        let render = client.render.as_ref().expect("installed");
        let mut terrain = render.terrain.borrow_mut();
        terrain.prepare_culling_walk(&camera);
        assert!(terrain.section_reached_by_walk(next), "the walk ran");
        assert!(
            !terrain.section_reached_by_walk(beyond),
            "after a restore the walk saw through a loaded section that is not meshed yet"
        );
        // Control: with the set as `unload_world` leaves it, the same walk sees through, so the
        // check above depends on the set.
        terrain.set_world_columns(std::iter::empty());
        terrain.prepare_culling_walk(&camera);
        assert!(
            terrain.section_reached_by_walk(beyond),
            "the control walk should see through absent columns"
        );
    }
}
