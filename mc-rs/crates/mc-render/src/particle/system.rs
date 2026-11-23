//! The particle arena: spawn/tick/expire, capacity, and the density gate.
//!
//! java: `client/particle/EffectRenderer.java` (`fxLayers`, `addEffect`, `updateEffects`),
//! `client/renderer/RenderGlobal.spawnParticle` (the density/range gate).

use std::time::{SystemTime, UNIX_EPOCH};

use mc_math::{BlockPos, DVec3, JavaRandom};
use mc_world::World;

use super::spawn;
use super::tick;
use super::tuning::{self, GroupTuning, ParticleGroup, ParticleTuning, Tuned};
use super::types::{AlphaBucket, FxLayer, Particle, ParticleKind, PER_BUCKET_CAP};

const EXPLOSION_NORMAL: i32 = 0;
const EXPLOSION_LARGE: i32 = 1;
const EXPLOSION_HUGE: i32 = 2;
const SMOKE_NORMAL: i32 = 11;

/// The part of a player entity that particle simulation observes.
///
/// `EntityCloudFX` uses its feet position and vertical motion after its own collision move. The
/// caller supplies every non-spectating player currently present in the client world; this small
/// snapshot keeps the particle crate independent from client ECS types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticlePlayer {
    /// `EntityPlayer.posX/Y/Z` (the feet position).
    pub pos: DVec3,
    /// `EntityPlayer.motionX/Y/Z` from the current tick.
    pub velocity: DVec3,
}

/// `GameSettings.particleSetting`: `0` = all, `1` = decreased, `2` = minimal.
/// java: `client/settings/GameSettings.java` field `particleSetting`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleDensity {
    /// Every particle spawns (subject only to the 16-block range check).
    All,
    /// java: `RenderGlobal.spawnParticle` — a `1/3` chance per spawn attempt to be demoted to
    /// [`ParticleDensity::Minimal`] for that attempt.
    Decreased,
    /// Non-range-ignoring particles never spawn.
    Minimal,
}

impl ParticleDensity {
    /// From [`crate::GameSettings`]'s raw `particles` field. Any value other than `1`/`2` is
    /// `All`, matching vanilla's unchecked `int` read (`GameSettings.particleSetting`).
    #[must_use]
    pub fn from_raw(v: i32) -> Self {
        match v {
            1 => Self::Decreased,
            2 => Self::Minimal,
            _ => Self::All,
        }
    }
}

fn layer_slot(l: FxLayer) -> u8 {
    match l {
        FxLayer::Misc => 0,
        FxLayer::Terrain => 1,
        // Both originate in vanilla fxLayers[3]. `Footprint` is a separate Rust draw state only
        // because its renderer binds another texture mid-pass; it must share layer 3's 4000 cap.
        FxLayer::LitExplosion | FxLayer::Footprint => 2,
    }
}

/// Number of `addEffect` cap buckets: three vanilla texture layers x two alpha buckets.
const BUCKETS: usize = 6;

/// The `(layer, alpha)` cap bucket as a dense index. java: `fxLayers[i][j]`.
fn bucket_index(p: &Particle) -> usize {
    usize::from(layer_slot(p.layer)) * 2 + usize::from(p.alpha_bucket() == AlphaBucket::Opaque)
}

/// A JVM-nanoTime-shaped, non-reproducible seed — matching vanilla's own `new Random()`
/// default-seeding, which is also never fixed. See `spawn`'s module docs.
fn entropy_seed() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// The live particle set. java: `EffectRenderer`.
///
/// # Cap bookkeeping (PART-1)
///
/// `particles` stays one `Vec` in global spawn order, because that order is both the tick (RNG)
/// order and, filtered per bucket, the draw order. Vanilla's `addEffect` size check is an O(1)
/// `List.size()`; scanning the whole arena per spawn made bursts quadratic, and `Vec::remove`
/// memmoved ~1 MB per evicting spawn at the cap. Instead:
///
/// * `bucket_len` counts every arena member per bucket (exactly what the old full scan counted,
///   including the never-reached "primed crit died" case). Alpha only changes inside `tick`, which
///   recounts while it compacts, so the counts are exact at every spawn.
/// * Eviction still removes the *first member of the bucket in arena order* (the old
///   `position(..)` pick), found from a per-bucket cursor that only moves forward between
///   compactions, and is recorded in `evicted` instead of being memmoved out immediately.
/// * Every public mutator ends with [`Self::flush_evictions`], so [`Self::particles`] and every
///   other observer only ever see the compacted arena: the same elements, in the same order, as
///   the old `Vec::remove` produced. Burst producers ([`Self::spawn_explosion_effects`], tick
///   children, `packet`'s loops) defer the flush to the end of the burst, which turns `B`
///   evicting spawns from `B` memmoves into one compaction pass.
// Replay checkpoints copy the whole system.
#[derive(Clone)]
pub struct ParticleSystem {
    rng: JavaRandom,
    particles: Vec<Particle>,
    density: ParticleDensity,
    /// Arena members per cap bucket, evicted-but-not-yet-compacted entries excluded.
    bucket_len: [usize; BUCKETS],
    /// Per bucket: no un-evicted member of that bucket sits below this arena index.
    evict_cursor: [usize; BUCKETS],
    /// Arena indices evicted by the cap since the last compaction (distinct, unsorted).
    evicted: Vec<usize>,
    /// Reused per-tick buffer for particles spawned by other particles' `onUpdate`.
    spawn_scratch: Vec<Particle>,
    /// The particles mod's per-group amount/size table. `None`
    /// while the mod is off or changes nothing at the funnel: every spawn path then runs its
    /// unchanged vanilla branch after this one `Option` test. See [`super::tuning`].
    tuning: Option<Box<ParticleTuning>>,
    /// The particles mod's "Use particle physics". `true` is
    /// vanilla: particles collide with blocks. `false` moves every particle the way vanilla moves
    /// a `noClip` one (no collision sweep): particle physics off.
    physics: bool,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticleSystem {
    /// An empty system, RNG seeded from the process clock (production use).
    #[must_use]
    pub fn new() -> Self {
        Self::with_rng(JavaRandom::new(entropy_seed()))
    }

    fn with_rng(rng: JavaRandom) -> Self {
        Self {
            rng,
            particles: Vec::new(),
            density: ParticleDensity::All,
            bucket_len: [0; BUCKETS],
            evict_cursor: [0; BUCKETS],
            evicted: Vec::new(),
            spawn_scratch: Vec::new(),
            tuning: None,
            physics: true,
        }
    }

    /// An empty system with a fixed RNG seed, for deterministic tests. The *jitter sequence*
    /// this produces has no vanilla equivalent to match (see `spawn`'s module docs) — this is
    /// for reproducing **this port's own** test runs, not vanilla's.
    #[must_use]
    pub fn with_seed(seed: i64) -> Self {
        Self::with_rng(JavaRandom::new(seed))
    }

    /// Current [`ParticleDensity`]. java: `GameSettings.particleSetting`, mirrored here rather
    /// than read live from [`crate::GameSettings`] so this crate's `particle/**` never has to
    /// reach into `settings.rs` (owned by S10) at spawn time.
    #[must_use]
    pub fn density(&self) -> ParticleDensity {
        self.density
    }

    /// Sets the density, e.g. from a video-options change. java: `GameSettings.particleSetting`
    pub fn set_density(&mut self, density: ParticleDensity) {
        self.density = density;
    }

    /// Direct access to the shared jitter/spawn RNG, for callers ([`super::packet`]) building
    /// particles with [`super::spawn`] before handing them to [`ParticleSystem::spawn`].
    pub fn rng_mut(&mut self) -> &mut JavaRandom {
        &mut self.rng
    }

    // ---- the particles mod -----------------------------------

    /// Install, update or remove the particles mod's per-group table. `None`, or values that
    /// change nothing at the funnel ([`ParticleTuning::is_vanilla`]), leaves no table at all, so
    /// every spawn path is vanilla's again. An update keeps the mod's own decision stream.
    pub fn set_tuning(&mut self, groups: Option<[GroupTuning; ParticleGroup::COUNT]>) {
        match groups.filter(|groups| !ParticleTuning::is_vanilla(groups)) {
            None => self.tuning = None,
            Some(groups) => match self.tuning.as_deref_mut() {
                Some(tuning) => tuning.set_groups(groups),
                None => self.tuning = Some(Box::new(ParticleTuning::new(groups))),
            },
        }
    }

    /// The particles mod's installed table, if any.
    #[must_use]
    pub fn tuning(&self) -> Option<&ParticleTuning> {
        self.tuning.as_deref()
    }

    /// The particles mod's "Use particle physics". `true` (the default, vanilla):
    /// particles collide with blocks. `false`: every particle skips its block collision sweep
    /// and moves as vanilla moves a `noClip` particle (`Entity.moveEntity`'s early branch), so
    /// it passes through floors and walls and is never on the ground. Spawning, RNG draws,
    /// lifetimes and everything that is not collision stay the same.
    pub fn set_physics(&mut self, on: bool) {
        self.physics = on;
    }

    /// Whether particles collide with blocks ([`Self::set_physics`]).
    #[must_use]
    pub fn physics(&self) -> bool {
        self.physics
    }

    /// The particles mod's decision for one gated attempt of `group` in a burst of `burst`
    /// attempts. `None` without a table: take the vanilla branch.
    pub fn tune_group(&mut self, group: ParticleGroup, burst: usize) -> Option<Tuned> {
        Some(self.tuning.as_deref_mut()?.tune(group, burst))
    }

    /// The particles mod's half of one gated spawn attempt: `tuned.copies` particles, each a
    /// fresh `make` from this system's RNG with its render size scaled
    /// ([`tuning::scale_render`]), added through [`Self::spawn_deferred`] so the cap holds. The
    /// caller flushes, as for any deferred spawn. Returns how many were added.
    pub fn spawn_tuned_deferred(
        &mut self,
        world: &World,
        tuned: Tuned,
        mut make: impl FnMut(&mut JavaRandom) -> Option<Particle>,
    ) -> u32 {
        let mut added = 0;
        for _ in 0..tuned.copies {
            // A factory that declines (no icon, say) declines every copy the same way.
            let Some(mut particle) = make(&mut self.rng) else {
                break;
            };
            tuning::scale_render(&mut particle, tuned.size);
            self.spawn_deferred(world, particle);
            added += 1;
        }
        added
    }

    /// One gated spawn attempt of vanilla particle type `particle_id`, built by `make`.
    ///
    /// Vanilla (no table): `make` once with this system's RNG, then [`Self::spawn`], exactly the
    /// sequence a caller writing those two lines gets. With the particles mod's table: its
    /// copies and size for the type's group. Returns whether anything was added.
    pub fn spawn_attempt(
        &mut self,
        world: &World,
        particle_id: i32,
        make: impl FnMut(&mut JavaRandom) -> Option<Particle>,
    ) -> bool {
        let added = self.spawn_attempt_deferred(world, particle_id, 1, make);
        self.flush_evictions();
        added
    }

    /// [`Self::spawn_attempt`] for one member of a burst of `burst` attempts (the particles
    /// mod limits how far a burst is multiplied; see [`tuning::BURST_LIMIT`]). Deferred: the
    /// caller must [`Self::flush_evictions`] after the burst.
    pub fn spawn_attempt_deferred(
        &mut self,
        world: &World,
        particle_id: i32,
        burst: usize,
        mut make: impl FnMut(&mut JavaRandom) -> Option<Particle>,
    ) -> bool {
        if let Some(tuning) = self.tuning.as_deref_mut() {
            let tuned = tuning.tune_id(particle_id, burst);
            return self.spawn_tuned_deferred(world, tuned, make) > 0;
        }
        let Some(particle) = make(&mut self.rng) else {
            return false;
        };
        self.spawn_deferred(world, particle);
        true
    }

    /// java: `RenderGlobal.spawnParticle`'s gate — the density demotion coin flip, then either
    /// the unconditional pass for `ignore_range` particles or the 16-block range check plus the
    /// (possibly demoted) density check.
    ///
    /// `cam_pos` is the render-view entity's **feet** position — vanilla measures
    /// `mc.getRenderViewEntity().posX/posY/posZ - coord` against `256.0`
    /// (`RenderGlobal.java:2081-2098`), i.e. the raw tick position, not the interpolated camera
    /// eye, which is `+1.62` in Y. Do **not** pass [`crate::frame::Camera::pos`]; every
    /// production caller passes `Predictor::state.pos`.
    #[must_use]
    pub fn should_spawn(
        &mut self,
        ignore_range: bool,
        cam_pos: DVec3,
        particle_pos: DVec3,
    ) -> bool {
        let mut level = self.density;
        if level == ParticleDensity::Decreased && self.rng.next_i32_bound(3) == 0 {
            level = ParticleDensity::Minimal;
        }
        if ignore_range {
            return true;
        }
        if (cam_pos - particle_pos).length_squared() > 256.0 {
            return false;
        }
        level != ParticleDensity::Minimal
    }

    /// Adds a particle, evicting the oldest particle sharing its texture-layer/alpha bucket if
    /// that bucket is already at [`PER_BUCKET_CAP`]. java: `EffectRenderer.addEffect`.
    ///
    /// A handful of kinds run one tick synchronously before ever being drawn — vanilla's
    /// `EntityCrit2FX` constructor calls `this.onUpdate()` itself. `world` is only used for that
    /// priming tick's collision sweep.
    pub fn spawn(&mut self, world: &World, p: Particle) {
        self.spawn_deferred(world, p);
        self.flush_evictions();
    }

    /// [`Self::spawn`] for one member of a burst: the cap check and eviction are O(1), and an
    /// evicted particle is only *recorded*. The caller **must** call
    /// [`Self::flush_evictions`] after the burst, before anything reads [`Self::particles`];
    /// every element and the arena order are then exactly what the same sequence of
    /// [`Self::spawn`] calls leaves. RNG use is identical to [`Self::spawn`] (only the crit
    /// priming tick draws from it).
    pub fn spawn_deferred(&mut self, world: &World, p: Particle) {
        let key = bucket_index(&p);
        if self.bucket_len[key] >= PER_BUCKET_CAP {
            self.evict_oldest(key);
        }
        let prime = matches!(p.kind, ParticleKind::Crit { .. });
        self.particles.push(p);
        let idx = self.particles.len() - 1;
        if prime {
            // Ticking in place is the old `remove(len - 1)` / tick / `push` sequence without the
            // two moves: `tick_one` sees only this particle, the world and the RNG.
            let p = &mut self.particles[idx];
            if self.physics {
                tick::tick_one(p, world, &mut self.rng);
            } else {
                tick::tick_one_without_collision(p, world, &mut self.rng, &[]);
            }
        }
        // Counted under its post-priming bucket, as the old full scan would see it next spawn.
        self.bucket_len[bucket_index(&self.particles[idx])] += 1;
        // Bound the arena. An S2A particle count is an uncapped server `i32`, so one packet could
        // otherwise grow it without limit before the burst's single flush. Flushing here leaves
        // exactly what the same sequence of `spawn` calls leaves, so the result is unchanged.
        if self.evicted.len() >= PER_BUCKET_CAP {
            self.flush_evictions();
        }
    }

    /// Records the eviction of the first un-evicted member of bucket `key` in arena order —
    /// the element the old `particles.iter().position(|q| bucket_of(q) == key)` removed.
    ///
    /// Members below `evict_cursor[key]` have all been evicted already, and new members are only
    /// ever pushed at the end, so the cursor moves forward monotonically until the next
    /// compaction; a burst of evictions therefore scans each arena slot at most once per bucket.
    /// Evicted entries keep their data (and therefore their bucket), so the scan cannot pick one
    /// twice: the only ones at or above the cursor belong to other buckets.
    fn evict_oldest(&mut self, key: usize) {
        let start = self.evict_cursor[key];
        if let Some(offset) = self.particles[start..]
            .iter()
            .position(|q| bucket_index(q) == key)
        {
            let idx = start + offset;
            self.evicted.push(idx);
            self.evict_cursor[key] = idx + 1;
            self.bucket_len[key] -= 1;
        }
    }

    /// Compacts particles evicted by [`Self::spawn_deferred`] out of the arena in one
    /// order-preserving pass. A no-op when nothing was evicted.
    pub fn flush_evictions(&mut self) {
        if self.evicted.is_empty() {
            return;
        }
        if self.evicted.len() == 1 {
            // A lone `spawn()` at the cap: exactly the old `Vec::remove` (one bulk memmove),
            // which beats `retain`'s element-wise moves when only one slot goes.
            self.particles.remove(self.evicted[0]);
            self.evicted.clear();
            self.evict_cursor = [0; BUCKETS];
            return;
        }
        self.evicted.sort_unstable();
        let mut next = self.evicted.iter().copied().peekable();
        let mut index = 0usize;
        // `Vec::retain` visits every element exactly once, in order.
        self.particles.retain(|_| {
            let keep = next.peek() != Some(&index);
            if !keep {
                next.next();
            }
            index += 1;
            keep
        });
        self.evicted.clear();
        // Arena indices shifted; "no live member below 0" is trivially true again.
        self.evict_cursor = [0; BUCKETS];
    }

    /// Advances every particle by one tick: physics/behaviour, expiry, and anything a kind
    /// spawns as a side effect (lava's smoke, a huge explosion's debris, firework trails) —
    /// re-entering [`ParticleSystem::spawn`] so the new particles get the same cap enforcement.
    /// java: `EffectRenderer.updateEffects`/`updateEffectAlphaLayer`.
    pub fn tick(&mut self, world: &World) {
        self.tick_with_players(world, &[]);
    }

    /// Advances every particle, also supplying the client world's non-spectating players to
    /// particle types which query them. The slice is deliberately a world snapshot rather than
    /// an iterator so each cloud sees the same entity state, as it does during vanilla's
    /// `EffectRenderer.updateEffects` pass.
    ///
    /// java: `EntityCloudFX.onUpdate` -> `World.getClosestPlayerToEntity(this, 2.0D)`.
    pub fn tick_with_players(&mut self, world: &World, players: &[ParticlePlayer]) {
        // Every in-crate mutator already flushed; this only guards an external
        // `spawn_deferred` caller that forgot to (evicted entries are not marked `dead`).
        self.flush_evictions();
        let mut spawned = std::mem::take(&mut self.spawn_scratch);
        spawned.clear();
        let physics = self.physics;
        for p in &mut self.particles {
            if p.dead {
                continue;
            }
            let out = if physics {
                tick::tick_one_with_players(p, world, &mut self.rng, players)
            } else {
                tick::tick_one_without_collision(p, world, &mut self.rng, players)
            };
            spawned.extend(out.spawn);
        }
        // Alpha (and so the bucket) only changes in the loop above: recount while compacting.
        let mut bucket_len = [0usize; BUCKETS];
        self.particles.retain(|p| {
            let keep = !p.dead;
            if keep {
                bucket_len[bucket_index(p)] += 1;
            }
            keep
        });
        self.bucket_len = bucket_len;
        self.evict_cursor = [0; BUCKETS];
        // Particles mod: a huge explosion's fireballs take the explosion group's
        // thinning and size. One `Option` test per tick while the mod has no table.
        if let Some(tuning) = self.tuning.as_deref_mut() {
            tuning.tune_children(&mut spawned);
        }
        for p in spawned.drain(..) {
            self.spawn_deferred(world, p);
        }
        self.flush_evictions();
        self.spawn_scratch = spawned;
    }

    /// Removes every particle. java: `EffectRenderer.clearEffects`
    pub fn clear(&mut self) {
        self.particles.clear();
        self.bucket_len = [0; BUCKETS];
        self.evict_cursor = [0; BUCKETS];
        self.evicted.clear();
    }

    /// Every live particle, for [`super::phase`] to batch and draw.
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        debug_assert!(
            self.evicted.is_empty(),
            "ParticleSystem::spawn_deferred without a matching flush_evictions"
        );
        &self.particles
    }

    /// Live particles per `(layer, alpha)` cap bucket — the six `fxLayers[i][j].size()`.
    #[must_use]
    pub fn bucket_counts(&self) -> [usize; BUCKETS] {
        self.bucket_len
    }

    /// Number of live particles across every bucket. java: `EffectRenderer.getStatistics`
    #[must_use]
    pub fn len(&self) -> usize {
        self.particles.len() - self.evicted.len()
    }

    /// True when no particles are live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Spawns the client-side particle half of `Explosion.doExplosionB(true)`.
    ///
    /// The server's `S27PacketExplosion` carries only the origin, strength, and affected block
    /// offsets; vanilla reconstructs the rest locally in `NetHandlerPlayClient.handleExplosion`
    /// by constructing an [`Explosion`] with `isSmoking = true` and calling `doExplosionB(true)`.
    /// The packet itself never fans these effects back over the wire.
    ///
    /// `affected_blocks` must be the absolute positions recovered from the packet's signed-byte
    /// offsets. `origin` is the packet's widened wire-float position (not a floored block
    /// position). The caller is responsible for setting this system's current particle density,
    /// just as the received-particle path does.
    ///
    /// The return value is the number of particle spawn attempts that passed range/density
    /// gating; the arena may still evict an older particle when a bucket is full. Vanilla emits
    /// one origin fireball and two particles per affected block, in that order.
    ///
    /// Java: `world/Explosion.doExplosionB` (`:176-212`) reached from
    /// `client/network/NetHandlerPlayClient.handleExplosion` (`:1078-1085`).
    pub fn spawn_explosion_effects(
        &mut self,
        world: &World,
        cam_pos: DVec3,
        origin: DVec3,
        strength: f32,
        affected_blocks: &[BlockPos],
    ) -> usize {
        mc_core::profile_scope!("particles.spawn_explosion");
        // Java's client-side Explosion constructor hard-codes isSmoking=true. The origin branch
        // chooses HUGE for strength >= 2, otherwise LARGE; the first offset is passed as the
        // large-explosion scale input, while HUGE ignores its constructor offsets.
        let origin_id = if strength >= 2.0 {
            EXPLOSION_HUGE
        } else {
            EXPLOSION_LARGE
        };
        // The particles mod's burst limit counts this burst's attempts (unused while it is off).
        let burst = 1 + 2 * affected_blocks.len();
        let mut added = 0;
        if self.spawn_explosion_particle(
            world,
            cam_pos,
            origin_id,
            origin,
            DVec3::new(1.0, 0.0, 0.0),
            burst,
        ) {
            added += 1;
        }

        for &block in affected_blocks {
            // These are deliberately drawn before either particle is constructed. That is the
            // exact order in Explosion.doExplosionB: three position nextFloat calls, then the
            // two draws in `rand.nextFloat() * rand.nextFloat() + 0.3F`, then EXPLOSION_NORMAL
            // and SMOKE_NORMAL.
            let (block_pos, velocity) = {
                let rng = self.rng_mut();
                let d0 = f64::from(block.x) + f64::from(rng.next_f32());
                let d1 = f64::from(block.y) + f64::from(rng.next_f32());
                let d2 = f64::from(block.z) + f64::from(rng.next_f32());
                let mut d3 = d0 - origin.x;
                let mut d4 = d1 - origin.y;
                let mut d5 = d2 - origin.z;
                // MathHelper.sqrt_double returns float before Java widens it back to double.
                let d6 = f64::from(mc_math::sqrt_d(d3 * d3 + d4 * d4 + d5 * d5));
                d3 /= d6;
                d4 /= d6;
                d5 /= d6;
                let mut d7 = 0.5 / (d6 / f64::from(strength) + 0.1);
                // Java's multiplication and addition are float operations, then the result is
                // widened by the `(double)` cast in doExplosionB.
                let random_scale = rng.next_f32() * rng.next_f32() + 0.3_f32;
                d7 *= f64::from(random_scale);
                (
                    DVec3::new(d0, d1, d2),
                    DVec3::new(d3 * d7, d4 * d7, d5 * d7),
                )
            };
            let midpoint = (block_pos + origin) * 0.5;

            if self.spawn_explosion_particle(
                world,
                cam_pos,
                EXPLOSION_NORMAL,
                midpoint,
                velocity,
                burst,
            ) {
                added += 1;
            }
            if self.spawn_explosion_particle(
                world,
                cam_pos,
                SMOKE_NORMAL,
                block_pos,
                velocity,
                burst,
            ) {
                added += 1;
            }
        }
        // One compaction for the whole burst (PART-1): nothing above reads the arena.
        self.flush_evictions();
        added
    }

    /// Constructs one of the four no-argument families used by an explosion after applying the
    /// same density/range gate as `RenderGlobal.spawnParticle`. `burst` is the burst's attempt
    /// count, for the particles mod's burst limit.
    fn spawn_explosion_particle(
        &mut self,
        world: &World,
        cam_pos: DVec3,
        particle_id: i32,
        pos: DVec3,
        speed: DVec3,
        burst: usize,
    ) -> bool {
        // java: `EnumParticleTypes` flags all three explosion types `shouldIgnoreRange`
        // (`EXPLOSION_NORMAL("explode", 0, true)`), so `doExplosionB`'s per-block debris spawns
        // at any distance and particle setting; only its SMOKE_NORMAL is range- and
        // density-gated (`RenderGlobal.spawnEntityFX:2084-2091`). The debris used to take the smoke's gate, so TNT beyond 16 blocks, or a
        // third of it on "Decreased", showed no debris.
        let ignore_range = matches!(
            particle_id,
            EXPLOSION_NORMAL | EXPLOSION_LARGE | EXPLOSION_HUGE
        );
        if !self.should_spawn(ignore_range, cam_pos, pos) {
            return false;
        }
        // Particles mod: every particle of an S27 burst, its smoke included, is in the
        // explosion group. Without a table this is skipped and the vanilla lines below run.
        if let Some(tuning) = self.tuning.as_deref_mut() {
            let tuned = if particle_id == SMOKE_NORMAL {
                tuning.tune(ParticleGroup::Explosion, burst)
            } else {
                tuning.tune_id(particle_id, burst)
            };
            self.spawn_tuned_deferred(world, tuned, |rng| {
                Some(explosion_particle(rng, particle_id, pos, speed))
            });
            return true;
        }
        let particle = {
            let rng = self.rng_mut();
            explosion_particle(rng, particle_id, pos, speed)
        };
        // The caller, `spawn_explosion_effects`, flushes once after the whole burst.
        self.spawn_deferred(world, particle);
        true
    }
}

/// The four no-argument families an explosion burst builds.
fn explosion_particle(
    rng: &mut JavaRandom,
    particle_id: i32,
    pos: DVec3,
    speed: DVec3,
) -> Particle {
    match particle_id {
        EXPLOSION_NORMAL => spawn::explode(rng, pos, speed),
        EXPLOSION_LARGE => spawn::explode_large(rng, pos, speed.x as f32),
        EXPLOSION_HUGE => spawn::huge_explode(rng, pos),
        SMOKE_NORMAL => spawn::smoke(rng, pos, speed, 1.0),
        _ => unreachable!("spawn_explosion_particle called with non-explosion id"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::spawn;

    #[test]
    fn explosion_emits_origin_and_two_particles_per_affected_block() {
        let world = World::new();
        let mut system = ParticleSystem::with_seed(0xE13);
        let blocks = [BlockPos::new(2, 65, -1), BlockPos::new(-3, 64, 4)];
        let origin = DVec3::new(1.25, 64.5, -2.75);

        let added = system.spawn_explosion_effects(&world, origin, origin, 4.0, &blocks);

        assert_eq!(added, 5);
        assert_eq!(system.len(), 5);
        let particles = system.particles();
        assert!(matches!(
            particles[0].kind,
            ParticleKind::HugeExplode { .. }
        ));
        assert!(matches!(particles[1].kind, ParticleKind::Explode));
        assert!(matches!(particles[2].kind, ParticleKind::Smoke { .. }));
        assert!(matches!(particles[3].kind, ParticleKind::Explode));
        assert!(matches!(particles[4].kind, ParticleKind::Smoke { .. }));
        assert_eq!(particles[0].pos, origin);

        // Position is the same jittered point for the paired debris/smoke particles, while the
        // debris particle itself is halfway back toward the origin.
        assert_eq!(particles[1].pos * 2.0 - origin, particles[2].pos);
        assert_eq!(particles[3].pos * 2.0 - origin, particles[4].pos);
        assert_ne!(particles[1].motion, DVec3::ZERO);
        assert_ne!(particles[2].motion, DVec3::ZERO);
    }

    #[test]
    fn explosion_strength_below_two_uses_large_origin_particle_and_empty_is_one() {
        let world = World::new();
        let mut system = ParticleSystem::with_seed(17);
        let origin = DVec3::new(10.0, 70.0, 3.0);

        assert_eq!(
            system.spawn_explosion_effects(&world, DVec3::ZERO, origin, 1.999, &[]),
            1
        );
        assert!(matches!(
            system.particles()[0].kind,
            ParticleKind::LargeExplode { .. }
        ));

        // Both origin families carry the vanilla ignore-range flag. Even with minimal density
        // and a camera 400 blocks away, the origin particle still survives; this catches a
        // mutant that routes the origin through the normal range gate.
        let mut minimal = ParticleSystem::with_seed(17);
        minimal.set_density(ParticleDensity::Minimal);
        assert_eq!(
            minimal.spawn_explosion_effects(&world, DVec3::ZERO, origin, 4.0, &[]),
            1
        );
    }

    #[test]
    fn density_all_never_demotes() {
        let mut sys = ParticleSystem::with_seed(1);
        sys.set_density(ParticleDensity::All);
        for _ in 0..200 {
            assert!(sys.should_spawn(false, DVec3::ZERO, DVec3::ZERO));
        }
    }

    #[test]
    fn density_minimal_blocks_ranged_particles_but_not_ignore_range_ones() {
        let mut sys = ParticleSystem::with_seed(1);
        sys.set_density(ParticleDensity::Minimal);
        assert!(!sys.should_spawn(false, DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)));
        assert!(sys.should_spawn(true, DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn decreased_density_spawns_roughly_two_thirds_of_the_time() {
        let mut sys = ParticleSystem::with_seed(7);
        sys.set_density(ParticleDensity::Decreased);
        let mut spawned = 0;
        const N: u32 = 20_000;
        for _ in 0..N {
            if sys.should_spawn(false, DVec3::ZERO, DVec3::ZERO) {
                spawned += 1;
            }
        }
        let frac = f64::from(spawned) / f64::from(N);
        assert!((frac - 2.0 / 3.0).abs() < 0.02, "{frac}");
    }

    #[test]
    fn range_check_is_the_16_block_radius() {
        let mut sys = ParticleSystem::with_seed(1);
        sys.set_density(ParticleDensity::All);
        assert!(sys.should_spawn(false, DVec3::ZERO, DVec3::new(15.9, 0.0, 0.0)));
        assert!(!sys.should_spawn(false, DVec3::ZERO, DVec3::new(16.1, 0.0, 0.0)));
    }

    #[test]
    fn spawn_and_tick_round_trip_and_expiry_removes_particles() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(3);
        let mut r = JavaRandom::new(3);
        let mut p = spawn::generic(&mut r, DVec3::new(0.0, 90.0, 0.0), DVec3::ZERO);
        p.max_age = 2;
        sys.spawn(&world, p);
        assert_eq!(sys.len(), 1);
        // `age++ >= maxAge` reads the *old* age, so with `max_age = 2` death lands on the 3rd
        // tick (old age 0, 1, then 2 >= 2).
        sys.tick(&world);
        assert_eq!(sys.len(), 1);
        sys.tick(&world);
        assert_eq!(sys.len(), 1);
        sys.tick(&world);
        assert_eq!(
            sys.len(),
            0,
            "particle should have expired and been removed"
        );
    }

    #[test]
    fn tick_with_players_forwards_the_world_snapshot_to_clouds() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(31);
        let mut r = JavaRandom::new(31);
        let mut cloud = spawn::cloud(&mut r, DVec3::new(0.0, 3.0, 0.0), DVec3::ZERO);
        cloud.motion = DVec3::ZERO;
        cloud.max_age = 100;
        sys.spawn(&world, cloud);

        sys.tick_with_players(
            &world,
            &[ParticlePlayer {
                pos: DVec3::new(0.0, 2.0, 0.0),
                velocity: DVec3::new(0.0, 3.0, 0.0),
            }],
        );

        let cloud = &sys.particles()[0];
        assert!((cloud.pos.y - 2.8).abs() < 1e-12);
        assert!((cloud.motion.y - 0.6).abs() < 1e-12);
    }

    #[test]
    fn crit_particles_are_primed_with_one_tick_at_spawn() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(5);
        let mut r = JavaRandom::new(5);
        let p = spawn::crit(&mut r, DVec3::new(0.0, 90.0, 0.0), DVec3::ZERO, 1.0, false);
        sys.spawn(&world, p);
        assert_eq!(
            sys.particles()[0].age,
            1,
            "constructor calls onUpdate() once"
        );
    }

    /// A burst far past the cap (one S2A packet can carry any `i32` count) keeps the arena
    /// bounded while it runs, and still ends exactly where sequential spawns end.
    #[test]
    fn a_huge_deferred_burst_keeps_the_arena_bounded_and_exact() {
        let world = World::new();
        let mut immediate = ParticleSystem::with_seed(9);
        let mut deferred = ParticleSystem::with_seed(9);
        let mut r = JavaRandom::new(9);
        for i in 0..(6 * super::PER_BUCKET_CAP) {
            let mut p = spawn::generic(&mut r, DVec3::new(i as f64, 90.0, 0.0), DVec3::ZERO);
            p.max_age = 10_000; // never expires during this test
            immediate.spawn(&world, p.clone());
            deferred.spawn_deferred(&world, p);
            assert!(
                deferred.particles.len() <= 2 * super::PER_BUCKET_CAP,
                "arena grew to {} during the burst",
                deferred.particles.len()
            );
        }
        deferred.flush_evictions();
        assert_eq!(immediate.particles(), deferred.particles());
        assert_eq!(deferred.bucket_counts(), recount(deferred.particles()));
    }

    #[test]
    fn per_bucket_cap_evicts_the_oldest_matching_particle() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(9);
        let mut r = JavaRandom::new(9);
        for i in 0..(super::PER_BUCKET_CAP + 5) {
            let mut p = spawn::generic(&mut r, DVec3::new(i as f64, 90.0, 0.0), DVec3::ZERO);
            p.max_age = 10_000; // never expires during this test
            sys.spawn(&world, p);
        }
        assert_eq!(sys.len(), super::PER_BUCKET_CAP);
        // The five oldest (x = 0..5) should have been evicted first.
        assert!(sys.particles().iter().all(|p| p.pos.x >= 5.0));
    }

    #[test]
    fn huge_explode_spawn_chain_stays_within_the_system() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(11);
        let mut r = JavaRandom::new(11);
        let p = spawn::huge_explode(&mut r, DVec3::new(0.0, 80.0, 0.0));
        sys.spawn(&world, p);
        sys.tick(&world);
        assert!(
            sys.len() >= 6,
            "the huge-explode particle plus its 6 spawned children"
        );
    }

    /// The pre-PART-1 arena, kept verbatim as the parity oracle: a full-arena count per spawn
    /// and `Vec::remove` eviction of the first same-bucket particle.
    struct ReferenceArena {
        rng: JavaRandom,
        particles: Vec<Particle>,
    }

    impl ReferenceArena {
        fn bucket_of(p: &Particle) -> (u8, AlphaBucket) {
            (layer_slot(p.layer), p.alpha_bucket())
        }

        fn spawn(&mut self, world: &World, p: Particle) {
            let key = Self::bucket_of(&p);
            if self
                .particles
                .iter()
                .filter(|q| Self::bucket_of(q) == key)
                .count()
                >= PER_BUCKET_CAP
            {
                if let Some(idx) = self
                    .particles
                    .iter()
                    .position(|q| Self::bucket_of(q) == key)
                {
                    self.particles.remove(idx);
                }
            }
            let prime = matches!(p.kind, ParticleKind::Crit { .. });
            self.particles.push(p);
            if prime {
                let idx = self.particles.len() - 1;
                let mut particle = self.particles.remove(idx);
                tick::tick_one(&mut particle, world, &mut self.rng);
                self.particles.push(particle);
            }
        }

        fn tick(&mut self, world: &World) {
            let mut spawned = Vec::new();
            for p in &mut self.particles {
                if p.dead {
                    continue;
                }
                let out = tick::tick_one_with_players(p, world, &mut self.rng, &[]);
                spawned.extend(out.spawn);
            }
            self.particles.retain(|p| !p.dead);
            for p in spawned {
                self.spawn(world, p);
            }
        }
    }

    /// A deterministic mixed stream: misc opaque/translucent, terrain, footprints (sharing the
    /// lit-explosion cap bucket), primed crits, and fading trail sparks that migrate between
    /// alpha buckets during `tick` and spawn children.
    fn mixed_particle(g: &mut JavaRandom, i: usize) -> Particle {
        let pos = DVec3::new(
            f64::from(g.next_f32()) * 8.0,
            90.0,
            f64::from(g.next_f32()) * 8.0,
        );
        let mut p = match i % 11 {
            0 => spawn::crit(g, pos, DVec3::ZERO, 1.0, false),
            1 => spawn::footstep(g, pos),
            2 => spawn::firework_spark(g, pos, DVec3::ZERO, [1.0, 0.5, 0.25], None, true, false),
            3 => {
                let mut p = spawn::generic(g, pos, DVec3::ZERO);
                p.layer = FxLayer::Terrain;
                p
            }
            4 => {
                let mut p = spawn::generic(g, pos, DVec3::ZERO);
                p.color[3] = 0.5;
                p
            }
            _ => spawn::generic(g, pos, DVec3::ZERO),
        };
        if !matches!(
            p.kind,
            ParticleKind::FireworkSpark { .. } | ParticleKind::Crit { .. }
        ) {
            // Long-lived so the misc-opaque bucket reaches the cap and keeps evicting.
            p.max_age = 2_000 + (i as u32 % 7);
        }
        p
    }

    fn recount(particles: &[Particle]) -> [usize; BUCKETS] {
        let mut counts = [0; BUCKETS];
        for p in particles {
            counts[bucket_index(p)] += 1;
        }
        counts
    }

    #[test]
    fn o1_cap_bookkeeping_matches_the_linear_scan_arena_element_for_element() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(0x5EED);
        let mut reference = ReferenceArena {
            rng: JavaRandom::new(0x5EED),
            particles: Vec::new(),
        };
        let mut g = JavaRandom::new(99);
        let mut i = 0usize;
        // Rounds of spawning past the 4000 cap, with ticks in between (bucket migration,
        // expiry, children re-entering the cap), alternating immediate and deferred spawns.
        // ~7/11 of the stream is misc-opaque, so rounds 2 (immediate) and 3 (deferred) evict.
        for round in 0..4 {
            for _ in 0..2_300 {
                let p = mixed_particle(&mut g, i);
                i += 1;
                reference.spawn(&world, p.clone());
                if round % 2 == 0 {
                    sys.spawn(&world, p);
                } else {
                    sys.spawn_deferred(&world, p);
                }
            }
            sys.flush_evictions();
            assert_eq!(
                sys.particles(),
                &reference.particles[..],
                "round {round} spawn"
            );
            assert_eq!(sys.bucket_counts(), recount(sys.particles()));
            for _ in 0..3 {
                reference.tick(&world);
                sys.tick(&world);
            }
            assert_eq!(
                sys.particles(),
                &reference.particles[..],
                "round {round} tick"
            );
            assert_eq!(sys.bucket_counts(), recount(sys.particles()));
            assert_eq!(sys.len(), reference.particles.len());
        }
        assert!(
            sys.bucket_counts().contains(&PER_BUCKET_CAP),
            "the stream must actually exercise eviction"
        );
    }

    /// PART-1 timing: a 1,200-spawn TNT-sized burst into 300 live particles, then into a full
    /// bucket, old arena vs new. Numbers depend on profile and machine; printed, not asserted.
    #[test]
    #[ignore = "microbenchmark: run with --include-ignored --nocapture"]
    fn microbench_part1_burst_spawn() {
        use std::time::Instant;
        let world = World::new();
        let mut g = JavaRandom::new(1);
        let mut make = |n: usize| -> Vec<Particle> {
            (0..n)
                .map(|i| {
                    let mut p =
                        spawn::generic(&mut g, DVec3::new(i as f64, 90.0, 0.0), DVec3::ZERO);
                    p.max_age = 10_000;
                    p
                })
                .collect()
        };
        let (live, burst, fill) = (make(300), make(1_200), make(PER_BUCKET_CAP));
        for (label, prefill) in [("300 live", &live), ("full bucket", &fill)] {
            let mut reference = ReferenceArena {
                rng: JavaRandom::new(2),
                particles: Vec::new(),
            };
            let mut immediate = ParticleSystem::with_seed(2);
            let mut deferred = ParticleSystem::with_seed(2);
            for p in prefill {
                reference.spawn(&world, p.clone());
                immediate.spawn(&world, p.clone());
                deferred.spawn(&world, p.clone());
            }
            let (b1, b2, b3) = (burst.clone(), burst.clone(), burst.clone());
            let t = Instant::now();
            for p in b1 {
                reference.spawn(&world, p);
            }
            let old = t.elapsed();
            let t = Instant::now();
            for p in b2 {
                immediate.spawn(&world, p);
            }
            let new_immediate = t.elapsed();
            let t = Instant::now();
            for p in b3 {
                deferred.spawn_deferred(&world, p);
            }
            deferred.flush_evictions();
            let new_deferred = t.elapsed();
            assert_eq!(deferred.particles(), &reference.particles[..]);
            eprintln!(
                "PART-1 burst of 1200 into {label}: old {old:?}, new spawn() {new_immediate:?}, \
                 new spawn_deferred+flush {new_deferred:?}"
            );
        }
    }

    #[test]
    fn deferred_burst_equals_immediate_spawns_and_keeps_counts_exact() {
        let world = World::new();
        let mut immediate = ParticleSystem::with_seed(4);
        let mut deferred = ParticleSystem::with_seed(4);
        let mut g = JavaRandom::new(5);
        // ~7/11 misc-opaque: ~5,000 into that bucket, so ~1,000 deferred evictions.
        for i in 0..(2 * PER_BUCKET_CAP) {
            let p = mixed_particle(&mut g, i);
            immediate.spawn(&world, p.clone());
            deferred.spawn_deferred(&world, p);
            assert_eq!(immediate.len(), deferred.len());
        }
        deferred.flush_evictions();
        assert_eq!(immediate.particles(), deferred.particles());
        assert_eq!(deferred.bucket_counts(), recount(deferred.particles()));
        deferred.clear();
        assert_eq!(deferred.bucket_counts(), [0; BUCKETS]);
        assert!(deferred.is_empty());
    }

    /// An explosion's debris (`EXPLOSION_NORMAL`) ignores range and the particle
    /// setting as in vanilla; only its smoke is gated. Fails on the old gate, which dropped the
    /// debris together with the smoke.
    #[test]
    fn explosion_debris_ignores_range_and_density_but_its_smoke_does_not() {
        let world = World::new();
        let origin = DVec3::new(0.5, 70.0, 0.5);
        let blocks = [BlockPos::new(1, 70, 0), BlockPos::new(0, 71, 1)];
        let far_camera = DVec3::new(400.0, 70.0, 0.0);
        for density in [
            ParticleDensity::All,
            ParticleDensity::Decreased,
            ParticleDensity::Minimal,
        ] {
            let mut sys = ParticleSystem::with_seed(21);
            sys.set_density(density);
            let added = sys.spawn_explosion_effects(&world, far_camera, origin, 4.0, &blocks);
            assert_eq!(added, 3, "{density:?}: the emitter and both debris");
            let kinds: Vec<_> = sys.particles().iter().map(|p| p.kind.clone()).collect();
            assert!(matches!(kinds[0], ParticleKind::HugeExplode { .. }));
            assert!(matches!(kinds[1], ParticleKind::Explode));
            assert!(matches!(kinds[2], ParticleKind::Explode));
        }
        // Up close with every particle on, the smoke comes back: 1 + 2 per block.
        let mut sys = ParticleSystem::with_seed(21);
        assert_eq!(
            sys.spawn_explosion_effects(&world, origin, origin, 4.0, &blocks),
            5
        );
    }

    // ---- "Use particle physics"---------------------------

    /// A 17x17 stone floor whose top face is at `y + 1`, with its chunks loaded first
    /// (`World::set_block` is a no-op in an unloaded chunk).
    fn stone_floor(world: &mut World, y: i32) {
        let stone = mc_data::BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
        for x in -8..=8 {
            for z in -8..=8 {
                let pos = BlockPos::new(x, y, z);
                if !world.is_chunk_loaded(pos.chunk_pos()) {
                    let len = mc_world::expected_chunk_data_len(1, world.has_sky_light(), true);
                    world
                        .apply_chunk_data(pos.chunk_pos(), true, 1, &vec![0u8; len])
                        .expect("a zeroed payload of the exact expected length decodes");
                }
                world.set_block(pos, stone);
            }
        }
    }

    fn falling_block_bit(seed: i64) -> Particle {
        let icon = super::super::types::ParticleIcon {
            min_u: 0.0,
            max_u: 1.0,
            min_v: 0.0,
            max_v: 1.0,
        };
        let mut r = JavaRandom::new(seed);
        let mut p = spawn::digging(
            &mut r,
            DVec3::new(0.5, 64.5, 0.5),
            DVec3::ZERO,
            icon,
            spawn::DEFAULT_DIGGING_TINT,
            false,
            BlockPos::new(0, 64, 0),
        );
        p.max_age = 1_000; // outlives the test
        p
    }

    /// With physics on (the default, vanilla) a block-break particle lands on a stone floor;
    /// with it off the same particle falls through the floor and is never on the ground. Its own
    /// `no_clip` flag is untouched, and the crit priming tick at spawn follows the setting too.
    #[test]
    fn particle_physics_off_lets_particles_fall_through_blocks() {
        let mut world = World::new();
        stone_floor(&mut world, 63); // top face at y = 64
        assert!(ParticleSystem::new().physics(), "vanilla is the default");
        let mut vanilla = ParticleSystem::with_seed(4);
        let mut off = ParticleSystem::with_seed(4);
        off.set_physics(false);
        assert!(!off.physics());
        vanilla.spawn(&world, falling_block_bit(4));
        off.spawn(&world, falling_block_bit(4));
        for _ in 0..40 {
            vanilla.tick(&world);
            off.tick(&world);
        }
        let landed = &vanilla.particles()[0];
        assert!(landed.on_ground, "{landed:?}");
        assert!((landed.pos.y - 64.0).abs() < 1e-9, "{}", landed.pos.y);
        let fell = &off.particles()[0];
        assert!(!fell.on_ground, "{fell:?}");
        assert!(fell.pos.y < 60.0, "fell through the floor: {}", fell.pos.y);
        assert!(
            !fell.no_clip,
            "the particle's own flag is restored after each tick"
        );

        // A crit is primed with one tick inside `spawn`; that tick obeys the setting as well.
        let crit = |seed| {
            let mut r = JavaRandom::new(seed);
            spawn::crit(
                &mut r,
                DVec3::new(0.5, 64.1, 0.5),
                DVec3::new(0.0, -1.25, 0.0),
                1.0,
                false,
            )
        };
        let mut vanilla = ParticleSystem::with_seed(6);
        let mut off = ParticleSystem::with_seed(6);
        off.set_physics(false);
        vanilla.spawn(&world, crit(6));
        off.spawn(&world, crit(6));
        assert!(
            (vanilla.particles()[0].pos.y - 64.0).abs() < 1e-9,
            "{}",
            vanilla.particles()[0].pos.y
        );
        assert!(
            off.particles()[0].pos.y < 63.8,
            "{}",
            off.particles()[0].pos.y
        );

        // Back on: the next tick collides again (the particle below the floor stays there, but
        // one dropped onto the floor lands as in vanilla).
        off.set_physics(true);
        off.clear();
        off.spawn(&world, falling_block_bit(4));
        for _ in 0..40 {
            off.tick(&world);
        }
        assert!(off.particles()[0].on_ground);
    }

    /// Timing: one tick of a heavy fight's particle set (TNT debris and smoke, block
    /// debris on a floor, crits, potion swirls), with physics on (vanilla) and off. Printed,
    /// not asserted; run optimised (`--profile perf-ab`) for meaningful numbers.
    #[test]
    #[ignore = "microbenchmark: run with --include-ignored --nocapture"]
    fn microbench_f6_heavy_fight_tick() {
        use std::time::Instant;
        let mut world = World::new();
        stone_floor(&mut world, 63); // top face at y = 64
        let mut sys = ParticleSystem::with_seed(0xF6);
        let mut r = JavaRandom::new(0xF6);
        let blocks: Vec<BlockPos> = (0..245)
            .map(|i| BlockPos::new(i % 7 - 3, 65 + (i / 49) % 5, (i / 7) % 7 - 3))
            .collect();
        let origin = DVec3::new(0.5, 66.5, 0.5);
        sys.spawn_explosion_effects(&world, origin, origin, 4.0, &blocks);
        for i in 0..6 {
            let mut bit = falling_block_bit(i);
            bit.pos.x += f64::from(i as i32 % 3) - 1.0;
            for _ in 0..64 {
                let mut p = bit.clone();
                p.motion = DVec3::new(
                    f64::from(r.next_f32()) * 0.2 - 0.1,
                    f64::from(r.next_f32()) * 0.2,
                    f64::from(r.next_f32()) * 0.2 - 0.1,
                );
                sys.spawn(&world, p);
            }
        }
        for _ in 0..200 {
            let p = spawn::crit(&mut r, origin, DVec3::new(0.3, 0.1, -0.2), 1.0, true);
            sys.spawn(&world, p);
        }
        for _ in 0..200 {
            let p = spawn::spell(
                &mut r,
                origin,
                DVec3::new(0.0, 0.1, 0.0),
                spawn::SpellVariant::Normal,
            );
            sys.spawn(&world, p);
        }
        // Let the debris land and the set settle into a typical mid-burst state.
        for _ in 0..4 {
            sys.tick(&world);
        }
        let live = sys.len();
        const N: u32 = 200;
        for (label, physics) in [("physics on", true), ("physics off", false)] {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..N {
                let mut copy = sys.clone();
                copy.set_physics(physics);
                let t = Instant::now();
                copy.tick(&world);
                total += t.elapsed();
                std::hint::black_box(copy.len());
            }
            let per_tick = total / N;
            eprintln!(
                "F6 heavy-fight tick, {label}: {per_tick:?} per tick for {live} \
                 particles ({:.1} ns each)",
                per_tick.as_nanos() as f64 / live as f64
            );
        }
    }

    /// Timing: vanilla's 1,000-sample ambient display tick
    /// (`World::do_void_fog_particles_with_rain`, the client's `client.void_fog_particles`
    /// scope) over an island-like column set. Printed, not asserted; run with
    /// `--profile perf-ab`.
    #[test]
    #[ignore = "microbenchmark: run with --include-ignored --nocapture"]
    fn microbench_f6_ambient_display_tick() {
        use std::time::Instant;
        let mut world = World::new();
        let stone = mc_data::BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
        let torch = mc_data::BlockState::new(mc_data::block_by_name("torch").unwrap(), 5);
        for cx in -2..=1 {
            for cz in -2..=1 {
                let pos = BlockPos::new(cx * 16, 0, cz * 16);
                let len = mc_world::expected_chunk_data_len(1, world.has_sky_light(), true);
                world
                    .apply_chunk_data(pos.chunk_pos(), true, 1, &vec![0u8; len])
                    .expect("blank chunk");
            }
        }
        // An island: stone from y 58 to 73 within 12 blocks, a few walls and torches on top.
        for x in -12..=12 {
            for z in -12..=12 {
                for y in 58..=73 {
                    world.set_block(BlockPos::new(x, y, z), stone);
                }
                if (x + z) % 7 == 0 {
                    world.set_block(BlockPos::new(x, 74, z), stone);
                    world.set_block(BlockPos::new(x, 75, z), stone);
                }
                if x % 6 == 0 && z % 6 == 0 {
                    world.set_block(BlockPos::new(x, 74, z), torch);
                }
            }
        }
        let origin = BlockPos::new(0, 75, 0);
        let mut sample_rng = JavaRandom::new(1);
        let mut callback_rng = JavaRandom::new(2);
        const N: u32 = 2_000;
        let mut effects = 0usize;
        let t = Instant::now();
        for _ in 0..N {
            let out = world.do_void_fog_particles_with_rain(
                origin,
                &mut sample_rng,
                &mut callback_rng,
                false,
                false,
            );
            effects += out.len();
        }
        let per_call = t.elapsed() / N;
        eprintln!(
            "F6 ambient display tick: {per_call:?} per call (1,000 samples), {:.2} effects/call",
            effects as f64 / f64::from(N)
        );
        // The live client runs it once per tick after a frame's worth of rendering, so its world
        // data starts cold: evict the caches (64 MiB of writes) before each timed call.
        let mut evict = vec![0u8; 64 << 20];
        let mut cold = std::time::Duration::ZERO;
        const COLD: u32 = 200;
        for i in 0..COLD {
            for chunk in evict.chunks_mut(64) {
                chunk[0] = chunk[0].wrapping_add(i as u8);
            }
            std::hint::black_box(&evict);
            let t = Instant::now();
            let out = world.do_void_fog_particles_with_rain(
                origin,
                &mut sample_rng,
                &mut callback_rng,
                false,
                false,
            );
            cold += t.elapsed();
            std::hint::black_box(out.len());
        }
        eprintln!(
            "F6 ambient display tick, cold caches: {:?} per call",
            cold / COLD
        );

        // A built-up spot (a fortress, a BedWars base): nearly every sampled cell is a block.
        let planks = mc_data::BlockState::new(mc_data::block_by_name("planks").unwrap(), 0);
        let glass = mc_data::BlockState::new(mc_data::block_by_name("glass").unwrap(), 0);
        for x in -15i32..=15 {
            for z in -15i32..=15 {
                for y in 60i32..=90 {
                    let state = match (x + y + z).rem_euclid(5) {
                        0 => planks,
                        1 => glass,
                        _ => stone,
                    };
                    world.set_block(BlockPos::new(x, y, z), state);
                }
            }
        }
        let t = Instant::now();
        for _ in 0..N {
            let out = world.do_void_fog_particles_with_rain(
                origin,
                &mut sample_rng,
                &mut callback_rng,
                false,
                false,
            );
            std::hint::black_box(out.len());
        }
        eprintln!(
            "F6 ambient display tick, solid surroundings: {:?} per call",
            t.elapsed() / N
        );
    }

    /// Away from blocks the setting changes nothing at all: the same spawns, RNG draws,
    /// positions, motion and lifetimes, bit for bit, tick after tick (only collision differs).
    #[test]
    fn particle_physics_off_is_bit_identical_in_open_air() {
        let world = World::new();
        let mut on = ParticleSystem::with_seed(11);
        let mut off = ParticleSystem::with_seed(11);
        off.set_physics(false);
        let origin = DVec3::new(0.5, 200.0, 0.5);
        let blocks: Vec<BlockPos> = (0..40)
            .map(|i| BlockPos::new(i % 7 - 3, 200 + i / 7, i % 5 - 2))
            .collect();
        for system in [&mut on, &mut off] {
            system.spawn_explosion_effects(&world, origin, origin, 4.0, &blocks);
            let mut r = JavaRandom::new(3);
            for _ in 0..16 {
                let p = spawn::crit(&mut r, origin, DVec3::new(0.3, 0.2, -0.1), 1.0, true);
                system.spawn(&world, p);
            }
        }
        assert!(on.len() > 80);
        assert_eq!(on.particles(), off.particles());
        for _ in 0..60 {
            on.tick(&world);
            off.tick(&world);
            assert_eq!(on.particles(), off.particles());
        }
    }
}
