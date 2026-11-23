//! The particles mod: per-group **amount** and render **size** for
//! the particle families PvP players care about, plus the extra crit and sharpness emitters a
//! local hit adds ([`HitEmitters`]).
//!
//! # Off means vanilla
//!
//! [`super::system::ParticleSystem`] keeps an `Option<Box<ParticleTuning>>` that is `None` while
//! the mod is off **or** every group is at its vanilla values. Each spawn path tests it once per
//! gated spawn attempt and, when it is `None`, runs its unchanged vanilla branch, so the particle
//! system's `JavaRandom` stream and the spawn sequence are exactly what they were before the mod
//! existed. [`HitEmitters`] (on the client's session) is a single `bool` test per local attack
//! while it adds nothing.
//!
//! # Where the table applies
//!
//! On every spawn attempt that has **passed** vanilla's density and range gate
//! (`RenderGlobal.spawnParticle`, [`super::system::ParticleSystem::should_spawn`]): the
//! client-generated funnel ([`super::packet::spawn_one_with`]), S2A packets
//! ([`super::packet::apply`]), the potion swirls the client ticks itself, the S27 explosion burst
//! ([`super::system::ParticleSystem::spawn_explosion_effects`]) and the two ungated block effects
//! ([`super::packet::add_block_destroy_effects`], [`super::packet::add_block_hit_effects`]).
//! The particles setting "Minimal" therefore still hides what vanilla hides: the mod never
//! brings a gated-out particle back.
//!
//! # Amount
//!
//! A gated attempt of a group with amount `a` yields `floor(a)` particles plus one more with
//! probability `frac(a)`. That coin is rolled on the mod's own [`ModRng`], never on the particle
//! system's `JavaRandom`. `0` hides the group; `1` is vanilla. Each copy is a fresh build from the
//! system RNG, so copies get their own jitter, motion and lifetime and spread the way vanilla's
//! own particles do. Crits and sharpness are the exception above `1`: their amount is the number
//! of emitters each hit **you** land spawns ([`HitEmitters`]); at the funnel they only thin
//! (`a < 1`) or hide (`a = 0`), from every source.
//!
//! # Size
//!
//! Render scale only: [`scale_render`] multiplies `Particle::scale` and the kind's ramp input
//! (`peak_scale`, or `scale_factor` for the large explosion), which only
//! [`super::billboard`]'s `effective_scale` reads. The collision box (`Particle::size`), the
//! lifetime and every tick are untouched.
//!
//! # The particle cap
//!
//! Every copy enters through `ParticleSystem::spawn_deferred`, so vanilla's 4000-per-bucket
//! `EffectRenderer.addEffect` cap (evict the oldest) holds whatever the amount. In one burst
//! (an S2A packet's `count`, an S27 explosion) the multiplier is additionally limited so the
//! burst never builds more than [`BURST_LIMIT`] particles that it would only evict again.

use super::packet;
use super::types::{Particle, ParticleKind, PER_BUCKET_CAP};

/// One burst never multiplies past this many particles: [`PER_BUCKET_CAP`] is all a bucket can
/// hold, so anything beyond it would only evict the burst's own earlier particles.
pub const BURST_LIMIT: usize = PER_BUCKET_CAP;

/// The particle groups the menu edits, in menu order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleGroup {
    /// `CRIT` (9): critical-hit stars.
    Crit,
    /// `CRIT_MAGIC` (10): the blue sharpness (enchantment) sparkles.
    Sharp,
    /// `SPELL_MOB` (15) and `SPELL_MOB_AMBIENT` (16): potion-effect swirls around entities.
    Swirl,
    /// `SPELL` (13) and `SPELL_INSTANT` (14): splash-potion clouds.
    Splash,
    /// `LAVA` (27) pops and `DRIP_LAVA` (19) drips.
    Lava,
    /// Block debris from `addBlockDestroyEffects` and `addBlockHitEffects` (not sprint dust).
    Block,
    /// `EXPLOSION_NORMAL`/`_LARGE`/`_HUGE` (0-2), plus every particle of an S27 explosion burst.
    /// Vanilla also uses `EXPLOSION_NORMAL` for the small poof of a dying mob or player.
    Explosion,
}

impl ParticleGroup {
    /// How many groups there are.
    pub const COUNT: usize = 7;

    /// Every group, in menu order (`ALL[g.index()] == g`).
    pub const ALL: [ParticleGroup; ParticleGroup::COUNT] = [
        ParticleGroup::Crit,
        ParticleGroup::Sharp,
        ParticleGroup::Swirl,
        ParticleGroup::Splash,
        ParticleGroup::Lava,
        ParticleGroup::Block,
        ParticleGroup::Explosion,
    ];

    /// Position in [`ParticleGroup::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable short name (the settings table's key).
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            ParticleGroup::Crit => "crit",
            ParticleGroup::Sharp => "sharp",
            ParticleGroup::Swirl => "swirl",
            ParticleGroup::Splash => "splash",
            ParticleGroup::Lava => "lava",
            ParticleGroup::Block => "block",
            ParticleGroup::Explosion => "explosion",
        }
    }

    /// The group a vanilla particle type (wire id) belongs to, if any. Block debris has no id of
    /// its own (`BLOCK_CRACK` is also sprint dust), so [`ParticleGroup::Block`] is never returned.
    #[must_use]
    pub const fn of_particle_id(particle_id: i32) -> Option<ParticleGroup> {
        match particle_id {
            packet::CRIT => Some(ParticleGroup::Crit),
            packet::CRIT_MAGIC => Some(ParticleGroup::Sharp),
            packet::SPELL_MOB | packet::SPELL_MOB_AMBIENT => Some(ParticleGroup::Swirl),
            packet::SPELL | packet::SPELL_INSTANT => Some(ParticleGroup::Splash),
            packet::LAVA | packet::DRIP_LAVA => Some(ParticleGroup::Lava),
            packet::EXPLOSION_NORMAL | packet::EXPLOSION_LARGE | packet::EXPLOSION_HUGE => {
                Some(ParticleGroup::Explosion)
            }
            _ => None,
        }
    }
}

/// One group's settings, as multipliers: `amount` 1.0 and `size` 1.0 are vanilla.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupTuning {
    /// Particles per gated spawn attempt (`0.0` hides the group). For crits and sharpness:
    /// emitters per hit you land.
    pub amount: f32,
    /// Render-scale multiplier.
    pub size: f32,
}

impl GroupTuning {
    /// Vanilla: one particle per attempt, at its own size.
    pub const VANILLA: GroupTuning = GroupTuning {
        amount: 1.0,
        size: 1.0,
    };
}

impl Default for GroupTuning {
    fn default() -> Self {
        GroupTuning::VANILLA
    }
}

/// The mod's own random stream (xorshift64*). It decides fractional amounts, so the particle
/// system's `JavaRandom` never sees an extra draw for a decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModRng(u64);

impl ModRng {
    /// A stream from `seed` (a zero seed is replaced: xorshift has no zero state).
    #[must_use]
    pub const fn new(seed: u64) -> ModRng {
        ModRng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        // The top 24 bits of the scrambled state: exactly representable, never 1.0.
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / 16_777_216.0
    }

    /// `floor(amount)` plus one more with probability `frac(amount)`: an integer count whose
    /// mean is `amount`. No draw for a whole number; `0` for zero, negative or NaN.
    pub fn count(&mut self, amount: f32) -> u32 {
        if amount.is_nan() || amount <= 0.0 {
            return 0;
        }
        let whole = amount.floor();
        let frac = amount - whole;
        let mut n = whole as u32;
        if frac > 0.0 && self.next_f32() < frac {
            n += 1;
        }
        n
    }
}

/// What the table does with one gated spawn attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tuned {
    /// Particles to build (each a fresh build from the system RNG). `0` drops the attempt.
    pub copies: u32,
    /// Render-scale multiplier for each of them ([`scale_render`]).
    pub size: f32,
}

impl Tuned {
    /// Exactly vanilla: one particle at its own size.
    pub const ONE: Tuned = Tuned {
        copies: 1,
        size: 1.0,
    };
}

/// Multiply a particle's **render** scale by `size`: `Particle::scale` and the ramp input of the
/// kinds whose on-screen size is computed from one (see `billboard::effective_scale`). The
/// collision box, the lifetime and the motion are left alone, and nothing in `tick` or `physics`
/// reads either field, so the simulation is unchanged. A no-op for `1.0`.
pub fn scale_render(particle: &mut Particle, size: f32) {
    if size == 1.0 {
        return;
    }
    particle.scale *= size;
    match &mut particle.kind {
        ParticleKind::Smoke { peak_scale }
        | ParticleKind::Crit { peak_scale }
        | ParticleKind::Flame { peak_scale }
        | ParticleKind::Lava { peak_scale }
        | ParticleKind::Reddust { peak_scale }
        | ParticleKind::Cloud { peak_scale }
        | ParticleKind::Heart { peak_scale }
        | ParticleKind::Note { peak_scale }
        | ParticleKind::SnowShovel { peak_scale } => *peak_scale *= size,
        ParticleKind::LargeExplode { scale_factor, .. } => *scale_factor *= size,
        _ => {}
    }
}

/// The per-group table the particle system applies at its spawn funnel (see the module docs).
#[derive(Debug, Clone)]
pub struct ParticleTuning {
    /// Funnel amount per group: the menu amount, except crits and sharpness, which are capped at
    /// `1.0` here (above that they are emitters per hit, handled by [`HitEmitters`]).
    amount: [f32; ParticleGroup::COUNT],
    /// Render size per group.
    size: [f32; ParticleGroup::COUNT],
    rng: ModRng,
}

impl ParticleTuning {
    /// The seed of the decision stream. Fixed, so a test (or a replay with the mod on) is
    /// reproducible; what it decides is cosmetic either way.
    const SEED: u64 = 0x5041_5254_4943_4C45;

    /// A table from the menu's per-group values.
    #[must_use]
    pub fn new(groups: [GroupTuning; ParticleGroup::COUNT]) -> ParticleTuning {
        let mut tuning = ParticleTuning {
            amount: [1.0; ParticleGroup::COUNT],
            size: [1.0; ParticleGroup::COUNT],
            rng: ModRng::new(Self::SEED),
        };
        tuning.set_groups(groups);
        tuning
    }

    /// Replace the per-group values, keeping the decision stream going.
    pub fn set_groups(&mut self, groups: [GroupTuning; ParticleGroup::COUNT]) {
        for group in ParticleGroup::ALL {
            let i = group.index();
            self.amount[i] = Self::funnel_amount(group, groups[i].amount);
            self.size[i] = sanitize_size(groups[i].size);
        }
    }

    fn funnel_amount(group: ParticleGroup, amount: f32) -> f32 {
        let amount = sanitize_amount(amount);
        match group {
            ParticleGroup::Crit | ParticleGroup::Sharp => amount.min(1.0),
            _ => amount,
        }
    }

    /// Whether these menu values change nothing at the funnel, so the particle system can keep
    /// no table at all. True for the mod's defaults (crits and sharpness above 100% only add
    /// emitters per hit, which the funnel does not see).
    #[must_use]
    pub fn is_vanilla(groups: &[GroupTuning; ParticleGroup::COUNT]) -> bool {
        ParticleGroup::ALL.iter().all(|&group| {
            let g = groups[group.index()];
            Self::funnel_amount(group, g.amount) == 1.0 && sanitize_size(g.size) == 1.0
        })
    }

    /// The funnel's effective values for one group (crits and sharpness capped at 1.0).
    #[must_use]
    pub fn group(&self, group: ParticleGroup) -> GroupTuning {
        GroupTuning {
            amount: self.amount[group.index()],
            size: self.size[group.index()],
        }
    }

    /// One gated attempt of `group`, in a burst of `burst` attempts (1 for a lone spawn).
    pub fn tune(&mut self, group: ParticleGroup, burst: usize) -> Tuned {
        let i = group.index();
        let amount = burst_amount(self.amount[i], burst);
        Tuned {
            copies: self.rng.count(amount),
            size: self.size[i],
        }
    }

    /// One gated attempt of vanilla particle type `particle_id`, in a burst of `burst`
    /// attempts. An ungrouped type is [`Tuned::ONE`] (no draw).
    ///
    /// `EXPLOSION_HUGE` is an invisible emitter that spawns six `EXPLOSION_LARGE` fireballs a
    /// tick: amount `0` drops it, an amount above `1` adds whole emitters, and thinning below
    /// `1` and the size act on its fireballs instead ([`ParticleTuning::tune_children`]).
    pub fn tune_id(&mut self, particle_id: i32, burst: usize) -> Tuned {
        let Some(group) = ParticleGroup::of_particle_id(particle_id) else {
            return Tuned::ONE;
        };
        if particle_id == packet::EXPLOSION_HUGE {
            let amount = self.amount[group.index()];
            let copies = if amount > 0.0 {
                self.rng.count(burst_amount(amount, burst).max(1.0))
            } else {
                0
            };
            return Tuned { copies, size: 1.0 };
        }
        self.tune(group, burst)
    }

    /// Apply the explosion group to particles that other particles spawned during a tick: the
    /// `EXPLOSION_LARGE` fireballs of an `EXPLOSION_HUGE` emitter (the only source of that kind
    /// in `tick`). Thins them below amount `1.0` and scales them; everything else is untouched.
    pub fn tune_children(&mut self, children: &mut Vec<Particle>) {
        let i = ParticleGroup::Explosion.index();
        let (keep, size) = (self.amount[i].min(1.0), self.size[i]);
        if keep >= 1.0 && size == 1.0 {
            return;
        }
        let rng = &mut self.rng;
        children.retain_mut(|particle| {
            if !matches!(particle.kind, ParticleKind::LargeExplode { .. }) {
                return true;
            }
            if keep < 1.0 && rng.next_f32() >= keep {
                return false;
            }
            scale_render(particle, size);
            true
        });
    }
}

/// The amount actually used for one attempt of a burst of `burst` attempts: amounts above 1 are
/// limited so the whole burst builds at most [`BURST_LIMIT`] particles (never below vanilla's 1).
fn burst_amount(amount: f32, burst: usize) -> f32 {
    if amount <= 1.0 || burst <= 1 {
        return amount;
    }
    amount.min((BURST_LIMIT as f32 / burst as f32).max(1.0))
}

/// Amounts are clamped to `0..=10` (NaN is vanilla).
fn sanitize_amount(amount: f32) -> f32 {
    if amount.is_nan() {
        1.0
    } else {
        amount.clamp(0.0, 10.0)
    }
}

/// Sizes are clamped to `0.1..=5` (NaN is vanilla).
fn sanitize_size(size: f32) -> f32 {
    if size.is_nan() {
        1.0
    } else {
        size.clamp(0.1, 5.0)
    }
}

/// How many extra `EntityParticleEmitter`s a hit adds (a "particle multiplier").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitConfig {
    /// Crit emitters per hit you land, as a multiple of vanilla's one (`1.0` = vanilla).
    pub crit: f32,
    /// Sharpness emitters per hit you land (`1.0` = vanilla).
    pub sharp: f32,
    /// Crit emitters on every hit, not only on vanilla's critical hits.
    pub always_crit: bool,
    /// Sharpness emitters on every hit, not only with a Sharpness weapon.
    pub always_sharp: bool,
}

impl HitConfig {
    /// Adds nothing.
    pub const VANILLA: HitConfig = HitConfig {
        crit: 1.0,
        sharp: 1.0,
        always_crit: false,
        always_sharp: false,
    };

    /// Whether any hit can get an extra emitter with these values.
    #[must_use]
    pub fn adds_emitters(&self) -> bool {
        self.crit > 1.0
            || self.sharp > 1.0
            || (self.always_crit && self.crit > 0.0)
            || (self.always_sharp && self.sharp > 0.0)
    }
}

impl Default for HitConfig {
    fn default() -> Self {
        HitConfig::VANILLA
    }
}

/// The extra crit and sharpness emitters for hits the local player lands. It lives on the
/// client's `Session`, whose `attack_entity_with_knockback` asks [`HitEmitters::extra`] after
/// vanilla's own emitters; [`HitEmitters::default`] adds nothing.
///
/// For a type with amount `n > 0` (the [`HitConfig`] value):
/// * vanilla's condition holds (a real crit; a weapon with Sharpness): `max(n, 1) - 1` extra,
///   on top of the one emitter vanilla shows for it (locally for a player target, as the
///   server's S0B echo for a mob);
/// * it does not, but "always" is on: `max(n, 1)` extra;
/// * otherwise none. `n = 0` adds none (the funnel hides the type everywhere).
///
/// So `n = 1` with "always" off is exactly vanilla. Fractions round on the mod's own stream. An
/// amount below 1 thins the type at the funnel instead, from every source.
#[derive(Debug, Clone, PartialEq)]
pub struct HitEmitters {
    config: HitConfig,
    active: bool,
    rng: ModRng,
}

impl Default for HitEmitters {
    fn default() -> Self {
        HitEmitters {
            config: HitConfig::VANILLA,
            active: false,
            rng: ModRng::new(0x4849_5450_4152_5449),
        }
    }
}

impl HitEmitters {
    /// The current values.
    #[must_use]
    pub fn config(&self) -> HitConfig {
        self.config
    }

    /// Replace the values (the stream keeps going).
    pub fn set_config(&mut self, config: HitConfig) {
        self.config = config;
        self.active = config.adds_emitters();
    }

    /// Whether a hit can get any extra emitter: the attack path's whole cost while it is false.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// `(crit, sharpness)` extra emitters for one hit on a living target. `crit_condition` is
    /// vanilla's critical-hit flag, `sharp_condition` its `getModifierForCreature > 0`.
    pub fn extra(&mut self, crit_condition: bool, sharp_condition: bool) -> (u32, u32) {
        if !self.active {
            return (0, 0);
        }
        let config = self.config;
        let crit = extra_for(
            &mut self.rng,
            config.crit,
            crit_condition,
            config.always_crit,
        );
        let sharp = extra_for(
            &mut self.rng,
            config.sharp,
            sharp_condition,
            config.always_sharp,
        );
        (crit, sharp)
    }
}

fn extra_for(rng: &mut ModRng, amount: f32, condition: bool, always: bool) -> u32 {
    let amount = sanitize_amount(amount);
    if amount <= 0.0 {
        return 0;
    }
    let wanted = amount.max(1.0);
    if condition {
        rng.count(wanted - 1.0)
    } else if always {
        rng.count(wanted)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::spawn;
    use glam::DVec3;
    use mc_math::JavaRandom;

    fn groups_with(group: ParticleGroup, tuning: GroupTuning) -> [GroupTuning; 7] {
        let mut groups = [GroupTuning::VANILLA; 7];
        groups[group.index()] = tuning;
        groups
    }

    #[test]
    fn every_group_round_trips_through_its_index_and_ids_map_to_the_spec_table() {
        for (i, group) in ParticleGroup::ALL.iter().enumerate() {
            assert_eq!(group.index(), i);
        }
        let expect = [
            (packet::CRIT, Some(ParticleGroup::Crit)),
            (packet::CRIT_MAGIC, Some(ParticleGroup::Sharp)),
            (packet::SPELL_MOB, Some(ParticleGroup::Swirl)),
            (packet::SPELL_MOB_AMBIENT, Some(ParticleGroup::Swirl)),
            (packet::SPELL, Some(ParticleGroup::Splash)),
            (packet::SPELL_INSTANT, Some(ParticleGroup::Splash)),
            (packet::LAVA, Some(ParticleGroup::Lava)),
            (packet::DRIP_LAVA, Some(ParticleGroup::Lava)),
            (packet::EXPLOSION_NORMAL, Some(ParticleGroup::Explosion)),
            (packet::EXPLOSION_LARGE, Some(ParticleGroup::Explosion)),
            (packet::EXPLOSION_HUGE, Some(ParticleGroup::Explosion)),
            (packet::BLOCK_CRACK, None),
            (packet::SPELL_WITCH, None),
            (packet::SMOKE_NORMAL, None),
            (packet::FLAME, None),
            (-1, None),
            (9999, None),
        ];
        for (id, group) in expect {
            assert_eq!(ParticleGroup::of_particle_id(id), group, "id {id}");
        }
    }

    #[test]
    fn counts_average_to_the_amount_and_whole_amounts_draw_nothing() {
        let mut rng = ModRng::new(7);
        let before = rng.clone();
        assert_eq!(rng.count(0.0), 0);
        assert_eq!(rng.count(-3.0), 0);
        assert_eq!(rng.count(f32::NAN), 0);
        assert_eq!(rng.count(1.0), 1);
        assert_eq!(rng.count(3.0), 3);
        assert_eq!(rng, before, "whole amounts must not consume the stream");
        let trials = 40_000;
        let total: u32 = (0..trials).map(|_| rng.count(2.25)).sum();
        let mean = f64::from(total) / f64::from(trials);
        assert!((mean - 2.25).abs() < 0.02, "{mean}");
        for _ in 0..1000 {
            let n = rng.count(0.4);
            assert!(n <= 1);
            let r = rng.next_f32();
            assert!((0.0..1.0).contains(&r));
        }
    }

    #[test]
    fn the_defaults_and_crit_emitter_amounts_need_no_funnel_table() {
        let mut defaults = [GroupTuning::VANILLA; 7];
        defaults[ParticleGroup::Crit.index()].amount = 2.0;
        defaults[ParticleGroup::Sharp.index()].amount = 2.0;
        assert!(ParticleTuning::is_vanilla(&defaults));
        assert!(!ParticleTuning::is_vanilla(&groups_with(
            ParticleGroup::Crit,
            GroupTuning {
                amount: 0.0,
                size: 1.0
            }
        )));
        assert!(!ParticleTuning::is_vanilla(&groups_with(
            ParticleGroup::Lava,
            GroupTuning {
                amount: 1.0,
                size: 1.5
            }
        )));
        assert!(!ParticleTuning::is_vanilla(&groups_with(
            ParticleGroup::Swirl,
            GroupTuning {
                amount: 0.5,
                size: 1.0
            }
        )));
    }

    #[test]
    fn ungrouped_types_are_untouched_and_crit_amount_only_thins_at_the_funnel() {
        let mut tuning = ParticleTuning::new(
            [GroupTuning {
                amount: 3.0,
                size: 2.0,
            }; 7],
        );
        assert_eq!(tuning.tune_id(packet::FLAME, 1), Tuned::ONE);
        assert_eq!(tuning.tune_id(packet::BLOCK_CRACK, 1), Tuned::ONE);
        // Crits above 1 are emitters per hit, so the funnel passes each crit particle once.
        assert_eq!(
            tuning.tune_id(packet::CRIT, 1),
            Tuned {
                copies: 1,
                size: 2.0
            }
        );
        assert_eq!(
            tuning.tune_id(packet::LAVA, 1),
            Tuned {
                copies: 3,
                size: 2.0
            }
        );
    }

    #[test]
    fn a_burst_never_multiplies_past_the_limit_but_keeps_vanilla_and_thinning() {
        let groups = groups_with(
            ParticleGroup::Splash,
            GroupTuning {
                amount: 10.0,
                size: 1.0,
            },
        );
        let mut tuning = ParticleTuning::new(groups);
        let burst = 2_000;
        let total: u32 = (0..burst)
            .map(|_| tuning.tune_id(packet::SPELL, burst).copies)
            .sum();
        assert!(total as usize <= BURST_LIMIT + 1, "{total}");
        assert!(
            total as usize >= burst,
            "never below vanilla's one per attempt"
        );
        // A huge burst still gets exactly one each, never zero.
        assert_eq!(tuning.tune_id(packet::SPELL, 1_000_000).copies, 1);
        // Thinning is not limited.
        tuning.set_groups(groups_with(
            ParticleGroup::Splash,
            GroupTuning {
                amount: 0.0,
                size: 1.0,
            },
        ));
        assert_eq!(tuning.tune_id(packet::SPELL, 1_000_000).copies, 0);
    }

    #[test]
    fn a_huge_explosion_emitter_is_kept_whole_and_its_fireballs_are_thinned_and_scaled() {
        let mut tuning = ParticleTuning::new(groups_with(
            ParticleGroup::Explosion,
            GroupTuning {
                amount: 0.5,
                size: 2.0,
            },
        ));
        assert_eq!(tuning.tune_id(packet::EXPLOSION_HUGE, 1).copies, 1);
        let mut rng = JavaRandom::new(3);
        let mut children: Vec<Particle> = (0..400)
            .map(|i| {
                if i % 2 == 0 {
                    spawn::explode_large(&mut rng, DVec3::ZERO, 0.0)
                } else {
                    spawn::smoke(&mut rng, DVec3::ZERO, DVec3::ZERO, 1.0)
                }
            })
            .collect();
        let is_fireball = |p: &Particle| matches!(p.kind, ParticleKind::LargeExplode { .. });
        let smoke_before: Vec<Particle> = children
            .iter()
            .filter(|p| !is_fireball(p))
            .cloned()
            .collect();
        tuning.tune_children(&mut children);
        let smoke_after: Vec<Particle> = children
            .iter()
            .filter(|p| !is_fireball(p))
            .cloned()
            .collect();
        assert_eq!(smoke_after, smoke_before, "other children are untouched");
        let fireballs = children.len() - smoke_after.len();
        assert!(
            (60..140).contains(&fireballs),
            "about half kept: {fireballs}"
        );
        for p in children.iter().filter(|p| is_fireball(p)) {
            let ParticleKind::LargeExplode { scale_factor, .. } = p.kind else {
                unreachable!()
            };
            assert!((scale_factor - 2.0).abs() < 1e-6, "{scale_factor}");
        }

        tuning.set_groups(groups_with(
            ParticleGroup::Explosion,
            GroupTuning {
                amount: 0.0,
                size: 1.0,
            },
        ));
        assert_eq!(tuning.tune_id(packet::EXPLOSION_HUGE, 1).copies, 0);
        tuning.set_groups(groups_with(
            ParticleGroup::Explosion,
            GroupTuning {
                amount: 3.0,
                size: 1.0,
            },
        ));
        assert_eq!(tuning.tune_id(packet::EXPLOSION_HUGE, 1).copies, 3);
    }

    #[test]
    fn scale_render_touches_render_fields_only() {
        let mut rng = JavaRandom::new(11);
        let crit = spawn::crit(&mut rng, DVec3::ZERO, DVec3::ZERO, 1.0, false);
        let mut scaled = crit.clone();
        scale_render(&mut scaled, 3.0);
        let (ParticleKind::Crit { peak_scale: a }, ParticleKind::Crit { peak_scale: b }) =
            (&crit.kind, &scaled.kind)
        else {
            unreachable!()
        };
        assert!((b - a * 3.0).abs() < 1e-6);
        assert!((scaled.scale - crit.scale * 3.0).abs() < 1e-6);
        assert_eq!(
            scaled.size, crit.size,
            "the collision box is not render scale"
        );
        assert_eq!(scaled.max_age, crit.max_age);
        assert_eq!(scaled.motion, crit.motion);

        let mut unchanged = crit.clone();
        scale_render(&mut unchanged, 1.0);
        assert_eq!(unchanged, crit);
    }

    #[test]
    fn hit_emitters_follow_the_sol_rule_and_vanilla_adds_nothing() {
        let mut hits = HitEmitters::default();
        assert!(!hits.is_active());
        assert_eq!(hits.extra(true, true), (0, 0));

        let config = |crit: f32, sharp: f32, always_crit: bool, always_sharp: bool| HitConfig {
            crit,
            sharp,
            always_crit,
            always_sharp,
        };
        // 1x with "always" off: vanilla, even with both conditions.
        hits.set_config(config(1.0, 1.0, false, false));
        assert!(!hits.is_active());
        assert_eq!(hits.extra(true, true), (0, 0));

        // Multiplier 0, 1 and 3 on a real crit: the hit's total is the multiplier.
        for (amount, extra) in [(0.0, 0), (1.0, 0), (3.0, 2)] {
            hits.set_config(config(amount, 1.0, false, false));
            assert_eq!(hits.extra(true, false).0, extra, "crit x{amount}");
            assert_eq!(hits.extra(false, false).0, 0, "no crit, no extra");
        }
        // "Always": the full amount on a hit without the condition, n - 1 with it.
        hits.set_config(config(1.0, 3.0, false, true));
        assert_eq!(hits.extra(false, false), (0, 3));
        assert_eq!(hits.extra(false, true), (0, 2));
        // "Always" at 0 adds nothing: the type is hidden.
        hits.set_config(config(1.0, 0.0, false, true));
        assert!(!hits.is_active());
        assert_eq!(hits.extra(false, false), (0, 0));
        // Below 1 the funnel thins; the attack still shows the "always" emitter to be thinned.
        hits.set_config(config(0.5, 1.0, true, false));
        assert_eq!(hits.extra(false, false), (1, 0));
        assert_eq!(hits.extra(true, false), (0, 0));
    }
}

/// The table through the real spawn paths ([`super::packet`], [`super::system`]): counts, the
/// untouched vanilla stream while off, render-only size and the particle cap.
#[cfg(test)]
mod spawn_path_tests {
    use super::*;
    use crate::frame::Camera;
    use crate::particle::billboard::{self, ParticleVertex};
    use crate::particle::packet::{self as pk, IconLookup, S2APacketParticles};
    use crate::particle::spawn::{self, SpellVariant};
    use crate::particle::system::ParticleSystem;
    use crate::particle::types::ParticleIcon;
    use glam::{DVec3, Vec3};
    use mc_data::BlockState;
    use mc_math::{BlockPos, Facing};
    use mc_world::World;

    const ICON: ParticleIcon = ParticleIcon {
        min_u: 0.0,
        max_u: 1.0,
        min_v: 0.0,
        max_v: 1.0,
    };

    struct Icons;
    impl IconLookup for Icons {
        fn block_icon(&self, _state: BlockState) -> Option<ParticleIcon> {
            Some(ICON)
        }
        fn item_icon(&self, _item_id: i32, _damage: i32) -> Option<ParticleIcon> {
            Some(ICON)
        }
    }

    /// The view position every spawn below is measured against (all spawns are in range).
    fn cam() -> DVec3 {
        DVec3::new(0.0, 64.0, 0.0)
    }

    fn at() -> DVec3 {
        DVec3::new(1.0, 65.0, 1.0)
    }

    fn stone() -> BlockState {
        BlockState::new(mc_data::block_by_name("stone").unwrap(), 0)
    }

    fn table(group: ParticleGroup, amount: f32, size: f32) -> [GroupTuning; 7] {
        let mut groups = [GroupTuning::VANILLA; 7];
        groups[group.index()] = GroupTuning { amount, size };
        groups
    }

    /// A system with `group` at `amount`, and a table installed even when that is vanilla's
    /// `1.0` (another group gets a non-vanilla size).
    fn tuned(seed: i64, group: ParticleGroup, amount: f32) -> ParticleSystem {
        let mut groups = table(group, amount, 1.0);
        let other = if group == ParticleGroup::Swirl {
            ParticleGroup::Lava
        } else {
            ParticleGroup::Swirl
        };
        groups[other.index()].size = 2.0;
        let mut system = ParticleSystem::with_seed(seed);
        system.set_tuning(Some(groups));
        assert!(system.tuning().is_some());
        system
    }

    fn packet(particle_id: i32, count: i32) -> S2APacketParticles {
        S2APacketParticles {
            particle_id,
            long_distance: false,
            pos: at(),
            offset: Vec3::new(0.5, 0.5, 0.5),
            speed: 0.1,
            count,
            args: vec![],
        }
    }

    /// A world holding one stone block at `pos` (for the mining shard).
    fn world_with_stone(pos: BlockPos) -> World {
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        world.set_block(pos, stone());
        world
    }

    #[test]
    fn spawn_counts_follow_the_amount_on_every_path() {
        let world = World::new();
        let mined_pos = BlockPos::new(2, 64, 2);
        let mined = world_with_stone(mined_pos);
        for (amount, factor) in [(0.0_f32, 0_usize), (1.0, 1), (3.0, 3)] {
            // S2A burst: 40 lava pops, all in range.
            let mut sys = tuned(1, ParticleGroup::Lava, amount);
            pk::apply(&mut sys, &world, cam(), &packet(pk::LAVA, 40), &Icons);
            assert_eq!(sys.len(), 40 * factor, "S2A burst x{amount}");

            // S2A single (count 0).
            let mut sys = tuned(2, ParticleGroup::Splash, amount);
            pk::apply(&mut sys, &world, cam(), &packet(pk::SPELL, 0), &Icons);
            assert_eq!(sys.len(), factor, "S2A single x{amount}");

            // The client funnel; its adjust hook runs once per particle built.
            let mut sys = tuned(3, ParticleGroup::Lava, amount);
            let mut adjusted = 0;
            let added = pk::spawn_one_with(
                &mut sys,
                &world,
                cam(),
                pk::DRIP_LAVA,
                false,
                at(),
                DVec3::ZERO,
                &[],
                &Icons,
                |_| adjusted += 1,
            );
            assert_eq!((added, sys.len(), adjusted), (factor > 0, factor, factor));

            // A potion swirl (`spawn_attempt`, as `sync::watched` calls it).
            let mut sys = tuned(4, ParticleGroup::Swirl, amount);
            sys.spawn_attempt(&world, pk::SPELL_MOB, |rng| {
                Some(spawn::spell(rng, at(), DVec3::ONE, SpellVariant::Mob))
            });
            assert_eq!(sys.len(), factor, "swirl x{amount}");

            // Block breaking: 64 per broken block, one shard per mining tick.
            let mut sys = tuned(5, ParticleGroup::Block, amount);
            let added = pk::add_block_destroy_effects(
                &mut sys,
                &world,
                BlockPos::new(0, 64, 0),
                stone(),
                spawn::DEFAULT_DIGGING_TINT,
                &Icons,
            );
            assert_eq!((added, sys.len()), (64 * factor, 64 * factor));
            let mut sys = tuned(6, ParticleGroup::Block, amount);
            let hit = pk::add_block_hit_effects(&mut sys, &mined, mined_pos, Facing::Up, &Icons);
            assert_eq!((hit, sys.len()), (factor > 0, factor), "mining x{amount}");

            // S27: the origin fireball plus two particles per affected block.
            let mut sys = tuned(7, ParticleGroup::Explosion, amount);
            let blocks = [BlockPos::new(1, 64, 0), BlockPos::new(0, 64, 1)];
            let passed = sys.spawn_explosion_effects(&world, cam(), at(), 1.0, &blocks);
            assert_eq!(passed, 5, "gated attempts do not depend on the amount");
            assert_eq!(sys.len(), 5 * factor, "explosion x{amount}");
        }
    }

    #[test]
    fn crit_amounts_only_thin_at_the_funnel_and_zero_hides_every_source() {
        let world = World::new();
        for group in [ParticleGroup::Crit, ParticleGroup::Sharp] {
            let id = if group == ParticleGroup::Crit {
                pk::CRIT
            } else {
                pk::CRIT_MAGIC
            };
            for (amount, expect) in [(0.0, 0), (1.0, 1), (3.0, 1), (10.0, 1)] {
                let mut sys = tuned(8, group, amount);
                pk::apply(&mut sys, &world, cam(), &packet(id, 0), &Icons);
                pk::spawn_one(&mut sys, &world, cam(), id, at(), DVec3::ZERO, &[], &Icons);
                assert_eq!(sys.len(), 2 * expect, "{group:?} x{amount}");
            }
        }
    }

    /// Everything the spawn paths can do, in one fixed order. Returns the arena right after the
    /// spawns, then runs 12 ticks (a huge explosion's fireballs included).
    fn scenario(sys: &mut ParticleSystem) -> Vec<Particle> {
        let world = World::new();
        let mined_pos = BlockPos::new(3, 64, -2);
        let mined = world_with_stone(mined_pos);
        let ids = [
            pk::CRIT,
            pk::CRIT_MAGIC,
            pk::SPELL,
            pk::SPELL_INSTANT,
            pk::SPELL_MOB,
            pk::LAVA,
            pk::DRIP_LAVA,
            pk::EXPLOSION_NORMAL,
            pk::EXPLOSION_HUGE,
            pk::FLAME,
            pk::SMOKE_NORMAL,
        ];
        for id in ids {
            pk::apply(sys, &world, cam(), &packet(id, 0), &Icons);
            pk::apply(sys, &world, cam(), &packet(id, 12), &Icons);
            pk::spawn_one_with(
                sys,
                &world,
                cam(),
                id,
                false,
                at(),
                DVec3::new(0.1, 0.2, 0.3),
                &[],
                &Icons,
                |p| p.color[0] *= 0.5,
            );
        }
        sys.spawn_attempt(&world, pk::SPELL_MOB_AMBIENT, |rng| {
            Some(spawn::spell(
                rng,
                at(),
                DVec3::ONE,
                SpellVariant::AmbientMob,
            ))
        });
        pk::add_block_destroy_effects(
            sys,
            &world,
            BlockPos::new(0, 64, 0),
            stone(),
            spawn::DEFAULT_DIGGING_TINT,
            &Icons,
        );
        pk::add_block_hit_effects(sys, &mined, mined_pos, Facing::North, &Icons);
        let blocks = [BlockPos::new(1, 64, 0), BlockPos::new(0, 65, 1)];
        sys.spawn_explosion_effects(&world, cam(), at(), 4.0, &blocks);
        let spawned = sys.particles().to_vec();
        for _ in 0..12 {
            sys.tick(&world);
        }
        spawned
    }

    #[test]
    fn with_the_mod_off_or_at_vanilla_values_the_rng_stream_and_particles_are_untouched() {
        let mut defaults = [GroupTuning::VANILLA; 7];
        defaults[ParticleGroup::Crit.index()].amount = 2.0;
        defaults[ParticleGroup::Sharp.index()].amount = 2.0;
        let setups: [(&str, Option<[GroupTuning; 7]>); 3] = [
            ("off", None),
            ("on, every group vanilla", Some([GroupTuning::VANILLA; 7])),
            ("on, the mod's defaults", Some(defaults)),
        ];
        let mut reference = ParticleSystem::with_seed(0x0FF);
        let reference_spawned = scenario(&mut reference);
        assert!(
            reference_spawned.len() > 100,
            "the scenario must spawn plenty"
        );
        assert!(
            !reference.is_empty(),
            "and leave some alive after the ticks"
        );
        let reference_draw = reference.rng_mut().next_i64();
        for (label, groups) in setups {
            let mut sys = ParticleSystem::with_seed(0x0FF);
            // Switching on with a real table and back must leave nothing behind either.
            sys.set_tuning(Some(table(ParticleGroup::Lava, 3.0, 2.0)));
            sys.set_tuning(groups);
            assert!(
                sys.tuning().is_none(),
                "{label}: no table at vanilla values"
            );
            let spawned = scenario(&mut sys);
            assert_eq!(spawned, reference_spawned, "{label}: the spawns");
            assert_eq!(
                sys.particles(),
                reference.particles(),
                "{label}: after the ticks"
            );
            assert_eq!(
                sys.rng_mut().next_i64(),
                reference_draw,
                "{label}: RNG stream"
            );
        }
        // Negative control: a real table does change the outcome.
        let mut sys = ParticleSystem::with_seed(0x0FF);
        sys.set_tuning(Some(table(ParticleGroup::Lava, 3.0, 1.0)));
        assert_ne!(scenario(&mut sys), reference_spawned);
    }

    /// The refactored vanilla branches against the pre-mod sequences, written out with the
    /// public builders: the S2A burst loop, the client funnel with its adjust hook, and the
    /// mining shard. A group at `100%/100%` inside an installed table also takes no extra draw.
    #[test]
    fn the_vanilla_branches_keep_the_pre_mod_draw_order() {
        let world = World::new();
        let make_reference = |seed: i64| {
            let mut sys = ParticleSystem::with_seed(seed);
            let p = packet(pk::LAVA, 12);
            for _ in 0..p.count {
                let rng = sys.rng_mut();
                let dpos = DVec3::new(
                    rng.next_gaussian() * f64::from(p.offset.x),
                    rng.next_gaussian() * f64::from(p.offset.y),
                    rng.next_gaussian() * f64::from(p.offset.z),
                );
                let _speed = DVec3::new(
                    rng.next_gaussian() * f64::from(p.speed),
                    rng.next_gaussian() * f64::from(p.speed),
                    rng.next_gaussian() * f64::from(p.speed),
                );
                let pos = p.pos + dpos;
                if !sys.should_spawn(false, cam(), pos) {
                    continue;
                }
                let particle = spawn::lava(sys.rng_mut(), pos);
                sys.spawn_deferred(&world, particle);
            }
            sys.flush_evictions();
            if sys.should_spawn(false, cam(), at()) {
                let speed = DVec3::new(0.2, 0.1, 0.3);
                let mut particle = spawn::spell(sys.rng_mut(), at(), speed, SpellVariant::Normal);
                particle.color[1] = 0.25;
                particle.motion *= 3.0;
                sys.spawn(&world, particle);
            }
            sys
        };
        let run = |sys: &mut ParticleSystem| {
            pk::apply(sys, &world, cam(), &packet(pk::LAVA, 12), &Icons);
            pk::spawn_one_with(
                sys,
                &world,
                cam(),
                pk::SPELL,
                false,
                at(),
                DVec3::new(0.2, 0.1, 0.3),
                &[],
                &Icons,
                |particle| {
                    particle.color[1] = 0.25;
                    particle.motion *= 3.0;
                },
            );
        };
        let mut reference = make_reference(77);
        let mut plain = ParticleSystem::with_seed(77);
        run(&mut plain);
        // Lava and splash at 100% inside a table that scales the swirl group.
        let mut tabled = ParticleSystem::with_seed(77);
        tabled.set_tuning(Some(table(ParticleGroup::Swirl, 1.0, 2.0)));
        run(&mut tabled);
        assert!(reference.len() > 5);
        assert_eq!(plain.particles(), reference.particles());
        assert_eq!(tabled.particles(), reference.particles());
        let draw = reference.rng_mut().next_i64();
        assert_eq!(plain.rng_mut().next_i64(), draw);
        assert_eq!(tabled.rng_mut().next_i64(), draw);

        // The mining shard (`EffectRenderer.addBlockHitEffects`), Up face.
        let pos = BlockPos::new(4, 64, -3);
        let mined = world_with_stone(pos);
        let mut reference = ParticleSystem::with_seed(91);
        {
            let state = stone();
            let bounds = mc_data::block_props(state).bounds;
            let rng = reference.rng_mut();
            let x = f64::from(pos.x)
                + rng.next_f64() * (bounds.max_x - bounds.min_x - 2.0 * 0.1)
                + 0.1
                + bounds.min_x;
            let _y = rng.next_f64();
            let z = f64::from(pos.z)
                + rng.next_f64() * (bounds.max_z - bounds.min_z - 2.0 * 0.1)
                + 0.1
                + bounds.min_z;
            let y = f64::from(pos.y) + bounds.max_y + 0.1;
            let tint = crate::terrain::tint::digging_tint_from_color_multiplier(&mined, state, pos);
            let mut particle = spawn::digging(
                rng,
                DVec3::new(x, y, z),
                DVec3::ZERO,
                ICON,
                tint,
                false,
                pos,
            );
            particle.motion.x *= 0.2;
            particle.motion.y = (particle.motion.y - 0.1) * 0.2 + 0.1;
            particle.motion.z *= 0.2;
            particle.size.0 *= 0.6;
            particle.size.1 *= 0.6;
            particle.scale *= 0.6;
            reference.spawn(&mined, particle);
        }
        let mut plain = ParticleSystem::with_seed(91);
        assert!(pk::add_block_hit_effects(
            &mut plain,
            &mined,
            pos,
            Facing::Up,
            &Icons
        ));
        assert_eq!(plain.particles(), reference.particles());
        assert_eq!(plain.rng_mut().next_i64(), reference.rng_mut().next_i64());
    }

    fn quad_edge(quad: &[ParticleVertex; 4]) -> f32 {
        (Vec3::from(quad[1].pos) - Vec3::from(quad[0].pos)).length()
    }

    /// Size multiplies the drawn quad and nothing else: the sized copy moves, collides, ages and
    /// dies exactly like the vanilla one, tick for tick.
    #[test]
    fn size_scales_the_drawn_quad_and_nothing_else() {
        let world = World::new();
        let mut camera = Camera {
            pos: DVec3::new(0.0, 66.0, -4.0),
            ..Camera::default()
        };
        camera.update();
        type Spawner = fn(&mut ParticleSystem, &World);
        type Member = fn(&ParticleKind) -> bool;
        let cases: [(ParticleGroup, Spawner, Member); 5] = [
            (
                ParticleGroup::Crit,
                |s, w| {
                    let speed = DVec3::new(0.1, 0.2, 0.1);
                    pk::spawn_one(s, w, cam(), pk::CRIT, at(), speed, &[], &Icons);
                },
                |k| matches!(k, ParticleKind::Crit { .. }),
            ),
            (
                ParticleGroup::Lava,
                |s, w| pk::apply(s, w, cam(), &packet(pk::LAVA, 0), &Icons),
                |k| matches!(k, ParticleKind::Lava { .. }),
            ),
            (
                ParticleGroup::Splash,
                |s, w| pk::apply(s, w, cam(), &packet(pk::SPELL, 3), &Icons),
                |k| matches!(k, ParticleKind::Spell { .. }),
            ),
            (
                ParticleGroup::Explosion,
                |s, w| {
                    s.spawn_explosion_effects(w, cam(), at(), 4.0, &[BlockPos::new(1, 64, 0)]);
                },
                // The burst's debris, smoke and the huge emitter's fireballs; the emitter
                // itself is invisible.
                |k| {
                    matches!(
                        k,
                        ParticleKind::Explode
                            | ParticleKind::Smoke { .. }
                            | ParticleKind::LargeExplode { .. }
                    )
                },
            ),
            (
                ParticleGroup::Block,
                |s, w| {
                    let pos = BlockPos::new(0, 64, 0);
                    let tint = spawn::DEFAULT_DIGGING_TINT;
                    pk::add_block_destroy_effects(s, w, pos, stone(), tint, &Icons);
                },
                |k| matches!(k, ParticleKind::Digging { .. }),
            ),
        ];
        for (group, spawn_fn, member) in cases {
            let mut plain = ParticleSystem::with_seed(21);
            let mut sized = ParticleSystem::with_seed(21);
            sized.set_tuning(Some(table(group, 1.0, 2.0)));
            spawn_fn(&mut plain, &world);
            spawn_fn(&mut sized, &world);
            let mut scaled_seen = 0;
            for tick in 0..12 {
                let (a, b) = (plain.particles(), sized.particles());
                assert_eq!(a.len(), b.len(), "{group:?} tick {tick}");
                for (a, b) in a.iter().zip(b) {
                    if !member(&a.kind) {
                        assert_eq!(a, b, "{group:?} tick {tick}: other kinds are untouched");
                        continue;
                    }
                    scaled_seen += 1;
                    // Halving the render fields gives back the vanilla particle exactly...
                    let mut back = b.clone();
                    scale_render(&mut back, 0.5);
                    assert_eq!(&back, a, "{group:?} tick {tick}: only render fields differ");
                    assert_eq!(b.scale, a.scale * 2.0);
                    // ...and the quad on screen is twice as wide.
                    let edge_a = quad_edge(&billboard::quad(a, &camera, 0.5, 15.0, 15.0));
                    let edge_b = quad_edge(&billboard::quad(b, &camera, 0.5, 15.0, 15.0));
                    assert!(
                        (edge_b - 2.0 * edge_a).abs() <= 1e-4 * edge_a.max(1e-3),
                        "{group:?} tick {tick}: edge {edge_a} -> {edge_b}"
                    );
                }
                plain.tick(&world);
                sized.tick(&world);
            }
            assert!(scaled_seen > 0, "{group:?}: nothing was sized");
        }
    }

    /// Vanilla's 4000-per-bucket cap holds whatever the amount: a 10x block-break storm keeps
    /// exactly the newest 4000 shards, oldest evicted first.
    #[test]
    fn a_multiplied_burst_keeps_the_4000_cap_and_evicts_the_oldest() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(5);
        sys.set_tuning(Some(table(ParticleGroup::Block, 10.0, 1.0)));
        let blocks: Vec<BlockPos> = (0..10).map(|i| BlockPos::new(i * 2, 64, 0)).collect();
        let mut added = 0;
        for &pos in &blocks {
            let tint = spawn::DEFAULT_DIGGING_TINT;
            added += pk::add_block_destroy_effects(&mut sys, &world, pos, stone(), tint, &Icons);
        }
        assert_eq!(added, 6_400, "640 shards per block at 10x");
        assert_eq!(sys.len(), PER_BUCKET_CAP);
        assert!(sys.bucket_counts().iter().all(|&n| n <= PER_BUCKET_CAP));
        // 2400 evicted: blocks 0-2 entirely (1920) and 480 of block 3.
        let source = |p: &Particle| match p.kind {
            ParticleKind::Digging { source_pos, .. } => source_pos,
            _ => unreachable!(),
        };
        assert_eq!(source(&sys.particles()[0]), blocks[3]);
        assert_eq!(source(&sys.particles()[PER_BUCKET_CAP - 1]), blocks[9]);
        let from_block_3 = sys
            .particles()
            .iter()
            .filter(|p| source(p) == blocks[3])
            .count();
        assert_eq!(from_block_3, 640 - 480);

        // One huge S2A packet at 10x stays bounded too (the burst limit, then the cap).
        let mut sys = ParticleSystem::with_seed(6);
        sys.set_tuning(Some(table(ParticleGroup::Splash, 10.0, 1.0)));
        pk::apply(&mut sys, &world, cam(), &packet(pk::SPELL, 3_000), &Icons);
        let counts = sys.bucket_counts();
        assert!(counts.iter().all(|&n| n <= PER_BUCKET_CAP), "{counts:?}");
        assert_eq!(sys.len(), counts.iter().sum::<usize>());
        assert!(sys.len() >= 3_000, "never below vanilla's one per attempt");
    }
}
