//! Particle construction — one function per vanilla `EntityFX` constructor (or constructor
//! family), matching field-for-field.
//!
//! # One `Random` stream, not two
//!
//! Vanilla draws particle jitter from **two** independent, both-unseeded sources: each
//! `EntityFX`'s own `this.rand` (`new Random()`, freshly constructed and never given a fixed
//! seed) and the JVM-wide static `Math.random()`. Neither is reproducible even between two runs
//! of *vanilla itself* — `new Random()` seeds from `System.nanoTime()` — so there is no sequence
//! to reproduce bit-for-bit, only the **algorithm** (mc-math's `JavaRandom`, per §7 S05) and the
//! **distributions/formulas** each draw feeds. Every function below takes one `&mut JavaRandom`
//! and uses it for both roles; a caller that wants the production entropy-seeded stream gets one
//! from [`super::system::ParticleSystem::new`], and tests pass a fixed seed for reproducibility
//! of the *test*, not of vanilla.
//!
//! java: `client/particle/Entity*FX.java` constructors, `EffectRenderer.addBlockDestroyEffects`/
//! `addBlockHitEffects`.

use glam::DVec3;
use mc_math::JavaRandom;

use super::types::{Particle, ParticleIcon, ParticleKind};

/// Vanilla's `EntityFX(World, x, y, z)` three-arg base constructor's field setup, shared by
/// every subclass regardless of which further constructor they call.
fn base_ctor(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    // Keep the draw order from EntityFX's three-argument constructor.  Java writes the two
    // texture-jitter fields first, then scale, then max age; these are all draws from the same
    // test/production stream here, so moving one field changes every subsequent random value.
    let jitter_x = rng.next_f32() * 3.0;
    let jitter_y = rng.next_f32() * 3.0;
    let scale = (rng.next_f32() * 0.5 + 0.5) * 2.0;
    // The expression is float arithmetic in Java (`4.0F`, `0.9F`, `0.1F`), not a double
    // calculation merely fed by nextFloat().  Narrow before the integer conversion.
    let max_age = (4.0_f32 / (rng.next_f32() * 0.9 + 0.1)) as u32;

    Particle {
        pos,
        prev_pos: pos,
        motion: DVec3::ZERO,
        age: 0,
        max_age,
        scale,
        gravity: 0.0, // java: `protected float particleGravity;` — unset unless a subclass sets it.
        color: [1.0, 1.0, 1.0, 1.0],
        tex_x: 0,
        tex_y: 0,
        jitter_x,
        jitter_y,
        size: (0.2, 0.2),
        on_ground: false,
        no_clip: false,
        dead: false,
        layer: super::types::FxLayer::Misc,
        icon: None,
        kind: ParticleKind::Generic,
    }
}

/// java: `EntityFX(World, x, y, z, xSpeed, ySpeed, zSpeed)` — the base ctor plus the
/// randomised-velocity composition every non-overriding subclass inherits.
fn apply_speed_jitter(rng: &mut JavaRandom, speed: DVec3) -> DVec3 {
    let jitter = DVec3::new(
        rng.next_f64() * 2.0 - 1.0,
        rng.next_f64() * 2.0 - 1.0,
        rng.next_f64() * 2.0 - 1.0,
    ) * 0.400_000_005_960_464_5;
    let motion = speed + jitter;
    // Java is `(float)(Math.random() + Math.random() + 1.0D) * 0.15F`: the sum narrows to
    // float before the multiplication.  MathHelper.sqrt_double likewise returns a float.
    let f = f64::from((rng.next_f64() + rng.next_f64() + 1.0) as f32 * 0.15_f32);
    let len = f64::from(mc_math::sqrt_d(motion.length_squared()));
    if len == 0.0 {
        return DVec3::new(0.0, 0.100_000_001_490_116_12, 0.0);
    }
    DVec3::new(
        motion.x / len * f * 0.400_000_005_960_464_5,
        motion.y / len * f * 0.400_000_005_960_464_5 + 0.100_000_001_490_116_12,
        motion.z / len * f * 0.400_000_005_960_464_5,
    )
}

/// java: `EntityFX.setParticleTextureIndex` (`:246-256`) — `indexX = i % 16; indexY = i / 16`.
///
/// Rust's `%` and `/` truncate toward zero, exactly as Java's do, so they are written plainly.
/// This used to use `rem_euclid`/`div_euclid`, which disagree for negative `i` — and `i == -1`
/// is reached on the **last live tick of every animated family** (`7 - age*8/maxAge` with
/// `age == maxAge`), where vanilla gives `(15, 0)` and Euclidean division gives `(15, -1)`.
/// A negative row index put V outside `[0,1]`, which the sampler's `ClampToEdge` smeared into
/// a one-texel strip. `F-P2-client.particle-SWEEP-12`.
fn set_tex_index(p: &mut Particle, index: i32) {
    p.tex_x = index % 16;
    p.tex_y = index / 16;
}

fn grey(v: f32) -> [f32; 4] {
    [v, v, v, 1.0]
}

/// The fallback for `EnumParticleTypes` this port has not given a dedicated family
/// (`MOB_APPEARANCE`). Plain `EntityFX` behaviour:
/// no gravity (vanilla's unset default), full white, misc-atlas index 0.
#[must_use]
pub fn generic(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, speed);
    p
}

/// java: `EntitySmokeFX` (`scale_factor = 1.0`) and `EntityCritFX` (`scale_factor = 2.5`,
/// registered for `SMOKE_LARGE`).
#[must_use]
pub fn smoke(rng: &mut JavaRandom, pos: DVec3, speed: DVec3, scale_factor: f32) -> Particle {
    let mut p = base_ctor(rng, pos);
    let jitter = apply_speed_jitter(rng, DVec3::ZERO);
    p.motion = jitter * 0.100_000_001_490_116_12 + speed;
    p.color = grey((rng.next_f64() * 0.300_000_011_920_928_96) as f32);
    p.scale *= 0.75 * scale_factor;
    let peak_scale = p.scale;
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.max_age = (p.max_age as f32 * scale_factor) as u32;
    p.kind = ParticleKind::Smoke { peak_scale };
    p
}

/// java: `EntityCrit2FX` (`CRIT`, `scale_factor = 1.0`) and `EntityCrit2FX.MagicFactory`
/// (`CRIT_MAGIC`) — pass `magic = true` for the latter, which recolours toward purple and
/// advances the texture column by one.
///
/// Vanilla calls `onUpdate()` once, synchronously, inside the constructor. This function does
/// not (it has no `World` to sweep collision against) — callers spawning through
/// [`super::system::ParticleSystem::spawn`] get that extra tick applied there.
#[must_use]
pub fn crit(
    rng: &mut JavaRandom,
    pos: DVec3,
    speed: DVec3,
    scale_factor: f32,
    magic: bool,
) -> Particle {
    let mut p = base_ctor(rng, pos);
    let jitter = apply_speed_jitter(rng, DVec3::ZERO);
    p.motion = jitter * 0.100_000_001_490_116_12 + speed * 0.4;
    p.color = grey((rng.next_f64() * 0.300_000_011_920_928_96 + 0.600_000_023_841_857_9) as f32);
    p.scale *= 0.75 * scale_factor;
    let peak_scale = p.scale;
    p.max_age = (6.0 / (rng.next_f64() * 0.8 + 0.6)) as u32;
    p.max_age = (p.max_age as f32 * scale_factor) as u32;
    set_tex_index(&mut p, 65);
    if magic {
        p.color = [p.color[0] * 0.3, p.color[1] * 0.8, p.color[2], p.color[3]];
        p.tex_x += 1;
    }
    p.kind = ParticleKind::Crit { peak_scale };
    p
}

/// java: `EntityFlameFX`.
#[must_use]
pub fn flame(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, speed);
    p.motion = p.motion * 0.009_999_999_776_482_582 + speed;
    // Deliberately not resynced with `prev_pos`, matching vanilla: the very first interpolated
    // frame lerps from the pre-jitter spawn point.
    // Java multiplies the float difference by `0.05F` before widening the result to double.
    // Widening the difference first changes the low bits of the spawn position.
    p.pos.x += f64::from((rng.next_f32() - rng.next_f32()) * 0.05_f32);
    p.pos.y += f64::from((rng.next_f32() - rng.next_f32()) * 0.05_f32);
    p.pos.z += f64::from((rng.next_f32() - rng.next_f32()) * 0.05_f32);
    let peak_scale = p.scale;
    p.color = [1.0, 1.0, 1.0, 1.0];
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32 + 4;
    p.no_clip = true;
    set_tex_index(&mut p, 48);
    p.kind = ParticleKind::Flame { peak_scale };
    p
}

/// java: `EntityLavaFX`.
#[must_use]
pub fn lava(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, DVec3::ZERO) * 0.800_000_011_920_929;
    p.motion.y = f64::from(rng.next_f32() * 0.4 + 0.05);
    p.color = [1.0, 1.0, 1.0, 1.0];
    p.scale *= rng.next_f32() * 2.0 + 0.2;
    let peak_scale = p.scale;
    p.max_age = (16.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    set_tex_index(&mut p, 49);
    p.kind = ParticleKind::Lava { peak_scale };
    p
}

/// java: `EntityReddustFX`. `r`/`g`/`b` are the wire packet's colour (yes — vanilla reuses the
/// `xSpeed`/`ySpeed`/`zSpeed` particle-spawn parameters as an RGB triple for this one type).
/// java quirk preserved: a zero **red** channel is treated as "unset" and forced to `1.0`
/// (`EntityReddustFX:24`, which tests the parameter that later multiplies `particleRed`).
#[must_use]
pub fn reddust(
    rng: &mut JavaRandom,
    pos: DVec3,
    r: f32,
    g: f32,
    b: f32,
    scale_factor: f32,
) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, DVec3::ZERO) * 0.100_000_001_490_116_12;
    // java: `EntityReddustFX` ctor (`:24-27`) tests `p_i46350_9_`, which is the parameter that
    // goes on to multiply `particleRed` (`:30`) — the **red** channel, not green. The factory
    // binds `xSpeedIn -> p_i46349_8_ -> p_i46350_9_`, so a packet with `offsetX == 0` is the one
    // vanilla reads as "colour unset" and forces to full. This used to clamp `g`, which both
    // recoloured every `offsetX == 0` redstone particle (vanilla: white-ish red; here: green)
    // and left genuine `offsetX == 0` at black. `F-P2-client.particle-SWEEP-3`,
    // `F-P2-client.particle-03-4`.
    let r = if r == 0.0 { 1.0 } else { r };
    let f = rng.next_f64() as f32 * 0.4 + 0.6;
    p.color = [
        ((rng.next_f64() * 0.200_000_002_980_232_24) as f32 + 0.8) * r * f,
        ((rng.next_f64() * 0.200_000_002_980_232_24) as f32 + 0.8) * g * f,
        ((rng.next_f64() * 0.200_000_002_980_232_24) as f32 + 0.8) * b * f,
        1.0,
    ];
    p.scale *= 0.75 * scale_factor;
    let peak_scale = p.scale;
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.max_age = (p.max_age as f32 * scale_factor) as u32;
    p.kind = ParticleKind::Reddust { peak_scale };
    p
}

/// java: `EntityPortalFX`.
#[must_use]
pub fn portal(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    let _ = apply_speed_jitter(rng, speed); // drawn and discarded, matching `super(...)`'s waste.
    p.motion = speed;
    let f = rng.next_f32() * 0.6 + 0.4;
    p.scale = rng.next_f32() * 0.2 + 0.5;
    p.color = [1.0 * f * 0.9, 1.0 * f * 0.3, 1.0 * f, 1.0];
    p.max_age = (rng.next_f64() * 10.0) as u32 + 40;
    p.no_clip = true;
    set_tex_index(&mut p, (rng.next_f64() * 8.0) as i32);
    p.kind = ParticleKind::Portal { origin: pos };
    p
}

/// java: `EntityEnchantmentTableParticleFX`. Vanilla starts the particle at `pos + speed` and
/// eases it back to `pos` over its life — `speed` here is a displacement, not a velocity.
#[must_use]
pub fn enchant_glyph(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    let _ = apply_speed_jitter(rng, speed);
    p.motion = speed;
    p.pos = pos + speed;
    p.prev_pos = p.pos;
    let f = rng.next_f32() * 0.6 + 0.4;
    p.scale = rng.next_f32() * 0.5 + 0.2;
    p.color = [1.0 * f * 0.9, 1.0 * f * 0.9, 1.0 * f, 1.0];
    p.max_age = (rng.next_f64() * 10.0) as u32 + 30;
    p.no_clip = true;
    set_tex_index(&mut p, (rng.next_f64() * 26.0 + 1.0 + 224.0) as i32);
    p.kind = ParticleKind::EnchantGlyph { origin: pos };
    p
}

/// java: `EntityBubbleFX` (`WATER_BUBBLE`).
#[must_use]
pub fn bubble(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    // EntityBubbleFX invokes EntityFX's seven-argument constructor, whose inherited randomised
    // velocity is immediately replaced by the bubble-specific motion below.  The five draws in
    // that superclass (three axis jitters plus the two Math.random() samples for its magnitude)
    // are nevertheless observable on the shared particle stream and must be consumed first.
    let _ = apply_speed_jitter(rng, speed);
    set_tex_index(&mut p, 32);
    p.size = (0.02, 0.02);
    p.scale *= rng.next_f32() * 0.6 + 0.2;
    // Java keeps these operations in double precision: the source literal is the widened float
    // value `0.019999999552965164D`, but neither the random sample nor the product is narrowed.
    // Each axis draws its own independent `Math.random()`, so this cannot be written as one
    // vectorised expression without changing which draw lands on which axis.
    p.motion = DVec3::new(
        speed.x * 0.200_000_002_980_232_24
            + (rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
        speed.y * 0.200_000_002_980_232_24
            + (rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
        speed.z * 0.200_000_002_980_232_24
            + (rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
    );
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.kind = ParticleKind::Bubble;
    p
}

/// java: `EntityRainFX` (`WATER_DROP`).
#[must_use]
pub fn rain(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    // java: `super(world, x, y, z, 0.0D, 0.0D, 0.0D)` then `motionX *= 0.3; motionY =
    // Math.random()*0.2+0.1; motionZ *= 0.3;`.
    //
    // The comment that used to sit here said the `*= 0.3` "operates on the base ctor's motion,
    // which is always zero for a fresh particle". That is wrong, and it was wrong about which
    // constructor runs: passing three explicit `0.0D` speeds selects the **seven**-argument
    // `EntityFX(World, D,D,D, D,D,D)`, the randomising one, which leaves `motionX/Z` at
    // `jitter/|jitter| * f * 0.4` with `f = (rand + rand + 1) * 0.15`. Confirmed by `javap -c` in
    // the audit (`F-P2-client.particle-SWEEP-4`, `F-P2-client.particle-03-5`). Every rain drop
    // and every water splash therefore drifts up to ~0.054 blocks/tick sideways; with zero drift
    // a splash was a column of droplets on one vertical line instead of a fan of spray.
    let jitter = apply_speed_jitter(rng, DVec3::ZERO);
    p.motion = DVec3::new(
        jitter.x * 0.300_000_011_920_928_96,
        rng.next_f64() * 0.200_000_002_980_232_24 + 0.100_000_001_490_116_12,
        jitter.z * 0.300_000_011_920_928_96,
    );
    p.color = [1.0, 1.0, 1.0, 1.0];
    set_tex_index(&mut p, 19 + rng.next_i32_bound(4));
    p.size = (0.01, 0.01);
    p.gravity = 0.06;
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.kind = ParticleKind::Rain { is_splash: false };
    p
}

/// java: `EntitySplashFX` (`WATER_SPLASH`) — an `EntityRainFX` with heavier gravity, a bumped
/// texture column, and (when spawned with a purely horizontal velocity) an extra upward kick.
#[must_use]
pub fn splash(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = rain(rng, pos);
    p.gravity = 0.04;
    p.tex_x += 1;
    if speed.y == 0.0 && (speed.x != 0.0 || speed.z != 0.0) {
        p.motion = DVec3::new(speed.x, speed.y + 0.1, speed.z);
    }
    p.kind = ParticleKind::Rain { is_splash: true };
    p
}

/// java: `EntityDropParticleFX.WaterFactory`/`LavaFactory` (`DRIP_WATER`/`DRIP_LAVA`).
#[must_use]
pub fn drip(rng: &mut JavaRandom, pos: DVec3, is_water: bool) -> Particle {
    let mut p = base_ctor(rng, pos);
    // EntityDropParticleFX invokes EntityFX's seven-argument constructor with three literal
    // zero speeds. Its randomised inherited velocity is immediately overwritten below, but the
    // three axis jitters and two Math.random() magnitude draws still advance the shared stream.
    let _ = apply_speed_jitter(rng, DVec3::ZERO);
    p.motion = DVec3::ZERO;
    p.color = if is_water {
        [0.0, 0.0, 1.0, 1.0]
    } else {
        [1.0, 0.0, 0.0, 1.0]
    };
    set_tex_index(&mut p, 113);
    p.size = (0.01, 0.01);
    p.gravity = 0.06;
    p.max_age = (64.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.kind = ParticleKind::Drip {
        is_water,
        bob_timer: 40,
    };
    p
}

/// java: `EntityCloudFX` — the drifting rings `EntityHugeExplodeFX` spawns.
#[must_use]
pub fn cloud(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    const F: f32 = 2.5;
    let mut p = base_ctor(rng, pos);
    let jitter = apply_speed_jitter(rng, DVec3::ZERO);
    p.motion = jitter * 0.100_000_001_490_116_12 + speed;
    p.color = grey(1.0 - (rng.next_f64() * 0.300_000_011_920_928_96) as f32);
    p.scale *= 0.75 * F;
    let peak_scale = p.scale;
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.3)) as u32;
    p.max_age = (p.max_age as f32 * F) as u32;
    p.kind = ParticleKind::Cloud { peak_scale };
    p
}

/// java: `EntityExplodeFX` (`EXPLOSION_NORMAL`).
#[must_use]
pub fn explode(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    // EntityExplodeFX calls the seven-argument EntityFX constructor. The inherited velocity is
    // replaced by the explosion-specific jitter immediately afterward, so only its five random
    // draws are observable here.
    let _ = apply_speed_jitter(rng, speed);
    p.motion = DVec3::new(
        speed.x + (rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
        speed.y + (rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
        speed.z + (rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
    );
    p.color = grey(rng.next_f32() * 0.3 + 0.7);
    p.scale = rng.next_f32() * rng.next_f32() * 6.0 + 1.0;
    p.max_age = (16.0 / (f64::from(rng.next_f32()) * 0.8 + 0.2)) as u32 + 2;
    p.kind = ParticleKind::Explode;
    p
}

/// java: `EntityHugeExplodeFX` (`EXPLOSION_HUGE`) — invisible; ticking it spawns
/// [`explode_large`] particles for 8 ticks. `getFXLayer() == 1` but it renders nothing.
#[must_use]
pub fn huge_explode(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = DVec3::ZERO;
    p.layer = super::types::FxLayer::Terrain;
    p.kind = ParticleKind::HugeExplode { elapsed: 0, max: 8 };
    p
}

/// java: `EntityLargeExplodeFX` (`EXPLOSION_LARGE`) — renders on [`super::types::FxLayer::LitExplosion`].
#[must_use]
pub fn explode_large(rng: &mut JavaRandom, pos: DVec3, scale_input: f32) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = DVec3::ZERO;
    let max = 6 + rng.next_i32_bound(4) as u32;
    p.color = grey(rng.next_f32() * 0.6 + 0.4);
    p.max_age = max;
    p.layer = super::types::FxLayer::LitExplosion;
    p.kind = ParticleKind::LargeExplode {
        max,
        scale_factor: 1.0 - scale_input * 0.5,
    };
    p
}

/// The four `EntitySpellParticleFX` factories differ only in initial colour/alpha and
/// `base_tex`; this covers all of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpellVariant {
    /// `EntitySpellParticleFX.Factory` (`SPELL`). `base_tex = 128`.
    Normal,
    /// `InstantFactory` (`SPELL_INSTANT`). `base_tex = 144`, no recolour.
    Instant,
    /// `MobFactory` (`SPELL_MOB`). `base_tex = 128`, recoloured from `speed` as RGB.
    Mob,
    /// `AmbientMobFactory` (`SPELL_MOB_AMBIENT`). `base_tex = 128`, recoloured from `speed`,
    /// alpha forced to `0.15`.
    AmbientMob,
    /// `WitchFactory` (`SPELL_WITCH`). `base_tex = 144`, a randomised purple.
    Witch,
}

/// java: `EntitySpellParticleFX` and its four factories.
#[must_use]
pub fn spell(rng: &mut JavaRandom, pos: DVec3, speed: DVec3, variant: SpellVariant) -> Particle {
    let dx = 0.5 - rng.next_f64();
    let dz = 0.5 - rng.next_f64();
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, DVec3::new(dx, speed.y, dz));
    p.motion.y *= 0.200_000_002_980_232_24;
    if speed.x == 0.0 && speed.z == 0.0 {
        p.motion.x *= 0.100_000_001_490_116_12;
        p.motion.z *= 0.100_000_001_490_116_12;
    }
    p.scale *= 0.75;
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    let base_tex = match variant {
        SpellVariant::Normal | SpellVariant::Mob | SpellVariant::AmbientMob => 128,
        SpellVariant::Instant | SpellVariant::Witch => 144,
    };
    match variant {
        SpellVariant::Mob | SpellVariant::AmbientMob => {
            p.color = [speed.x as f32, speed.y as f32, speed.z as f32, p.color[3]];
        }
        SpellVariant::Witch => {
            let f = rng.next_f32() * 0.5 + 0.35;
            p.color = [1.0 * f, 0.0, 1.0 * f, p.color[3]];
        }
        SpellVariant::Normal | SpellVariant::Instant => {}
    }
    if matches!(variant, SpellVariant::AmbientMob) {
        p.color[3] = 0.15;
    }
    p.kind = ParticleKind::Spell { base_tex };
    p
}

/// java: `EntityBreakingFX` — `ITEM_CRACK`, and the slime-ball/snowball "poof" factories.
/// `icon` is the item's particle sprite ([`super::phase`] resolves it from `mc_assets`).
#[must_use]
pub fn breaking(rng: &mut JavaRandom, pos: DVec3, speed: DVec3, icon: ParticleIcon) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, DVec3::ZERO) * 0.100_000_001_490_116_12 + speed;
    p.icon = Some(icon);
    p.color = [1.0, 1.0, 1.0, 1.0];
    p.gravity = 1.0; // java: `Blocks.snow.blockParticleGravity`, vanilla's constant `1.0F`.
    p.scale /= 2.0;
    p.layer = super::types::FxLayer::Terrain;
    p.kind = ParticleKind::Breaking;
    p
}

/// java: `EntityDiggingFX` (`BLOCK_CRACK`, `addBlockDestroyEffects`/`addBlockHitEffects`) and
/// `EntityBlockDustFX` (`BLOCK_DUST`, `is_dust = true`: keeps `speed` as the exact velocity
/// instead of randomising it).
///
/// `tint` is the block's colour multiplier already folded in (vanilla's flat `0.6` grey, times
/// `Block.getRenderColor` for the handful of blocks that override it, and *not* applied at all
/// for grass — a documented vanilla quirk, `EntityDiggingFX.setBlockPos`/`func_174845_l` both
/// early-return for `Blocks.grass`). [`DEFAULT_DIGGING_TINT`] is the common case.
#[must_use]
#[allow(clippy::too_many_arguments)] // `EntityDiggingFX`'s own constructor plus `setBlockPos`.
pub fn digging(
    rng: &mut JavaRandom,
    pos: DVec3,
    speed: DVec3,
    icon: ParticleIcon,
    tint: [f32; 3],
    is_dust: bool,
    source_pos: mc_math::BlockPos,
) -> Particle {
    let mut p = base_ctor(rng, pos);
    // `EntityBlockDustFX` calls the seven-argument `EntityDiggingFX` superclass first.  That
    // superclass calls the randomising `EntityFX` constructor, so the inherited velocity draws
    // must be consumed even though BlockDust immediately overwrites all three motion fields with
    // the supplied speed.  This matters because packet particles share this RNG stream: skipping
    // those discarded draws shifts every later particle's constructor state.
    let inherited_motion = apply_speed_jitter(rng, speed);
    p.motion = if is_dust { speed } else { inherited_motion };
    p.icon = Some(icon);
    p.color = [tint[0], tint[1], tint[2], 1.0];
    p.gravity = 1.0;
    p.scale /= 2.0;
    p.layer = super::types::FxLayer::Terrain;
    p.kind = ParticleKind::Digging {
        is_dust,
        source_pos,
    };
    p
}

/// The flat grey vanilla applies to every block-break particle before any per-block colour
/// multiplier. java: `EntityDiggingFX` ctor — `particleRed = particleGreen = particleBlue = 0.6F`.
pub const DEFAULT_DIGGING_TINT: [f32; 3] = [0.6, 0.6, 0.6];

/// java: `EntityFirework.SparkFX`. Colour, fade colour, trail and twinkle are vanilla
/// post-construction setters (`setColour`, `setFadeColour`, `setTrail`, `setTwinkle`); this takes
/// them as arguments since Rust has no equivalent of mutating a freshly-`new`'d object inline.
#[must_use]
pub fn firework_spark(
    rng: &mut JavaRandom,
    pos: DVec3,
    speed: DVec3,
    color: [f32; 3],
    fade_color: Option<[f32; 3]>,
    trail: bool,
    twinkle: bool,
) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = speed;
    p.scale *= 0.75;
    p.max_age = 48 + rng.next_i32_bound(12) as u32;
    p.color = [color[0], color[1], color[2], 0.99];
    p.kind = ParticleKind::FireworkSpark {
        base_tex: 160,
        trail,
        twinkle,
        fade_color,
    };
    p
}

/// java: `EntityBreakingFX(World, x, y, z, Item)` — the **five**-argument constructor, which
/// `SlimeFactory` and `SnowballFactory` use. Unlike [`breaking`] (the eight-argument form
/// `ITEM_CRACK` goes through) it never folds the packet's speed in at all: those two types
/// discard `xSpeed`/`ySpeed`/`zSpeed` entirely and keep only the base constructor's jitter.
#[must_use]
pub fn breaking_item(rng: &mut JavaRandom, pos: DVec3, icon: ParticleIcon) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, DVec3::ZERO);
    p.icon = Some(icon);
    p.color = [1.0, 1.0, 1.0, 1.0];
    p.gravity = 1.0;
    p.scale /= 2.0;
    p.layer = super::types::FxLayer::Terrain;
    p.kind = ParticleKind::Breaking;
    p
}

/// java: `EntityAuraFX` — `TOWN_AURA` and `SUSPENDED_DEPTH` (`Factory`), and `VILLAGER_HAPPY`
/// (`HappyVillagerFactory`, which then forces texture index 82 and white). Pass
/// `happy_villager = true` for the latter.
#[must_use]
pub fn aura(rng: &mut JavaRandom, pos: DVec3, speed: DVec3, happy_villager: bool) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, speed);
    let f = rng.next_f32() * 0.1 + 0.2;
    p.color = grey(f);
    set_tex_index(&mut p, 0);
    p.size = (0.02, 0.02);
    p.scale *= rng.next_f32() * 0.6 + 0.5;
    p.motion *= 0.019_999_999_552_965_164;
    p.max_age = (20.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.no_clip = true;
    if happy_villager {
        // java: `HappyVillagerFactory` — `setParticleTextureIndex(82)` then
        // `setRBGColorF(1.0F, 1.0F, 1.0F)`, both applied *after* construction.
        set_tex_index(&mut p, 82);
        p.color = [1.0, 1.0, 1.0, 1.0];
    }
    p.kind = ParticleKind::Aura;
    p
}

/// java: `EntitySuspendFX` (`SUSPENDED`). Spawns `0.125` **below** the requested Y and keeps no
/// velocity at all — vanilla writes `motionX = xSpeedIn * 0.0D`, i.e. literally zero.
#[must_use]
pub fn suspend(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    let mut p = base_ctor(rng, DVec3::new(pos.x, pos.y - 0.125, pos.z));
    p.color = [0.4, 0.4, 0.7, 1.0];
    set_tex_index(&mut p, 0);
    p.size = (0.01, 0.01);
    p.scale *= rng.next_f32() * 0.6 + 0.2;
    p.motion = DVec3::ZERO;
    p.max_age = (16.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.kind = ParticleKind::Suspend;
    p
}

/// java: `EntityFishWakeFX` (`WATER_WAKE`).
///
/// The constructor's first three `motion` statements are dead in vanilla — it computes a jittered
/// velocity and then overwrites all three components with the raw arguments three lines later
/// (`EntityFishWakeFX.java:23-25`). The discarded seven-argument superclass draws still advance
/// the shared particle stream, including the temporary upward Math.random() assignment.
#[must_use]
pub fn fish_wake(rng: &mut JavaRandom, pos: DVec3, speed: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    // EntityFishWakeFX passes literal zero speeds to EntityFX's seven-argument constructor. The
    // inherited X/Z values are scaled and the inherited Y value is replaced by the next random
    // sample before all three components are overwritten with `speed`; consume that whole chain
    // in Java source order so a following particle sees the same RNG state.
    let inherited = apply_speed_jitter(rng, DVec3::ZERO);
    let discarded_x = inherited.x * 0.300_000_011_920_928_96;
    let discarded_y = rng.next_f64() * 0.200_000_002_980_232_24 + 0.100_000_001_490_116_12;
    let discarded_z = inherited.z * 0.300_000_011_920_928_96;
    let _ = (discarded_x, discarded_y, discarded_z);
    p.color = [1.0, 1.0, 1.0, 1.0];
    set_tex_index(&mut p, 19);
    p.size = (0.01, 0.01);
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.gravity = 0.0;
    p.motion = speed;
    p.kind = ParticleKind::FishWake;
    p
}

/// java: `EntityHeartFX` — `HEART` (`Factory`) and `VILLAGER_ANGRY` (`AngryVillagerFactory`,
/// which spawns `0.5` higher and uses texture index 81). `scale` is vanilla's `2.0F` default.
#[must_use]
pub fn heart(rng: &mut JavaRandom, pos: DVec3, scale: f32, angry: bool) -> Particle {
    // java: `AngryVillagerFactory` passes `yCoordIn + 0.5D` into the constructor.
    let pos = if angry {
        DVec3::new(pos.x, pos.y + 0.5, pos.z)
    } else {
        pos
    };
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, DVec3::ZERO) * 0.009_999_999_776_482_582;
    p.motion.y += 0.1;
    p.scale *= 0.75 * scale;
    let peak_scale = p.scale;
    p.max_age = 16;
    set_tex_index(&mut p, if angry { 81 } else { 80 });
    p.kind = ParticleKind::Heart { peak_scale };
    p
}

/// java: `EntityNoteFX` (`NOTE`). `note` is the packet's `xSpeed`, which vanilla reuses as the
/// note's pitch fraction (`TileEntityNote` sends `note / 24.0`); `scale` is its `2.0F` default.
#[must_use]
pub fn note(rng: &mut JavaRandom, pos: DVec3, note: f32, scale: f32) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, DVec3::ZERO) * 0.009_999_999_776_482_582;
    p.motion.y += 0.2;
    // java: `MathHelper.sin((note + phase) * PI * 2) * 0.65F + 0.35F` for each channel, with
    // phases 0, 1/3, 2/3 — the rainbow the note block cycles through as its pitch climbs.
    let channel =
        |phase: f32| mc_math::sin((note + phase) * std::f32::consts::PI * 2.0) * 0.65 + 0.35;
    p.color = [
        channel(0.0),
        channel(0.333_333_34),
        channel(0.666_666_7),
        1.0,
    ];
    p.scale *= 0.75 * scale;
    let peak_scale = p.scale;
    p.max_age = 6;
    set_tex_index(&mut p, 64);
    p.kind = ParticleKind::Note { peak_scale };
    p
}

/// java: `EntitySnowShovelFX` (`SNOW_SHOVEL`). `scale_factor` is vanilla's `1.0F` default.
#[must_use]
pub fn snow_shovel(rng: &mut JavaRandom, pos: DVec3, speed: DVec3, scale_factor: f32) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.motion = apply_speed_jitter(rng, speed) * 0.100_000_001_490_116_12 + speed;
    p.color = grey(1.0 - (rng.next_f64() * 0.300_000_011_920_928_96) as f32);
    p.scale *= 0.75 * scale_factor;
    let peak_scale = p.scale;
    p.max_age = (8.0 / (rng.next_f64() * 0.8 + 0.2)) as u32;
    p.max_age = (p.max_age as f32 * scale_factor) as u32;
    p.kind = ParticleKind::SnowShovel { peak_scale };
    p
}

/// java: `Barrier` (`BARRIER`) — the barrier item's own icon, held motionless for 80 ticks.
/// `icon` is `getItemModelMesher().getParticleIcon(Item.getItemFromBlock(Blocks.barrier))`.
#[must_use]
pub fn barrier(rng: &mut JavaRandom, pos: DVec3, icon: ParticleIcon) -> Particle {
    let mut p = base_ctor(rng, pos);
    // Barrier selects EntityFX's seven-argument constructor. Its randomised inherited motion is
    // immediately overwritten below, but the five double draws still advance world.rand.
    let _ = apply_speed_jitter(rng, DVec3::ZERO);
    p.icon = Some(icon);
    p.color = [1.0, 1.0, 1.0, 1.0];
    p.motion = DVec3::ZERO;
    p.gravity = 0.0;
    p.max_age = 80;
    p.layer = super::types::FxLayer::Terrain;
    p.kind = ParticleKind::Barrier;
    p
}

/// java: `EntityFootStepFX` (`FOOTSTEP`). Its seven-argument base constructor does make random
/// motion/age draws, but then the subclass zeroes all motion and ignores `particleMaxAge` in
/// favour of its independent, fixed `footstepMaxAge = 200` counter. We store that counter in the
/// common `age`/`max_age` fields because no base behaviour reads it for this kind.
#[must_use]
pub fn footstep(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    // Calling the seven-argument Java constructor consumes its random draws before these values
    // are overwritten. They are invisible on the footprint itself but still belong in this
    // port's shared construction stream.
    let _ = apply_speed_jitter(rng, DVec3::ZERO);
    p.motion = DVec3::ZERO;
    p.age = 0;
    p.max_age = 200;
    p.layer = super::types::FxLayer::Footprint;
    p.kind = ParticleKind::Footstep;
    p
}

/// java: `MobAppearance` (`MOB_APPEARANCE`). The custom render method draws an elder guardian,
/// not a particle quad, so this particle is an invisible controller in vanilla layer 3.
#[must_use]
pub fn mob_appearance(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    let _ = apply_speed_jitter(rng, DVec3::ZERO);
    p.motion = DVec3::ZERO;
    p.gravity = 0.0;
    p.max_age = 30;
    p.layer = super::types::FxLayer::LitExplosion;
    p.kind = ParticleKind::MobAppearance;
    p
}

/// java: `EntityFirework.OverlayFX` — the single-frame flash at the burst point.
#[must_use]
pub fn firework_overlay(rng: &mut JavaRandom, pos: DVec3) -> Particle {
    let mut p = base_ctor(rng, pos);
    p.max_age = 4;
    p.kind = ParticleKind::FireworkOverlay;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> JavaRandom {
        JavaRandom::new(12345)
    }

    #[test]
    fn base_ctor_replays_java_draw_order_and_float_age_math() {
        let mut old_order_differences = 0usize;

        for seed in 0..4096 {
            let mut actual_rng = JavaRandom::new(seed);
            let actual = base_ctor(&mut actual_rng, DVec3::ZERO);

            // java: EntityFX(World, x, y, z), in source order.  Keep every operation in the
            // same f32 domain as the Java `nextFloat()` expression.
            let mut oracle_rng = JavaRandom::new(seed);
            let jitter_x = oracle_rng.next_f32() * 3.0;
            let jitter_y = oracle_rng.next_f32() * 3.0;
            let scale = (oracle_rng.next_f32() * 0.5 + 0.5) * 2.0;
            let max_age = (4.0_f32 / (oracle_rng.next_f32() * 0.9 + 0.1)) as u32;

            assert_eq!(actual.jitter_x.to_bits(), jitter_x.to_bits(), "seed {seed}");
            assert_eq!(actual.jitter_y.to_bits(), jitter_y.to_bits(), "seed {seed}");
            assert_eq!(actual.scale.to_bits(), scale.to_bits(), "seed {seed}");
            assert_eq!(actual.max_age, max_age, "seed {seed}");

            // Negative control: the old port read maxAge/scale before texture jitter and did the
            // maxAge division in f64.  This must disagree somewhere in the same corpus, or this
            // oracle would not be able to detect a regression back to that implementation.
            let mut mutant_rng = JavaRandom::new(seed);
            let mutant_max_age = (4.0 / (f64::from(mutant_rng.next_f32()) * 0.9 + 0.1)) as u32;
            let mutant_scale = (mutant_rng.next_f32() * 0.5 + 0.5) * 2.0;
            let mutant_jitter_x = mutant_rng.next_f32() * 3.0;
            let mutant_jitter_y = mutant_rng.next_f32() * 3.0;
            if mutant_max_age != max_age
                || mutant_scale.to_bits() != scale.to_bits()
                || mutant_jitter_x.to_bits() != jitter_x.to_bits()
                || mutant_jitter_y.to_bits() != jitter_y.to_bits()
            {
                old_order_differences += 1;
            }
        }

        assert!(
            old_order_differences > 0,
            "the pre-fix constructor stream never diverged from Java in 4096 seeds"
        );
    }

    #[test]
    fn speed_jitter_replays_java_float_narrowing_and_sqrt() {
        let speed = DVec3::new(0.37, -0.21, 0.91);
        let mut old_arithmetic_differences = 0usize;

        for seed in 0..4096 {
            let mut actual_rng = JavaRandom::new(seed);
            let actual = apply_speed_jitter(&mut actual_rng, speed);

            let mut oracle_rng = JavaRandom::new(seed);
            let jitter = DVec3::new(
                oracle_rng.next_f64() * 2.0 - 1.0,
                oracle_rng.next_f64() * 2.0 - 1.0,
                oracle_rng.next_f64() * 2.0 - 1.0,
            ) * 0.400_000_005_960_464_5;
            let motion = speed + jitter;
            let f =
                f64::from((oracle_rng.next_f64() + oracle_rng.next_f64() + 1.0) as f32 * 0.15_f32);
            let length_squared = motion.x * motion.x + motion.y * motion.y + motion.z * motion.z;
            // java: MathHelper.sqrt_double returns `(float)Math.sqrt(value)`.
            let length = f64::from(mc_math::sqrt_d(length_squared));
            let expected = if length == 0.0 {
                DVec3::new(0.0, 0.100_000_001_490_116_12, 0.0)
            } else {
                DVec3::new(
                    motion.x / length * f * 0.400_000_005_960_464_5,
                    motion.y / length * f * 0.400_000_005_960_464_5 + 0.100_000_001_490_116_12,
                    motion.z / length * f * 0.400_000_005_960_464_5,
                )
            };

            assert_eq!(actual.x.to_bits(), expected.x.to_bits(), "seed {seed}, x");
            assert_eq!(actual.y.to_bits(), expected.y.to_bits(), "seed {seed}, y");
            assert_eq!(actual.z.to_bits(), expected.z.to_bits(), "seed {seed}, z");

            // Negative control: both old operations were left in f64.  Replay the same random
            // stream, rather than consuming the oracle's post-helper draws, so this really tests
            // the old arithmetic and not a shifted sequence.
            let mut mutant_rng = JavaRandom::new(seed);
            let _ = mutant_rng.next_f64();
            let _ = mutant_rng.next_f64();
            let _ = mutant_rng.next_f64();
            let old_f = (mutant_rng.next_f64() + mutant_rng.next_f64() + 1.0) * 0.15;
            let old_length = length_squared.sqrt();
            let old = if old_length == 0.0 {
                DVec3::new(0.0, 0.100_000_001_490_116_12, 0.0)
            } else {
                DVec3::new(
                    motion.x / old_length * old_f * 0.400_000_005_960_464_5,
                    motion.y / old_length * old_f * 0.400_000_005_960_464_5
                        + 0.100_000_001_490_116_12,
                    motion.z / old_length * old_f * 0.400_000_005_960_464_5,
                )
            };
            if old.x.to_bits() != expected.x.to_bits()
                || old.y.to_bits() != expected.y.to_bits()
                || old.z.to_bits() != expected.z.to_bits()
            {
                old_arithmetic_differences += 1;
            }
        }

        assert!(
            old_arithmetic_differences > 0,
            "the pre-fix double helper never diverged from Java in 4096 seeds"
        );
    }

    #[test]
    fn constructors_preserve_vanilla_double_motion_constants() {
        let speed = DVec3::new(0.37, -0.21, 0.91);
        let mut old_literal_differences = [0usize; 3];

        for seed in 0..4096 {
            // java: EntityFlameFX multiplies the inherited velocity by the float literal widened
            // by the JVM to this exact double before adding the supplied speed.
            let mut actual_rng = JavaRandom::new(seed);
            let actual = flame(&mut actual_rng, DVec3::ZERO, speed);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let inherited = apply_speed_jitter(&mut oracle_rng, speed);
            let expected = inherited * 0.009_999_999_776_482_582 + speed;
            assert_eq!(
                actual.motion.x.to_bits(),
                expected.x.to_bits(),
                "flame {seed}, x"
            );
            assert_eq!(
                actual.motion.y.to_bits(),
                expected.y.to_bits(),
                "flame {seed}, y"
            );
            assert_eq!(
                actual.motion.z.to_bits(),
                expected.z.to_bits(),
                "flame {seed}, z"
            );
            let old = inherited * 0.01 + speed;
            if old != expected {
                old_literal_differences[0] += 1;
            }

            let expected_pos = DVec3::new(
                f64::from((oracle_rng.next_f32() - oracle_rng.next_f32()) * 0.05_f32),
                f64::from((oracle_rng.next_f32() - oracle_rng.next_f32()) * 0.05_f32),
                f64::from((oracle_rng.next_f32() - oracle_rng.next_f32()) * 0.05_f32),
            );
            assert_eq!(
                actual.pos.x.to_bits(),
                expected_pos.x.to_bits(),
                "flame {seed}, pos x"
            );
            assert_eq!(
                actual.pos.y.to_bits(),
                expected_pos.y.to_bits(),
                "flame {seed}, pos y"
            );
            assert_eq!(
                actual.pos.z.to_bits(),
                expected_pos.z.to_bits(),
                "flame {seed}, pos z"
            );
            let mut mutant_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut mutant_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut mutant_rng, speed);
            let old_pos = DVec3::new(
                f64::from(mutant_rng.next_f32() - mutant_rng.next_f32()) * 0.05,
                f64::from(mutant_rng.next_f32() - mutant_rng.next_f32()) * 0.05,
                f64::from(mutant_rng.next_f32() - mutant_rng.next_f32()) * 0.05,
            );
            if old_pos != expected_pos {
                old_literal_differences[0] += 1;
            }

            // java: EntityBubbleFX does not retain the inherited randomised velocity; each axis is
            // `speed * 0.20000000298023224D + (random * 2.0D - 1.0D) * 0.019999999552965164D`.
            let mut actual_rng = JavaRandom::new(seed);
            let actual = bubble(&mut actual_rng, DVec3::ZERO, speed);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut oracle_rng, speed);
            let _ = oracle_rng.next_f32(); // scale *= rand.nextFloat() * 0.6F + 0.2F
            let expected = DVec3::new(
                speed.x * 0.200_000_002_980_232_24
                    + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
                speed.y * 0.200_000_002_980_232_24
                    + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
                speed.z * 0.200_000_002_980_232_24
                    + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
            );
            assert_eq!(
                actual.motion.x.to_bits(),
                expected.x.to_bits(),
                "bubble {seed}, x"
            );
            assert_eq!(
                actual.motion.y.to_bits(),
                expected.y.to_bits(),
                "bubble {seed}, y"
            );
            assert_eq!(
                actual.motion.z.to_bits(),
                expected.z.to_bits(),
                "bubble {seed}, z"
            );
            let mut mutant_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut mutant_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut mutant_rng, speed);
            let _ = mutant_rng.next_f32();
            let old = DVec3::new(
                speed.x * 0.200_000_002_980_232_24
                    + f64::from(((mutant_rng.next_f64() * 2.0 - 1.0) as f32) * 0.02_f32),
                speed.y * 0.200_000_002_980_232_24
                    + f64::from(((mutant_rng.next_f64() * 2.0 - 1.0) as f32) * 0.02_f32),
                speed.z * 0.200_000_002_980_232_24
                    + f64::from(((mutant_rng.next_f64() * 2.0 - 1.0) as f32) * 0.02_f32),
            );
            if old != expected {
                old_literal_differences[1] += 1;
            }

            // java: EntityRainFX scales inherited X/Z by 0.30000001192092896D and assigns
            // `Math.random() * 0.20000000298023224D + 0.10000000149011612D` to Y.
            let mut actual_rng = JavaRandom::new(seed);
            let actual = rain(&mut actual_rng, DVec3::ZERO);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let inherited = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let expected = DVec3::new(
                inherited.x * 0.300_000_011_920_928_96,
                oracle_rng.next_f64() * 0.200_000_002_980_232_24 + 0.100_000_001_490_116_12,
                inherited.z * 0.300_000_011_920_928_96,
            );
            assert_eq!(
                actual.motion.x.to_bits(),
                expected.x.to_bits(),
                "rain {seed}, x"
            );
            assert_eq!(
                actual.motion.y.to_bits(),
                expected.y.to_bits(),
                "rain {seed}, y"
            );
            assert_eq!(
                actual.motion.z.to_bits(),
                expected.z.to_bits(),
                "rain {seed}, z"
            );
            let mut mutant_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut mutant_rng, DVec3::ZERO);
            let inherited = apply_speed_jitter(&mut mutant_rng, DVec3::ZERO);
            let old = DVec3::new(
                inherited.x * 0.300_000_011_920_928_96,
                mutant_rng.next_f64() * 0.2 + 0.1,
                inherited.z * 0.300_000_011_920_928_96,
            );
            if old != expected {
                old_literal_differences[2] += 1;
            }
        }

        for (family, count) in ["flame", "bubble", "rain"]
            .into_iter()
            .zip(old_literal_differences)
        {
            assert!(
                count > 0,
                "old {family} motion literals never diverged from Java in 4096 seeds"
            );
        }
    }

    #[test]
    fn bubble_consumes_inherited_velocity_draws_before_overwriting_motion() {
        let speed = DVec3::new(0.37, -0.21, 0.91);

        for seed in 0..256 {
            let mut actual_rng = JavaRandom::new(seed);
            let actual = bubble(&mut actual_rng, DVec3::ZERO, speed);
            let following = generic(&mut actual_rng, DVec3::ZERO, DVec3::ZERO);

            let mut oracle_rng = JavaRandom::new(seed);
            let mut expected = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let inherited = apply_speed_jitter(&mut oracle_rng, speed);
            expected.tex_x = 32 % 16;
            expected.tex_y = 32 / 16;
            expected.size = (0.02, 0.02);
            expected.scale *= oracle_rng.next_f32() * 0.6 + 0.2;
            expected.motion = DVec3::new(
                speed.x * 0.200_000_002_980_232_24
                    + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
                speed.y * 0.200_000_002_980_232_24
                    + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
                speed.z * 0.200_000_002_980_232_24
                    + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.019_999_999_552_965_164,
            );
            expected.max_age = (8.0 / (oracle_rng.next_f64() * 0.8 + 0.2)) as u32;
            expected.kind = ParticleKind::Bubble;
            let expected_following = generic(&mut oracle_rng, DVec3::ZERO, DVec3::ZERO);

            assert_eq!(
                actual.motion.x.to_bits(),
                expected.motion.x.to_bits(),
                "seed {seed}, x"
            );
            assert_eq!(
                actual.motion.y.to_bits(),
                expected.motion.y.to_bits(),
                "seed {seed}, y"
            );
            assert_eq!(
                actual.motion.z.to_bits(),
                expected.motion.z.to_bits(),
                "seed {seed}, z"
            );
            assert_eq!(
                actual.scale.to_bits(),
                expected.scale.to_bits(),
                "seed {seed}, scale"
            );
            assert_eq!(actual.max_age, expected.max_age, "seed {seed}, max age");
            assert_eq!(actual.kind, ParticleKind::Bubble, "seed {seed}, kind");
            assert_ne!(
                inherited, actual.motion,
                "seed {seed}: BubbleFX must overwrite super motion"
            );
            assert_eq!(
                following, expected_following,
                "seed {seed}: post-bubble RNG position"
            );

            // Control: an ordinary EntityFX seven-argument particle still retains the inherited
            // randomised velocity; the bubble-only overwrite must not alter this shared helper.
            let mut ordinary_rng = JavaRandom::new(seed);
            let ordinary = generic(&mut ordinary_rng, DVec3::ZERO, speed);
            let mut ordinary_oracle = JavaRandom::new(seed);
            let _ = base_ctor(&mut ordinary_oracle, DVec3::ZERO);
            let ordinary_expected = apply_speed_jitter(&mut ordinary_oracle, speed);
            assert_eq!(
                ordinary.motion, ordinary_expected,
                "seed {seed}: ordinary control"
            );
        }
    }

    #[test]
    fn explosion_motion_uses_vanilla_float_rounded_jitter_constant() {
        let speed = DVec3::new(0.37, -0.21, 0.91);
        let mut old_literal_differences = 0usize;

        for seed in 0..4096 {
            let mut actual_rng = JavaRandom::new(seed);
            let actual = explode(&mut actual_rng, DVec3::ZERO, speed);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut oracle_rng, speed);
            let expected = DVec3::new(
                speed.x + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
                speed.y + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
                speed.z + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
            );
            assert_eq!(
                actual.motion.x.to_bits(),
                expected.x.to_bits(),
                "seed {seed}, x"
            );
            assert_eq!(
                actual.motion.y.to_bits(),
                expected.y.to_bits(),
                "seed {seed}, y"
            );
            assert_eq!(
                actual.motion.z.to_bits(),
                expected.z.to_bits(),
                "seed {seed}, z"
            );

            let mut mutant_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut mutant_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut mutant_rng, speed);
            let old = DVec3::new(
                speed.x + (mutant_rng.next_f64() * 2.0 - 1.0) * 0.05,
                speed.y + (mutant_rng.next_f64() * 2.0 - 1.0) * 0.05,
                speed.z + (mutant_rng.next_f64() * 2.0 - 1.0) * 0.05,
            );
            if old != expected {
                old_literal_differences += 1;
            }
        }

        assert!(
            old_literal_differences > 0,
            "the pre-fix explosion jitter literal never diverged in 4096 seeds"
        );
    }

    #[test]
    fn discarded_seven_arg_constructor_draws_keep_particle_fields_and_rng_state() {
        let pos = DVec3::new(1.0, 70.0, -2.0);
        let speed = DVec3::new(0.37, -0.21, 0.91);

        for seed in 0..256 {
            // EntityDropParticleFX: the inherited velocity is discarded before the material
            // colour and the Math.random() max-age draw, but all five superclass draws remain.
            let mut actual_rng = JavaRandom::new(seed);
            let actual = drip(&mut actual_rng, pos, true);
            let actual_tail = actual_rng.next_i32();

            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, pos);
            let _ = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let expected_max_age = (64.0 / (oracle_rng.next_f64() * 0.8 + 0.2)) as u32;
            let expected_tail = oracle_rng.next_i32();

            assert_eq!(
                actual.motion,
                DVec3::ZERO,
                "drip seed {seed}: motion overwrite"
            );
            assert_eq!(
                actual.color,
                [0.0, 0.0, 1.0, 1.0],
                "drip seed {seed}: colour"
            );
            assert_eq!(
                actual.max_age, expected_max_age,
                "drip seed {seed}: max age"
            );
            assert_eq!(actual_tail, expected_tail, "drip seed {seed}: RNG state");

            // EntityExplodeFX: the inherited velocity is discarded by the three own jitter
            // assignments, which then feed the visible colour, scale, and lifetime draws.
            let mut actual_rng = JavaRandom::new(seed);
            let actual = explode(&mut actual_rng, pos, speed);
            let actual_tail = actual_rng.next_i32();

            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, pos);
            let _ = apply_speed_jitter(&mut oracle_rng, speed);
            let expected_motion = DVec3::new(
                speed.x + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
                speed.y + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
                speed.z + (oracle_rng.next_f64() * 2.0 - 1.0) * 0.050_000_000_745_058_06,
            );
            let expected_colour = oracle_rng.next_f32() * 0.3 + 0.7;
            let scale_a = oracle_rng.next_f32();
            let scale_b = oracle_rng.next_f32();
            let expected_scale = scale_a * scale_b * 6.0 + 1.0;
            let expected_max_age =
                (16.0 / (f64::from(oracle_rng.next_f32()) * 0.8 + 0.2)) as u32 + 2;
            let expected_tail = oracle_rng.next_i32();

            assert_eq!(
                actual.motion, expected_motion,
                "explode seed {seed}: motion"
            );
            assert_eq!(
                actual.color,
                [expected_colour, expected_colour, expected_colour, 1.0],
                "explode seed {seed}: colour"
            );
            assert_eq!(actual.scale, expected_scale, "explode seed {seed}: scale");
            assert_eq!(
                actual.max_age, expected_max_age,
                "explode seed {seed}: max age"
            );
            assert_eq!(actual_tail, expected_tail, "explode seed {seed}: RNG state");

            // EntityFishWakeFX: after the inherited motion and its overwritten upward sample,
            // maxAge is drawn, then every visible motion component is replaced by raw `speed`.
            let mut actual_rng = JavaRandom::new(seed);
            let actual = fish_wake(&mut actual_rng, pos, speed);
            let actual_tail = actual_rng.next_i32();

            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, pos);
            let inherited = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let _discarded_x = inherited.x * 0.300_000_011_920_928_96;
            let _discarded_y =
                oracle_rng.next_f64() * 0.200_000_002_980_232_24 + 0.100_000_001_490_116_12;
            let _discarded_z = inherited.z * 0.300_000_011_920_928_96;
            let expected_max_age = (8.0 / (oracle_rng.next_f64() * 0.8 + 0.2)) as u32;
            let expected_tail = oracle_rng.next_i32();

            assert_eq!(
                actual.motion, speed,
                "fish wake seed {seed}: motion overwrite"
            );
            assert_eq!(
                actual.color,
                [1.0, 1.0, 1.0, 1.0],
                "fish wake seed {seed}: colour"
            );
            assert_eq!(
                actual.max_age, expected_max_age,
                "fish wake seed {seed}: max age"
            );
            assert_eq!(
                actual_tail, expected_tail,
                "fish wake seed {seed}: RNG state"
            );
        }
    }

    #[test]
    fn constructor_colours_preserve_java_double_then_float_narrowing() {
        const THREE_TENTHS_AS_DOUBLE: f64 = 0.300_000_011_920_928_96;
        const CRIT_BASE_AS_DOUBLE: f64 = 0.600_000_023_841_857_9;
        const ONE_FIFTH_AS_DOUBLE: f64 = 0.200_000_002_980_232_24;
        let mut old_order_differences = [0usize; 5];

        for seed in 0..4096 {
            let mut actual_rng = JavaRandom::new(seed);
            let actual = smoke(&mut actual_rng, DVec3::ZERO, DVec3::ZERO, 1.0);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let sample = oracle_rng.next_f64();
            let expected = (sample * THREE_TENTHS_AS_DOUBLE) as f32;
            let old = sample as f32 * 0.3;
            assert_eq!(
                actual.color[0].to_bits(),
                expected.to_bits(),
                "smoke seed {seed}"
            );
            if old.to_bits() != expected.to_bits() {
                old_order_differences[0] += 1;
            }

            let mut actual_rng = JavaRandom::new(seed);
            let actual = crit(&mut actual_rng, DVec3::ZERO, DVec3::ZERO, 1.0, false);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let sample = oracle_rng.next_f64();
            let expected = (sample * THREE_TENTHS_AS_DOUBLE + CRIT_BASE_AS_DOUBLE) as f32;
            let old = sample as f32 * 0.3 + 0.6;
            assert_eq!(
                actual.color[0].to_bits(),
                expected.to_bits(),
                "crit seed {seed}"
            );
            if old.to_bits() != expected.to_bits() {
                old_order_differences[1] += 1;
            }

            let mut actual_rng = JavaRandom::new(seed);
            let actual = cloud(&mut actual_rng, DVec3::ZERO, DVec3::ZERO);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let sample = oracle_rng.next_f64();
            let expected = 1.0 - (sample * THREE_TENTHS_AS_DOUBLE) as f32;
            let old = 1.0 - sample as f32 * 0.3;
            assert_eq!(
                actual.color[0].to_bits(),
                expected.to_bits(),
                "cloud seed {seed}"
            );
            if old.to_bits() != expected.to_bits() {
                old_order_differences[2] += 1;
            }

            // Snow-shovel was already the one exact constructor in this family; retain it as a
            // control so a future bulk rewrite cannot pull it back into early f32 arithmetic.
            let mut actual_rng = JavaRandom::new(seed);
            let actual = snow_shovel(&mut actual_rng, DVec3::ZERO, DVec3::ZERO, 1.0);
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let sample = oracle_rng.next_f64();
            let expected = 1.0 - (sample * THREE_TENTHS_AS_DOUBLE) as f32;
            let old = 1.0 - sample as f32 * 0.3;
            assert_eq!(
                actual.color[0].to_bits(),
                expected.to_bits(),
                "snow-shovel seed {seed}"
            );
            if old.to_bits() != expected.to_bits() {
                old_order_differences[3] += 1;
            }

            let input = [0.37, 0.61, 0.83];
            let mut actual_rng = JavaRandom::new(seed);
            let actual = reddust(
                &mut actual_rng,
                DVec3::ZERO,
                input[0],
                input[1],
                input[2],
                1.0,
            );
            let mut oracle_rng = JavaRandom::new(seed);
            let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
            let _ = apply_speed_jitter(&mut oracle_rng, DVec3::ZERO);
            let f = oracle_rng.next_f64() as f32 * 0.4 + 0.6;
            for (channel, &input_value) in input.iter().enumerate() {
                let sample = oracle_rng.next_f64();
                let expected = ((sample * ONE_FIFTH_AS_DOUBLE) as f32 + 0.8) * input_value * f;
                let old = (sample as f32 * 0.2 + 0.8) * input_value * f;
                assert_eq!(
                    actual.color[channel].to_bits(),
                    expected.to_bits(),
                    "reddust seed {seed}, channel {channel}"
                );
                if old.to_bits() != expected.to_bits() {
                    old_order_differences[4] += 1;
                }
            }
        }

        for (family, count) in ["smoke", "crit", "cloud", "snow-shovel", "reddust"]
            .into_iter()
            .zip(old_order_differences)
        {
            assert!(
                count > 0,
                "{family} sample never distinguished early f32 narrowing"
            );
        }
    }

    #[test]
    fn base_ctor_defaults_match_vanilla_field_defaults() {
        let mut r = rng();
        let p = generic(&mut r, DVec3::ZERO, DVec3::ZERO);
        assert_eq!(
            p.gravity, 0.0,
            "unset particleGravity defaults to 0 in Java"
        );
        assert_eq!(p.size, (0.2, 0.2));
        assert_eq!(p.color, [1.0, 1.0, 1.0, 1.0]);
        assert!(p.jitter_x >= 0.0 && p.jitter_x < 3.0);
        assert!(p.jitter_y >= 0.0 && p.jitter_y < 3.0);
        assert!(p.scale >= 0.5 * 2.0 * 0.5 && p.scale <= 2.0);
        assert!(p.max_age >= 4 && p.max_age <= 40); // 4.0 / (0.1..1.0)
    }

    #[test]
    fn speed_jitter_is_never_nan_even_from_zero_input() {
        let mut r = rng();
        for _ in 0..1000 {
            let m = apply_speed_jitter(&mut r, DVec3::ZERO);
            assert!(m.is_finite(), "{m:?}");
        }
    }

    #[test]
    fn tex_index_matches_java_div_mod_16() {
        let mut r = rng();
        let mut p = generic(&mut r, DVec3::ZERO, DVec3::ZERO);
        set_tex_index(&mut p, 48);
        assert_eq!((p.tex_x, p.tex_y), (0, 3));
        set_tex_index(&mut p, 65);
        assert_eq!((p.tex_x, p.tex_y), (1, 4));
    }

    #[test]
    fn crit_magic_recolours_and_advances_column() {
        let mut r = rng();
        let base = crit(&mut r.clone(), DVec3::ZERO, DVec3::ZERO, 1.0, false);
        let magic = crit(&mut r, DVec3::ZERO, DVec3::ZERO, 1.0, true);
        assert_eq!(base.tex_x, 65 % 16);
        assert_eq!(magic.tex_x, 66 % 16);
        assert!(matches!(magic.kind, ParticleKind::Crit { .. }));
    }

    #[test]
    fn splash_bumps_texture_and_gravity_from_rain() {
        let mut r = rng();
        let rn = rain(&mut r.clone(), DVec3::ZERO);
        let sp = splash(&mut r, DVec3::ZERO, DVec3::ZERO);
        assert_eq!(sp.tex_x, rn.tex_x + 1);
        assert_eq!(sp.gravity, 0.04);
        assert!(matches!(sp.kind, ParticleKind::Rain { is_splash: true }));
    }

    #[test]
    fn splash_with_horizontal_speed_gets_an_upward_kick() {
        let mut r = rng();
        let sp = splash(&mut r, DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(sp.motion, DVec3::new(1.0, 0.1, 0.0));
    }

    #[test]
    fn ambient_mob_spell_forces_low_alpha() {
        let mut r = rng();
        let p = spell(
            &mut r,
            DVec3::ZERO,
            DVec3::new(0.5, 0.0, 0.5),
            SpellVariant::AmbientMob,
        );
        assert_eq!(p.color[3], 0.15);
    }

    #[test]
    fn mob_spell_recolours_from_speed() {
        let mut r = rng();
        let p = spell(
            &mut r,
            DVec3::ZERO,
            DVec3::new(0.2, 0.4, 0.6),
            SpellVariant::Mob,
        );
        assert_eq!(p.color[0], 0.2_f32);
        assert_eq!(p.color[1], 0.4_f32);
        assert_eq!(p.color[2], 0.6_f32);
    }

    #[test]
    fn digging_and_dust_differ_only_in_velocity_handling() {
        let icon = ParticleIcon {
            min_u: 0.0,
            max_u: 0.0625,
            min_v: 0.0,
            max_v: 0.0625,
        };
        let mut r = rng();
        let dust = digging(
            &mut r,
            DVec3::ZERO,
            DVec3::new(0.3, 0.1, -0.2),
            icon,
            DEFAULT_DIGGING_TINT,
            true,
            mc_math::BlockPos::ORIGIN,
        );
        assert_eq!(
            dust.motion,
            DVec3::new(0.3, 0.1, -0.2),
            "dust keeps the exact given velocity"
        );
        assert_eq!(dust.color, [0.6, 0.6, 0.6, 1.0]);
        assert_eq!(dust.layer, super::super::types::FxLayer::Terrain);
    }

    #[test]
    fn dust_consumes_inherited_velocity_draws_before_overwriting_motion() {
        let icon = ParticleIcon {
            min_u: 0.0,
            max_u: 0.0625,
            min_v: 0.0,
            max_v: 0.0625,
        };
        let speed = DVec3::new(0.3, 0.1, -0.2);
        let seed = 0x5eed_i64;

        // EntityBlockDustFX keeps the caller's velocity, but its EntityDiggingFX superclass still
        // consumes the randomised EntityFX velocity draws before that overwrite.
        let mut actual_rng = JavaRandom::new(seed);
        let dust = digging(
            &mut actual_rng,
            DVec3::ZERO,
            speed,
            icon,
            DEFAULT_DIGGING_TINT,
            true,
            mc_math::BlockPos::ORIGIN,
        );
        let following = generic(&mut actual_rng, DVec3::ZERO, DVec3::ZERO);

        let mut oracle_rng = JavaRandom::new(seed);
        let _ = base_ctor(&mut oracle_rng, DVec3::ZERO);
        let inherited = apply_speed_jitter(&mut oracle_rng, speed);
        let expected_following = generic(&mut oracle_rng, DVec3::ZERO, DVec3::ZERO);

        assert_eq!(dust.motion, speed, "dust overwrites inherited velocity");
        assert_ne!(
            inherited, speed,
            "test seed must exercise the discarded velocity"
        );
        assert_eq!(
            following, expected_following,
            "dust must leave the shared RNG at the post-superclass position"
        );

        // Ordinary EntityDiggingFX keeps the inherited randomised motion, so the same constructor
        // stream remains the control path for BLOCK_CRACK and hit/destroy effects.
        let mut ordinary_rng = JavaRandom::new(seed);
        let ordinary = digging(
            &mut ordinary_rng,
            DVec3::ZERO,
            speed,
            icon,
            DEFAULT_DIGGING_TINT,
            false,
            mc_math::BlockPos::ORIGIN,
        );
        assert_eq!(ordinary.motion, inherited);
        assert_ne!(ordinary.motion, speed);
    }

    #[test]
    fn huge_explode_starts_invisible_at_tick_zero() {
        let mut r = rng();
        let p = huge_explode(&mut r, DVec3::ZERO);
        assert!(matches!(
            p.kind,
            ParticleKind::HugeExplode { elapsed: 0, max: 8 }
        ));
    }

    #[test]
    fn portal_and_enchant_glyph_store_their_easing_origin() {
        let mut r = rng();
        let origin = DVec3::new(1.0, 2.0, 3.0);
        let p = portal(&mut r, origin, DVec3::ZERO);
        assert!(matches!(p.kind, ParticleKind::Portal { origin: o } if o == origin));
        let p2 = enchant_glyph(&mut r, origin, DVec3::new(0.1, 0.2, 0.3));
        assert!(matches!(p2.kind, ParticleKind::EnchantGlyph { origin: o } if o == origin));
        // java: `posX = prevPosX = xCoordIn + xSpeedIn` — starts displaced, not at `origin`.
        assert!((p2.pos - (origin + DVec3::new(0.1, 0.2, 0.3))).length() < 1e-9);
    }

    #[test]
    fn firework_spark_lifetime_is_in_vanillas_range() {
        let mut r = rng();
        for _ in 0..200 {
            let p = firework_spark(
                &mut r,
                DVec3::ZERO,
                DVec3::ZERO,
                [1.0, 0.0, 0.0],
                None,
                false,
                false,
            );
            assert!(p.max_age >= 48 && p.max_age < 60, "{}", p.max_age);
        }
    }

    /// `F-P2-client.particle-SWEEP-4` / `F-P2-client.particle-03-5`: `EntityRainFX` calls the
    /// **seven**-argument randomising `EntityFX` constructor with three literal `0.0D` speeds, so
    /// `motionX/Z` are the jitter (bounded by `0.45 * 0.4 = 0.18`) times `0.3`. The port assigned
    /// exactly zero and its comment asserted that was correct.
    ///
    /// Bounds are asserted in both directions, so this cannot pass by the drift having been made
    /// arbitrarily large, and it is run over 200 seeds so it cannot pass on one lucky draw.
    #[test]
    fn rain_and_splash_keep_their_horizontal_drift() {
        let mut any_nonzero = false;
        for seed in 0..200 {
            let mut r = JavaRandom::new(seed);
            let p = rain(&mut r, DVec3::new(0.0, 70.0, 0.0));
            assert!(
                p.motion.x.abs() <= 0.055 && p.motion.z.abs() <= 0.055,
                "seed {seed}: drift {:?} exceeds 0.18 * 0.3",
                p.motion
            );
            assert!(
                p.motion.y >= 0.1 && p.motion.y <= 0.3,
                "seed {seed}: motionY is Math.random()*0.2+0.1, got {}",
                p.motion.y
            );
            if p.motion.x != 0.0 || p.motion.z != 0.0 {
                any_nonzero = true;
            }
        }
        assert!(
            any_nonzero,
            "every drop went straight up: the drift is still zero"
        );

        // `splash` is built on `rain` and only overwrites the motion for a purely-horizontal
        // spawn velocity, so the drift must survive the zero-velocity case.
        let mut r = JavaRandom::new(7);
        let s = splash(&mut r, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO);
        assert!(s.motion.x != 0.0 || s.motion.z != 0.0);
    }

    /// The new families' constants, read straight off the Java constructors. One test rather than
    /// nine, because each is three or four numbers.
    #[test]
    fn the_new_families_match_their_java_constructors() {
        let pos = DVec3::new(1.0, 70.0, 2.0);
        let icon = ParticleIcon {
            min_u: 0.0,
            max_u: 1.0,
            min_v: 0.0,
            max_v: 1.0,
        };

        // java: EntityAuraFX — size 0.02, noClip, grey in [0.2, 0.3), texture index 0.
        let mut r = JavaRandom::new(3);
        let a = aura(&mut r, pos, DVec3::ZERO, false);
        assert_eq!(a.size, (0.02, 0.02));
        assert!(a.no_clip);
        assert!(a.color[0] >= 0.2 && a.color[0] < 0.3, "{:?}", a.color);
        assert_eq!((a.tex_x, a.tex_y), (0, 0));

        // java: EntitySuspendFX — spawns 0.125 BELOW y, blue-grey, size 0.01, zero motion.
        let mut r = JavaRandom::new(3);
        let s = suspend(&mut r, pos);
        assert!((s.pos.y - (70.0 - 0.125)).abs() < 1e-9, "{}", s.pos.y);
        assert_eq!([s.color[0], s.color[1], s.color[2]], [0.4, 0.4, 0.7]);
        assert_eq!(s.size, (0.01, 0.01));
        assert_eq!(s.motion, DVec3::ZERO);

        // java: EntityFishWakeFX — gravity 0, texture 19, size 0.01, motion is the raw speed.
        let mut r = JavaRandom::new(3);
        let f = fish_wake(&mut r, pos, DVec3::new(0.5, -0.25, 0.75));
        assert_eq!(f.gravity, 0.0);
        assert_eq!((f.tex_x, f.tex_y), (3, 1), "19 == row 1, column 3");
        assert_eq!(f.motion, DVec3::new(0.5, -0.25, 0.75));

        // java: EntityHeartFX — maxAge 16, texture 80 (angry: 81, and y + 0.5), motionY += 0.1.
        let mut r = JavaRandom::new(3);
        let h = heart(&mut r, pos, 2.0, false);
        assert_eq!(h.max_age, 16);
        assert_eq!((h.tex_x, h.tex_y), (0, 5), "80 == row 5, column 0");
        assert!(h.motion.y > 0.09 && h.motion.y < 0.11, "{:?}", h.motion);
        let mut r = JavaRandom::new(3);
        let angry = heart(&mut r, pos, 2.0, true);
        assert_eq!((angry.tex_x, angry.tex_y), (1, 5), "81");
        assert!((angry.pos.y - 70.5).abs() < 1e-9);

        // java: EntityNoteFX — maxAge 6, texture 64, colour = three phase-shifted sines of the
        // note pitch. At note = 0 that is (0.35, 0.9129, -0.2129) before any clamp.
        let mut r = JavaRandom::new(3);
        let n = note(&mut r, pos, 0.0, 2.0);
        assert_eq!(n.max_age, 6);
        assert_eq!((n.tex_x, n.tex_y), (0, 4), "64 == row 4, column 0");
        assert!((n.color[0] - 0.35).abs() < 1e-4, "sin(0)*0.65 + 0.35");
        assert!(
            n.color[1] > n.color[0],
            "the 1/3-phase channel leads at note 0"
        );

        // java: Barrier — maxAge 80, gravity 0, motionless, terrain layer, white.
        let mut r = JavaRandom::new(3);
        let b = barrier(&mut r, pos, icon);
        assert_eq!(b.max_age, 80);
        assert_eq!(b.gravity, 0.0);
        assert_eq!(b.motion, DVec3::ZERO);
        assert_eq!(b.layer, super::super::types::FxLayer::Terrain);
        assert_eq!(b.kind, ParticleKind::Barrier);
        let actual_tail = r.next_i32();
        let mut oracle = JavaRandom::new(3);
        let _ = base_ctor(&mut oracle, pos);
        let _ = apply_speed_jitter(&mut oracle, DVec3::ZERO);
        assert_eq!(
            actual_tail,
            oracle.next_i32(),
            "Barrier discards the seven-argument EntityFX motion but retains its RNG draws"
        );

        // java: the five-arg EntityBreakingFX keeps the base ctor's jitter and never folds the
        // caller's speed in — unlike the eight-arg form `breaking` models.
        let mut r = JavaRandom::new(3);
        let item = breaking_item(&mut r, pos, icon);
        let mut r = JavaRandom::new(3);
        let cracked = breaking(&mut r, pos, DVec3::new(5.0, 5.0, 5.0), icon);
        assert!(
            item.motion.length() < 1.0 && cracked.motion.length() > 4.0,
            "the five-arg form must ignore the speed argument: {:?} vs {:?}",
            item.motion,
            cracked.motion
        );
    }
}
