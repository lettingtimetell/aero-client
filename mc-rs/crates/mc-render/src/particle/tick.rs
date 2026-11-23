//! Per-tick simulation — one function per vanilla `EntityFX.onUpdate` override.
//!
//! java: `client/particle/Entity*FX.java` `onUpdate`. Runs at 20 TPS (§7 S10's tick/frame
//! split); [`super::billboard`] handles everything that vanilla instead computes at render time
//! from `partialTicks` (scale ramps, the light boost, and the billboard quad itself).
//!
//! # Two "am I done" idioms, both reproduced exactly
//!
//! Most kinds use `if (particleAge++ >= particleMaxAge) setDead();` — post-increment, so
//! whatever reads `particleAge` afterward in the same tick sees the **new** value
//! ([`age_and_check_expiry`]). Three kinds (`EntityBubbleFX`, `EntityRainFX`,
//! `EntityDropParticleFX`) instead treat `particleMaxAge` itself as a countdown:
//! `if (particleMaxAge-- <= 0) setDead();` ([`countdown_and_check_expiry`]). `EntityPortalFX` and
//! `EntityEnchantmentTableParticleFX` are a third shape again: they read the **old** `particleAge`
//! to ease position, and only increment/check afterward — handled inline in each function since
//! nothing else shares it.

use glam::DVec3;
use mc_math::JavaRandom;
use mc_world::World;

use super::physics;
use super::spawn;
use super::system::ParticlePlayer;
use super::types::{Particle, ParticleKind, BASE_GRAVITY_ACCEL, DEFAULT_DRAG, GROUND_FRICTION};

/// New particles a tick spawned (lava's smoke wisps, a huge explosion's debris, a firework's
/// trail children). Empty for the overwhelming majority of ticks.
#[derive(Debug, Default)]
pub struct TickOutcome {
    /// Particles to add to the system, already fully constructed.
    pub spawn: Vec<Particle>,
}

fn age_and_check_expiry(p: &mut Particle) {
    let expired = p.age >= p.max_age;
    p.age += 1;
    if expired {
        p.dead = true;
    }
}

fn countdown_and_check_expiry(p: &mut Particle) {
    if p.max_age == 0 {
        p.dead = true;
    } else {
        p.max_age -= 1;
    }
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

/// Runs collision (or, for `no_clip` particles, an unclipped move) and applies its result:
/// new position, resting motion, `on_ground`.
fn collide(p: &mut Particle, world: &World) {
    let bb = p.bounding_box();
    let r = physics::sweep(world, bb, p.motion, p.no_clip);
    p.set_pos_from_box(r.bb);
    p.motion = r.motion;
    p.on_ground = r.on_ground;
}

/// java: `EntityFX.onUpdate` — the base behaviour used directly by `Generic`, `Digging`, and
/// `Breaking` (none of which override it).
fn default_tick(p: &mut Particle, world: &World) {
    p.prev_pos = p.pos;
    age_and_check_expiry(p);
    p.motion.y -= BASE_GRAVITY_ACCEL * f64::from(p.gravity);
    collide(p, world);
    p.motion *= DEFAULT_DRAG;
    if p.on_ground {
        p.motion.x *= GROUND_FRICTION;
        p.motion.z *= GROUND_FRICTION;
    }
}

/// Advance one particle by one tick. `rng` backs every `this.rand`/`Math.random()` draw a
/// `onUpdate` override makes (see [`super::spawn`]'s module docs on why one stream is correct
/// here). Returns any particles this tick spawned.
pub fn tick_one(p: &mut Particle, world: &World, rng: &mut JavaRandom) -> TickOutcome {
    tick_one_with_players(p, world, rng, &[])
}

/// [`tick_one_with_players`] with block collision off: the particles mod's "Use particle
/// physics" switched off (`ParticleSystem::set_physics`). The particle moves exactly as a
/// `noClip` particle does (vanilla's `Entity.moveEntity` early branch: the raw motion, never on
/// the ground), and everything else its `onUpdate` does is unchanged, RNG draws included. Its own
/// `no_clip` flag is restored afterwards, so switching the setting back on is exact.
pub fn tick_one_without_collision(
    p: &mut Particle,
    world: &World,
    rng: &mut JavaRandom,
    players: &[ParticlePlayer],
) -> TickOutcome {
    let no_clip = p.no_clip;
    p.no_clip = true;
    let out = tick_one_with_players(p, world, rng, players);
    p.no_clip = no_clip;
    out
}

/// Like [`tick_one`], additionally exposing the client world's non-spectating players to
/// particle behaviours which perform a vanilla entity lookup.
pub fn tick_one_with_players(
    p: &mut Particle,
    world: &World,
    rng: &mut JavaRandom,
    players: &[ParticlePlayer],
) -> TickOutcome {
    let mut out = TickOutcome::default();
    match p.kind.clone() {
        ParticleKind::Generic | ParticleKind::Digging { .. } | ParticleKind::Breaking => {
            default_tick(p, world);
        }

        // java: `EntityFootStepFX.onUpdate` — no base gravity, drag, collision or motion; it
        // simply pre-increments its separate counter and dies on equality (not >=).
        ParticleKind::Footstep => {
            p.age += 1;
            if p.age == p.max_age {
                p.dead = true;
            }
        }

        // `MobAppearance.onUpdate` calls the base implementation. Its synthetic guardian is
        // created after this first tick and is otherwise never ticked, so the appearance handoff
        // uses the particle age and a neutral guardian pose.
        ParticleKind::MobAppearance => default_tick(p, world),

        ParticleKind::Smoke { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            set_tex_index(p, 7 - (p.age * 8 / p.max_age.max(1)) as i32);
            p.motion.y += 0.004;
            collide(p, world);
            if p.pos.y == p.prev_pos.y {
                p.motion.x *= 1.1;
                p.motion.z *= 1.1;
            }
            p.motion *= 0.96;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::Crit { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            collide(p, world);
            p.color[1] = (f64::from(p.color[1]) * 0.96) as f32;
            p.color[2] = (f64::from(p.color[2]) * 0.9) as f32;
            p.motion *= 0.7;
            p.motion.y -= 0.02;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::Flame { .. } => {
            // java: `EntityFlameFX`'s constructor jitters `posX/Y/Z` but not the bounding box
            // `EntityFX.setPosition` built, and the `noClip` `Entity.moveEntity` rebuilds the
            // position from that box (`resetPositionToBB`). So the jitter only lasts until the
            // first tick, and until then `prev_pos` still holds the box's un-jittered centre.
            let box_anchor = if p.age == 0 { p.prev_pos } else { p.pos };
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            p.pos = box_anchor;
            collide(p, world); // no_clip, so this is an unclipped move.
            p.motion *= 0.96;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::Lava { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            let f = p.age as f32 / p.max_age.max(1) as f32;
            if rng.next_f32() > f {
                out.spawn.push(spawn::smoke(rng, p.pos, p.motion, 1.0));
            }
            p.motion.y -= 0.03;
            collide(p, world);
            p.motion *= 0.999;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::Reddust { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            set_tex_index(p, 7 - (p.age * 8 / p.max_age.max(1)) as i32);
            collide(p, world);
            if p.pos.y == p.prev_pos.y {
                p.motion.x *= 1.1;
                p.motion.z *= 1.1;
            }
            p.motion *= 0.96;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::Portal { origin } => {
            p.prev_pos = p.pos;
            let f = p.age as f32 / p.max_age.max(1) as f32;
            let f = 1.0 - (-f + f * f * 2.0);
            p.pos = origin + p.motion * f64::from(f);
            p.pos.y += f64::from(1.0 - f);
            let expired = p.age >= p.max_age;
            p.age += 1;
            if expired {
                p.dead = true;
            }
        }

        ParticleKind::EnchantGlyph { origin } => {
            p.prev_pos = p.pos;
            let age_frac = p.age as f32 / p.max_age.max(1) as f32;
            let f = 1.0 - age_frac;
            let f1 = age_frac * age_frac * age_frac * age_frac;
            p.pos = origin + p.motion * f64::from(f);
            p.pos.y -= f64::from(f1 * 1.2);
            let expired = p.age >= p.max_age;
            p.age += 1;
            if expired {
                p.dead = true;
            }
        }

        ParticleKind::Bubble => {
            p.prev_pos = p.pos;
            p.motion.y += 0.002;
            collide(p, world);
            p.motion *= 0.85;
            let pos = mc_math::BlockPos::from_vec3(p.pos);
            let material = mc_data::block_props(world.block_at(pos)).material;
            if material != mc_data::Material::Water {
                p.dead = true;
            }
            countdown_and_check_expiry(p);
        }

        ParticleKind::Rain { .. } => {
            p.prev_pos = p.pos;
            p.motion.y -= f64::from(p.gravity);
            collide(p, world);
            p.motion *= DEFAULT_DRAG;
            countdown_and_check_expiry(p);
            if p.on_ground {
                if rng.next_f64() < 0.5 {
                    p.dead = true;
                }
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
            rain_submerge_check(p, world);
        }

        ParticleKind::Drip {
            is_water,
            bob_timer,
        } => {
            p.prev_pos = p.pos;
            p.color = if is_water {
                [0.2, 0.3, 1.0, p.color[3]]
            } else {
                let bt = 40 - bob_timer + 16;
                [
                    1.0,
                    16.0 / bt.max(1) as f32,
                    4.0 / (40 - bob_timer + 8).max(1) as f32,
                    p.color[3],
                ]
            };
            p.motion.y -= f64::from(p.gravity);
            // java: `if (this.bobTimer-- > 0)` — a **post-decrement in the condition**, so it
            // fires whichever branch is taken and keeps running negative for the rest of the
            // particle's life. The colour formula above reads it, so a lava drip's green channel
            // walks down from `16/56` and it visibly cools from orange toward red. Decrementing
            // only inside the `if` froze the colour at the moment the bob ended, which is most of
            // a 64-320 tick lifetime. `F-P2-client.particle-SWEEP-11`.
            let was_bobbing = bob_timer > 0;
            let bob_timer = bob_timer - 1;
            if was_bobbing {
                p.motion *= 0.02;
                set_tex_index(p, 113);
            } else {
                set_tex_index(p, 112);
            }
            collide(p, world);
            p.motion *= DEFAULT_DRAG;
            countdown_and_check_expiry(p);
            if p.on_ground {
                if is_water {
                    p.dead = true;
                    out.spawn.push(spawn::splash(rng, p.pos, DVec3::ZERO));
                } else {
                    set_tex_index(p, 114);
                }
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
            p.kind = ParticleKind::Drip {
                is_water,
                bob_timer,
            };
            drip_submerge_check(p, world);
        }

        ParticleKind::Cloud { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            set_tex_index(p, 7 - (p.age * 8 / p.max_age.max(1)) as i32);
            collide(p, world);
            p.motion *= 0.96;
            // `World.getClosestPlayerToEntity(this, 2.0D)` uses strict distance `< 2^2`, then
            // EntityCloudFX pulls only when the cloud sits above that player's bounding-box
            // minimum Y (which is the player's feet position).
            if let Some(player) = players
                .iter()
                .filter(|player| (player.pos - p.pos).length_squared() < 4.0)
                .min_by(|a, b| {
                    (a.pos - p.pos)
                        .length_squared()
                        .total_cmp(&(b.pos - p.pos).length_squared())
                })
                .copied()
                .filter(|player| p.pos.y > player.pos.y)
            {
                p.pos.y += (player.pos.y - p.pos.y) * 0.2;
                p.motion.y += (player.velocity.y - p.motion.y) * 0.2;
            }
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::Explode => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            set_tex_index(p, 7 - (p.age * 8 / p.max_age.max(1)) as i32);
            p.motion.y += 0.004;
            collide(p, world);
            p.motion *= 0.9;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::HugeExplode { elapsed, max } => {
            for _ in 0..6 {
                let d = DVec3::new(
                    (rng.next_f64() - rng.next_f64()) * 4.0,
                    (rng.next_f64() - rng.next_f64()) * 4.0,
                    (rng.next_f64() - rng.next_f64()) * 4.0,
                );
                let frac = elapsed as f32 / max.max(1) as f32;
                out.spawn.push(spawn::explode_large(rng, p.pos + d, frac));
            }
            let elapsed = elapsed + 1;
            if elapsed >= max {
                p.dead = true;
            }
            p.kind = ParticleKind::HugeExplode { elapsed, max };
        }

        ParticleKind::LargeExplode { max, .. } => {
            // java: `++field_70581_a; if (field_70581_a == maximumTime) setDead();` — a pure
            // frame counter with no motion or collision. Reuses `age` as that counter; `kind`
            // (and its `max`) is left untouched since neither field changes over its life.
            p.prev_pos = p.pos;
            p.age += 1;
            if p.age >= max {
                p.dead = true;
            }
        }

        ParticleKind::Spell { base_tex } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            set_tex_index(p, base_tex + 7 - (p.age * 8 / p.max_age.max(1)) as i32);
            p.motion.y += 0.004;
            collide(p, world);
            if p.pos.y == p.prev_pos.y {
                p.motion.x *= 1.1;
                p.motion.z *= 1.1;
            }
            p.motion *= 0.96;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        ParticleKind::FireworkSpark {
            base_tex,
            trail,
            twinkle,
            fade_color,
        } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            if p.age > p.max_age / 2 {
                p.color[3] =
                    1.0 - (p.age as f32 - (p.max_age / 2) as f32) / p.max_age.max(1) as f32;
                if let Some(fc) = fade_color {
                    p.color[0] += (fc[0] - p.color[0]) * 0.2;
                    p.color[1] += (fc[1] - p.color[1]) * 0.2;
                    p.color[2] += (fc[2] - p.color[2]) * 0.2;
                }
            }
            set_tex_index(p, base_tex + 7 - (p.age * 8 / p.max_age.max(1)) as i32);
            p.motion.y -= 0.004;
            collide(p, world);
            p.motion *= 0.91;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
            if trail && p.age < p.max_age / 2 && (p.age + p.max_age).is_multiple_of(2) {
                let mut child = spawn::firework_spark(
                    rng,
                    p.pos,
                    DVec3::ZERO,
                    [p.color[0], p.color[1], p.color[2]],
                    fade_color,
                    false,
                    twinkle,
                );
                child.age = child.max_age / 2;
                out.spawn.push(child);
            }
            p.kind = ParticleKind::FireworkSpark {
                base_tex,
                trail,
                twinkle,
                fade_color,
            };
        }

        ParticleKind::FireworkOverlay => {
            default_tick(p, world);
        }

        // java: `EntityAuraFX.onUpdate` — move, damp all three axes by `0.99`, then the
        // countdown. No gravity term at all, and `noClip` is set at spawn.
        ParticleKind::Aura => {
            p.prev_pos = p.pos;
            collide(p, world);
            p.motion *= 0.99;
            countdown_and_check_expiry(p);
        }

        // java: `EntitySuspendFX.onUpdate` — move, die if the block it now sits in is not water,
        // then the countdown. The two death conditions are separate `if`s in vanilla, not an
        // if/else, so a particle can satisfy both in the same tick.
        ParticleKind::Suspend => {
            p.prev_pos = p.pos;
            collide(p, world);
            let block = world.block_at(mc_math::BlockPos::from_vec3(p.pos));
            if mc_data::block_props(block).material != mc_data::Material::Water {
                p.dead = true;
            }
            countdown_and_check_expiry(p);
        }

        // java: `EntityFishWakeFX.onUpdate`. `i = 60 - particleMaxAge` grows as the countdown
        // runs down, driving both the size ramp and the 4-frame texture cycle 19..22.
        ParticleKind::FishWake => {
            p.prev_pos = p.pos;
            p.motion.y -= f64::from(p.gravity);
            collide(p, world);
            p.motion *= 0.980_000_019_073_486_3;
            let i = 60i64 - i64::from(p.max_age);
            let f = i as f32 * 0.001;
            p.size = (f, f);
            set_tex_index(p, 19 + (i.rem_euclid(4)) as i32);
            countdown_and_check_expiry(p);
        }

        // java: `EntityHeartFX.onUpdate` — the `Note`/`Reddust` shape with a `0.86` damp and no
        // texture animation (the index is fixed at spawn).
        ParticleKind::Heart { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            collide(p, world);
            if p.pos.y == p.prev_pos.y {
                p.motion.x *= 1.1;
                p.motion.z *= 1.1;
            }
            p.motion *= 0.860_000_014_305_114_7;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        // java: `EntityNoteFX.onUpdate` — as `Heart`, with a `0.66` damp.
        ParticleKind::Note { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            collide(p, world);
            if p.pos.y == p.prev_pos.y {
                p.motion.x *= 1.1;
                p.motion.z *= 1.1;
            }
            p.motion *= 0.660_000_026_226_043_7;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        // java: `EntitySnowShovelFX.onUpdate`. Its own `-0.03` gravity term (not
        // `BASE_GRAVITY_ACCEL * particleGravity`) and a `0.99` damp.
        ParticleKind::SnowShovel { .. } => {
            p.prev_pos = p.pos;
            age_and_check_expiry(p);
            set_tex_index(p, 7 - (p.age * 8 / p.max_age.max(1)) as i32);
            p.motion.y -= 0.03;
            collide(p, world);
            p.motion *= 0.990_000_009_536_743_2;
            if p.on_ground {
                p.motion.x *= GROUND_FRICTION;
                p.motion.z *= GROUND_FRICTION;
            }
        }

        // java: `Barrier` does not override `onUpdate`, so it takes `EntityFX`'s — which, with
        // `particleGravity == 0` and zero motion, is just the age check.
        ParticleKind::Barrier => {
            default_tick(p, world);
        }
    }
    out
}

/// `BlockLiquid.getLiquidHeightPercent` (`BlockLiquid.java:48-57`). Despite its name this is the
/// fraction of a liquid cell that is *air*: levels `0..=7` map to `(level + 1) / 9`, while a
/// falling level (`>= 8`) is treated as a source (`0`). The arithmetic stays in `f32` until the
/// result is widened because that is the order in both particle classes' bytecode.
fn liquid_height_percent(meta: u8) -> f32 {
    let level = if meta >= 8 { 0 } else { meta };
    f32::from(level + 1) / 9.0
}

/// `EntityRainFX.onUpdate:58-84`: rain dies below a liquid's rendered surface or below a solid
/// block's state-dependent render-bound top face. The latter is deliberately `bounds.max_y`, not
/// the collision shape: vanilla calls `setBlockBoundsBasedOnState` then `getBlockBoundsMaxY`.
fn rain_submerge_check(p: &mut Particle, world: &World) {
    let pos = mc_math::BlockPos::from_vec3(p.pos);
    let state = world.block_at(pos);
    let props = mc_data::block_props(state);
    let surface = if props.material.is_liquid() {
        f64::from(pos.y) + f64::from(1.0_f32 - liquid_height_percent(state.meta()))
    } else if props.material.is_solid() {
        crate::terrain::selected_bounding_box(world, pos, state)
            .max
            .y
    } else {
        return;
    };
    if p.pos.y < surface {
        p.dead = true;
    }
}

/// `EntityDropParticleFX.onUpdate:124-141`: liquids use the same rendered surface as rain, but
/// every solid uses the top of its entire block cell (`d0` remains zero in vanilla). Keeping this
/// separate from [`rain_submerge_check`] is load-bearing for partial blocks such as slabs.
fn drip_submerge_check(p: &mut Particle, world: &World) {
    let pos = mc_math::BlockPos::from_vec3(p.pos);
    let state = world.block_at(pos);
    let material = mc_data::block_props(state).material;
    let air_fraction = if material.is_liquid() {
        f64::from(liquid_height_percent(state.meta()))
    } else if material.is_solid() {
        0.0
    } else {
        return;
    };
    if p.pos.y < f64::from(pos.y + 1) - air_fraction {
        p.dead = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mc_data::{block_by_name, BlockState};
    use mc_math::BlockPos;

    fn rng() -> JavaRandom {
        JavaRandom::new(99)
    }

    fn place(world: &mut World, pos: BlockPos, state: BlockState) {
        let chunk = pos.chunk_pos();
        if !world.is_chunk_loaded(chunk) {
            let len = mc_world::expected_chunk_data_len(1, world.has_sky_light(), true);
            world
                .apply_chunk_data(chunk, true, 1, &vec![0; len])
                .expect("an exact-length blank chunk payload decodes");
        }
        world.set_block(pos, state);
        assert_eq!(world.block_at(pos), state, "test block write must land");
    }

    #[test]
    fn generic_falls_and_dies_at_max_age() {
        let world = World::new();
        let mut r = rng();
        let mut p = spawn::digging(
            &mut r,
            DVec3::new(0.0, 100.0, 0.0),
            DVec3::ZERO,
            super::super::types::ParticleIcon {
                min_u: 0.0,
                max_u: 1.0,
                min_v: 0.0,
                max_v: 1.0,
            },
            spawn::DEFAULT_DIGGING_TINT,
            true,
            mc_math::BlockPos::new(0, 100, 0),
        );
        p.max_age = 3;
        // `age++ >= maxAge` checks the *old* age before incrementing, so death lands on the
        // tick where `age` reaches `3` going in — the 4th call, not the 3rd.
        for i in 0..4 {
            let out = tick_one(&mut p, &world, &mut r);
            assert!(out.spawn.is_empty());
            assert_eq!(p.dead, i == 3);
        }
        assert!(p.dead);
        assert!(p.pos.y < 100.0, "gravity pulled it down");
    }

    #[test]
    fn crit_colour_fade_widens_to_double_before_multiplication() {
        let discriminating = |multiplier: f64| {
            (0x3f00_0000_u32..=0x3f80_0000)
                .step_by(997)
                .map(f32::from_bits)
                .find(|&value| (f64::from(value) * multiplier) as f32 != value * multiplier as f32)
                .expect("sampled normal f32 values must expose the narrowing-order difference")
        };
        let green = discriminating(0.96);
        let blue = discriminating(0.9);
        let expected_green = (f64::from(green) * 0.96) as f32;
        let expected_blue = (f64::from(blue) * 0.9) as f32;
        assert_ne!(expected_green, green * 0.96_f32);
        assert_ne!(expected_blue, blue * 0.9_f32);

        let world = World::new();
        let mut r = rng();
        let mut p = spawn::crit(&mut r, DVec3::ZERO, DVec3::ZERO, 1.0, false);
        p.color[1] = green;
        p.color[2] = blue;
        p.max_age = 100;
        tick_one(&mut p, &world, &mut r);
        assert_eq!(p.color[1].to_bits(), expected_green.to_bits());
        assert_eq!(p.color[2].to_bits(), expected_blue.to_bits());
    }

    #[test]
    fn smoke_animates_texture_index_downward_and_fades_in() {
        let world = World::new();
        let mut r = rng();
        let mut p = spawn::smoke(&mut r, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO, 1.0);
        p.max_age = 8;
        // `EntitySmokeFX`'s constructor never calls `setParticleTextureIndex`, so the very first
        // frame (before any tick) would show vanilla's `Entity`-default index `0` too.
        assert_eq!(p.tex_x + p.tex_y * 16, 0);
        tick_one(&mut p, &world, &mut r);
        let start_tex = p.tex_x + p.tex_y * 16;
        assert_eq!(start_tex, 6, "age is 1 after the first tick: 7 - 1*8/8 = 6");
        for _ in 0..6 {
            tick_one(&mut p, &world, &mut r);
        }
        let end_tex = p.tex_x + p.tex_y * 16;
        assert!(end_tex < start_tex, "{end_tex} should be < {start_tex}");
    }

    /// `EntityCloudFX.onUpdate`: look up the closest player only after moving and damping, then
    /// ease the cloud down toward that player's feet and its Y velocity toward player motion.
    #[test]
    fn cloud_homes_to_the_nearest_player_after_its_move() {
        let world = World::new();
        let mut r = rng();
        let mut cloud = spawn::cloud(&mut r, DVec3::new(0.0, 2.0, 0.0), DVec3::ZERO);
        cloud.motion = DVec3::new(0.0, 1.0, 0.0);
        cloud.max_age = 100;

        tick_one_with_players(
            &mut cloud,
            &world,
            &mut r,
            &[
                ParticlePlayer {
                    pos: DVec3::new(0.0, 2.0, 0.0),
                    velocity: DVec3::new(0.0, 4.0, 0.0),
                },
                ParticlePlayer {
                    pos: DVec3::new(0.0, 0.0, 0.0),
                    velocity: DVec3::new(0.0, -9.0, 0.0),
                },
            ],
        );

        // The cloud first moves 2 -> 3, damps 1 -> .96, then selects the player at y=2 rather
        // than the farther y=0 player. Both calculations are double precision in vanilla.
        assert!((cloud.pos.y - 2.8).abs() < 1e-12);
        assert!((cloud.motion.y - 1.568).abs() < 1e-12);
    }

    /// `World.getClosestPlayer` requires `d1 < distance * distance`; a player exactly two
    /// blocks below is eligible under the Cloud's height condition but must not be selected.
    #[test]
    fn cloud_homing_excludes_the_exact_two_block_boundary() {
        let world = World::new();
        let mut r = rng();
        let mut cloud = spawn::cloud(&mut r, DVec3::new(0.0, 3.0, 0.0), DVec3::ZERO);
        cloud.motion = DVec3::ZERO;
        cloud.max_age = 100;

        tick_one_with_players(
            &mut cloud,
            &world,
            &mut r,
            &[ParticlePlayer {
                pos: DVec3::new(0.0, 1.0, 0.0),
                velocity: DVec3::new(0.0, 7.0, 0.0),
            }],
        );

        assert_eq!(cloud.pos.y, 3.0);
        assert_eq!(cloud.motion.y, 0.0);
    }

    /// java: `EntityFlameFX` jitters its position by up to 0.05 per axis but not its bounding
    /// box, and the `noClip` `moveEntity` rebuilds the position from the box. A vanilla flame
    /// therefore sits on its spawn point (plus its tiny motion) from the first tick on.
    #[test]
    fn flame_spawn_jitter_is_dropped_by_the_first_move() {
        let world = World::new();
        let spawn_at = DVec3::new(10.5, 64.7, -3.5);
        let mut largest_jitter = 0.0_f64;
        for seed in 0..32 {
            let mut r = JavaRandom::new(seed);
            let mut p = spawn::flame(&mut r, spawn_at, DVec3::ZERO);
            p.max_age = 100;
            let jittered = p.pos;
            largest_jitter = largest_jitter.max((jittered - spawn_at).length());
            assert_eq!(p.prev_pos, spawn_at, "seed {seed}");
            let motion = p.motion;
            tick_one(&mut p, &world, &mut r);
            // `onUpdate` copies the jittered position into prevPos before it moves.
            assert_eq!(p.prev_pos, jittered, "seed {seed}");
            let expected = spawn_at + motion;
            assert!(
                (p.pos - expected).length() < 1e-9,
                "seed {seed}: {:?}, expected {expected:?}",
                p.pos
            );
            // Later ticks move on from there.
            let (before, motion) = (p.pos, p.motion);
            tick_one(&mut p, &world, &mut r);
            assert!((p.pos - (before + motion)).length() < 1e-9, "seed {seed}");
        }
        // Control: the jitter is large enough that keeping it would fail the checks above.
        assert!(largest_jitter > 0.01, "largest jitter {largest_jitter}");
    }

    #[test]
    fn lava_occasionally_spawns_smoke() {
        let world = World::new();
        let mut r = rng();
        let mut p = spawn::lava(&mut r, DVec3::new(0.0, 70.0, 0.0));
        p.max_age = 50;
        let mut spawned_any = false;
        for _ in 0..50 {
            let out = tick_one(&mut p, &world, &mut r);
            if !out.spawn.is_empty() {
                spawned_any = true;
                assert!(matches!(out.spawn[0].kind, ParticleKind::Smoke { .. }));
            }
        }
        assert!(
            spawned_any,
            "50 ticks should trigger at least one smoke puff"
        );
    }

    #[test]
    fn portal_eases_back_to_origin_and_dies_on_time() {
        let world = World::new();
        let mut r = rng();
        let origin = DVec3::new(5.0, 70.0, 5.0);
        let mut p = spawn::portal(&mut r, origin, DVec3::new(1.0, 0.0, 0.0));
        p.max_age = 10;
        // Portal reads the *old* `age` to ease position and only checks/increments afterward,
        // so (like the post-increment kinds, just via a different code shape) death lands one
        // tick later than `max_age` ticks.
        for _ in 0..11 {
            tick_one(&mut p, &world, &mut r);
        }
        assert!(p.dead);
    }

    #[test]
    fn bubble_dies_immediately_outside_water() {
        let world = World::new(); // no water anywhere
        let mut r = rng();
        let mut p = spawn::bubble(&mut r, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO);
        tick_one(&mut p, &world, &mut r);
        assert!(p.dead);
    }

    #[test]
    fn huge_explode_spawns_six_large_explosions_per_tick_for_eight_ticks() {
        let world = World::new();
        let mut r = rng();
        let mut p = spawn::huge_explode(&mut r, DVec3::ZERO);
        let mut total = 0;
        for i in 0..8 {
            let out = tick_one(&mut p, &world, &mut r);
            assert_eq!(out.spawn.len(), 6);
            total += out.spawn.len();
            assert_eq!(p.dead, i == 7);
        }
        assert_eq!(total, 48);
        assert!(p.dead);
    }

    #[test]
    fn firework_trail_child_starts_half_faded_and_inherits_only_twinkle() {
        let world = World::new();
        for parent_twinkle in [false, true] {
            let mut r = rng();
            let mut p = spawn::firework_spark(
                &mut r,
                DVec3::new(0.0, 70.0, 0.0),
                DVec3::ZERO,
                [1.0, 0.5, 0.0],
                Some([0.0, 0.5, 1.0]),
                true,
                parent_twinkle,
            );
            p.max_age = 20; // even, so `(age + max_age) % 2 == 0` happens on even ages too
            let mut found_child = false;
            for _ in 0..9 {
                let out = tick_one(&mut p, &world, &mut r);
                for child in out.spawn {
                    found_child = true;
                    assert_eq!(child.age, child.max_age / 2);
                    match child.kind {
                        ParticleKind::FireworkSpark {
                            trail,
                            twinkle,
                            fade_color,
                            ..
                        } => {
                            // Java constructs a fresh child (so trail stays false), copies the
                            // fade colour when present, and explicitly assigns parent.twinkle.
                            assert!(!trail);
                            assert_eq!(twinkle, parent_twinkle);
                            assert_eq!(fade_color, Some([0.0, 0.5, 1.0]));
                        }
                        other => panic!("trail child had wrong kind: {other:?}"),
                    }
                }
            }
            assert!(found_child, "a trailing spark should have spawned a child");
        }
    }

    #[test]
    fn large_explode_is_a_pure_frame_counter() {
        let world = World::new();
        let mut r = rng();
        let mut p = spawn::explode_large(&mut r, DVec3::ZERO, 0.0);
        let max = match p.kind {
            ParticleKind::LargeExplode { max, .. } => max,
            _ => unreachable!(),
        };
        for i in 0..max {
            tick_one(&mut p, &world, &mut r);
            assert_eq!(p.dead, i + 1 >= max);
        }
    }

    /// `F-P2-client.particle-SWEEP-12`: `7 - age*8/maxAge` is `-1` on the particle's last live
    /// tick, and vanilla's truncating `i % 16 / i / 16` gives `(15, 0)` there. `rem_euclid` /
    /// `div_euclid` gave `(15, -1)`, and a negative row index puts V outside `[0, 1]`, which the
    /// sampler's `ClampToEdge` smears into a one-texel strip.
    ///
    /// The negative control is the first assertion: a *positive* index still decomposes the same
    /// way, so this cannot pass by `set_tex_index` having been turned into a constant.
    #[test]
    fn the_last_tick_of_an_animated_particle_lands_on_cell_15_0_not_row_minus_one() {
        let mut r0 = rng();
        let mut p = spawn::smoke(&mut r0, DVec3::ZERO, DVec3::ZERO, 1.0);
        set_tex_index(&mut p, 65);
        assert_eq!((p.tex_x, p.tex_y), (1, 4), "positive indices are unchanged");

        set_tex_index(&mut p, -1);
        assert_eq!(
            (p.tex_x, p.tex_y),
            (-1, 0),
            "java truncates toward zero: -1 % 16 == -1 and -1 / 16 == 0"
        );
        // The *row* is what matters for appearance: with `div_euclid` this was `-1`, putting V
        // outside `[0, 1]` where `ClampToEdge` smeared it. Column `-1` is vanilla's own value and
        // wraps to column 15 under the `Repeat` address mode `phase.rs` now sets.
        assert_eq!(p.tex_y, 0, "a negative row puts V outside the atlas");

        // And through the real tick: a smoke particle reaches `age == max_age` on its last live
        // tick, where `7 - 8 == -1`.
        let world = World::new();
        let mut r = rng();
        let mut p = spawn::smoke(&mut r, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO, 1.0);
        p.max_age = 4;
        for _ in 0..5 {
            if p.dead {
                break;
            }
            tick_one(&mut p, &world, &mut r);
        }
        assert!(p.tex_y >= 0, "tex_y went negative during a real lifetime");
    }

    /// `F-P2-client.particle-SWEEP-11`: `if (this.bobTimer-- > 0)` decrements whichever branch is
    /// taken, so `bobTimer` keeps going negative and the colour formula
    /// `16 / (40 - bobTimer + 16)` keeps shrinking — a lava drip cools from orange toward red.
    /// Decrementing only inside the `if` froze it.
    ///
    /// The negative control is the water drip: its colour is a constant in vanilla, so it must
    /// *not* move. That rules out "every drip's colour now drifts".
    #[test]
    fn a_lava_drip_keeps_cooling_after_its_bob_ends() {
        let world = World::new();
        let mut r = rng();
        let mut lava = spawn::drip(&mut r, DVec3::new(0.0, 70.0, 0.0), false);
        lava.max_age = 200;
        // Run past the 40-tick bob.
        for _ in 0..45 {
            tick_one(&mut lava, &world, &mut r);
        }
        let green_at_45 = lava.color[1];
        for _ in 0..60 {
            tick_one(&mut lava, &world, &mut r);
        }
        let green_at_105 = lava.color[1];
        assert!(
            green_at_105 < green_at_45 - 1e-4,
            "green must keep falling after the bob: {green_at_45} -> {green_at_105}"
        );

        let mut water = spawn::drip(&mut r, DVec3::new(0.0, 70.0, 0.0), true);
        water.max_age = 200;
        water.no_clip = true;
        for _ in 0..45 {
            tick_one(&mut water, &world, &mut r);
        }
        assert_eq!(
            [water.color[0], water.color[1], water.color[2]],
            [0.2, 0.3, 1.0],
            "a water drip's colour is a constant in vanilla"
        );
    }

    /// `F-P2-client.particle-SWEEP-10`: both vanilla checks use the precise liquid surface,
    /// rather than killing a particle anywhere in a liquid cell. The two probes straddle level
    /// four's surface, so replacing this with the old material-only check kills the upper probe,
    /// while dropping the check entirely lets the lower probe survive.
    #[test]
    fn rain_and_drips_die_at_the_liquid_surface_not_the_top_of_its_cell() {
        assert_eq!(
            liquid_height_percent(7).to_bits(),
            (8.0_f32 / 9.0).to_bits(),
            "the highest flowing level retains its metadata"
        );
        assert_eq!(
            liquid_height_percent(8).to_bits(),
            (1.0_f32 / 9.0).to_bits(),
            "a falling-liquid metadata value resets to source level before division"
        );

        let mut world = World::new();
        let cell = BlockPos::new(0, 70, 0);
        let water = BlockState::new(block_by_name("flowing_water").unwrap(), 4);
        place(&mut world, cell, water);

        let surface = 70.0 + f64::from(1.0_f32 - (5.0_f32 / 9.0));
        let mut r = rng();
        for is_rain in [true, false] {
            let mut below = if is_rain {
                spawn::rain(&mut r, DVec3::new(0.5, surface - 1.0e-6, 0.5))
            } else {
                spawn::drip(&mut r, DVec3::new(0.5, surface - 1.0e-6, 0.5), true)
            };
            let mut above = if is_rain {
                spawn::rain(&mut r, DVec3::new(0.5, surface + 1.0e-6, 0.5))
            } else {
                spawn::drip(&mut r, DVec3::new(0.5, surface + 1.0e-6, 0.5), true)
            };

            if is_rain {
                rain_submerge_check(&mut below, &world);
                rain_submerge_check(&mut above, &world);
            } else {
                drip_submerge_check(&mut below, &world);
                drip_submerge_check(&mut above, &world);
            }
            assert!(below.dead, "a particle below the water surface must die");
            assert!(
                !above.dead,
                "a particle above the water surface must remain alive"
            );
        }
    }

    /// Rain and drips intentionally diverge for partial solids. `EntityRainFX` uses the block's
    /// render-bound top; `EntityDropParticleFX` leaves `d0 = 0`, making its threshold the cell's
    /// top. The below-slab rain is the negative control for simply disabling rain's solid check.
    #[test]
    fn rain_uses_partial_block_bounds_but_drips_use_the_whole_solid_cell() {
        let mut world = World::new();
        let cell = BlockPos::new(0, 70, 0);
        let bottom_slab = BlockState::new(block_by_name("stone_slab").unwrap(), 0);
        place(&mut world, cell, bottom_slab);
        assert_eq!(bottom_slab.props().bounds.max_y, 0.5);

        let mut r = rng();
        let mut rain_below = spawn::rain(&mut r, DVec3::new(0.5, 70.49, 0.5));
        let mut rain_above = spawn::rain(&mut r, DVec3::new(0.5, 70.75, 0.5));
        let mut drip_above = spawn::drip(&mut r, DVec3::new(0.5, 70.75, 0.5), true);

        rain_submerge_check(&mut rain_below, &world);
        rain_submerge_check(&mut rain_above, &world);
        drip_submerge_check(&mut drip_above, &world);

        assert!(rain_below.dead, "rain below the slab top must die");
        assert!(!rain_above.dead, "rain above the slab top must survive");
        assert!(
            drip_above.dead,
            "a drip inside any part of a solid block cell must die"
        );
    }

    #[test]
    fn rain_uses_the_connected_wall_height_not_the_static_full_block_top() {
        let mut world = World::new();
        let cell = BlockPos::new(0, 70, 0);
        let wall = BlockState::new(block_by_name("cobblestone_wall").unwrap(), 0);
        for pos in [cell, cell.north(), cell.south()] {
            place(&mut world, pos, wall);
        }

        let live = crate::terrain::selected_bounding_box(&world, cell, wall);
        assert_eq!(live.max.y, 70.0 + 13.0 / 16.0);
        let mut particle = spawn::rain(&mut rng(), DVec3::new(0.5, 70.9, 0.5));
        rain_submerge_check(&mut particle, &world);
        assert!(
            !particle.dead,
            "a straight connected wall ends at 13/16; rain above it remains alive"
        );
    }

    /// The behaviour of every family added in this wave, in one pass: each must age out rather
    /// than live forever, and none may panic in `tick_one`.
    #[test]
    fn the_new_families_all_terminate() {
        let world = World::new();
        let mut r = rng();
        let pos = DVec3::new(0.0, 70.0, 0.0);
        let icon = super::super::types::ParticleIcon {
            min_u: 0.0,
            max_u: 1.0,
            min_v: 0.0,
            max_v: 1.0,
        };
        let mut particles = vec![
            spawn::aura(&mut r, pos, DVec3::ZERO, false),
            spawn::aura(&mut r, pos, DVec3::ZERO, true),
            spawn::fish_wake(&mut r, pos, DVec3::ZERO),
            spawn::heart(&mut r, pos, 2.0, false),
            spawn::heart(&mut r, pos, 2.0, true),
            spawn::note(&mut r, pos, 0.5, 2.0),
            spawn::snow_shovel(&mut r, pos, DVec3::ZERO, 1.0),
            spawn::barrier(&mut r, pos, icon),
            spawn::breaking_item(&mut r, pos, icon),
            spawn::footstep(&mut r, pos),
            spawn::mob_appearance(&mut r, pos),
        ];
        // `suspend` dies immediately out of water, which is correct and would mask the others.
        for p in &mut particles {
            let mut ticks = 0;
            while !p.dead && ticks < 500 {
                tick_one(p, &world, &mut r);
                ticks += 1;
            }
            assert!(p.dead, "{:?} never died in 500 ticks", p.kind);
        }

        // java: `EntitySuspendFX.onUpdate` kills it the first tick it is not in water.
        let mut suspended = spawn::suspend(&mut r, pos);
        tick_one(&mut suspended, &world, &mut r);
        assert!(suspended.dead, "a suspended mote out of water dies at once");
    }

    #[test]
    fn footstep_has_its_own_exact_200_tick_equality_expiry() {
        let world = World::new();
        let mut rng = rng();
        let mut p = spawn::footstep(&mut rng, DVec3::new(1.0, 70.0, 2.0));
        for _ in 0..199 {
            tick_one(&mut p, &world, &mut rng);
        }
        assert_eq!(p.age, 199);
        assert!(!p.dead, "footstep is live until its counter equals 200");
        tick_one(&mut p, &world, &mut rng);
        assert_eq!(p.age, 200);
        assert!(p.dead, "footstep dies exactly when ++age == maxAge");
        assert_eq!(p.motion, DVec3::ZERO, "footsteps never invoke base physics");
    }
}
