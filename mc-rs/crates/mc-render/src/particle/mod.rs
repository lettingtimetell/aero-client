//! Particle simulation and batched rendering — S18.
//!
//! java: `client/particle/` — `EffectRenderer.java` (the arena, GL state, per-frame draw loop)
//! and 30-odd `EntityFX` subclasses (per-family spawn/tick/render behaviour).
//!
//! # Layout
//!
//! | module | java source | what it owns |
//! |---|---|---|
//! | [`types`] | `EntityFX`'s fields | [`types::Particle`], [`types::ParticleKind`], [`types::FxLayer`] |
//! | [`physics`] | `Entity.moveEntity` | the collision sweep |
//! | [`spawn`] | every `EntityFX` constructor | one function per family |
//! | [`tick`] | every `onUpdate` override | per-tick behaviour, dispatched on [`types::ParticleKind`] |
//! | [`billboard`] | `renderParticle`, `ActiveRenderInfo` | camera-facing quad generation |
//! | [`system`] | `EffectRenderer` | the arena: spawn/tick/expire/cap, the density gate |
//! | [`packet`] | `S2APacketParticles`, `EnumParticleTypes` | server-triggered particle decode + dispatch |
//! | [`phase`] | `EffectRenderer.renderParticles`/`renderLitParticles` | the `PARTICLES` wgpu phase |
//! | [`tuning`] | (none: the particles mod) | per-group amount/size, extra hit emitters |
//!
//! # A minimal integration
//!
//! ```no_run
//! use mc_render::particle::{system::ParticleSystem, phase::ParticlePhase, spawn};
//! use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder};
//! use mc_math::JavaRandom;
//! use mc_world::World;
//! use glam::DVec3;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let gpu = Gpu::headless(64, 64, GpuOptions::default())?;
//! let mut phase = ParticlePhase::new(&gpu);
//! phase.install_misc_atlas(&gpu, &vec![255u8; 16 * 16 * 4], 16, 16);
//!
//! let world = World::new();
//! let mut particles = ParticleSystem::new();
//! let mut rng = JavaRandom::new(0);
//! let p = spawn::smoke(&mut rng, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO, 1.0);
//! particles.spawn(&world, p);
//!
//! // Once per tick:
//! particles.tick(&world);
//!
//! // Once per frame, before `PhaseList::render`:
//! let cam = Camera::default();
//! phase.upload(&gpu, particles.particles(), &cam, 0.0, &world);
//!
//! let mut phases = PhaseList::new();
//! phases.insert(PhaseOrder::PARTICLES, Box::new(phase));
//! # Ok(())
//! # }
//! ```

pub mod appearance;
pub mod billboard;
pub mod packet;
pub mod phase;
pub mod physics;
pub mod spawn;
pub mod system;
pub mod tick;
// The particles mod's per-group amount/size and hit emitters.
pub mod tuning;
pub mod types;

pub use appearance::{mob_appearance_draw, MobAppearanceDraw, MOB_APPEARANCE_TEXTURE};
pub use billboard::ParticleVertex;
pub use packet::{IconLookup, S2APacketParticles};
pub use phase::ParticlePhase;
pub use system::{ParticleDensity, ParticlePlayer, ParticleSystem};
pub use types::{AlphaBucket, FxLayer, Particle, ParticleIcon, ParticleKind};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use glam::DVec3;
    use mc_math::JavaRandom;
    use mc_world::World;

    /// A full spawn -> several ticks -> death cycle through the public API only, exercising
    /// [`ParticleSystem`], [`spawn`], and [`tick`] together rather than each in isolation.
    #[test]
    fn full_lifecycle_through_the_public_api() {
        let world = World::new();
        let mut system = ParticleSystem::with_seed(42);
        let mut rng = JavaRandom::new(42);

        let mut p = spawn::explode(&mut rng, DVec3::new(0.0, 90.0, 0.0), DVec3::ZERO);
        p.max_age = 5;
        system.spawn(&world, p);
        assert_eq!(system.len(), 1);

        // Post-increment expiry: dies once `age` (checked before incrementing) reaches 5, i.e.
        // on the 6th tick.
        for _ in 0..6 {
            system.tick(&world);
        }
        assert_eq!(system.len(), 0);
    }

    /// A server-driven block-break burst (the shape `EffectRenderer.addBlockDestroyEffects`
    /// produces: 64 digging particles) round-tripped through density gating and the arena.
    #[test]
    fn block_destroy_burst_respects_density() {
        let world = World::new();
        let mut system = ParticleSystem::with_seed(7);
        system.set_density(ParticleDensity::Minimal);
        let icon = ParticleIcon {
            min_u: 0.0,
            max_u: 0.5,
            min_v: 0.0,
            max_v: 0.5,
        };

        for i in 0..64 {
            let pos = DVec3::new(f64::from(i) * 0.01, 64.0, 0.0);
            if !system.should_spawn(false, DVec3::ZERO, pos) {
                continue;
            }
            let rng = system.rng_mut();
            let particle = spawn::digging(
                rng,
                pos,
                DVec3::ZERO,
                icon,
                spawn::DEFAULT_DIGGING_TINT,
                false,
                mc_math::BlockPos::from_vec3(pos),
            );
            system.spawn(&world, particle);
        }
        assert_eq!(
            system.len(),
            0,
            "minimal density drops every range-respecting spawn attempt"
        );
    }
}
