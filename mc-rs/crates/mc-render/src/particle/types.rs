//! Common particle state — the fields every vanilla `EntityFX` subclass carries, plus the
//! per-family behaviour tag that drives [`super::tick`] and [`super::billboard`].
//!
//! java: `client/particle/EntityFX.java`

use glam::DVec3;
use mc_math::Aabb;

/// Vanilla's default per-tick downward acceleration multiplier, applied as
/// `motionY -= 0.04 * particleGravity`. java: `EntityFX.onUpdate`
pub const BASE_GRAVITY_ACCEL: f64 = 0.04;

/// The three drag/friction constants every default `EntityFX.onUpdate` applies, in order:
/// per-tick velocity retention, and the extra retention while resting on the ground.
pub const DEFAULT_DRAG: f64 = 0.98;
/// Extra X/Z retention applied on top of [`DEFAULT_DRAG`] while `on_ground`.
pub const GROUND_FRICTION: f64 = 0.7;

/// Vanilla's four `fxLayers` texture buckets. java: `EffectRenderer.fxLayers` (outer dimension).
///
/// Layer index 2 is allocated by vanilla (`fxLayers[4][2]`) but nothing ever sets an
/// `EntityFX`'s `getFXLayer()` to `2`; it is dead vanilla code and has no counterpart here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FxLayer {
    /// `textures/particle/particles.png` — a 16x16 grid of misc particle icons. Layer 0.
    /// Smoke, flame, lava, portal, reddust, crit, bubble, rain, drip, spell, firework spark.
    Misc,
    /// The block/item atlas — block-break, block-dust, and item-break particles. Layer 1.
    Terrain,
    /// `textures/entity/explosion.png`, a 4x4 animated grid, drawn through
    /// `EffectRenderer.renderLitParticles` rather than the main blended batch. Layer 3.
    LitExplosion,
    /// `textures/particle/footprint.png` — `EntityFootStepFX`'s one-off, ground-aligned
    /// texture. Vanilla files footsteps in layer 3, but its custom `renderParticle` binds this
    /// texture and begins/draws its own quad, so it needs a separate draw state here.
    Footprint,
}

/// Depth-mask bucket. java: `EffectRenderer.fxLayers[i][j]` — `j=0` is alpha `< 1`
/// (`GlStateManager.depthMask(false)`), `j=1` is opaque (`depthMask(true)`).
///
/// [`Particle::alpha_bucket`] recomputes this from the live alpha value every frame, mirroring
/// `EntityFX.setAlphaF`'s `moveToAlphaLayer`/`moveToNoAlphaLayer` calls (which fire exactly when
/// alpha crosses the `1.0` boundary) without needing to replicate the move itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaBucket {
    /// `alpha < 1.0`: blended, no depth write.
    Translucent,
    /// `alpha == 1.0`: blended (vanilla still calls `enableBlend`), depth write on.
    Opaque,
}

/// Per-bucket particle cap. java: `EffectRenderer.addEffect` — `if (list.size() >= 4000)
/// list.remove(0)`.
pub const PER_BUCKET_CAP: usize = 4000;

/// A sprite's placement inside the [`FxLayer::Terrain`] atlas, as plain UV fractions — exactly
/// the numbers `EntityFX.renderParticle`'s icon branch reads off `TextureAtlasSprite`.
///
/// This is **not** `mc_assets::Sprite`: `particle/**` does not depend on how a block's particle
/// texture was resolved (baked-model quad lookup, a direct atlas stitch, or a test fixture), only
/// on where it landed. See [`super::phase`] for how a real one is produced from `mc_assets`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleIcon {
    /// java: `TextureAtlasSprite.getMinU()`
    pub min_u: f32,
    /// java: `getMaxU()`
    pub max_u: f32,
    /// java: `getMinV()`
    pub min_v: f32,
    /// java: `getMaxV()`
    pub max_v: f32,
}

impl ParticleIcon {
    /// java: `TextureAtlasSprite.getInterpolatedU` — `minU + (maxU - minU) * u / 16.0`.
    #[must_use]
    pub fn interpolated_u(&self, u: f32) -> f32 {
        self.min_u + (self.max_u - self.min_u) * (u / 16.0)
    }

    /// java: `getInterpolatedV`
    #[must_use]
    pub fn interpolated_v(&self, v: f32) -> f32 {
        self.min_v + (self.max_v - self.min_v) * (v / 16.0)
    }
}

/// The behavioural family a particle belongs to, plus whatever per-family state its `onUpdate`
/// and `renderParticle` overrides need. Maps one-to-one onto a vanilla `EntityFX` subclass.
#[derive(Debug, Clone, PartialEq)]
pub enum ParticleKind {
    /// No override: base `EntityFX.onUpdate`/`renderParticle`. Used directly by nothing in
    /// vanilla's registry, but it is the fallback for `EnumParticleTypes` this port has not
    /// given a dedicated family (footstep, heart, snowball poof, ...) — see [`super::spawn`].
    Generic,
    /// `EntitySmokeFX` (also `EntityCritFX`, which is just `EntitySmokeFX` with `scale_factor =
    /// 2.5`, registered as `SMOKE_LARGE`). Fades in over the first ~3% of its life, animates
    /// texture index 7 -> 0, and doubles X/Z drift when vertically stalled.
    Smoke {
        /// The peak scale this particle ramps up to. java: `smokeParticleScale`
        peak_scale: f32,
    },
    /// `EntityCrit2FX` (`CRIT` and `CRIT_MAGIC`). Fully overrides `onUpdate`: no world gravity
    /// term, its own decay constants, and a green/blue colour fade.
    Crit {
        /// java: `field_174839_a` — the scale the age-based ramp multiplies against.
        peak_scale: f32,
    },
    /// `EntityFlameFX`. `noClip = true`; scale shrinks quadratically and blocklight ramps up to
    /// full bright ([`Particle::light_boost`]) as it ages.
    Flame {
        /// java: `flameScale`
        peak_scale: f32,
    },
    /// `EntityLavaFX`. Always full-bright; occasionally emits a `SMOKE_NORMAL` child while
    /// airborne; scale shrinks as `1 - f*f`.
    Lava {
        /// java: `lavaParticleScale`
        peak_scale: f32,
    },
    /// `EntityReddustFX` (`REDSTONE`). Same fade-in/animate pattern as [`ParticleKind::Smoke`]
    /// but with an externally supplied colour (the wire packet or `spawn` caller sets RGB).
    Reddust {
        /// java: `reddustParticleScale`
        peak_scale: f32,
    },
    /// `EntityPortalFX`. `noClip = true`; position is re-derived every tick from an eased
    /// fraction of age rather than integrated from `motion`.
    Portal {
        /// The particle's fixed origin. java: `portalPosX/Y/Z`
        origin: DVec3,
    },
    /// `EntityEnchantmentTableParticleFX`. Same re-derived-position pattern as
    /// [`ParticleKind::Portal`], with an extra downward arc term.
    EnchantGlyph {
        /// java: `coordX/Y/Z`
        origin: DVec3,
    },
    /// `EntityBubbleFX` (`WATER_BUBBLE`). Dies the instant it leaves water, independent of the
    /// normal age/gravity path (it has its own tiny upward acceleration and no gravity field).
    Bubble,
    /// `EntityRainFX` (`WATER_DROP`) and `EntitySplashFX` (`WATER_SPLASH`, which is a `RainFX`
    /// with `particleGravity = 0.04` and a bumped texture index). Dies on a coin flip when it
    /// lands, or when it sinks below the liquid/solid surface it is falling through.
    Rain {
        /// `EntitySplashFX` sets this; `EntityRainFX` does not.
        is_splash: bool,
    },
    /// `EntityDropParticleFX` (`DRIP_WATER`/`DRIP_LAVA`). Holds still for `bob_timer` ticks,
    /// then falls; water drips spawn a splash and die on landing, lava drips just change frame.
    Drip {
        /// `true` for `DRIP_WATER`, `false` for `DRIP_LAVA`.
        is_water: bool,
        /// Ticks remaining before the drop starts falling. java: `bobTimer`, counts to 0.
        bob_timer: i32,
    },
    /// `EntityCloudFX` — the drifting ring `EntityHugeExplodeFX` spawns. Eases toward the
    /// nearest player's feet.
    Cloud {
        /// java: `field_70569_a`
        peak_scale: f32,
    },
    /// `EntityExplodeFX` (`EXPLOSION_NORMAL`, `"explode"`). The everyday "poof" particle.
    Explode,
    /// `EntityHugeExplodeFX` (`EXPLOSION_HUGE`). Invisible; spawns six `EXPLOSION_LARGE`
    /// particles per tick for 8 ticks, then dies. `getFXLayer() == 1` in vanilla but it never
    /// renders anything itself (`renderParticle` is overridden to a no-op).
    HugeExplode {
        /// Ticks elapsed. java: `timeSinceStart`
        elapsed: u32,
        /// Always 8. java: `maximumTime`
        max: u32,
    },
    /// `EntityLargeExplodeFX` (`EXPLOSION_LARGE`). Renders on [`FxLayer::LitExplosion`] with a
    /// 4x4 animated frame from `textures/entity/explosion.png`; 6-9 tick lifetime.
    LargeExplode {
        /// java: `field_70584_aq` — `6 + rand.nextInt(4)`.
        max: u32,
        /// java: `field_70582_as` — `1.0 - xSpeed * 0.5`, scales the quad half-width.
        scale_factor: f32,
    },
    /// The `EntitySpellParticleFX` family (`SPELL`, `SPELL_INSTANT`, `SPELL_MOB`,
    /// `SPELL_MOB_AMBIENT`, `SPELL_WITCH`) — potion-effect swirl particles. The four factories
    /// differ only in `base_tex` (128 normal, 144 instant/witch) and initial colour/alpha, which
    /// [`super::spawn`] applies before the particle starts ticking.
    Spell {
        /// java: `baseSpellTextureIndex`
        base_tex: i32,
    },
    /// `EntityDiggingFX` / `EntityBlockDustFX` (`BLOCK_CRACK`/`BLOCK_DUST`, and the
    /// `addBlockDestroyEffects`/`addBlockHitEffects` mining calls). Uses the block's particle
    /// icon on [`FxLayer::Terrain`] with the 4-way jitter sub-rect. `is_dust` selects
    /// `EntityBlockDustFX`'s "keep the exact given velocity" behaviour over `EntityDiggingFX`'s
    /// normal randomised-velocity `EntityFX` construction.
    Digging {
        /// `true` for `EntityBlockDustFX`, `false` for plain digging/hit particles.
        is_dust: bool,
        /// java: `EntityDiggingFX.sourcePos`, set by `setBlockPos(pos)` on the
        /// `addBlockDestroyEffects`/`addBlockHitEffects` path and by `func_174845_l()` — which
        /// uses `new BlockPos(posX, posY, posZ)`, the particle's own spawn cell — on the packet
        /// path. Read only by `getBrightnessForRender`, as the light to fall back on when the
        /// particle's own cell reports zero.
        source_pos: mc_math::BlockPos,
    },
    /// `EntityBreakingFX` (`ITEM_CRACK`, and the `SlimeFactory`/`SnowballFactory` variants).
    /// Item-icon particle on [`FxLayer::Terrain`].
    Breaking,
    /// `EntityFirework.SparkFX`. Twinkles, optionally spawns trail children, fades toward
    /// `fade_color` after half its life.
    FireworkSpark {
        /// java: `baseTextureIndex`, always 160.
        base_tex: i32,
        /// java: `trail`
        trail: bool,
        /// java: `twinkle`
        twinkle: bool,
        /// java: `fadeColourRed/Green/Blue`, `hasFadeColour`.
        fade_color: Option<[f32; 3]>,
    },
    /// `EntityFirework.OverlayFX` — the single-frame flash drawn at the burst point.
    FireworkOverlay,
    /// `EntityAuraFX` (`TOWN_AURA`, `SUSPENDED_DEPTH`, and — with texture index 82 and white
    /// colour — `VILLAGER_HAPPY`). A tiny, near-motionless mote: `noClip`, `0.99` drag on all
    /// three axes, and the *countdown* expiry idiom (`if (particleMaxAge-- <= 0) setDead()`),
    /// not the `particleAge++` one.
    Aura,
    /// `EntitySuspendFX` (`SUSPENDED`) — the blue-grey underwater mote. Motionless, dies the
    /// instant the block it sits in stops being water, countdown expiry.
    Suspend,
    /// `EntityFishWakeFX` (`WATER_WAKE`). Zero gravity, `0.98` drag, and a size/texture ramp
    /// driven by `60 - particleMaxAge` as the countdown runs down.
    FishWake,
    /// `EntityHeartFX` (`HEART`, and `VILLAGER_ANGRY` at texture index 81). Rises, `0.86` drag,
    /// fixed 16-tick life, with the same `clamp(age/maxAge*32)` scale ramp `Smoke` uses.
    Heart {
        /// java: `particleScaleOverTime`
        peak_scale: f32,
    },
    /// `EntityNoteFX` (`NOTE`). Rises faster than a heart, `0.66` drag, fixed 6-tick life,
    /// colour derived from the note pitch by three phase-shifted sines.
    Note {
        /// java: `noteParticleScale`
        peak_scale: f32,
    },
    /// `EntitySnowShovelFX` (`SNOW_SHOVEL`). A near-white puff with its own `-0.03` gravity term
    /// and `0.99` drag; animates texture index `7 -> 0` like `Smoke` and ramps scale like it too.
    SnowShovel {
        /// java: `snowDigParticleScale`
        peak_scale: f32,
    },
    /// `Barrier` (`BARRIER`) — the barrier item's icon, drawn at a **fixed** half-block size with
    /// the sprite's full UV rect (no 4-way jitter sub-rect), motionless, for 80 ticks.
    Barrier,
    /// `EntityFootStepFX` (`FOOTSTEP`). Unlike every other particle family this is not a
    /// camera-facing billboard: it renders a fixed 0.25-block square on the ground using its
    /// standalone `footprint.png`, fading from alpha 0.2 to zero over exactly 200 ticks.
    Footstep,
    /// `MobAppearance` (`MOB_APPEARANCE`). The particle itself is an invisible 30-tick
    /// controller; its elder-guardian geometry is handed to the entity phase each frame.
    MobAppearance,
}

/// One live particle. java: `client/particle/EntityFX.java` and its subclasses' extra fields,
/// flattened onto one struct with [`ParticleKind`] carrying the behavioural differences.
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    /// Current position. java: `posX/Y/Z`
    pub pos: DVec3,
    /// Position at the start of this tick, for render interpolation. java: `prevPosX/Y/Z`
    pub prev_pos: DVec3,
    /// Velocity in blocks/tick. java: `motionX/Y/Z`
    pub motion: DVec3,
    /// Ticks this particle has existed. java: `particleAge`
    pub age: u32,
    /// Ticks after which this particle expires. java: `particleMaxAge`
    pub max_age: u32,
    /// Render scale. java: `particleScale`
    pub scale: f32,
    /// Multiplies [`BASE_GRAVITY_ACCEL`]. java: `particleGravity`
    pub gravity: f32,
    /// `[r, g, b, a]`, each `0.0..=1.0`. java: `particleRed/Green/Blue/Alpha`
    pub color: [f32; 4],
    /// Misc-atlas cell column. java: `particleTextureIndexX`. Only meaningful for
    /// [`FxLayer::Misc`] particles; icon-based ([`FxLayer::Terrain`]) particles ignore it.
    pub tex_x: i32,
    /// Misc-atlas cell row. java: `particleTextureIndexY`
    pub tex_y: i32,
    /// Per-particle sub-cell jitter, `0.0..3.0`. java: `particleTextureJitterX`
    pub jitter_x: f32,
    /// java: `particleTextureJitterY`
    pub jitter_y: f32,
    /// Collision box width/height in blocks. java: `Entity.width`/`height`, set via `setSize`.
    pub size: (f32, f32),
    /// Set by the last [`super::physics::sweep`] call. java: `Entity.onGround`
    pub on_ground: bool,
    /// Skips collision entirely when true. java: `Entity.noClip`
    pub no_clip: bool,
    /// Marked for removal at the end of this tick. java: `Entity.isDead` / `setDead`
    pub dead: bool,
    /// Which texture batch this particle draws in.
    pub layer: FxLayer,
    /// The atlas rect for [`FxLayer::Terrain`] particles ([`ParticleKind::Digging`] and
    /// [`ParticleKind::Breaking`]). `None` for every other layer. java: `particleIcon`
    pub icon: Option<ParticleIcon>,
    /// The behavioural family and its private state.
    ///
    /// A handful of vanilla fields are deliberately **not** stored here because vanilla itself
    /// only ever computes them inside `renderParticle`/`getBrightnessForRender`, from `age`,
    /// `max_age`, and that call's `partialTicks` — never reading them back from simulation state.
    /// [`super::billboard`] recomputes both fresh every frame instead of caching a value that
    /// would need re-deriving per-frame anyway:
    ///
    /// * The render **scale ramp** ([`ParticleKind::Smoke`], [`ParticleKind::Crit`],
    ///   [`ParticleKind::Flame`], [`ParticleKind::Lava`], [`ParticleKind::Cloud`],
    ///   [`ParticleKind::Reddust`], [`ParticleKind::Portal`], [`ParticleKind::EnchantGlyph`]) —
    ///   each variant's `peak_scale`/`origin` is the ramp's input, not [`Particle::scale`].
    ///   [`Particle::scale`] stays the spawn-time value and is what non-ramping kinds render at.
    /// * The **light boost** toward full-bright (flame, lava, portal, enchantment glyph, the
    ///   large explosion) — a simplified stand-in for vanilla's `getBrightnessForRender` byte
    ///   manipulation, since without S20's real lightmap texture (see the `light_curve`
    ///   placeholder note in `terrain.wgsl`) there is no lightmap to write an exact byte into.
    pub kind: ParticleKind,
}

impl Particle {
    /// The collision/render bounding box, vanilla's `Entity.setPosition`: centred in X/Z, feet
    /// at `pos.y`.
    #[must_use]
    pub fn bounding_box(&self) -> Aabb {
        let hw = f64::from(self.size.0) / 2.0;
        let h = f64::from(self.size.1);
        Aabb::from_bounds(
            self.pos.x - hw,
            self.pos.y,
            self.pos.z - hw,
            self.pos.x + hw,
            self.pos.y + h,
            self.pos.z + hw,
        )
    }

    /// Recentres [`Particle::pos`] from a bounding box that has just been moved by the collision
    /// sweep — the inverse of [`Particle::bounding_box`]. java: `Entity.setPosition` as called
    /// from `Entity.moveEntity` with the swept box's centre-bottom.
    pub fn set_pos_from_box(&mut self, bb: Aabb) {
        let hw = f64::from(self.size.0) / 2.0;
        self.pos = DVec3::new(bb.min.x + hw, bb.min.y, bb.min.z + hw);
    }

    /// java: `EntityFX.setAlphaF` — which layer this particle draws in this frame, purely a
    /// function of its current alpha.
    #[must_use]
    pub fn alpha_bucket(&self) -> AlphaBucket {
        if self.color[3] < 1.0 {
            AlphaBucket::Translucent
        } else {
            AlphaBucket::Opaque
        }
    }

    /// World-space position interpolated between the previous and current tick.
    /// java: `prevPosX + (posX - prevPosX) * partialTicks` (and Y, Z).
    #[must_use]
    pub fn interpolated_pos(&self, partial_ticks: f32) -> DVec3 {
        self.prev_pos + (self.pos - self.prev_pos) * f64::from(partial_ticks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(kind: ParticleKind) -> Particle {
        Particle {
            pos: DVec3::ZERO,
            prev_pos: DVec3::ZERO,
            motion: DVec3::ZERO,
            age: 0,
            max_age: 10,
            scale: 1.0,
            gravity: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            tex_x: 0,
            tex_y: 0,
            jitter_x: 0.0,
            jitter_y: 0.0,
            size: (0.2, 0.2),
            on_ground: false,
            no_clip: false,
            dead: false,
            layer: FxLayer::Misc,
            icon: None,
            kind,
        }
    }

    #[test]
    fn bounding_box_is_centred_in_xz_feet_at_y() {
        let mut p = base(ParticleKind::Generic);
        p.pos = DVec3::new(1.0, 2.0, 3.0);
        p.size = (0.2, 0.4);
        let bb = p.bounding_box();
        // `size` is `f32`, widened to `f64` before halving (matching Java's `(double)this.width
        // / 2.0D`), so the result carries `f32`-precision error rather than landing on an exact
        // decimal — approximate rather than exact comparison.
        assert!((bb.min - DVec3::new(0.9, 2.0, 2.9)).length() < 1e-6);
        assert!((bb.max - DVec3::new(1.1, 2.4, 3.1)).length() < 1e-6);
    }

    #[test]
    fn set_pos_from_box_round_trips() {
        let mut p = base(ParticleKind::Generic);
        p.pos = DVec3::new(5.0, 10.0, -3.0);
        p.size = (0.2, 0.2);
        let mut bb = p.bounding_box();
        bb = bb.offset(1.0, -1.0, 0.5);
        p.set_pos_from_box(bb);
        assert!((p.pos - DVec3::new(6.0, 9.0, -2.5)).length() < 1e-9);
    }

    #[test]
    fn alpha_bucket_follows_alpha_exactly_at_one() {
        let mut p = base(ParticleKind::Generic);
        p.color[3] = 1.0;
        assert_eq!(p.alpha_bucket(), AlphaBucket::Opaque);
        p.color[3] = 0.999;
        assert_eq!(p.alpha_bucket(), AlphaBucket::Translucent);
    }

    #[test]
    fn interpolation_matches_java_lerp() {
        let mut p = base(ParticleKind::Generic);
        p.prev_pos = DVec3::new(0.0, 0.0, 0.0);
        p.pos = DVec3::new(2.0, 0.0, 0.0);
        assert_eq!(p.interpolated_pos(0.25), DVec3::new(0.5, 0.0, 0.0));
        assert_eq!(p.interpolated_pos(1.0), p.pos);
        assert_eq!(p.interpolated_pos(0.0), p.prev_pos);
    }
}
