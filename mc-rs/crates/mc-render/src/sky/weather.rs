//! Rain, snow, and lightning. java: `EntityRenderer.renderRainSnow` (line 1594 of the
//! 2,050-line file — confirmed by research to live in `EntityRenderer.java`, not
//! `RenderGlobal.java`, despite the latter owning the rest of the sky/weather geometry).
//!
//! `RenderPhase::record` is given `gpu`/`enc`/`view`/`cam` only — no world reference — so
//! [`build_columns`] is a pure function decoupled behind [`ColumnEnvironment`], exactly the way
//! [`crate::terrain::mesh_section`] is a pure function of `&World`. A caller (S21, eventually)
//! calls [`build_columns`] once per tick and hands the result to [`WeatherPhase::upload`]; the
//! phase itself only draws whatever was last uploaded. This mirrors
//! [`crate::terrain::phase::TerrainRenderer::upload`]'s external-driver pattern.

use glam::{DVec3, Vec2, Vec3};
use mc_math::JavaRandom;

use crate::sky::environment::{Dimension, Environment};
use crate::sky::textures::{self, SkyTexture};
use crate::sky::WorldColumns;
use crate::{Camera, DepthAttachment, Gpu, RenderPhase};
use mc_world::world::{ambient_particle, AmbientEffect, AmbientParticleEffect, AmbientSoundEffect};

/// The most precipitation samples one [`add_rain_particles`] pass takes:
/// a rain level of about 18 (100 samples at the vanilla maximum of 1). See the cap's comment.
pub const MAX_RAIN_SAMPLES: i32 = 1 << 15;

/// The result of one `EntityRenderer.addRainParticles` pass.
///
/// The render crate produces the same particle/sound-neutral recipes as the world-side ambient
/// callbacks. `rain_sound_counter` is returned because vanilla keeps this counter in the
/// renderer, incrementing it only when at least one solid precipitation impact was sampled.
#[derive(Debug, Clone, PartialEq)]
pub struct RainAmbientPass {
    /// Water-drop/smoke particle and positional rain-sound requests.
    pub effects: Vec<AmbientEffect>,
    /// The counter value to persist into the next client tick.
    pub rain_sound_counter: i32,
}

/// Runs Java 1.8.9's local rain-impact producer.
///
/// This is separate from [`build_columns`]: the latter builds the translucent curtain, while
/// this method samples raw precipitation-height blocks around the view entity and emits impact
/// particles plus `ambient.weather.rain`. `random` is the renderer-owned stream and is reseeded
/// from `renderer_update_count * 312987231L` each call, exactly as `EntityRenderer` does. The
/// world itself is read-only; the caller owns particle density/range filtering and audio sinks.
#[must_use]
pub fn add_rain_particles(
    world: &mc_world::World,
    camera_pos: DVec3,
    renderer_update_count: i64,
    fancy_graphics: bool,
    particle_setting: i32,
    rain_strength: f32,
    rain_sound_counter: i32,
) -> RainAmbientPass {
    let mut strength = rain_strength;
    if !fancy_graphics {
        strength /= 2.0;
    }
    let mut counter = rain_sound_counter;
    if strength == 0.0 {
        return RainAmbientPass {
            effects: Vec::new(),
            rain_sound_counter: counter,
        };
    }

    let seed = renderer_update_count.wrapping_mul(312_987_231);
    let mut random = JavaRandom::new(seed);
    let camera_block = mc_math::BlockPos::from_f64(camera_pos.x, camera_pos.y, camera_pos.z);
    let range = 10;
    let mut sound_x = 0.0;
    let mut sound_y = 0.0;
    let mut sound_z = 0.0;
    let mut solid_impacts = 0_i32;
    let mut effects = Vec::new();
    let mut count = (100.0 * strength * strength) as i32;
    match particle_setting {
        1 => count >>= 1,
        2 => count = 0,
        _ => {}
    }
    // The rain level is a server float (S2B reason 7) that vanilla
    // stores unclamped, so a huge one makes this `(int)(100 * f * f)` loop `Integer.MAX_VALUE`
    // samples long on every tick: a permanent freeze in vanilla, and here an `effects` list that
    // grows until memory runs out. Real rain levels are 0..1 (100 samples); the cap is a level
    // of about 18, so every level a server plausibly sends samples exactly as before.
    count = count.min(MAX_RAIN_SAMPLES);

    let columns = WorldColumns(world);
    for _ in 0..count.max(0) {
        let sample = camera_block.add(
            random.next_i32_bound(range) - random.next_i32_bound(range),
            0,
            random.next_i32_bound(range) - random.next_i32_bound(range),
        );
        let precipitation_y = columns.precipitation_height(sample.x, sample.z);
        let precipitation = mc_math::BlockPos::new(sample.x, precipitation_y, sample.z);
        // SKY-6: Java's first `&&` term, tested before the reads the rest of the condition
        // needs. Those reads are pure and this sample's draws all sit inside the branch, so an
        // impact outside ±10 of the camera (on void maps: most of them, landing on empty
        // columns at height 0) skips them with identical results and RNG state.
        if precipitation.y > camera_block.y + range || precipitation.y < camera_block.y - range {
            continue;
        }
        let below = precipitation.down();
        // Java's renderer reads the loaded world column; an absent client chunk must not leak
        // the mutable shared blank-chunk scratch state into rain particles/audio.
        let state = world
            .loaded_block_at(below)
            .unwrap_or(mc_data::BlockState::AIR);
        let biome = mc_data::biome(world.biome_at(precipitation));
        let temperature = columns.temperature_at(precipitation.x, precipitation.y, precipitation.z);
        if biome.rain && !biome.snow && temperature >= 0.15 {
            let dx = random.next_f64();
            let dz = random.next_f64();
            let props = state.props();
            if props.material == mc_data::Material::Lava {
                effects.push(AmbientEffect::Particle(AmbientParticleEffect {
                    particle_id: ambient_particle::SMOKE_NORMAL,
                    position: mc_math::DVec3::new(
                        f64::from(precipitation.x) + dx,
                        f64::from(precipitation.y) + 0.1 - props.bounds.min_y,
                        f64::from(precipitation.z) + dz,
                    ),
                    velocity: mc_math::DVec3::ZERO,
                }));
            } else if props.material != mc_data::Material::Air {
                solid_impacts += 1;
                if random.next_i32_bound(solid_impacts) == 0 {
                    sound_x = f64::from(below.x) + dx;
                    sound_y = f64::from(below.y) + 0.1 + props.bounds.max_y - 1.0;
                    sound_z = f64::from(below.z) + dz;
                }
                effects.push(AmbientEffect::Particle(AmbientParticleEffect {
                    particle_id: ambient_particle::WATER_DROP,
                    position: mc_math::DVec3::new(
                        f64::from(below.x) + dx,
                        f64::from(below.y) + 0.1 + props.bounds.max_y,
                        f64::from(below.z) + dz,
                    ),
                    velocity: mc_math::DVec3::ZERO,
                }));
            }
        }
    }

    if solid_impacts > 0 {
        let trigger = random.next_i32_bound(3) < counter;
        if trigger {
            // java: the post-increment is part of the trigger expression, then the successful
            // branch immediately resets `rainSoundCounter` before choosing the close/distant
            // pitch.  A steadily increasing counter would make every later rain pass audible.
            counter = 0;
            let precipitation_here = columns.precipitation_height(camera_block.x, camera_block.z);
            let (volume, pitch) =
                if sound_y > f64::from(camera_block.y + 1) && precipitation_here > camera_block.y {
                    (0.1, 0.5)
                } else {
                    (0.2, 1.0)
                };
            effects.push(AmbientEffect::Sound(AmbientSoundEffect {
                name: "ambient.weather.rain",
                position: mc_math::DVec3::new(sound_x, sound_y, sound_z),
                volume,
                pitch,
            }));
        } else {
            counter = counter.wrapping_add(1);
        }
    }
    RainAmbientPass {
        effects,
        rain_sound_counter: counter,
    }
}

/// What a column of world state [`build_columns`] needs, decoupled from `mc_world::World` so the
/// column-selection/seeding logic is testable without building a real world.
pub trait ColumnEnvironment {
    /// Biome temperature at this column, evaluated at `y`, for the rain-vs-snow split.
    /// java: `BiomeGenBase.getFloatTemperature(new BlockPos(l1, k2, k1))`
    /// (`EntityRenderer.renderRainSnow:1667-1669`) — note the Y is `k2`, the *bottom* of the
    /// streak's bottom, not necessarily the column's precipitation height, and
    /// `getTemperatureAtHeight` on the
    /// result is an identity function (`WorldChunkManager.java:120-123`).
    fn temperature_at(&self, x: i32, y: i32, z: i32) -> f32;

    /// Whether precipitation in this column falls as snow at `y`.
    ///
    /// Real worlds override this through [`mc_data::Biome::snows_at`], keeping the biome's
    /// precipitation flags and strict temperature boundary in one canonical predicate. The
    /// default preserves the compact temperature-only environments used by focused tests.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn snows_at(&self, x: i32, y: i32, z: i32) -> bool {
        !(self.temperature_at(x, y, z) >= 0.15)
    }
    /// The Y to spawn precipitation from — vanilla's `World.getPrecipitationHeight`, the top
    /// opaque block in the column (so indoor columns effectively never show rain below a roof).
    fn precipitation_height(&self, x: i32, z: i32) -> i32;
    /// Combined light (`0..15`) at a position, for the streak's brightness tint.
    fn light_at(&self, x: i32, y: i32, z: i32) -> u8;

    /// Whether this column's biome has any precipitation at all.
    /// java: `EntityRenderer.renderRainSnow:1639` gates every column on
    /// `biomegenbase.canRain() || biomegenbase.getEnableSnow()` **before** the temperature test
    /// picks rain over snow. Without it, rain falls in deserts, mesas and mushroom islands.
    ///
    /// The default is `true` purely so the hand-built `ColumnEnvironment`s in tests need not
    /// care; [`crate::sky::WorldColumns`] overrides it with the real biome flags.
    fn has_precipitation(&self, _x: i32, _z: i32) -> bool {
        true
    }
}

/// One column's precomputed precipitation streak, ready to turn into a quad.
#[derive(Debug, Clone, Copy)]
pub struct PrecipColumn {
    /// Column X.
    pub x: i32,
    /// Column Z.
    pub z: i32,
    /// Top of the streak. java: `l2 = max(floor(camY) + radius, precipitationHeight)`.
    pub top_y: i32,
    /// Bottom of the streak. java: `k2 = max(floor(camY) - radius, precipitationHeight)`.
    ///
    /// The pair is a band **around the camera**, not a fixed drop from the ground: vanilla's
    /// rain falls past the player's face and stops at ground level. Drawing a fixed length
    /// downward from `precipitationHeight` put every streak at or below foot level, where the
    /// shared depth buffer removed it.
    pub bottom_y: i32,
    /// `true` = snow, `false` = rain. java: `temperature < 0.15F`
    pub is_snow: bool,
    /// `0.0..1.0`, distance-faded and rain-strength-scaled.
    pub alpha: f32,
    /// `0..15` combined light, pre-fetched so the phase does not need `&World`.
    pub light: u8,
    /// java: `EntityRenderer.rendererUpdateCount`, narrowed to Java's signed 32-bit counter.
    /// Combined with [`WeatherPhase::partial_ticks`] for continuous rain/snow texture motion.
    pub animation_tick: i32,
    /// Half of the constructor's `rainXCoords`/`rainYCoords` entry for this column.
    /// `None` is the centre entry, where vanilla computes `0/0` and submits a NaN quad that
    /// rasterises nothing. This is captured when columns are built because the table index uses
    /// the tick position, while rendering subtracts the separately interpolated camera position.
    pub billboard_half_offset: Option<(f64, f64)>,
}

/// java: the do-while column loop in `renderRainSnow` — search radius 10 with fancy graphics,
/// 5 without.
#[must_use]
pub fn search_radius(fancy: bool) -> i32 {
    if fancy {
        10
    } else {
        5
    }
}

/// java: `this.random.setSeed((long)(l1*l1*3121 + l1*45238971 ^ k1*k1*418711 + k1*13761))` — Java
/// `int` arithmetic (wrapping), the result sign-extended to the `long` seed `Random` takes.
#[must_use]
pub fn column_seed(x: i32, z: i32) -> i64 {
    let a = x
        .wrapping_mul(x)
        .wrapping_mul(3121)
        .wrapping_add(x.wrapping_mul(45_238_971));
    let b = z
        .wrapping_mul(z)
        .wrapping_mul(418_711)
        .wrapping_add(z.wrapping_mul(13_761));
    i64::from(a ^ b)
}

/// Build this tick's precipitation columns around `camera_pos`. Pure and deterministic for a
/// fixed `env`/`camera_pos`/`tick`. The exact billboard offsets and per-frame texture animation
/// are applied by the phase after [`WeatherPhase::partial_ticks`] is known.
#[must_use]
pub fn build_columns(
    env: &dyn ColumnEnvironment,
    camera_pos: DVec3,
    tick: i64,
    fancy: bool,
    rain_strength: f32,
) -> Vec<PrecipColumn> {
    // java: `if (f > 0.0F)`. The negated comparison preserves Java's false branch for NaN;
    // using `<=` would let a NaN strength through and manufacture NaN vertex alpha/UV state.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(rain_strength > 0.0) {
        return Vec::new();
    }
    let radius = search_radius(fancy);
    let cam_x = camera_pos.x.floor() as i32;
    let cam_y = camera_pos.y.floor() as i32;
    let cam_z = camera_pos.z.floor() as i32;
    let mut out = Vec::new();

    // java: z (`k1`) is the outer loop and x (`l1`) the inner loop. Transparent precipitation
    // is submitted in this order, so retain it even though each column is computed independently.
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let x = cam_x + dx;
            let z = cam_z + dz;
            // java: `d6 = (l1 + 0.5F) - entity.posX`, `f3 = sqrt(d6*d6 + d7*d7) / i1` — measured
            // from the *fractional* camera position to the column centre, not from the integer
            // offset, and **not** clipped to a circle: `f3` runs up to `sqrt(2)` at the corners
            // of the square loop and those columns are still drawn, just faint. The alpha
            // formula takes them to zero on its own at `f3 = sqrt(2)`.
            let dist = column_distance(x, z, camera_pos, radius);

            // java: `if (biomegenbase.canRain() || biomegenbase.getEnableSnow())` — the gate
            // that comes *before* the temperature split.
            if !env.has_precipitation(x, z) {
                continue;
            }

            // java: `j2` = precipitation height, `k2 = max(j - i1, j2)`, `l2 = max(j + i1, j2)`,
            // and the whole column is skipped when `k2 == l2` — which is exactly the case where
            // the column's top is at or above the camera plus the search radius.
            let precip_height = env.precipitation_height(x, z);
            let k2 = (cam_y - radius).max(precip_height);
            let l2 = (cam_y + radius).max(precip_height);
            if k2 == l2 {
                continue;
            }

            // `k2` is the lower/bottom Y and `l2` the upper/top Y. The bytecode samples the
            // biome temperature at `k2` despite older comments calling that bound the top.
            // Real world columns route this through `Biome::snows_at`, which owns both the
            // `(rain || snow)` gate and the strict temperature boundary. Test environments use
            // the trait's temperature-only default after the gate above.
            let is_snow = env.snows_at(x, k2, z);
            let alpha = if is_snow {
                ((1.0 - dist * dist) * 0.3 + 0.5) * rain_strength
            } else {
                ((1.0 - dist * dist) * 0.5 + 0.5) * rain_strength
            };
            if alpha <= 0.0 {
                continue;
            }

            let (top_y, bottom_y) = (l2, k2);
            // java: `i3 = j2 < l ? l : j2` (`l = floor(interpolated camera y)`) — the light is
            // sampled at the camera's own height whenever the column top is below it, so rain
            // seen from above is lit by the sky, not by whatever is at ground level.
            let light_y = precip_height.max(cam_y);
            let light = if is_snow {
                // java: snow is blended 3:1 toward full-bright rather than using raw light.
                let raw = env.light_at(x, light_y, z);
                ((u16::from(raw) * 3 + 15) / 4) as u8
            } else {
                env.light_at(x, light_y, z)
            };

            out.push(PrecipColumn {
                x,
                z,
                top_y,
                bottom_y,
                is_snow,
                alpha: alpha.clamp(0.0, 1.0),
                light,
                animation_tick: tick as i32,
                billboard_half_offset: streak_half_offset(dx, dz),
            });
        }
    }
    out
}

/// Java's `f3` distance from a precipitation column to the view entity.
///
/// `EntityRenderer.renderRainSnow` first narrows the integer column coordinate through
/// `(float)l1 + 0.5F` (and likewise for z), then widens that value before subtracting the
/// entity's double position. Keeping the centre as a Rust `f64` is visibly different near the
/// world border, where an f32 has a two-block ULP.
fn column_distance(x: i32, z: i32, camera_pos: DVec3, radius: i32) -> f32 {
    let d6 = f64::from(x as f32 + 0.5) - camera_pos.x;
    let d7 = f64::from(z as f32 + 0.5) - camera_pos.z;
    (d6 * d6 + d7 * d7).sqrt() as f32 / radius as f32
}

/// An active lightning bolt. java: `EntityLightningBolt` — the entity carries a single `long`,
/// `boltVertex`, which is the *only* input to its geometry; the renderer seeds two `java.util.Random`s
/// from it and derives everything else.
///
/// The bolt is always 128 blocks tall (8 levels of 16) and is anchored at the entity's feet, so
/// there is no "ground Y" to supply — vanilla draws upward from `posY` into the sky, it does not
/// draw down from a cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightningStrike {
    /// World position of the struck block. java: the entity's `posX`/`posY`/`posZ`, which
    /// `doRender` receives as `x`/`y`/`z` (camera-relative) and immediately offsets by `+0.5` in
    /// x and z to centre on the block.
    pub position: DVec3,
    /// java: `EntityLightningBolt.boltVertex`, `rand.nextLong()` at spawn and again each time the
    /// bolt re-strikes.
    pub bolt_vertex: i64,
}

/// The client-visible lifetime of one `EntityLightningBolt`.
///
/// `RenderLightningBolt` has no `lightningState` visibility gate: the entity continues to draw
/// through its negative, dark interval and only disappears when `setDead()` runs.  Keeping this
/// small state machine beside the retained render strikes prevents the sky-flash countdown from
/// accidentally becoming the bolt's lifetime.
#[derive(Debug, Clone, PartialEq)]
struct LightningLifetime {
    /// java: `EntityLightningBolt.lightningState`, initially `2`.
    lightning_state: i32,
    /// java: `EntityLightningBolt.boltLivingTime`, initially `rand.nextInt(3) + 1`.
    bolt_living_time: i32,
    /// The entity's random stream after its constructor and first sound update. Subsequent draws
    /// preserve Java's `nextInt(10)` / `nextLong()` re-strike order.
    random: JavaRandom,
}

/// <<NW-RQ:lightning>> A copy of a [`WeatherPhase`]'s live lightning entities
/// ([`WeatherPhase::lightning_snapshot`]), for replay checkpoints.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LightningSnapshot {
    lightning: Vec<LightningStrike>,
    lifetimes: Vec<LightningLifetime>,
}

impl LightningSnapshot {
    /// How many bolts the copy holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lightning.len()
    }

    /// Whether the copy holds no bolt.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lightning.is_empty()
    }
}

/// The two observable outcomes of one `EntityLightningBolt.onUpdate` tick.
///
/// `setDead()` and the client-only `setLastLightningBolt(2)` happen at different points in the
/// Java state machine. Keeping both results explicit prevents the flash signal from being
/// inferred from the retained-entity result (or from the independent countdown).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LightningTick {
    /// Whether the weather entity remains in `World.weatherEffects` after this tick.
    alive: bool,
    /// Whether Java's client-side `setLastLightningBolt(2)` runs after this tick.
    flash: bool,
}

/// Advance the retained weather entities without requiring a GPU-backed phase.
///
/// Keeping this CPU seam separate makes the return value testable: a caller must re-arm the
/// world flash when *any* entity reports the client-side signal, while a dark retained entity
/// contributes no signal.
fn tick_lightning_entities(
    lightning: &mut Vec<LightningStrike>,
    lifetimes: &mut Vec<LightningLifetime>,
) -> bool {
    debug_assert_eq!(lightning.len(), lifetimes.len());
    let mut index = 0;
    let mut flash = false;
    while index < lightning.len() {
        let tick = lifetimes[index].tick(&mut lightning[index]);
        flash |= tick.flash;
        if tick.alive {
            index += 1;
        } else {
            lightning.remove(index);
            lifetimes.remove(index);
        }
    }
    flash
}

impl LightningLifetime {
    fn new(seed: i64) -> Self {
        let mut random = JavaRandom::new(seed);
        Self {
            lightning_state: 2,
            bolt_living_time: random.next_i32_bound(3) + 1,
            random,
        }
    }

    /// Advance `EntityLightningBolt.onUpdate`'s lifetime branch.
    ///
    /// Returns the entity's alive/dead result plus the client-only flash signal. Sound, fire, and
    /// damage belong to the event/world driver; this is deliberately only the render-entity
    /// state that selects its current geometry seed and eventual removal.
    fn tick(&mut self, strike: &mut LightningStrike) -> LightningTick {
        self.lightning_state -= 1;
        if self.lightning_state < 0 {
            if self.bolt_living_time == 0 {
                return LightningTick {
                    alive: false,
                    flash: false,
                };
            }
            if self.lightning_state < -self.random.next_i32_bound(10) {
                self.bolt_living_time -= 1;
                self.lightning_state = 1;
                strike.bolt_vertex = self.random.next_i64();
            }
        }
        LightningTick {
            alive: true,
            flash: self.lightning_state >= 0,
        }
    }
}

/// Blocks per bolt level. java: the `i1 * 16` / `(i1 + 1) * 16` vertex heights.
const BOLT_LEVEL_HEIGHT: f64 = 16.0;
/// Levels in a bolt. java: `new double[8]`.
const BOLT_LEVELS: usize = 8;

/// One `GL_TRIANGLE_STRIP` from `RenderLightningBolt.doRender` — a single tapering, four-sided
/// tube segment spanning one 16-block level, in **block-local** coordinates (the strike position
/// is the origin; add [`LightningStrike::position`] to place it).
///
/// The ten vertices are exactly the ten `worldrenderer.pos(...)` calls of one `tessellator.draw()`,
/// in order: the `j1` loop emits a lower/upper pair per corner and revisits corner 0 to close the
/// tube, so the strip is `(lower0, upper0, lower1, upper1, ... lower0', upper0')`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoltSegment {
    /// The strip's ten vertices, in emission order.
    pub verts: [DVec3; 10],
}

impl BoltSegment {
    /// The strip as eight triangles' worth of vertex indices. `GL_TRIANGLE_STRIP` semantics:
    /// triangle `i` is `(i, i+1, i+2)`. Winding alternates, which is why the vanilla pipeline
    /// draws lightning with culling irrelevant (it disables lighting and never faces-culls).
    pub const TRIANGLES: usize = 8;
}

/// Vanilla's lightning colour. java: `.color(0.45F, 0.45F, 0.5F, 0.3F)` on every one of the
/// bolt's vertices — a pale blue-white at 30% alpha, drawn additively
/// (`GlStateManager.blendFunc(770, 1)` = `GL_SRC_ALPHA, GL_ONE`), so overlapping tubes accumulate
/// into the bright core.
pub const BOLT_COLOR: [f32; 4] = [0.45, 0.45, 0.5, 0.3];

/// Whether the weather render phase has work in this dimension.
///
/// Precipitation belongs to a surface world, but lightning is an entry in vanilla's
/// `World.weatherEffects` list and `RenderGlobal.renderEntities` renders that list regardless
/// of the dimension.  Keeping the two gates separate matters for plugin-spawned lightning in
/// the Nether or End: the bolt must draw there without reviving stale overworld rain columns.
#[must_use]
fn weather_phase_enabled(dimension: Dimension, has_lightning: bool, has_border: bool) -> bool {
    dimension.has_weather() || has_lightning || has_border
}

/// Build a bolt's full geometry. java: `RenderLightningBolt.doRender` (lines 21-140), transcribed
/// term for term.
///
/// Structure, since it is not obvious from the loop nest: an outer `k1` loop draws the bolt four
/// times at increasing tube radius (`0.1 + k1 * 0.2`), which is what produces the glowing core —
/// each pass re-seeds its own `Random` from `boltVertex` so the four tubes follow the *same* path.
/// Inside it, `j = 0` is the main bolt down all eight levels with a `nextInt(11) - 5` per-level
/// walk, and `j = 1`/`j = 2` are two short branches (three levels each, starting one and two
/// levels below the top) with a much wider `nextInt(31) - 15` walk. `j = 0` additionally tapers
/// the tube by `i1 * 0.1 + 1.0`, so the bolt is fattest in the sky and thinnest at the ground.
///
/// Returns 56 segments (4 × (8 + 3 + 3)) for every input.
#[must_use]
pub fn lightning_bolt_segments(bolt_vertex: i64) -> Vec<BoltSegment> {
    // java: the `for (int i = 7; i >= 0; --i)` pre-pass that walks the bolt's spine and records
    // each level's (x, z) offset. Note `d0`/`d1` keep accumulating past the last store, so they
    // end up one step beyond `adouble[0]`; the render loop subtracts that final value, which is
    // what anchors the bolt's *bottom* near the entity rather than its top.
    let mut spine_x = [0.0f64; BOLT_LEVELS];
    let mut spine_z = [0.0f64; BOLT_LEVELS];
    let mut d0 = 0.0f64;
    let mut d1 = 0.0f64;
    let mut random = JavaRandom::new(bolt_vertex);
    for i in (0..BOLT_LEVELS).rev() {
        spine_x[i] = d0;
        spine_z[i] = d1;
        d0 += f64::from(random.next_i32_bound(11) - 5);
        d1 += f64::from(random.next_i32_bound(11) - 5);
    }

    let mut out = Vec::with_capacity(4 * (BOLT_LEVELS + 3 + 3));
    for k1 in 0..4 {
        // java: `Random random1 = new Random(entity.boltVertex)` — re-seeded per width pass, so
        // all four tubes trace the identical path.
        let mut random1 = JavaRandom::new(bolt_vertex);
        for j in 0..3 {
            let k = if j > 0 { 7 - j } else { 7 };
            let l = if j > 0 { k - 2 } else { 0 };

            let mut d2 = spine_x[k as usize] - d0;
            let mut d3 = spine_z[k as usize] - d1;

            for i1 in (l..=k).rev() {
                let d4 = d2;
                let d5 = d3;
                if j == 0 {
                    d2 += f64::from(random1.next_i32_bound(11) - 5);
                    d3 += f64::from(random1.next_i32_bound(11) - 5);
                } else {
                    d2 += f64::from(random1.next_i32_bound(31) - 15);
                    d3 += f64::from(random1.next_i32_bound(31) - 15);
                }

                // Half-widths of the segment's two ends. `d6` belongs to the *upper* vertex of
                // the pair (height `(i1 + 1) * 16`, offsets `d4`/`d5`) and `d7` to the lower one,
                // which reads backwards until you notice the strip is emitted bottom-then-top.
                let mut d6 = 0.1 + f64::from(k1) * 0.2;
                if j == 0 {
                    d6 *= f64::from(i1) * 0.1 + 1.0;
                }
                let mut d7 = 0.1 + f64::from(k1) * 0.2;
                if j == 0 {
                    d7 *= f64::from(i1 - 1) * 0.1 + 1.0;
                }

                let mut verts = [DVec3::ZERO; 10];
                for j1 in 0..5usize {
                    // java: the corner walk — (-,-), (+,-), (+,+), (-,+), then (-,-) again.
                    let mut d8 = 0.5 - d6;
                    let mut d9 = 0.5 - d6;
                    if j1 == 1 || j1 == 2 {
                        d8 += d6 * 2.0;
                    }
                    if j1 == 2 || j1 == 3 {
                        d9 += d6 * 2.0;
                    }
                    let mut d10 = 0.5 - d7;
                    let mut d11 = 0.5 - d7;
                    if j1 == 1 || j1 == 2 {
                        d10 += d7 * 2.0;
                    }
                    if j1 == 2 || j1 == 3 {
                        d11 += d7 * 2.0;
                    }

                    verts[j1 * 2] =
                        DVec3::new(d10 + d2, f64::from(i1) * BOLT_LEVEL_HEIGHT, d11 + d3);
                    verts[j1 * 2 + 1] =
                        DVec3::new(d8 + d4, f64::from(i1 + 1) * BOLT_LEVEL_HEIGHT, d9 + d5);
                }
                out.push(BoltSegment { verts });
            }
        }
    }
    out
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WeatherVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

impl WeatherVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<WeatherVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4],
    };
}

/// One frame of the vanilla world-border forcefield. The client driver owns the packet/state
/// clock and computes these camera-independent values; this phase only turns them into the four
/// clipped wall strips and submits the texture draw.
///
/// java: `RenderGlobal.renderWorldBorder` (`:1679-1795`). `player_x`/`player_z` are the raw
/// render-view entity position used by the near-border gate and alpha calculation, while
/// `render_x`/`render_z` are the separately interpolated coordinates used to clip each wall's
/// visible segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldBorderFrame {
    /// Current clamped west/east border coordinates.
    pub min_x: f64,
    /// Current clamped east border coordinate.
    pub max_x: f64,
    /// Current clamped north/south border coordinates.
    pub min_z: f64,
    /// Current clamped south border coordinate.
    pub max_z: f64,
    /// Raw render-view entity X/Z for the near-border alpha input.
    pub player_x: f64,
    /// Raw render-view entity Z for the near-border alpha input.
    pub player_z: f64,
    /// Interpolated render-view entity X/Z for the visible segment bounds.
    pub render_x: f64,
    /// Interpolated render-view entity Z for the visible segment bounds.
    pub render_z: f64,
    /// `renderDistanceChunks * 16`, used to limit the emitted wall strips.
    pub render_distance: f64,
    /// `Math.pow(1 - closestDistance / render_distance, 4)` narrowed to Java float colour alpha.
    pub alpha: f32,
    /// The status RGB extracted from `EnumBorderStatus.getID()`; alpha is supplied separately.
    pub color: [f32; 4],
    /// `(Minecraft.getSystemTime() % 3000) / 3000`, for forcefield U/V animation.
    pub texture_phase: f32,
}

/// Build the camera-relative forcefield vertices in Java's four-wall and one-block segment
/// order. Kept pure so transition gates, clipping, UV progression, and a no-wall mutant can be
/// tested without constructing a GPU.
fn world_border_vertices(frame: WorldBorderFrame, cam: DVec3) -> Vec<WeatherVertex> {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(frame.render_distance > 0.0)
        || !frame.alpha.is_finite()
        || frame.alpha <= 0.0
        || !frame.render_distance.is_finite()
    {
        return Vec::new();
    }

    let mut out = Vec::new();
    let d0 = frame.render_distance;
    let z_start = frame.render_z - d0;
    let z_end = frame.render_z + d0;
    let x_start = frame.render_x - d0;
    let x_end = frame.render_x + d0;
    let z_min = z_start.floor().max(frame.min_z);
    let z_max = z_end.ceil().min(frame.max_z);
    let x_min = x_start.floor().max(frame.min_x);
    let x_max = x_end.ceil().min(frame.max_x);

    // Java starts each wall's U phase at zero and advances by 0.5 per block. Do not merge
    // adjacent quads: a fractional end segment needs its own half-width and UV span.
    if frame.render_x > frame.max_x - d0 {
        emit_world_border_wall(&mut out, frame.max_x, z_min, z_max, true, frame, cam);
    }
    if frame.render_x < frame.min_x + d0 {
        emit_world_border_wall(&mut out, frame.min_x, z_min, z_max, true, frame, cam);
    }
    if frame.render_z > frame.max_z - d0 {
        emit_world_border_wall(&mut out, frame.max_z, x_min, x_max, false, frame, cam);
    }
    if frame.render_z < frame.min_z + d0 {
        emit_world_border_wall(&mut out, frame.min_z, x_min, x_max, false, frame, cam);
    }
    out
}

/// Append one wall's one-block quads. `constant_is_x` selects X walls; otherwise the constant
/// coordinate is Z. The order is the four `POSITION_TEX` calls from `renderWorldBorder`.
fn emit_world_border_wall(
    out: &mut Vec<WeatherVertex>,
    constant: f64,
    start: f64,
    end: f64,
    constant_is_x: bool,
    frame: WorldBorderFrame,
    cam: DVec3,
) {
    let mut cursor = start;
    let mut u = 0.0_f32;
    while cursor < end {
        let width = (1.0_f64).min(end - cursor);
        let u_next = u + width as f32 * 0.5;
        let top_a = if constant_is_x {
            DVec3::new(constant, 256.0, cursor)
        } else {
            DVec3::new(cursor, 256.0, constant)
        };
        let top_b = if constant_is_x {
            DVec3::new(constant, 256.0, cursor + width)
        } else {
            DVec3::new(cursor + width, 256.0, constant)
        };
        let bottom_b = if constant_is_x {
            DVec3::new(constant, 0.0, cursor + width)
        } else {
            DVec3::new(cursor + width, 0.0, constant)
        };
        let bottom_a = if constant_is_x {
            DVec3::new(constant, 0.0, cursor)
        } else {
            DVec3::new(cursor, 0.0, constant)
        };
        let color = [frame.color[0], frame.color[1], frame.color[2], frame.alpha];
        let positions = [top_a, top_b, bottom_b, bottom_a].map(|pos| (pos - cam).as_vec3());
        let uvs = [
            [frame.texture_phase + u, frame.texture_phase],
            [frame.texture_phase + u_next, frame.texture_phase],
            [frame.texture_phase + u_next, frame.texture_phase + 128.0],
            [frame.texture_phase + u, frame.texture_phase + 128.0],
        ];
        // GPURES-4: built on the stack; this ran a heap `collect` per one-block quad.
        let quad: [WeatherVertex; 4] = std::array::from_fn(|corner| WeatherVertex {
            pos: positions[corner].to_array(),
            uv: uvs[corner],
            color,
        });
        for &index in &[0usize, 1, 2, 0, 2, 3] {
            out.push(quad[index]);
        }
        cursor += width;
        u = u_next;
    }
}

/// `POSITION_COLOR`, matching `lightning.wgsl`. java: `DefaultVertexFormats.POSITION_COLOR`, the
/// format `RenderLightningBolt` begins its strips with.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BoltVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

impl BoltVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<BoltVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
    };
}

/// Turn one bolt's segments into camera-relative triangles. `GL_TRIANGLE_STRIP` expansion over
/// the segment's ten vertices, eight triangles per strip, with GL's winding rule: triangle `i`
/// is `(i, i+1, i+2)` for even `i` and `(i+1, i, i+2)` for odd `i`, so every triangle of a strip
/// faces the same way and the pipeline's back-face cull matches vanilla's.
fn bolt_triangles(strike: LightningStrike, cam: DVec3, out: &mut Vec<BoltVertex>) {
    for segment in lightning_bolt_segments(strike.bolt_vertex) {
        let local = segment.verts.map(|v| (strike.position + v - cam).as_vec3());
        for i in 0..BoltSegment::TRIANGLES {
            let (a, b) = if i.is_multiple_of(2) {
                (i, i + 1)
            } else {
                (i + 1, i)
            };
            for v in [local[a], local[b], local[i + 2]] {
                out.push(BoltVertex {
                    pos: v.to_array(),
                    color: BOLT_COLOR,
                });
            }
        }
    }
}

/// Half-width/direction of vanilla's one precipitation plane for a column offset.
///
/// `EntityRenderer` builds two 32x32 tables once in its constructor:
/// `rainXCoords = -dz / sqrt(dx^2 + dz^2)` and
/// `rainYCoords =  dx / sqrt(dx^2 + dz^2)`. `renderRainSnow` multiplies both by `0.5D`.
/// The centre entry divides zero by zero and therefore emits a NaN quad in Java; returning
/// `None` skips the same non-rasterising primitive without feeding NaNs to wgpu.
fn streak_half_offset(dx: i32, dz: i32) -> Option<(f64, f64)> {
    let x = dx as f32;
    let z = dz as f32;
    let len = (x * x + z * z).sqrt();
    if len == 0.0 {
        return None;
    }
    Some((f64::from(-z / len) * 0.5, f64::from(x / len) * 0.5))
}

/// Java's rain texture phase before the quarter-block Y term is added.
fn rain_scroll(col: &PrecipColumn, partial_ticks: f32, random: &mut JavaRandom) -> f64 {
    // java: `(rendererUpdateCount + x*x*3121 + x*45238971 + z*z*418711 + z*13761 & 31)`.
    // Every operation before `& 31` is signed 32-bit integer arithmetic.
    let hash = col
        .animation_tick
        .wrapping_add(col.x.wrapping_mul(col.x).wrapping_mul(3_121))
        .wrapping_add(col.x.wrapping_mul(45_238_971))
        .wrapping_add(col.z.wrapping_mul(col.z).wrapping_mul(418_711))
        .wrapping_add(col.z.wrapping_mul(13_761))
        & 31;
    (f64::from(hash) + f64::from(partial_ticks)) / 32.0 * (3.0 + random.next_f64())
}

/// Java's snow `(u_offset, v_offset)` before the quarter-block Y term is added.
fn snow_scroll(col: &PrecipColumn, partial_ticks: f32, random: &mut JavaRandom) -> (f64, f64) {
    // Preserve the source/bytecode's float-vs-double boundaries. In particular, the second
    // Gaussian is multiplied by `f1` as an f32 before that product widens to f64.
    let f1 = col.animation_tick as f32 + partial_ticks;
    let d8 = ((col.animation_tick & 511) as f32 + partial_ticks) / 512.0;
    let u_random = random.next_f64();
    let u_gaussian = random.next_gaussian() as f32;
    let u = u_random + f64::from(f1) * 0.01 * f64::from(u_gaussian);
    let v_random = random.next_f64();
    let v_gaussian = random.next_gaussian() as f32;
    let v = f64::from(d8) + v_random + f64::from(f1 * v_gaussian) * 0.001;
    (u, v)
}

/// The separate colour/alpha blend factors used by `EntityRenderer.renderRainSnow`.
///
/// java: `GlStateManager.tryBlendFuncSeparate(770, 771, 1, 0)` — colour is ordinary
/// source-alpha compositing, while the destination alpha is replaced by the source alpha.
/// WGPU's [`wgpu::BlendState::ALPHA_BLENDING`] uses `OneMinusSrcAlpha` for the destination
/// alpha, which is a different framebuffer result whenever two translucent weather layers
/// overlap. Keep this in one helper so the pipeline declaration and its regression stay tied
/// to the Java call rather than to a similarly named convenience constant.
fn weather_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// The forcefield's fixed-function blend from `RenderGlobal.renderWorldBorder`.
///
/// Java's `(770, 1, 1, 0)` is source-alpha/additive for colour and source/zero for alpha. The
/// latter keeps the border from polluting the framebuffer alpha used by later passes.
fn world_border_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrecipitationKind {
    Rain,
    Snow,
}

impl PrecipitationKind {
    fn of(column: &PrecipColumn) -> Self {
        if column.is_snow {
            Self::Snow
        } else {
            Self::Rain
        }
    }
}

/// One contiguous Java precipitation draw batch. Vanilla's `j1` state machine calls
/// `tessellator.draw()` exactly when the column's rain/snow kind changes; keeping those runs
/// instead of collecting the two kinds globally preserves transparent compositing order.
#[derive(Debug, Clone)]
struct PrecipitationBatch {
    kind: PrecipitationKind,
    vertices: Vec<WeatherVertex>,
}

#[cfg(test)]
fn streak_quads(
    columns: &[PrecipColumn],
    cam: DVec3,
    partial_ticks: f32,
    out_rain: &mut Vec<WeatherVertex>,
    out_snow: &mut Vec<WeatherVertex>,
) {
    for col in columns {
        let dst = if col.is_snow {
            &mut *out_snow
        } else {
            &mut *out_rain
        };
        emit_streak_quad(col, cam, partial_ticks, dst);
    }
}

/// Emit one six-vertex precipitation quad into an already-selected type batch.
fn emit_streak_quad(
    col: &PrecipColumn,
    cam: DVec3,
    partial_ticks: f32,
    dst: &mut Vec<WeatherVertex>,
) {
    let Some((half_x, half_z)) = col.billboard_half_offset else {
        return;
    };

    // java: all four positions are `(double)l1 +/- d3 + 0.5D` and the analogous z term.
    // This is one radial billboard plane, not two crossed axis-aligned planes.
    let ex = col.x as f64 + 0.5 - cam.x;
    let ez = col.z as f64 + 0.5 - cam.z;
    let top = col.top_y as f64 - cam.y;
    let bottom = col.bottom_y as f64 - cam.y;
    let brightness = f32::from(col.light) / 15.0;
    let rgba = [brightness, brightness, brightness, col.alpha];

    let mut random = JavaRandom::new(column_seed(col.x, col.z));
    let (u_offset, v_offset) = if col.is_snow {
        snow_scroll(col, partial_ticks, &mut random)
    } else {
        (0.0, rain_scroll(col, partial_ticks, &mut random))
    };

    let p = [
        Vec3::new((ex - half_x) as f32, bottom as f32, (ez - half_z) as f32),
        Vec3::new((ex + half_x) as f32, bottom as f32, (ez + half_z) as f32),
        Vec3::new((ex + half_x) as f32, top as f32, (ez + half_z) as f32),
        Vec3::new((ex - half_x) as f32, top as f32, (ez - half_z) as f32),
    ];
    let bottom_v = col.bottom_y as f64 * 0.25 + v_offset;
    let top_v = col.top_y as f64 * 0.25 + v_offset;
    let uv = [
        Vec2::new(u_offset as f32, bottom_v as f32),
        Vec2::new((1.0 + u_offset) as f32, bottom_v as f32),
        Vec2::new((1.0 + u_offset) as f32, top_v as f32),
        Vec2::new(u_offset as f32, top_v as f32),
    ];
    let v = |i: usize| WeatherVertex {
        pos: p[i].to_array(),
        uv: uv[i].to_array(),
        color: rgba,
    };
    dst.push(v(0));
    dst.push(v(1));
    dst.push(v(2));
    dst.push(v(0));
    dst.push(v(2));
    dst.push(v(3));
}

/// Reproduce vanilla's contiguous `j1` precipitation batches in z-outer/x-inner order.
fn ordered_streak_batches(
    columns: &[PrecipColumn],
    cam: DVec3,
    partial_ticks: f32,
) -> Vec<PrecipitationBatch> {
    let mut batches = Vec::new();
    for col in columns {
        if col.billboard_half_offset.is_none() {
            continue;
        }
        let kind = PrecipitationKind::of(col);
        if batches
            .last()
            .is_none_or(|batch: &PrecipitationBatch| batch.kind != kind)
        {
            batches.push(PrecipitationBatch {
                kind,
                vertices: Vec::new(),
            });
        }
        emit_streak_quad(
            col,
            cam,
            partial_ticks,
            &mut batches
                .last_mut()
                .expect("the kind batch was just created or retained")
                .vertices,
        );
    }
    batches
}

fn bind_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    tex: &SkyTexture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("weather group1"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Rain/snow streaks, registered at [`crate::PhaseOrder::WEATHER`] — after opaque terrain,
/// entities and particles, so precipitation is correctly hidden behind roofs and walls by the
/// shared depth buffer, and before translucent terrain, so water blends over it as in vanilla
/// (`EntityRenderer.renderWorldPass:1451-1467`).
pub struct WeatherPhase {
    /// This frame's environment (only [`Environment::dimension`]/`weather` are read).
    pub environment: Environment,
    /// This frame's interpolation fraction, used by vanilla's rain and snow UV formulas.
    pub partial_ticks: f32,
    columns: Vec<PrecipColumn>,
    lightning: Vec<LightningStrike>,
    /// One exact `EntityLightningBolt` lifetime per retained render strike.
    lightning_lifetimes: Vec<LightningLifetime>,
    /// Border wall state for this frame. The driver clears it when no live border is near enough
    /// to render, so an old world's forcefield cannot survive a disconnect.
    border: Option<WorldBorderFrame>,

    pipeline: wgpu::RenderPipeline,
    bolt_pipeline: wgpu::RenderPipeline,
    border_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Java's global `GameSettings.anaglyph` upload switch for resident weather resources.
    anaglyph: bool,
    rain_tex: Option<(SkyTexture, wgpu::BindGroup)>,
    snow_tex: Option<(SkyTexture, wgpu::BindGroup)>,
    forcefield_tex: Option<(SkyTexture, wgpu::BindGroup)>,
    /// GPURES-4 / R1-17: one retained, grow-only stream for every weather draw of a frame
    /// (precipitation batches, bolts, border wall) instead of a new buffer per draw per frame.
    vertices: WeatherVertexStream,
}

/// The weather phase's retained vertex stream: a reused CPU staging `Vec` and a
/// `VERTEX | COPY_DST` buffer written with one `write_buffer` per frame. Both grow to fit and
/// give oversized memory back (see [`Self::release_if_oversized`]).
#[derive(Default)]
struct WeatherVertexStream {
    bytes: Vec<u8>,
    buffer: Option<wgpu::Buffer>,
}

impl WeatherVertexStream {
    /// Above this many bytes the stream gives memory back once a frame needs less than a
    /// quarter of it (phase 4d review S2). Lightning is uncapped (about 37.6 KB per bolt), so a
    /// server-sent burst must not pin its peak for the rest of the session; ordinary rain, bolts
    /// and the border stay far below this, so steady play never reallocates.
    const SHRINK_ABOVE: u64 = 1 << 20;

    /// Drop memory this frame's bytes no longer need: an oversized staging capacity, and an
    /// oversized GPU buffer (the next [`Self::upload`] reallocates it at this frame's size).
    fn release_if_oversized(&mut self) {
        let needed = self.bytes.len() as u64;
        if self.bytes.capacity() as u64 > Self::SHRINK_ABOVE
            && needed * 4 < self.bytes.capacity() as u64
        {
            self.bytes.shrink_to(self.bytes.len());
        }
        if self
            .buffer
            .as_ref()
            .is_some_and(|buffer| buffer.size() > Self::SHRINK_ABOVE && needed * 4 < buffer.size())
        {
            self.buffer = None;
        }
    }

    /// Append `vertices`, returning their byte range in this frame's stream.
    fn push<T: bytemuck::Pod>(&mut self, vertices: &[T]) -> std::ops::Range<u64> {
        let start = self.bytes.len() as u64;
        self.bytes.extend_from_slice(bytemuck::cast_slice(vertices));
        start..self.bytes.len() as u64
    }

    /// Upload this frame's bytes, growing the buffer to the next power of two when needed.
    ///
    /// A staged write lands before any of this frame's commands run (as `write_buffer` does),
    /// so every draw this frame records sees exactly these bytes, as it did when each draw owned
    /// a fresh buffer. The buffer is read only by this phase's pass in the frame encoder, which
    /// is the upload ring's contract (`crate::upload`).
    fn upload(&mut self, gpu: &Gpu) -> Option<&wgpu::Buffer> {
        self.release_if_oversized();
        if self.bytes.is_empty() {
            return None;
        }
        let needed = self.bytes.len() as u64;
        if self
            .buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < needed)
        {
            self.buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("weather vertices"),
                size: needed.next_power_of_two().max(64 * 1024),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buffer = self.buffer.as_ref()?;
        gpu.write_buffer_staged(buffer, 0, &self.bytes);
        Some(buffer)
    }
}

impl WeatherPhase {
    /// Build the phase. Rain/snow textures start unset — see [`WeatherPhase::load_textures`].
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("weather"),
                source: wgpu::ShaderSource::Wgsl(include_str!("weather.wgsl").into()),
            });

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("weather group1"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("weather layout"),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("weather"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(WeatherVertex::LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        // java: tryBlendFuncSeparate(770, 771, 1, 0), including the
                        // destination-alpha replacement rather than WGPU's convenience state.
                        blend: Some(weather_blend_state()),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        // java: `RenderGlobal.renderWorldBorder` uses the same POSITION_TEX stream as weather,
        // but switches to additive colour blending and a -3 polygon offset before submitting
        // the forcefield quads. Reuse the weather shader/layout (including its 0.1 alpha test)
        // while keeping those fixed-function differences in a dedicated pipeline.
        let border_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("world border"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(WeatherVertex::LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    // java: `doPolygonOffset(-3.0F, -3.0F)`.
                    bias: wgpu::DepthBiasState {
                        constant: -3,
                        slope_scale: -3.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        // java: `tryBlendFuncSeparate(770, 1, 1, 0)`.
                        blend: Some(world_border_blend_state()),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("weather sampler"),
            // java: rain.png, snow.png and forcefield.png are plain `SimpleTexture`s with no
            // `.mcmeta`, so `TextureUtil.setTextureBlurMipmap(false, false)` selects
            // `GL_NEAREST` for both filters.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            // `SimpleTexture` uploads rain.png/snow.png with clamp=false. Both coordinates must
            // repeat: V spans one quarter-texture per world block and snow also jitters U.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let bolt_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lightning"),
                source: wgpu::ShaderSource::Wgsl(include_str!("lightning.wgsl").into()),
            });
        let bolt_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lightning layout"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let bolt_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("lightning"),
                layout: Some(&bolt_layout),
                vertex: wgpu::VertexState {
                    module: &bolt_shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(BoltVertex::LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    // java: `EntityRenderer.renderWorldPass` opens with `enableCull()` and
                    // nothing before `RenderGlobal.renderEntities`' weather-effect loop turns it
                    // off (the cloud renderers and the world border restore it), while
                    // `RenderLightningBolt` only touches texture/lighting/blend state. So the
                    // bolt's back faces are culled: each tube adds only its near (outward,
                    // counter-clockwise) walls. `bolt_triangles` keeps GL's strip winding.
                    cull_mode: Some(wgpu::Face::Back),
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::DEPTH_FORMAT,
                    // java: `RenderLightningBolt` never calls `depthMask(false)`, so every strip
                    // writes depth. Drawn narrow tube first, each wider tube's near wall is
                    // nearer and passes `GL_LEQUAL`; where branches cross, whichever was drawn
                    // first in front hides the later one. Without the write (and the cull above)
                    // every wall of every tube added up: twice vanilla's layers, a white core.
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bolt_shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        // java: GlStateManager.blendFunc(770, 1) == (GL_SRC_ALPHA, GL_ONE).
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        Self {
            environment: Environment::default(),
            partial_ticks: 0.0,
            columns: Vec::new(),
            lightning: Vec::new(),
            lightning_lifetimes: Vec::new(),
            border: None,
            pipeline,
            bolt_pipeline,
            border_pipeline,
            bind_group_layout,
            sampler,
            anaglyph: false,
            rain_tex: None,
            snow_tex: None,
            forcefield_tex: None,
            vertices: WeatherVertexStream::default(),
        }
    }

    /// Build the phase with Java's global upload-time anaglyph switch selected.
    ///
    /// The flag is retained before any resource-pack textures are loaded, so rain, snow, and
    /// the world-border forcefield are converted on their first upload. Use
    /// [`Self::set_anaglyph_with_gpu`] when changing an installed phase live.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let mut phase = Self::new(gpu);
        phase.anaglyph = anaglyph;
        phase
    }

    /// Load `rain.png`/`snow.png`. Left unset (nothing drawn) if missing.
    pub fn load_textures(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        if let Ok(tex) = textures::load_with_anaglyph(
            gpu,
            rm,
            "textures/environment/rain.png",
            "rain",
            self.anaglyph,
        ) {
            let group = bind_group(gpu, &self.bind_group_layout, &tex, &self.sampler);
            self.rain_tex = Some((tex, group));
        }
        if let Ok(tex) = textures::load_with_anaglyph(
            gpu,
            rm,
            "textures/environment/snow.png",
            "snow",
            self.anaglyph,
        ) {
            let group = bind_group(gpu, &self.bind_group_layout, &tex, &self.sampler);
            self.snow_tex = Some((tex, group));
        }
        if let Ok(tex) = textures::load_with_anaglyph(
            gpu,
            rm,
            "textures/misc/forcefield.png",
            "forcefield",
            self.anaglyph,
        ) {
            let group = bind_group(gpu, &self.bind_group_layout, &tex, &self.sampler);
            self.forcefield_tex = Some((tex, group));
        }
    }

    /// Whether Java's upload-time anaglyph conversion is selected for weather textures.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global transform and replay resident rain, snow, and forcefield textures
    /// from their retained decoded source bytes.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        for (texture, _) in [&self.rain_tex, &self.snow_tex, &self.forcefield_tex]
            .into_iter()
            .flatten()
        {
            texture.set_anaglyph(gpu, enabled);
        }
    }

    /// Set the transform for subsequent resource uploads without touching resident GPU
    /// textures. Use [`Self::set_anaglyph_with_gpu`] for a live setting change.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Replace the frame's border wall state. `None` is the near-border/render-distance gate's
    /// normal result and explicitly removes the previous frame's forcefield.
    pub fn set_world_border(&mut self, border: Option<WorldBorderFrame>) {
        self.border = border;
    }

    /// Current border wall state, primarily for deterministic driver/render tests.
    #[must_use]
    pub fn world_border(&self) -> Option<WorldBorderFrame> {
        self.border
    }

    /// Replace this tick's precomputed columns. Called once per tick by the driver (S21),
    /// exactly like [`crate::terrain::phase::TerrainRenderer::upload`].
    pub fn upload(&mut self, columns: Vec<PrecipColumn>) {
        self.columns = columns;
    }

    /// Register a newly spawned `EntityLightningBolt` render entity.
    ///
    /// Call [`WeatherPhase::tick_lightning`] once per client tick.  The supplied vertex seed is
    /// drawn immediately, then the phase owns the Java 2→1→0/negative/re-strike/delete lifetime.
    pub fn add_lightning_strike(&mut self, strike: LightningStrike) {
        self.lightning_lifetimes
            .push(LightningLifetime::new(strike.bolt_vertex));
        self.lightning.push(strike);
    }

    /// Register a strike whose local `EntityLightningBolt.rand` sequence is available.
    ///
    /// `SpawnGlobalEntity` does not transmit `boltVertex`: vanilla's client constructs its own
    /// weather entity and therefore its own unseeded local `Random`.  The client driver uses a
    /// deterministic stand-in for that otherwise wall-clock-seeded source, consumes the
    /// constructor's vertex/lifetime/sound draws, and hands the remaining stream here. Keeping
    /// the stream rather than re-seeding from `bolt_vertex` preserves the entity's subsequent
    /// `nextInt(10)` / `nextLong()` re-strike order.
    pub fn add_lightning_strike_with_entity_random(
        &mut self,
        strike: LightningStrike,
        bolt_living_time: i32,
        random: JavaRandom,
    ) {
        self.lightning_lifetimes.push(LightningLifetime {
            lightning_state: 2,
            bolt_living_time,
            random,
        });
        self.lightning.push(strike);
    }

    /// Advance all retained lightning entities by one 20 Hz client tick.
    ///
    /// Java draws weather-effect entities throughout their negative `lightningState` interval;
    /// this method therefore retains a strike until its own `boltLivingTime` reaches zero and the
    /// following negative tick executes `setDead()`.  It is intentionally independent of the
    /// world's two-tick lightning-flash counter.
    ///
    /// Returns `true` when at least one entity finished this tick with
    /// `lightningState >= 0`, the exact condition for Java's client-side
    /// `World.setLastLightningBolt(2)` call. A negative dark interval returns `false`, even
    /// though its entity remains retained and drawable.
    #[must_use]
    pub fn tick_lightning(&mut self) -> bool {
        let flashed = tick_lightning_entities(&mut self.lightning, &mut self.lightning_lifetimes);
        if self.lightning.is_empty() {
            self.release_stream();
        }
        flashed
    }

    /// Drop all active lightning strikes.
    pub fn clear_lightning(&mut self) {
        self.lightning.clear();
        self.lightning_lifetimes.clear();
        self.release_stream();
    }

    /// Give a lightning burst's stream memory back even when `record` no longer runs: in a
    /// dimension without weather the phase disables itself once the last bolt is gone, and a
    /// disconnect stops it too (phase 5a review S4). `record` repacks the stream from scratch
    /// every frame, so emptying it between frames loses nothing; a stream of ordinary size is
    /// left alone.
    fn release_stream(&mut self) {
        self.vertices.bytes.clear();
        self.vertices.release_if_oversized();
    }

    /// Currently registered strikes, for a caller (or a test) to inspect.
    #[must_use]
    pub fn lightning_strikes(&self) -> &[LightningStrike] {
        &self.lightning
    }

    /// <<NW-RQ:lightning>> Copy the live lightning entities (strikes, lifetimes and their random
    /// streams) for a replay checkpoint: they are ticked here,
    /// outside the replayed session, so a restore puts them back with
    /// [`Self::restore_lightning`] instead of losing a bolt that was still alive.
    #[must_use]
    pub fn lightning_snapshot(&self) -> LightningSnapshot {
        LightningSnapshot {
            lightning: self.lightning.clone(),
            lifetimes: self.lightning_lifetimes.clone(),
        }
    }

    /// <<NW-RQ:lightning>> Put a [`Self::lightning_snapshot`] back.
    pub fn restore_lightning(&mut self, snapshot: LightningSnapshot) {
        self.lightning = snapshot.lightning;
        self.lightning_lifetimes = snapshot.lifetimes;
        if self.lightning.is_empty() {
            self.release_stream();
        }
    }
}

impl RenderPhase for WeatherPhase {
    fn name(&self) -> &'static str {
        "sky/weather"
    }

    fn enabled(&self) -> bool {
        weather_phase_enabled(
            self.environment.dimension,
            !self.lightning.is_empty(),
            self.border.is_some(),
        )
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        // `renderRainSnow` is a surface-world effect, unlike weather-effect entities below.
        // A retained precipitation upload from an Overworld frame must not leak into a Nether
        // or End frame merely because a lightning bolt keeps this phase active there.
        let precipitation_batches = if self.environment.dimension.has_weather() {
            ordered_streak_batches(&self.columns, cam.pos, self.partial_ticks)
        } else {
            Vec::new()
        };

        // java: RenderLightningBolt.doRender — untextured, additively blended `POSITION_COLOR`
        // triangle strips, generated from the entity's `boltVertex` alone.
        let mut bolt_verts = Vec::new();
        for strike in &self.lightning {
            bolt_triangles(*strike, cam.pos, &mut bolt_verts);
        }

        let border_verts = self
            .border
            .map_or_else(Vec::new, |border| world_border_vertices(border, cam.pos));

        if precipitation_batches.is_empty() && bolt_verts.is_empty() && border_verts.is_empty() {
            // A frame with no weather still gives back what an earlier burst left behind.
            self.vertices.bytes.clear();
            self.vertices.release_if_oversized();
            return;
        }

        // GPURES-4: pack every draw of this frame into the retained stream, in draw order, and
        // upload it once before the pass. Draws that are skipped (a texture not loaded) are not
        // packed, so the stream holds exactly the bytes the per-draw buffers used to hold.
        self.vertices.bytes.clear();
        let mut precipitation_ranges = Vec::with_capacity(precipitation_batches.len());
        for batch in &precipitation_batches {
            let texture = match batch.kind {
                PrecipitationKind::Rain => &self.rain_tex,
                PrecipitationKind::Snow => &self.snow_tex,
            };
            if let Some((_, group)) = texture {
                let range = self.vertices.push(batch.vertices.as_slice());
                precipitation_ranges.push((group, range, batch.vertices.len() as u32));
            }
        }
        let bolt_range =
            (!bolt_verts.is_empty()).then(|| self.vertices.push(bolt_verts.as_slice()));
        let border_draw = match &self.forcefield_tex {
            Some((_, group)) if !border_verts.is_empty() => {
                Some((group, self.vertices.push(border_verts.as_slice())))
            }
            _ => None,
        };
        let Some(buffer) = self.vertices.upload(gpu) else {
            // Nothing drawable (only texture-less precipitation or border); the old path opened
            // an empty pass here, which loads and stores the attachments unchanged.
            drop(gpu.begin_pass(enc, view, "sky/weather", DepthAttachment::Shared));
            return;
        };

        let mut pass = gpu.begin_pass(enc, view, "sky/weather", DepthAttachment::Shared);
        if let Some(range) = bolt_range {
            // java: the bolt is a weather *entity*, drawn by `RenderGlobal.renderEntities` long
            // before `renderRainSnow`, and it writes depth; rain behind it then fails the depth
            // test. Needs no texture at all, so it runs whether or not `load_textures` was ever
            // called.
            pass.set_pipeline(&self.bolt_pipeline);
            pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
            pass.set_vertex_buffer(0, buffer.slice(range));
            pass.draw(0..bolt_verts.len() as u32, 0..1);
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);

        // java: the tessellator is flushed whenever `j1` changes from rain to snow or back.
        // Do not partition all rain and all snow into two global draws: that changes the
        // source-over result where translucent planes overlap.
        for (group, range, count) in precipitation_ranges {
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, buffer.slice(range));
            pass.draw(0..count, 0..1);
        }
        if let Some((group, range)) = border_draw {
            // A missing resource-pack texture is a no-draw fallback, but retaining the CPU
            // frame still means the next successful reload can draw the exact current wall.
            pass.set_pipeline(&self.border_pipeline);
            pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, buffer.slice(range));
            pass.draw(0..border_verts.len() as u32, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{BlockId, BlockState};

    /// GPURES-4: the stack-built border quads are bit-identical to the heap-collected quads the
    /// wall used to build, for all four walls, fractional end segments and a moving camera.
    #[test]
    fn world_border_stack_quads_match_the_collected_reference() {
        // Pristine `emit_world_border_wall`'s construction, heap `collect` included.
        fn reference_wall(
            out: &mut Vec<WeatherVertex>,
            constant: f64,
            start: f64,
            end: f64,
            constant_is_x: bool,
            frame: WorldBorderFrame,
            cam: DVec3,
        ) {
            let mut cursor = start;
            let mut u = 0.0_f32;
            while cursor < end {
                let width = (1.0_f64).min(end - cursor);
                let u_next = u + width as f32 * 0.5;
                let corner = |along: f64, y: f64| {
                    if constant_is_x {
                        DVec3::new(constant, y, along)
                    } else {
                        DVec3::new(along, y, constant)
                    }
                };
                let color = [frame.color[0], frame.color[1], frame.color[2], frame.alpha];
                let positions = [
                    corner(cursor, 256.0),
                    corner(cursor + width, 256.0),
                    corner(cursor + width, 0.0),
                    corner(cursor, 0.0),
                ]
                .map(|pos| (pos - cam).as_vec3());
                let uvs = [
                    [frame.texture_phase + u, frame.texture_phase],
                    [frame.texture_phase + u_next, frame.texture_phase],
                    [frame.texture_phase + u_next, frame.texture_phase + 128.0],
                    [frame.texture_phase + u, frame.texture_phase + 128.0],
                ];
                let quad: [WeatherVertex; 4] = positions
                    .into_iter()
                    .zip(uvs)
                    .map(|(pos, uv)| WeatherVertex {
                        pos: pos.to_array(),
                        uv,
                        color,
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("world-border quad has four vertices");
                for &index in &[0usize, 1, 2, 0, 2, 3] {
                    out.push(quad[index]);
                }
                cursor += width;
                u = u_next;
            }
        }
        for (render, cam) in [
            ((7.5, 7.25), DVec3::ZERO),
            ((0.6, 15.9), DVec3::new(3.25, 70.5, -2.125)),
            ((12.0, 0.4), DVec3::new(-100.0, 0.0, 33.75)),
        ] {
            let frame = WorldBorderFrame {
                min_x: 0.0,
                max_x: 15.5,
                min_z: 0.25,
                max_z: 16.0,
                player_x: render.0,
                player_z: render.1,
                render_x: render.0,
                render_z: render.1,
                render_distance: 9.75,
                alpha: 0.75,
                color: [0.125, 0.625, 1.0, 1.0],
                texture_phase: 0.375,
            };
            let d0 = frame.render_distance;
            let z_min = (frame.render_z - d0).floor().max(frame.min_z);
            let z_max = (frame.render_z + d0).ceil().min(frame.max_z);
            let x_min = (frame.render_x - d0).floor().max(frame.min_x);
            let x_max = (frame.render_x + d0).ceil().min(frame.max_x);
            let mut expected = Vec::new();
            let mut walls = 0;
            if frame.render_x > frame.max_x - d0 {
                reference_wall(&mut expected, frame.max_x, z_min, z_max, true, frame, cam);
                walls += 1;
            }
            if frame.render_x < frame.min_x + d0 {
                reference_wall(&mut expected, frame.min_x, z_min, z_max, true, frame, cam);
                walls += 1;
            }
            if frame.render_z > frame.max_z - d0 {
                reference_wall(&mut expected, frame.max_z, x_min, x_max, false, frame, cam);
                walls += 1;
            }
            if frame.render_z < frame.min_z + d0 {
                reference_wall(&mut expected, frame.min_z, x_min, x_max, false, frame, cam);
                walls += 1;
            }
            let actual = world_border_vertices(frame, cam);
            assert!(
                walls >= 2 && !expected.is_empty(),
                "the fixture must emit several walls"
            );
            assert_eq!(
                bytemuck::cast_slice::<WeatherVertex, u8>(&actual),
                bytemuck::cast_slice::<WeatherVertex, u8>(&expected),
                "render={render:?} cam={cam:?}"
            );
        }
    }

    /// GPURES-4: the retained stream hands each draw exactly the bytes its old private buffer
    /// held, at contiguous ranges in draw order, and a new frame starts from an empty stream.
    #[test]
    fn retained_weather_stream_packs_each_draw_at_its_own_range() {
        let rain = [WeatherVertex {
            pos: [1.0, 2.0, 3.0],
            uv: [0.25, 0.5],
            color: [1.0; 4],
        }; 6];
        let bolt = [BoltVertex {
            pos: [-1.0, 0.0, 9.5],
            color: [0.45, 0.45, 0.5, 0.3],
        }; 3];
        let border = [WeatherVertex {
            pos: [0.0; 3],
            uv: [7.5, 128.0],
            color: [0.1, 0.2, 0.3, 0.4],
        }; 12];
        let mut stream = WeatherVertexStream::default();
        for _frame in 0..2 {
            stream.bytes.clear();
            let ranges = [
                stream.push(rain.as_slice()),
                stream.push(bolt.as_slice()),
                stream.push(border.as_slice()),
            ];
            let expected: [&[u8]; 3] = [
                bytemuck::cast_slice(&rain),
                bytemuck::cast_slice(&bolt),
                bytemuck::cast_slice(&border),
            ];
            let mut next = 0;
            for (range, bytes) in ranges.iter().zip(expected) {
                assert_eq!(
                    range.start, next,
                    "draws are packed back to back in draw order"
                );
                assert_eq!(
                    &stream.bytes[range.start as usize..range.end as usize],
                    bytes
                );
                assert_eq!(
                    range.start % 4,
                    0,
                    "vertex-buffer offsets stay 4-byte aligned"
                );
                next = range.end;
            }
            assert_eq!(stream.bytes.len() as u64, next);
        }
    }

    /// Phase 4d review S2: the stream gives a burst's memory back. A small frame after a
    /// lightning storm shrinks both the GPU buffer and the CPU staging, and a frame with no
    /// weather at all releases them; an ordinary small frame never reallocates.
    #[test]
    fn weather_stream_gives_a_bursts_memory_back() {
        let Ok(gpu) = Gpu::headless(4, 4, crate::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let bolt = BoltVertex {
            pos: [0.0; 3],
            color: [0.45, 0.45, 0.5, 0.3],
        };
        let burst = vec![bolt; 200_000];
        let small = [bolt; 3];
        let frame = |stream: &mut WeatherVertexStream, vertices: &[BoltVertex]| {
            stream.bytes.clear();
            stream.push(vertices);
            stream.upload(&gpu).map(wgpu::Buffer::size)
        };
        let mut stream = WeatherVertexStream::default();
        let peak = frame(&mut stream, &burst).expect("burst uploaded");
        assert!(peak > WeatherVertexStream::SHRINK_ABOVE);

        assert_eq!(frame(&mut stream, &small), Some(64 * 1024));
        assert!(stream.bytes.capacity() as u64 <= WeatherVertexStream::SHRINK_ABOVE);
        assert_eq!(frame(&mut stream, &small), Some(64 * 1024));

        frame(&mut stream, &burst).expect("burst uploaded");
        stream.bytes.clear();
        stream.release_if_oversized();
        assert!(stream.buffer.is_none());
        assert!(stream.bytes.capacity() as u64 <= WeatherVertexStream::SHRINK_ABOVE);
    }

    /// Phase 5a review S4: the phase gives a burst's stream memory back from `tick_lightning`
    /// once the last bolt is gone and from `clear_lightning`, because `record` stops running
    /// when a dimension without weather disables the phase (and after a disconnect).
    #[test]
    fn lightning_ticks_and_clears_give_the_stream_memory_back() {
        let Ok(gpu) = Gpu::headless(4, 4, crate::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let bolt = BoltVertex {
            pos: [0.0; 3],
            color: [0.45, 0.45, 0.5, 0.3],
        };
        let burst = vec![bolt; 200_000];
        for clear in [false, true] {
            let mut phase = WeatherPhase::new(&gpu);
            phase.vertices.bytes.clear();
            phase.vertices.push(burst.as_slice());
            phase.vertices.upload(&gpu).expect("burst uploaded");
            if clear {
                phase.clear_lightning();
            } else {
                let _ = phase.tick_lightning();
            }
            assert!(phase.vertices.buffer.is_none(), "clear={clear}");
            assert!(
                phase.vertices.bytes.capacity() as u64 <= WeatherVertexStream::SHRINK_ABOVE,
                "clear={clear}"
            );
        }
    }

    #[test]
    fn weather_resource_owner_routes_rain_snow_and_forcefield_through_anaglyph_uploads() {
        let source = include_str!("weather.rs");
        let loader = concat!("textures::load_with_anaglyph", "(");
        assert_eq!(
            source.matches(loader).count(),
            3,
            "rain, snow, and forcefield must use the anaglyph-aware loader"
        );
        for path in [
            "textures/environment/rain.png",
            "textures/environment/snow.png",
            "textures/misc/forcefield.png",
        ] {
            let owner_call = source
                .split_once(path)
                .map(|(_, rest)| rest)
                .expect("weather resource path must have an owner call");
            assert!(
                owner_call.contains("self.anaglyph"),
                "weather resource owner lost anaglyph state for {path}"
            );
        }
        let setter = concat!("texture", ".set_anaglyph(gpu, enabled)");
        assert!(
            source.contains(setter),
            "live weather toggles must replay every resident source texture"
        );
    }

    /// A camera above the test worlds' ground (`height: 64`), because vanilla skips any column
    /// whose top is at or above `camera y + radius`.
    const CAM: DVec3 = DVec3::new(0.0, 80.0, 0.0);

    struct FixedEnv {
        temperature: f32,
        height: i32,
    }

    impl ColumnEnvironment for FixedEnv {
        fn temperature_at(&self, _x: i32, _y: i32, _z: i32) -> f32 {
            self.temperature
        }
        fn precipitation_height(&self, _x: i32, _z: i32) -> i32 {
            self.height
        }
        fn light_at(&self, _x: i32, _y: i32, _z: i32) -> u8 {
            15
        }
    }

    /// Fill every column sampled by `add_rain_particles` around the origin.  Keeping this
    /// fixture resident is important: the real client has a mutable shared blank chunk for
    /// packet compatibility, but rain impacts are only valid in loaded columns.
    fn rain_world(ground: BlockState) -> mc_world::World {
        let mut world = mc_world::World::new();
        for chunk_x in -1..=0 {
            for chunk_z in -1..=0 {
                world.load_empty_chunk(mc_math::ChunkPos::new(chunk_x, chunk_z));
            }
        }
        for x in -10..10 {
            for z in -10..10 {
                world.set_block(mc_math::BlockPos::new(x, 63, z), ground);
            }
        }
        world
    }

    fn has_particle(effects: &[AmbientEffect], particle_id: i32) -> bool {
        effects.iter().any(|effect| {
            matches!(
                effect,
                AmbientEffect::Particle(AmbientParticleEffect { particle_id: id, .. })
                    if *id == particle_id
            )
        })
    }

    fn has_sound(effects: &[AmbientEffect], name: &str) -> bool {
        effects
            .iter()
            .any(|effect| matches!(effect, AmbientEffect::Sound(sound) if sound.name == name))
    }

    #[test]
    fn no_rain_strength_means_no_columns() {
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        let cols = build_columns(&env, DVec3::ZERO, 0, false, 0.0);
        assert!(cols.is_empty());
    }

    #[test]
    fn rain_impact_pass_has_dry_lava_and_audio_branches() {
        let eye = DVec3::new(0.0, 64.0, 0.0);
        let dry_world = rain_world(BlockState::new(BlockId(1), 0));
        let dry = add_rain_particles(&dry_world, eye, 0, true, 0, 0.0, 7);
        assert!(dry.effects.is_empty());
        assert_eq!(
            dry.rain_sound_counter, 7,
            "dry weather does not advance audio state"
        );

        let lava_world = rain_world(BlockState::new(BlockId(10), 0));
        let lava = add_rain_particles(&lava_world, eye, 0, true, 0, 1.0, 0);
        assert!(has_particle(&lava.effects, ambient_particle::SMOKE_NORMAL));
        assert!(!has_particle(&lava.effects, ambient_particle::WATER_DROP));
        assert_eq!(
            lava.rain_sound_counter, 0,
            "lava impacts are smoke, not rain audio"
        );

        let stone_world = rain_world(BlockState::new(BlockId(1), 0));
        let mut heard = None;
        for tick in 0..64 {
            let pass = add_rain_particles(&stone_world, eye, tick, true, 0, 1.0, 1);
            if has_sound(&pass.effects, "ambient.weather.rain") {
                heard = Some(pass);
                break;
            }
        }
        let heard = heard.expect("solid precipitation eventually chooses a rain sound");
        assert_eq!(
            heard.rain_sound_counter, 0,
            "the successful Java post-increment branch resets the counter"
        );
        assert!(has_particle(&heard.effects, ambient_particle::WATER_DROP));
    }

    #[test]
    fn non_finite_rain_strength_takes_java_no_weather_branch() {
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        // java: `if (f > 0.0F)` is false for NaN. The old `f <= 0.0F` guard let this
        // through and produced columns whose alpha was NaN instead of drawing nothing.
        assert!(build_columns(&env, CAM, 0, false, f32::NAN).is_empty());
    }

    #[test]
    fn non_finite_temperature_falls_through_to_vanillas_snow_branch() {
        let env = FixedEnv {
            temperature: f32::NAN,
            height: 64,
        };
        // java: `temperature >= 0.15F` selects rain; NaN makes that comparison false, so the
        // `else` snow path wins. A direct `temperature < 0.15F` Rust port reverses this case.
        let cols = build_columns(&env, CAM, 0, false, 1.0);
        assert!(!cols.is_empty());
        assert!(cols.iter().all(|column| column.is_snow));
    }

    #[test]
    fn precipitation_batches_preserve_vanillas_contiguous_type_order() {
        let base = PrecipColumn {
            x: 1,
            z: 0,
            top_y: 74,
            bottom_y: 64,
            is_snow: false,
            alpha: 1.0,
            light: 15,
            animation_tick: 0,
            billboard_half_offset: streak_half_offset(1, 0),
        };
        let columns = [
            base,
            PrecipColumn {
                is_snow: true,
                x: 2,
                billboard_half_offset: streak_half_offset(2, 0),
                ..base
            },
            PrecipColumn {
                x: 3,
                billboard_half_offset: streak_half_offset(3, 0),
                ..base
            },
        ];
        let batches = ordered_streak_batches(&columns, DVec3::ZERO, 0.25);
        let kinds: Vec<_> = batches.iter().map(|batch| batch.kind).collect();
        assert_eq!(
            kinds,
            [
                PrecipitationKind::Rain,
                PrecipitationKind::Snow,
                PrecipitationKind::Rain
            ]
        );
        assert!(batches.iter().all(|batch| batch.vertices.len() == 6));

        // Negative control for the former implementation: collecting by texture would produce
        // Rain/Rain/Snow, which is observably a different source-over order at overlap seams.
        assert_ne!(
            kinds,
            [
                PrecipitationKind::Rain,
                PrecipitationKind::Rain,
                PrecipitationKind::Snow
            ]
        );
    }

    #[test]
    fn weather_blend_matches_javas_separate_colour_and_alpha_factors() {
        let blend = weather_blend_state();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::Zero);

        // Negative control: WGPU's convenience state keeps destination alpha with
        // OneMinusSrcAlpha, unlike Java's factor pair (1, 0).
        assert_ne!(blend, wgpu::BlendState::ALPHA_BLENDING);
    }

    /// java: `EntityRenderer.renderRainSnow` draws a column only when `k2 != l2`, and
    /// `k2 = max(floor(camY) - radius, precipHeight)`, `l2 = max(floor(camY) + radius,
    /// precipHeight)`. Both collapse onto `precipHeight` as soon as the column's top is at or
    /// above `camY + radius`, so a camera that far below the terrain sees no rain at all.
    /// mc-rs used to draw it anyway, which is why every test below now sits the camera above
    /// the ground.
    #[test]
    fn a_camera_below_the_column_top_draws_no_rain() {
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        // Camera at y = 0, ground at 64: k2 = l2 = 64.
        assert!(build_columns(&env, DVec3::ZERO, 0, false, 1.0).is_empty());
        // Camera at y = 59 with radius 5: k2 = l2 = 64 still — the boundary case.
        assert!(build_columns(&env, DVec3::new(0.0, 59.0, 0.0), 0, false, 1.0).is_empty());
        // One block higher and it draws.
        assert!(!build_columns(&env, DVec3::new(0.0, 60.0, 0.0), 0, false, 1.0).is_empty());
    }

    /// java: the temperature is sampled at `k2`, the bottom of the streak, not always at the
    /// column's precipitation height — so flying high over a mild biome turns its rain to snow.
    #[test]
    fn the_rain_snow_split_is_decided_at_the_streaks_bottom_not_always_the_ground() {
        /// Reports the Y it was asked about, scaled so the caller can read it back.
        struct HeightProbe;
        impl ColumnEnvironment for HeightProbe {
            fn temperature_at(&self, _x: i32, y: i32, _z: i32) -> f32 {
                // java: BiomeGenBase.getFloatTemperature for an Extreme Hills column.
                0.2 - (y as f32 - 64.0) * 0.05 / 30.0
            }
            fn precipitation_height(&self, _x: i32, _z: i32) -> i32 {
                70
            }
            fn light_at(&self, _x: i32, _y: i32, _z: i32) -> u8 {
                15
            }
        }
        // Camera just above the ground: k2 = 70, temp = 0.19 -> rain.
        let low = build_columns(&HeightProbe, DVec3::new(0.0, 75.0, 0.0), 0, true, 1.0);
        assert!(!low.is_empty() && low.iter().all(|c| !c.is_snow));
        // Camera at 150, fancy: k2 = 140, temp = 0.073 -> snow. Sampling at the ground would
        // still say 0.19 and keep it raining, which is what mc-rs did before.
        let high = build_columns(&HeightProbe, DVec3::new(0.0, 150.0, 0.0), 0, true, 1.0);
        assert!(!high.is_empty() && high.iter().all(|c| c.is_snow));
    }

    /// java: `EntityRenderer.renderRainSnow:1642-1650` and the four `pos(...)` calls at
    /// `:1687-1690` — the quad spans `y in [k2, l2]`, a band from ground level up to `radius`
    /// blocks **above** the camera. It used to span a fixed 20 blocks *downward* from the
    /// precipitation height, i.e. entirely at or below the player's feet and buried in terrain,
    /// where the shared depth buffer removed it.
    ///
    /// Negative control: on flat ground with the camera well above it, the top of the streak
    /// must be above the camera and the bottom at the ground — if either collapsed onto
    /// `precipitation_height` the band would have zero height, and `build_columns` would have
    /// skipped the column entirely via `k2 == l2`.
    #[test]
    fn a_streak_spans_a_band_around_the_camera_not_a_drop_from_the_ground() {
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        let cam = DVec3::new(0.0, 80.0, 0.0);
        let cols = build_columns(&env, cam, 0, true, 1.0);
        assert!(!cols.is_empty());
        for c in &cols {
            // java: `k2 = max(80 - 10, 64) = 70`, `l2 = max(80 + 10, 64) = 90`.
            assert_eq!((c.bottom_y, c.top_y), (70, 90), "{c:?}");
        }
        assert!(
            f64::from(cols[0].top_y) > cam.y,
            "rain must reach above the camera"
        );

        // Standing on the ground, the band's floor clamps to the terrain rather than going
        // below it: `k2 = max(64 - 10, 64) = 64`.
        let cols = build_columns(&env, DVec3::new(0.0, 64.0, 0.0), 0, true, 1.0);
        assert!(!cols.is_empty());
        assert_eq!((cols[0].bottom_y, cols[0].top_y), (64, 74));
    }

    #[test]
    fn warm_biome_gets_rain_cold_biome_gets_snow() {
        let warm = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        let cold = FixedEnv {
            temperature: 0.0,
            height: 64,
        };
        let rain_cols = build_columns(&warm, CAM, 0, false, 1.0);
        let snow_cols = build_columns(&cold, CAM, 0, false, 1.0);
        assert!(!rain_cols.is_empty());
        assert!(!snow_cols.is_empty());
        assert!(rain_cols.iter().all(|c| !c.is_snow));
        assert!(snow_cols.iter().all(|c| c.is_snow));
    }

    #[test]
    fn temperature_threshold_matches_vanillas_015() {
        let just_above = FixedEnv {
            temperature: 0.15,
            height: 64,
        };
        let just_below = FixedEnv {
            temperature: 0.14,
            height: 64,
        };
        assert!(!build_columns(&just_above, CAM, 0, false, 1.0)[0].is_snow);
        assert!(build_columns(&just_below, CAM, 0, false, 1.0)[0].is_snow);
    }

    #[test]
    fn fancy_graphics_doubles_the_search_radius() {
        assert_eq!(search_radius(false), 5);
        assert_eq!(search_radius(true), 10);
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        let fast = build_columns(&env, CAM, 0, false, 1.0);
        let fancy = build_columns(&env, CAM, 0, true, 1.0);
        assert!(fancy.len() > fast.len());
    }

    #[test]
    fn alpha_fades_toward_the_edge_of_the_search_radius() {
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        let cols = build_columns(&env, CAM, 0, true, 1.0);
        let centre = cols.iter().find(|c| c.x == 0 && c.z == 0).unwrap();
        let edge = cols.iter().max_by_key(|c| c.x * c.x + c.z * c.z).unwrap();
        assert!(centre.alpha > edge.alpha, "{centre:?} vs {edge:?}");
    }

    /// Java narrows `(float) l1 + 0.5F` before it subtracts `entity.posX`. At ordinary
    /// positions the narrowing is invisible, but around the playable border an f32's two-block
    /// ULP moves a column centre and changes its fade. This is a deliberately far-world witness
    /// for the tempting, old all-f64 expression.
    #[test]
    fn far_world_precipitation_fade_uses_the_java_f32_column_centre() {
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        let camera = DVec3::new(29_999_984.25, 80.0, -29_999_984.75);
        let x = camera.x.floor() as i32 + 1;
        let z = camera.z.floor() as i32;
        let column = build_columns(&env, camera, 0, false, 1.0)
            .into_iter()
            .find(|column| (column.x, column.z) == (x, z))
            .expect("the target column lies inside the 5-block square");

        // Direct transcription of Java's two float centres, then `sqrt_double`'s f32 result.
        let java_dx = f64::from(x as f32 + 0.5) - camera.x;
        let java_dz = f64::from(z as f32 + 0.5) - camera.z;
        let java_dist = (java_dx * java_dx + java_dz * java_dz).sqrt() as f32 / 5.0;
        let java_alpha = (1.0 - java_dist * java_dist) * 0.5 + 0.5;
        assert_eq!(column.alpha.to_bits(), java_alpha.to_bits());

        // Negative control: retaining the Rust f64 centre gives a different fade for this very
        // same submitted column, so the oracle cannot pass the pre-fix expression by accident.
        let exact_dx = f64::from(x) + 0.5 - camera.x;
        let exact_dz = f64::from(z) + 0.5 - camera.z;
        let exact_dist = (exact_dx * exact_dx + exact_dz * exact_dz).sqrt() as f32 / 5.0;
        let exact_alpha = (1.0 - exact_dist * exact_dist) * 0.5 + 0.5;
        assert_ne!(column.alpha.to_bits(), exact_alpha.to_bits());
    }

    #[test]
    fn columns_keep_vanillas_z_outer_x_inner_submission_order() {
        let env = FixedEnv {
            temperature: 0.8,
            height: 64,
        };
        let cols = build_columns(&env, CAM, 0, false, 1.0);
        assert!(cols.len() >= 2);
        assert_eq!((cols[0].x, cols[0].z), (-5, -5));
        assert_eq!((cols[1].x, cols[1].z), (-4, -5));
        // Negative control for accidentally restoring the old x-outer order.
        assert_ne!((cols[1].x, cols[1].z), (-5, -4));
    }

    #[test]
    fn rain_direction_table_matches_the_java_constructor() {
        // dx=3,dz=4: Java stores (-4/5, 3/5), then renderRainSnow halves both values.
        let (half_x, half_z) = streak_half_offset(3, 4).unwrap();
        assert_eq!(half_x, f64::from(-0.8_f32) * 0.5);
        assert_eq!(half_z, f64::from(0.3_f32));
        // The Java table's centre is 0/0 (NaN); the port deliberately skips that one primitive.
        assert_eq!(streak_half_offset(0, 0), None);
    }

    #[test]
    fn rain_and_snow_scroll_match_a_java_189_oracle() {
        let rain = PrecipColumn {
            x: 3,
            z: 4,
            top_y: 74,
            bottom_y: 64,
            is_snow: false,
            alpha: 1.0,
            light: 15,
            animation_tick: 37,
            billboard_half_offset: streak_half_offset(3, 4),
        };
        let mut random = JavaRandom::new(column_seed(rain.x, rain.z));
        assert_eq!(
            rain_scroll(&rain, 0.25, &mut random).to_bits(),
            0x3fd7_fb68_363e_6a3c
        );

        let snow = PrecipColumn {
            is_snow: true,
            ..rain
        };
        let mut random = JavaRandom::new(column_seed(snow.x, snow.z));
        let (u, v) = snow_scroll(&snow, 0.25, &mut random);
        assert_eq!(u.to_bits(), 0x3fe3_a65b_8ea2_569c);
        assert_eq!(v.to_bits(), 0x3fec_4c1a_12a0_9639);

        // Negative control for the old eight-tile animation: both formulas move within a tick.
        let mut at_zero = JavaRandom::new(column_seed(rain.x, rain.z));
        let mut at_quarter = JavaRandom::new(column_seed(rain.x, rain.z));
        assert_ne!(
            rain_scroll(&rain, 0.0, &mut at_zero).to_bits(),
            rain_scroll(&rain, 0.25, &mut at_quarter).to_bits()
        );
    }

    #[test]
    fn each_precipitation_column_emits_one_radial_plane() {
        let col = PrecipColumn {
            x: 3,
            z: 4,
            top_y: 74,
            bottom_y: 64,
            is_snow: false,
            alpha: 0.75,
            light: 15,
            animation_tick: 37,
            billboard_half_offset: streak_half_offset(3, 4),
        };
        let mut rain = Vec::new();
        let mut snow = Vec::new();
        streak_quads(&[col], DVec3::ZERO, 0.25, &mut rain, &mut snow);
        assert_eq!(
            rain.len(),
            6,
            "vanilla emits one quad, not two crossed quads"
        );
        assert!(snow.is_empty());
        assert_eq!(rain[0].pos, [3.9, 64.0, 4.2]);
        assert_eq!(rain[1].pos, [3.1, 64.0, 4.8]);
        assert_eq!(rain[2].pos, [3.1, 74.0, 4.8]);
        assert_eq!(rain[0].uv[0], 0.0);
        assert_eq!(rain[1].uv[0], 1.0);
        assert_eq!(rain[2].uv[1] - rain[1].uv[1], 2.5);

        // Vanilla indexes the direction table with the tick position but translates vertices
        // by the interpolated position. Crossing a block boundary during interpolation must
        // translate this plane without selecting a different table entry.
        rain.clear();
        streak_quads(
            &[col],
            DVec3::new(1.1, 0.0, 0.0),
            0.25,
            &mut rain,
            &mut snow,
        );
        let span_x = rain[1].pos[0] - rain[0].pos[0];
        let span_z = rain[1].pos[2] - rain[0].pos[2];
        assert!((span_x + 0.8).abs() < 1e-6, "{span_x}");
        assert!((span_z - 0.6).abs() < 1e-6, "{span_z}");

        // The centre table entry is NaN in Java and rasterises nothing. Skipping it must not
        // accidentally suppress a neighbouring, valid column as well.
        let centre = PrecipColumn {
            x: 0,
            z: 0,
            billboard_half_offset: None,
            ..col
        };
        rain.clear();
        streak_quads(&[centre, col], DVec3::ZERO, 0.25, &mut rain, &mut snow);
        assert_eq!(rain.len(), 6);
    }

    #[test]
    fn column_seed_matches_javas_int_wraparound_formula() {
        // Hand-computed for x=1,000,000 z=1,000,000, where naive i64 arithmetic (no wrapping)
        // would disagree with Java's 32-bit int overflow.
        let x = 1_000_000i32;
        let z = 1_000_000i32;
        let a = x
            .wrapping_mul(x)
            .wrapping_mul(3121)
            .wrapping_add(x.wrapping_mul(45_238_971));
        let b = z
            .wrapping_mul(z)
            .wrapping_mul(418_711)
            .wrapping_add(z.wrapping_mul(13_761));
        assert_eq!(column_seed(x, z), i64::from(a ^ b));
    }

    #[test]
    fn column_seed_is_deterministic_and_position_dependent() {
        assert_eq!(column_seed(5, 7), column_seed(5, 7));
        assert_ne!(column_seed(5, 7), column_seed(7, 5));
    }

    #[test]
    fn snow_light_is_blended_toward_full_bright() {
        struct DimEnv;
        impl ColumnEnvironment for DimEnv {
            fn temperature_at(&self, _: i32, _: i32, _: i32) -> f32 {
                0.0
            }
            fn precipitation_height(&self, _: i32, _: i32) -> i32 {
                64
            }
            fn light_at(&self, _: i32, _: i32, _: i32) -> u8 {
                0
            }
        }
        let cols = build_columns(&DimEnv, CAM, 0, false, 1.0);
        assert!(
            !cols.is_empty(),
            "the light assertion must inspect at least one column"
        );
        // 3:1 blend of 0 toward 15 -> (0*3+15)/4 = 3.
        assert!(cols.iter().all(|c| c.light == 3));
    }

    // ------------------------------------------------------- lightning: RenderLightningBolt

    #[test]
    fn lightning_keeps_the_weather_phase_alive_outside_the_overworld() {
        // `RenderGlobal` renders World.weatherEffects after the normal entity pass without a
        // `WorldProvider.isSurfaceWorld` gate.  The phase therefore stays live for a bolt in
        // both skyless dimensions, but an otherwise-empty phase stays off there.
        for dimension in [Dimension::Nether, Dimension::End] {
            assert!(
                weather_phase_enabled(dimension, true, false),
                "a lightning bolt must render in {dimension:?}"
            );
            assert!(
                !weather_phase_enabled(dimension, false, false),
                "ordinary rain/snow remains disabled in {dimension:?}"
            );
        }
        assert!(weather_phase_enabled(Dimension::Overworld, false, false));
    }

    #[test]
    fn border_phase_is_live_in_skyless_dimensions_and_wall_quads_are_clipped() {
        let frame = WorldBorderFrame {
            min_x: 0.0,
            max_x: 16.0,
            min_z: 0.0,
            max_z: 16.0,
            player_x: 15.0,
            player_z: 8.0,
            render_x: 15.0,
            render_z: 8.0,
            render_distance: 4.0,
            alpha: 0.5,
            color: [1.0, 0.0, 0.0, 1.0],
            texture_phase: 0.25,
        };
        let vertices = world_border_vertices(frame, DVec3::ZERO);
        // Only the east wall is near the render-view position; [4,12] contributes eight
        // one-block quads, each expanded to six triangle-list vertices.
        assert_eq!(vertices.len(), 8 * 6);
        assert!(vertices.iter().all(|vertex| vertex.pos[0] == 16.0));
        assert_eq!(vertices[0].uv, [0.25, 0.25]);
        assert_eq!(vertices[1].uv, [0.75, 0.25]);
        assert_eq!(vertices[2].uv, [0.75, 128.25]);
        assert_eq!(vertices[5].uv, [0.25, 128.25]);
        assert_eq!(vertices[6].uv, [0.75, 0.25]);

        assert!(weather_phase_enabled(Dimension::Nether, false, true));
        assert!(weather_phase_enabled(Dimension::End, false, true));
        // Biting negative control: border work must not depend on the ordinary weather gate.
        assert!(!weather_phase_enabled(Dimension::Nether, false, false));
    }

    #[test]
    fn lightning_entity_starts_with_vanillas_two_visible_state_ticks() {
        // `EntityLightningBolt` begins at 2, then `onUpdate` decrements before doing any
        // negative-state work.  Rendering itself has no state gate, so both 1 and 0 remain a
        // visible entity state.
        let mut strike = LightningStrike {
            position: DVec3::ZERO,
            bolt_vertex: 7,
        };
        let mut lifetime = LightningLifetime {
            lightning_state: 2,
            bolt_living_time: 3,
            random: JavaRandom::new(99),
        };

        let first_tick = lifetime.tick(&mut strike);
        assert!(first_tick.alive);
        assert!(first_tick.flash);
        assert_eq!(lifetime.lightning_state, 1);
        let second_tick = lifetime.tick(&mut strike);
        assert!(second_tick.alive);
        assert!(second_tick.flash);
        assert_eq!(lifetime.lightning_state, 0);
        assert_eq!(strike.bolt_vertex, 7);
    }

    #[test]
    fn lightning_entity_tick_reports_flash_without_confusing_dark_or_dead_ticks() {
        // This CPU seam is the exact value WeatherPhase::tick_lightning returns to the client
        // driver. The initial visible states re-arm the world flash; a retained negative state
        // does not; a re-strike does; and final death does not. The negative controls here force
        // every divergent branch instead of only sampling the common two-tick window.
        let strike = LightningStrike {
            position: DVec3::ZERO,
            bolt_vertex: 7,
        };
        let mut strikes = vec![strike];
        let mut lifetimes = vec![LightningLifetime {
            lightning_state: 2,
            bolt_living_time: 2,
            random: JavaRandom::new(99),
        }];

        assert!(tick_lightning_entities(&mut strikes, &mut lifetimes));
        assert_eq!(lifetimes[0].lightning_state, 1);
        assert!(tick_lightning_entities(&mut strikes, &mut lifetimes));
        assert_eq!(lifetimes[0].lightning_state, 0);

        let (seed, wait) = (0_i64..100)
            .map(|seed| {
                let mut random = JavaRandom::new(seed);
                (seed, random.next_i32_bound(10))
            })
            .find(|(_, wait)| *wait > 0)
            .expect("JavaRandom must produce a nonzero nextInt(10) in this small range");
        lifetimes[0] = LightningLifetime {
            lightning_state: 1 - wait,
            bolt_living_time: 1,
            random: JavaRandom::new(seed),
        };
        assert!(!tick_lightning_entities(&mut strikes, &mut lifetimes));
        assert_eq!(lifetimes[0].lightning_state, -wait);

        lifetimes[0] = LightningLifetime {
            lightning_state: -wait,
            bolt_living_time: 1,
            random: JavaRandom::new(seed),
        };
        assert!(tick_lightning_entities(&mut strikes, &mut lifetimes));
        assert_eq!(lifetimes[0].lightning_state, 1);

        lifetimes[0] = LightningLifetime {
            lightning_state: 0,
            bolt_living_time: 0,
            random: JavaRandom::new(seed),
        };
        assert!(!tick_lightning_entities(&mut strikes, &mut lifetimes));
        assert!(strikes.is_empty());
        assert!(lifetimes.is_empty());
    }

    #[test]
    fn lightning_negative_window_waits_on_the_strict_java_comparison() {
        // Java's `lightningState < -rand.nextInt(10)` is strict.  The equality tick is a dark
        // but still-rendered entity tick, not a re-strike or despawn.  Find a stable seed whose
        // first bounded draw is nonzero so this genuinely exercises a negative state.
        let (seed, wait) = (0_i64..100)
            .map(|seed| {
                let mut random = JavaRandom::new(seed);
                (seed, random.next_i32_bound(10))
            })
            .find(|(_, wait)| *wait > 0)
            .expect("JavaRandom must produce a nonzero nextInt(10) in this small range");
        let original_vertex = 0x1234_5678_i64;
        let mut strike = LightningStrike {
            position: DVec3::ZERO,
            bolt_vertex: original_vertex,
        };
        let mut lifetime = LightningLifetime {
            // `tick` decrements first, leaving exactly `-wait` for the comparison.
            lightning_state: 1 - wait,
            bolt_living_time: 2,
            random: JavaRandom::new(seed),
        };

        let tick = lifetime.tick(&mut strike);
        assert!(tick.alive);
        assert!(!tick.flash);
        assert_eq!(lifetime.lightning_state, -wait);
        assert_eq!(lifetime.bolt_living_time, 2);
        assert_eq!(strike.bolt_vertex, original_vertex);
    }

    #[test]
    fn lightning_restrike_consumes_java_wait_then_rerolls_the_vertex() {
        // On the next negative tick Java consumes `nextInt(10)`, decrements
        // `boltLivingTime`, restores state 1, then consumes `nextLong` for the new geometry.
        let (seed, wait) = (0_i64..100)
            .map(|seed| {
                let mut random = JavaRandom::new(seed);
                (seed, random.next_i32_bound(10))
            })
            .find(|(_, wait)| *wait > 0)
            .expect("JavaRandom must produce a nonzero nextInt(10) in this small range");
        let mut expected = JavaRandom::new(seed);
        assert_eq!(expected.next_i32_bound(10), wait);
        let expected_vertex = expected.next_i64();
        let mut strike = LightningStrike {
            position: DVec3::ZERO,
            bolt_vertex: -1,
        };
        let mut lifetime = LightningLifetime {
            // After `tick` this is `-wait - 1`, strictly below Java's threshold.
            lightning_state: -wait,
            bolt_living_time: 2,
            random: JavaRandom::new(seed),
        };

        let tick = lifetime.tick(&mut strike);
        assert!(tick.alive);
        assert!(tick.flash);
        assert_eq!(lifetime.lightning_state, 1);
        assert_eq!(lifetime.bolt_living_time, 1);
        assert_eq!(strike.bolt_vertex, expected_vertex);
    }

    #[test]
    fn lightning_despawns_only_after_the_last_restrike_window() {
        // `boltLivingTime == 0` is checked only after decrementing into a negative state.  It
        // is not the sky flash counter and it must not clear unrelated strikes.
        let mut strike = LightningStrike {
            position: DVec3::ZERO,
            bolt_vertex: 3,
        };
        let mut lifetime = LightningLifetime {
            lightning_state: 0,
            bolt_living_time: 0,
            random: JavaRandom::new(3),
        };
        let tick = lifetime.tick(&mut strike);
        assert!(!tick.alive);
        assert!(!tick.flash);
        assert_eq!(lifetime.lightning_state, -1);
        assert_eq!(strike.bolt_vertex, 3);
    }

    #[test]
    fn a_bolt_has_vanillas_exact_segment_count() {
        // 4 width passes x (8 main-bolt levels + 3 + 3 branch levels) = 56 `tessellator.draw()`
        // calls in `RenderLightningBolt.doRender`, for every possible `boltVertex`.
        for seed in [0i64, 1, -1, 42, i64::MIN, i64::MAX, 0x5DEECE66D] {
            assert_eq!(lightning_bolt_segments(seed).len(), 56, "seed {seed}");
        }
    }

    #[test]
    fn a_bolt_spans_exactly_eight_sixteen_block_levels() {
        let segments = lightning_bolt_segments(42);
        let ys: Vec<f64> = segments
            .iter()
            .flat_map(|s| s.verts.iter().map(|v| v.y))
            .collect();
        let min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // java: `y + (double)(i1 * 16)` with i1 in 0..=7, and the upper vertex at (i1 + 1) * 16.
        assert!((min - 0.0).abs() < 1e-9, "{min}");
        assert!((max - 128.0).abs() < 1e-9, "{max}");
        // Every y is a multiple of 16 — nothing interpolates between levels.
        assert!(ys.iter().all(|y| (y / 16.0).fract().abs() < 1e-9));
    }

    #[test]
    fn the_main_bolts_first_segment_matches_a_hand_run_of_the_java() {
        // Hand-derived from `RenderLightningBolt.doRender` with boltVertex = 0, using
        // `JavaRandom` (identical LCG) for the two `nextInt(11) - 5` sequences the Java draws.
        let seed = 0i64;

        // Pre-pass: `for (int i = 7; i >= 0; --i) { adouble[i] = d0; ...; d0 += nextInt(11)-5; }`
        let mut r = JavaRandom::new(seed);
        let mut spine_x = [0.0f64; 8];
        let mut spine_z = [0.0f64; 8];
        let (mut d0, mut d1) = (0.0f64, 0.0f64);
        for i in (0..8).rev() {
            spine_x[i] = d0;
            spine_z[i] = d1;
            d0 += f64::from(r.next_i32_bound(11) - 5);
            d1 += f64::from(r.next_i32_bound(11) - 5);
        }

        // First render segment: k1 = 0, j = 0, i1 = 7, with a freshly re-seeded Random.
        let mut r1 = JavaRandom::new(seed);
        let d4 = spine_x[7] - d0;
        let d5 = spine_z[7] - d1;
        let d2 = d4 + f64::from(r1.next_i32_bound(11) - 5);
        let d3 = d5 + f64::from(r1.next_i32_bound(11) - 5);
        let d6 = 0.1 * (7.0 * 0.1 + 1.0); // upper vertex half-width
        let d7 = 0.1 * (6.0 * 0.1 + 1.0); // lower vertex half-width

        let seg = lightning_bolt_segments(seed)[0];
        // j1 = 0 emits the (-,-) corner: lower vertex first, then upper.
        assert!((seg.verts[0].x - (0.5 - d7 + d2)).abs() < 1e-12, "{seg:?}");
        assert!((seg.verts[0].y - 7.0 * 16.0).abs() < 1e-12);
        assert!((seg.verts[0].z - (0.5 - d7 + d3)).abs() < 1e-12);
        assert!((seg.verts[1].x - (0.5 - d6 + d4)).abs() < 1e-12);
        assert!((seg.verts[1].y - 8.0 * 16.0).abs() < 1e-12);
        assert!((seg.verts[1].z - (0.5 - d6 + d5)).abs() < 1e-12);
        // j1 = 4 closes the tube back onto the j1 = 0 corner.
        assert_eq!(seg.verts[8], seg.verts[0]);
        assert_eq!(seg.verts[9], seg.verts[1]);
    }

    #[test]
    fn the_main_bolt_tapers_toward_the_ground_and_the_branches_do_not() {
        let segments = lightning_bolt_segments(1234);
        // Segment 0 is k1=0/j=0/i1=7 (top of the main bolt); segment 7 is i1=0 (the ground end).
        let width = |s: &BoltSegment| (s.verts[2].x - s.verts[0].x).abs();
        assert!(
            width(&segments[0]) > width(&segments[7]),
            "top {} should be fatter than bottom {}",
            width(&segments[0]),
            width(&segments[7])
        );
        // Segments 8..=10 are j=1's branch, whose half-width is the constant `0.1 + k1 * 0.2`.
        let branch: Vec<f64> = segments[8..11].iter().map(width).collect();
        assert!(
            branch.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9),
            "{branch:?}"
        );
    }

    #[test]
    fn the_four_width_passes_trace_the_same_path_at_increasing_radius() {
        let segments = lightning_bolt_segments(99);
        // Segment 0 and segment 14 are the same (j=0, i1=7) level at k1=0 and k1=1.
        let a = segments[0];
        let b = segments[14];
        let centre = |s: &BoltSegment| (s.verts[0] + s.verts[4]) * 0.5;
        let ca = centre(&a);
        let cb = centre(&b);
        assert!((ca - cb).length() < 1e-9, "{ca:?} vs {cb:?}");
        let width = |s: &BoltSegment| (s.verts[2].x - s.verts[0].x).abs();
        // java: 0.1 + k1 * 0.2 -> 0.1 then 0.3, tripling the tube.
        assert!(
            width(&b) > width(&a) * 2.5,
            "{} vs {}",
            width(&b),
            width(&a)
        );
    }

    #[test]
    fn bolt_geometry_is_deterministic_and_seed_dependent() {
        assert_eq!(lightning_bolt_segments(7), lightning_bolt_segments(7));
        assert_ne!(lightning_bolt_segments(7), lightning_bolt_segments(8));
    }
}
