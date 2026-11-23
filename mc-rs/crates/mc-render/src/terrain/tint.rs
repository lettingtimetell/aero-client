//! Biome tinting: the grass/foliage colormaps, water colour, and the per-block
//! `colorMultiplier` dispatch that `BakedQuad::tint_index` selects.
//!
//! Vanilla resolves a tint in three steps:
//!
//! 1. `Block.colorMultiplier(world, pos, tintIndex)` picks *which* colour source applies —
//!    grass, foliage, water, a constant, or nothing.
//! 2. `BiomeColorHelper` averages that source over the 3×3 column around the block, which is
//!    what makes biome borders fade instead of snapping.
//! 3. `ColorizerGrass`/`ColorizerFoliage` look the biome's temperature/rainfall up in a
//!    256×256 colormap PNG.
//!
//! Step 3 needs `textures/colormap/{grass,foliage}.png` from the resource pack, so the
//! colormaps are installed once at startup ([`install_colormaps`]) into a process-global,
//! mirroring `ColorizerGrass.setGrassBiomeColorizer`. Before that, the buffers have vanilla's
//! zero-filled initial value, so every lookup is black — [`colormaps_installed`] reports which
//! state you are in. On a later reload, each failed map retains its previous buffer, as vanilla
//! does.
//
// java: world/ColorizerGrass.java, world/ColorizerFoliage.java,
//       world/biome/BiomeColorHelper.java, world/biome/BiomeGenBase.java,
//       block/BlockGrass.java, BlockLeaves.java, BlockVine.java, BlockTallGrass.java,
//       BlockDoublePlant.java, BlockReed.java, BlockLilyPad.java, BlockStem.java,
//       BlockRedstoneWire.java, BlockLiquid.java

use mc_assets::model::HotSwap;
use mc_assets::{AssetError, ResourceLocation, ResourceManager};
use mc_data::{block_props, BlockState, Material};
use mc_math::BlockPos;
use mc_world::World;

/// A 256×256 packed-RGB lookup table, as `ColorizerGrass`/`ColorizerFoliage` hold one.
#[derive(Debug, Clone)]
pub struct Colormap {
    /// Row-major `0x00RRGGBB`, exactly `65536` entries.
    pixels: Vec<u32>,
}

impl Colormap {
    fn zeroed() -> Self {
        Self {
            pixels: vec![0; 65_536],
        }
    }

    /// Decode a colormap PNG. Vanilla reads these with `TextureUtil.readImageData`, which
    /// discards alpha and keeps the image row-major.
    ///
    /// # Errors
    /// If the image cannot be decoded, or is not 256×256.
    pub fn from_png(name: &str, bytes: &[u8]) -> Result<Self, AssetError> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| AssetError::Image {
                location: name.to_owned(),
                message: e.to_string(),
            })?
            .to_rgba8();
        if img.width() != 256 || img.height() != 256 {
            return Err(AssetError::Image {
                location: name.to_owned(),
                message: format!(
                    "colormap must be 256x256, got {}x{}",
                    img.width(),
                    img.height()
                ),
            });
        }
        let pixels = img
            .pixels()
            .map(|p| (u32::from(p[0]) << 16) | (u32::from(p[1]) << 8) | u32::from(p[2]))
            .collect();
        Ok(Self { pixels })
    }

    /// java: `ColorizerGrass.getGrassColor` / `ColorizerFoliage.getFoliageColor` — the shared
    /// body of both, which differ only in which buffer they read.
    ///
    /// `temperature` and `rainfall` are already clamped to `0..=1` by the caller, as
    /// `BiomeGenBase.getGrassColorAtPos` does.
    #[must_use]
    pub fn lookup(&self, temperature: f64, rainfall: f64) -> u32 {
        let rainfall = rainfall * temperature;
        let i = ((1.0 - temperature) * 255.0) as i32;
        let j = ((1.0 - rainfall) * 255.0) as i32;
        let k = (j << 8) | i;
        // java: ColorizerGrass returns -65281 (magenta) when the index is out of range, and
        // ColorizerFoliage does not check at all. The check is unreachable for the clamped
        // 0..=1 inputs `BiomeGenBase` passes, so this is a safety net, not a behaviour.
        if k < 0 || k as usize >= self.pixels.len() {
            return 0x00FF_00FF;
        }
        self.pixels[k as usize]
    }
}

/// The two colormaps vanilla loads at startup.
#[derive(Debug, Clone)]
pub struct Colormaps {
    /// `textures/colormap/grass.png`.
    pub grass: Colormap,
    /// `textures/colormap/foliage.png`.
    pub foliage: Colormap,
}

static COLORMAPS: HotSwap<Colormaps> = HotSwap::new();

std::thread_local! {
    /// Per-thread copy of `COLORMAPS`, validated against its generation. Same reasoning as
    /// `mc_assets::model`'s registry cache: `color_multiplier` runs on `rayon` workers and the
    /// 3×3 averaging makes nine lookups per tinted block, so a lock per lookup would put every
    /// worker on one cache line.
    static CACHED_COLORMAPS: std::cell::Cell<(u64, Option<&'static Colormaps>)> =
        const { std::cell::Cell::new((0, None)) };
}

/// The generation [`with_thread_colormaps`] parks in
/// `CACHED_COLORMAPS`. A real generation counts installs and never reaches it, so the ordinary
/// path's compare can never accept it and the override costs that path nothing.
const OVERRIDE_GENERATION: u64 = u64::MAX;

fn colormaps() -> Option<&'static Colormaps> {
    let generation = COLORMAPS.generation();
    CACHED_COLORMAPS.with(|cache| {
        let (cached_generation, cached) = cache.get();
        if cached_generation == generation || cached_generation == OVERRIDE_GENERATION {
            return cached;
        }
        let current = COLORMAPS.load();
        cache.set((generation, current));
        current
    })
}

/// Run `f` with **this thread's** [`grass_color`]/[`foliage_color`] (and so the mesher's tints)
/// reading `maps` instead of the installed colormaps, then restore what the thread saw before,
/// also on a panic. Nests. Nothing global changes: other threads and [`colormap_generation`]
/// are untouched.
///
/// The colormap twin of `mc_assets::model::with_thread_registry`, for the Aero title screen's
/// vanilla-textured background.
pub fn with_thread_colormaps<R>(maps: &'static Colormaps, f: impl FnOnce() -> R) -> R {
    struct Restore((u64, Option<&'static Colormaps>));
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0;
            let _ = CACHED_COLORMAPS.try_with(|cache| cache.set(previous));
        }
    }
    let previous = CACHED_COLORMAPS.with(|cache| cache.replace((OVERRIDE_GENERATION, Some(maps))));
    let _restore = Restore(previous);
    f()
}

/// Whether [`with_thread_colormaps`] is active on this thread.
#[must_use]
pub fn thread_colormaps_overridden() -> bool {
    CACHED_COLORMAPS.with(|cache| cache.get().0 == OVERRIDE_GENERATION)
}

/// Publish the colormaps for [`grass_color`]/[`foliage_color`] to read. **Only the first call
/// wins**, matching `mc_assets::install`'s single-shot behaviour; use [`reinstall_colormaps`] to
/// replace them on a resource-pack reload.
pub fn install_colormaps(maps: Colormaps) {
    COLORMAPS.install(maps);
}

/// Replace the colormaps, for a resource-pack reload. Returns the new generation.
///
/// java: `ColorizerGrass`/`ColorizerFoliage` are re-fed by `GrassColorReloadListener` and
/// `FoliageColorReloadListener` on every `reloadResources`, which is exactly this.
///
/// The previous set is leaked; see [`mc_assets::model::HotSwap`] for why that is required rather
/// than sloppy. Two 256 KiB tables per distinct pair: A reload whose
/// maps equal a set published before (the same packs again, or a pack without colormaps)
/// publishes that set again instead of leaking a copy, so every reload no longer costs 512 KiB.
pub fn reinstall_colormaps(maps: Colormaps) -> u64 {
    COLORMAPS.reinstall_dedup(maps, |new, published| {
        new.grass.pixels == published.grass.pixels && new.foliage.pixels == published.foliage.pixels
    })
}

/// How many colormap sets have been published (and are kept).
#[must_use]
pub fn distinct_colormap_sets() -> usize {
    COLORMAPS.distinct_published()
}

/// How many times colormaps have been installed: `0` before [`install_colormaps`], one more per
/// [`reinstall_colormaps`].
#[must_use]
pub fn colormap_generation() -> u64 {
    COLORMAPS.generation()
}

/// Decode both colormaps from a resource manager without publishing them.
///
/// # Errors
/// If either PNG is missing or malformed.
pub fn read_colormaps(rm: &ResourceManager) -> Result<Colormaps, AssetError> {
    let grass = read_colormap(rm, "textures/colormap/grass.png")?;
    let foliage = read_colormap(rm, "textures/colormap/foliage.png")?;
    Ok(Colormaps { grass, foliage })
}

fn read_colormap(rm: &ResourceManager, path: &str) -> Result<Colormap, AssetError> {
    let loc = ResourceLocation::minecraft(path);
    Colormap::from_png(path, &rm.load(&loc)?)
}

/// Load and install both colormaps from a resource manager. First call wins.
///
/// # Errors
/// If either PNG is missing or malformed.
pub fn load_colormaps(rm: &ResourceManager) -> Result<(), AssetError> {
    install_colormaps(read_colormaps(rm)?);
    Ok(())
}

/// Load the colormaps from a resource manager and **replace** each one that loaded successfully.
///
/// # Errors
/// If either PNG is missing or malformed. As in vanilla's two independent reload listeners, a
/// successful sibling is still installed and a failed map keeps its previous value (or the
/// zero-filled initial buffer on the first load).
pub fn reload_colormaps(rm: &ResourceManager) -> Result<u64, AssetError> {
    let grass = read_colormap(rm, "textures/colormap/grass.png");
    let foliage = read_colormap(rm, "textures/colormap/foliage.png");
    match (grass, foliage) {
        (Ok(grass), Ok(foliage)) => Ok(reinstall_colormaps(Colormaps { grass, foliage })),
        (Ok(grass), Err(error)) => {
            let maps = merge_partial_reload(colormaps(), Some(grass), None)
                .expect("one colormap loaded successfully");
            reinstall_colormaps(maps);
            Err(error)
        }
        (Err(error), Ok(foliage)) => {
            let maps = merge_partial_reload(colormaps(), None, Some(foliage))
                .expect("one colormap loaded successfully");
            reinstall_colormaps(maps);
            Err(error)
        }
        (Err(error), Err(_)) => Err(error),
    }
}

fn merge_partial_reload(
    previous: Option<&Colormaps>,
    grass: Option<Colormap>,
    foliage: Option<Colormap>,
) -> Option<Colormaps> {
    if grass.is_none() && foliage.is_none() {
        return None;
    }
    Some(Colormaps {
        grass: grass
            .unwrap_or_else(|| previous.map_or_else(Colormap::zeroed, |maps| maps.grass.clone())),
        foliage: foliage
            .unwrap_or_else(|| previous.map_or_else(Colormap::zeroed, |maps| maps.foliage.clone())),
    })
}

/// Whether [`install_colormaps`] or [`reinstall_colormaps`] has published a map. Tint lookups are
/// total either way; before the first successful load they read vanilla's zero-filled buffers.
#[must_use]
pub fn colormaps_installed() -> bool {
    COLORMAPS.generation() > 0
}

/// Vanilla's un-tinted white, `0xFFFFFF`. java: `Block.colorMultiplier`'s default.
pub const NO_TINT: u32 = 16_777_215;

// ---------------------------------------------------------------------------
// `BiomeGenBase.temperatureNoise` and `BiomeGenBase.GRASS_COLOR_NOISE`.
// ---------------------------------------------------------------------------

/// java: `world/gen/NoiseGeneratorSimplex.java`, the 2-D single-sample path (`func_151605_a`)
/// only. The 3-D array path (`func_151606_a`) is worldgen and is not needed here.
struct SimplexNoise {
    /// java: `field_151608_f`, the 512-entry doubled permutation.
    permutation: [i32; 512],
}

/// java: `NoiseGeneratorSimplex.field_151611_e` — 12 gradients; only `[0]` and `[1]` are read
/// by the 2-D sample.
const SIMPLEX_GRAD_2D: [[f64; 2]; 12] = [
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [0.0, 1.0],
    [0.0, -1.0],
];

impl SimplexNoise {
    /// java: `NoiseGeneratorSimplex(Random)`. The three `nextDouble` offsets are only read by
    /// the 3-D path, but they **must** still be drawn: they advance the LCG before the
    /// permutation shuffle, so dropping them changes every subsequent sample.
    fn new(rng: &mut mc_math::JavaRandom) -> Self {
        let _x_offset = rng.next_f64() * 256.0;
        let _y_offset = rng.next_f64() * 256.0;
        let _z_offset = rng.next_f64() * 256.0;
        let mut permutation = [0i32; 512];
        for (i, slot) in permutation.iter_mut().take(256).enumerate() {
            *slot = i as i32;
        }
        for l in 0..256usize {
            // java: `int j = rand.nextInt(256 - l) + l;` then swap, then mirror into l+256.
            let j = rng.next_i32_bound(256 - l as i32) as usize + l;
            permutation.swap(l, j);
            permutation[l + 256] = permutation[l];
        }
        Self { permutation }
    }

    /// java: `NoiseGeneratorSimplex.func_151607_a` — a floor that returns `-1` for an exact
    /// `0.0`, because the test is `> 0.0` and not `>= 0.0`. Reproduced deliberately.
    fn java_floor(v: f64) -> i32 {
        if v > 0.0 {
            v as i32
        } else {
            v as i32 - 1
        }
    }

    /// java: `NoiseGeneratorSimplex.func_151605_a(double, double)`.
    #[allow(clippy::many_single_char_names)]
    fn sample(&self, x: f64, z: f64) -> f64 {
        // java: `field_151614_a = Math.sqrt(3.0)`, then `d3` and `d5` recomputed inline.
        let sqrt3 = 3.0f64.sqrt();
        let d3 = 0.5 * (sqrt3 - 1.0);
        let d4 = (x + z) * d3;
        let i = Self::java_floor(x + d4);
        let j = Self::java_floor(z + d4);
        let d5 = (3.0 - sqrt3) / 6.0;
        let d6 = f64::from(i + j) * d5;
        let d7 = f64::from(i) - d6;
        let d8 = f64::from(j) - d6;
        let d9 = x - d7;
        let d10 = z - d8;
        let (k, l): (i32, i32) = if d9 > d10 { (1, 0) } else { (0, 1) };
        let d11 = d9 - f64::from(k) + d5;
        let d12 = d10 - f64::from(l) + d5;
        let d13 = d9 - 1.0 + 2.0 * d5;
        let d14 = d10 - 1.0 + 2.0 * d5;
        let i1 = (i & 255) as usize;
        let j1 = (j & 255) as usize;
        let p = &self.permutation;
        let k1 = (p[i1 + p[j1] as usize] % 12) as usize;
        let l1 = (p[i1 + k as usize + p[j1 + l as usize] as usize] % 12) as usize;
        let i2 = (p[i1 + 1 + p[j1 + 1] as usize] % 12) as usize;
        let corner = |t: f64, g: usize, a: f64, b: f64| {
            if t < 0.0 {
                0.0
            } else {
                let t2 = t * t;
                t2 * t2 * (SIMPLEX_GRAD_2D[g][0] * a + SIMPLEX_GRAD_2D[g][1] * b)
            }
        };
        let d0 = corner(0.5 - d9 * d9 - d10 * d10, k1, d9, d10);
        let d1 = corner(0.5 - d11 * d11 - d12 * d12, l1, d11, d12);
        let d2 = corner(0.5 - d13 * d13 - d14 * d14, i2, d13, d14);
        70.0 * (d0 + d1 + d2)
    }
}

/// java: `BiomeGenBase.temperatureNoise = new NoiseGeneratorPerlin(new Random(1234L), 1)`.
///
/// A one-octave `NoiseGeneratorPerlin` is `simplex.func_151605_a(x, z) / 1.0`, i.e. the simplex
/// sample unchanged (`world/gen/NoiseGeneratorPerlin.java:22-33`).
///
/// Both of vanilla's colour noises are seeded from **hard-coded literals**, not from the world
/// seed, so they are fixed functions of `(x, z)` in every 1.8.9 world and need no worldgen state
/// whatsoever — the three comments in this tree that called them "worldgen state a multiplayer
/// client does not carry" were wrong.
fn temperature_noise() -> &'static SimplexNoise {
    static N: std::sync::OnceLock<SimplexNoise> = std::sync::OnceLock::new();
    N.get_or_init(|| SimplexNoise::new(&mut mc_math::JavaRandom::new(1234)))
}

/// java: `BiomeGenBase.GRASS_COLOR_NOISE = new NoiseGeneratorPerlin(new Random(2345L), 1)`.
fn grass_color_noise() -> &'static SimplexNoise {
    static N: std::sync::OnceLock<SimplexNoise> = std::sync::OnceLock::new();
    N.get_or_init(|| SimplexNoise::new(&mut mc_math::JavaRandom::new(2345)))
}

/// The `noise` term [`mc_data::Biome::temperature_at`] takes: java's
/// `(float)(temperatureNoise.func_151601_a(x / 8.0, z / 8.0) * 4.0)`
/// (`BiomeGenBase.getFloatTemperature`, `BiomeGenBase.java:404-418`).
#[must_use]
pub fn temperature_noise_at(x: i32, z: i32) -> f32 {
    (temperature_noise().sample(f64::from(x) / 8.0, f64::from(z) / 8.0) * 4.0) as f32
}

/// java: `BiomeGenSwamp.getGrassColorAtPos` — `GRASS_COLOR_NOISE` at the `0.0225` scale, whose
/// sign picks between the swamp's two grass colours.
fn swamp_grass_color(x: i32, z: i32) -> u32 {
    let d0 = grass_color_noise().sample(f64::from(x) * 0.0225, f64::from(z) * 0.0225);
    if d0 < -0.1 {
        5_011_004
    } else {
        6_975_545
    }
}

/// `BiomeGenMutated.getGrassColorAtPos`/`getFoliageColorAtPos` are
/// `return this.baseBiome.<same>(pos);` — the *base* biome's override runs, with the base
/// biome's own temperature and rainfall. `createMutation` registers the mutation at
/// `biomeID + 128` (`BiomeGenBase.java:548-556`), so the base is recoverable from the id.
///
/// Not every mutation is a `BiomeGenMutated`: Sunflower Plains, Flower Forest, Ice Plains
/// Spikes, the two Mega Spruce taigas, the two Extreme Hills variants and the three Mesa
/// variants each build a fresh instance of the *ordinary* biome class in an overridden
/// `createMutatedBiome`, and those do **not** delegate. Only these four classes do:
/// `BiomeGenMutated` itself, `BiomeGenSavanna$Mutated`, and the two anonymous
/// `BiomeGenForest$1`/`$2` subclasses (`BlockGenForest.createMutatedBiome`).
fn delegating_base(biome_id: u8) -> Option<&'static mc_data::Biome> {
    let class = mc_data::biome(biome_id).java_class;
    let short = class.rsplit('.').next().unwrap_or(class);
    if !matches!(
        short,
        "BiomeGenMutated" | "BiomeGenSavanna$Mutated" | "BiomeGenForest$1" | "BiomeGenForest$2"
    ) {
        return None;
    }
    let base = mc_data::biome(biome_id.checked_sub(128)?);
    base.registered.then_some(base)
}

/// `BiomeGenBase.getGrassColorAtPos` for one column, including the biomes that override it.
#[must_use]
pub fn grass_color(biome_id: u8, pos: BlockPos) -> u32 {
    let b = delegating_base(biome_id).unwrap_or_else(|| mc_data::biome(biome_id));
    match b.java_class {
        // java: BiomeGenSwamp.getGrassColorAtPos — GRASS_COLOR_NOISE picks 5011004 below -0.1.
        c if c.ends_with("BiomeGenSwamp") => swamp_grass_color(pos.x, pos.z),
        // java: BiomeGenMesa.getGrassColorAtPos
        c if c.contains("BiomeGenMesa") => 9_470_285,
        _ => {
            let noise = temperature_noise_at(pos.x, pos.z);
            let base = colormap_grass(b.temperature_at(pos.y, noise), b.rainfall);
            // java: BiomeGenForest.getGrassColorAtPos, field_150632_aF == 3. Of the registered
            // biomes only id 29 has that type; Roofed Forest M delegates to this same base id.
            if b.id == 29 {
                ((base & 16_711_422) + 2_634_762) >> 1
            } else {
                base
            }
        }
    }
}

/// `BiomeGenBase.getFoliageColorAtPos` for one column.
#[must_use]
pub fn foliage_color(biome_id: u8, pos: BlockPos) -> u32 {
    let b = delegating_base(biome_id).unwrap_or_else(|| mc_data::biome(biome_id));
    match b.java_class {
        c if c.ends_with("BiomeGenSwamp") => 6_975_545,
        c if c.contains("BiomeGenMesa") => 10_387_789,
        _ => colormap_foliage(
            b.temperature_at(pos.y, temperature_noise_at(pos.x, pos.z)),
            b.rainfall,
        ),
    }
}

fn colormap_grass(temperature: f32, rainfall: f32) -> u32 {
    match colormaps() {
        Some(m) => m.grass.lookup(
            f64::from(temperature.clamp(0.0, 1.0)),
            f64::from(rainfall.clamp(0.0, 1.0)),
        ),
        // java: ColorizerGrass.grassBuffer starts as `new int[65536]`, so a failed first
        // resource load leaves a zero-filled (black) map. Its reload listener swallows the
        // IOException; a later failed reload leaves the previously installed map intact.
        None => 0,
    }
}

fn colormap_foliage(temperature: f32, rainfall: f32) -> u32 {
    match colormaps() {
        Some(m) => m.foliage.lookup(
            f64::from(temperature.clamp(0.0, 1.0)),
            f64::from(rainfall.clamp(0.0, 1.0)),
        ),
        // java: ColorizerFoliage.foliageBuffer has the same zero-filled initial state.
        None => 0,
    }
}

/// Which biome colour source a block wants. java: the `colorMultiplier` overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// `BiomeColorHelper.getGrassColorAtPos` — averaged over 3×3.
    Grass,
    /// `BiomeColorHelper.getFoliageColorAtPos` — averaged over 3×3.
    Foliage,
    /// `biome.getGrassColorAtPos(pos)` for this column only, no averaging.
    GrassHere,
    /// `biome.getFoliageColorAtPos(pos)` for this column only, no averaging.
    FoliageHere,
    /// `BiomeColorHelper.getWaterColorAtPos` — averaged over 3×3.
    Water,
    /// A fixed colour.
    Constant(u32),
}

/// The 3×3 column average vanilla applies to grass, foliage and water.
/// java: `BiomeColorHelper.getColorAtPos`.
fn averaged(world: &World, pos: BlockPos, f: impl Fn(u8, BlockPos) -> u32) -> u32 {
    let mut r = 0u32;
    let mut g = 0u32;
    let mut b = 0u32;
    for dx in -1..=1 {
        for dz in -1..=1 {
            // java: the resolver is handed the *sampled* column's `BlockPos`, not the centre —
            // `BiomeColorHelper.getColorAtPos` passes `blockpos$mutableblockpos` through.
            let p = pos.add(dx, 0, dz);
            let c = f(biome_at(world, p), p);
            r += (c & 0x00FF_0000) >> 16;
            g += (c & 0x0000_FF00) >> 8;
            b += c & 0xFF;
        }
    }
    ((r / 9) & 255) << 16 | ((g / 9) & 255) << 8 | ((b / 9) & 255)
}

/// The mutable/provider-aware form of [`averaged`]. `BiomeColorHelper` obtains each of these
/// nine values through `World.getBiomeGenForCoords`, so an absent neighbour and a position above
/// the build ceiling must reach the caller's generated-biome source instead of quietly becoming
/// Plains.
fn averaged_with_generated(
    world: &mut World,
    pos: BlockPos,
    generated: &impl Fn(BlockPos) -> Option<u8>,
    f: impl Fn(u8, BlockPos) -> u32,
) -> u32 {
    let mut r = 0u32;
    let mut g = 0u32;
    let mut b = 0u32;
    for dx in -1..=1 {
        for dz in -1..=1 {
            let p = pos.add(dx, 0, dz);
            let c = biome_at_with_generated_ref(world, p, generated);
            let c = f(c, p);
            r += (c & 0x00FF_0000) >> 16;
            g += (c & 0x0000_FF00) >> 8;
            b += c & 0xFF;
        }
    }
    ((r / 9) & 255) << 16 | ((g / 9) & 255) << 8 | ((b / 9) & 255)
}

/// The biome id at a terrain sample. Unregistered resident ids fall back to Plains (`1`), matching
/// `Chunk.getBiome`'s null substitution; absent columns and invalid Y are delegated to the
/// dimension-selected source installed on [`World`].
#[must_use]
pub fn biome_at(world: &World, pos: BlockPos) -> u8 {
    world.biome_at(pos)
}

/// Resolves a terrain biome through the same generated-biome boundary as
/// [`mc_world::World::biome_at_with_generated`].
///
/// The ordinary terrain mesher deliberately keeps its `&World`/parallel-read-only contract, so
/// it cannot cache a `0xFF` biome or ask a seed-dependent `WorldChunkManager`. Render callers
/// which do have an authoritative source should use this function (or
/// [`color_multiplier_with_generated`]): valid resident columns cache the source result, while
/// unloaded columns and Y outside `0..=255` ask the source directly. `None` is the manager's
/// explicit Plains default, not the normal generated answer.
#[must_use]
pub fn biome_at_with_generated(
    world: &mut World,
    pos: BlockPos,
    generated: impl Fn(BlockPos) -> Option<u8>,
) -> u8 {
    biome_at_with_generated_ref(world, pos, &generated)
}

#[inline]
fn biome_at_with_generated_ref(
    world: &mut World,
    pos: BlockPos,
    generated: &impl Fn(BlockPos) -> Option<u8>,
) -> u8 {
    world.biome_at_with_generated(pos, generated)
}

/// `Block.colorMultiplier(world, pos, tint_index)` for the block being rendered.
///
/// Returns a packed `0x00RRGGBB` multiplier; [`NO_TINT`] means "leave the texture alone".
/// `tint_index` comes from [`mc_assets::model::BakedQuad::tint_index`] — in 1.8.9 every
/// vanilla block ignores its value and only the *presence* of a tint index matters, which is
/// why this takes the state rather than the index.
#[must_use]
pub fn color_multiplier(world: &World, state: BlockState, pos: BlockPos) -> u32 {
    let Some(source) = tint_source(world, state, pos) else {
        return NO_TINT;
    };
    match source {
        Source::Grass => averaged(world, pos, grass_color),
        Source::Foliage => averaged(world, pos, foliage_color),
        Source::GrassHere => grass_color(biome_at(world, pos), pos),
        Source::FoliageHere => foliage_color(biome_at(world, pos), pos),
        Source::Water => averaged(world, pos, |b, _| mc_data::biome(b).water_color),
        Source::Constant(c) => c,
    }
}

/// [`color_multiplier`] with a live generated-biome source for the `World.getBiomeGenForCoords`
/// fallback branch.
///
/// The existing read-only mesher cannot call this without changing its parallel driver contract.
/// This explicit mutable seam nevertheless gives every terrain tint source (including all nine
/// `BiomeColorHelper` samples) the exact loaded/unloaded/invalid-Y boundary now, without
/// pretending an unloaded column is Plains.
#[must_use]
pub fn color_multiplier_with_generated(
    world: &mut World,
    state: BlockState,
    pos: BlockPos,
    generated: impl Fn(BlockPos) -> Option<u8>,
) -> u32 {
    let Some(source) = tint_source(world, state, pos) else {
        return NO_TINT;
    };
    match source {
        Source::Grass => averaged_with_generated(world, pos, &generated, grass_color),
        Source::Foliage => averaged_with_generated(world, pos, &generated, foliage_color),
        Source::GrassHere => grass_color(biome_at_with_generated_ref(world, pos, &generated), pos),
        Source::FoliageHere => {
            foliage_color(biome_at_with_generated_ref(world, pos, &generated), pos)
        }
        Source::Water => {
            averaged_with_generated(world, pos, &generated, |b, _| mc_data::biome(b).water_color)
        }
        Source::Constant(c) => c,
    }
}

/// `Block.getRenderColor(IBlockState)` — the **position-independent** sibling of
/// [`color_multiplier`], and a different override set.
///
/// Vanilla uses this wherever there is no world position to sample a biome from: item icons in
/// the inventory, and `EntityDiggingFX.func_174845_l()`, which is what every packet-driven
/// `BLOCK_CRACK`/`BLOCK_DUST` particle goes through. Where `colorMultiplier` reads the live
/// biome colormap, `getRenderColor` returns the flat "basic" constants
/// (`ColorizerFoliage.getFoliageColorBasic()` and friends) — so a birch leaf particle is the
/// same colour in a jungle as in a taiga, which is vanilla's behaviour, not an approximation.
///
/// Returns [`NO_TINT`] for the blocks that do not override it. java: `Block.getRenderColor`
/// (`block/Block.java:938`) and its seven overrides.
#[must_use]
pub fn render_color(state: BlockState) -> u32 {
    let class = mc_data::block_info(state.block()).java_class;
    let short = class.rsplit('.').next().unwrap_or(class);
    match short {
        // java: BlockLeaves.getRenderColor -> ColorizerFoliage.getFoliageColorBasic().
        "BlockLeaves" | "BlockNewLeaf" => FOLIAGE_BASIC,
        // java: BlockOldLeaf.getRenderColor — spruce and birch have their own constants, the
        // other two `BlockPlanks.EnumType` variants fall through to `super`.
        "BlockOldLeaf" => match state.meta() & 3 {
            1 => FOLIAGE_PINE,
            2 => FOLIAGE_BIRCH,
            _ => FOLIAGE_BASIC,
        },
        // java: BlockVine.getRenderColor -> getFoliageColorBasic().
        "BlockVine" => FOLIAGE_BASIC,
        // java: BlockLilyPad.getRenderColor — 7455580, *not* colorMultiplier's 2129968.
        "BlockLilyPad" => 7_455_580,
        // java: BlockStem.getRenderColor, the same green-to-orange ramp colorMultiplier uses.
        "BlockStem" => stem_color(state.meta() & 7),
        // java: BlockTallGrass.getRenderColor — DEAD_BUSH (meta 0) is white, the other two
        // variants take the colormap at the fixed point (0.5, 1.0).
        "BlockTallGrass" => {
            if state.meta() & 3 == 0 {
                NO_TINT
            } else {
                colormap_grass_at(0.5, 1.0)
            }
        }
        // java: BlockGrass.getRenderColor -> getBlockColor() -> getGrassColor(0.5, 1.0). Listed
        // for completeness: both `EntityDiggingFX` colour paths early-return for `Blocks.grass`,
        // so this value can never actually reach a particle.
        "BlockGrass" => colormap_grass_at(0.5, 1.0),
        _ => NO_TINT,
    }
}

/// java: `ColorizerFoliage.getFoliageColorBasic()`
const FOLIAGE_BASIC: u32 = 4_764_952;
/// java: `ColorizerFoliage.getFoliageColorPine()`
const FOLIAGE_PINE: u32 = 6_396_257;
/// java: `ColorizerFoliage.getFoliageColorBirch()`
const FOLIAGE_BIRCH: u32 = 8_431_445;

/// `ColorizerGrass.getGrassColor(t, r)` at a literal temperature/rainfall pair, with no biome in
/// the way — what `getRenderColor`'s two grass overrides call.
fn colormap_grass_at(temperature: f64, rainfall: f64) -> u32 {
    match colormaps() {
        Some(m) => m.grass.lookup(temperature, rainfall),
        // java: the static grass buffer is zero-filled before its first successful reload.
        None => 0,
    }
}

/// `Block.getRenderColor`/`colorMultiplier` packed `0x00RRGGBB` folded onto vanilla's flat `0.6`
/// digging-particle grey, as `EntityDiggingFX.setBlockPos`/`func_174845_l` do it:
/// `particleRed *= (i >> 16 & 255) / 255.0F`, and so on.
///
/// `Blocks.grass` is exempt in both vanilla methods — they `return this` before touching the
/// colour — so callers must pass [`NO_TINT`] for it rather than its (non-white) render colour.
#[must_use]
pub fn digging_tint(color: u32) -> [f32; 3] {
    let base = crate::particle::spawn::DEFAULT_DIGGING_TINT;
    [
        base[0] * ((color >> 16) & 255) as f32 / 255.0,
        base[1] * ((color >> 8) & 255) as f32 / 255.0,
        base[2] * (color & 255) as f32 / 255.0,
    ]
}

/// The tint an `EntityDiggingFX` built by the `BLOCK_CRACK`/`BLOCK_DUST` factories ends up with.
/// java: `EntityDiggingFX.func_174845_l()` — `0.6` grey times `getRenderColor(state)`, except
/// for `Blocks.grass`, which returns before the multiply.
#[must_use]
pub fn digging_tint_from_render_color(state: BlockState) -> [f32; 3] {
    if is_grass(state) {
        return crate::particle::spawn::DEFAULT_DIGGING_TINT;
    }
    digging_tint(render_color(state))
}

/// The tint `EffectRenderer.addBlockDestroyEffects` produces. java:
/// `EntityDiggingFX.setBlockPos(pos)` — `0.6` grey times `colorMultiplier(world, pos)`, with the
/// same `Blocks.grass` exemption.
#[must_use]
pub fn digging_tint_from_color_multiplier(
    world: &World,
    state: BlockState,
    pos: BlockPos,
) -> [f32; 3] {
    if is_grass(state) {
        return crate::particle::spawn::DEFAULT_DIGGING_TINT;
    }
    digging_tint(color_multiplier(world, state, pos))
}

/// java: `this.sourceState.getBlock() == Blocks.grass` — the one block both `EntityDiggingFX`
/// colour paths exempt. `BlockGrass` backs no other registry entry in 1.8.9.
fn is_grass(state: BlockState) -> bool {
    mc_data::block_info(state.block())
        .java_class
        .ends_with("BlockGrass")
}

/// The `colorMultiplier` override table, keyed on the originating Java class the way
/// `mc_data` records it. Blocks with no override return `None` (white).
fn tint_source(world: &World, state: BlockState, pos: BlockPos) -> Option<Source> {
    let props = block_props(state);
    // java: BlockLiquid.colorMultiplier — lava is untinted, water takes the biome water colour.
    if props.material == Material::Water {
        return Some(Source::Water);
    }
    if props.material == Material::Lava {
        return None;
    }

    let class = mc_data::block_info(state.block()).java_class;
    let short = class.rsplit('.').next().unwrap_or(class);
    match short {
        // java: BlockGrass, BlockLeaves — 3x3 averaged.
        "BlockGrass" => Some(Source::Grass),
        "BlockLeaves" | "BlockNewLeaf" => Some(Source::Foliage),
        // java: BlockOldLeaf.colorMultiplier (`BlockOldLeaf.java:49-69`) — spruce and birch
        // short-circuit to the *same* two flat constants `getRenderColor` uses, before `super`
        // ever reaches `BiomeColorHelper`. Only oak and jungle are biome-tinted. `getWoodType`
        // is `(meta & 3) % 4`, so the variant is `meta & 3`.
        "BlockOldLeaf" => {
            let actual = world.block_at(pos);
            if actual.block() != state.block() {
                // Java reads `worldIn.getBlockState(pos)` and delegates to BlockLeaves when the
                // caller's BlockOldLeaf instance no longer occupies the position.
                Some(Source::Foliage)
            } else {
                match actual.meta() & 3 {
                    1 => Some(Source::Constant(FOLIAGE_PINE)),
                    2 => Some(Source::Constant(FOLIAGE_BIRCH)),
                    _ => Some(Source::Foliage),
                }
            }
        }
        // java: BlockDoublePlant.colorMultiplier — only the GRASS/FERN variants tint. An upper
        // half gets its variant from the stored state below when that is the same block.
        "BlockDoublePlant" => {
            let actual = world.block_at(pos);
            let source = if actual.block() != state.block() {
                // `getVariant` returns FERN for a position that is no longer this block. That
                // surprising vanilla fallback still selects the grass tint.
                None
            } else if stored_property(actual, "half") == Some("upper") {
                let below = world.block_at(pos.down());
                if below.block() == actual.block() {
                    Some(below)
                } else {
                    Some(actual)
                }
            } else {
                Some(actual)
            };
            if source.is_none() {
                return Some(Source::Grass);
            }
            let source = source.expect("checked above");
            // Read the generated storage property, not `meta & 7` and not `getStateFromMeta`'s
            // default. `BLOCK_STATE_IDS` freezes the last state registered for each packed id,
            // so every stored upper half carries `variant=paeonia`. Java copies that exact
            // property from `pos.down()` without actualising the neighbour first. Thus an orphan
            // and an upper-on-upper pair are both untinted.
            if matches!(
                stored_property(source, "variant"),
                Some("double_grass" | "double_fern")
            ) {
                Some(Source::Grass)
            } else {
                None
            }
        }
        // java: BlockFlowerPot.colorMultiplier delegates to its contained ItemBlock. Of the
        // 22 flower-pot models in 1.8.9 only `flower_pot_fern` has tinted quads, and that payload
        // is Blocks.tallgrass meta 2, whose multiplier is the unaveraged column grass colour.
        "BlockFlowerPot" => world
            .tile_entity(pos)
            .and_then(mc_world::TileEntity::flower_pot)
            .filter(|pot| pot.contents() == "fern")
            .map(|_| Source::GrassHere),
        // java: BlockTallGrass, BlockReed — the *unaveraged* single-column form.
        "BlockTallGrass" | "BlockReed" => Some(Source::GrassHere),
        // java: BlockVine.colorMultiplier — also unaveraged.
        "BlockVine" => Some(Source::FoliageHere),
        // java: BlockLilyPad.colorMultiplier
        "BlockLilyPad" => Some(Source::Constant(2_129_968)),
        // java: BlockStem.colorMultiplier -> getRenderColor: green fading to orange with age.
        "BlockStem" => {
            let actual = world.block_at(pos);
            (actual.block() == state.block())
                .then(|| Source::Constant(stem_color(actual.meta() & 7)))
        }
        // java: BlockRedstoneWire.colorMultiplier(power)
        "BlockRedstoneWire" => {
            let actual = world.block_at(pos);
            (actual.block() == state.block())
                .then(|| Source::Constant(redstone_color(actual.meta() & 15)))
        }
        _ => None,
    }
}

fn stored_property(state: BlockState, name: &str) -> Option<&'static str> {
    mc_data::state_properties(state)
        .iter()
        .find_map(|(property, value)| (*property == name).then_some(*value))
}

/// java: `BlockStem.getRenderColor` — `i * 32` red, `255 - i * 8` green, `i * 4` blue.
fn stem_color(age: u8) -> u32 {
    let i = u32::from(age);
    let r = i * 32;
    let g = 255 - i * 8;
    let b = i * 4;
    (r << 16) | (g << 8) | b
}

/// java: `BlockRedstoneWire.colorMultiplier(int)`.
fn redstone_color(power: u8) -> u32 {
    let f = f32::from(power) / 15.0;
    let f1 = if power == 0 { 0.3 } else { f * 0.6 + 0.4 };
    let f2 = (f * f * 0.7 - 0.5).max(0.0);
    let f3 = (f * f * 0.6 - 0.7).max(0.0);
    // java: MathHelper.clamp_int((int)(f * 255.0F), 0, 255) — truncate, then clamp.
    let to_byte = |v: f32| ((v * 255.0) as i32).clamp(0, 255) as u32;
    (to_byte(f1) << 16) | (to_byte(f2) << 8) | to_byte(f3)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both noise generators against the **real 1.8.9 sources**: `NoiseGenerator.java`,
    /// `NoiseGeneratorSimplex.java` and `NoiseGeneratorPerlin.java` compiled straight out of
    /// `MCP-919-main/src/minecraft/net/minecraft/world/gen/` and run under a JDK, printing raw
    /// IEEE bits so the comparison is exact rather than epsilon'd.
    ///
    /// Sample: the full 16x16 cross product of
    /// `x, z in {0, +-1, +-3, +-5, 7, 8, +-8, 16, 17, +-100, 1234, -1234, 4321, -4321,
    /// 63, 64, +-30_000_000}` — both signs, both sides of the `func_151607_a` `> 0.0` floor
    /// quirk, both sides of the `/8.0` and `*0.0225` scales, and the world-border extremes.
    ///
    /// Negative control: `every_row_is_bit_exact` also counts how many rows are non-zero. If
    /// the generator degenerated to a constant (a wrong permutation shuffle, a dropped
    /// `nextDouble`), the table would still "match" on the zero rows, so the count must stay
    /// at its measured value.
    #[test]
    fn simplex_noise_matches_the_vanilla_generators_bit_for_bit() {
        // (x, z, bits of the temperature term `noise(x/8, z/8) * 4` as f32,
        //         bits of `GRASS_COLOR_NOISE(x*0.0225, z*0.0225)` as f64)
        const ORACLE: &[(i32, i32, u32, u64)] = &[
            (0, 0, 0x00000000u32, 0x0000000000000000u64),
            (0, 1, 0xbff69b05u32, 0x0000000000000000u64),
            (0, -1, 0x3ff69b9eu32, 0x0000000000000000u64),
            (0, 8, 0xbffd7cfdu32, 0xbf62193b5af9cb13u64),
            (0, -8, 0xbffd7cfdu32, 0xbf621889ebe1a200u64),
            (0, 17, 0xbfe45e24u32, 0xbfca593d561965b0u64),
            (0, 100, 0x3fdac65du32, 0x3fd454b5b464d416u64),
            (0, -100, 0x3fda37f5u32, 0xbfd4549d0dda8fe9u64),
            (0, 4321, 0xc0189fb7u32, 0x3fc00560271a4fdeu64),
            (0, -1234, 0x3f9bde5au32, 0x3fe9f39989713a25u64),
            (0, -30000000, 0x3fe177d8u32, 0x3fd533aba2a099a4u64),
            (0, 30000000, 0x3fecdab5u32, 0xbec0ee1b216bdd65u64),
            (0, 5, 0x3fc3e7bfu32, 0x0000000000000000u64),
            (0, -5, 0x3fbb33ebu32, 0x0000000000000000u64),
            (0, 64, 0x3fb4a6c4u32, 0xbfcecf0a93d3ad5cu64),
            (0, 63, 0x4036ac71u32, 0xbfcfda01f2c697cau64),
            (1, 0, 0xbff69b39u32, 0x3fb9191cbf4f3671u64),
            (1, 1, 0xc058d7e5u32, 0x3fb8ff1a91f396dcu64),
            (1, -1, 0x3aca8bdau32, 0x3fb8ff1a91f396dau64),
            (1, 8, 0xbfb2bf1cu32, 0x3fb2d73a5ce3ac86u64),
            (1, -8, 0xc01a277eu32, 0x3fb258a0b3628b25u64),
            (1, 17, 0x00000000u32, 0xbfc45481dc791a81u64),
            (1, 100, 0x3fc499a9u32, 0x3fd5748cb0e593aau64),
            (1, -100, 0x3f78383eu32, 0xbfd31e95f26945f8u64),
            (1, 4321, 0xbf994257u32, 0x3fc2657f5a697a79u64),
            (1, -1234, 0xbeac7057u32, 0x3fe8a493035e543au64),
            (1, -30000000, 0x400c4895u32, 0x3fd5ca00f98b450du64),
            (1, 30000000, 0x3fb5351fu32, 0xbfb5a01c1d0ff0a4u64),
            (1, 5, 0x3f5a52f4u32, 0x3fb6a63eca9a35d5u64),
            (1, -5, 0x400ab813u32, 0x3fb6a63cbc866ebfu64),
            (1, 64, 0xbf2fce4fu32, 0xbfc424e4c8dc4fe2u64),
            (1, 63, 0x3f8ae4eeu32, 0xbfc600af90934537u64),
            (-1, 0, 0x3ff69b9eu32, 0xbfb9191cbf4f3666u64),
            (-1, 1, 0x3b333bb7u32, 0xbfb8ff1a91f396d1u64),
            (-1, -1, 0x4057317fu32, 0xbfb8ff1a91f396d1u64),
            (-1, 8, 0xc01a5a39u32, 0xbfb40a0cc619cda0u64),
            (-1, -8, 0xbfb5c21eu32, 0xbfb3866e3a77044du64),
            (-1, 17, 0xc004ba2fu32, 0xbfd0476ec6a89ba5u64),
            (-1, 100, 0x4017b5cau32, 0x3fd31ebf2e6530f3u64),
            (-1, -100, 0x400765e2u32, 0xbfd574827c51a85eu64),
            (-1, 4321, 0xc008a88du32, 0x3fbb593985c95e6fu64),
            (-1, -1234, 0x400aeacfu32, 0x3feae71f62a8054eu64),
            (-1, -30000000, 0x3f5ea900u32, 0x3fd47421f55d1faau64),
            (-1, 30000000, 0x400c4511u32, 0x3fb704e7e03f7732u64),
            (-1, 5, 0x400955b3u32, 0xbfb6a640d8adfcddu64),
            (-1, -5, 0xbd4b1f97u32, 0xbfb6a63eca9a35c8u64),
            (-1, 64, 0x40369e6eu32, 0xbfd4f2336c9d9fa6u64),
            (-1, 63, 0x406911dau32, 0xbfd51b514eb4e3a9u64),
            (7, 0, 0x4015de6eu32, 0x3fe1f96bd09c2677u64),
            (7, 1, 0x3fbf976du32, 0x3fe1e7dc110e5aa8u64),
            (7, -1, 0x3fab5226u32, 0x3fe1e2c8fef48c21u64),
            (7, 8, 0xc02b92f8u32, 0x3fd9c738fdf25301u64),
            (7, -8, 0xb642850cu32, 0x3fda223e6273b798u64),
            (7, 17, 0xc0130edau32, 0xbf7ac4d0ec30ef46u64),
            (7, 100, 0xbb1efff4u32, 0x3fd8cbb3d65e6951u64),
            (7, -100, 0xbf149365u32, 0xbfc92d54aac985a5u64),
            (7, 4321, 0xbe9b8f00u32, 0x3fce1a8845da2db3u64),
            (7, -1234, 0xc066e3a1u32, 0x3fd440ec0cb6fc5du64),
            (7, -30000000, 0xbf7742a7u32, 0x3fd58ef7bac391dfu64),
            (7, 30000000, 0x4018ea34u32, 0xbfda37a1503be7e5u64),
            (7, 5, 0x3dc7ef7du32, 0x3fe00214b60b1468u64),
            (7, -5, 0xc02cf29au32, 0x3fe00b1f12d38429u64),
            (7, 64, 0xc0137e2cu32, 0x3fc8a00af2a5c222u64),
            (7, 63, 0xbf42a1d1u32, 0x3fc12cdf3a52646eu64),
            (8, 0, 0x3fb169ceu32, 0x3fe334933f419198u64),
            (8, 1, 0x3eb18615u32, 0x3fe325f35c042a4eu64),
            (8, -1, 0x3f71d45au32, 0x3fe3163ffb92b840u64),
            (8, 8, 0xc0648f8bu32, 0x3fdae070c0035117u64),
            (8, -8, 0x00000000u32, 0x3fdb704b6fea7762u64),
            (8, 17, 0xc0163826u32, 0xbf8bc721dac817dbu64),
            (8, 100, 0xbf8fcdddu32, 0x3fd8ab10d09c42a9u64),
            (8, -100, 0xbf9aa977u32, 0xbfc8075e62688d4du64),
            (8, 4321, 0xc00e144au32, 0x3fcf7380b49ae466u64),
            (8, -1234, 0xc02960b8u32, 0x3fcc7babcfd30b96u64),
            (8, -30000000, 0xbf7920adu32, 0x3fd4e6a84616d0d7u64),
            (8, 30000000, 0x4012cf31u32, 0xbfdb9d5d9f5e6d54u64),
            (8, 5, 0xbf75ceedu32, 0x3fe10258f2d28d08u64),
            (8, -5, 0xc01f8da8u32, 0x3fe0fcc3dcff7bb8u64),
            (8, 64, 0xbfe8f934u32, 0x3fcc5fb51546cd30u64),
            (8, 63, 0xbf1bddaeu32, 0x3fc419d96cbbc70cu64),
            (16, 0, 0xbd493176u32, 0x3fd3625a57eb7ab0u64),
            (16, 1, 0xbfe45d52u32, 0x3fd41b5d4847e2d0u64),
            (16, -1, 0x3f207922u32, 0x3fd216a37ab9ca37u64),
            (16, 8, 0xbf70e94cu32, 0x3fb8ee2b1b146bb2u64),
            (16, -8, 0xc03169ceu32, 0x3f9b7910e3bf8d50u64),
            (16, 17, 0xbfce3b4fu32, 0xbfd5c5f77363d518u64),
            (16, 100, 0x4009cafbu32, 0x3fd27505c55b753au64),
            (16, -100, 0xc02105e7u32, 0xbfd00687ccf5cc0bu64),
            (16, 4321, 0x3ed4162fu32, 0x3fd07dc30c6ce789u64),
            (16, -1234, 0x3f078037u32, 0xbfd593605f60f3e0u64),
            (16, -30000000, 0x3fff7400u32, 0x3fb3609b0095363bu64),
            (16, 30000000, 0xbd1b7e0du32, 0xbfdd6e474a1de514u64),
            (16, 5, 0x3fb877acu32, 0x3fd00c5ac30033b4u64),
            (16, -5, 0x4006f3d2u32, 0x3fc2f2712019c3a9u64),
            (16, 64, 0x3fc6eb83u32, 0x3fd3d0a8ecdbc751u64),
            (16, 63, 0x3f7072ddu32, 0x3fcc361b360f2ce7u64),
            (100, 0, 0xbf9493f7u32, 0xbfe07e43e1390f53u64),
            (100, 1, 0xc05468fdu32, 0xbfddc2d018763993u64),
            (100, -1, 0x3fc0bd87u32, 0xbfe1bd63586eed67u64),
            (100, 8, 0xbf7c2b44u32, 0x3fba31ecd07d07fau64),
            (100, -8, 0x3f237e38u32, 0xbfe10f2b13008937u64),
            (100, 17, 0x3f71ba5fu32, 0x3fe1fcc7d3f4b604u64),
            (100, 100, 0x3fde06e7u32, 0x3fcf6d0ee96b8ad7u64),
            (100, -100, 0x3f7f75adu32, 0xbfd19a3219308ac3u64),
            (100, 4321, 0x402085b4u32, 0x3fddc3f2a6c876b4u64),
            (100, -1234, 0xc007997du32, 0xbfc9b2fcb1515595u64),
            (100, -30000000, 0xbfe5b6e7u32, 0x3fe4daa14e108296u64),
            (100, 30000000, 0xbfd09958u32, 0x3fe2b68ff81eb57bu64),
            (100, 5, 0x3f98f7ebu32, 0xbfc5a501144366f7u64),
            (100, -5, 0xc017dfd2u32, 0xbfe3174a4f7bbe78u64),
            (100, 64, 0xbfdd3aaau32, 0x3fe2e0c68e8848fcu64),
            (100, 63, 0xc015af61u32, 0x3fe25b191ac5c0e3u64),
            (-100, 0, 0x3ecad065u32, 0xbfe068382bacd715u64),
            (-100, 1, 0x4002638eu32, 0xbfe17a3d0564b83fu64),
            (-100, -1, 0xbf9ae120u32, 0xbfddba5bda63301bu64),
            (-100, 8, 0xc0558f0cu32, 0xbfd252952e24e486u64),
            (-100, -8, 0xbff690fdu32, 0x3fba31ecd07d07edu64),
            (-100, 17, 0x3f9e9bafu32, 0x3fe00fde4ea6a1a6u64),
            (-100, 100, 0xbf7f75adu32, 0xbfd19a3219308ac4u64),
            (-100, -100, 0x3fbfbdc2u32, 0xbfcf6d0ee96b8ad7u64),
            (-100, 4321, 0x3dd9de44u32, 0x3fd8cbe478d11d82u64),
            (-100, -1234, 0x3f94f1d8u32, 0xbfda552da043c760u64),
            (-100, -30000000, 0x400dcd7au32, 0x3fd67629ddafaadbu64),
            (-100, 30000000, 0xbfc799c1u32, 0xbfe416aef27890cfu64),
            (-100, 5, 0xc0184d3eu32, 0xbfe00fecec8e17bdu64),
            (-100, -5, 0xbfabffd9u32, 0xbfc5a50114436702u64),
            (-100, 64, 0x3eb2b74eu32, 0x3fe1ea1d260700b4u64),
            (-100, 63, 0x3fea9ac1u32, 0x3fe35621b6d71e00u64),
            (1234, 0, 0x3e7b009du32, 0x3fc726da47ee2fe8u64),
            (1234, 1, 0x3f8090eeu32, 0x3fcf8c86e47eb884u64),
            (1234, -1, 0x3d1bf5fdu32, 0x3fbb912ca4149537u64),
            (1234, 8, 0xc002f3b8u32, 0x3fe12389689f8c1eu64),
            (1234, -8, 0x402b8c61u32, 0xbfe08c39ef6c92a6u64),
            (1234, 17, 0xbf9c24d1u32, 0x3fe3d08fad74f7b2u64),
            (1234, 100, 0xc012fd89u32, 0x3fd743dff61397aau64),
            (1234, -100, 0x3da53df6u32, 0x3fddb0b1b70b5f32u64),
            (1234, 4321, 0xc0044f09u32, 0x3fd30104355656d5u64),
            (1234, -1234, 0xc02e938au32, 0x3fadd0b98ee7cbb5u64),
            (1234, -30000000, 0xbfe89974u32, 0xbfc94aa45f6117cau64),
            (1234, 30000000, 0x401e5b83u32, 0x3fe3456a683682c0u64),
            (1234, 5, 0x3f329da1u32, 0x3fdc4869261e5342u64),
            (1234, -5, 0x40368331u32, 0xbfcf5374854a0422u64),
            (1234, 64, 0x400350adu32, 0x3fe4982b994397c1u64),
            (1234, 63, 0x3f4574d6u32, 0x3fe5451b0778b5d1u64),
            (-4321, 0, 0xbf65906du32, 0xbfc6ad4b6188c76eu64),
            (-4321, 1, 0xbed8bd2fu32, 0xbfc73ac637ccf8f3u64),
            (-4321, -1, 0xbf994257u32, 0xbfc6b6cb19ba4100u64),
            (-4321, 8, 0xbfc63b9eu32, 0xbfd5b3a9e16ea2d5u64),
            (-4321, -8, 0xc0031efeu32, 0xbfcf7395d7774f02u64),
            (-4321, 17, 0x404fb93fu32, 0xbfe3f2ad9911a0f4u64),
            (-4321, 100, 0xbf50eb07u32, 0x3fde8b29a96c3a8fu64),
            (-4321, -100, 0xc024d8a9u32, 0xbfd535710b0b6e36u64),
            (-4321, 4321, 0xbab33bb7u32, 0x3fdf0435af959ff6u64),
            (-4321, -1234, 0x3fc19b66u32, 0xbfbf72d8410f19d5u64),
            (-4321, -30000000, 0xbf46a0bdu32, 0xbfb0878a514d9390u64),
            (-4321, 30000000, 0x3fb8ff8cu32, 0xbfd30217bcb80a4fu64),
            (-4321, 5, 0x40095fa4u32, 0xbfcfa8b4716f1b84u64),
            (-4321, -5, 0xbf450c06u32, 0xbfcb1334be353e51u64),
            (-4321, 64, 0x402d44d8u32, 0x3fb180150acde944u64),
            (-4321, 63, 0x3f934b1bu32, 0x3fae0e0307f7707cu64),
            (30000000, 0, 0xc0672947u32, 0x3ddf1bffc93975e3u64),
            (30000000, 1, 0xc03d5250u32, 0xbfb59b96cdd32c7cu64),
            (30000000, -1, 0xc0259774u32, 0x3fb704e7e03f7732u64),
            (30000000, 8, 0xbffdb32du32, 0xbfd99fe8f6dbe5ebu64),
            (30000000, -8, 0x00000000u32, 0x3fe614b98148ac90u64),
            (30000000, 17, 0xbff7b1d8u32, 0xbfcf0de3bfb176dau64),
            (30000000, 100, 0xbfc909d4u32, 0xbfcc556ffeafc50eu64),
            (30000000, -100, 0xbfd476cbu32, 0x3fe3ceafc0223deeu64),
            (30000000, 4321, 0xc002f754u32, 0xbfe3bebd644ebe4fu64),
            (30000000, -1234, 0xbfe52697u32, 0x3fc7f510f20e5ae8u64),
            (30000000, -30000000, 0x00000000u32, 0x3df1481960000000u64),
            (30000000, 30000000, 0xbfc090bcu32, 0x3f97a83105c5538bu64),
            (30000000, 5, 0x3f8edb21u32, 0xbfd548dac9eb164cu64),
            (30000000, -5, 0x403ddec1u32, 0x3fdd87f59267c184u64),
            (30000000, 64, 0x3fcb843cu32, 0x3fba9388bbed55d2u64),
            (30000000, 63, 0xbe615fcdu32, 0x3fc5b9d6609bdd12u64),
            (-30000000, 0, 0x406177d8u32, 0xbfd533aba2a0999fu64),
            (-30000000, 1, 0x40259aebu32, 0xbfd5ca00f98b450du64),
            (-30000000, -1, 0x401e9b4eu32, 0xbfd4743b41969da2u64),
            (-30000000, 8, 0x31846a85u32, 0xbfd4e6a84616d0d7u64),
            (-30000000, -8, 0xbfd5c7f9u32, 0xbfc8a16dcfd88fefu64),
            (-30000000, 17, 0x4003b419u32, 0xbf839508c6c83df8u64),
            (-30000000, 100, 0xbf847133u32, 0x3fe44b7296fcd5f8u64),
            (-30000000, -100, 0xbfd92b51u32, 0xbfe998dc72fc4eefu64),
            (-30000000, 4321, 0x3f8a952cu32, 0xbfd3633e16f8d3b6u64),
            (-30000000, -1234, 0x401d3cb7u32, 0x3fe4c1a026a307f6u64),
            (-30000000, -30000000, 0xbfff74a8u32, 0xbfdd1e17e838cb57u64),
            (-30000000, 30000000, 0xb08dbf35u32, 0xbdf1481960000000u64),
            (-30000000, 5, 0xc03ddc87u32, 0xbfd65cc783fcb0c4u64),
            (-30000000, -5, 0xc013a09cu32, 0xbfd03aafaef16d59u64),
            (-30000000, 64, 0x00000000u32, 0xbfddc9088261ec21u64),
            (-30000000, 63, 0xbf59a005u32, 0xbfde23aeb4d6320eu64),
            (3, 0, 0xbfb3ed0eu32, 0x3fd2386fb302db47u64),
            (3, 1, 0xc0120442u32, 0x3fd225669cecb19au64),
            (3, -1, 0xbf381d3au32, 0x3fd225669cecb19au64),
            (3, 8, 0x3f4486f2u32, 0x3fcb9a21c8bf42d7u64),
            (3, -8, 0xbff05a41u32, 0x3fcb09dbae7ccc1du64),
            (3, 17, 0x403744d4u32, 0xbfb3bdb0d1e237abu64),
            (3, 100, 0x400a8bebu32, 0x3fd74bd7e01353c6u64),
            (3, -100, 0x3f82c66eu32, 0xbfd09e783ada30b9u64),
            (3, 4321, 0x40122038u32, 0x3fc6ea44be6b9bc3u64),
            (3, -1234, 0xc01a3a69u32, 0x3fe4e8261d3a7631u64),
            (3, -30000000, 0x3fb428efu32, 0x3fd670f3af6919a3u64),
            (3, 30000000, 0x3fbfe6beu32, 0xbfcd870e34868d50u64),
            (3, 5, 0x3f17fd55u32, 0x3fd06d98d61220c8u64),
            (3, -5, 0x3fdfce1cu32, 0x3fd06d967f4c2703u64),
            (3, 64, 0xc0042eeau32, 0xbf806d02214fb5fau64),
            (3, 63, 0xbf61b861u32, 0xbfa2d650c0738ce6u64),
            (-3, 0, 0x401cf7fau32, 0xbfd2386fb302db47u64),
            (-3, 1, 0x400db02bu32, 0xbfd225669cecb19au64),
            (-3, -1, 0x3fb801a5u32, 0xbfd225669cecb19au64),
            (-3, 8, 0xbff4ce3cu32, 0xbfcc918c30077047u64),
            (-3, -8, 0x3ed5ac40u32, 0xbfcbae3100b71d9du64),
            (-3, 17, 0x3f5b6434u32, 0xbfd6432fbe3f9ef7u64),
            (-3, 100, 0x3f84215eu32, 0x3fd09eaf67461ffcu64),
            (-3, -100, 0xbeb385f2u32, 0xbfd74bd7cdca5584u64),
            (-3, 4321, 0xbf90ca75u32, 0x3fb3055dcede714bu64),
            (-3, -1234, 0xbe21e4c4u32, 0x3febe8347634f516u64),
            (-3, -30000000, 0x3f3e7b6cu32, 0x3fd285aa76e5c214u64),
            (-3, 30000000, 0x3fa2d7e5u32, 0x3fd1c7f586592a81u64),
            (-3, 5, 0x3ff22a70u32, 0xbfd06df53d32d686u64),
            (-3, -5, 0xbfefbc06u32, 0xbfd06d98d61220c5u64),
            (-3, 64, 0x3ef715bbu32, 0xbfe00b6faff09df5u64),
            (-3, 63, 0x3ee93ed6u32, 0xbfdfb547d9622b9bu64),
            (63, 0, 0xc0124d7bu32, 0x3fc314cf216f79d1u64),
            (63, 1, 0xc00cccddu32, 0x3fc1ee6a00a374f8u64),
            (63, -1, 0xbfded85du32, 0x3fc4bfcce8f23c4fu64),
            (63, 8, 0xc03a6945u32, 0x3fc55e0db6bd6fbcu64),
            (63, -8, 0xbda69550u32, 0x3fd60bba8ec731e6u64),
            (63, 17, 0xbffe4e90u32, 0x3fcc287579158333u64),
            (63, 100, 0xbf9ae6eau32, 0x3fd665ab34fdf496u64),
            (63, -100, 0xc0697680u32, 0x3fe4d5d9b9c1b07du64),
            (63, 4321, 0xbf1f45dfu32, 0x3fe422f49b6992dbu64),
            (63, -1234, 0x4018bb46u32, 0x3fe2841173ef36aau64),
            (63, -30000000, 0x3fb138f7u32, 0x3fcfd875fa154379u64),
            (63, 30000000, 0xbe615fcdu32, 0x3fe4073596648af2u64),
            (63, 5, 0x3ff6569du32, 0x3fc203f30f448ce8u64),
            (63, -5, 0x402d5b25u32, 0x3fd01568a71c379du64),
            (63, 64, 0x40564f50u32, 0xbfe00b41268c58f3u64),
            (63, 63, 0x4059f274u32, 0xbfdfedeb0429634fu64),
            (64, 0, 0xbf34a6c4u32, 0x3fc58307690d8f83u64),
            (64, 1, 0xbf2d1d6bu32, 0x3fc53694a49f068au64),
            (64, -1, 0xbf1173d0u32, 0x3fc651e9191d6fb8u64),
            (64, 8, 0xc06ec79cu32, 0x3fcde5303bd309d2u64),
            (64, -8, 0x3f38c7a9u32, 0x3fd425699032fee0u64),
            (64, 17, 0xc019bbe2u32, 0x3fd3c704ff653cd3u64),
            (64, 100, 0xbfdd3aa1u32, 0x3fdb0d976b7336e3u64),
            (64, -100, 0xc0298374u32, 0x3fe558aa8b2ff0c9u64),
            (64, 4321, 0xbf89ef2bu32, 0x3fe17ef884afe95fu64),
            (64, -1234, 0x400f2e53u32, 0x3fe2f2f8bda28eb6u64),
            (64, -30000000, 0x3ffb0ae3u32, 0x3fcd31488939d24du64),
            (64, 30000000, 0x3fcae9c8u32, 0x3fe309d7beae0cf5u64),
            (64, 5, 0x3eb8427bu32, 0x3fc87bdc00d73b7au64),
            (64, -5, 0x401f6f83u32, 0x3fce6da95382d5b0u64),
            (64, 64, 0x40261c10u32, 0xbfdfeeb5989d3559u64),
            (64, 63, 0x4054e236u32, 0xbfdf955594401316u64),
        ];
        let mut nonzero_temp = 0;
        let mut nonzero_grass = 0;
        for &(x, z, tbits, gbits) in ORACLE {
            let t = temperature_noise_at(x, z);
            assert_eq!(
                t.to_bits(),
                tbits,
                "temperatureNoise at ({x}, {z}): got {t}, want {}",
                f32::from_bits(tbits)
            );
            let g = grass_color_noise().sample(f64::from(x) * 0.0225, f64::from(z) * 0.0225);
            assert_eq!(
                g.to_bits(),
                gbits,
                "GRASS_COLOR_NOISE at ({x}, {z}): got {g}, want {}",
                f64::from_bits(gbits)
            );
            if tbits != 0 {
                nonzero_temp += 1;
            }
            if gbits != 0 {
                nonzero_grass += 1;
            }
        }
        assert_eq!(ORACLE.len(), 256);
        assert_eq!(nonzero_temp, 250, "the oracle itself went flat");
        assert_eq!(nonzero_grass, 251, "the oracle itself went flat");
    }

    #[test]
    fn colormap_index_matches_vanilla_arithmetic() {
        // java: ColorizerGrass.getGrassColor — index = (int)((1-rain*temp)*255) << 8
        //                                            | (int)((1-temp)*255)
        let mut pixels = vec![0u32; 65536];
        // temperature 0.8, rainfall 0.4 (plains): rain*temp = 0.32
        let i = ((1.0 - 0.8f64) * 255.0) as usize; // 51
        let j = ((1.0 - 0.32f64) * 255.0) as usize; // 173
        pixels[(j << 8) | i] = 0x00AB_CDEF;
        let map = Colormap { pixels };
        assert_eq!(map.lookup(0.8, 0.4), 0x00AB_CDEF);
        // Every clamped input lands inside the table; nothing panics and nothing reaches the
        // out-of-range sentinel.
        for t in [0.0f64, 0.25, 0.5, 1.0] {
            for r in [0.0f64, 0.25, 0.5, 1.0] {
                assert_ne!(map.lookup(t, r), 0x00FF_00FF, "t={t} r={r}");
            }
        }
    }

    #[test]
    fn an_unloaded_colormap_has_vanillas_zero_filled_initial_value() {
        assert!(
            !colormaps_installed(),
            "this test exercises the pre-reload static-buffer state"
        );
        assert_eq!(colormap_grass(0.8, 0.4), 0);
        assert_eq!(colormap_foliage(0.8, 0.4), 0);
        assert_eq!(colormap_grass_at(0.5, 1.0), 0);
        // Negative controls for both historical substitutes: magenta was the old terrain
        // fallback and white was the old item-render fallback. Vanilla produces neither.
        assert_ne!(colormap_grass(0.8, 0.4), 0x00FF_00FF);
        assert_ne!(colormap_grass_at(0.5, 1.0), NO_TINT);
    }

    #[test]
    fn the_two_vanilla_colormap_reload_listeners_fail_independently() {
        let solid = |value| Colormap {
            pixels: vec![value; 65_536],
        };
        let previous = Colormaps {
            grass: solid(0x0011_1111),
            foliage: solid(0x0022_2222),
        };
        let grass_only = merge_partial_reload(Some(&previous), Some(solid(0x0033_3333)), None)
            .expect("one successful listener");
        assert_eq!(grass_only.grass.lookup(0.8, 0.4), 0x0033_3333);
        assert_eq!(
            grass_only.foliage.lookup(0.8, 0.4),
            0x0022_2222,
            "the failed foliage listener must retain its previous buffer"
        );

        let foliage_only =
            merge_partial_reload(None, None, Some(solid(0x0044_4444))).expect("foliage loaded");
        assert_eq!(
            foliage_only.grass.lookup(0.8, 0.4),
            0,
            "a first-load failure keeps vanilla's zero-filled grass buffer"
        );
        assert_eq!(foliage_only.foliage.lookup(0.8, 0.4), 0x0044_4444);
        assert!(
            merge_partial_reload(Some(&previous), None, None).is_none(),
            "two failed listeners must not publish a redundant generation"
        );
    }

    fn state_of(name: &str) -> BlockState {
        mc_data::block_by_name(name)
            .unwrap_or_else(|| panic!("{name} is not a registered block"))
            .default_state()
    }

    #[test]
    fn untinted_blocks_report_no_tint_source() {
        let w = World::new();
        assert_eq!(
            tint_source(&w, state_of("minecraft:stone"), BlockPos::ORIGIN),
            None
        );
    }

    #[test]
    fn grass_and_leaves_take_the_averaged_sources() {
        let w = World::new();
        let p = BlockPos::ORIGIN;
        assert_eq!(
            tint_source(&w, state_of("minecraft:grass"), p),
            Some(Source::Grass)
        );
        assert_eq!(
            tint_source(&w, state_of("minecraft:leaves"), p),
            Some(Source::Foliage)
        );
        assert_eq!(
            tint_source(&w, state_of("minecraft:water"), p),
            Some(Source::Water)
        );
        assert_eq!(tint_source(&w, state_of("minecraft:lava"), p), None);
    }

    #[test]
    fn an_unregistered_chunk_biome_falls_back_to_plains_for_tinting() {
        let mut w = World::new();
        w.set_has_sky_light(true);
        let payload = crate::terrain::fixture::blank_column_payload(true, 42);
        w.apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 0xFFFF, &payload)
            .expect("odd-biome fixture");
        let p = BlockPos::new(2, 64, 2);
        let raw = w.chunk(p.chunk_pos()).expect("fixture chunk").biomes()
            [((p.z & 15) << 4 | (p.x & 15)) as usize];
        assert_eq!(
            raw, 42,
            "negative control: the fixture must carry an odd raw id"
        );
        assert_eq!(biome_at(&w, p), 1);
        assert_ne!(raw, biome_at(&w, p), "the fallback path must bite");
        assert_eq!(mc_data::biome(biome_at(&w, p)).name, "Plains");
    }

    #[test]
    fn generated_tint_biome_seam_respects_loaded_unloaded_invalid_y_and_negative_columns() {
        let mut world = crate::terrain::fixture::blank_world(0);
        let loaded = BlockPos::new(2, 64, 2);
        assert_eq!(
            biome_at_with_generated(&mut world, loaded, |_| -> Option<u8> {
                panic!("a received loaded biome must not query the generated source")
            }),
            1
        );

        // Missing negative columns must invoke the source with the true world position, not
        // local 0..15 coordinates and not the old unconditional Plains substitute.
        let unloaded = BlockPos::new(-17, 64, -33);
        assert_eq!(
            biome_at_with_generated(&mut world, unloaded, |got| {
                assert_eq!(got, unloaded);
                Some(6)
            }),
            6
        );
        assert!(!world.is_chunk_loaded(unloaded.chunk_pos()));

        // The column is loaded, but vanilla's `World.isValid` rejects y == 256 before the
        // chunk lookup. Therefore the generator wins over its received Plains byte.
        let above_top = BlockPos::new(2, 256, 2);
        assert_eq!(
            biome_at_with_generated(&mut world, above_top, |_| Some(2)),
            2
        );
        assert_eq!(
            biome_at(&world, loaded),
            1,
            "invalid-Y lookups do not rewrite chunk data"
        );
    }

    #[test]
    fn generated_color_multiplier_uses_the_provider_for_each_missing_tint_column() {
        let mut world = World::new();
        let water = state_of("minecraft:water");
        let pos = BlockPos::new(-17, 64, -33);
        let expected = mc_data::biome(6).water_color;
        assert_eq!(
            color_multiplier_with_generated(&mut world, water, pos, |got| {
                assert_eq!(got.y, pos.y, "BiomeColorHelper keeps the sample Y");
                Some(6)
            }),
            expected,
            "all nine absent neighbour columns use the source's swamp water colour"
        );
        assert_ne!(
            expected,
            mc_data::biome(1).water_color,
            "the Plains control must bite"
        );
    }

    #[test]
    fn a_potted_fern_delegates_to_tallgrass_column_tint() {
        let mut w = crate::terrain::fixture::blank_world(1);
        let p = BlockPos::new(2, 64, 2);
        let flower_pot = state_of("minecraft:flower_pot");
        w.set_block(p, flower_pot);

        let payload = |item_id, data| mc_world::TileEntity {
            kind: mc_world::TileKind::FlowerPot,
            data: mc_world::TileData::FlowerPot(mc_world::FlowerPotState { item_id, data }),
        };
        w.set_tile_entity(p, payload(31, 2));
        assert_eq!(
            tint_source(&w, flower_pot, p),
            Some(Source::GrassHere),
            "tallgrass meta 2 is the potted fern"
        );

        // Negative controls: neither the untinted tallgrass variants nor another potted
        // ItemBlock delegates to grass, even though all have a valid tile entity. Keeping the
        // unsupported meta 1 here bites a broad `item_id == 31` transcription.
        w.set_tile_entity(p, payload(31, 0));
        assert_eq!(tint_source(&w, flower_pot, p), None);
        w.set_tile_entity(p, payload(31, 1));
        assert_eq!(tint_source(&w, flower_pot, p), None);
        w.set_tile_entity(p, payload(38, 0));
        assert_eq!(tint_source(&w, flower_pot, p), None);
        w.set_tile_entity(p, payload(99, 0));
        assert_eq!(tint_source(&w, flower_pot, p), None);
    }

    fn biome_named(name: &str) -> u8 {
        mc_data::biomes()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("{name} is not a registered biome"))
            .id
    }

    #[test]
    fn mesa_and_swamp_ignore_the_colormap_entirely() {
        // These return constants in vanilla, so they are correct even with no colormap loaded.
        let mesa = biome_named("Mesa");
        assert_eq!(grass_color(mesa, BlockPos::new(0, 64, 0)), 9_470_285);
        assert_eq!(foliage_color(mesa, BlockPos::new(0, 64, 0)), 10_387_789);
        let swamp = biome_named("Swampland");
        assert_eq!(foliage_color(swamp, BlockPos::new(0, 64, 0)), 6_975_545);
    }

    /// java: `BiomeGenSwamp.getGrassColorAtPos` picks between two constants on the sign of
    /// `GRASS_COLOR_NOISE - 0.1`. Before this change the whole branch was dropped and every swamp
    /// column got `6975545`.
    ///
    /// Sample: the 200x200 patch `x, z in [-100, 100)`, matched against the same patch run
    /// through the real Java generator (`Oracle.java`, `SWAMP_BELOW 17090 / 40000`). The count
    /// is the negative control: a generator that had gone flat, or a `<=` instead of `<`, or a
    /// dropped `* 0.0225`, all move it.
    #[test]
    fn swamp_grass_takes_the_grass_colour_noise_branch() {
        let swamp = biome_named("Swampland");
        let mut below = 0;
        for x in -100..100 {
            for z in -100..100 {
                let c = grass_color(swamp, BlockPos::new(x, 64, z));
                assert!(
                    c == 5_011_004 || c == 6_975_545,
                    "swamp grass at ({x}, {z}) is {c}"
                );
                if c == 5_011_004 {
                    below += 1;
                }
            }
        }
        assert_eq!(below, 17_090, "vanilla's own generator says 17090 of 40000");
    }

    /// java: `BiomeGenMutated.getGrassColorAtPos` forwards to `baseBiome`, so Swampland M (134)
    /// must produce swampland's constants and not a colormap lookup.
    ///
    /// Negative control: the mutations that are *not* `BiomeGenMutated` subclasses must not
    /// delegate, and with no colormap installed a non-delegating Swampland M would return the
    /// out-of-range sentinel `0x00FF_00FF` instead — which is exactly what it did before.
    #[test]
    fn mutated_biomes_delegate_their_tint_to_the_base_biome() {
        assert!(!colormaps_installed(), "this test needs the sentinel path");
        let swamp_m = biome_named("Swampland M");
        assert_eq!(swamp_m, 134);
        for pos in [BlockPos::new(0, 64, 0), BlockPos::new(37, 70, -91)] {
            assert_eq!(grass_color(swamp_m, pos), grass_color(6, pos));
            assert_ne!(grass_color(swamp_m, pos), 0x00FF_00FF);
            assert_eq!(foliage_color(swamp_m, pos), 6_975_545);
        }
        // The four delegating classes, and only those.
        let delegating: Vec<u8> = (128u8..=255)
            .filter(|&id| delegating_base(id).is_some())
            .collect();
        assert_eq!(
            delegating,
            vec![130, 133, 134, 149, 151, 155, 156, 157, 158, 163, 164]
        );
        for &id in &delegating {
            assert_eq!(delegating_base(id).expect("base").id, id - 128);
        }
        // Sunflower Plains, Ice Plains Spikes and the Mesa/Taiga/Hills variants build ordinary
        // biome instances in an overridden `createMutatedBiome` and do not forward.
        for id in [129u8, 131, 132, 140, 160, 161, 162, 165, 166, 167] {
            assert!(
                delegating_base(id).is_none(),
                "biome {id} must not delegate"
            );
        }
    }

    #[test]
    fn roofed_forest_darkening_is_keyed_to_the_vanilla_type_not_its_name() {
        assert!(
            !colormaps_installed(),
            "this test needs the zero-filled map"
        );
        let p = BlockPos::new(0, 64, 0);
        let expected = 2_634_762 >> 1;
        assert_eq!(grass_color(29, p), expected);
        assert_eq!(
            grass_color(157, p),
            expected,
            "mutated biome delegates to id 29"
        );
        // Ordinary Forest and Flower Forest both share the class but not forest type 3.
        assert_eq!(grass_color(4, p), 0);
        assert_eq!(grass_color(132, p), 0);
    }

    /// java: `BlockOldLeaf.colorMultiplier` short-circuits spruce and birch to two flat
    /// constants before `super` ever reaches `BiomeColorHelper`.
    #[test]
    fn spruce_and_birch_leaves_are_never_biome_tinted() {
        let mut w = World::new();
        let p = BlockPos::ORIGIN;
        // World::set_block mirrors EmptyChunk and ignores writes to an unloaded column.
        w.load_empty_chunk(p.chunk_pos());
        let leaves = state_of("minecraft:leaves").block();
        let meta = |m: u8| BlockState::new(leaves, m);
        w.set_block(p, meta(1));
        assert_eq!(
            tint_source(&w, meta(1), p),
            Some(Source::Constant(FOLIAGE_PINE))
        );
        w.set_block(p, meta(2));
        assert_eq!(
            tint_source(&w, meta(2), p),
            Some(Source::Constant(FOLIAGE_BIRCH))
        );
        // Oak and jungle still take the colormap, and so do acacia/dark oak (`leaves2`).
        w.set_block(p, meta(0));
        assert_eq!(tint_source(&w, meta(0), p), Some(Source::Foliage));
        w.set_block(p, meta(3));
        assert_eq!(tint_source(&w, meta(3), p), Some(Source::Foliage));
        w.set_block(p, state_of("minecraft:leaves2"));
        assert_eq!(
            tint_source(&w, state_of("minecraft:leaves2"), p),
            Some(Source::Foliage)
        );
        // The decay/check-decay bits (4 and 8) must not disturb the variant.
        w.set_block(p, meta(1 | 4 | 8));
        assert_eq!(
            tint_source(&w, meta(1 | 4 | 8), p),
            Some(Source::Constant(FOLIAGE_PINE))
        );
    }

    #[test]
    fn virtual_tint_overrides_read_the_world_state_not_a_stale_caller_state() {
        let mut w = World::new();
        let p = BlockPos::new(0, 64, 0);
        w.load_empty_chunk(p.chunk_pos());

        let old_leaves = state_of("minecraft:leaves").block();
        let pine = BlockState::new(old_leaves, 1);
        let birch = BlockState::new(old_leaves, 2);
        w.set_block(p, birch);
        assert_eq!(
            tint_source(&w, pine, p),
            Some(Source::Constant(FOLIAGE_BIRCH)),
            "BlockOldLeaf.colorMultiplier reads worldIn.getBlockState(pos)"
        );
        w.set_block(p, state_of("minecraft:stone"));
        assert_eq!(
            tint_source(&w, pine, p),
            Some(Source::Foliage),
            "a stale old-leaf state delegates to BlockLeaves"
        );

        let double_plant = state_of("minecraft:double_plant");
        w.set_block(p, state_of("minecraft:stone"));
        assert_eq!(
            tint_source(&w, double_plant, p),
            Some(Source::Grass),
            "BlockDoublePlant.getVariant returns FERN when the world no longer has the block"
        );

        let stem = state_of("minecraft:pumpkin_stem");
        w.set_block(p, state_of("minecraft:stone"));
        assert_eq!(
            tint_source(&w, stem, p),
            None,
            "BlockStem.getRenderColor delegates to white for a mismatched world state"
        );

        let redstone = state_of("minecraft:redstone_wire");
        w.set_block(p, state_of("minecraft:stone"));
        assert_eq!(
            tint_source(&w, redstone, p),
            None,
            "BlockRedstoneWire.colorMultiplier delegates to white for a mismatched world state"
        );
    }

    /// java: an upper double-plant half whose lower half is missing keeps its own stored VARIANT.
    /// `BLOCK_STATE_IDS` freezes that property as PAEONIA for every upper metadata. The low bits
    /// are a *facing*, so reading them as a variant would tint facings 2 and 3 grass-green.
    #[test]
    fn an_orphan_double_plant_upper_half_does_not_tint() {
        let mut w = World::new();
        let p = BlockPos::new(0, 64, 0);
        w.load_empty_chunk(p.chunk_pos());
        let block = state_of("minecraft:double_plant").block();
        let meta = |m: u8| BlockState::new(block, m);
        for facing in 0..4u8 {
            w.set_block(p, meta(8 | facing));
            assert_eq!(
                tint_source(&w, meta(8 | facing), p),
                None,
                "orphan upper half facing {facing}"
            );
        }
        // A *lower* half still tints on its own variant: grass(2) and fern(3) only.
        w.set_block(p, meta(2));
        assert_eq!(tint_source(&w, meta(2), p), Some(Source::Grass));
        w.set_block(p, meta(3));
        assert_eq!(tint_source(&w, meta(3), p), Some(Source::Grass));
        w.set_block(p, meta(0));
        assert_eq!(tint_source(&w, meta(0), p), None);
        w.set_block(p, meta(5));
        assert_eq!(tint_source(&w, meta(5), p), None);
    }

    #[test]
    fn a_double_plant_upper_copies_the_neighbours_stored_variant_not_its_meta_bits() {
        let mut w = crate::terrain::fixture::blank_world(0);
        let p = BlockPos::new(2, 65, 2);
        let block = state_of("minecraft:double_plant").block();
        let meta = |m: u8| BlockState::new(block, m);

        for variant in 0..=5u8 {
            w.set_block(p, meta(8));
            w.set_block(p.down(), meta(variant));
            assert_eq!(
                tint_source(&w, meta(8), p),
                matches!(variant, 2 | 3).then_some(Source::Grass),
                "lower-half variant {variant}"
            );
        }

        // Hostile upper-on-upper input: metas 10/11 have low bits 2/3, but their generated
        // stored VARIANT is paeonia. A `below.meta() & 7` implementation fails both controls.
        for facing in 0..4u8 {
            let below = meta(8 | facing);
            assert_eq!(stored_property(below, "variant"), Some("paeonia"));
            w.set_block(p, meta(8));
            w.set_block(p.down(), below);
            assert_eq!(tint_source(&w, meta(8), p), None, "upper facing {facing}");
        }
    }

    #[test]
    fn stem_and_redstone_colours_match_vanilla() {
        // java: BlockStem.getRenderColor at age 0 -> 0x00FF00, at age 7 -> (224, 199, 28)
        assert_eq!(stem_color(0), 0x0000_FF00);
        assert_eq!(stem_color(7), (224 << 16) | (199 << 8) | 28);
        // java: BlockRedstoneWire.colorMultiplier(0) -> f1 = 0.3, f2/f3 clamp to 0
        assert_eq!(redstone_color(0), (76 << 16));
    }

    /// The override is this thread's
    /// alone, nests, restores what the thread saw before (also on a panic), and never moves the
    /// installed generation. No test in this binary installs colormaps, so the global is stable.
    #[test]
    fn a_thread_colormap_override_is_thread_scoped_and_restores() {
        let marked = |rgb: u32| -> &'static Colormaps {
            let map = Colormap {
                pixels: vec![rgb; 65_536],
            };
            Box::leak(Box::new(Colormaps {
                grass: map.clone(),
                foliage: map,
            }))
        };
        let (a, b) = (marked(0x12_34_56), marked(0x65_43_21));
        let pos = BlockPos::new(3, 64, -7);
        let probe = move || (grass_color(1, pos), foliage_color(1, pos));
        let generation = colormap_generation();
        let global = probe();
        assert!(!thread_colormaps_overridden());

        let seen = with_thread_colormaps(a, || {
            assert!(thread_colormaps_overridden());
            let other_thread = std::thread::spawn(move || (thread_colormaps_overridden(), probe()))
                .join()
                .expect("probe thread");
            assert_eq!(
                other_thread,
                (false, global),
                "another thread saw the override"
            );
            (probe(), with_thread_colormaps(b, probe), probe())
        });
        let (ma, mb) = ((0x12_34_56, 0x12_34_56), (0x65_43_21, 0x65_43_21));
        assert_eq!(seen, (ma, mb, ma));
        assert!(
            !thread_colormaps_overridden(),
            "the override outlived its scope"
        );
        assert_eq!(probe(), global);

        let caught = std::panic::catch_unwind(|| with_thread_colormaps(a, || panic!("boom")));
        assert!(caught.is_err());
        assert!(
            !thread_colormaps_overridden(),
            "a panic left the override in place"
        );
        assert_eq!(probe(), global);
        assert_eq!(colormap_generation(), generation);
    }
}
