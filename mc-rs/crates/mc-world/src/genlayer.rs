//! The client-side 1.8.9 overworld biome provider.
//!
//! `WorldChunkManager` is normally thought of as a world-generation class, but the vanilla
//! multiplayer client constructs one anyway: `WorldClient` registers itself with its provider
//! and the provider evaluates the `GenLayer` stack whenever a column is not resident.  The
//! server-supplied biome bytes remain authoritative in [`crate::world::World`]; this module is
//! only the exact seed-0 fallback used by that client-side provider branch.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use mc_data::biome;

const LCG_MULTIPLIER: i64 = 6_364_136_223_846_793_005;
const LCG_ADDEND: i64 = 1_442_695_040_888_963_407;
const BIOME_CACHE_CLEANUP_INTERVAL_MS: i64 = 7_500;
const BIOME_CACHE_IDLE_TIMEOUT_MS: i64 = 30_000;
const BIOME_CACHE_WIDTH: i32 = 16;
const BIOME_CACHE_AREA: usize = (BIOME_CACHE_WIDTH * BIOME_CACHE_WIDTH) as usize;

/// How far from the origin (in blocks, either axis) a cache block may lie
/// for [`OverworldBiomeProvider::build_cache_block`] to fill it with one 16x16 layer request.
///
/// A `GenLayer` value is a function of its position only, which is why vanilla can fill a
/// `BiomeCache.Block` with one `getBiomeGenAt(.., 16, 16, ..)`. The one exception is integer
/// overflow: the layers derive parent areas from `origin + width` and halvings, and multiply the
/// parent coordinates back by the same factor (zoom `* 2`, Voronoi `* 4`), so next to the `i32`
/// edge those wrap differently for a 16-wide request than for a 1-wide one. Every intermediate
/// stays within about a hundred of the requested area's coordinates, so within `2^26` nothing comes
/// near `2^31`. Vanilla's world border is at 30 million (`< 2^25`).
const BIOME_BLOCK_QUERY_MAX_ABS: i32 = 1 << 26;

/// Kill switch: `MC_NW_F5_BIOME_BLOCK=0` fills every cache block with 256
/// one-cell requests, as before. Read once.
fn biome_block_query_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("MC_NW_F5_BIOME_BLOCK")
                .as_deref()
                .map(str::trim),
            Ok("0")
        )
    })
}

/// Whether a cache block at this origin can be filled by one 16x16 request
/// with exactly the ids 256 one-cell requests give (see [`BIOME_BLOCK_QUERY_MAX_ABS`]).
fn biome_block_query_is_exact(origin_x: i32, origin_z: i32) -> bool {
    let inside = |v: i32| (-BIOME_BLOCK_QUERY_MAX_ABS..=BIOME_BLOCK_QUERY_MAX_ABS).contains(&v);
    inside(origin_x) && inside(origin_z)
}

type CacheClock = Arc<dyn Fn() -> i64 + Send + Sync>;

/// Observable cache counters used by focused parity controls.
///
/// The counters deliberately live beside the cache instead of in a global diagnostic so tests
/// can prove a repeated lookup is a hit, an idle lookup rebuilds exactly once, and a negative
/// chunk coordinate does not alias its positive counterpart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BiomeCacheStats {
    /// Number of cached 16x16 columns currently resident.
    pub(crate) entries: usize,
    /// Number of block-coordinate lookups served by an existing cache block.
    pub(crate) hits: u64,
    /// Number of 16x16 blocks generated and inserted.
    pub(crate) rebuilds: u64,
    /// Number of blocks removed by the Java idle/clock-reversal cleanup rule.
    pub(crate) evictions: u64,
    /// Number of cleanup passes whose 7.5-second cadence was reached.
    pub(crate) cleanup_passes: u64,
}

#[derive(Clone)]
struct BiomeCache {
    state: Arc<Mutex<BiomeCacheState>>,
    clock: CacheClock,
}

#[derive(Default)]
struct BiomeCacheState {
    blocks: HashMap<u64, BiomeCacheBlock>,
    last_cleanup_time: i64,
    stats: BiomeCacheStats,
}

struct BiomeCacheBlock {
    /// Java `BiomeCache.Block.biomes`, in local index `x & 15 | (z & 15) << 4` order.
    biomes: [u8; BIOME_CACHE_AREA],
    /// Java `BiomeCache.Block.rainfallValues`, generated alongside `biomes`.
    rainfall: [f32; BIOME_CACHE_AREA],
    /// Java `BiomeCache.Block.lastAccessTime`.
    last_access_time: i64,
}

impl BiomeCache {
    fn with_clock(clock: CacheClock) -> Self {
        Self {
            state: Arc::new(Mutex::new(BiomeCacheState::default())),
            clock,
        }
    }

    #[inline]
    fn now(&self) -> i64 {
        (self.clock)()
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, BiomeCacheState> {
        // A poisoned cache must not take the client world down.  The generated layer is
        // immutable and every entry is disposable, so retaining the state is the least
        // surprising recovery and mirrors Java's best-effort cache semantics.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn cleanup_if_due(state: &mut BiomeCacheState, now: i64) {
        let elapsed = now.saturating_sub(state.last_cleanup_time);
        if elapsed <= BIOME_CACHE_CLEANUP_INTERVAL_MS && now >= state.last_cleanup_time {
            return;
        }

        state.last_cleanup_time = now;
        state.stats.cleanup_passes = state.stats.cleanup_passes.saturating_add(1);
        let stale_keys: Vec<u64> = state
            .blocks
            .iter()
            .filter_map(|(key, block)| {
                let age = now.saturating_sub(block.last_access_time);
                let stale = age > BIOME_CACHE_IDLE_TIMEOUT_MS || now < block.last_access_time;
                stale.then_some(*key)
            })
            .collect();
        state.stats.evictions = state
            .stats
            .evictions
            .saturating_add(stale_keys.len() as u64);
        for key in stale_keys {
            state.blocks.remove(&key);
        }
        state.stats.entries = state.blocks.len();
    }

    #[cfg(test)]
    fn stats(&self) -> BiomeCacheStats {
        let mut state = self.lock_state();
        state.stats.entries = state.blocks.len();
        state.stats
    }

    fn cleanup(&self) {
        let mut state = self.lock_state();
        Self::cleanup_if_due(&mut state, self.now());
    }
}

fn system_time_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

/// Exact default 1.8.9 overworld biome provider used by the multiplayer client.
///
/// The layer graph is immutable after construction. Java stores each layer's mutable
/// `worldGenSeed`/`chunkSeed`; here those values are derived into local variables for each
/// request, which preserves the result while making generation safe for parallel terrain
/// meshing. The provider additionally owns Java's `BiomeCache`: one generated 16x16 block holds
/// both biome ids and rainfall values, refreshes its access timestamp on every lookup, and is
/// eligible for the same 7.5-second cleanup cadence/30-second idle eviction as vanilla.
#[derive(Clone)]
pub(crate) struct OverworldBiomeProvider {
    /// Immutable once seeded, so it is shared rather than owned: `Clone` (taken by every render
    /// snapshot, i.e. every frame that dispatches mesh work) is one reference-count increment
    /// instead of a deep copy of ~80 boxed nodes, and `World::new` reuses the graph built for
    /// the same `(seed, biome size)` instead of rebuilding it. See [`shared_layer_graph`].
    layer: Arc<Node>,
    biome_cache: BiomeCache,
}

impl OverworldBiomeProvider {
    /// Build `GenLayer.initializeAllBiomeGenerators(0L, WorldType.DEFAULT, "")`.
    #[must_use]
    pub(crate) fn new(seed: i64) -> Self {
        Self::with_clock(seed, Arc::new(system_time_millis))
    }

    /// Build the provider for a packet world type with the default generator options.
    ///
    /// Java's `WorldType.LARGE_BIOMES` changes the biome zoom count from four to six, and
    /// `WorldType.DEFAULT_1_1` makes `GenLayerBiome` pick warm-climate biomes from its legacy
    /// six-entry table; the other packet world types use the ordinary default graph here because
    /// Join Game and Respawn do not carry generator options (CUSTOMIZED with empty options is
    /// DEFAULT's graph, which `BiomeTypeRefGen` checks).
    pub(crate) fn new_for_world_type(seed: i64, world_type: &str) -> Self {
        Self::with_clock_for_world_type(seed, Arc::new(system_time_millis), world_type)
    }

    /// Build a provider with an injected millisecond clock.
    ///
    /// The production path uses the wall clock, as `MinecraftServer.getCurrentTimeMillis()`
    /// does. Keeping the clock as an injected `Send + Sync` callback makes stale/fresh and clock
    /// reversal behavior deterministic in focused controls without introducing a process-global
    /// test clock or changing biome generation.
    pub(crate) fn with_clock(seed: i64, clock: CacheClock) -> Self {
        Self::with_clock_for_world_type(seed, clock, "default")
    }

    fn with_clock_for_world_type(seed: i64, clock: CacheClock, world_type: &str) -> Self {
        let biome_size = if world_type.eq_ignore_ascii_case("largeBiomes") {
            6
        } else {
            4
        };
        // java: `GenLayerBiome`'s constructor swaps its warm table when
        // `p_i45560_4_ == WorldType.DEFAULT_1_1`.
        let warm = if world_type.eq_ignore_ascii_case("default_1_1") {
            WarmBiomes::Legacy
        } else {
            WarmBiomes::Default
        };
        Self {
            layer: shared_layer_graph(seed, biome_size, warm),
            biome_cache: BiomeCache::with_clock(clock),
        }
    }

    /// Return the biome id from the provider's `biomeIndexLayer` at one block column.
    #[must_use]
    pub(crate) fn biome_at(&self, x: i32, z: i32) -> u8 {
        self.cached_values(x, z).0
    }

    /// Return Java's quantized `WorldChunkManager.getRainfall` value at one block column.
    ///
    /// This is intentionally sourced from the same cache block as [`Self::biome_at`]. The Java
    /// manager constructs `BiomeCache.Block` by filling rainfall first and biome ids second;
    /// both arrays therefore rebuild together and cannot drift when a provider lookup is
    /// repeated or an idle block is evicted.
    #[must_use]
    pub(crate) fn rainfall_at(&self, x: i32, z: i32) -> f32 {
        self.cached_values(x, z).1
    }

    /// Run Java `WorldChunkManager.cleanupCache()`'s cadence-controlled pass.
    pub(crate) fn cleanup_cache(&self) {
        self.biome_cache.cleanup();
    }

    /// Return cache controls for focused parity tests and diagnostics.
    #[cfg(test)]
    pub(crate) fn cache_stats(&self) -> BiomeCacheStats {
        self.biome_cache.stats()
    }

    fn cached_values(&self, x: i32, z: i32) -> (u8, f32) {
        let chunk_x = x >> 4;
        let chunk_z = z >> 4;
        let key = biome_cache_key(chunk_x, chunk_z);
        let index = cache_index(x, z);
        let now = self.biome_cache.now();
        let mut state = self.biome_cache.lock_state();
        BiomeCache::cleanup_if_due(&mut state, now);

        if let Some(block) = state.blocks.get_mut(&key) {
            block.last_access_time = now;
            let values = (block.biomes[index], block.rainfall[index]);
            state.stats.hits = state.stats.hits.saturating_add(1);
            return values;
        }

        // Holding this short mutex while a 16x16 layer request is generated prevents duplicate
        // rebuilds when parallel mesher workers arrive at a new chunk simultaneously. The layer
        // itself is immutable; only this disposable cache state is serialized.
        let block = self.build_cache_block(chunk_x, chunk_z, now);
        state.blocks.insert(key, block);
        state.stats.rebuilds = state.stats.rebuilds.saturating_add(1);
        state.stats.entries = state.blocks.len();
        let block = state
            .blocks
            .get(&key)
            .expect("inserted biome cache block must be present");
        (block.biomes[index], block.rainfall[index])
    }

    fn build_cache_block(&self, chunk_x: i32, chunk_z: i32, now: i64) -> BiomeCacheBlock {
        // Java's `x << 4` is an int operation. Wrapping multiplication retains that behavior at
        // the extreme signed coordinate edges while `>> 4` above preserves negative chunk floor.
        let origin_x = chunk_x.wrapping_mul(BIOME_CACHE_WIDTH);
        let origin_z = chunk_z.wrapping_mul(BIOME_CACHE_WIDTH);
        let mut biomes = [0; BIOME_CACHE_AREA];
        let mut rainfall = [0.0; BIOME_CACHE_AREA];
        // One 16x16 request, as vanilla's `BiomeCache.Block` makes, wherever
        // it gives exactly the ids of the one-cell requests below (see
        // `BIOME_BLOCK_QUERY_MAX_ABS`). 256 one-cell requests ran the whole layer graph 256 times:
        // ~9 ms per block in an optimised build, paid on the game thread on the first frame of
        // every world (the camera's column is not loaded yet, so sky and fog colour read this
        // fallback) and by mesher workers for every unloaded neighbour column.
        let whole_block = (biome_block_query_enabled()
            && biome_block_query_is_exact(origin_x, origin_z))
        .then(|| {
            self.layer
                .get_ints(origin_x, origin_z, BIOME_CACHE_WIDTH, BIOME_CACHE_WIDTH)
        });
        for local_z in 0..BIOME_CACHE_WIDTH {
            for local_x in 0..BIOME_CACHE_WIDTH {
                let index = (local_z * BIOME_CACHE_WIDTH + local_x) as usize;
                let x = origin_x.wrapping_add(local_x);
                let z = origin_z.wrapping_add(local_z);
                // The public provider historically evaluated one column at a time. Keeping that
                // exact query shape inside the 16x16 cache is important at the signed i32 edge:
                // several GenLayer area implementations legitimately inspect `origin + width`
                // and therefore can overflow differently from a one-cell request. The cache is
                // an acceleration/storage seam, not permission to change an existing biome id.
                let raw_id = match &whole_block {
                    Some(raw) => raw[index],
                    None => self.layer.get_ints(x, z, 1, 1)[0],
                };
                let id = canonical_generated_biome(raw_id);
                biomes[index] = id;
                rainfall[index] = java_rainfall(id);
            }
        }
        BiomeCacheBlock {
            biomes,
            rainfall,
            last_access_time: now,
        }
    }
}

/// `GenLayerBiome.field_151623_c`, the table a warm (climate 1) cell picks its biome from.
///
/// The constructor replaces the default `{desert, desert, desert,
/// savanna, savanna, plains}` with `{desert, forest, extremeHills, swampland, plains, taiga}`
/// for `WorldType.DEFAULT_1_1`; that is the only thing the legacy type changes in the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WarmBiomes {
    Default,
    Legacy,
}

impl WarmBiomes {
    fn table(self) -> [i32; 6] {
        match self {
            Self::Default => [2, 2, 2, 35, 35, 1],
            Self::Legacy => [2, 4, 3, 6, 1, 5],
        }
    }
}

/// The seeded `GenLayer` graph for `(seed, biome_size, warm table)`, built once per process.
///
/// The graph is a pure function of those inputs (`biome_size` and the warm table are the only
/// things the packet world type changes, see [`OverworldBiomeProvider::new_for_world_type`]) and
/// is never mutated after `init_world_gen_seed`: every request derives its chunk seeds into
/// locals. Sharing it therefore cannot change a biome id. The per-provider [`BiomeCache`] is
/// deliberately *not* shared here — a new `World` still starts with an empty cache, exactly as
/// before.
///
/// The client only ever asks for seed 0 with biome size 4 or 6 and one of two warm tables; the
/// small bound keeps a test that sweeps seeds from growing this without limit.
/// `(seed, biome size, warm table, graph)` entries shared by [`shared_layer_graph`].
type SharedGraphs = Mutex<Vec<(i64, i32, WarmBiomes, Arc<Node>)>>;

fn shared_layer_graph(seed: i64, biome_size: i32, warm: WarmBiomes) -> Arc<Node> {
    const MAX_SHARED_GRAPHS: usize = 8;
    static GRAPHS: std::sync::OnceLock<SharedGraphs> = std::sync::OnceLock::new();
    let mut graphs = GRAPHS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, _, _, graph)) = graphs
        .iter()
        .find(|(s, size, table, _)| *s == seed && *size == biome_size && *table == warm)
    {
        return Arc::clone(graph);
    }
    let mut layer = default_layer_graph(biome_size, warm);
    layer.init_world_gen_seed(seed);
    let layer = Arc::new(layer);
    if graphs.len() >= MAX_SHARED_GRAPHS {
        graphs.remove(0);
    }
    graphs.push((seed, biome_size, warm, Arc::clone(&layer)));
    layer
}

#[inline]
fn biome_cache_key(chunk_x: i32, chunk_z: i32) -> u64 {
    // Java's LongHashMap key: `(long)x & 0xffffffffL | ((long)z & 0xffffffffL) << 32`.
    u64::from(chunk_x as u32) | (u64::from(chunk_z as u32) << 32)
}

#[inline]
fn cache_index(x: i32, z: i32) -> usize {
    ((z & 15) as usize) << 4 | (x & 15) as usize
}

#[inline]
fn canonical_generated_biome(id: i32) -> u8 {
    // Java's `getBiomeFromBiomeList(id, ocean)` uses Ocean for a missing/invalid registry row.
    // Registered mutated ids are intentionally retained.
    if (0..=255).contains(&id) {
        let id = id as u8;
        if biome(id).id == id {
            return id;
        }
    }
    0
}

#[inline]
pub(crate) fn java_rainfall(id: u8) -> f32 {
    // BiomeGenBase.getIntRainfall() truncates `rainfall * 65536` before returning the float
    // rainfall array value. Rust's f32 table values are already the canonical registry values,
    // but retaining this quantization matters for exact WorldChunkManager parity.
    let rainfall = (biome(id).rainfall * 65_536.0) as u32;
    rainfall as f32 / 65_536.0
}

#[derive(Clone)]
enum Layer {
    Island,
    FuzzyZoom(Node),
    AddIsland(Node),
    Zoom(Node),
    RemoveTooMuchOcean(Node),
    AddSnow(Node),
    AddMushroomIsland(Node),
    Edge(Node, EdgeMode),
    DeepOcean(Node),
    Biome(Node, WarmBiomes),
    BiomeEdge(Node),
    Hills(Node, Node),
    RiverInit(Node),
    River(Node),
    Smooth(Node),
    RareBiome(Node),
    Shore(Node),
    RiverMix(Node, Node),
    Voronoi(Node),
}

/// A layer node carrying the Java base/world seed pair.
#[derive(Clone)]
struct Node {
    base_seed: i64,
    world_gen_seed: i64,
    layer: Box<Layer>,
}

#[derive(Clone, Copy)]
enum EdgeMode {
    CoolWarm,
    HeatIce,
    Special,
}

impl Node {
    fn new(seed: i64, layer: Layer) -> Self {
        let mut base = seed;
        for _ in 0..3 {
            base = base
                .wrapping_mul(base.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_ADDEND))
                .wrapping_add(seed);
        }
        Self {
            base_seed: base,
            world_gen_seed: 0,
            layer: Box::new(layer),
        }
    }

    /// Seed `GenLayerHills`'s second input the way Java leaves it:
    /// its own zoom layers keep `worldGenSeed == 0` (they are only reachable through the hills
    /// layer, which never seeds them), while the shared river-init layer under them is seeded
    /// as the river chain seeds it.
    fn init_hills_river_input(&mut self, seed: i64) {
        match self.layer.as_mut() {
            Layer::Zoom(parent) => parent.init_hills_river_input(seed),
            _ => self.init_world_gen_seed(seed),
        }
    }

    fn init_world_gen_seed(&mut self, seed: i64) {
        // `GenLayerRiverMix` initializes both of its independent branches before initializing
        // itself. Every other layer initializes its single parent first, then applies its own
        // three LCG rounds.
        match self.layer.as_mut() {
            Layer::RiverMix(biome_layer, river_layer) => {
                biome_layer.init_world_gen_seed(seed);
                river_layer.init_world_gen_seed(seed);
            }
            Layer::Island => {}
            Layer::Hills(parent, river) => {
                parent.init_world_gen_seed(seed);
                // java: `GenLayerHills` does not override
                // `initWorldGenSeed`, and `GenLayer.initWorldGenSeed` recurses into `parent`
                // only, so its second input is never seeded through it. That input is
                // `GenLayerZoom.magnify(1000L, genlayerriverinit, 2)`: two zoom layers that
                // nothing else references, so they keep `worldGenSeed == 0`, over the river-init
                // layer the river chain shares and seeds.
                river.init_hills_river_input(seed);
            }
            Layer::FuzzyZoom(parent)
            | Layer::AddIsland(parent)
            | Layer::Zoom(parent)
            | Layer::RemoveTooMuchOcean(parent)
            | Layer::AddSnow(parent)
            | Layer::AddMushroomIsland(parent)
            | Layer::Edge(parent, _)
            | Layer::DeepOcean(parent)
            | Layer::BiomeEdge(parent)
            | Layer::RiverInit(parent)
            | Layer::River(parent)
            | Layer::Smooth(parent)
            | Layer::RareBiome(parent)
            | Layer::Shore(parent)
            | Layer::Voronoi(parent) => parent.init_world_gen_seed(seed),
            Layer::Biome(parent, _) => parent.init_world_gen_seed(seed),
        }
        self.world_gen_seed = seed;
        for _ in 0..3 {
            self.world_gen_seed = self
                .world_gen_seed
                .wrapping_mul(
                    self.world_gen_seed
                        .wrapping_mul(LCG_MULTIPLIER)
                        .wrapping_add(LCG_ADDEND),
                )
                .wrapping_add(self.base_seed);
        }
    }

    fn chunk_seed(&self, x: i32, z: i32) -> i64 {
        let x = i64::from(x);
        let z = i64::from(z);
        let mut seed = self.world_gen_seed;
        seed = seed
            .wrapping_mul(seed.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_ADDEND))
            .wrapping_add(x);
        seed = seed
            .wrapping_mul(seed.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_ADDEND))
            .wrapping_add(z);
        seed = seed
            .wrapping_mul(seed.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_ADDEND))
            .wrapping_add(x);
        seed = seed
            .wrapping_mul(seed.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_ADDEND))
            .wrapping_add(z);
        seed
    }

    fn next_int(&self, chunk_seed: &mut i64, bound: i32) -> i32 {
        debug_assert!(bound > 0);
        // Java's remainder is sign-preserving; Rust's `%` has the same rule.  The adjustment is
        // the exact `if (i < 0) i += bound` in GenLayer.nextInt.
        let mut value = (*chunk_seed >> 24) % i64::from(bound);
        if value < 0 {
            value += i64::from(bound);
        }
        let current = *chunk_seed;
        *chunk_seed = current
            .wrapping_mul(
                current
                    .wrapping_mul(LCG_MULTIPLIER)
                    .wrapping_add(LCG_ADDEND),
            )
            .wrapping_add(self.world_gen_seed);
        value as i32
    }

    fn get_ints(&self, area_x: i32, area_y: i32, area_width: i32, area_height: i32) -> Vec<i32> {
        if area_width <= 0 || area_height <= 0 {
            return Vec::new();
        }
        match self.layer.as_ref() {
            Layer::Island => island(self, area_x, area_y, area_width, area_height),
            Layer::FuzzyZoom(parent) => {
                zoom(self, parent, area_x, area_y, area_width, area_height, true)
            }
            Layer::AddIsland(parent) => {
                add_island(self, parent, area_x, area_y, area_width, area_height)
            }
            Layer::Zoom(parent) => {
                zoom(self, parent, area_x, area_y, area_width, area_height, false)
            }
            Layer::RemoveTooMuchOcean(parent) => {
                remove_too_much_ocean(self, parent, area_x, area_y, area_width, area_height)
            }
            Layer::AddSnow(parent) => {
                add_snow(self, parent, area_x, area_y, area_width, area_height)
            }
            Layer::AddMushroomIsland(parent) => {
                add_mushroom_island(self, parent, area_x, area_y, area_width, area_height)
            }
            Layer::Edge(parent, mode) => {
                edge(self, parent, *mode, area_x, area_y, area_width, area_height)
            }
            Layer::DeepOcean(parent) => deep_ocean(parent, area_x, area_y, area_width, area_height),
            Layer::Biome(parent, warm) => {
                biome_layer(self, parent, *warm, area_x, area_y, area_width, area_height)
            }
            Layer::BiomeEdge(parent) => {
                biome_edge(self, parent, area_x, area_y, area_width, area_height)
            }
            Layer::Hills(parent, river_init) => hills(
                self,
                parent,
                river_init,
                area_x,
                area_y,
                area_width,
                area_height,
            ),
            Layer::RiverInit(parent) => {
                river_init(self, parent, area_x, area_y, area_width, area_height)
            }
            Layer::River(parent) => river(self, parent, area_x, area_y, area_width, area_height),
            Layer::Smooth(parent) => smooth(self, parent, area_x, area_y, area_width, area_height),
            Layer::RareBiome(parent) => {
                rare_biome(self, parent, area_x, area_y, area_width, area_height)
            }
            Layer::Shore(parent) => shore(parent, area_x, area_y, area_width, area_height),
            Layer::RiverMix(biome_layer, river_layer) => river_mix(
                biome_layer,
                river_layer,
                area_x,
                area_y,
                area_width,
                area_height,
            ),
            Layer::Voronoi(parent) => {
                voronoi(self, parent, area_x, area_y, area_width, area_height)
            }
        }
    }
}

fn unary(seed: i64, layer: Layer) -> Node {
    Node::new(seed, layer)
}

fn default_layer_graph(biome_size: i32, warm: WarmBiomes) -> Node {
    unary(
        10,
        Layer::Voronoi(default_river_mix_graph(biome_size, warm)),
    )
}

fn default_river_mix_graph(biome_size: i32, warm: WarmBiomes) -> Node {
    // This is a direct spelling of GenLayer.initializeAllBiomeGenerators for
    // WorldType.DEFAULT and an empty generator-options string.  Keeping the construction order
    // visible is important: several layers reuse the same river-init branch and each reuse has
    // its own zoom nodes/base seed, exactly as the Java code does. LARGE_BIOMES only changes
    // `biome_size` and DEFAULT_1_1 only changes GenLayerBiome's warm table (`warm`).
    let mut layer = unary(1, Layer::Island);
    layer = unary(2000, Layer::FuzzyZoom(layer));
    let mut add_island = unary(1, Layer::AddIsland(layer));
    add_island = unary(2001, Layer::Zoom(add_island));
    add_island = unary(2, Layer::AddIsland(add_island));
    add_island = unary(50, Layer::AddIsland(add_island));
    add_island = unary(70, Layer::AddIsland(add_island));
    let mut layer = unary(2, Layer::RemoveTooMuchOcean(add_island));
    layer = unary(2, Layer::AddSnow(layer));
    layer = unary(3, Layer::AddIsland(layer));
    layer = unary(2, Layer::Edge(layer, EdgeMode::CoolWarm));
    layer = unary(2, Layer::Edge(layer, EdgeMode::HeatIce));
    layer = unary(3, Layer::Edge(layer, EdgeMode::Special));
    layer = unary(2002, Layer::Zoom(layer));
    layer = unary(2003, Layer::Zoom(layer));
    layer = unary(4, Layer::AddIsland(layer));
    layer = unary(5, Layer::AddMushroomIsland(layer));
    layer = unary(4, Layer::DeepOcean(layer));
    let genlayer4 = layer;

    let river_init = unary(100, Layer::RiverInit(genlayer4.clone()));
    let biome = unary(200, Layer::Biome(genlayer4.clone(), warm));
    let biome = magnify(1000, biome, 2);
    let biome = unary(1000, Layer::BiomeEdge(biome));
    let hills_river = magnify(1000, river_init.clone(), 2);
    let hills = unary(1000, Layer::Hills(biome, hills_river));
    let river = magnify(1000, river_init.clone(), 2);
    let river = magnify(1000, river, 4);
    let river = unary(1, Layer::River(river));
    let river = unary(1000, Layer::Smooth(river));

    let mut hills = unary(1001, Layer::RareBiome(hills));
    for k in 0..i64::from(biome_size) {
        hills = unary(1000 + k, Layer::Zoom(hills));
        if k == 0 {
            hills = unary(3, Layer::AddIsland(hills));
        }
        if k == 1 {
            hills = unary(1000, Layer::Shore(hills));
        }
    }
    let hills = unary(1000, Layer::Smooth(hills));
    unary(100, Layer::RiverMix(hills, river))
}

fn magnify(seed: i64, mut layer: Node, times: i32) -> Node {
    for i in 0..times {
        layer = unary(seed.wrapping_add(i64::from(i)), Layer::Zoom(layer));
    }
    layer
}

fn output_size(width: i32, height: i32) -> usize {
    (width as usize).saturating_mul(height as usize)
}

#[inline]
fn at(data: &[i32], width: i32, x: i32, y: i32) -> i32 {
    data[(x as usize) + (y as usize) * (width as usize)]
}

#[inline]
fn set(data: &mut [i32], width: i32, x: i32, y: i32, value: i32) {
    data[(x as usize) + (y as usize) * (width as usize)] = value;
}

fn island(layer: &Node, area_x: i32, area_y: i32, width: i32, height: i32) -> Vec<i32> {
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            set(
                &mut out,
                width,
                x,
                y,
                (layer.next_int(&mut seed, 10) == 0) as i32,
            );
        }
    }
    if area_x > -width && area_x <= 0 && area_y > -height && area_y <= 0 {
        let x = -area_x;
        let y = -area_y;
        set(&mut out, width, x, y, 1);
    }
    out
}

fn zoom(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
    fuzzy: bool,
) -> Vec<i32> {
    let px = area_x >> 1;
    let py = area_y >> 1;
    let pw = (width >> 1) + 2;
    let ph = (height >> 1) + 2;
    let parent_values = parent.get_ints(px, py, pw, ph);
    let zoom_width = (pw - 1) << 1;
    let zoom_height = (ph - 1) << 1;
    let mut zoomed = vec![0; output_size(zoom_width, zoom_height)];
    for y in 0..(ph - 1) {
        let mut out_index = (y << 1) * zoom_width;
        let mut i2 = 0;
        let mut north_west = at(&parent_values, pw, 0, y);
        let mut south_west = at(&parent_values, pw, 0, y + 1);
        while i2 < pw - 1 {
            let mut seed = layer.chunk_seed((i2 + px).wrapping_mul(2), (y + py).wrapping_mul(2));
            let north_east = at(&parent_values, pw, i2 + 1, y);
            let south_east = at(&parent_values, pw, i2 + 1, y + 1);
            set(
                &mut zoomed,
                zoom_width,
                out_index % zoom_width,
                out_index / zoom_width,
                north_west,
            );
            // Both Java zoom variants choose the vertical edge randomly; only
            // the centre selection differs for GenLayerZoomFuzzy.
            let vertical = select_random(layer, &mut seed, [north_west, south_west]);
            set(
                &mut zoomed,
                zoom_width,
                (out_index + zoom_width) % zoom_width,
                (out_index + zoom_width) / zoom_width,
                vertical,
            );
            let horizontal = select_random(layer, &mut seed, [north_west, north_east]);
            set(
                &mut zoomed,
                zoom_width,
                (out_index + 1) % zoom_width,
                (out_index + 1) / zoom_width,
                horizontal,
            );
            let centre = if fuzzy {
                select_random(
                    layer,
                    &mut seed,
                    [north_west, north_east, south_west, south_east],
                )
            } else {
                select_mode_or_random(
                    layer, &mut seed, north_west, north_east, south_west, south_east,
                )
            };
            set(
                &mut zoomed,
                zoom_width,
                (out_index + 1 + zoom_width) % zoom_width,
                (out_index + 1 + zoom_width) / zoom_width,
                centre,
            );
            out_index += 2;
            i2 += 1;
            north_west = north_east;
            south_west = south_east;
        }
    }
    let mut out = vec![0; output_size(width, height)];
    let xoff = area_x & 1;
    let yoff = area_y & 1;
    for y in 0..height {
        let source_start = ((y + yoff) * zoom_width + xoff) as usize;
        let dest_start = (y * width) as usize;
        out[dest_start..dest_start + width as usize]
            .copy_from_slice(&zoomed[source_start..source_start + width as usize]);
    }
    out
}

fn select_random<const N: usize>(layer: &Node, seed: &mut i64, values: [i32; N]) -> i32 {
    values[layer.next_int(seed, N as i32) as usize]
}

fn select_mode_or_random(layer: &Node, seed: &mut i64, a: i32, b: i32, c: i32, d: i32) -> i32 {
    if b == c && c == d {
        b
    } else if a == b && a == c
        || a == b && a == d
        || a == c && a == d
        || a == b && c != d
        || a == c && b != d
        || a == d && b != c
    {
        a
    } else if b == c && a != d || b == d && a != c {
        b
    } else if c == d && a != b {
        c
    } else {
        select_random(layer, seed, [a, b, c, d])
    }
}

fn add_island(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let ph = height + 2;
    let values = parent.get_ints(area_x.wrapping_sub(1), area_y.wrapping_sub(1), pw, ph);
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let nw = at(&values, pw, x, y);
            let ne = at(&values, pw, x + 2, y);
            let sw = at(&values, pw, x, y + 2);
            let se = at(&values, pw, x + 2, y + 2);
            let centre = at(&values, pw, x + 1, y + 1);
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let value = if centre != 0 || nw == 0 && ne == 0 && sw == 0 && se == 0 {
                if centre > 0 && (nw == 0 || ne == 0 || sw == 0 || se == 0) {
                    if layer.next_int(&mut seed, 5) == 0 {
                        if centre == 4 {
                            4
                        } else {
                            0
                        }
                    } else {
                        centre
                    }
                } else {
                    centre
                }
            } else {
                let mut bound = 1;
                let mut picked = 1;
                if nw != 0 {
                    if layer.next_int(&mut seed, bound) == 0 {
                        picked = nw;
                    }
                    bound += 1;
                }
                if ne != 0 {
                    if layer.next_int(&mut seed, bound) == 0 {
                        picked = ne;
                    }
                    bound += 1;
                }
                if sw != 0 {
                    if layer.next_int(&mut seed, bound) == 0 {
                        picked = sw;
                    }
                    bound += 1;
                }
                if se != 0 && layer.next_int(&mut seed, bound) == 0 {
                    picked = se;
                }
                if layer.next_int(&mut seed, 3) == 0 {
                    picked
                } else if picked == 4 {
                    4
                } else {
                    0
                }
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn remove_too_much_ocean(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let north = at(&values, pw, x + 1, y);
            let east = at(&values, pw, x + 2, y + 1);
            let west = at(&values, pw, x, y + 1);
            let south = at(&values, pw, x + 1, y + 2);
            let centre = at(&values, pw, x + 1, y + 1);
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let value = if centre == 0
                && north == 0
                && east == 0
                && west == 0
                && south == 0
                && layer.next_int(&mut seed, 2) == 0
            {
                1
            } else {
                centre
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn add_snow(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let value = at(&values, pw, x + 1, y + 1);
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let value = if value == 0 {
                0
            } else {
                let r = layer.next_int(&mut seed, 6);
                if r == 0 {
                    4
                } else if r <= 1 {
                    3
                } else {
                    1
                }
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn add_mushroom_island(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let nw = at(&values, pw, x, y);
            let ne = at(&values, pw, x + 2, y);
            let sw = at(&values, pw, x, y + 2);
            let se = at(&values, pw, x + 2, y + 2);
            let centre = at(&values, pw, x + 1, y + 1);
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let value = if centre == 0
                && nw == 0
                && ne == 0
                && sw == 0
                && se == 0
                && layer.next_int(&mut seed, 100) == 0
            {
                14
            } else {
                centre
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn edge(
    layer: &Node,
    parent: &Node,
    mode: EdgeMode,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let centre = at(&values, pw, x + 1, y + 1);
            let value = match mode {
                EdgeMode::CoolWarm => {
                    if centre == 1 {
                        let n = at(&values, pw, x + 1, y);
                        let e = at(&values, pw, x + 2, y + 1);
                        let w = at(&values, pw, x, y + 1);
                        let s = at(&values, pw, x + 1, y + 2);
                        if [n, e, w, s].iter().any(|v| *v == 3 || *v == 4) {
                            2
                        } else {
                            centre
                        }
                    } else {
                        centre
                    }
                }
                EdgeMode::HeatIce => {
                    if centre == 4 {
                        let n = at(&values, pw, x + 1, y);
                        let e = at(&values, pw, x + 2, y + 1);
                        let w = at(&values, pw, x, y + 1);
                        let s = at(&values, pw, x + 1, y + 2);
                        if [n, e, w, s].iter().any(|v| *v == 1 || *v == 2) {
                            3
                        } else {
                            centre
                        }
                    } else {
                        centre
                    }
                }
                EdgeMode::Special => {
                    let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
                    if centre != 0 && layer.next_int(&mut seed, 13) == 0 {
                        centre | ((1 + layer.next_int(&mut seed, 15)) << 8 & 3840)
                    } else {
                        centre
                    }
                }
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn deep_ocean(parent: &Node, area_x: i32, area_y: i32, width: i32, height: i32) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let neighbours = [
                at(&values, pw, x + 1, y),
                at(&values, pw, x + 2, y + 1),
                at(&values, pw, x, y + 1),
                at(&values, pw, x + 1, y + 2),
            ];
            let centre = at(&values, pw, x + 1, y + 1);
            let value = if centre == 0 && neighbours.iter().filter(|v| **v == 0).count() > 3 {
                24
            } else {
                centre
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn biome_layer(
    layer: &Node,
    parent: &Node,
    warm: WarmBiomes,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let values = parent.get_ints(area_x, area_y, width, height);
    let warm = warm.table();
    let temperate = [4, 29, 3, 1, 27, 6];
    let cool = [4, 3, 5, 1];
    let icy = [12, 12, 12, 30];
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let raw = at(&values, width, x, y);
            let climate = (raw & 3840) >> 8;
            let raw = raw & !3840;
            let value = if is_oceanic(raw) || raw == 14 {
                raw
            } else if raw == 1 {
                if climate > 0 {
                    if layer.next_int(&mut seed, 3) == 0 {
                        39
                    } else {
                        38
                    }
                } else {
                    warm[layer.next_int(&mut seed, warm.len() as i32) as usize]
                }
            } else if raw == 2 {
                if climate > 0 {
                    21
                } else {
                    temperate[layer.next_int(&mut seed, temperate.len() as i32) as usize]
                }
            } else if raw == 3 {
                if climate > 0 {
                    32
                } else {
                    cool[layer.next_int(&mut seed, cool.len() as i32) as usize]
                }
            } else if raw == 4 {
                icy[layer.next_int(&mut seed, icy.len() as i32) as usize]
            } else {
                14
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn biome_edge(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let centre = at(&values, pw, x + 1, y + 1);
            let n = at(&values, pw, x + 1, y);
            let e = at(&values, pw, x + 2, y + 1);
            let w = at(&values, pw, x, y + 1);
            let s = at(&values, pw, x + 1, y + 2);
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let value = if replace_edge_if_necessary(&values, &mut out, x, y, width, centre, 3, 20)
                || replace_edge(&values, &mut out, x, y, width, centre, 38, 37)
                || replace_edge(&values, &mut out, x, y, width, centre, 39, 37)
                || replace_edge(&values, &mut out, x, y, width, centre, 32, 5)
            {
                at(&out, width, x, y)
            } else if centre == 2 {
                if [n, e, w, s].iter().all(|v| *v != 12) {
                    centre
                } else {
                    34
                }
            } else if centre == 6 {
                let no_desert = [n, e, w, s].iter().all(|v| *v != 2);
                let no_cold = [n, e, w, s].iter().all(|v| *v != 30);
                let no_ice = [n, e, w, s].iter().all(|v| *v != 12);
                if no_desert && no_cold && no_ice {
                    if [n, e, w, s]
                        .iter()
                        .all(|value| !biome_edge_jungle_neighbour(*value))
                    {
                        centre
                    } else {
                        23
                    }
                } else {
                    1
                }
            } else {
                centre
            };
            let _ = &mut seed;
            // The replacement helpers above write the destination for their matching centre.
            // Ordinary branches write it here; keeping this assignment unconditional also
            // makes the control flow resistant to a future helper that returns a stale slot.
            set(&mut out, width, x, y, value);
        }
    }
    out
}

/// `GenLayerBiomeEdge` compares neighbours to the exact `BiomeGenBase.jungle`
/// singleton (ID 21).  This is intentionally narrower than `GenLayerShore`,
/// whose `instanceof BiomeGenJungle` check also accepts JungleHills/JungleEdge.
#[inline]
fn biome_edge_jungle_neighbour(id: i32) -> bool {
    id == 21
}

// These parameters mirror the Java layer's source/destination window and
// target/replacement values; keeping them separate preserves auditability.
#[allow(clippy::too_many_arguments)]
fn replace_edge_if_necessary(
    values: &[i32],
    out: &mut [i32],
    x: i32,
    y: i32,
    width: i32,
    centre: i32,
    target: i32,
    replacement: i32,
) -> bool {
    if !biomes_equal(centre, target) {
        return false;
    }
    let pw = width + 2;
    let neighbours = [
        at(values, pw, x + 1, y),
        at(values, pw, x + 2, y + 1),
        at(values, pw, x, y + 1),
        at(values, pw, x + 1, y + 2),
    ];
    let value = if neighbours
        .iter()
        .all(|neighbour| can_neighbour(*neighbour, target))
    {
        centre
    } else {
        replacement
    };
    set(out, width, x, y, value);
    true
}

#[allow(clippy::too_many_arguments)]
fn replace_edge(
    values: &[i32],
    out: &mut [i32],
    x: i32,
    y: i32,
    width: i32,
    centre: i32,
    target: i32,
    replacement: i32,
) -> bool {
    if centre != target {
        return false;
    }
    let pw = width + 2;
    let neighbours = [
        at(values, pw, x + 1, y),
        at(values, pw, x + 2, y + 1),
        at(values, pw, x, y + 1),
        at(values, pw, x + 1, y + 2),
    ];
    let value = if neighbours
        .iter()
        .all(|neighbour| biomes_equal(*neighbour, target))
    {
        centre
    } else {
        replacement
    };
    set(out, width, x, y, value);
    true
}

/// The default 1.8.9 shore layer. The class checks deliberately use the generated Java class
/// names: mutated biomes delegate `getBiomeClass()` to their base biome, just as vanilla does.
fn shore(parent: &Node, area_x: i32, area_y: i32, width: i32, height: i32) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let centre = at(&values, pw, x + 1, y + 1);
            let neighbours = [
                at(&values, pw, x + 1, y),
                at(&values, pw, x + 2, y + 1),
                at(&values, pw, x, y + 1),
                at(&values, pw, x + 1, y + 2),
            ];
            let value = if centre == 14 {
                if neighbours.iter().all(|value| *value != 0) {
                    centre
                } else {
                    15
                }
            } else if is_jungle(centre) {
                if neighbours.iter().all(|value| jungle_neighbour(*value)) {
                    if neighbours.iter().all(|value| !is_oceanic(*value)) {
                        centre
                    } else {
                        16
                    }
                } else {
                    23
                }
            } else if centre != 3 && centre != 34 && centre != 20 {
                if registered(centre) && mc_data::biome(centre as u8).snow {
                    shore_edge_value(&neighbours, centre, 26)
                } else if centre != 37 && centre != 38 {
                    if centre != 0 && centre != 24 && centre != 7 && centre != 6 {
                        if neighbours.iter().all(|value| !is_oceanic(*value)) {
                            centre
                        } else {
                            16
                        }
                    } else {
                        centre
                    }
                } else if neighbours.iter().all(|value| !is_oceanic(*value)) {
                    if neighbours.iter().all(|value| is_mesa(*value)) {
                        centre
                    } else {
                        2
                    }
                } else {
                    centre
                }
            } else {
                shore_edge_value(&neighbours, centre, 25)
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

#[inline]
fn shore_edge_value(neighbours: &[i32; 4], centre: i32, edge: i32) -> i32 {
    if is_oceanic(centre) || neighbours.iter().all(|value| !is_oceanic(*value)) {
        centre
    } else {
        edge
    }
}

#[inline]
fn jungle_neighbour(id: i32) -> bool {
    class_of(id) == Some("net.minecraft.world.biome.BiomeGenJungle")
        || id == 23
        || id == 21
        || id == 22
        || id == 4
        || id == 5
        || is_oceanic(id)
}

fn hills(
    layer: &Node,
    parent: &Node,
    river_init: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let river_values = river_init.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let k = at(&values, pw, x + 1, y + 1);
            let l = at(&river_values, pw, x + 1, y + 1);
            let flag = (l - 2).rem_euclid(29) == 0;
            // java: `if (k != 0 && l >= 2 && (l - 2) % 29 == 1 && k <
            // 128) { aint2 = getBiome(k + 128) != null ? k + 128 : k; }`. The registry test is
            // inside the branch: a cell whose mutation does not exist (deep ocean, 152) keeps
            // `k` and draws no `nextInt(3)`. Folding the test into the condition sent such a cell
            // on to the hills roll instead (a deep-ocean cell could become plains).
            let value = if k != 0 && l >= 2 && (l - 2).rem_euclid(29) == 1 && k < 128 {
                if registered(k + 128) {
                    k + 128
                } else {
                    k
                }
            } else if layer.next_int(&mut seed, 3) != 0 && !flag {
                k
            } else {
                let mut replacement = hill_variant(k, layer, &mut seed);
                if flag && replacement != k {
                    replacement = if registered(replacement + 128) {
                        replacement + 128
                    } else {
                        k
                    };
                }
                if replacement == k {
                    k
                } else {
                    let neighbours = [
                        at(&values, pw, x + 1, y),
                        at(&values, pw, x + 2, y + 1),
                        at(&values, pw, x, y + 1),
                        at(&values, pw, x + 1, y + 2),
                    ];
                    if neighbours.iter().filter(|v| biomes_equal(**v, k)).count() >= 3 {
                        replacement
                    } else {
                        k
                    }
                }
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn hill_variant(k: i32, layer: &Node, seed: &mut i64) -> i32 {
    match k {
        2 => 17,
        4 => 18,
        27 => 28,
        29 => 1,
        5 => 19,
        32 => 33,
        30 => 31,
        1 => {
            if layer.next_int(seed, 3) == 0 {
                18
            } else {
                4
            }
        }
        12 => 13,
        21 => 22,
        0 => 24,
        3 => 34,
        35 => 36,
        38 | 39 => 37,
        24 => {
            if layer.next_int(seed, 3) == 0 {
                if layer.next_int(seed, 2) == 0 {
                    1
                } else {
                    4
                }
            } else {
                24
            }
        }
        _ => k,
    }
}

fn river_init(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let values = parent.get_ints(area_x, area_y, width, height);
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let value = if at(&values, width, x, y) > 0 {
                layer.next_int(&mut seed, 299_999) + 2
            } else {
                0
            };
            set(&mut out, width, x, y, value);
        }
    }
    out
}

fn river(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let west = river_value(at(&values, pw, x, y + 1));
            let east = river_value(at(&values, pw, x + 2, y + 1));
            let north = river_value(at(&values, pw, x + 1, y));
            let south = river_value(at(&values, pw, x + 1, y + 2));
            let centre = river_value(at(&values, pw, x + 1, y + 1));
            set(
                &mut out,
                width,
                x,
                y,
                if centre == west && centre == north && centre == east && centre == south {
                    -1
                } else {
                    7
                },
            );
        }
    }
    let _ = layer;
    out
}

#[inline]
fn river_value(value: i32) -> i32 {
    if value >= 2 {
        2 + (value & 1)
    } else {
        value
    }
}

fn smooth(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let west = at(&values, pw, x, y + 1);
            let east = at(&values, pw, x + 2, y + 1);
            let north = at(&values, pw, x + 1, y);
            let south = at(&values, pw, x + 1, y + 2);
            let mut centre = at(&values, pw, x + 1, y + 1);
            if west == east && north == south {
                let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
                centre = if layer.next_int(&mut seed, 2) == 0 {
                    west
                } else {
                    north
                };
            } else {
                if west == east {
                    centre = west;
                }
                if north == south {
                    centre = north;
                }
            }
            set(&mut out, width, x, y, centre);
        }
    }
    out
}

fn rare_biome(
    layer: &Node,
    parent: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let pw = width + 2;
    let values = parent.get_ints(
        area_x.wrapping_sub(1),
        area_y.wrapping_sub(1),
        pw,
        height + 2,
    );
    let mut out = vec![0; output_size(width, height)];
    for y in 0..height {
        for x in 0..width {
            let mut seed = layer.chunk_seed(area_x.wrapping_add(x), area_y.wrapping_add(y));
            let value = at(&values, pw, x + 1, y + 1);
            set(
                &mut out,
                width,
                x,
                y,
                if layer.next_int(&mut seed, 57) == 0 && value == 1 {
                    129
                } else {
                    value
                },
            );
        }
    }
    out
}

fn river_mix(
    biome_layer: &Node,
    river_layer: &Node,
    area_x: i32,
    area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    let biomes = biome_layer.get_ints(area_x, area_y, width, height);
    let rivers = river_layer.get_ints(area_x, area_y, width, height);
    let mut out = vec![0; output_size(width, height)];
    for i in 0..out.len() {
        let b = biomes[i];
        let r = rivers[i];
        out[i] = if b != 0 && b != 24 {
            if r == 7 {
                if b == 12 {
                    11
                } else if b != 14 && b != 15 {
                    r & 255
                } else {
                    15
                }
            } else {
                b
            }
        } else {
            b
        };
    }
    out
}

fn voronoi(
    layer: &Node,
    parent: &Node,
    mut area_x: i32,
    mut area_y: i32,
    width: i32,
    height: i32,
) -> Vec<i32> {
    area_x = area_x.wrapping_sub(2);
    area_y = area_y.wrapping_sub(2);
    let px = area_x >> 2;
    let py = area_y >> 2;
    let pw = (width >> 2) + 2;
    let ph = (height >> 2) + 2;
    let values = parent.get_ints(px, py, pw, ph);
    let zoom_width = (pw - 1) << 2;
    let zoom_height = (ph - 1) << 2;
    let mut zoomed = vec![0; output_size(zoom_width, zoom_height)];
    for y in 0..(ph - 1) {
        let mut west = at(&values, pw, 0, y);
        let mut south_west = at(&values, pw, 0, y + 1);
        for x in 0..(pw - 1) {
            let mut seed = layer.chunk_seed((x + px).wrapping_mul(4), (y + py).wrapping_mul(4));
            let d1 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6;
            let d2 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6;
            let mut seed = layer.chunk_seed((x + px + 1).wrapping_mul(4), (y + py).wrapping_mul(4));
            let d3 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6 + 4.0;
            let d4 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6;
            let mut seed = layer.chunk_seed((x + px).wrapping_mul(4), (y + py + 1).wrapping_mul(4));
            let d5 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6;
            let d6 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6 + 4.0;
            let mut seed =
                layer.chunk_seed((x + px + 1).wrapping_mul(4), (y + py + 1).wrapping_mul(4));
            let d7 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6 + 4.0;
            let d8 = (f64::from(layer.next_int(&mut seed, 1024)) / 1024.0 - 0.5) * 3.6 + 4.0;
            let east = at(&values, pw, x + 1, y) & 255;
            let south_east = at(&values, pw, x + 1, y + 1) & 255;
            for iy in 0..4 {
                for ix in 0..4 {
                    let a = f64::from(iy) - d2;
                    let b = f64::from(ix) - d1;
                    let d9 = a * a + b * b;
                    let a = f64::from(iy) - d4;
                    let b = f64::from(ix) - d3;
                    let d10 = a * a + b * b;
                    let a = f64::from(iy) - d6;
                    let b = f64::from(ix) - d5;
                    let d11 = a * a + b * b;
                    let a = f64::from(iy) - d8;
                    let b = f64::from(ix) - d7;
                    let d12 = a * a + b * b;
                    let value = if d9 < d10 && d9 < d11 && d9 < d12 {
                        west
                    } else if d10 < d9 && d10 < d11 && d10 < d12 {
                        east
                    } else if d11 < d9 && d11 < d10 && d11 < d12 {
                        south_west
                    } else {
                        south_east
                    };
                    set(&mut zoomed, zoom_width, x * 4 + ix, y * 4 + iy, value);
                }
            }
            west = east;
            south_west = south_east;
        }
    }
    let mut out = vec![0; output_size(width, height)];
    let xoff = area_x & 3;
    let yoff = area_y & 3;
    for y in 0..height {
        let source_start = ((y + yoff) * zoom_width + xoff) as usize;
        let dest_start = (y * width) as usize;
        out[dest_start..dest_start + width as usize]
            .copy_from_slice(&zoomed[source_start..source_start + width as usize]);
    }
    out
}

#[inline]
fn registered(id: i32) -> bool {
    (0..=255).contains(&id) && biome(id as u8).id == id as u8
}

#[inline]
fn canonical_id(id: i32) -> Option<i32> {
    if registered(id) {
        Some(id)
    } else if id >= 128 && registered(id - 128) {
        Some(id - 128)
    } else {
        None
    }
}

/// java: `BiomeGenBase.getBiomeClass()`, which the layers compare (`isEqualTo`, the jungle
/// checks, `getTempCategory`).
///
/// mc-data records a mutated biome's own Java class, and for most
/// mutations that is `BiomeGenMutated`, whose `getBiomeClass()` returns `baseBiome
/// .getBiomeClass()`: Jungle M (149) answers `BiomeGenJungle`. Reading the raw class made a
/// jungle next to Jungle M a jungle edge (the large-biomes Java samples caught it). The mesa
/// mutations are real `BiomeGenMesa` instances and keep their own class.
#[inline]
fn class_of(id: i32) -> Option<&'static str> {
    let id = canonical_id(id)?;
    let class = biome(id as u8).java_class;
    if class == "net.minecraft.world.biome.BiomeGenMutated" && id >= 128 {
        return canonical_id(id - 128).map(|base| biome(base as u8).java_class);
    }
    Some(class)
}

#[inline]
fn biomes_equal(a: i32, b: i32) -> bool {
    if a == b {
        return true;
    }
    if a != 38 && a != 39 {
        class_of(a).is_some() && class_of(a) == class_of(b)
    } else {
        b == 38 || b == 39
    }
}

#[inline]
fn is_oceanic(id: i32) -> bool {
    id == 0 || id == 10 || id == 24
}

#[inline]
// java: `GenLayerShore.func_151633_d` is `getBiome(id) instanceof BiomeGenMesa`.
// The three mesa mutations (165..=167) satisfy it: `BiomeGenMesa` overrides
// `createMutatedBiome` to build another `BiomeGenMesa` (`BiomeGenMesa.java:317-331`), not a
// `BiomeGenMutated`. Excluding them turned a mesa next to a mutated mesa into desert (the
// large-biomes Java fixture caught four such cells).
fn is_mesa(id: i32) -> bool {
    matches!(id, 37..=39 | 165..=167)
}

#[inline]
fn is_jungle(id: i32) -> bool {
    class_of(id) == Some("net.minecraft.world.biome.BiomeGenJungle")
        || id == 23
        || id == 22
        || id == 21
}

fn can_neighbour(a: i32, b: i32) -> bool {
    if biomes_equal(a, b) {
        return true;
    }
    if !registered(a) || !registered(b) {
        return false;
    }
    // TempCategory is based on the biome's nominal temperature, not the current block height.
    // Mutated rows copy the base temperature in BiomeGenMutated, so the generated table is a
    // direct equivalent of BiomeGenBase.getTempCategory here.
    let ca = temp_category(a);
    let cb = temp_category(b);
    ca == cb || ca == TempCategory::Medium || cb == TempCategory::Medium
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TempCategory {
    Ocean,
    Cold,
    Medium,
    Warm,
}

fn temp_category(id: i32) -> TempCategory {
    // BiomeGenOcean overrides BiomeGenBase.getTempCategory and returns OCEAN for
    // ocean, frozen-ocean, and deep-ocean (including any mutated row that delegates
    // its category to an ocean base).  It must remain distinct from MEDIUM: Java's
    // GenLayerBiomeEdge only treats MEDIUM as a bridge between otherwise incompatible
    // temperature categories.
    if class_of(id) == Some("net.minecraft.world.biome.BiomeGenOcean") {
        return TempCategory::Ocean;
    }
    let temperature = mc_data::biome(id as u8).temperature;
    if temperature < 0.2 {
        TempCategory::Cold
    } else if temperature < 1.0 {
        TempCategory::Medium
    } else {
        TempCategory::Warm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    fn test_clock() -> (Arc<AtomicI64>, CacheClock) {
        let now = Arc::new(AtomicI64::new(0));
        let clock_now = Arc::clone(&now);
        let clock: CacheClock = Arc::new(move || clock_now.load(Ordering::SeqCst));
        (now, clock)
    }

    #[test]
    fn ocean_biomes_use_java_ocean_temp_category() {
        for id in [0, 10, 24] {
            assert_eq!(temp_category(id), TempCategory::Ocean, "biome id {id}");
        }
        assert_eq!(temp_category(12), TempCategory::Cold);
        assert_eq!(temp_category(1), TempCategory::Medium);
        assert_eq!(temp_category(2), TempCategory::Warm);
    }

    #[test]
    fn ocean_neighbor_controls_match_java_categories() {
        // OCEAN is not MEDIUM, so it cannot bridge cold or warm neighbors.  It does
        // still neighbor another ocean (same Java BiomeGenOcean class) and MEDIUM.
        assert!(can_neighbour(0, 10));
        assert!(can_neighbour(24, 1));
        assert!(!can_neighbour(10, 12));
        assert!(!can_neighbour(0, 2));
        assert!(!can_neighbour(24, 30));
    }

    #[test]
    fn seeded_provider_edge_sample_matches_java() {
        let provider = OverworldBiomeProvider::new(0);
        assert_eq!(provider.biome_at(-4_097, 8_193), 0);
        assert_eq!(provider.biome_at(-2_147_483_648, 0), 24);
    }

    #[test]
    fn provider_is_deterministic_at_negative_and_far_coordinates() {
        let provider = OverworldBiomeProvider::new(0);
        let samples = [
            (0, 0, 4),
            (-1, -1, 4),
            (1, 1, 4),
            (15, 15, 4),
            (16, 16, 4),
            (-16, -16, 4),
            (-17, -17, 4),
            (-65, 79, 1),
            (-4097, 8193, 0),
            (1_000_000, -1_000_000, 16),
            (2_147_000_000, -2_147_000_000, 0),
            (-2_147_483_648, -2_147_483_648, 30),
            (2_147_483_647, 2_147_483_647, 30),
            (0, 1_000_000, 3),
            (1_000_000, 0, 0),
            (-2_147_483_648, 0, 24),
            (0, -2_147_483_648, 24),
            (-2_147_483_647, -2_147_483_647, 30),
            (-2_147_483_640, -2_147_483_640, 6),
            (2_147_483_640, 2_147_483_640, 30),
            (2_147_483_647, 0, 24),
            (0, 2_147_483_647, 24),
        ];
        for (x, z, expected) in samples {
            assert_eq!(provider.biome_at(x, z), expected, "sample ({x},{z})");
            assert_eq!(provider.biome_at(x, z), expected, "second pass ({x},{z})");
        }
    }

    #[test]
    fn biome_cache_reuses_full_chunk_for_negative_coordinates_and_rainfall() {
        let (_now, clock) = test_clock();
        let provider = OverworldBiomeProvider::with_clock(0, clock);
        let chunk_x = -2;
        let chunk_z = -3;
        let origin_x = chunk_x * BIOME_CACHE_WIDTH;
        let origin_z = chunk_z * BIOME_CACHE_WIDTH;
        let raw = provider
            .layer
            .get_ints(origin_x, origin_z, BIOME_CACHE_WIDTH, BIOME_CACHE_WIDTH);

        let x = -17;
        let z = -33;
        let index = cache_index(x, z);
        let expected = canonical_generated_biome(raw[index]);
        assert_eq!(provider.biome_at(x, z), expected);
        assert_eq!(provider.rainfall_at(x, z), java_rainfall(expected));

        // -31/-47 remain in the same Java `>> 4` chunk as -17/-33.  This must be a hit, not a
        // second generation, and the key must remain distinct from the positive chunk at zero.
        let second_index = cache_index(-31, -47);
        assert_eq!(
            provider.biome_at(-31, -47),
            canonical_generated_biome(raw[second_index])
        );
        assert_eq!(provider.biome_at(0, 0), {
            let positive = provider
                .layer
                .get_ints(0, 0, BIOME_CACHE_WIDTH, BIOME_CACHE_WIDTH);
            canonical_generated_biome(positive[0])
        });

        let stats = provider.cache_stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.rebuilds, 2);
        assert_eq!(
            stats.hits, 2,
            "rainfall and second negative lookup share the first block"
        );
    }

    #[test]
    fn providers_share_one_seeded_graph_without_changing_a_biome_or_sharing_a_cache() {
        let (_now, clock) = test_clock();
        let provider = OverworldBiomeProvider::with_clock(0, clock);
        let other = OverworldBiomeProvider::new(0);
        assert!(
            Arc::ptr_eq(&provider.layer, &other.layer),
            "the same (seed, biome size) must reuse one graph instead of rebuilding it"
        );
        assert!(
            Arc::ptr_eq(&provider.layer, &provider.clone().layer),
            "a render-snapshot clone must not deep-copy the graph"
        );
        let large = OverworldBiomeProvider::new_for_world_type(0, "largeBiomes");
        assert!(!Arc::ptr_eq(&provider.layer, &large.layer));
        let legacy = OverworldBiomeProvider::new_for_world_type(0, "default_1_1");
        assert!(!Arc::ptr_eq(&provider.layer, &legacy.layer));
        assert!(!Arc::ptr_eq(&large.layer, &legacy.layer));

        // A privately built graph, which is how every provider used to be made, answers
        // identically: sharing is storage, not a different generator. Chunk-sized areas, as
        // vanilla asks for: a 3-wide area at an unaligned x would run the voronoi copy past its
        // 4x4 working grid, which Java's 256-entry `IntCache` arrays only hide.
        let mut fresh = default_layer_graph(4, WarmBiomes::Default);
        fresh.init_world_gen_seed(0);
        for (x, z) in [(0, 0), (-17, -33), (1_234, -5_678), (-100_000, 77_777)] {
            assert_eq!(
                provider.layer.get_ints(x, z, 16, 16),
                fresh.get_ints(x, z, 16, 16),
                "({x},{z})"
            );
        }

        // Only the graph is shared. Independently built providers keep independent caches.
        let _ = provider.biome_at(0, 0);
        assert_eq!(provider.cache_stats().entries, 1);
        assert_eq!(other.cache_stats().entries, 0);
    }

    #[test]
    fn biome_cache_keeps_fresh_blocks_and_rebuilds_idle_blocks() {
        let (now, clock) = test_clock();
        let provider = OverworldBiomeProvider::with_clock(0, clock);
        let initial = provider.biome_at(0, 0);
        assert_eq!(provider.cache_stats().rebuilds, 1);

        now.store(BIOME_CACHE_IDLE_TIMEOUT_MS, Ordering::SeqCst);
        provider.cleanup_cache();
        assert_eq!(provider.biome_at(0, 0), initial);
        let boundary = provider.cache_stats();
        assert_eq!(
            boundary.evictions, 0,
            "Java uses a strict > 30-second idle bound"
        );
        assert_eq!(boundary.rebuilds, 1);

        now.store(BIOME_CACHE_IDLE_TIMEOUT_MS * 2 + 1, Ordering::SeqCst);
        provider.cleanup_cache();
        assert_eq!(provider.cache_stats().evictions, 1);
        assert_eq!(provider.biome_at(0, 0), initial);
        let rebuilt = provider.cache_stats();
        assert_eq!(rebuilt.rebuilds, 2);
        assert_eq!(rebuilt.entries, 1);
    }

    #[test]
    fn biome_cache_clock_reversal_evicts_old_entries_without_global_time() {
        let (now, clock) = test_clock();
        let provider = OverworldBiomeProvider::with_clock(0, clock);
        let _ = provider.biome_at(-1, -1);
        now.store(10_000, Ordering::SeqCst);
        let _ = provider.biome_at(-1, -1);
        assert_eq!(provider.cache_stats().rebuilds, 1);

        now.store(9_000, Ordering::SeqCst);
        provider.cleanup_cache();
        let stats = provider.cache_stats();
        assert_eq!(
            stats.evictions, 1,
            "Java removes entries whose access time is in the future"
        );
        assert_eq!(stats.entries, 0);
        let _ = provider.biome_at(-1, -1);
        assert_eq!(provider.cache_stats().rebuilds, 2);
    }

    /// Inside `BIOME_BLOCK_QUERY_MAX_ABS` one 16x16 request returns exactly
    /// the ids of the 256 one-cell requests the cache used to make, so filling a block either way
    /// is the same block: random blocks (both signs, near and far), the SkyWars bench's spawn,
    /// the world border and the range's own edges, on both world types' graphs.
    #[test]
    fn a_whole_block_request_equals_one_cell_requests_inside_the_exact_range() {
        let per_cell = |layer: &Node, ox: i32, oz: i32| -> Vec<i32> {
            let mut out = Vec::with_capacity(BIOME_CACHE_AREA);
            for local_z in 0..BIOME_CACHE_WIDTH {
                for local_x in 0..BIOME_CACHE_WIDTH {
                    out.push(layer.get_ints(ox + local_x, oz + local_z, 1, 1)[0]);
                }
            }
            out
        };
        let edge = BIOME_BLOCK_QUERY_MAX_ABS;
        let mut origins = vec![
            (0, 0),
            (-16, -16),
            (-3_904, 2_048),
            (29_999_984, -30_000_000),
            (-30_000_000, 29_999_984),
            (edge, edge),
            (-edge, -edge),
            (edge - 16, -edge),
        ];
        let mut state: u64 = 0x5EED_F5F5;
        for _ in 0..40 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = ((state >> 33) as i32 % (edge / 16)) * 16;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let z = ((state >> 33) as i32 % (edge / 16)) * 16;
            // Most players are near spawn: half the samples within 4096 blocks.
            if origins.len() % 2 == 0 {
                origins.push((x % 4_096, z % 4_096));
            } else {
                origins.push((x, z));
            }
        }
        for provider in [
            OverworldBiomeProvider::new(0),
            OverworldBiomeProvider::new_for_world_type(0, "largeBiomes"),
            OverworldBiomeProvider::new_for_world_type(0, "default_1_1"),
        ] {
            for &(ox, oz) in &origins {
                assert!(
                    biome_block_query_is_exact(ox, oz),
                    "({ox},{oz}) is in range"
                );
                let whole = provider
                    .layer
                    .get_ints(ox, oz, BIOME_CACHE_WIDTH, BIOME_CACHE_WIDTH);
                assert_eq!(
                    whole,
                    per_cell(&provider.layer, ox, oz),
                    "block at ({ox},{oz})"
                );
            }
        }
        assert!(!biome_block_query_is_exact(edge + 16, 0));
        assert!(!biome_block_query_is_exact(0, -edge - 16));
        assert!(!biome_block_query_is_exact(i32::MIN, 0));

        // Negative control: at the `i32` edge the two request shapes really do disagree, so the
        // comparison above can see a difference and the range guard is needed.
        let provider = OverworldBiomeProvider::new(0);
        let near_edge = [
            i32::MIN,
            i32::MIN + 16,
            i32::MIN + 64,
            i32::MAX - 15,
            i32::MAX - 31,
        ];
        let edge_blocks: Vec<(i32, i32)> = near_edge
            .iter()
            .flat_map(|&x| near_edge.iter().map(move |&z| (x, z)))
            .chain(near_edge.iter().flat_map(|&v| [(v, 0), (0, v)]))
            .collect();
        let differing = edge_blocks
            .iter()
            .filter(|&&(ox, oz)| {
                let whole = provider
                    .layer
                    .get_ints(ox, oz, BIOME_CACHE_WIDTH, BIOME_CACHE_WIDTH);
                let cells: Vec<i32> = (0..BIOME_CACHE_WIDTH)
                    .flat_map(|dz| (0..BIOME_CACHE_WIDTH).map(move |dx| (dx, dz)))
                    .map(|(dx, dz)| {
                        provider
                            .layer
                            .get_ints(ox.wrapping_add(dx), oz.wrapping_add(dz), 1, 1)[0]
                    })
                    .collect();
                whole != cells
            })
            .count();
        assert!(
            differing > 0,
            "the edge blocks must show the overflow difference"
        );
    }

    /// The cache fills a block from one request inside the range, and the
    /// provider's answers are the ones the per-cell fill gives (the sample list of
    /// `provider_is_deterministic_at_negative_and_far_coordinates` covers the `i32` edge, which
    /// stays on one-cell requests).
    #[test]
    fn cached_lookups_match_the_per_cell_fill_near_spawn() {
        let provider = OverworldBiomeProvider::new(0);
        for (x, z) in [
            (-3_892, 2_059),
            (-3_860, 2_100),
            (0, 0),
            (-1, -1),
            (777, -12_345),
        ] {
            let raw = provider.layer.get_ints(x, z, 1, 1)[0];
            assert_eq!(
                provider.biome_at(x, z),
                canonical_generated_biome(raw),
                "({x},{z})"
            );
        }
    }

    /// The client's seed-0 fallback for each packet world type, against
    /// the real 1.8.9 `WorldChunkManager(0L, type, "").getBiomeGenerator(pos)` on a 33x33 grid
    /// (`javaref/BiomeTypeRefGen.java`, `generate-biome-types.ps1`). DEFAULT_1_1 differs from
    /// DEFAULT at 164 of the 1,089 samples (warm cells and everything later layers derive from
    /// them); the generator also checks that CUSTOMIZED and AMPLIFIED with a client's empty
    /// options equal DEFAULT. Negative control: before the legacy warm table, every one of those
    /// 164 DEFAULT_1_1 samples failed.
    #[test]
    fn packet_world_type_fallbacks_match_java_world_chunk_manager() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/biome/world_types_seed0.txt"
        ));
        let (samples, legacy_differs) = check_world_type_samples(fixture);
        assert_eq!(samples, 1089);
        assert_eq!(
            legacy_differs, 164,
            "the fixture exercises the legacy table"
        );
    }

    /// The same check over extra `BiomeTypeRefGen ... random <count> <range>` files named in
    /// `MC_W18_BIOME_SAMPLES` (`;`-separated paths); skipped when unset.
    #[test]
    #[ignore = "reads MC_W18_BIOME_SAMPLES; run with --ignored"]
    fn packet_world_type_fallbacks_match_extra_java_samples() {
        let Ok(paths) = std::env::var("MC_W18_BIOME_SAMPLES") else {
            eprintln!("skip: MC_W18_BIOME_SAMPLES is not set");
            return;
        };
        for path in paths.split(';').filter(|path| !path.is_empty()) {
            let text = std::fs::read_to_string(path).expect("sample file");
            let (samples, legacy_differs) = check_world_type_samples(&text);
            eprintln!("{path}: {samples} samples, {legacy_differs} differ for default_1_1");
            assert!(samples > 0);
        }
    }

    /// Check every `x z default default_1_1 largeBiomes` line; returns the sample count and how
    /// many samples the legacy table changes.
    fn check_world_type_samples(fixture: &str) -> (usize, usize) {
        let providers = [
            (
                "default",
                OverworldBiomeProvider::new_for_world_type(0, "default"),
            ),
            (
                "default_1_1",
                OverworldBiomeProvider::new_for_world_type(0, "default_1_1"),
            ),
            (
                "largeBiomes",
                OverworldBiomeProvider::new_for_world_type(0, "largeBiomes"),
            ),
        ];
        let mut samples = 0;
        let mut legacy_differs = 0;
        let mut mismatches = Vec::new();
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let values: Vec<i32> = line
                .split_whitespace()
                .map(|value| value.parse().expect("fixture integer"))
                .collect();
            let [x, z, default, legacy, large] = values[..] else {
                panic!("fixture line {line:?}");
            };
            for ((name, provider), expected) in providers.iter().zip([default, legacy, large]) {
                let got = i32::from(provider.biome_at(x, z));
                if got != expected {
                    mismatches.push(format!("{name} at ({x},{z}): rust {got}, java {expected}"));
                }
            }
            samples += 1;
            legacy_differs += usize::from(default != legacy);
        }
        assert!(
            mismatches.is_empty(),
            "{} mismatches:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
        (samples, legacy_differs)
    }
}
