//! The client [`World`]: chunk storage, block/light access, and the `hecs` entity arena.
//!
//! java: `world/World.java`, `client/multiplayer/WorldClient.java`,
//! `client/multiplayer/ChunkProviderClient.java`
//!
//! `World.java` is exactly the cyclic mutable graph the arena-index rule exists to avoid: it hands out
//! live references to chunks, entities, and itself in every direction. This is read for
//! *behaviour* only — [`World`] here is a plain struct with no back-references, entities are
//! `hecs::Entity` arena indices, and every system that mutates it takes `&mut World`.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

use mc_data::block::Property;
use mc_data::{can_catch_fire, BlockId, BlockState, Material};
use mc_math::{floor_f64, Aabb, BlockPos, ChunkPos, ChunkSectionPos, Facing, JavaRandom};

use crate::chunk::Chunk;
use crate::entity::EntityBundle;
use crate::genlayer::{java_rainfall, OverworldBiomeProvider};
use crate::map::{MapData, MapUpdateError};
use crate::WorldError;

// Exact world copies for replay checkpoints. A child of this module so
// it can copy every private field; the file sits next to this one.
#[path = "snapshot.rs"]
pub(crate) mod snapshot;

const DEBUG_WORLD_LEVEL_TYPE: &str = "debug_all_block_states";
const DEBUG_WORLD_BARRIER: BlockState = BlockState::new(BlockId(166), 0);

/// The DEBUG_WORLD tuple table is immutable after generation.  Share one expanded table across
/// all client worlds and render snapshots: a snapshot must copy the Arc handle, not allocate and
/// clone 7,806 property vectors just because `World::render_snapshot` starts from `World::new`.
static DEBUG_WORLD_STATES: OnceLock<Arc<[mc_data::DebugWorldState]>> = OnceLock::new();

#[inline]
fn shared_debug_world_states() -> Arc<[mc_data::DebugWorldState]> {
    DEBUG_WORLD_STATES
        .get_or_init(|| Arc::from(mc_data::block::debug_world_states().collect::<Vec<_>>()))
        .clone()
}

// java: BiomeGenBase.temperatureNoise = new NoiseGeneratorPerlin(new Random(1234L), 1).
// Keep only the two-dimensional, single-sample path used by getFloatTemperature.  The
// generator is fixed by the literal seed and is not tied to the client world's (always absent)
// world seed.
struct TemperatureNoise {
    permutation: [u8; 512],
}

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

impl TemperatureNoise {
    /// java: `NoiseGeneratorSimplex(Random)`, including the three offset draws that precede the
    /// permutation shuffle even though the 2-D sample path never reads those offsets.
    fn new(rng: &mut JavaRandom) -> Self {
        let _x_offset = rng.next_f64() * 256.0;
        let _y_offset = rng.next_f64() * 256.0;
        let _z_offset = rng.next_f64() * 256.0;
        let mut permutation = [0; 512];
        for (i, slot) in permutation.iter_mut().take(256).enumerate() {
            *slot = i as u8;
        }
        for l in 0..256usize {
            // java: `int j = rand.nextInt(256 - l) + l;` then swap, then mirror into `l + 256`.
            let j = rng.next_i32_bound(256 - l as i32) as usize + l;
            permutation.swap(l, j);
            permutation[l + 256] = permutation[l];
        }
        Self { permutation }
    }

    /// java: `NoiseGeneratorSimplex.func_151607_a` deliberately maps exact zero to `-1`.
    #[inline]
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

fn temperature_noise() -> &'static TemperatureNoise {
    static NOISE: std::sync::OnceLock<TemperatureNoise> = std::sync::OnceLock::new();
    NOISE.get_or_init(|| TemperatureNoise::new(&mut JavaRandom::new(1234)))
}

/// The fixed Java `BiomeGenBase.temperatureNoise` contribution to one world column.
///
/// java: `getFloatTemperature` samples its one-octave `NoiseGeneratorPerlin` at `x / 8.0`
/// and `z / 8.0`, then multiplies the `double` result by `4.0` before the `float` height
/// falloff. The `mc_data::Biome::temperature_at` caller supplies this term only above `y = 64`.
#[inline]
fn temperature_noise_at(x: i32, z: i32) -> f32 {
    (temperature_noise().sample(f64::from(x) / 8.0, f64::from(z) / 8.0) * 4.0) as f32
}

/// `hecs::Entity` is the arena index the arena-index rule requires: opaque, `Copy`, and meaningless
/// once its generation is recycled by `despawn`. Re-exported here so callers never need a
/// direct `hecs` dependency just to hold one.
pub type Entity = hecs::Entity;

/// Return the canonical 1.8.9 spelling for a packet world type.
///
/// java: `WorldType.parseWorldType` followed by the `S01`/`S07` packet readers' fallback to
/// `WorldType.DEFAULT`. The protocol crate already performs this normalization when it decodes
/// wire packets; keeping the small boundary helper here makes direct lifecycle callers obey the
/// same rule without trusting an unnormalised synthetic packet.
#[must_use]
pub fn canonical_level_type_name(level_type: &str) -> &'static str {
    const WORLD_TYPES: [&str; 7] = [
        "default",
        "flat",
        "largeBiomes",
        "amplified",
        "customized",
        "debug_all_block_states",
        "default_1_1",
    ];

    WORLD_TYPES
        .into_iter()
        .find(|name| level_type.eq_ignore_ascii_case(name))
        .unwrap_or("default")
}

/// The client-side biome source selected by the current dimension.
///
/// java: `WorldProvider.registerWorldChunkManager` and the three vanilla
/// `WorldChunkManager` implementations.  Resident chunk biome bytes always win;
/// this source is consulted only for the same cases where vanilla's
/// `World.getBiomeGenForCoords` leaves the chunk path: an absent column or a
/// position outside the valid `y = 0..256` range.
///
/// `OverworldSeed0` is deliberately named for the client world's actual seed:
/// `NetHandlerPlayClient.handleJoinGame` constructs the client `WorldInfo` with
/// literal seed `0L`. Its exact 1.8.9 GenLayer graph is kept in this crate so immutable
/// terrain/sky readers do not silently turn a generated answer into Plains. Servers'
/// resident biome arrays still take precedence over it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BiomeSource {
    /// The client-side default/overworld source (seed `0L`).
    #[default]
    OverworldSeed0,
    /// `WorldProvider.registerWorldChunkManager`'s FLAT branch on a multiplayer client.
    ///
    /// `NetHandlerPlayClient` constructs the client `WorldInfo` with empty generator options;
    /// Java's `FlatGeneratorInfo.createFlatGeneratorFromString("")` therefore resolves to the
    /// default flat preset, whose fixed biome is Plains (id `1`). Server-supplied resident
    /// biome bytes still take precedence over this fallback.
    FlatPlains,
    /// `WorldProvider.registerWorldChunkManager`'s DEBUG branch, whose fixed biome is Plains
    /// (id `1`). This represents only the biome manager: DEBUG's server-side block-grid
    /// synthesis is not a client fallback and remains outside this world storage layer.
    DebugPlains,
    /// `WorldProviderHell`'s constant `BiomeGenBase.hell` (id `8`).
    Nether,
    /// `WorldProviderEnd`'s constant `BiomeGenBase.sky` (id `9`).
    End,
}

/// One chunk-column entry in an `S26PacketMapChunkBulk`-shaped batch.
///
/// A tiny "how S06's future packet type maps onto our unpack function" seam: once
/// `mc-protocol` lands, `S26PacketMapChunkBulk`'s per-chunk fields slot straight into this
/// struct's fields with no translation.
#[derive(Debug, Clone, Copy)]
pub struct BulkChunkEntry<'a> {
    /// Chunk column X.
    pub chunk_x: i32,
    /// Chunk column Z.
    pub chunk_z: i32,
    /// Section bitmask (`dataSize` in the Java).
    pub primary_bitmask: u16,
    /// This chunk's slice of the packet's raw byte payload.
    pub data: &'a [u8],
}

/// One client-local side effect a block activation asks its caller to play.
///
/// Java's `World.playAuxSFXAtEntity` reaches `RenderGlobal.playAuxSFX` immediately even on a
/// multiplayer client.  `mc-world` deliberately cannot own the audio mixer, so an activation
/// reports the exact level-event tuple to the application layer instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalBlockEffect {
    /// The `playAuxSFX` id, e.g. `1003` (door open) or `1006` (door close).
    pub id: i32,
    /// The block position passed to `playAuxSFXAtEntity`.
    pub pos: BlockPos,
    /// The event payload.  Trapdoors pass zero.
    pub data: i32,
}

/// The client-visible result of a block's `onBlockActivated` body.
///
/// `consumed` is `PlayerControllerMP.onPlayerRightClick`'s boolean: it suppresses item use and
/// makes the caller swing the arm.  The optional effect must be sent to the local audio/effects
/// fan-out on the same click, before the server's later block-change packet arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalBlockActivation {
    /// Java `Block.onBlockActivated`'s return value.
    pub consumed: bool,
    /// The local `World.playAuxSFXAtEntity` request, when this activation has one.
    pub effect: Option<LocalBlockEffect>,
}

/// The fields stored alongside a client-created `map_<id>` record.
///
/// java: `MapData.xCenter`, `MapData.zCenter`, and `MapData.dimension`. Multiplayer clients use
/// `SaveDataMemoryStorage`, whose `getUniqueDataId("map")` always returns `0`; the wire map packet
/// carries only the pixels/scale/icons, so these creation-time fields live beside [`MapData`]
/// rather than being reconstructed when an S34 update arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapDataMetadata {
    /// `MapData.xCenter`, calculated from the player's X at scale zero.
    pub x_center: i32,
    /// `MapData.zCenter`, calculated from the player's Z at scale zero.
    pub z_center: i32,
    /// `MapData.dimension`, Java's signed byte cast of the provider dimension id.
    pub dimension: i8,
}

/// Java `EnumParticleTypes` ids used by the client-side ambient block callbacks.
///
/// Keeping these wire ids in `mc-world` avoids a dependency cycle with `mc-render`; the client
/// maps each recipe into the existing particle factory after applying its ordinary density and
/// range gates.
pub mod ambient_particle {
    /// `EnumParticleTypes.SUSPENDED`.
    pub const SUSPENDED: i32 = 7;
    /// `EnumParticleTypes.SMOKE_NORMAL`.
    pub const SMOKE_NORMAL: i32 = 11;
    /// `EnumParticleTypes.SMOKE_LARGE`.
    pub const SMOKE_LARGE: i32 = 12;
    /// `EnumParticleTypes.DRIP_WATER`.
    pub const DRIP_WATER: i32 = 18;
    /// `EnumParticleTypes.DRIP_LAVA`.
    pub const DRIP_LAVA: i32 = 19;
    /// `EnumParticleTypes.TOWN_AURA`.
    pub const TOWN_AURA: i32 = 22;
    /// `EnumParticleTypes.PORTAL`.
    pub const PORTAL: i32 = 24;
    /// `EnumParticleTypes.ENCHANTMENT_TABLE`.
    pub const ENCHANTMENT_TABLE: i32 = 25;
    /// `EnumParticleTypes.FLAME`.
    pub const FLAME: i32 = 26;
    /// `EnumParticleTypes.LAVA`.
    pub const LAVA: i32 = 27;
    /// `EnumParticleTypes.REDSTONE`.
    pub const REDSTONE: i32 = 30;
    /// `EnumParticleTypes.BARRIER`.
    pub const BARRIER: i32 = 35;
    /// `EnumParticleTypes.WATER_DROP`.
    pub const WATER_DROP: i32 = 39;
}

/// One particle recipe emitted by a block's `randomDisplayTick` callback.
///
/// `velocity` is the callback's three `spawnParticle` speed arguments. For redstone dust Java
/// interprets the same triple as RGB, so it intentionally remains in this field and is not
/// normalised or fed through a generic velocity helper here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientParticleEffect {
    /// Java `EnumParticleTypes` wire id.
    pub particle_id: i32,
    /// Exact world-space spawn position.
    pub position: mc_math::DVec3,
    /// Java `xSpeed`, `ySpeed`, and `zSpeed` triple.
    pub velocity: mc_math::DVec3,
}

/// One positional sound request emitted by a block's `randomDisplayTick` callback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientSoundEffect {
    /// Resource key passed to `WorldClient.playSound`.
    pub name: &'static str,
    /// Exact world-space sound origin.
    pub position: mc_math::DVec3,
    /// Java volume argument.
    pub volume: f32,
    /// Java pitch argument.
    pub pitch: f32,
}

/// The two effect kinds a client-side random block display tick can produce.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmbientEffect {
    /// A particle spawn request.
    Particle(AmbientParticleEffect),
    /// A positional sound request.
    Sound(AmbientSoundEffect),
}

/// Number of `doVoidFogParticles` samples vanilla performs per client tick.
pub const RANDOM_DISPLAY_TICK_SAMPLES: usize = 1_000;

/// Exclusive half-width of the `nextInt(16) - nextInt(16)` sample range.
pub const RANDOM_DISPLAY_TICK_RADIUS: i32 = 16;

/// One client-local result of `Block.onBlockClicked` on the first left-click of a new dig.
///
/// The Java callback is a virtual block operation, not another digging packet.  The world owns
/// only the block/state half here; the application layer owns player food/achievement state and
/// particle/audio fan-out.  In particular, `NoteBlock` is intentionally absent: its callback is
/// guarded by `!world.isRemote`, so a multiplayer client performs no local note action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalBlockClick {
    /// `BlockRedstoneOre.activate`: six redstone particles and, for unlit ore, an ore-to-lit state
    /// prediction.  The particle fan-out remains an application/render concern.
    RedstoneOre {
        /// The clicked ore position.
        pos: BlockPos,
    },
    /// `BlockDragonEgg.teleport`: the remote-world branch emits 128 portal particles after its
    /// random destination search, without moving the egg locally.
    DragonEgg {
        /// The clicked egg position, used as the source of the portal burst.
        pos: BlockPos,
    },
    /// `BlockCake.eatCake`: the application must first apply `EntityPlayer.canEat(false)`; the
    /// world helper [`World::consume_cake_slice`] then performs the exact bites/air state write.
    Cake {
        /// The clicked cake position.
        pos: BlockPos,
        /// The current `BlockCake.BITES` metadata, retained for stale-state rejection.
        bites: u8,
    },
}

/// The client-side world: loaded chunks, the entity arena, and the mesher's dirty set.
///
/// java: `World` + `WorldClient` + `ChunkProviderClient`, collapsed into one non-cyclic struct.
pub struct World {
    // `pub(crate)` so [`crate::light`] — the second `impl World` block, which exists so the
    // A12-01 flood fill does not have to be written into this file — can reach it.
    /// Columns and their sections are shared with render snapshots until written. Every writer
    /// goes through `Arc::make_mut`, so a meshing worker keeps the state it was handed.
    ///
    /// [`mc_core::FastMap`]: this is the hottest map in the client — every
    /// `block_at`, light read and write, collision cell and mesher read on the workers probes it,
    /// and SipHash on a `ChunkPos` cost more than the probe itself. Keys are server-chosen, so the
    /// hasher stays seeded, and its iteration order stays what `std`'s was: random per process
    /// (`loaded_chunks` callers that need an order sort, see `light.rs`).
    pub(crate) chunks: mc_core::FastMap<ChunkPos, Arc<Chunk>>,
    /// java: `ChunkProviderClient.blankChunk`, the one `EmptyChunk` every absent column shares
    /// for this world's lifetime (a new `World` is a new `WorldClient`, hence a new one).
    ///
    /// `EmptyChunk` hard-codes `getBlock`, metadata, opacity, light, height, sky, entity and
    /// tile accessors, but inherits `getBlockState`, `setBlockState` and `fillChunk`. So block
    /// *states* written into an absent column (an S23/S22, a partial S21, a local prediction)
    /// live here at `(x & 15, y, z & 15)` and [`World::block_at`] reads them back in every
    /// absent column; the hard-coded accessors stay hard-coded (see `light.rs`). An unload of a
    /// resident column (`doPreChunk(false)`) never touches it. `pub(crate)` for `light.rs`.
    pub(crate) blank_chunk: Chunk,
    /// The `hecs` entity arena. Public per the frozen §7 contract: systems query it directly
    /// rather than going through `World` wrapper methods for every component access.
    pub ecs: hecs::World,
    /// Sections touched since the mesher (S11) last drained this set. `pub(crate)` for
    /// [`crate::light`]'s invalidation helpers. Up to eight inserts per
    /// light write; [`World::drain_dirty_sections`] hands it out sorted, so its own order is
    /// never observed.
    pub(crate) dirty: mc_core::FastSet<ChunkSectionPos>,
    /// Whether the current dimension has a sky (hazard H6): the overworld does, the Nether and
    /// the End do not. Drives both the chunk-packet unpacker (whether to read the sky-light
    /// region at all) and [`World::light_at`] (whether sky light contributes). S14 sets this
    /// from the dimension field of `S01PacketJoinGame`/`S07PacketRespawn` once it exists; it
    /// defaults to the overworld's `true` so a `World` is useful in tests without that wiring.
    has_sky_light: bool,
    /// The dimension-selected source for generated biomes in absent/out-of-range columns.
    ///
    /// Resident chunk biome bytes remain authoritative and are not replaced when this changes;
    /// changing dimensions replaces the world in the client, while this setter also makes the
    /// transition explicit for the packet path and focused tests.
    biome_source: BiomeSource,
    /// Canonical `WorldType` name supplied by the join/respawn lifecycle.
    ///
    /// java: `Chunk.getBlockState` checks `worldObj.getWorldType() == WorldType.DEBUG_WORLD`
    /// before consulting either a resident section or the client's blank chunk. Keeping the
    /// terrain type separately from [`BiomeSource`] matters because Nether/End providers win
    /// biome selection while the wire world type still controls this DEBUG_WORLD lookup.
    level_type: &'static str,
    /// The valid property-rich block-state list used by Java's
    /// `ChunkProviderDebug.field_177464_a`.
    ///
    /// It is cached per client world so a mesher lookup is O(1), while `mc-data` itself retains
    /// allocation-free generated tables. The list is only consulted when [`Self::level_type`]
    /// is `debug_all_block_states`.
    debug_world_states: Arc<[mc_data::DebugWorldState]>,
    /// Java's `field_177462_b` (`ceil(sqrt(states.len()))`).
    debug_world_grid_width: i32,
    /// Java's `field_181039_c` (`ceil(states.len() / width)`).
    debug_world_grid_height: i32,
    /// Exact seed-0 `GenLayer` fallback used by the default client overworld provider.
    ///
    /// The layer graph is immutable after construction and its Java-shaped BiomeCache is behind
    /// a `Send + Sync` mutex, so `World::biome_at` remains a read-only operation to callers even
    /// when terrain/tint workers evaluate it in parallel.
    overworld_biome_provider: OverworldBiomeProvider,
    /// java: `Chunk.chunkTileEntityMap` (`world/chunk/Chunk.java:74`), flattened across columns.
    ///
    /// Vanilla keys it per chunk so unload drops it for free; here it is one flat map because
    /// the read that matters runs once per *visible tile entity per frame* and a map-of-maps
    /// would cost a second hash for every one of them. [`crate::tileent`] owns everything that
    /// touches it — including [`World::drop_tile_entities_in_chunk`], which is the unload's
    /// explicit replacement for vanilla's free drop. A [`mc_core::FastMap`] for
    /// the same per-frame reason; every iteration of it that can be observed
    /// sorts first or is order-free (`tick_pistons`, chest adjacency).
    pub(crate) tile_entities: mc_core::FastMap<mc_math::BlockPos, crate::tileent::TileEntity>,
    /// java: `World.lightUpdateBlockList` (`world/World.java:151`, allocated 32,768 long at
    /// `:158`) — the flood fill's scratch queue, a field rather than a local so a light update
    /// costs no allocation. [`World::check_light_for`] takes it out and puts it back.
    pub(crate) light_update_block_list: Vec<i32>,
    /// The round-robin cursor for [`World::tick_chunk_light`], as a stack of columns still owed
    /// a visit this cycle. Refilled (sorted, so a replay is deterministic) when it empties.
    /// java: `WorldClient.previousActiveChunkSet`, which serves the same "everyone gets a turn
    /// before anyone gets a second" purpose.
    pub(crate) light_tick_queue: Vec<ChunkPos>,
    /// The columns already consumed from the current `WorldClient.activeChunkSet` relight cycle.
    ///
    /// Java retains only entries that are still active, clears the set once it catches up to the
    /// complete active set, and then starts a new at-most-ten-columns-per-tick cycle.
    ///
    /// Held as a set sorted by `(x, z)` (no duplicates) so retention and the cycle scan are merge
    /// walks against the equally sorted active list rather than SipHash probes.
    pub(crate) previous_active_light_chunks: Vec<ChunkPos>,
    /// The cached `activeChunkSet` and cycle cursor behind
    /// [`World::tick_chunk_relight_active_around`].
    pub(crate) active_light_chunks: crate::light::ActiveLightChunks,
    /// Run every `checkLightFor` through the original one-hashed-lookup-per-access transcription
    /// instead of [`crate::light`]'s resolved-column fast path. Both produce bit-identical
    /// light, dirty sets and storage; this exists so differential tests and `light_perf` can run
    /// the reference engine on demand. DEBUG_WORLD always uses the reference engine, because its
    /// `getBlockState` ignores chunk storage.
    pub(crate) reference_light_engine: bool,
    /// Client-side `map_<id>` records. java: `WorldClient.mapStorage`, whose
    /// `SaveDataMemoryStorage` backing is an in-memory map on multiplayer clients.
    map_data: HashMap<i32, MapData>,
    /// Creation-time center/dimension fields for client-created map records. S34 does not carry
    /// these fields and therefore must not replace them when it updates the corresponding map.
    map_data_metadata: HashMap<i32, MapDataMetadata>,
    /// The client world's `World.rand`, used by tile animations such as the enchantment-table
    /// page target.  It is stateful rather than re-seeded per tile so the Java draw order is
    /// retained when several tables tick in one world pass.
    pub(crate) tile_entity_random: JavaRandom,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world with no loaded chunks or entities, defaulting to an overworld-shaped
    /// sky (see [`World::has_sky_light`]).
    #[must_use]
    pub fn new() -> Self {
        let debug_world_states = shared_debug_world_states();
        let (debug_world_grid_width, debug_world_grid_height) =
            debug_world_grid_dimensions(debug_world_states.len());

        Self {
            chunks: mc_core::FastMap::default(),
            blank_chunk: Chunk::empty(),
            ecs: hecs::World::new(),
            dirty: mc_core::FastSet::default(),
            has_sky_light: true,
            biome_source: BiomeSource::OverworldSeed0,
            level_type: "default",
            debug_world_states,
            debug_world_grid_width,
            debug_world_grid_height,
            overworld_biome_provider: OverworldBiomeProvider::new(0),
            tile_entities: mc_core::FastMap::default(),
            light_update_block_list: Vec::new(),
            light_tick_queue: Vec::new(),
            previous_active_light_chunks: Vec::new(),
            active_light_chunks: crate::light::ActiveLightChunks::default(),
            reference_light_engine: false,
            map_data: HashMap::new(),
            map_data_metadata: HashMap::new(),
            tile_entity_random: JavaRandom::new(0),
        }
    }

    /// Clone the part of this client world that terrain meshing is allowed to read.
    ///
    /// A terrain worker may outlive the frame that dispatched it.  Passing `&World` to such a
    /// worker would therefore either hold the game-world borrow across frames or require a lock
    /// around every block/light lookup.  Java solves the same lifetime boundary by giving a
    /// `ChunkRenderWorker` a `RegionRenderCache`; this is the Rust-side equivalent at the
    /// section-job boundary.  The returned value owns the resident chunk columns and immutable
    /// biome/debug settings, so it is safe to move into a worker and share between its two
    /// threads.
    ///
    /// Entity/map/light-work state is intentionally not copied.  None of those are read by
    /// `mc-render`'s terrain mesher, and copying them would make a render snapshot look like a
    /// second game world to callers.  Blank-column and biome-provider state *is* copied because
    /// unloaded-neighbour, tint, and generated-biome reads are part of a mesh's observable input.
    ///
    /// The client takes one of these on every frame with mesh work queued, so its cost must not
    /// scale with the world. Columns are shared rather than copied, which leaves one reference
    /// count per column. Tile entities are limited to flower pots, the only tile payload the
    /// mesher and tinter read; everything else in that map can hold arbitrary sign text and NBT.
    #[must_use]
    pub fn render_snapshot(&self) -> Self {
        let mut snapshot = Self::new();
        snapshot.chunks = self.chunks.clone();
        snapshot.blank_chunk = self.blank_chunk.clone();
        snapshot.has_sky_light = self.has_sky_light;
        snapshot.biome_source = self.biome_source;
        snapshot.level_type = self.level_type;
        snapshot.debug_world_states = self.debug_world_states.clone();
        snapshot.debug_world_grid_width = self.debug_world_grid_width;
        snapshot.debug_world_grid_height = self.debug_world_grid_height;
        snapshot.overworld_biome_provider = self.overworld_biome_provider.clone();
        snapshot.tile_entities = self
            .tile_entities
            .iter()
            .filter(|(_, tile)| tile.flower_pot().is_some())
            .map(|(pos, tile)| (*pos, tile.clone()))
            .collect();
        snapshot
    }

    /// [`Self::render_snapshot`] restricted to what meshing `sections` can read: the columns in
    /// the 3×3 neighbourhood of each section's column, and the flower pots in them.
    ///
    /// A section's mesh never reads more than three blocks outside the section horizontally
    /// (`BlockPortal`'s culling probe; blocks, light, AO corners, slab/stair neighbour light,
    /// actual states, the 3×3 biome tint average and fluid corners are all within two), so every
    /// read lands in one of those nine columns. They are the same `Arc`s the full snapshot would
    /// hold, and a column outside them is never asked about, so the meshes are identical
    /// (mc-render's `tests/mesher_fast_path_parity.rs` compares the two).
    ///
    /// The point is cost (round-2 #31 / MESH-3 "full"): the full snapshot clones the whole
    /// column map — one reference-count increment on a cold cache line per resident column,
    /// ~600–1,100 at render distance 12–16, plus the same again when it is dropped — on every
    /// frame with mesh work queued. This one is proportional to the sections dispatched this
    /// frame, and a main-thread write to a column copies it only if an in-flight job can still
    /// read it.
    #[must_use]
    pub fn render_snapshot_for_sections(&self, sections: &[ChunkSectionPos]) -> Self {
        // Sections of one column share a neighbourhood: expand each column once.
        let mut columns: Vec<(i32, i32)> = sections.iter().map(|s| (s.x, s.z)).collect();
        columns.sort_unstable();
        columns.dedup();
        let mut chunks: mc_core::FastMap<ChunkPos, Arc<Chunk>> = mc_core::FastMap::default();
        for (x, z) in columns {
            for dx in -1..=1 {
                for dz in -1..=1 {
                    let pos = ChunkPos::new(x + dx, z + dz);
                    if chunks.contains_key(&pos) {
                        continue;
                    }
                    if let Some(chunk) = self.chunks.get(&pos) {
                        chunks.insert(pos, Arc::clone(chunk));
                    }
                }
            }
        }
        let mut snapshot = Self::new();
        snapshot.blank_chunk = self.blank_chunk.clone();
        snapshot.has_sky_light = self.has_sky_light;
        snapshot.biome_source = self.biome_source;
        snapshot.level_type = self.level_type;
        snapshot.debug_world_states = self.debug_world_states.clone();
        snapshot.debug_world_grid_width = self.debug_world_grid_width;
        snapshot.debug_world_grid_height = self.debug_world_grid_height;
        snapshot.overworld_biome_provider = self.overworld_biome_provider.clone();
        snapshot.tile_entities = self
            .tile_entities
            .iter()
            .filter(|(pos, tile)| {
                tile.flower_pot().is_some() && chunks.contains_key(&pos.chunk_pos())
            })
            .map(|(pos, tile)| (*pos, tile.clone()))
            .collect();
        snapshot.chunks = chunks;
        snapshot
    }

    // ---- block access (frozen §7 API) --------------------------------------------------

    /// The block state at `p`: `World.getBlockState`. Air for out-of-world `y`.
    ///
    /// A column the client holds no chunk for answers from the shared blank chunk, at
    /// `(x & 15, y, z & 15)`: `ChunkProviderClient.provideChunk` hands every absent column the
    /// same `EmptyChunk`, which overrides `getBlock` but **not** the inherited `getBlockState`
    /// (`EmptyChunk.java`, `Chunk.java:586-621`). So a write into one absent column (see
    /// [`World::set_block`]) reads back at the same local position in every absent column, as
    /// it does in vanilla. An empty blank chunk costs one section probe here.
    #[must_use]
    pub fn block_at(&self, p: BlockPos) -> BlockState {
        // java: Chunk.getBlockState checks WorldType.DEBUG_WORLD before looking at either the
        // resident section or the client blank chunk. This is deliberately before the ordinary
        // height guard: Java's DEBUG branch returns air for every non-grid Y, including an
        // out-of-range position, and never exposes packet storage in this world type.
        if let Some(state) = self.debug_world_block_at(p) {
            return state;
        }
        if !(0..256).contains(&p.y) {
            return BlockState::AIR;
        }
        let (x, z) = local_xz(p);
        match self.chunks.get(&p.chunk_pos()) {
            Some(chunk) => chunk.block_at(x, p.y as u8, z),
            // java: `World.getBlockState` returns air for `!isValid(pos)` (|x|, |z| beyond the
            // 30,000,000 limit) before it asks the provider.
            None if !is_valid_xz(p) => BlockState::AIR,
            None => self.blank_chunk.block_at(x, p.y as u8, z),
        }
    }

    /// The block metadata at `p`, matching `Chunk.getBlockMetadata` and `EmptyChunk`'s fixed
    /// zero result. An absent column never consults packet scratch storage.
    #[must_use]
    pub fn block_metadata_at(&self, p: BlockPos) -> u8 {
        if !(0..256).contains(&p.y) {
            return 0;
        }
        let (x, z) = local_xz(p);
        self.chunks
            .get(&p.chunk_pos())
            .map_or(0, |chunk| chunk.block_at(x, p.y as u8, z).meta())
    }

    /// The light opacity at `p`, matching `Chunk.getBlockLightOpacity` and `EmptyChunk`'s fixed
    /// `255` result. A loaded column reads its actual block; an absent column always returns the
    /// hard-coded EmptyChunk value even if packet decoding previously used a temporary scratch.
    #[must_use]
    pub fn block_light_opacity_at(&self, p: BlockPos) -> i32 {
        let Some(chunk) = self.chunks.get(&p.chunk_pos()) else {
            return 255;
        };
        if !(0..256).contains(&p.y) {
            return 0;
        }
        let (x, z) = local_xz(p);
        chunk.light_opacity(x, p.y, z)
    }

    /// Java `Chunk.getBlockState`'s DEBUG_WORLD branch, including the
    /// `ChunkProviderDebug.func_177461_b` state-grid mapping.
    ///
    /// `Some` means DEBUG_WORLD owns the lookup (including an explicit air result); `None`
    /// leaves the ordinary resident/blank-chunk path in [`World::block_at`] untouched.
    #[inline]
    fn debug_world_block_at(&self, p: BlockPos) -> Option<BlockState> {
        if !self.is_debug_world() {
            return None;
        }

        Some(match p.y {
            60 => DEBUG_WORLD_BARRIER,
            70 => self.debug_world_state_at(p.x, p.z),
            _ => BlockState::AIR,
        })
    }

    /// Java `ChunkProviderDebug.func_177461_b`, with Java's strict-positive odd-coordinate
    /// checks and inclusive width/height bounds retained exactly.
    #[inline]
    fn debug_world_state_at(&self, x: i32, z: i32) -> BlockState {
        self.debug_world_state_entry_at(x, z)
            .map_or(BlockState::AIR, mc_data::DebugWorldState::stored)
    }

    /// The property-rich Java `IBlockState` entry at a DEBUG_WORLD grid coordinate.
    ///
    /// This is intentionally separate from [`Self::debug_world_state_at`]: the latter is the
    /// packed-state compatibility seam used by block access, while the renderer needs the
    /// complete tuple to distinguish metadata-colliding fence, wire, stair, fire, and flower-pot
    /// entries.
    #[inline]
    fn debug_world_state_entry_at(&self, x: i32, z: i32) -> Option<&mc_data::DebugWorldState> {
        if x <= 0 || z <= 0 || x % 2 == 0 || z % 2 == 0 {
            return None;
        }

        let x = x / 2;
        let z = z / 2;
        if x > self.debug_world_grid_width || z > self.debug_world_grid_height {
            return None;
        }

        // Java computes this as an int expression and then calls MathHelper.abs_int. Use the
        // wrapping forms so extreme (modded/untrusted) positions retain Java's two's-complement
        // overflow behavior instead of panicking in a debug Rust build.
        let index = x
            .wrapping_mul(self.debug_world_grid_width)
            .wrapping_add(z)
            .wrapping_abs();
        if index < 0 {
            return None;
        }
        self.debug_world_states.get(index as usize)
    }

    /// Returns the full property tuple used by Java's DEBUG_WORLD model lookup at `p`.
    ///
    /// `None` means that `p` is not on the state plane or does not map to a grid cell.  A mapped
    /// air cell still returns `Some`, matching Java's explicit first entry in the debug list.
    #[inline]
    #[must_use]
    pub fn debug_world_actual_state_at(
        &self,
        p: BlockPos,
    ) -> Option<mc_data::block::ActualBlockState> {
        if !self.is_debug_world() || p.y != 70 {
            return None;
        }
        self.debug_world_state_entry_at(p.x, p.z)
            .map(mc_data::DebugWorldState::actual_state)
    }

    /// The resident block state at `p`, or `None` when its column is not loaded or the
    /// coordinate is outside the Java world height. In DEBUG_WORLD, a loaded column returns the
    /// same synthetic barrier/grid/air result as [`World::block_at`], while the residency gate
    /// remains intact. Ambient/weather consumers use this view because the public
    /// [`World::block_at`] returns air for an absent provider column, so this view keeps the
    /// residency gate explicit for ambient/weather consumers.
    #[inline]
    #[must_use]
    pub fn loaded_block_at(&self, p: BlockPos) -> Option<BlockState> {
        if !(0..256).contains(&p.y) {
            return None;
        }
        // One probe answers both the residency gate and the read (PART-6).
        let chunk = self.chunks.get(&p.chunk_pos())?;
        self.loaded_block_in(chunk, p)
    }

    /// [`World::loaded_block_at`] for a `p` with a valid Y whose resident column is `chunk`.
    #[inline]
    fn loaded_block_in(&self, chunk: &Chunk, p: BlockPos) -> Option<BlockState> {
        if let Some(state) = self.debug_world_block_at(p) {
            return Some(state);
        }
        let (x, z) = local_xz(p);
        let state = chunk.block_at(x, p.y as u8, z);
        state.is_legal().then_some(state)
    }

    /// The biome at `p`, using `generated` for the exact client-side world-manager fallback.
    ///
    /// This is the `World.getBiomeGenForCoords` split: a valid Y in a loaded column calls
    /// [`Chunk::biome_with_generated`] and therefore caches a `0xFF` sentinel; an unloaded or
    /// out-of-world position calls `WorldChunkManager.getBiomeGenerator` directly and leaves all
    /// chunk storage untouched. `generated` returns `None` only to model the manager's explicit
    /// `BiomeGenBase.plains` default.
    ///
    /// This mutable compatibility API still accepts an explicit callback for callers that need a
    /// custom provider; the read-only live path is [`World::biome_at`], which installs the exact
    /// seed-0 overworld graph (or the dimension/level-type provider selected by the lifecycle).
    #[must_use]
    pub fn biome_at_with_generated(
        &mut self,
        p: BlockPos,
        generated: impl FnOnce(BlockPos) -> Option<u8>,
    ) -> u8 {
        // java: World.isBlockLoaded(pos) first rejects y < 0 / y >= 256. It then either goes
        // through the resident chunk (which owns the cache) or invokes the provider directly;
        // `ChunkProviderClient.blankChunk` is not consulted by this particular World method.
        if (0..256).contains(&p.y) {
            if let Some(chunk) = self.chunks.get_mut(&p.chunk_pos()).map(Arc::make_mut) {
                return chunk.biome_with_generated(p, generated);
            }
        }
        generated(p).map_or(1, |id| mc_data::biome(id).id)
    }

    /// The source used when a biome is not supplied by a resident chunk.
    #[inline]
    #[must_use]
    pub const fn biome_source(&self) -> BiomeSource {
        self.biome_source
    }

    /// Whether this client world has the wire `WorldType.DEBUG_WORLD` terrain type.
    ///
    /// The terrain type is independent from [`BiomeSource`]: Nether/End providers take
    /// precedence for biome generation, while Java's `Chunk.getBlockState` still keys its
    /// barrier/state-grid short circuit from `WorldInfo.getTerrainType()`. A renderer that needs
    /// Java's DEBUG_WORLD model lookup bypass can use this seam without reaching into storage.
    #[inline]
    #[must_use]
    pub fn is_debug_world(&self) -> bool {
        self.level_type == DEBUG_WORLD_LEVEL_TYPE
    }

    /// Selects the generated-biome source without touching resident chunk bytes.
    ///
    /// This is the client lifecycle seam for `S01 Join Game`/`S07 Respawn`.
    /// `dimension_id == -1` selects the Nether constant and `1` selects the End
    /// constant; every other id uses the client-side overworld source.
    pub fn set_biome_source(&mut self, source: BiomeSource) {
        self.biome_source = source;
        self.has_sky_light = !matches!(source, BiomeSource::Nether | BiomeSource::End);
    }

    /// Selects the dimension's default biome source and sky-light mode from the wire id.
    pub fn set_dimension_id(&mut self, dimension_id: i32) {
        self.set_dimension_and_level_type(dimension_id, "default");
    }

    /// Selects the client provider for a dimension and its canonical packet world type.
    ///
    /// java: `WorldProvider.registerWorld` reads both the dimension-specific provider and the
    /// `WorldInfo` terrain type. Nether/End providers always win over the terrain type. The
    /// surface provider uses the seed-0 graph for ordinary world types, but its FLAT and DEBUG
    /// branches install a fixed Plains manager. There is no generator-options field in Join or
    /// Respawn, so a multiplayer client can only observe FLAT's default (empty-options) preset;
    /// custom flat terrain and all block placement remain server-authoritative.
    ///
    /// Deliberate difference: a dimension id other than -1, 0 or 1 is
    /// treated as the surface. Java has no provider for it (`getProviderForDimension` returns
    /// null), so `new WorldClient(...)` throws inside the scheduled Join Game/Respawn task;
    /// `Util.runTask` only logs that, which leaves vanilla with no world after a join (stuck on
    /// the connecting screen) or with the old world after a respawn. No protocol-47 server sends
    /// such an id, and reproducing a broken client state is not a parity goal.
    pub fn set_dimension_and_level_type(&mut self, dimension_id: i32, level_type: &str) {
        let level_type = canonical_level_type_name(level_type);
        // `WorldType.LARGE_BIOMES` (six zooms) and `WorldType.DEFAULT_1_1` (GenLayerBiome's
        // legacy warm table) have distinct GenLayer graphs even though their provider is still
        // the ordinary overworld manager. Rebuild this seed-0 client provider at the lifecycle
        // boundary so Join Game/Respawn world-type swaps cannot retain DEFAULT's graph.
        self.overworld_biome_provider = OverworldBiomeProvider::new_for_world_type(0, level_type);
        self.level_type = level_type;
        self.set_biome_source(match dimension_id {
            -1 => BiomeSource::Nether,
            1 => BiomeSource::End,
            _ => match level_type {
                "flat" => BiomeSource::FlatPlains,
                "debug_all_block_states" => BiomeSource::DebugPlains,
                // DEFAULT/LARGE_BIOMES/AMPLIFIED/CUSTOMIZED/default_1_1 all use the seed-0
                // `WorldChunkManager` built above for this level type (no generator options
                // reach a client, so AMPLIFIED and CUSTOMIZED are DEFAULT's graph).
                _ => BiomeSource::OverworldSeed0,
            },
        });
    }

    /// The canonical packet world type selected by the join/respawn lifecycle.
    #[inline]
    #[must_use]
    pub const fn level_type(&self) -> &'static str {
        self.level_type
    }

    /// Reads a biome through the installed dimension source.
    ///
    /// Resident, non-sentinel chunk bytes remain authoritative. A `0xFF` chunk
    /// sentinel, an absent chunk, or an invalid Y is passed to the selected
    /// source directly, matching `World.getBiomeGenForCoords`'s provider branch
    /// while preserving the immutable `&World` contract used by rayon meshing.
    #[must_use]
    pub fn biome_at(&self, p: BlockPos) -> u8 {
        if (0..256).contains(&p.y) {
            if let Some(chunk) = self.chunks.get(&p.chunk_pos()) {
                let raw = chunk.biomes()[((p.z & 15) as usize) << 4 | (p.x & 15) as usize];
                if raw != 0xFF {
                    return mc_data::biome(raw).id;
                }
            }
        }
        match self.biome_source {
            BiomeSource::Nether => 8,
            BiomeSource::End => 9,
            BiomeSource::FlatPlains | BiomeSource::DebugPlains => 1,
            BiomeSource::OverworldSeed0 => self.overworld_biome_provider.biome_at(p.x, p.z),
        }
    }

    /// The Java `WorldChunkManager` rainfall value at `p`'s column.
    ///
    /// Resident chunk biome bytes remain authoritative for the ordinary overworld provider,
    /// including the Java rainfall quantization (`getIntRainfall() / 65536.0F`). The FLAT and
    /// DEBUG providers are `WorldChunkManagerHell` instances, however, so their fixed constructor
    /// rainfall (0.5F and 0.0F respectively) wins even when a resident chunk is present. For
    /// absent or sentinel columns this follows the installed provider; the default overworld
    /// path shares the provider's 16x16 biome/rainfall cache with [`World::biome_at`].
    #[must_use]
    pub fn rainfall_at(&self, p: BlockPos) -> f32 {
        // Java `WorldChunkManagerHell.getRainfall` never reads resident chunk bytes. Its
        // constructor value is 0.5F for FLAT and 0.0F for DEBUG, while Nether/End providers use
        // 0.0F as well.
        match self.biome_source {
            BiomeSource::FlatPlains => return 0.5,
            BiomeSource::DebugPlains | BiomeSource::Nether | BiomeSource::End => return 0.0,
            BiomeSource::OverworldSeed0 => {}
        }

        if (0..256).contains(&p.y) {
            if let Some(chunk) = self.chunks.get(&p.chunk_pos()) {
                let raw = chunk.biomes()[((p.z & 15) as usize) << 4 | (p.x & 15) as usize];
                if raw != 0xFF {
                    return java_rainfall(raw);
                }
            }
        }

        match self.biome_source {
            BiomeSource::Nether | BiomeSource::End => 0.0,
            BiomeSource::FlatPlains => 0.5,
            BiomeSource::DebugPlains => 0.0,
            BiomeSource::OverworldSeed0 => self.overworld_biome_provider.rainfall_at(p.x, p.z),
        }
    }

    /// Run the installed overworld provider's Java-compatible cache cleanup cadence.
    ///
    /// Lookups already perform the cadence check, matching the observable result of vanilla's
    /// `WorldChunkManager.cleanupCache` when a caller is not running a separate world tick.
    /// This explicit seam lets client lifecycle/tick code request the same cleanup pass without
    /// exposing cache storage or making time a global.
    pub fn cleanup_biome_cache(&self) {
        self.overworld_biome_provider.cleanup_cache();
    }

    /// Sets the block state at `p` and marks the affected section(s) dirty for the mesher:
    /// `World.setBlockState(pos, state, 3)`. A no-op outside `y = 0..=255`.
    /// `S23PacketBlockChange`/`S22PacketMultiBlockChange` and every local prediction route
    /// through this.
    ///
    /// A column the client holds no chunk for is **not** a no-op: `World.setBlockState` writes
    /// into whatever `getChunkFromBlockCoords` returns, which for an absent column is
    /// `ChunkProviderClient`'s one shared `EmptyChunk`, and `EmptyChunk` inherits
    /// `Chunk.setBlockState` unchanged (`Chunk.java:660-697`). The block therefore lands in the
    /// blank chunk at `(x & 15, y, z & 15)` and [`World::block_at`] (collision, raytrace,
    /// placement) reads it back in every absent column. That is how a bridge over a void column
    /// the server "sent" as an all-air S21 (bitmask 0, i.e. an unload) stays solid under the
    /// player, together with the server's S23 for it. Nothing else happens: `checkLight`
    /// returns at once (`isAreaLoaded(.., false)` fails on an empty chunk), `markBlockForUpdate`
    /// needs `chunk.isPopulated()` (false), and `EmptyChunk`'s tile-entity hooks are no-ops.
    pub fn set_block(&mut self, p: BlockPos, s: BlockState) {
        if !(0..256).contains(&p.y) {
            return;
        }
        if !self.chunks.contains_key(&p.chunk_pos()) {
            // java: `World.setBlockState` refuses `!isValid(pos)` before it asks the provider.
            if is_valid_xz(p) {
                let (x, z) = local_xz(p);
                // `Chunk.setBlockState` on the blank chunk: the no-op and air-into-a-hole
                // refusals, the precipitation-map reset and the storage write. Its light
                // follow-ups are `EmptyChunk` no-ops or unobservable there, so the outcome that
                // tells a resident column what to relight is dropped.
                let _ = self.blank_chunk.set_block_relit(x, p.y, z, s);
            }
            return;
        }
        let old = self.block_at(p);
        if old == s {
            return;
        }
        // The whole of `Chunk.setBlockState` + `World.setBlockState`, heightmap
        // maintenance and both light flood fills included. See [`crate::light`].
        self.set_block_lit(p, s);
        // Only mirror `Chunk.setBlockState`'s tile-entity lifecycle after the state really
        // landed in a resident column. java: `Chunk.java:696-770`; the client removes the old
        // provider when its block changes, then creates the new provider inline with the new
        // metadata. `EmptyChunk` overrides its tile-entity accessors to no-op, handled above.
        if self.block_at(p) == s {
            self.reconcile_tile_entity_after_block_change(p, old, s);
        }
    }

    /// Applies farmland's neighbour reaction after a block was placed above it.
    ///
    /// java: `BlockFarmland.onNeighborBlockChange` calls `Material.isSolid()` for the block at
    /// `pos.up()` and replaces the farmland with dirt when that predicate is true.  This is an
    /// explicit operation rather than part of [`World::set_block`], and the client never calls
    /// it: `World.setBlockState` notifies neighbours only `if (!this.isRemote && (flags & 1) !=
    /// 0)` (`World.java:381`), so neither a packet write nor a predicted placement runs any
    /// `onNeighborBlockChange` on a vanilla client. The server's own update arrives as S23.
    ///
    /// Returns whether farmland was replaced.  The farmland identity guard makes repeated calls
    /// inert after the first replacement.
    pub fn react_farmland_to_solid_above(&mut self, farmland_pos: BlockPos) -> bool {
        if self.block_at(farmland_pos).block() != BlockId(60)
            || !self.block_at(farmland_pos.up()).props().material.is_solid()
        {
            return false;
        }

        self.set_block(farmland_pos, BlockState::new(BlockId(3), 0));
        true
    }

    /// Performs `BlockBush.checkAndDropBlock` for the bush currently at `pos`.
    ///
    /// Both `BlockBush.onNeighborBlockChange` and `BlockBush.updateTick` enter this exact
    /// virtual method in vanilla.  The latter has no random-tick scheduler in `WorldClient`,
    /// and `World.setBlockState` sends neighbour notifications only when `!isRemote`; therefore
    /// client packet application must **not** call this from [`World::set_block`].  It is kept as
    /// an explicit world operation for the places that do model either trigger, rather than
    /// quietly giving every incoming block-change packet server-world side effects.
    ///
    /// Returns whether `checkAndDropBlock` removed a bush state.  This client world deliberately
    /// has no item-entity drop side effect: `Block.dropBlockAsItem` reaches its spawn body only
    /// behind `!worldIn.isRemote`, then the block body sets the state to air.
    pub fn check_and_drop_bush(&mut self, pos: BlockPos) -> bool {
        let state = self.block_at(pos);
        let bush = state.props().block;
        if !family::bush(bush) || self.bush_can_block_stay_for_state(state, pos) {
            return false;
        }

        if bush != BlockId(175) {
            // java: `BlockBush.checkAndDropBlock`: drop (a no-op on the client), then air with
            // flags 3. `set_block` is this world's flags-3 state-write analogue.
            self.set_block(pos, BlockState::AIR);
            return true;
        }

        // java: `BlockDoublePlant.checkAndDropBlock`.  Its override removes the other half
        // first (flags 2), then removes the lower half (flags 3) and drops only for a lower-half
        // trigger.  Both flag values have the same storage/meshing result in this client world;
        // preserving the ordering matters because malformed, half-present plants are observable.
        let upper = state.meta() & 8 != 0;
        let other = if upper { pos } else { pos.up() };
        let lower = if upper { pos.down() } else { pos };
        let other_is_plant = if upper {
            true
        } else {
            self.block_at(other).props().block == bush
        };
        let lower_is_plant = if upper {
            self.block_at(lower).props().block == bush
        } else {
            true
        };

        if other_is_plant {
            self.set_block(other, BlockState::AIR);
        }
        if lower_is_plant {
            self.set_block(lower, BlockState::AIR);
        }
        // The state at `pos` is a double plant, so one of the two branches above always removed
        // it.  Keep the explicit return value so callers can distinguish a stable plant from a
        // survival failure without re-reading the world.
        true
    }

    /// Applies a decoded `S22PacketMultiBlockChange` entry list to the chunk at `chunk_pos`.
    ///
    /// `crammed` is the packet's per-entry position short, packed exactly as the wire does —
    /// `x = crammed >> 12 & 15`, `z = crammed >> 8 & 15`, `y = crammed & 255`
    /// (`S22PacketMultiBlockChange.BlockUpdateData.getPos`) — so S06/S14 can hand this
    /// function the raw field with no translation. `state` is already-decoded via
    /// `Block.BLOCK_STATE_IDS`, which packs `block_id << 4 | meta` — the *same* packing as
    /// [`mc_data::BlockState`], **not** [`mc_data::BlockState::from_java_state_id`]'s.
    pub fn apply_multi_block_change(&mut self, chunk_pos: ChunkPos, changes: &[(u16, BlockState)]) {
        // java: `NetHandlerPlayClient.handleMultiBlockChange` calls
        // `WorldClient.invalidateRegionAndSetBlock` once per entry, i.e. a full
        // `World.setBlockState(pos, state, 3)` — light included — in wire order. The batched
        // [`World::apply_block_changes`] reorders that light work (all writes first, one relight
        // per column, `checkLight` sorted by position), and the result is not always the same:
        // over stale stored light 1 of 40 random craters ended with a different block-light
        // nibble (09-relight findings). Client light gates C08 (mushroom placement), so S22 takes
        // vanilla's per-entry path. It was faster than the batch on a measured 257-block crater;
        // it does give up per-column coalescing for packets that stack many entries in a column.
        for &(crammed, state) in changes {
            self.set_block(multi_block_change_pos(chunk_pos, crammed), state);
        }
    }

    /// Applies decoded block states as one client-side update batch.
    ///
    /// This is the construction/update counterpart to [`World::set_block`].  The latter keeps
    /// vanilla's one-write-at-a-time lighting cadence for local prediction; this method is for
    /// packet batches and large fixture setup, where chunk storage can be written in wire order
    /// and the expensive world-wide relight work safely coalesced afterward.
    pub fn apply_block_changes(&mut self, changes: &[(BlockPos, BlockState)]) {
        self.apply_block_changes_lit(changes);
    }

    // ---- light access (frozen §7 API) ---------------------------------------------------

    /// The effective light level (`0..=15`) at `p`: `max(sky_light, block_light)`, matching
    /// `World.getLight(pos)` with no darkening subtracted (`Chunk.getLightSubtracted(pos, 0)`,
    /// java: `World.java:604-618`, `Chunk.java:818-842`).
    ///
    /// Server-authoritative: this crate stores whatever the server sent and never propagates
    /// or recomputes light, per S08's task list. Sky light is ignored outside dimensions that
    /// have one ([`World::has_sky_light`], hazard H6) — asking for it there always yields `0`,
    /// same as `extendedblockstorage.getExtSkylightValue` would if `provider.getHasNoSky()`.
    #[must_use]
    pub fn light_at(&self, p: BlockPos) -> u8 {
        self.sky_light_at(p).max(self.block_light_at(p))
    }

    /// The raw block-light nibble at `p`. `0` for out-of-world `y`, an unloaded chunk, or an
    /// unloaded section — vanilla's blank-chunk/blank-section default for `EnumSkyBlock.BLOCK`.
    #[must_use]
    pub fn block_light_at(&self, p: BlockPos) -> u8 {
        if !(0..256).contains(&p.y) {
            return 0;
        }
        let (x, z) = local_xz(p);
        self.chunks
            .get(&p.chunk_pos())
            .and_then(|c| c.section((p.y >> 4) as u8))
            .map_or(0, |s| s.block_light(x, (p.y & 15) as u8, z))
    }

    /// The raw sky-light nibble at `p`, or `0` outright when [`World::has_sky_light`] is
    /// `false`. An unloaded chunk/section reports full brightness (`15`) when the dimension
    /// has sky, matching vanilla's blank-chunk default for `EnumSkyBlock.SKY`
    /// (`EnumSkyBlock.SKY.defaultLightValue`, `Chunk.getLightSubtracted`'s null-storage arm).
    #[must_use]
    pub fn sky_light_at(&self, p: BlockPos) -> u8 {
        if !self.has_sky_light || !(0..256).contains(&p.y) {
            return 0;
        }
        let (x, z) = local_xz(p);
        match self.chunks.get(&p.chunk_pos()) {
            None => 15,
            Some(chunk) => match chunk.section((p.y >> 4) as u8) {
                Some(section) => section.sky_light(x, (p.y & 15) as u8, z),
                // java: Chunk.getLightSubtracted's null-storage arm distinguishes an empty
                // section below the height map from open sky. EmptyChunk (the unloaded case
                // above) still reports EnumSkyBlock.SKY's default 15.
                None if chunk.can_see_sky(x, p.y, z) => 15,
                None => 0,
            },
        }
    }

    /// Whether the current dimension has a sky (hazard H6: the Nether and the End do not, and
    /// their chunk packets omit the sky-light region entirely). Set this from the join/respawn
    /// dimension id before feeding it chunk packets, or block/sky light for the wrong
    /// dimension will silently read as garbage-shifted bytes.
    #[inline]
    #[must_use]
    pub fn has_sky_light(&self) -> bool {
        self.has_sky_light
    }

    /// Sets the dimension's sky flag. java: `WorldProvider.getHasNoSky()`, inverted.
    #[inline]
    pub fn set_has_sky_light(&mut self, has_sky_light: bool) {
        self.has_sky_light = has_sky_light;
    }

    // ---- filled-map storage -------------------------------------------------------------

    /// Returns the client-side `map_<id>` record, if an `S34PacketMaps` packet has created it.
    #[inline]
    #[must_use]
    pub fn map_data(&self, id: i32) -> Option<&MapData> {
        self.map_data.get(&id)
    }

    /// Returns every client-side map record in unspecified hash-map order.
    pub fn map_data_records(&self) -> impl Iterator<Item = (i32, &MapData)> + '_ {
        self.map_data.iter().map(|(id, map)| (*id, map))
    }

    /// Returns the creation-time fields for a client-side `map_<id>` record, when the map was
    /// created locally. Server-only S34 map records have no center/dimension sidecar because
    /// those fields are not present on the packet.
    #[inline]
    #[must_use]
    pub fn map_data_metadata(&self, id: i32) -> Option<&MapDataMetadata> {
        self.map_data_metadata.get(&id)
    }

    /// Create the client world's fresh `map_0` record and return its memory-storage id.
    ///
    /// java: `ItemEmptyMap.onItemRightClick` plus `SaveDataMemoryStorage.getUniqueDataId`.
    /// `MapData.calculateMapCenter` is evaluated at scale zero before the provider dimension is
    /// stored. An incoming S34 for id `0` updates the pixel record but leaves this metadata
    /// untouched, matching Java's `MapData.setMapdataTo`.
    pub fn create_empty_map_data(&mut self, player_x: f64, player_z: f64, dimension: i32) -> i32 {
        const MAP_ID: i32 = 0;
        const MAP_SIZE: i32 = 128;

        let x_cell = floor_f64((player_x + 64.0) / f64::from(MAP_SIZE));
        let z_cell = floor_f64((player_z + 64.0) / f64::from(MAP_SIZE));
        let metadata = MapDataMetadata {
            x_center: x_cell
                .wrapping_mul(MAP_SIZE)
                .wrapping_add(MAP_SIZE / 2)
                .wrapping_sub(64),
            z_center: z_cell
                .wrapping_mul(MAP_SIZE)
                .wrapping_add(MAP_SIZE / 2)
                .wrapping_sub(64),
            dimension: dimension as i8,
        };

        // SaveDataMemoryStorage replaces an existing same-name record; a second client-side
        // empty map therefore resets map_0's scale/pixels and its creation metadata exactly as
        // the Java callback does.
        self.map_data.insert(MAP_ID, MapData::new(MAP_ID));
        self.map_data_metadata.insert(MAP_ID, metadata);
        MAP_ID
    }

    /// Loads or creates `map_<packet.map_id>` and applies an `S34PacketMaps` update.
    ///
    /// java: `NetHandlerPlayClient.handleMaps` -> `ItemMap.loadMapData` ->
    /// `S34PacketMaps.setMapdataTo`. The renderer is intentionally a separate consumer: once
    /// it owns a map texture it can read [`Self::map_data`] after this call.
    pub fn apply_map_packet(
        &mut self,
        packet: &mc_protocol::play::s2c::Maps,
    ) -> Result<(), MapUpdateError> {
        // Validate against a scratch record first. This ensures an impossible packet cannot
        // create `map_<id>` or partially replace an existing record.
        let mut next = self
            .map_data
            .get(&packet.map_id)
            .cloned()
            .unwrap_or_else(|| MapData::new(packet.map_id));
        next.apply_packet(packet)?;
        self.map_data.insert(packet.map_id, next);
        Ok(())
    }

    // ---- chunk load/unload -----------------------------------------------------------

    /// The chunk at `pos`, if loaded.
    #[inline]
    #[must_use]
    pub fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos).map(Arc::as_ref)
    }

    /// Whether a chunk is currently loaded at `pos`.
    #[inline]
    #[must_use]
    pub fn is_chunk_loaded(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// The number of currently loaded chunk columns.
    #[inline]
    #[must_use]
    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Every loaded chunk column, in **unspecified order**.
    ///
    /// java: `ChunkProviderClient.chunkListing`, which `makeString` and `getLoadedChunkCount`
    /// read — vanilla keeps the same list this iterates.
    ///
    /// Added `S21c → S08` row: without it, anything that wants to visit the whole
    /// loaded world (a parity scan, an F3 count, a mesher's dirty walk) has to guess a bounding
    /// box of chunk coordinates and probe [`World::chunk`] for each, which is both slower and
    /// silently *incomplete* — a column outside the guessed box is skipped rather than reported.
    ///
    /// The order is a `HashMap`'s and is not stable across runs, so a caller that needs
    /// determinism must sort. Nothing is promised beyond "each loaded column exactly once", and
    /// `loaded_chunks().count() == loaded_chunk_count()`.
    pub fn loaded_chunks(&self) -> impl Iterator<Item = (ChunkPos, &Chunk)> + '_ {
        self.chunks
            .iter()
            .map(|(pos, chunk)| (*pos, chunk.as_ref()))
    }

    /// Removes a chunk and marks its column — and, as vanilla does, the ring of columns around
    /// it — **dirty**, so the mesher rebuilds them as empty and the departing geometry actually
    /// disappears. java: `ChunkProviderClient.unloadChunk` + `WorldClient.doPreChunk`'s
    /// `markBlockRangeForRenderUpdate(x*16, 0, z*16, x*16+15, 256, z*16+15)` (`:166`)
    ///
    /// This previously did the *opposite* — it `remove`d those sections from the
    /// dirty set, so nothing ever told the renderer the column was gone and its GPU meshes
    /// stayed drawn until the budget evictor happened to reach them.
    pub fn unload_chunk(&mut self, pos: ChunkPos) {
        self.chunks.remove(&pos);
        self.mark_column_and_neighbours_dirty(pos);
        // java: free in vanilla — `chunkTileEntityMap` lives on the `Chunk` and goes with it.
        // See [`World::tile_entities`] for why this one is flat and has to say so.
        self.drop_tile_entities_in_chunk(pos);
    }

    /// Loads `pos` as a freshly-loaded, empty chunk column: no sections (all air) and biomes at
    /// [`Chunk::empty`]'s `0xFF` "unset" sentinel. Replaces whatever chunk was previously loaded
    /// at `pos`, the same as a real full-column packet would.
    ///
    /// There is no direct vanilla API equivalent. A single `S21PacketChunkData` with an empty
    /// full-column bitmask is the *unload* signal (see [`World::apply_chunk_data`]), while an
    /// empty-bitmask entry in `S26PacketMapChunkBulk` really does load an empty column (see
    /// [`World::apply_chunk_bulk`]). This helper exists so hand-built worlds need neither packet.
    ///
    /// Added `S13 → S08` row: [`World::set_block`] on an unloaded position writes
    /// is a no-op (not a resident column), and the only way for
    /// a caller outside this crate to create residency used to be faking a chunk
    /// packet through [`World::apply_chunk_data`] — indirect, and one bitmask/length bug away
    /// from a confusing [`WorldError`]. `mc_physics::scene::ensure_chunk` did exactly that; this
    /// method is the direct way to say what it meant, in the same "test ergonomics, not a
    /// vanilla behaviour" spirit as [`World::loaded_chunks`].
    pub fn load_empty_chunk(&mut self, pos: ChunkPos) {
        // `tile_entities` is flat in this Rust world, whereas vanilla keeps the map on each
        // Chunk. Replacing a resident column therefore has to drop its old entries explicitly,
        // just as `apply_chunk_data` does before installing a fresh full-column Chunk.
        self.drop_tile_entities_in_chunk(pos);
        self.chunks.insert(pos, Arc::new(Chunk::empty()));
        self.mark_chunk_dirty(pos);
    }

    /// Applies one `S21PacketChunkData` payload.
    ///
    /// `ground_up_continuous && primary_bitmask == 0` is the wire's chunk-unload shape
    /// (`NetHandlerPlayClient.handleChunkData`'s `doPreChunk(x, z, false)` branch); every
    /// other `ground_up_continuous` packet **replaces** whatever chunk was loaded at `pos`
    /// with a fresh one before filling it, exactly as `WorldClient.doPreChunk(x, z, true)` ->
    /// `ChunkProviderClient.loadChunk` unconditionally allocates a new `Chunk`. A
    /// non-`ground_up_continuous` packet patches the existing chunk in place; when `pos` is not
    /// loaded, it patches the shared blank chunk instead (`fillChunk` on `EmptyChunk`), so its
    /// sections show through every absent column. It never makes an absent column resident.
    ///
    /// Uses [`World::has_sky_light`] for whether the payload's sky-light region is present —
    /// set that from the dimension before calling this (H6).
    pub fn apply_chunk_data(
        &mut self,
        pos: ChunkPos,
        ground_up_continuous: bool,
        primary_bitmask: u16,
        data: &[u8],
    ) -> Result<(), WorldError> {
        if ground_up_continuous && primary_bitmask == 0 {
            self.unload_chunk(pos);
            return Ok(());
        }
        if ground_up_continuous {
            self.chunks.insert(pos, Arc::new(Chunk::empty()));
            // `doPreChunk(x, z, true)` -> `ChunkProviderClient.loadChunk` allocates a brand-new
            // `Chunk`, so its tile-entity map starts empty; the server re-sends the description
            // packets right behind this one (`PlayerManager.java:526-530`).
            self.drop_tile_entities_in_chunk(pos);
        }
        let has_sky_light = self.has_sky_light;
        if ground_up_continuous {
            let chunk = self
                .chunks
                .get_mut(&pos)
                .map(Arc::make_mut)
                .expect("full-column insertion immediately precedes lookup");
            chunk.fill_from_packet(true, primary_bitmask, has_sky_light, data)?;
            // java: `NetHandlerPlayClient.handleChunkData:768` — `if (!packetIn.func_149274_i() ||
            // !(provider instanceof WorldProviderSurface)) chunk.resetRelightChecks()`. A full
            // packet in the Nether/End arms the round-robin relight sweep.
            if !has_sky_light {
                chunk.reset_relight_checks();
            }
        } else if let Some(chunk) = self.chunks.get_mut(&pos).map(Arc::make_mut) {
            chunk.fill_from_packet(false, primary_bitmask, has_sky_light, data)?;
            // A partial packet arms the round-robin relight sweep for a resident column.
            chunk.reset_relight_checks();
        } else {
            // java: `getChunkFromChunkCoords` hands an absent column the provider's shared
            // `EmptyChunk`, whose `fillChunk` is the inherited one (`Chunk.java:1307-1380`): the
            // packet's sections overwrite the blank chunk's, and every absent column then reads
            // them through `getBlockState` (see `World::block_at`). Its `generateHeightMap` is an
            // `EmptyChunk` no-op; the one Rust runs here is unobservable on the blank chunk. A
            // short payload is rejected before anything is written, as before.
            self.blank_chunk
                .fill_from_packet(false, primary_bitmask, has_sky_light, data)?;
        }
        // java: `NetHandlerPlayClient.handleChunkData:768` — partial packets and non-surface
        // full packets both arm relight work. The absent-column branch above intentionally has
        // no resident light work to reset.
        if !ground_up_continuous || !has_sky_light {
            if let Some(chunk) = self.chunks.get_mut(&pos).map(Arc::make_mut) {
                chunk.reset_relight_checks();
            }
        }
        self.mark_chunk_dirty(pos);
        Ok(())
    }

    /// Applies an `S26PacketMapChunkBulk` payload: every entry is a full-column send
    /// (`ground_up_continuous = true` always, per `S21PacketChunkData.getExtractedData`'s
    /// call site in `S26PacketMapChunkBulk`'s constructor). `packet_has_sky_light` is the
    /// packet's `isOverworld` flag and is used only for its payload layout. The world's dimension
    /// sky state remains owned by join/respawn, matching `Chunk.fillChunk` consulting the
    /// `WorldProvider` rather than mutating it from this server-controlled packet field.
    pub fn apply_chunk_bulk(
        &mut self,
        packet_has_sky_light: bool,
        chunks: &[BulkChunkEntry<'_>],
    ) -> Result<(), WorldError> {
        for entry in chunks {
            let pos = ChunkPos::new(entry.chunk_x, entry.chunk_z);
            // java: `handleMapChunkBulk` unconditionally calls `doPreChunk(..., true)` for
            // every entry. In particular, bitmask zero here loads an all-air column; only the
            // single-S21 handler interprets full+zero as an unload.
            self.chunks.insert(pos, Arc::new(Chunk::empty()));
            self.drop_tile_entities_in_chunk(pos);
            let chunk = self
                .chunks
                .get_mut(&pos)
                .map(Arc::make_mut)
                .expect("bulk insertion immediately precedes lookup");
            chunk.fill_from_packet(
                true,
                entry.primary_bitmask,
                packet_has_sky_light,
                entry.data,
            )?;
            // The relight gate is based on the actual WorldProvider, not S26's layout flag.
            if !self.has_sky_light {
                chunk.reset_relight_checks();
            }
            self.mark_chunk_dirty(pos);
        }
        Ok(())
    }

    /// Installs a column that [`Chunk::from_full_column_packet`] unpacked (usually off the game
    /// thread) with **this world's** sky-light layout — the caller checks that — at `pos`.
    ///
    /// Same world effects as the successful full-column branch of [`World::apply_chunk_data`]
    /// and as one [`World::apply_chunk_bulk`] entry: the column replaces whatever was loaded, the
    /// column's tile entities are dropped, a sky-less dimension arms the relight sweep, and the
    /// 3×3 neighbourhood is marked dirty. Neither the tile-entity drop nor the dirty marks read
    /// the column, so doing the unpack first is unobservable (#15 / NET-1).
    pub fn install_full_column(&mut self, pos: ChunkPos, mut chunk: Chunk) {
        // java: `handleChunkData:768` / the bulk relight gate, on the actual WorldProvider.
        if !self.has_sky_light {
            chunk.reset_relight_checks();
        }
        self.chunks.insert(pos, Arc::new(chunk));
        self.drop_tile_entities_in_chunk(pos);
        self.mark_chunk_dirty(pos);
    }

    // ---- dirty set (feeds S11) ---------------------------------------------------------

    /// Every section touched since the last drain, in no particular order.
    pub fn dirty_sections(&self) -> impl Iterator<Item = ChunkSectionPos> + '_ {
        self.dirty.iter().copied()
    }

    /// Takes and clears the dirty set. The mesher (S11) calls this once per frame/tick to get
    /// its re-mesh work list.
    ///
    /// Returns a **`BTreeSet`**, not the `HashSet` it is stored in, so the work list is in a
    /// deterministic order. The set is keyed by position and iterating a `HashSet` yields
    /// `RandomState`'s order, which differs between processes — that order reached
    /// `MeshQueue::take_nearest` and, wherever two sections tie on distance from a stationary
    /// camera, decided which of them was built first and therefore which upload batch each landed
    /// in. Two replays of one capture produced identical geometry from different batch groupings.
    /// The tie-break in `take_nearest` is the real fix; this removes the nondeterminism at source.
    /// Found by T0.1's replay determinism harness.
    pub fn drain_dirty_sections(&mut self) -> BTreeSet<ChunkSectionPos> {
        std::mem::take(&mut self.dirty).into_iter().collect()
    }

    /// A newly-loaded column has to re-mesh its **neighbours** too, or their faces
    /// on the shared boundary keep the geometry and the sky-light-15 they were built with while
    /// this column was still "unloaded = air". The body is
    /// [`World::mark_column_and_neighbours_dirty`], which is vanilla's actual box.
    fn mark_chunk_dirty(&mut self, pos: ChunkPos) {
        self.mark_column_and_neighbours_dirty(pos);
    }

    // The former `mark_dirty_for_block` lives in [`crate::light`] as `World::mark_dirty_box`
    // now: GAP A12-09 replaced its six face-neighbour tests with vanilla's actual box —
    // `(p-1)..(p+1)` bucketed by 16 through `ViewFrustum.markBlocksForUpdate`'s triple nested
    // loop, which is up to eight sections and includes the diagonals AO samples from.

    // ---- entity arena (frozen §7 API) ---------------------------------------------------

    /// Spawns a new entity with the given component bundle and returns its arena index.
    pub fn spawn(&mut self, b: EntityBundle) -> Entity {
        self.ecs
            .spawn((b.position, b.velocity, b.rotation, b.kind, b.metadata))
    }

    /// Despawns an entity. Errors if it does not exist (already despawned, or never spawned
    /// in this `World`).
    pub fn despawn(&mut self, e: Entity) -> Result<(), WorldError> {
        Ok(self.ecs.despawn(e)?)
    }
}

// =====================================================================================
// Placement predicates
// =====================================================================================

/// Vanilla block ids this module names. Written out rather than looked up by registry name so
/// the hot predicates stay branch-only. Cross-checked against `Block.registerBlocks`.
mod block_id {
    use mc_data::BlockId;

    pub const GRASS: BlockId = BlockId(2);
    pub const DIRT: BlockId = BlockId(3);
    pub const SAND: BlockId = BlockId(12);
    pub const GLASS: BlockId = BlockId(20);
    pub const REEDS: BlockId = BlockId(83);
    pub const GLOWSTONE: BlockId = BlockId(89);
    pub const CACTUS: BlockId = BlockId(81);
    pub const ICE: BlockId = BlockId(79);
    pub const PACKED_ICE: BlockId = BlockId(174);
    pub const STAINED_GLASS: BlockId = BlockId(95);
    pub const COBBLESTONE_WALL: BlockId = BlockId(139);
    pub const FARMLAND: BlockId = BlockId(60);
    pub const SOUL_SAND: BlockId = BlockId(88);
    pub const MYCELIUM: BlockId = BlockId(110);
    pub const HARDENED_CLAY: BlockId = BlockId(172);
    pub const STAINED_HARDENED_CLAY: BlockId = BlockId(159);
    /// `Blocks.water` is the *still* block (9). `Blocks.flowing_water` (8) is a different
    /// `Block` object, and `BlockLilyPad.canPlaceBlockOn` compares by identity.
    pub const WATER: BlockId = BlockId(9);
    pub const SNOW_LAYER: BlockId = BlockId(78);
    pub const HOPPER: BlockId = BlockId(154);
}

/// The `instanceof` tests the placement predicates make, resolved to the block ids that
/// actually satisfy them in 1.8.9.
///
/// A Java `instanceof BlockSlab` cannot be answered from `BlockInfo::java_class`, which names
/// only the *concrete* class (`BlockHalfStoneSlab`, never `BlockSlab`). Each list below was
/// produced by reflecting `Block.getBlockById(0..255)` against the family class under the real
/// 1.8.9 classes, so it is the registry's own answer and not a reading of the source.
mod family {
    use mc_data::BlockId;

    /// `instanceof BlockStairs`.
    pub fn stairs(id: BlockId) -> bool {
        matches!(
            id.0,
            53 | 67 | 108 | 109 | 114 | 128 | 134 | 135 | 136 | 156 | 163 | 164 | 180
        )
    }

    /// `instanceof BlockSlab` - both the single and the double form of all three families.
    pub fn slab(id: BlockId) -> bool {
        matches!(id.0, 43 | 44 | 125 | 126 | 181 | 182)
    }

    /// `instanceof BlockFence`. Fence *gates* are a separate class and are deliberately absent.
    pub fn fence(id: BlockId) -> bool {
        matches!(id.0, 85 | 113 | 188 | 189 | 190 | 191 | 192)
    }

    /// `instanceof BlockBush`, the class whose `canPlaceBlockAt` adds the ground test.
    pub fn bush(id: BlockId) -> bool {
        matches!(
            id.0,
            6 | 31 | 32 | 37 | 38 | 39 | 40 | 59 | 104 | 105 | 111 | 115 | 141 | 142 | 175
        )
    }
}

impl World {
    // ---- the two `World` statics the placement predicates lean on -----------------------

    /// java: `World.doesBlockHaveSolidTopSurface(IBlockAccess, BlockPos)` (`World.java:2419`).
    ///
    /// Reads through `IBlockAccess`, so - unlike [`World::is_block_normal_cube`] - it has no
    /// chunk-residency escape hatch: an unloaded position is simply air.
    ///
    /// The four `instanceof` arms after the opaque-full-cube test are reached only when that
    /// test fails, which is why a *double* slab (opaque and full) never reaches the `BlockSlab`
    /// arm - it has no `HALF` property to read and asking for one would throw in Java.
    #[must_use]
    pub fn does_block_have_solid_top_surface(&self, pos: BlockPos) -> bool {
        let state = self.block_at(pos);
        let props = state.props();
        if props.material.info().opaque && props.full_cube {
            return true;
        }
        let id = props.block;
        if family::stairs(id) {
            // java: `BlockStairs.HALF`, `EnumHalf.TOP` - `getStateFromMeta` reads `meta & 4`.
            return state.meta() & 4 != 0;
        }
        if family::slab(id) {
            // java: `BlockSlab.HALF`, `EnumBlockHalf.TOP` - `meta & 8`.
            return state.meta() & 8 != 0;
        }
        if id == block_id::HOPPER {
            return true;
        }
        if id == block_id::SNOW_LAYER {
            // java: `LAYERS == 7`, and `BlockSnow.getMetaFromState` stores `layers - 1`, so the
            // eight-layer (full-height) state is meta 7 and this one is meta 6.
            return state.meta() == 6;
        }
        false
    }

    /// java: `World.isBlockNormalCube(BlockPos, boolean)` (`World.java:2429`).
    ///
    /// **Not the same predicate as `Block.isNormalCube()`**, which also requires
    /// `!canProvidePower()`. This one is `material.isOpaque() && isFullCube()` only, so a
    /// redstone block is a "normal cube" here and is not one there. `BlockTorch` is the caller
    /// that cares.
    ///
    /// `default` is returned for an invalid position or an unreceived chunk, matching
    /// `ChunkProviderClient`'s blank `EmptyChunk` whose `isEmpty()` is true.
    #[must_use]
    pub fn is_block_normal_cube(&self, pos: BlockPos, default: bool) -> bool {
        // java: `World.isValid` - the 30-million-block horizontal limit and `0 <= y < 256`.
        if !(0..256).contains(&pos.y)
            || pos.x < -30_000_000
            || pos.z < -30_000_000
            || pos.x >= 30_000_000
            || pos.z >= 30_000_000
        {
            return default;
        }
        if !self.is_chunk_loaded(pos.chunk_pos()) {
            return default;
        }
        let props = self.block_at(pos).props();
        props.material.info().opaque && props.full_cube
    }

    /// Runs `BlockTrapDoor.onBlockActivated`'s complete client-side body.
    ///
    /// The caller supplies the dispatch boundary (the clicked block and `!sneaking || hand is
    /// empty` guard from `PlayerControllerMP`), while this method owns the exact state mutation
    /// and local level event.  Wooden trapdoors flip only metadata bit four (`OPEN`); iron
    /// trapdoors return `true` without changing state or making a sound.  Non-trapdoors return
    /// `None` so the broader activation dispatcher can continue with other virtual bodies.
    ///
    /// java: `BlockTrapDoor.onBlockActivated` (`BlockTrapDoor.java:127-140`).
    #[must_use]
    pub fn activate_trapdoor(&mut self, pos: BlockPos) -> Option<LocalBlockActivation> {
        let state = self.block_at(pos);
        if !matches!(state.block().0, 96 | 167) {
            return None;
        }

        // Java checks `this.blockMaterial == Material.iron`, not the state.  The two registered
        // BlockTrapDoor instances are ids 96 (wood) and 167 (iron); retaining the material check
        // makes the condition mirror the virtual body's source exactly.
        if state.props().material == Material::Iron {
            return Some(LocalBlockActivation {
                consumed: true,
                effect: None,
            });
        }

        // `BlockTrapDoor.OPEN` is a typed boolean property.  Java's `cycleProperty(OPEN)` keeps
        // `FACING` and `HALF` untouched while selecting the sibling state, rather than making
        // this caller know the metadata bit layout.
        let toggled = state
            .cycle_property(Property::boolean("open"))
            .expect("registered trapdoor state must declare OPEN");
        self.set_block(pos, toggled);
        let opened = toggled.meta() & 4 != 0;
        Some(LocalBlockActivation {
            consumed: true,
            effect: Some(LocalBlockEffect {
                id: if opened { 1003 } else { 1006 },
                pos,
                data: 0,
            }),
        })
    }

    /// Runs `BlockDoor.onBlockActivated`'s client-visible body.
    ///
    /// Iron doors consume the click without changing state. Wooden doors clicked on their
    /// upper half resolve and mutate only the matching lower half, while the auxiliary effect
    /// remains anchored at the original clicked position. A malformed upper half whose lower
    /// block is not the same door returns Java's `false` result without a local mutation.
    ///
    /// java: `BlockDoor.onBlockActivated` (`BlockDoor.java:156-177`).
    #[must_use]
    pub fn activate_door(&mut self, pos: BlockPos) -> Option<LocalBlockActivation> {
        let clicked = self.block_at(pos);
        if !matches!(clicked.block().0, 64 | 71 | 193 | 194 | 195 | 196 | 197) {
            return None;
        }
        // Java checks the BlockDoor instance's material, not the metadata-dependent state
        // properties. Raw upper-half metadata is not a canonical registered storage state,
        // so block identity is the faithful (and total) discriminator here.
        if clicked.block() == BlockId(71) {
            return Some(LocalBlockActivation {
                consumed: true,
                effect: None,
            });
        }

        let lower_pos = if clicked.meta() & 8 != 0 {
            pos.down()
        } else {
            pos
        };
        let lower = self.block_at(lower_pos);
        if lower.block() != clicked.block() || lower.meta() & 8 != 0 {
            return Some(LocalBlockActivation {
                consumed: false,
                effect: None,
            });
        }

        let toggled = lower
            .cycle_property(Property::boolean("open"))
            .expect("registered lower door state must declare OPEN");
        self.set_block(lower_pos, toggled);
        Some(LocalBlockActivation {
            consumed: true,
            effect: Some(LocalBlockEffect {
                id: if toggled.meta() & 4 != 0 { 1003 } else { 1006 },
                pos,
                data: 0,
            }),
        })
    }

    /// Runs `BlockFenceGate.onBlockActivated`'s complete client-side body.
    ///
    /// A closed gate opens immediately. If the player is standing on the gate's back side,
    /// vanilla first reverses the stored horizontal facing to the player's facing; closing an
    /// open gate never changes facing. The `POWERED` bit and every unrelated metadata bit are
    /// preserved, and all six wooden gate registrations share this body.
    ///
    /// java: `BlockFenceGate.onBlockActivated` (`BlockFenceGate.java:148-170`).
    #[must_use]
    pub fn activate_fence_gate(
        &mut self,
        pos: BlockPos,
        player_yaw: f32,
    ) -> Option<LocalBlockActivation> {
        let state = self.block_at(pos);
        if !matches!(state.block().0, 107 | 183..=187) {
            return None;
        }

        let mut meta = state.meta();
        let effect_id = if meta & 4 != 0 {
            meta &= !4;
            1006
        } else {
            let player_facing = Facing::from_angle(f64::from(player_yaw))
                .horizontal_index()
                .expect("yaw always produces a horizontal facing");
            if meta & 3 == (player_facing ^ 2) {
                meta = (meta & !3) | player_facing;
            }
            meta |= 4;
            1003
        };

        self.set_block(pos, BlockState::new(state.block(), meta));
        Some(LocalBlockActivation {
            consumed: true,
            effect: Some(LocalBlockEffect {
                id: effect_id,
                pos,
                data: 0,
            }),
        })
    }

    /// Runs `BlockButton.onBlockActivated`'s client-visible state body.
    ///
    /// Unlike the server-only tick/neighbor work, the button callback has no `isRemote` guard:
    /// an unpressed wood or stone button immediately cycles `POWERED` on the client and returns
    /// `true`, while an already-pressed button still returns `true` without another write. The
    /// server's later block update remains authoritative and can overwrite this prediction.
    /// Button click audio is intentionally not fabricated here: Java's multiplayer client's
    /// `World.playSoundEffect` reaches `RenderGlobal.playSound`, whose body is empty.
    ///
    /// java: `BlockButton.onBlockActivated` (`BlockButton.java:167-182`).
    #[must_use]
    pub fn activate_button(&mut self, pos: BlockPos) -> Option<LocalBlockActivation> {
        let state = self.block_at(pos);
        if !matches!(state.block().0, 77 | 143) {
            return None;
        }

        // Java returns true for an already-powered button, suppressing the held item's fallback
        // without scheduling another local write.
        if state.meta() & 8 != 0 {
            return Some(LocalBlockActivation {
                consumed: true,
                effect: None,
            });
        }

        let powered = state
            .cycle_property(Property::boolean("powered"))
            .expect("registered button state must declare POWERED");
        self.set_block(pos, powered);
        Some(LocalBlockActivation {
            consumed: true,
            effect: None,
        })
    }

    /// Runs the client-side half of `BlockSign.onBlockActivated`.
    ///
    /// `BlockSign` returns `true` immediately when `worldIn.isRemote`, before it looks up the
    /// sign tile entity or calls the server-only `TileEntitySign.executeCommand` body.  The
    /// client therefore consumes a standing or wall-sign click without a local mutation or
    /// level event.  The application-layer caller supplies the Java `!sneaking || held == null`
    /// guard and uses [`LocalBlockActivation::consumed`] to suppress the later item-use path.
    ///
    /// java: `BlockSign.onBlockActivated` (`BlockSign.java:85-95`).
    #[must_use]
    pub fn activate_sign(&self, pos: BlockPos) -> Option<LocalBlockActivation> {
        if matches!(self.block_at(pos).block().0, 63 | 68) {
            Some(LocalBlockActivation {
                consumed: true,
                effect: None,
            })
        } else {
            None
        }
    }

    /// Dispatches the client-visible `Block.onBlockClicked` overrides.
    ///
    /// Java calls this exactly once when a fresh survival dig begins and the previous local
    /// damage is zero.  This method is deliberately read-only: `clickBlock` has an immutable
    /// world view while it decides the digging packet, and the returned marker lets the caller
    /// apply only the local effects it can own.  `RedstoneOre`'s state write is exposed through
    /// [`World::activate_redstone_ore_click`], while cake needs the player's `canEat(false)`
    /// result and dragon-egg particles need the application's particle RNG.
    ///
    /// The complete 1.8.9 override set is represented: redstone ore (73/74), dragon egg (122),
    /// and cake (92) are client-relevant; note blocks (25) are server-gated and stairs delegate
    /// to their ordinary model block, whose base callback is empty.
    #[must_use]
    pub fn on_block_clicked(&self, pos: BlockPos) -> Option<LocalBlockClick> {
        let state = self.block_at(pos);
        match state.block().0 {
            73 | 74 => Some(LocalBlockClick::RedstoneOre { pos }),
            92 => Some(LocalBlockClick::Cake {
                pos,
                bites: state.meta() & 7,
            }),
            122 => Some(LocalBlockClick::DragonEgg { pos }),
            // java: BlockNote.onBlockClicked is `if (!worldIn.isRemote)`, and BlockStairs
            // delegates to a model block that has no client-local onBlockClicked body in 1.8.9.
            _ => None,
        }
    }

    /// Applies `BlockRedstoneOre.activate`'s client-local state half.
    ///
    /// Both lit and unlit ore request the particle burst.  Only unlit ore changes state; the
    /// later server block update remains authoritative.  Particle generation belongs to the
    /// application/render owner and is therefore not fabricated here.
    pub fn activate_redstone_ore_click(&mut self, pos: BlockPos) -> bool {
        if self.block_at(pos).block().0 != 73 {
            return false;
        }
        self.set_block(pos, BlockState::new(BlockId(74), 0));
        true
    }

    /// Applies `BlockCake.eatCake`'s block-state half after the caller has passed
    /// `EntityPlayer.canEat(false)`.
    ///
    /// The caller owns food stats and the achievement.  A stale/non-cake position is rejected,
    /// making it safe to run after an intervening server block update.
    pub fn consume_cake_slice(&mut self, pos: BlockPos) -> bool {
        let state = self.block_at(pos);
        if state.block().0 != 92 {
            return false;
        }
        let bites = state.meta() & 7;
        if bites < 6 {
            self.set_block(pos, BlockState::new(BlockId(92), bites + 1));
        } else {
            self.set_block(pos, BlockState::AIR);
        }
        true
    }

    // ---- the placement chain -------------------------------------------------------------

    /// java: `Block.canPlaceBlockOnSide(World, BlockPos, EnumFacing)` (`Block.java:841`) and
    /// all seven vanilla overrides of it — five real ones (`BlockButton`, `BlockLever`,
    /// `BlockTrapDoor`, `BlockTripWireHook`, `BlockVine`) and two constant `false`s
    /// (`BlockPistonExtension`, `BlockPistonMoving`).
    ///
    /// This is the second half of `World.canBlockBePlaced` (`World.java:3153`):
    ///
    /// ```text
    /// block.getMaterial().isReplaceable() && blockIn.canReplace(this, pos, side, itemStackIn)
    /// ```
    ///
    /// `Block.canReplace` (`:833`) is a one-line delegate to this, so a caller reproducing
    /// `canBlockBePlaced` wants
    /// `existing.props().material.is_replaceable() && world.can_place_block_on_side(..)`.
    /// The default arm re-tests `replaceable` itself, which is why the pre-override code could
    /// collapse the whole chain into that one field.
    ///
    /// `pos` is the cell the block would occupy - already stepped along `side` by
    /// `ItemBlock.canPlaceBlockOnSide` where that applies - and `side` is the face that was
    /// clicked, **reassigned to `Up` when the clicked block is a snow layer**
    /// (`ItemBlock.java:128-136`). That reassignment used to be unobservable and is not any
    /// more: `side` reaches the six overrides below.
    #[must_use]
    pub fn can_place_block_on_side(&self, block: BlockId, pos: BlockPos, side: Facing) -> bool {
        match block.0 {
            // java: `BlockPistonExtension:93` / `BlockPistonMoving:70` - constant `false`.
            34 | 36 => false,
            // java: `BlockButton.canPlaceBlockOnSide:66` and `BlockLever:52`, both
            // `func_181088_a(world, pos, side.getOpposite())`.
            69 | 77 | 143 => self.button_can_attach(pos, side.opposite()),
            // java: `BlockTrapDoor:203`.
            96 | 167 => {
                !side.axis().is_vertical()
                    && trapdoor_support(self.block_at(pos.relative(side.opposite())))
            }
            // java: `BlockTripWireHook:67`.
            131 => {
                side.axis().is_horizontal() && self.is_normal_cube_at(pos.relative(side.opposite()))
            }
            // java: `BlockVine:157` - a `switch` with no `DOWN` arm, so a vine clicked on a
            // floor is refused outright. The `UP` arm reads `pos.up()`, *not* the opposite.
            106 => match side {
                Facing::Up => vine_can_place_on(self.block_at(pos.up())),
                Facing::North | Facing::South | Facing::East | Facing::West => {
                    vine_can_place_on(self.block_at(pos.relative(side.opposite())))
                }
                Facing::Down => false,
            },
            _ => self.can_place_block_at(block, pos),
        }
    }

    /// java: `Block.canPlaceBlockAt(World, BlockPos)` (`Block.java:846`) and all 28 overrides
    /// of it (29 declarations in `net/minecraft/block/`, one of which is `Block`'s own).
    ///
    /// The default is `worldIn.getBlockState(pos).getBlock().blockMaterial.isReplaceable()` -
    /// a test on what is *already* at `pos`, not on the block being placed. Several overrides
    /// (chest, ladder, rails, torch, button, lever, tripwire hook, snow, reeds, fire, redstone
    /// wire, pressure plates) never call `super` and therefore drop that test entirely;
    /// `World.canBlockBePlaced` applies its own `isReplaceable` before reaching here, so the
    /// omission is unobservable through the placement path and deliberate here.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn can_place_block_at(&self, block: BlockId, pos: BlockPos) -> bool {
        let default = || self.block_at(pos).props().material.is_replaceable();
        let below = pos.down();

        match block.0 {
            // java: `BlockPistonExtension:85` / `BlockPistonMoving:62`.
            34 | 36 => false,

            // java: `BlockChest:329`. Never calls `super`: a chest refuses a cell only because
            // of its neighbours. `i <= 1` - one adjacent single chest is fine (that is how a
            // double chest is made), two are not, and an adjacent chest that is *already* half
            // of a double is refused immediately.
            54 | 146 => {
                let mut adjacent = 0;
                for side in Facing::HORIZONTALS {
                    let neighbour = pos.relative(side);
                    if self.block_at(neighbour).props().block == block {
                        if self.is_double_chest(block, neighbour) {
                            return false;
                        }
                        adjacent += 1;
                    }
                }
                adjacent <= 1
            }

            // java: `BlockLadder:82` - any of the four horizontal neighbours a normal cube.
            // `Block.isNormalCube()`, the *virtual*, so `!canProvidePower()` applies here and a
            // redstone block will not hold a ladder.
            65 => Facing::HORIZONTALS
                .iter()
                .any(|&s| self.is_normal_cube_at(pos.relative(s))),

            // java: `BlockRailBase:85`.
            27 | 28 | 66 | 157 => self.does_block_have_solid_top_surface(below),

            // java: `BlockTorch:71` - true if *any* allowed `FACING` value would hold it.
            // `BlockTorch.FACING` is `PropertyDirection.create("facing", <not DOWN>)`
            // (`BlockTorch:29`), so the loop never tries a ceiling torch.
            50 | 75 | 76 => Facing::VALUES
                .iter()
                .filter(|f| **f != Facing::Down)
                .any(|&f| self.torch_can_place_at(pos, f)),

            // java: `BlockButton:71` / `BlockLever:57` - every one of the six facings, `DOWN`
            // included (a button on a ceiling).
            69 | 77 | 143 => Facing::VALUES
                .iter()
                .any(|&f| self.button_can_attach(pos, f)),

            // java: `BlockTripWireHook:72`.
            131 => Facing::HORIZONTALS
                .iter()
                .any(|&s| self.is_normal_cube_at(pos.relative(s))),

            // java: `BlockPumpkin:116` - spelled out rather than `super &&`, same answer.
            86 | 91 => default() && self.does_block_have_solid_top_surface(below),

            // java: `BlockSnow:82`. Ice and packed ice refuse outright; leaves always accept; a
            // full-depth snow layer accepts; otherwise the support must be an opaque cube whose
            // material blocks movement.
            78 => {
                let state = self.block_at(below);
                let props = state.props();
                let id = props.block;
                if id == block_id::ICE || id == block_id::PACKED_ICE {
                    false
                } else if props.material == Material::Leaves {
                    true
                } else if id == block_id::SNOW_LAYER && state.meta() >= 6 {
                    // java: `LAYERS >= 7`, stored as `layers - 1`.
                    true
                } else {
                    props.opaque_cube && props.material.info().blocks_movement
                }
            }

            // java: `BlockCarpet:74` + `canBlockStay:101` - `!worldIn.isAirBlock(pos.down())`,
            // which is a *material* test (`World.isAirBlock` is `material == Material.air`),
            // not a block-id one. Only `Blocks.air` has it, so the two coincide here.
            171 => default() && self.block_at(below).props().material != Material::Air,

            // java: `BlockCactus:88` + `canBlockStay:104`. No horizontal neighbour may have a
            // solid material, and the support must be cactus or sand.
            81 => {
                if !default() {
                    return false;
                }
                if Facing::HORIZONTALS
                    .iter()
                    .any(|&s| self.block_at(pos.relative(s)).props().material.info().solid)
                {
                    return false;
                }
                let ground = self.block_at(below).props().block;
                ground == block_id::CACTUS || ground == block_id::SAND
            }

            // java: `BlockCake:105` + `canBlockStay:121`.
            92 => default() && self.block_at(below).props().material.info().solid,

            // java: `BlockFenceGate:47`.
            107 | 183..=187 => self.block_at(below).props().material.info().solid && default(),

            // java: `BlockDoor:285`. Both the lower cell and the one above it must be
            // replaceable, and y >= 255 is refused before anything is read.
            64 | 71 | 193..=197 => {
                pos.y < 255
                    && self.does_block_have_solid_top_surface(below)
                    && default()
                    && self.block_at(pos.up()).props().material.is_replaceable()
            }

            // java: `BlockRedstoneDiode:32`.
            93 | 94 | 149 | 150 => self.does_block_have_solid_top_surface(below) && default(),

            // java: `BlockRedstoneWire:98` - glowstone is the one non-solid-top exception.
            55 => {
                self.does_block_have_solid_top_surface(below)
                    || self.block_at(below).props().block == block_id::GLOWSTONE
            }

            // java: `BlockBasePressurePlate:88` + `canBePlacedOn:105`.
            70 | 72 | 147 | 148 => {
                self.does_block_have_solid_top_surface(below)
                    || family::fence(self.block_at(below).props().block)
            }

            // java: `BlockFlowerPot:169`.
            140 => default() && self.does_block_have_solid_top_surface(below),

            // java: `BlockSign.canPlaceBlockAt` (`BlockSign.java:98`) - `BlockContainer`'s
            // cactus-neighbour guard plus `Block.canPlaceBlockAt`'s replaceable-cell default.
            63 | 68 => self.can_place_sign_at(pos),

            // java: `BlockBanner.canPlaceBlockAt` (`BlockBanner.java:128`) has the same
            // inherited `BlockContainer.hasInvalidNeighbor` shape, but remains a separate arm
            // so the sign-specific helper does not accidentally claim banner semantics.
            176 | 177 => !self.has_invalid_neighbor(pos) && default(),

            // java: `BlockReed:63`.
            83 => {
                let ground = self.block_at(below).props().block;
                if ground == block_id::REEDS {
                    true
                } else if ground != block_id::GRASS
                    && ground != block_id::DIRT
                    && ground != block_id::SAND
                {
                    false
                } else {
                    Facing::HORIZONTALS.iter().any(|&s| {
                        self.block_at(pos.relative(s).down()).props().material == Material::Water
                    })
                }
            }

            // java: `BlockFire:366`.
            51 => {
                self.does_block_have_solid_top_surface(below)
                    || Facing::VALUES
                        .iter()
                        .any(|&f| can_catch_fire(self.block_at(pos.relative(f)).props().block))
            }

            // java: `BlockDoublePlant:61` - `BlockBush`'s chain plus an air cell above for the
            // upper half. `BlockDoublePlant` does not override `canPlaceBlockOn`, so the ground
            // test is `BlockBush`'s grass/dirt/farmland.
            175 => {
                default()
                    && self.bush_can_block_stay(block, pos)
                    && self.block_at(pos.up()).props().material == Material::Air
            }

            // java: `BlockMushroom:59` - `BlockBush`'s chain plus `canBlockStay`.
            39 | 40 => {
                default()
                    && bush_can_place_on(block, self.block_at(below).props().block)
                    && self.mushroom_can_block_stay(block, pos)
            }

            // java: `BlockBush:35`, for the remaining bushes.
            id if family::bush(BlockId(id)) => default() && self.bush_can_block_stay(block, pos),

            // java: `BlockStairs:615` - `this.modelBlock.canPlaceBlockAt(worldIn, pos)`. Every
            // vanilla stairs block's model block is a plain `Block` (planks 5, cobblestone 4,
            // brick 45, stonebrick 98, nether_brick 112, sandstone 24, quartz 155,
            // red_sandstone 179 - `Block.registerBlocks:1310/1325/1369/1370/1376/1390/1419` and
            // the five wood variants), none of which overrides `canPlaceBlockAt`, so the
            // delegation lands on this same default in all thirteen cases.
            _ => default(),
        }
    }

    /// java: `Block.isNormalCube()` read through the world at `pos`, i.e.
    /// `worldIn.getBlockState(pos).getBlock().isNormalCube()`. **Not**
    /// [`World::is_block_normal_cube`], which is `World`'s laxer static of the same name.
    /// Runs `BlockSign.canPlaceBlockAt` for either concrete sign block.
    ///
    /// The superclass check is deliberately on the material already at `pos`, not on the
    /// sign's own wood material. `BlockContainer.hasInvalidNeighbor` examines only the four
    /// horizontal neighbours and compares their material by identity with `Material.cactus`;
    /// vertical and diagonal cactus blocks do not invalidate a sign.
    ///
    /// java: `BlockSign.canPlaceBlockAt` (`BlockSign.java:98-101`) and
    /// `BlockContainer.hasInvalidNeighbor` (`BlockContainer.java:24-29`).
    #[must_use]
    pub fn can_place_sign_at(&self, pos: BlockPos) -> bool {
        !self.sign_has_invalid_neighbor(pos) && self.block_at(pos).props().material.is_replaceable()
    }

    /// Read `Block.isNormalCube()` for the live state at `pos`.
    ///
    /// ItemDoor's hinge chooser counts this predicate in both adjacent two-cell columns.
    #[must_use]
    pub fn is_normal_cube_at(&self, pos: BlockPos) -> bool {
        self.block_at(pos).props().normal_cube
    }

    /// java: `BlockButton.func_181088_a` (`BlockButton.java:84`), shared verbatim by
    /// `BlockLever.func_181090_a` (`BlockLever.java:70`).
    ///
    /// `facing` is the direction from `pos` towards the *support*. `DOWN` uses the solid-top
    /// test (a button standing on a floor); every other direction uses `isNormalCube`.
    fn button_can_attach(&self, pos: BlockPos, facing: Facing) -> bool {
        let support = pos.relative(facing);
        if facing == Facing::Down {
            self.does_block_have_solid_top_surface(support)
        } else {
            self.is_normal_cube_at(support)
        }
    }

    /// java: `BlockTorch.canPlaceAt` (`BlockTorch.java:84`) + `canPlaceOn` (`:58`).
    ///
    /// `facing` is the direction the torch *points*, so the support is at
    /// `pos.offset(facing.getOpposite())`. The horizontal case uses `World.isBlockNormalCube`
    /// with a `true` default - the laxer static, so a redstone block **does** hold a wall torch
    /// even though it will not hold a ladder - while the `UP` case adds fences, glass, stained
    /// glass and cobblestone walls to the solid-top test.
    fn torch_can_place_at(&self, pos: BlockPos, facing: Facing) -> bool {
        let support = pos.relative(facing.opposite());
        if facing.axis().is_horizontal() && self.is_block_normal_cube(support, true) {
            return true;
        }
        if facing != Facing::Up {
            return false;
        }
        if self.does_block_have_solid_top_surface(support) {
            return true;
        }
        let id = self.block_at(support).props().block;
        family::fence(id)
            || id == block_id::GLASS
            || id == block_id::COBBLESTONE_WALL
            || id == block_id::STAINED_GLASS
    }

    /// java: `BlockChest.isDoubleChest` (`BlockChest.java:380`).
    fn is_double_chest(&self, chest: BlockId, pos: BlockPos) -> bool {
        if self.block_at(pos).props().block != chest {
            return false;
        }
        Facing::HORIZONTALS
            .iter()
            .any(|&s| self.block_at(pos.relative(s)).props().block == chest)
    }

    /// java: `BlockContainer.hasInvalidNeighbor` (`BlockContainer.java:29`) - N, S, W, E only,
    /// and the test is on the *material*, so both cactus states qualify. This helper is public
    /// because the sign placement seam needs the exact valid-neighbour decision as a pure world
    /// query; banners reuse the same inherited Java body through the private wrapper.
    #[must_use]
    pub fn sign_has_invalid_neighbor(&self, pos: BlockPos) -> bool {
        [Facing::North, Facing::South, Facing::West, Facing::East]
            .iter()
            .any(|&s| self.block_at(pos.relative(s)).props().material == Material::Cactus)
    }

    /// The common `BlockContainer` cactus-neighbour predicate used by banner placement too.
    fn has_invalid_neighbor(&self, pos: BlockPos) -> bool {
        self.sign_has_invalid_neighbor(pos)
    }

    /// java: `BlockMushroom.canBlockStay` (`BlockMushroom.java:72`).
    ///
    /// Mycelium and podzol accept unconditionally; anything else needs `World.getLight(pos) <
    /// 13` as well as `isFullBlock()`.
    fn mushroom_can_block_stay(&self, mushroom: BlockId, pos: BlockPos) -> bool {
        if !(0..256).contains(&pos.y) {
            return false;
        }
        let state = self.block_at(pos.down());
        let id = state.props().block;
        if id == block_id::MYCELIUM {
            return true;
        }
        // java: `BlockDirt.VARIANT == PODZOL`, which is meta 2.
        if id == block_id::DIRT && state.meta() == 2 {
            return true;
        }
        self.light_at(pos) < 13 && bush_can_place_on(mushroom, id)
    }

    /// The shared `BlockBush.canBlockStay` ground predicate (`BlockBush.java:76`).
    ///
    /// `canPlaceBlockOn` is virtual, so `bush` is retained rather than hard-coding the base
    /// grass/dirt/farmland triplet: dead bushes, lily pads, crops, stems and nether wart all
    /// provide their own substrate predicate. Callers with additional state conditions (notably
    /// mushrooms and the two halves of a double plant) layer those conditions around this one.
    #[must_use]
    fn bush_can_block_stay(&self, bush: BlockId, pos: BlockPos) -> bool {
        bush_can_place_on(bush, self.block_at(pos.down()).props().block)
    }

    /// `BlockBush.canBlockStay` with the two virtual overrides that change its answer.
    fn bush_can_block_stay_for_state(&self, state: BlockState, pos: BlockPos) -> bool {
        let bush = state.props().block;
        match bush.0 {
            // java: `BlockDoublePlant.canBlockStay`.  Only a matching lower half supports an
            // upper half; a lower half needs the matching upper state as well as BlockBush's
            // ordinary substrate predicate.
            175 if state.meta() & 8 != 0 => self.block_at(pos.down()).props().block == BlockId(175),
            175 => {
                self.block_at(pos.up()).props().block == BlockId(175)
                    && self.bush_can_block_stay(bush, pos)
            }
            // java: `BlockMushroom.canBlockStay`, which replaces rather than augments the base
            // ground predicate (a full block in dim light, plus unconditional mycelium/podzol).
            39 | 40 => self.mushroom_can_block_stay(bush, pos),
            _ => self.bush_can_block_stay(bush, pos),
        }
    }
}

/// The `FACING` a torch's metadata stores. `BlockRedstoneTorch` inherits the same mapping.
///
/// Torch metadata is not an `EnumFacing` index, so `Facing::from_index(meta)` is wrong for every
/// torch except a south-facing one. It turned a standing torch (meta 5) into `EAST`, which put
/// its flame and smoke 0.27 blocks west of the flame and 0.22 blocks above it, and it put the
/// flame of an east-facing wall torch (meta 1) in the middle of the block.
// java: BlockTorch.java:244 (getStateFromMeta)
fn torch_facing(meta: u8) -> Facing {
    match meta {
        1 => Facing::East,
        2 => Facing::West,
        3 => Facing::South,
        4 => Facing::North,
        _ => Facing::Up,
    }
}

/// Client-side random display ticks and their render/audio-neutral recipes.
///
/// This is kept as a second `World` implementation block so the application layer can own the
/// particle arena and sound mixer. It mirrors `WorldClient.doVoidFogParticles`: the sampler uses
/// the persistent world RNG, while every block callback uses the fresh callback RNG passed by the
/// caller. In particular, particle construction must not be folded into this method because the
/// `EntityFX` factories have their own renderer-side random stream in vanilla.
impl World {
    /// Runs one vanilla `WorldClient.doVoidFogParticles` pass.
    ///
    /// `sample_rng` is the persistent world-sampling stream (`this.rand`); it consumes exactly
    /// six `nextInt(16)` draws per sample. `callback_rng` models the fresh `new Random()` handed
    /// to each `randomDisplayTick` body. The optional barrier branch is the creative-held-barrier
    /// reveal from the same Java method. Unloaded columns are omitted deliberately: vanilla's
    /// client `EmptyChunk` is air, and no packet scratch state may create ambient effects.
    #[must_use]
    pub fn do_void_fog_particles(
        &self,
        origin: BlockPos,
        sample_rng: &mut JavaRandom,
        callback_rng: &mut JavaRandom,
        reveal_barriers: bool,
    ) -> Vec<AmbientEffect> {
        self.do_void_fog_particles_with_rain(
            origin,
            sample_rng,
            callback_rng,
            reveal_barriers,
            false,
        )
    }

    /// `do_void_fog_particles` with the live rain gate used by `BlockLeaves.randomDisplayTick`.
    #[must_use]
    pub fn do_void_fog_particles_with_rain(
        &self,
        origin: BlockPos,
        sample_rng: &mut JavaRandom,
        callback_rng: &mut JavaRandom,
        reveal_barriers: bool,
        raining: bool,
    ) -> Vec<AmbientEffect> {
        let mut effects = Vec::new();
        // Every sample lies within origin ± 15 on x and z, so at most 3×3 columns: resolve them
        // once instead of two SipHash probes per sample (2,000+ per tick; PART-6/BOVL-5). Only
        // lookups move; every sample still draws its six values first.
        let columns = AmbientColumns::new(self, origin);
        for _ in 0..RANDOM_DISPLAY_TICK_SAMPLES {
            // java: `pos + rand.nextInt(16) - rand.nextInt(16)` on each axis. Keep the three
            // pairs separate; changing this order changes every later sample and every lit ore
            // callback that correctly consumes the world's sampling RNG. The sum is wrapping
            // `int` arithmetic in Java: `floor_double(posX)` of an S08-placed player saturates to
            // `Integer.MAX_VALUE`, and the samples wrap past it in every build profile.
            let pos = BlockPos::new(
                origin
                    .x
                    .wrapping_add(sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS))
                    .wrapping_sub(sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS)),
                origin
                    .y
                    .wrapping_add(sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS))
                    .wrapping_sub(sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS)),
                origin
                    .z
                    .wrapping_add(sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS))
                    .wrapping_sub(sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS)),
            );
            let state = columns.ambient_block_at(self, pos);
            if state == BlockState::AIR {
                continue;
            }
            // `random_display_tick_at_with_rain` would re-read this same state from this same
            // immutable world; of its rejections only the legality check can still fire.
            if state.is_legal() {
                effects.extend(self.random_display_tick_resolved(
                    pos,
                    state,
                    callback_rng,
                    sample_rng,
                    raining,
                ));
            }
            if reveal_barriers && state.props().block.0 == 166 {
                effects.push(AmbientEffect::Particle(AmbientParticleEffect {
                    particle_id: ambient_particle::BARRIER,
                    position: mc_math::DVec3::new(
                        f64::from(pos.x) + 0.5,
                        f64::from(pos.y) + 0.5,
                        f64::from(pos.z) + 0.5,
                    ),
                    velocity: mc_math::DVec3::ZERO,
                }));
            }
        }
        effects
    }

    /// Dispatches one loaded block's `randomDisplayTick` body.
    ///
    /// The callback receives its own random stream, and `world_rng` is used only by the one
    /// vanilla body (`BlockRedstoneOre`) that explicitly reaches back to `worldIn.rand` rather
    /// than consuming its callback argument. This method is useful to tests and to callers that
    /// already own the 1000-sample scheduler.
    #[must_use]
    pub fn random_display_tick_at(
        &self,
        pos: BlockPos,
        state: BlockState,
        callback_rng: &mut JavaRandom,
        world_rng: &mut JavaRandom,
    ) -> Vec<AmbientEffect> {
        self.random_display_tick_at_with_rain(pos, state, callback_rng, world_rng, false)
    }

    /// `random_display_tick_at` with the live world-rain gate for leaves.
    #[must_use]
    pub fn random_display_tick_at_with_rain(
        &self,
        pos: BlockPos,
        supplied_state: BlockState,
        callback_rng: &mut JavaRandom,
        world_rng: &mut JavaRandom,
        raining: bool,
    ) -> Vec<AmbientEffect> {
        let state = self.ambient_block_at(pos);
        if state == BlockState::AIR || !supplied_state.is_legal() || supplied_state != state {
            // The live world value is authoritative just like `getBlockState`; rejecting any
            // mismatched explicit state (including metadata) keeps direct callback audits from
            // dispatching a stale packet's class or variant against a newly changed block.
            return Vec::new();
        }
        self.random_display_tick_resolved(pos, state, callback_rng, world_rng, raining)
    }

    /// The `randomDisplayTick` dispatch for a `state` that is `ambient_block_at(pos)`, legal and
    /// not air.
    fn random_display_tick_resolved(
        &self,
        pos: BlockPos,
        state: BlockState,
        callback_rng: &mut JavaRandom,
        world_rng: &mut JavaRandom,
        raining: bool,
    ) -> Vec<AmbientEffect> {
        let id = state.props().block.0;
        let mut effects = Vec::new();
        let particle = |effects: &mut Vec<AmbientEffect>,
                        id: i32,
                        position: mc_math::DVec3,
                        velocity: mc_math::DVec3| {
            effects.push(AmbientEffect::Particle(AmbientParticleEffect {
                particle_id: id,
                position,
                velocity,
            }));
        };
        let sound = |effects: &mut Vec<AmbientEffect>,
                     name: &'static str,
                     position: mc_math::DVec3,
                     volume: f32,
                     pitch: f32| {
            effects.push(AmbientEffect::Sound(AmbientSoundEffect {
                name,
                position,
                volume,
                pitch,
            }));
        };
        let xyz = |x: f64, y: f64, z: f64| mc_math::DVec3::new(x, y, z);
        let zero = mc_math::DVec3::ZERO;
        let x = f64::from(pos.x);
        let y = f64::from(pos.y);
        let z = f64::from(pos.z);

        match id {
            // BlockBrewingStand.randomDisplayTick: three nextFloat draws and one smoke.
            117 => {
                particle(
                    &mut effects,
                    ambient_particle::SMOKE_NORMAL,
                    xyz(
                        x + 0.4 + f64::from(callback_rng.next_f32()) * 0.2,
                        y + 0.7 + f64::from(callback_rng.next_f32()) * 0.3,
                        z + 0.4 + f64::from(callback_rng.next_f32()) * 0.2,
                    ),
                    zero,
                );
            }
            // BlockEnchantmentTable.randomDisplayTick. Java mutates j from -1 to 2 for the
            // interior ring, so a while loop is clearer than a range that would visit skipped
            // cells and consume the wrong nextInt(16) draws.
            116 => {
                for i in -2..=2 {
                    let mut j = -2;
                    while j <= 2 {
                        if i > -2 && i < 2 && j == -1 {
                            j = 2;
                        }
                        if callback_rng.next_i32_bound(16) == 0 {
                            for k in 0..=1 {
                                let bookshelf = self.ambient_block_at(pos.add(i, k, j));
                                if bookshelf.props().block.0 != 47 {
                                    continue;
                                }
                                if self.ambient_block_at(pos.add(i / 2, 0, j / 2))
                                    != BlockState::AIR
                                {
                                    break;
                                }
                                particle(
                                    &mut effects,
                                    ambient_particle::ENCHANTMENT_TABLE,
                                    xyz(x + 0.5, y + 2.0, z + 0.5),
                                    xyz(
                                        f64::from(i) + f64::from(callback_rng.next_f32()) - 0.5,
                                        f64::from(k) - f64::from(callback_rng.next_f32()) - 1.0,
                                        f64::from(j) + f64::from(callback_rng.next_f32()) - 0.5,
                                    ),
                                );
                            }
                        }
                        j += 1;
                    }
                }
            }
            // BlockEndPortal.randomDisplayTick.
            119 => {
                particle(
                    &mut effects,
                    ambient_particle::SMOKE_NORMAL,
                    xyz(
                        x + f64::from(callback_rng.next_f32()),
                        y + 0.8,
                        z + f64::from(callback_rng.next_f32()),
                    ),
                    zero,
                );
            }
            // BlockEnderChest.randomDisplayTick.
            130 => {
                for _ in 0..3 {
                    let j = callback_rng.next_i32_bound(2) * 2 - 1;
                    let k = callback_rng.next_i32_bound(2) * 2 - 1;
                    let position = xyz(
                        x + 0.5 + 0.25 * f64::from(j),
                        y + f64::from(callback_rng.next_f32()),
                        z + 0.5 + 0.25 * f64::from(k),
                    );
                    let velocity = xyz(
                        f64::from(callback_rng.next_f32()) * f64::from(j),
                        (f64::from(callback_rng.next_f32()) - 0.5) * 0.125,
                        f64::from(callback_rng.next_f32()) * f64::from(k),
                    );
                    particle(&mut effects, ambient_particle::PORTAL, position, velocity);
                }
            }
            // BlockFire.randomDisplayTick.
            51 => {
                if callback_rng.next_i32_bound(24) == 0 {
                    sound(
                        &mut effects,
                        "fire.fire",
                        xyz(x + 0.5, y + 0.5, z + 0.5),
                        1.0 + callback_rng.next_f32(),
                        callback_rng.next_f32() * 0.7 + 0.3,
                    );
                }
                let below = pos.down();
                if !self.ambient_solid_top(below)
                    && !can_catch_fire(self.ambient_block_at(below).props().block)
                {
                    for (side, kind) in [
                        (Facing::West, 0_u8),
                        (Facing::East, 1),
                        (Facing::North, 2),
                        (Facing::South, 3),
                        (Facing::Up, 4),
                    ] {
                        if !can_catch_fire(self.ambient_block_at(pos.relative(side)).props().block)
                        {
                            continue;
                        }
                        for _ in 0..2 {
                            let a = callback_rng.next_f64();
                            let b = callback_rng.next_f64();
                            let c = callback_rng.next_f64();
                            let (px, py, pz) = match kind {
                                0 => (x + a * 0.1, y + b, z + c),
                                1 => (x + 1.0 - a * 0.1, y + b, z + c),
                                2 => (x + a, y + b, z + c * 0.1),
                                3 => (x + a, y + b, z + 1.0 - c * 0.1),
                                _ => (x + a, y + 1.0 - b * 0.1, z + c),
                            };
                            particle(
                                &mut effects,
                                ambient_particle::SMOKE_LARGE,
                                xyz(px, py, pz),
                                zero,
                            );
                        }
                    }
                } else {
                    for _ in 0..3 {
                        particle(
                            &mut effects,
                            ambient_particle::SMOKE_LARGE,
                            xyz(
                                x + callback_rng.next_f64(),
                                y + callback_rng.next_f64() * 0.5 + 0.5,
                                z + callback_rng.next_f64(),
                            ),
                            zero,
                        );
                    }
                }
            }
            // Only block id 62 has BlockFurnace.isBurning set in vanilla.
            62 => {
                let facing = Facing::from_index(i32::from(state.meta()));
                let facing = if facing.axis().is_horizontal() {
                    facing
                } else {
                    Facing::North
                };
                let d0 = x + 0.5;
                let d1 = y + callback_rng.next_f64() * 6.0 / 16.0;
                let d2 = z + 0.5;
                let d4 = callback_rng.next_f64() * 0.6 - 0.3;
                let (sx, sz) = match facing {
                    Facing::West => (d0 - 0.52, d2 + d4),
                    Facing::East => (d0 + 0.52, d2 + d4),
                    Facing::North => (d0 + d4, d2 - 0.52),
                    Facing::South => (d0 + d4, d2 + 0.52),
                    _ => (d0, d2),
                };
                particle(
                    &mut effects,
                    ambient_particle::SMOKE_NORMAL,
                    xyz(sx, d1, sz),
                    zero,
                );
                particle(&mut effects, ambient_particle::FLAME, xyz(sx, d1, sz), zero);
            }
            // BlockLeaves.randomDisplayTick. The weather gate is supplied by the environment
            // caller because `World` deliberately does not own the interpolated rain strength.
            18 | 161 => {
                if raining
                    && self.is_raining_at(pos.up(), true)
                    && !self.ambient_solid_top(pos.down())
                    && callback_rng.next_i32_bound(15) == 1
                {
                    particle(
                        &mut effects,
                        ambient_particle::DRIP_WATER,
                        xyz(
                            x + f64::from(callback_rng.next_f32()),
                            y - 0.05,
                            z + f64::from(callback_rng.next_f32()),
                        ),
                        zero,
                    );
                }
            }
            // BlockMycelium calls Block.randomDisplayTick first (empty), then this body.
            110 => {
                if callback_rng.next_i32_bound(10) == 0 {
                    particle(
                        &mut effects,
                        ambient_particle::TOWN_AURA,
                        xyz(
                            x + f64::from(callback_rng.next_f32()),
                            y + 1.1,
                            z + f64::from(callback_rng.next_f32()),
                        ),
                        zero,
                    );
                }
            }
            // BlockPortal.randomDisplayTick.
            90 => {
                if callback_rng.next_i32_bound(100) == 0 {
                    sound(
                        &mut effects,
                        "portal.portal",
                        xyz(x + 0.5, y + 0.5, z + 0.5),
                        0.5,
                        callback_rng.next_f32() * 0.4 + 0.8,
                    );
                }
                for _ in 0..4 {
                    let mut px = x + f64::from(callback_rng.next_f32());
                    let py = y + f64::from(callback_rng.next_f32());
                    let mut pz = z + f64::from(callback_rng.next_f32());
                    let mut vx = (f64::from(callback_rng.next_f32()) - 0.5) * 0.5;
                    let vy = (f64::from(callback_rng.next_f32()) - 0.5) * 0.5;
                    let mut vz = (f64::from(callback_rng.next_f32()) - 0.5) * 0.5;
                    let sign = callback_rng.next_i32_bound(2) * 2 - 1;
                    if self.ambient_block_at(pos.west()).props().block.0 != 90
                        && self.ambient_block_at(pos.east()).props().block.0 != 90
                    {
                        px = x + 0.5 + 0.25 * f64::from(sign);
                        vx = f64::from(callback_rng.next_f32()) * 2.0 * f64::from(sign);
                    } else {
                        pz = z + 0.5 + 0.25 * f64::from(sign);
                        vz = f64::from(callback_rng.next_f32()) * 2.0 * f64::from(sign);
                    }
                    particle(
                        &mut effects,
                        ambient_particle::PORTAL,
                        xyz(px, py, pz),
                        xyz(vx, vy, vz),
                    );
                }
            }
            // BlockRedstoneOre's private spawnParticles uses `worldIn.rand`, represented by
            // world_rng rather than callback_rng. It is only active for lit ore (id 74).
            74 => {
                for face in 0..6 {
                    let mut px = x + f64::from(world_rng.next_f32());
                    let mut py = y + f64::from(world_rng.next_f32());
                    let mut pz = z + f64::from(world_rng.next_f32());
                    if face == 0 && !self.ambient_block_at(pos.up()).props().opaque_cube {
                        py = y + 0.0625 + 1.0;
                    }
                    if face == 1 && !self.ambient_block_at(pos.down()).props().opaque_cube {
                        py = y - 0.0625;
                    }
                    if face == 2 && !self.ambient_block_at(pos.south()).props().opaque_cube {
                        pz = z + 0.0625 + 1.0;
                    }
                    if face == 3 && !self.ambient_block_at(pos.north()).props().opaque_cube {
                        pz = z - 0.0625;
                    }
                    if face == 4 && !self.ambient_block_at(pos.east()).props().opaque_cube {
                        px = x + 0.0625 + 1.0;
                    }
                    if face == 5 && !self.ambient_block_at(pos.west()).props().opaque_cube {
                        px = x - 0.0625;
                    }
                    if px < x || px > x + 1.0 || py < 0.0 || py > y + 1.0 || pz < z || pz > z + 1.0
                    {
                        particle(
                            &mut effects,
                            ambient_particle::REDSTONE,
                            xyz(px, py, pz),
                            zero,
                        );
                    }
                }
            }
            // BlockRedstoneRepeater.randomDisplayTick.
            94 => {
                let facing = Facing::from_horizontal(i32::from(state.meta() & 3));
                let mut px = x + 0.5 + (f64::from(callback_rng.next_f32()) - 0.5) * 0.2;
                let py = y + 0.4 + (f64::from(callback_rng.next_f32()) - 0.5) * 0.2;
                let mut pz = z + 0.5 + (f64::from(callback_rng.next_f32()) - 0.5) * 0.2;
                let mut speed = -5.0;
                if callback_rng.next_bool() {
                    speed = f32::from(((state.meta() >> 2) & 3) + 1) * 2.0 - 1.0;
                }
                speed /= 16.0;
                let (dx, _, dz) = facing.offset();
                px += f64::from(speed) * f64::from(dx);
                pz += f64::from(speed) * f64::from(dz);
                particle(
                    &mut effects,
                    ambient_particle::REDSTONE,
                    xyz(px, py, pz),
                    zero,
                );
            }
            // BlockRedstoneTorch.randomDisplayTick.
            76 => {
                let mut px = x + 0.5 + (callback_rng.next_f64() - 0.5) * 0.2;
                let mut py = y + 0.7 + (callback_rng.next_f64() - 0.5) * 0.2;
                let mut pz = z + 0.5 + (callback_rng.next_f64() - 0.5) * 0.2;
                let facing = torch_facing(state.meta());
                if facing.axis().is_horizontal() {
                    let opposite = facing.opposite();
                    let (dx, _, dz) = opposite.offset();
                    px += 0.27 * f64::from(dx);
                    py += 0.22;
                    pz += 0.27 * f64::from(dz);
                }
                particle(
                    &mut effects,
                    ambient_particle::REDSTONE,
                    xyz(px, py, pz),
                    zero,
                );
            }
            // BlockRedstoneWire.randomDisplayTick: speed carries the red dust RGB triple.
            55 if state.meta() & 15 != 0 => {
                let px = x + 0.5 + (f64::from(callback_rng.next_f32()) - 0.5) * 0.2;
                let py = y + 0.0625;
                let pz = z + 0.5 + (f64::from(callback_rng.next_f32()) - 0.5) * 0.2;
                let power = f32::from(state.meta() & 15) / 15.0;
                let red = power * 0.6 + 0.4;
                let green = (power * power * 0.7 - 0.5).max(0.0);
                let blue = (power * power * 0.6 - 0.7).max(0.0);
                particle(
                    &mut effects,
                    ambient_particle::REDSTONE,
                    xyz(px, py, pz),
                    xyz(f64::from(red), f64::from(green), f64::from(blue)),
                );
            }
            // BlockSponge.randomDisplayTick.
            19 if state.meta() & 1 != 0 => {
                let facing = Facing::from_index(callback_rng.next_i32_bound(6));
                if facing != Facing::Up && !self.ambient_solid_top(pos.relative(facing)) {
                    let mut px = x;
                    let mut py = y;
                    let mut pz = z;
                    if facing == Facing::Down {
                        py -= 0.05;
                        px += callback_rng.next_f64();
                        pz += callback_rng.next_f64();
                    } else {
                        py += callback_rng.next_f64() * 0.8;
                        if facing.axis() == mc_math::Axis::X {
                            pz += callback_rng.next_f64();
                            px += if facing == Facing::East { 1.0 } else { 0.05 };
                        } else {
                            px += callback_rng.next_f64();
                            pz += if facing == Facing::South { 1.0 } else { 0.05 };
                        }
                    }
                    particle(
                        &mut effects,
                        ambient_particle::DRIP_WATER,
                        xyz(px, py, pz),
                        zero,
                    );
                }
            }
            // BlockTorch.randomDisplayTick: no callback random draws at all.
            50 => {
                let facing = torch_facing(state.meta());
                let (px, py, pz) = if facing.axis().is_horizontal() {
                    let opposite = facing.opposite();
                    let (dx, _, dz) = opposite.offset();
                    (
                        x + 0.5 + 0.27 * f64::from(dx),
                        y + 0.7 + 0.22,
                        z + 0.5 + 0.27 * f64::from(dz),
                    )
                } else {
                    (x + 0.5, y + 0.7, z + 0.5)
                };
                particle(
                    &mut effects,
                    ambient_particle::SMOKE_NORMAL,
                    xyz(px, py, pz),
                    zero,
                );
                particle(&mut effects, ambient_particle::FLAME, xyz(px, py, pz), zero);
            }
            // BlockLiquid.randomDisplayTick: water and lava share the implementation.
            8..=11 => {
                let material = state.props().material;
                if material == Material::Water {
                    let level = state.meta();
                    if level > 0 && level < 8 {
                        if callback_rng.next_i32_bound(64) == 0 {
                            sound(
                                &mut effects,
                                "liquid.water",
                                xyz(x + 0.5, y + 0.5, z + 0.5),
                                callback_rng.next_f32() * 0.25 + 0.75,
                                callback_rng.next_f32() + 0.5,
                            );
                        }
                    } else if callback_rng.next_i32_bound(10) == 0 {
                        particle(
                            &mut effects,
                            ambient_particle::SUSPENDED,
                            xyz(
                                x + f64::from(callback_rng.next_f32()),
                                y + f64::from(callback_rng.next_f32()),
                                z + f64::from(callback_rng.next_f32()),
                            ),
                            zero,
                        );
                    }
                } else if material == Material::Lava {
                    let above = self.ambient_block_at(pos.up());
                    if above.props().material == Material::Air && !above.props().opaque_cube {
                        if callback_rng.next_i32_bound(100) == 0 {
                            let px = x + f64::from(callback_rng.next_f32());
                            let py = y + 1.0;
                            let pz = z + f64::from(callback_rng.next_f32());
                            particle(&mut effects, ambient_particle::LAVA, xyz(px, py, pz), zero);
                            sound(
                                &mut effects,
                                "liquid.lavapop",
                                xyz(px, py, pz),
                                0.2 + callback_rng.next_f32() * 0.2,
                                0.9 + callback_rng.next_f32() * 0.15,
                            );
                        }
                        if callback_rng.next_i32_bound(200) == 0 {
                            sound(
                                &mut effects,
                                "liquid.lava",
                                xyz(x, y, z),
                                0.2 + callback_rng.next_f32() * 0.2,
                                0.9 + callback_rng.next_f32() * 0.15,
                            );
                        }
                    }
                }
                if callback_rng.next_i32_bound(10) == 0 && self.ambient_solid_top(pos.down()) {
                    let below = self.ambient_block_at(pos.down().down());
                    let material = below.props().material;
                    if !material.blocks_movement() && !material.is_liquid() {
                        let particle_id = if state.props().material == Material::Water {
                            ambient_particle::DRIP_WATER
                        } else {
                            ambient_particle::DRIP_LAVA
                        };
                        particle(
                            &mut effects,
                            particle_id,
                            xyz(
                                x + f64::from(callback_rng.next_f32()),
                                y - 1.05,
                                z + f64::from(callback_rng.next_f32()),
                            ),
                            zero,
                        );
                    }
                }
            }
            _ => {}
        }
        effects
    }

    /// Returns the actual resident state for an ambient callback, erasing all out-of-range,
    /// illegal, and unloaded positions to air.
    #[inline]
    fn ambient_block_at(&self, pos: BlockPos) -> BlockState {
        self.loaded_block_at(pos).unwrap_or(BlockState::AIR)
    }

    /// `doesBlockHaveSolidTopSurface` through the loaded-only ambient view. Random display ticks
    /// must see an unloaded neighbour as air rather than consulting packet storage.
    #[inline]
    fn ambient_solid_top(&self, pos: BlockPos) -> bool {
        let state = self.ambient_block_at(pos);
        let props = state.props();
        if props.material.info().opaque && props.full_cube {
            return true;
        }
        match props.block.0 {
            id if family::stairs(BlockId(id)) => state.meta() & 4 != 0,
            id if family::slab(BlockId(id)) => state.meta() & 8 != 0,
            154 => true,             // hopper
            78 => state.meta() == 6, // eight-layer snow
            _ => false,
        }
    }

    /// Java `World.isRainingAt`, with the interpolated rain-state boolean supplied by the
    /// environment owner. The biome source and resident cache are both resolved through the
    /// installed world provider.
    #[must_use]
    pub fn is_raining_at(&self, pos: BlockPos, raining: bool) -> bool {
        if !raining || !self.can_see_sky(pos) {
            return false;
        }
        if self.precipitation_height(pos.x, pos.z) > pos.y {
            return false;
        }
        let biome = mc_data::biome(self.biome_at(pos));
        // `canSnowAt` rejects the cold half of the biome temperature range.  The client-side
        // weather gate is therefore not just the biome's precipitation flags: a cold, rainy
        // biome must become snow before Entity.isWet can count it as rain.
        !biome.snow
            && biome.temperature_at(pos.y, temperature_noise_at(pos.x, pos.z)) > 0.15
            && biome.rain
    }

    /// Scans an entity AABB for Java's three client-visible fire hazards.
    ///
    /// java: `World.isFlammableWithin` (`World.java:2041`). The upper bounds are exclusive after
    /// `floor(max + 1)`, and an out-of-height box is rejected before scanning. The cells are
    /// `World.getBlockState` (its `isAreaLoaded(.., true)` gate is always true on a client), so
    /// an absent column reads the shared blank chunk like every other block reader.
    #[must_use]
    pub fn is_flammable_within(&self, bb: &Aabb) -> bool {
        let min_x = floor_f64(bb.min.x);
        let max_x = floor_f64(bb.max.x + 1.0);
        let min_y = floor_f64(bb.min.y);
        let max_y = floor_f64(bb.max.y + 1.0);
        let min_z = floor_f64(bb.min.z);
        let max_z = floor_f64(bb.max.z + 1.0);
        if max_y < 0 || min_y >= 256 {
            return false;
        }
        for x in min_x..max_x {
            for y in min_y..max_y {
                for z in min_z..max_z {
                    let block = self.block_at(BlockPos::new(x, y, z)).props().block;
                    if matches!(block.0, 10 | 11 | 51) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// java: `BlockBush.canPlaceBlockOn(Block)` (`BlockBush.java:43`) and the five overrides of it.
///
/// `BlockCrops`, `BlockNetherWart` and `BlockStem` also override it; their blocks (59, 104, 105,
/// 115, 141, 142) have no `ItemBlock`, but they are answered anyway so the predicate is total
/// over the registry.
fn bush_can_place_on(bush: BlockId, ground: BlockId) -> bool {
    match bush.0 {
        // java: `BlockLilyPad:59` - identity against `Blocks.water`, the *still* block.
        111 => ground == block_id::WATER,
        // java: `BlockDeadBush:37`.
        32 => {
            ground == block_id::SAND
                || ground == block_id::HARDENED_CLAY
                || ground == block_id::STAINED_HARDENED_CLAY
                || ground == block_id::DIRT
        }
        // java: `BlockMushroom:67` - `ground.isFullBlock()`.
        39 | 40 => mc_data::block_props(BlockState::new(ground, 0)).full_block,
        // java: `BlockCrops:39` and `BlockStem:59` - farmland only.
        59 | 104 | 105 | 141 | 142 => ground == block_id::FARMLAND,
        // java: `BlockNetherWart:44` - soul sand only.
        115 => ground == block_id::SOUL_SAND,
        // java: `BlockBush:43`.
        _ => ground == block_id::GRASS || ground == block_id::DIRT || ground == block_id::FARMLAND,
    }
}

/// java: `BlockVine.canPlaceOn(Block)` (`BlockVine.java:175`).
fn vine_can_place_on(state: BlockState) -> bool {
    let props = state.props();
    props.full_cube && props.material.info().blocks_movement
}

/// java: `BlockTrapDoor.isValidSupportBlock(Block)` (`BlockTrapDoor.java:246`).
fn trapdoor_support(state: BlockState) -> bool {
    let props = state.props();
    if props.material.info().opaque && props.full_cube {
        return true;
    }
    let id = props.block;
    id == block_id::GLOWSTONE || family::slab(id) || family::stairs(id)
}

/// A block-local `(x, z)` pair, each `0..=15`, from a world position. Java's `& 15` on a
/// negative `i32` two's-complements to the same low-4-bits result Rust's `&` gives, so no
/// separate negative-coordinate case is needed.
#[inline]
fn local_xz(p: BlockPos) -> (u8, u8) {
    ((p.x & 15) as u8, (p.z & 15) as u8)
}

/// java: the horizontal half of `World.isValid` - `-30000000 <= x < 30000000`, same for z.
#[inline]
fn is_valid_xz(p: BlockPos) -> bool {
    (-30_000_000..30_000_000).contains(&p.x) && (-30_000_000..30_000_000).contains(&p.z)
}

/// The resident columns a `doVoidFogParticles` pass can sample, resolved once per pass.
///
/// Samples are `origin ± (RANDOM_DISPLAY_TICK_RADIUS - 1)` on x and z, which spans at most three
/// columns per axis. A position outside the window (impossible for a sample, but this is a
/// general reader) falls back to a map probe, so reads are exact for any position.
struct AmbientColumns<'w> {
    base: ChunkPos,
    columns: [[Option<&'w Chunk>; 3]; 3],
}

impl<'w> AmbientColumns<'w> {
    fn new(world: &'w World, origin: BlockPos) -> Self {
        let reach = RANDOM_DISPLAY_TICK_RADIUS - 1;
        let base = ChunkPos::new(
            origin.x.wrapping_sub(reach) >> 4,
            origin.z.wrapping_sub(reach) >> 4,
        );
        let mut columns = [[None; 3]; 3];
        for (dx, row) in (0i32..).zip(columns.iter_mut()) {
            for (dz, column) in (0i32..).zip(row.iter_mut()) {
                *column = world.chunk(ChunkPos::new(
                    base.x.wrapping_add(dx),
                    base.z.wrapping_add(dz),
                ));
            }
        }
        Self { base, columns }
    }

    /// Exactly `world.ambient_block_at(p)`.
    #[inline]
    fn ambient_block_at(&self, world: &World, p: BlockPos) -> BlockState {
        if !(0..256).contains(&p.y) {
            return BlockState::AIR;
        }
        let cpos = p.chunk_pos();
        let (dx, dz) = (
            cpos.x.wrapping_sub(self.base.x),
            cpos.z.wrapping_sub(self.base.z),
        );
        let chunk = if (0..3).contains(&dx) && (0..3).contains(&dz) {
            self.columns[dx as usize][dz as usize]
        } else {
            world.chunk(cpos)
        };
        chunk
            .and_then(|chunk| world.loaded_block_in(chunk, p))
            .unwrap_or(BlockState::AIR)
    }
}

/// Java's `ChunkProviderDebug` dimensions, including its float32 rounding points.
fn debug_world_grid_dimensions(state_count: usize) -> (i32, i32) {
    // Vanilla's registry always contains at least air. Keep a nonzero fallback here so a future
    // generated table cannot turn an unrelated `World::new` into a divide-by-zero panic.
    let count = state_count.max(1) as f32;
    let width = count.sqrt().ceil() as i32;
    let height = (count / width as f32).ceil() as i32;
    (width, height)
}

/// Decodes an `S22PacketMultiBlockChange` entry's crammed position short into a world
/// [`BlockPos`]. java: `S22PacketMultiBlockChange.BlockUpdateData.getPos`
#[inline]
#[must_use]
pub fn multi_block_change_pos(chunk_pos: ChunkPos, crammed: u16) -> BlockPos {
    let x = i32::from((crammed >> 12) & 0xF);
    let y = i32::from(crammed & 0xFF);
    let z = i32::from((crammed >> 8) & 0xF);
    BlockPos::new((chunk_pos.x << 4) + x, y, (chunk_pos.z << 4) + z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::{EntityKind, Position, Rotation, Velocity};
    use mc_data::{block_by_name, BlockId};
    use mc_math::DVec3;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn world_is_send_and_sync() {
        // The arena-index rule: no reference-counted interior-mutability wrapper anywhere in
        // mc-world. Neither of those two types (the pair the plan singles out by name in
        // lib.rs's module docs) is Send or Sync, so this compiling is a type-level proof the
        // invariant holds for every field reachable from `World`, not just a promise in a doc
        // comment.
        assert_send_sync::<World>();
    }

    #[test]
    fn trapdoor_activation_toggles_exactly_open_bit_and_requests_the_matching_local_effect() {
        let pos = BlockPos::new(-33, 71, 18);

        // Every legal `BlockTrapDoor` metadata value is exercised.  `OPEN` is only bit four;
        // each facing and top/bottom bit must survive the client prediction unchanged.
        for meta in 0_u8..16 {
            let mut world = World::new();
            world.load_empty_chunk(pos.chunk_pos());
            let before = BlockState::new(BlockId(96), meta);
            world.set_block(pos, before);

            let activation = world
                .activate_trapdoor(pos)
                .expect("wooden trapdoor must dispatch its virtual body");
            let after = BlockState::new(BlockId(96), meta ^ 4);
            assert_eq!(world.block_at(pos), after, "meta {meta:#x}");
            assert_eq!(
                activation,
                LocalBlockActivation {
                    consumed: true,
                    effect: Some(LocalBlockEffect {
                        id: if after.meta() & 4 != 0 { 1003 } else { 1006 },
                        pos,
                        data: 0,
                    }),
                },
                "meta {meta:#x}"
            );
        }
    }

    #[test]
    fn iron_trapdoor_activation_is_consumed_but_never_mutates_or_requests_a_sound() {
        let pos = BlockPos::new(8, 64, -4);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        let state = BlockState::new(BlockId(167), 0b1101);
        world.set_block(pos, state);

        assert_eq!(
            world.activate_trapdoor(pos),
            Some(LocalBlockActivation {
                consumed: true,
                effect: None,
            })
        );
        assert_eq!(
            world.block_at(pos),
            state,
            "iron body returns before cycling OPEN"
        );
    }

    #[test]
    fn wooden_door_activation_mutates_only_the_lower_half_and_keeps_clicked_effect_position() {
        let lower_pos = BlockPos::new(6, 70, -3);
        let upper_pos = lower_pos.up();

        for door in [64_u16, 193, 194, 195, 196, 197] {
            let mut lower_click = World::new();
            lower_click.load_empty_chunk(lower_pos.chunk_pos());
            let lower = BlockState::new(BlockId(door), 2);
            lower_click.set_block(lower_pos, lower);
            let activation = lower_click
                .activate_door(lower_pos)
                .expect("wooden lower half must dispatch");
            assert_eq!(
                lower_click.block_at(lower_pos),
                BlockState::new(BlockId(door), 6)
            );
            assert_eq!(lower_click.block_at(upper_pos), BlockState::AIR);
            assert_eq!(
                activation,
                LocalBlockActivation {
                    consumed: true,
                    effect: Some(LocalBlockEffect {
                        id: 1003,
                        pos: lower_pos,
                        data: 0,
                    }),
                }
            );

            let mut upper_click = World::new();
            upper_click.load_empty_chunk(lower_pos.chunk_pos());
            upper_click.set_block(lower_pos, lower);
            let upper = BlockState::new(BlockId(door), 0b1011);
            upper_click.set_block(upper_pos, upper);
            let activation = upper_click
                .activate_door(upper_pos)
                .expect("wooden upper half must dispatch");
            assert_eq!(
                upper_click.block_at(lower_pos),
                BlockState::new(BlockId(door), 6)
            );
            assert_eq!(
                upper_click.block_at(upper_pos),
                upper,
                "upper metadata is untouched"
            );
            assert_eq!(
                activation.effect,
                Some(LocalBlockEffect {
                    id: 1003,
                    pos: upper_pos,
                    data: 0,
                }),
                "the sound remains at the originally clicked upper position"
            );

            let close = upper_click
                .activate_door(upper_pos)
                .expect("the same upper half closes the lower door");
            assert_eq!(upper_click.block_at(lower_pos), lower);
            assert_eq!(upper_click.block_at(upper_pos), upper);
            assert_eq!(
                close.effect,
                Some(LocalBlockEffect {
                    id: 1006,
                    pos: upper_pos,
                    data: 0,
                })
            );
        }
    }

    #[test]
    fn iron_and_malformed_upper_door_activation_match_java_return_values() {
        let pos = BlockPos::new(2, 64, 9);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        for meta in 0..16 {
            // Activation receives chunk-storage states. Decode Java's full metadata domain
            // first so redundant upper bits 12..15 normalize to the representable 8..11.
            let iron = BlockState::from_java_state_id(71 | (meta << 12));
            world.set_block(pos, iron);
            assert_eq!(
                world.activate_door(pos),
                Some(LocalBlockActivation {
                    consumed: true,
                    effect: None,
                }),
                "iron door metadata {meta}"
            );
            assert_eq!(world.block_at(pos), iron, "iron door metadata {meta}");
        }

        let upper_pos = pos.up();
        let orphan = BlockState::new(BlockId(64), 8);
        world.set_block(pos, BlockState::AIR);
        world.set_block(upper_pos, orphan);
        assert_eq!(
            world.activate_door(upper_pos),
            Some(LocalBlockActivation {
                consumed: false,
                effect: None,
            })
        );
        assert_eq!(world.block_at(upper_pos), orphan);
        assert_eq!(world.activate_door(BlockPos::new(20, 64, 20)), None);
    }

    #[test]
    fn fence_gate_activation_reverses_only_from_behind_and_preserves_power() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(8, 70, 8);

        for gate in [107_u16, 183, 184, 185, 186, 187] {
            for (yaw, player_facing) in [
                (0.0_f32, 0_u8),
                (90.0, 1),
                (180.0, 2),
                (270.0, 3),
                (-90.0, 3),
                (-180.0, 2),
                (-270.0, 1),
                (-45.0, 0),
                (45.0, 1),
            ] {
                let opposite = player_facing ^ 2;
                world.set_block(pos, BlockState::new(BlockId(gate), opposite | 8));
                let opened = world
                    .activate_fence_gate(pos, yaw)
                    .expect("all wooden fence gates dispatch");
                assert_eq!(
                    world.block_at(pos),
                    BlockState::new(BlockId(gate), player_facing | 4 | 8),
                    "gate {gate}, yaw {yaw} must reverse an opposite facing before opening"
                );
                assert_eq!(
                    opened.effect,
                    Some(LocalBlockEffect {
                        id: 1003,
                        pos,
                        data: 0,
                    })
                );

                let closed = world
                    .activate_fence_gate(pos, yaw + 180.0)
                    .expect("an open gate closes regardless of player yaw");
                assert_eq!(
                    world.block_at(pos),
                    BlockState::new(BlockId(gate), player_facing | 8),
                    "closing preserves facing and POWERED"
                );
                assert_eq!(
                    closed.effect,
                    Some(LocalBlockEffect {
                        id: 1006,
                        pos,
                        data: 0,
                    })
                );

                let adjacent = player_facing ^ 1;
                world.set_block(pos, BlockState::new(BlockId(gate), adjacent | 8));
                world
                    .activate_fence_gate(pos, yaw)
                    .expect("non-opposite closed gate still opens");
                assert_eq!(
                    world.block_at(pos),
                    BlockState::new(BlockId(gate), adjacent | 4 | 8),
                    "a non-opposite facing is not rewritten"
                );
            }
        }

        let ordinary = BlockState::new(BlockId(1), 0);
        world.set_block(pos, ordinary);
        assert_eq!(world.activate_fence_gate(pos, 0.0), None);
        assert_eq!(world.block_at(pos), ordinary);
    }

    /// `BlockButton.onBlockActivated` has one client-visible mutation for both registrations:
    /// every legal orientation cycles only the `POWERED` bit, while an already-pressed button is
    /// still consumed without another write. Support is deliberately not revalidated here;
    /// Java's activation body never reads neighbours, even for a stale/unsupported button.
    #[test]
    fn button_activation_cycles_both_materials_and_server_reset_wins() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(8, 70, 8);

        for button in [BlockId(77), BlockId(143)] {
            for facing_meta in 0_u8..6 {
                let unpressed = BlockState::new(button, facing_meta);
                world.set_block(pos, unpressed);
                let activation = world.activate_button(pos).expect("button dispatch");
                assert_eq!(
                    activation,
                    LocalBlockActivation {
                        consumed: true,
                        effect: None,
                    },
                    "button {button:?}, facing meta {facing_meta}"
                );
                let pressed = BlockState::new(button, facing_meta | 8);
                assert_eq!(world.block_at(pos), pressed);

                // Pressing again is consumed but does not toggle an already-powered button.
                assert_eq!(world.activate_button(pos), Some(activation));
                assert_eq!(world.block_at(pos), pressed);

                // An incoming server block update is authoritative and clears the prediction.
                world.set_block(pos, unpressed);
                assert_eq!(world.block_at(pos), unpressed);
            }
        }

        // No support is present around this stale button, but Java's activation body still
        // toggles it; support validation belongs to placement/neighbor-drop logic.
        world.set_block(pos, BlockState::new(BlockId(77), 0));
        assert!(world.activate_button(pos).is_some());
        assert_eq!(world.block_at(pos).meta(), 8);

        let ordinary = BlockPos::new(9, 70, 8);
        let ordinary_state = BlockState::new(BlockId(1), 0);
        world.set_block(ordinary, ordinary_state);
        assert_eq!(world.activate_button(ordinary), None);
        assert_eq!(world.block_at(ordinary), ordinary_state);
    }

    #[test]
    fn trapdoor_activation_does_not_claim_unrelated_blocks() {
        let pos = BlockPos::new(0, 64, 0);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        let stone = BlockState::new(BlockId(1), 0);
        world.set_block(pos, stone);

        assert_eq!(world.activate_trapdoor(pos), None);
        assert_eq!(world.block_at(pos), stone);
    }

    /// `BlockSign.canPlaceBlockAt` is the replaceable-cell default plus
    /// `BlockContainer.hasInvalidNeighbor`: all four horizontal cactus neighbours reject both
    /// concrete sign blocks, while vertical/diagonal cactus and unrelated occupied cells do not
    /// accidentally enter that guard.
    #[test]
    fn sign_placement_uses_only_horizontal_cactus_neighbors_and_replaceable_target() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(8, 70, 8);
        let cactus = BlockState::new(BlockId(81), 0);
        let signs = [BlockId(63), BlockId(68)];

        for sign in signs {
            assert!(!world.sign_has_invalid_neighbor(pos));
            assert!(world.can_place_sign_at(pos));
            assert!(world.can_place_block_at(sign, pos));
            for side in Facing::HORIZONTALS {
                world.set_block(pos.relative(side), cactus);
                assert!(
                    world.sign_has_invalid_neighbor(pos),
                    "cactus on {side:?} must invalidate a sign"
                );
                assert!(!world.can_place_sign_at(pos));
                assert!(!world.can_place_block_at(sign, pos));
                world.set_block(pos.relative(side), BlockState::AIR);
            }

            // BlockContainer.hasInvalidNeighbor deliberately ignores the vertical and diagonal
            // cells. These controls also ensure the helper did not broaden the check to every
            // neighbour merely because cactus has a solid material.
            for ignored in [
                pos.up(),
                pos.down(),
                pos.relative(Facing::North).relative(Facing::East),
            ] {
                world.set_block(ignored, cactus);
                assert!(!world.sign_has_invalid_neighbor(pos));
                assert!(world.can_place_sign_at(pos));
                world.set_block(ignored, BlockState::AIR);
            }

            world.set_block(pos, BlockState::new(BlockId(1), 0));
            assert!(!world.sign_has_invalid_neighbor(pos));
            assert!(
                !world.can_place_sign_at(pos),
                "the inherited Block.canPlaceBlockAt check still rejects an occupied cell"
            );
            assert!(!world.can_place_block_at(sign, pos));
            world.set_block(pos, BlockState::AIR);
        }
    }

    /// The generic side predicate must reach the same `BlockSign` superclass/invalid-neighbour
    /// chain for every clicked face; this catches an accidental sign-only special case that
    /// bypasses `canPlaceBlockAt` when the future ItemSign caller supplies a side.
    #[test]
    fn sign_side_placement_delegates_to_the_sign_at_predicate() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(8, 70, 8);

        for sign in [BlockId(63), BlockId(68)] {
            for side in Facing::VALUES {
                assert_eq!(
                    world.can_place_block_on_side(sign, pos, side),
                    world.can_place_sign_at(pos),
                    "sign {sign:?} face {side:?}"
                );
            }
            world.set_block(pos.relative(Facing::West), BlockState::new(BlockId(81), 0));
            for side in Facing::VALUES {
                assert!(!world.can_place_block_on_side(sign, pos, side));
            }
            world.set_block(pos.relative(Facing::West), BlockState::AIR);
        }
    }

    /// The two provider-specific placement overrides that are easiest to accidentally flatten
    /// into the default predicate. Banners retain `BlockContainer.hasInvalidNeighbor` plus the
    /// replaceable target check; pressure plates ignore the target in their own override and
    /// accept only a solid top or a fence below (the outer `World.canBlockBePlaced` gate adds
    /// the target check when a real item is used).
    #[test]
    fn banner_and_pressure_plate_predicates_keep_their_java_overrides() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(8, 70, 8);
        let banners = [BlockId(176), BlockId(177)];
        let plates = [BlockId(70), BlockId(72), BlockId(147), BlockId(148)];

        for banner in banners {
            assert!(world.can_place_block_at(banner, pos));
            world.set_block(pos.relative(Facing::North), BlockState::new(BlockId(81), 0));
            assert!(!world.can_place_block_at(banner, pos));
            world.set_block(pos.relative(Facing::North), BlockState::AIR);
            world.set_block(pos, BlockState::new(BlockId(1), 0));
            assert!(!world.can_place_block_at(banner, pos));
            world.set_block(pos, BlockState::AIR);
        }

        for plate in plates {
            world.set_block(pos.down(), BlockState::new(BlockId(1), 0));
            assert!(
                world.can_place_block_at(plate, pos),
                "stone supports {plate:?}"
            );
            world.set_block(pos.down(), BlockState::new(BlockId(20), 0));
            assert!(
                !world.can_place_block_at(plate, pos),
                "glass rejects {plate:?}"
            );
            world.set_block(pos.down(), BlockState::new(BlockId(85), 0));
            assert!(
                world.can_place_block_at(plate, pos),
                "fence supports {plate:?}"
            );
            world.set_block(pos.down(), BlockState::AIR);
            assert!(
                !world.can_place_block_at(plate, pos),
                "air rejects {plate:?}"
            );

            // BlockBasePressurePlate does not call `super`: its direct predicate remains true
            // for an occupied target, while the caller's replaceable-material conjunct rejects
            // the actual ItemBlock placement.
            world.set_block(pos, BlockState::new(BlockId(1), 0));
            world.set_block(pos.down(), BlockState::new(BlockId(1), 0));
            assert!(world.can_place_block_at(plate, pos));
            world.set_block(pos, BlockState::AIR);
        }

        // The unoverridden Block predicate is the ordinary replaceable-cell default.
        assert!(world.can_place_block_at(BlockId(1), pos));
        world.set_block(pos, BlockState::new(BlockId(1), 0));
        assert!(!world.can_place_block_at(BlockId(1), pos));
    }

    /// On a client `BlockSign.onBlockActivated` returns `true` before touching the tile entity.
    /// The helper therefore consumes both sign variants even for a tile-less world write, does
    /// not request a local effect, and declines a non-sign block.
    #[test]
    fn sign_activation_is_consumed_without_tile_lookup_or_local_effect() {
        let mut world = World::new();
        let standing = BlockPos::new(8, 70, 8);
        let wall = BlockPos::new(9, 70, 9);
        world.load_empty_chunk(standing.chunk_pos());
        let standing_state = BlockState::new(BlockId(63), 7);
        let wall_state = BlockState::new(BlockId(68), 2);

        // These are resident sign blocks, so the normal block write creates their tile records;
        // Java's client activation arm still returns true before it reads either record.
        world.set_block(standing, standing_state);
        world.set_block(wall, wall_state);
        assert_eq!(world.tile_entity_count(), 2);
        for (pos, state) in [(standing, standing_state), (wall, wall_state)] {
            assert_eq!(
                world.activate_sign(pos),
                Some(LocalBlockActivation {
                    consumed: true,
                    effect: None,
                })
            );
            assert_eq!(world.block_at(pos), state, "activation is read-only");
        }

        world.set_block(BlockPos::new(10, 70, 10), BlockState::new(BlockId(1), 0));
        assert_eq!(world.activate_sign(BlockPos::new(10, 70, 10)), None);
    }

    #[test]
    fn block_click_dispatch_covers_client_overrides_and_negative_families() {
        let pos = BlockPos::new(12, 70, -5);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());

        // Base Block.onBlockClicked is a no-op. NoteBlock is also a no-op on a multiplayer
        // client because its override is guarded by !world.isRemote; stairs delegate to their
        // model block, whose inherited callback is empty.
        for state in [
            BlockState::new(BlockId(1), 0),  // ordinary stone
            BlockState::new(BlockId(25), 0), // note block
            BlockState::new(BlockId(53), 0), // oak stairs
        ] {
            world.set_block(pos, state);
            assert_eq!(world.on_block_clicked(pos), None, "state {state:?}");
        }

        // Both redstone variants run activate(), but only the unlit block predicts the lit state.
        for (id, expected) in [(73, true), (74, false)] {
            let state = BlockState::new(BlockId(id), 0);
            world.set_block(pos, state);
            assert_eq!(
                world.on_block_clicked(pos),
                Some(LocalBlockClick::RedstoneOre { pos })
            );
            assert_eq!(world.activate_redstone_ore_click(pos), expected);
            assert_eq!(world.block_at(pos).block().0, 74);
            if !expected {
                assert_eq!(world.block_at(pos), state, "lit ore must remain lit");
            }
        }

        // Cake retains the exact BITE metadata for the caller's canEat(false) and stale-state
        // check; consuming slices increments 0..5 and removes the cake at six bites.
        for bites in 0_u8..=6 {
            let state = BlockState::new(BlockId(92), bites);
            world.set_block(pos, state);
            assert_eq!(
                world.on_block_clicked(pos),
                Some(LocalBlockClick::Cake { pos, bites })
            );
            assert!(world.consume_cake_slice(pos));
            let expected = if bites < 6 {
                BlockState::new(BlockId(92), bites + 1)
            } else {
                BlockState::AIR
            };
            assert_eq!(world.block_at(pos), expected, "cake bites {bites}");
        }

        // The dragon egg callback is client-local particle feedback and must not move/mutate the
        // egg in this world helper.
        let egg = BlockState::new(BlockId(122), 0);
        world.set_block(pos, egg);
        assert_eq!(
            world.on_block_clicked(pos),
            Some(LocalBlockClick::DragonEgg { pos })
        );
        assert_eq!(world.block_at(pos), egg);
    }

    #[test]
    fn block_click_state_helpers_reject_stale_or_wrong_blocks() {
        let pos = BlockPos::new(-9, 63, 21);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        let lit = BlockState::new(BlockId(74), 0);
        world.set_block(pos, lit);
        assert!(!world.activate_redstone_ore_click(pos));
        assert_eq!(world.block_at(pos), lit);

        world.set_block(pos, BlockState::new(BlockId(1), 0));
        assert!(!world.consume_cake_slice(pos));
        assert_eq!(world.block_at(pos).block().0, 1);
    }

    #[test]
    fn biome_provider_uses_loaded_negative_column_cache_but_not_unloaded_or_invalid_y() {
        let mut world = World::new();
        let loaded = BlockPos::new(-17, 64, -33);
        world.load_empty_chunk(loaded.chunk_pos());

        assert_eq!(
            world.biome_at_with_generated(loaded, |got| {
                assert_eq!(got, loaded);
                Some(6) // swamp, proving this is not a hidden Plains fallback
            }),
            6
        );
        let chunk = world
            .chunk(loaded.chunk_pos())
            .expect("negative chunk loaded");
        assert_eq!(chunk.biomes()[15 * 16 + 15], 6);
        assert_eq!(
            world.biome_at_with_generated(BlockPos::new(-17, 200, -33), |_| -> Option<u8> {
                panic!("a loaded sentinel must cache its generated biome")
            }),
            6
        );

        let unloaded = BlockPos::new(-65, 64, 79);
        assert_eq!(
            world.biome_at_with_generated(unloaded, |got| {
                assert_eq!(got, unloaded);
                Some(2)
            }),
            2
        );
        assert!(!world.is_chunk_loaded(unloaded.chunk_pos()));

        // `World.isBlockLoaded` rejects this Y before looking up the resident chunk, so it
        // takes the provider directly and cannot overwrite the loaded column's cached byte.
        let outside_y = BlockPos::new(-17, 256, -33);
        assert_eq!(world.biome_at_with_generated(outside_y, |_| Some(4)), 4);
        assert_eq!(
            world
                .chunk(loaded.chunk_pos())
                .expect("still loaded")
                .biomes()[15 * 16 + 15],
            6
        );
    }

    /// `BlockBush.canPlaceBlockAt` first requires its target to be replaceable, then delegates
    /// to the shared `canBlockStay` substrate predicate. Flowers, saplings and tall grass do
    /// not override that predicate: grass, dirt and farmland are the complete allow-list.
    #[test]
    fn base_bushes_only_stay_on_grass_dirt_or_farmland() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(3, 70, 4);
        let below = pos.down();
        // 6 = sapling, 31 = tall grass, 37/38 = the two BlockFlower subclasses. These exercise
        // the distinct concrete classes which all inherit BlockBush's exact implementation.
        let base_bushes = [BlockId(6), BlockId(31), BlockId(37), BlockId(38)];

        for ground in [block_id::GRASS, block_id::DIRT, block_id::FARMLAND] {
            world.set_block(below, BlockState::new(ground, 0));
            for bush in base_bushes {
                assert!(
                    world.can_place_block_at(bush, pos),
                    "BlockBush {bush:?} must accept substrate {ground:?}"
                );
            }
        }

        // Biting negative controls: a solid cube and sand would both pass an accidental
        // "non-air"/"solid" implementation, while water catches an accidental lily-pad rule.
        for ground in [BlockId(1), block_id::SAND, block_id::WATER] {
            world.set_block(below, BlockState::new(ground, 0));
            for bush in base_bushes {
                assert!(
                    !world.can_place_block_at(bush, pos),
                    "BlockBush {bush:?} must reject substrate {ground:?}"
                );
            }
        }
    }

    /// `BlockFire.canPlaceBlockAt` accepts a horizontal neighbour precisely when
    /// `BlockFire.canCatchFire` reports nonzero encouragement.  This is intentionally not a
    /// material test: saplings and the two newer wooden stairs are the useful negative controls
    /// because they look combustible but `BlockFire.init` omits them.
    #[test]
    fn fire_placement_uses_blockfire_identity_map_not_material() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(3, 70, 4);
        let fire = BlockId(51);
        let neighbour = pos.relative(Facing::North);

        for state in [
            BlockState::new(BlockId(5), 5),   // a non-oak plank metadata
            BlockState::new(BlockId(17), 12), // a non-oak log metadata
            BlockState::new(BlockId(46), 0),  // TNT, no wood material involved
        ] {
            world.set_block(neighbour, state);
            assert!(
                world.can_place_block_at(fire, pos),
                "{state:?} is an exact BlockFire.init entry"
            );
        }

        for state in [
            BlockState::new(BlockId(1), 0),   // stone
            BlockState::new(BlockId(6), 0),   // sapling: plant material, omitted by init
            BlockState::new(BlockId(163), 0), // acacia stairs: wood material, omitted by init
            BlockState::new(BlockId(164), 0), // dark-oak stairs: wood material, omitted by init
        ] {
            world.set_block(neighbour, state);
            assert!(
                !world.can_place_block_at(fire, pos),
                "{state:?} is absent from BlockFire's identity maps"
            );
        }
    }

    /// `BlockBush.onNeighborBlockChange` and its random `updateTick` share
    /// `checkAndDropBlock`: an unsupported ordinary bush turns into air, while a valid one is
    /// left byte-for-byte alone.  Sand is deliberately a negative control because it is a
    /// plausible-looking plant substrate but is not in `BlockBush.canPlaceBlockOn`'s triplet.
    #[test]
    fn check_and_drop_bush_removes_an_unsupported_ordinary_bush_only() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(3, 70, 4);
        let flower = BlockState::new(BlockId(38), 7);

        world.set_block(pos.down(), BlockState::new(block_id::GRASS, 0));
        world.set_block(pos, flower);
        assert!(
            !world.check_and_drop_bush(pos),
            "valid ground must make both BlockBush trigger paths a no-op"
        );
        assert_eq!(world.block_at(pos), flower, "metadata survives the no-op");

        // This is the neighbour-change shape: support is replaced after the bush already
        // exists.  `set_block` itself is intentionally client-packet-pure; the caller invokes
        // this explicit BlockBush body when modelling the server-side notification.
        world.set_block(pos.down(), BlockState::new(block_id::SAND, 0));
        assert!(world.check_and_drop_bush(pos));
        assert_eq!(world.block_at(pos), BlockState::AIR);

        // The dispatch must not claim unrelated blocks just because their support is invalid.
        world.set_block(pos, BlockState::new(BlockId(1), 0));
        assert!(!world.check_and_drop_bush(pos));
        assert_eq!(world.block_at(pos), BlockState::new(BlockId(1), 0));
    }

    /// `BlockDoublePlant` overrides the base body: a lower-half failure removes its matching
    /// upper half first, and an orphaned upper half removes only itself.  This catches an easy
    /// but incorrect "treat 175 like every other bush" implementation.
    #[test]
    fn check_and_drop_bush_matches_double_plant_half_cleanup() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let lower = BlockPos::new(6, 70, 6);
        let upper = lower.up();
        let lower_state = BlockState::new(BlockId(175), 0);
        let upper_state = BlockState::new(BlockId(175), 8);

        world.set_block(lower.down(), BlockState::new(block_id::GRASS, 0));
        world.set_block(lower, lower_state);
        world.set_block(upper, upper_state);
        assert!(!world.check_and_drop_bush(lower));
        assert_eq!(world.block_at(lower), lower_state);
        assert_eq!(world.block_at(upper), upper_state);

        world.set_block(lower.down(), BlockState::new(block_id::SAND, 0));
        assert!(world.check_and_drop_bush(lower));
        assert_eq!(world.block_at(lower), BlockState::AIR);
        assert_eq!(world.block_at(upper), BlockState::AIR);

        // An upper half with no lower partner cannot erase a normal block below it.  Vanilla's
        // `block1 == this` guard is specifically what protects that malformed-world case.
        world.set_block(lower, BlockState::new(BlockId(1), 0));
        world.set_block(upper, upper_state);
        assert!(world.check_and_drop_bush(upper));
        assert_eq!(world.block_at(upper), BlockState::AIR);
        assert_eq!(world.block_at(lower), BlockState::new(BlockId(1), 0));
    }

    /// Mushroom survival is the other `canBlockStay` override: dim full blocks work, but the
    /// unconditional mycelium case must not accidentally be reduced to BlockBush's grass/dirt
    /// rule while sharing the `checkAndDropBlock` dispatch.
    #[test]
    fn check_and_drop_bush_uses_mushrooms_virtual_survival_predicate() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(9, 70, 9);
        let mushroom = BlockState::new(BlockId(39), 0);

        world.set_block(pos.down(), BlockState::new(block_id::MYCELIUM, 0));
        world.set_block(pos, mushroom);
        assert!(
            !world.check_and_drop_bush(pos),
            "mycelium bypasses the light/full-block branch"
        );

        world.set_block(pos.down(), BlockState::new(block_id::WATER, 0));
        assert!(world.check_and_drop_bush(pos));
        assert_eq!(world.block_at(pos), BlockState::AIR);
    }

    #[test]
    fn map_packets_create_and_update_the_worlds_named_map_record() {
        use mc_protocol::play::s2c::{MapPixelUpdate, Maps};

        let mut world = World::new();
        world
            .apply_map_packet(&Maps {
                map_id: -12,
                scale: 1,
                icons: vec![],
                update: Some(MapPixelUpdate {
                    width: 1,
                    height: 1,
                    min_x: 2,
                    min_y: 3,
                    data: vec![0x2A],
                }),
            })
            .unwrap();
        world
            .apply_map_packet(&Maps {
                map_id: -12,
                scale: 4,
                icons: vec![],
                update: None,
            })
            .unwrap();

        let map = world.map_data(-12).expect("S34 must load map_-12");
        assert_eq!(map.id, -12);
        assert_eq!(map.scale, 4);
        assert_eq!(map.colors[2 + 3 * crate::map::MAP_SIZE], 0x2A);
        assert_eq!(world.map_data_records().count(), 1);
    }

    #[test]
    fn client_empty_map_uses_memory_storage_id_and_java_centering() {
        let mut world = World::new();
        let map_id = world.create_empty_map_data(191.9, -64.1, -1);

        assert_eq!(map_id, 0, "SaveDataMemoryStorage always allocates map_0");
        let map = world.map_data(map_id).expect("local empty map record");
        assert_eq!(map.id, 0);
        assert_eq!(map.scale, 0);
        assert!(map.colors.iter().all(|color| *color == 0));
        assert_eq!(
            world.map_data_metadata(map_id),
            Some(&MapDataMetadata {
                x_center: 128,
                z_center: -128,
                dimension: -1,
            })
        );
    }

    #[test]
    fn map_packet_updates_keep_local_empty_map_metadata() {
        use mc_protocol::play::s2c::{MapPixelUpdate, Maps};

        let mut world = World::new();
        world.create_empty_map_data(0.0, 0.0, 1);
        world
            .apply_map_packet(&Maps {
                map_id: 0,
                scale: 2,
                icons: vec![],
                update: Some(MapPixelUpdate {
                    width: 1,
                    height: 1,
                    min_x: 0,
                    min_y: 0,
                    data: vec![0x2A],
                }),
            })
            .expect("valid S34 map update");

        assert_eq!(world.map_data(0).map(|map| map.scale), Some(2));
        assert_eq!(world.map_data(0).map(|map| map.colors[0]), Some(0x2A));
        assert_eq!(
            world.map_data_metadata(0),
            Some(&MapDataMetadata {
                x_center: 0,
                z_center: 0,
                dimension: 1,
            })
        );
    }

    /// `loaded_chunks` must agree with `loaded_chunk_count` and with `chunk`, and must follow
    /// loads and unloads. The alternative is a bounding-box probe
    /// that silently omits anything outside the box.
    #[test]
    fn loaded_chunks_visits_every_column_exactly_once() {
        let mut w = World::new();
        let payload = vec![0u8; crate::expected_chunk_data_len(0b1, true, true)];
        // Deliberately spread out, including negative coordinates and one column far outside any
        // plausible view-distance box.
        let placed = [
            ChunkPos::new(0, 0),
            ChunkPos::new(-3, 7),
            ChunkPos::new(2, -9),
            ChunkPos::new(1_000, -1_000),
        ];
        assert_eq!(w.loaded_chunks().count(), 0);
        for pos in placed {
            w.apply_chunk_data(pos, true, 0b1, &payload).unwrap();
        }

        let mut seen: Vec<ChunkPos> = w.loaded_chunks().map(|(pos, _)| pos).collect();
        seen.sort_by_key(|pos| (pos.x, pos.z));
        let mut expected = placed.to_vec();
        expected.sort_by_key(|pos| (pos.x, pos.z));
        assert_eq!(seen, expected);
        assert_eq!(w.loaded_chunks().count(), w.loaded_chunk_count());
        // The chunk handed back is the same one `chunk()` names.
        for (pos, chunk) in w.loaded_chunks() {
            assert!(std::ptr::eq(chunk, w.chunk(pos).expect("loaded")));
        }

        w.unload_chunk(ChunkPos::new(1_000, -1_000));
        assert_eq!(w.loaded_chunks().count(), 3);
        assert!(!w.loaded_chunks().any(|(pos, _)| pos.x == 1_000));
    }

    /// java: an absent column is `ChunkProviderClient.blankChunk`, one `EmptyChunk` shared by
    /// every absent column. It hard-codes `getBlock`/metadata/opacity/light/tile accessors but
    /// inherits `getBlockState`/`setBlockState`, so a written *state* reads back at the same
    /// local position in every absent column, while the hard-coded accessors do not move. (this test used to require that absent writes vanish, which vanilla never does.)
    #[test]
    fn unloaded_columns_share_the_blank_chunk_states_but_keep_hardcoded_accessors() {
        let mut w = World::new();
        let written = BlockPos::new(1_001, 65, -998); // local (9, 65, 10)
        let alias = BlockPos::new(-23, 65, 26); // another unloaded column, same local coordinate
        let stone = BlockState::new(BlockId(1), 0);

        w.set_block(written, stone);

        assert!(!w.is_chunk_loaded(written.chunk_pos()));
        assert!(!w.is_chunk_loaded(alias.chunk_pos()));
        assert_eq!(w.block_at(written), stone, "inherited Chunk.getBlockState");
        assert_eq!(
            w.block_at(alias),
            stone,
            "the same blank chunk at the same local cell"
        );
        assert_eq!(
            w.block_at(BlockPos::new(-23, 66, 26)),
            BlockState::AIR,
            "only that local cell"
        );
        assert_eq!(w.block_metadata_at(written), 0);
        assert_eq!(w.block_metadata_at(alias), 0);
        assert_eq!(w.block_light_opacity_at(written), 255);
        assert_eq!(w.block_light_opacity_at(alias), 255);
        assert_eq!(w.light_for(crate::SkyBlock::Sky, written), 15);
        assert_eq!(w.light_for(crate::SkyBlock::Block, alias), 0);
        assert_eq!(w.light_subtracted(written, 0), 0);
        w.set_light_for(crate::SkyBlock::Block, written, 13);
        assert_eq!(w.light_for(crate::SkyBlock::Block, written), 0);
        w.set_tile_entity(written, crate::TileEntity::new(crate::TileKind::Chest));
        assert!(w.tile_entity(written).is_none());
        assert!(w.tile_entity(alias).is_none());

        // Loading one real column does not inherit a write made while it was absent: it is a
        // fresh `Chunk`, and the blank chunk keeps serving the columns that are still absent.
        w.load_empty_chunk(alias.chunk_pos());
        assert_eq!(w.block_at(alias), BlockState::AIR);
        assert_eq!(w.block_light_opacity_at(alias), 0);
        w.set_block(alias, stone);
        assert_eq!(w.block_at(alias), stone);
        assert_eq!(w.block_metadata_at(alias), 0);
        assert_eq!(w.block_light_opacity_at(alias), 255);
        assert_eq!(w.block_at(written), stone);
        // A resident write never reaches the blank chunk.
        w.set_block(alias, BlockState::AIR);
        assert_eq!(w.block_at(written), stone);
        // `World.isValid` is checked before the provider: nothing beyond the world limit.
        let far = BlockPos::new(30_000_009, 65, 10); // local (9, 65, 10), invalid x
        assert_eq!(w.block_at(far), BlockState::AIR);
        w.set_block(far, BlockState::new(BlockId(20), 0));
        assert_eq!(w.block_at(written), stone, "an invalid write never lands");
    }

    /// The blank chunk's lifetime is the world's: an S21 unload (`doPreChunk(false)`) of a
    /// column, absent or resident, never clears it, writing air removes a blank block (with
    /// `Chunk.setBlockState`'s air-into-a-hole refusal), and a new `World` (a new `WorldClient`
    /// on JoinGame or a dimension change) starts with a fresh one. A partial S21 for an absent
    /// column fills the blank chunk's sections (inherited `fillChunk`).
    #[test]
    fn the_blank_chunk_survives_unloads_takes_partial_chunks_and_dies_with_the_world() {
        let mut w = World::new();
        let written = BlockPos::new(1_001, 65, -998); // local (9, 65, 10), absent
        let wool = BlockState::new(BlockId(35), 14);
        w.set_block(written, wool);

        w.apply_chunk_data(written.chunk_pos(), true, 0, &[])
            .unwrap();
        assert_eq!(
            w.block_at(written),
            wool,
            "an unload of an absent column changes nothing"
        );
        let resident = ChunkPos::new(4, 4);
        w.load_empty_chunk(resident);
        w.apply_chunk_data(resident, true, 0, &[]).unwrap();
        assert!(!w.is_chunk_loaded(resident));
        assert_eq!(
            w.block_at(written),
            wool,
            "nor does unloading a resident one"
        );
        assert_eq!(
            w.block_at(BlockPos::new(73, 65, 74)),
            wool,
            "and the unloaded column now reads the blank chunk too"
        );

        w.set_block(written, BlockState::AIR);
        assert_eq!(w.block_at(written), BlockState::AIR);
        // Air into a section the blank chunk never had is refused, not allocated.
        w.set_block(BlockPos::new(1, 200, 1), BlockState::AIR);
        assert!(w.blank_chunk.section(200 >> 4).is_none());

        // A partial (non-ground-up) S21 for an absent column patches the blank chunk.
        let payload = vec![0u8; crate::expected_chunk_data_len(0b1, true, false)];
        let mut stone_cell = payload.clone();
        // Section 0, local (0, 0, 0): little-endian `1 << 4`, stone.
        stone_cell[0] = 16;
        w.apply_chunk_data(ChunkPos::new(-50, 9), false, 0b1, &stone_cell)
            .unwrap();
        assert!(!w.is_chunk_loaded(ChunkPos::new(-50, 9)));
        let stone = BlockState::new(BlockId(1), 0);
        assert_eq!(w.block_at(BlockPos::new(-800, 0, 144)), stone);
        assert_eq!(w.block_at(BlockPos::new(160, 0, 160)), stone);

        let fresh = World::new();
        assert_eq!(fresh.block_at(BlockPos::new(160, 0, 160)), BlockState::AIR);
        assert_eq!(fresh.block_at(written), BlockState::AIR);
    }

    #[test]
    fn unloaded_empty_chunk_precipitation_ignores_absent_writes_and_resident_chunks_are_separate() {
        let mut w = World::new();
        let written = BlockPos::new(1_001, 65, -998); // local (9, 65, 10)
        let alias = BlockPos::new(-23, 65, 26); // another unloaded column, same local coordinate

        assert_eq!(w.precipitation_height(written.x, written.z), -1);
        w.set_block(written, BlockState::new(BlockId(1), 0));

        assert_eq!(w.precipitation_height(written.x, written.z), -1);
        assert_eq!(w.precipitation_height(alias.x, alias.z), -1);

        w.load_empty_chunk(alias.chunk_pos());
        assert_eq!(w.precipitation_height(alias.x, alias.z), -1);
        assert_eq!(w.precipitation_height(written.x, written.z), -1);
    }

    #[test]
    fn set_block_in_loaded_chunk_marks_dirty() {
        let mut w = World::new();
        // A full-column packet with a non-zero bitmask actually loads a chunk; bitmask 0 is
        // the wire's unload signal (see `unload_via_empty_bitmask_removes_the_chunk`), not an
        // empty chunk, so this must send at least one real section.
        let payload = vec![0u8; crate::expected_chunk_data_len(0b1, true, true)];
        w.apply_chunk_data(ChunkPos::new(0, 0), true, 0b1, &payload)
            .unwrap();
        w.drain_dirty_sections();
        let pos = BlockPos::new(1, 1, 1);
        w.set_block(pos, BlockState::new(BlockId(1), 0));
        assert_eq!(w.block_at(pos).block().0, 1);
        assert!(w.dirty_sections().any(|s| s == pos.section_pos()));
    }

    #[test]
    fn block_change_batch_matches_sequential_storage_light_and_dirty_state() {
        fn seeded_world() -> World {
            let mut world = World::new();
            // `checkLightFor` requires the complete 17-block neighbourhood to be resident.
            for cx in -1..=1 {
                for cz in -1..=1 {
                    world.load_empty_chunk(ChunkPos::new(cx, cz));
                }
            }
            world.drain_dirty_sections();
            world
        }

        let changes = vec![
            (BlockPos::new(0, 64, 0), BlockState::new(BlockId(1), 0)),
            (BlockPos::new(1, 64, 0), BlockState::new(BlockId(89), 0)),
            // Duplicate coordinates are legal at this API boundary; the final write must win.
            (BlockPos::new(1, 64, 0), BlockState::AIR),
            (BlockPos::new(0, 65, 0), BlockState::new(BlockId(20), 0)),
        ];

        let mut sequential = seeded_world();
        for &(position, state) in &changes {
            sequential.set_block(position, state);
        }

        let mut batched = seeded_world();
        batched.apply_block_changes(&changes);

        for x in -2..=2 {
            for y in 62..=67 {
                for z in -2..=2 {
                    let position = BlockPos::new(x, y, z);
                    assert_eq!(
                        batched.block_at(position),
                        sequential.block_at(position),
                        "block state differs at {position:?}"
                    );
                    assert_eq!(
                        batched.sky_light_at(position),
                        sequential.sky_light_at(position),
                        "sky light differs at {position:?}"
                    );
                    assert_eq!(
                        batched.block_light_at(position),
                        sequential.block_light_at(position),
                        "block light differs at {position:?}"
                    );
                }
            }
        }
        let batched_dirty = batched.drain_dirty_sections();
        let sequential_dirty = sequential.drain_dirty_sections();
        for &(position, state) in &changes {
            if state != BlockState::AIR || position == BlockPos::new(1, 64, 0) {
                assert!(
                    batched_dirty.contains(&position.section_pos()),
                    "batched write must dirty its changed section"
                );
            }
        }
        assert!(
            batched_dirty.len() <= sequential_dirty.len(),
            "coalescing should not dirty more sections than sequential writes"
        );
    }

    #[test]
    fn farmland_reacts_once_to_solid_material_above_and_ignores_non_solid_controls() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let farmland_pos = BlockPos::new(4, 64, 4);
        let above = farmland_pos.up();
        let farmland = BlockState::new(BlockId(60), 0);
        let dirt = BlockState::new(BlockId(3), 0);

        for solid in [BlockId(1), BlockId(20)] {
            world.set_block(farmland_pos, farmland);
            world.set_block(above, BlockState::new(solid, 0));
            world.drain_dirty_sections();

            assert!(
                world.react_farmland_to_solid_above(farmland_pos),
                "solid material for block {} must trigger the reaction",
                solid.0
            );
            assert_eq!(world.block_at(farmland_pos), dirt);
            assert!(
                world
                    .dirty_sections()
                    .any(|section| section == farmland_pos.section_pos()),
                "the dirt replacement must invalidate its mesh"
            );

            world.drain_dirty_sections();
            assert!(
                !world.react_farmland_to_solid_above(farmland_pos),
                "the dirt result makes a repeated callback inert"
            );
            assert_eq!(world.dirty_sections().count(), 0);
        }

        for non_solid in [BlockState::AIR, BlockState::new(BlockId(31), 1)] {
            world.set_block(farmland_pos, farmland);
            world.set_block(above, non_solid);
            assert!(!world.react_farmland_to_solid_above(farmland_pos));
            assert_eq!(world.block_at(farmland_pos), farmland);
        }

        world.set_block(farmland_pos, BlockState::new(BlockId(1), 0));
        world.set_block(above, BlockState::new(BlockId(20), 0));
        assert!(!world.react_farmland_to_solid_above(farmland_pos));
        assert_eq!(world.block_at(farmland_pos), BlockState::new(BlockId(1), 0));
    }

    #[test]
    fn set_block_out_of_world_height_is_a_no_op() {
        let mut w = World::new();
        let payload = vec![0u8; crate::expected_chunk_data_len(0b1, true, true)];
        w.apply_chunk_data(ChunkPos::new(0, 0), true, 0b1, &payload)
            .unwrap();
        w.set_block(BlockPos::new(0, 999, 0), BlockState::new(BlockId(1), 0));
        w.set_block(BlockPos::new(0, -1, 0), BlockState::new(BlockId(1), 0));
        assert_eq!(w.block_at(BlockPos::new(0, 999, 0)), BlockState::AIR);
    }

    /// Pre-[`World::load_empty_chunk`], the only way to make
    /// `set_block` stick in a *resident* hand-built world was to fake a chunk packet through
    /// `apply_chunk_data`. An unloaded write lands only in vanilla's shared blank-column scratch;
    /// this proves the helper creates real residency and leaves the column genuinely empty.
    #[test]
    fn load_empty_chunk_makes_set_block_stick_and_starts_all_air() {
        let mut w = World::new();
        let pos = ChunkPos::new(4, -2);
        assert!(!w.is_chunk_loaded(pos));

        w.load_empty_chunk(pos);
        assert!(w.is_chunk_loaded(pos));

        // Freshly loaded: no sections at all and the biome sentinel, exactly `Chunk::empty()`.
        let chunk = w.chunk(pos).expect("just loaded");
        assert!(chunk.sections().iter().all(Option::is_none));
        assert_eq!(chunk.biomes(), &[0xFFu8; 256]);

        // Every block in the column reads as air, not just the one this test is about to write.
        for y in [0i32, 1, 128, 255] {
            assert_eq!(
                w.block_at(BlockPos::new(pos.x * 16 + 3, y, pos.z * 16 + 5)),
                BlockState::AIR
            );
        }

        // The trap this method exists to route around: before loading the chunk, this write
        // would have been silently discarded (see `set_block_on_unloaded_chunk_is_a_no_op`).
        let target = BlockPos::new(pos.x * 16 + 3, 70, pos.z * 16 + 5);
        w.set_block(target, BlockState::new(BlockId(1), 0));
        assert_eq!(w.block_at(target).block().0, 1);
    }

    #[test]
    fn render_snapshot_shares_untouched_columns_and_isolates_written_ones() {
        let stone = BlockState::new(BlockId(1), 0);
        let written_column = ChunkPos::new(0, 0);
        let untouched_column = ChunkPos::new(1, 0);
        let written = BlockPos::new(1, 20, 1);
        let untouched = BlockPos::new(17, 20, 1);

        let mut live = World::new();
        live.load_empty_chunk(written_column);
        live.load_empty_chunk(untouched_column);
        live.set_block(written, stone);
        live.set_block(untouched, stone);

        let snapshot = live.render_snapshot();
        assert!(Arc::ptr_eq(
            live.chunks.get(&untouched_column).unwrap(),
            snapshot.chunks.get(&untouched_column).unwrap(),
        ));
        assert!(Arc::ptr_eq(
            live.chunks.get(&written_column).unwrap(),
            snapshot.chunks.get(&written_column).unwrap(),
        ));

        live.set_block(written, BlockState::AIR);
        assert_eq!(live.block_at(written), BlockState::AIR);
        assert_eq!(snapshot.block_at(written), stone);
        assert!(!Arc::ptr_eq(
            live.chunks.get(&written_column).unwrap(),
            snapshot.chunks.get(&written_column).unwrap(),
        ));
        assert!(Arc::ptr_eq(
            live.chunks.get(&untouched_column).unwrap(),
            snapshot.chunks.get(&untouched_column).unwrap(),
        ));

        // Light writes detach the same way as block writes.
        live.set_block(untouched, BlockState::AIR);
        assert_eq!(snapshot.block_at(untouched), stone);
    }

    #[test]
    fn render_snapshot_keeps_flower_pots_and_leaves_other_tile_entities_behind() {
        let column = ChunkPos::new(0, 0);
        let pot_pos = BlockPos::new(1, 70, 1);
        let chest_pos = BlockPos::new(3, 70, 3);
        let mut live = World::new();
        live.load_empty_chunk(column);
        live.set_block(chest_pos, BlockState::new(BlockId(54), 2));
        live.set_block(pot_pos, BlockState::new(BlockId(140), 0));
        live.set_tile_entity(
            pot_pos,
            crate::tileent::TileEntity {
                kind: crate::tileent::TileKind::FlowerPot,
                data: crate::tileent::TileData::FlowerPot(crate::tileent::FlowerPotState {
                    item_id: 38,
                    data: 0,
                }),
            },
        );
        assert!(live.tile_entity(chest_pos).is_some());
        assert!(live.tile_entity(pot_pos).is_some());

        let snapshot = live.render_snapshot();
        assert_eq!(
            snapshot
                .tile_entity(pot_pos)
                .and_then(|tile| tile.flower_pot()),
            live.tile_entity(pot_pos).and_then(|tile| tile.flower_pot()),
        );
        assert!(snapshot.tile_entity(chest_pos).is_none());
    }

    #[test]
    fn a_section_snapshot_shares_exactly_the_neighbourhood_columns_and_their_flower_pots() {
        let mut live = World::new();
        for x in -3..=3 {
            for z in -3..=3 {
                live.load_empty_chunk(ChunkPos::new(x, z));
            }
        }
        let pot = |item_id| crate::tileent::TileEntity {
            kind: crate::tileent::TileKind::FlowerPot,
            data: crate::tileent::TileData::FlowerPot(crate::tileent::FlowerPotState {
                item_id,
                data: 0,
            }),
        };
        let near_pot = BlockPos::new(-15, 70, 31); // column (-1, 1): inside the neighbourhood
        let far_pot = BlockPos::new(40, 70, 0); // column (2, 0): outside it
        for pos in [near_pot, far_pot] {
            live.set_block(pos, BlockState::new(BlockId(140), 0));
            live.set_tile_entity(pos, pot(38));
        }

        let snapshot = live.render_snapshot_for_sections(&[ChunkSectionPos::new(0, 4, 0)]);
        for x in -3..=3 {
            for z in -3..=3 {
                let column = ChunkPos::new(x, z);
                let inside = (-1..=1).contains(&x) && (-1..=1).contains(&z);
                match snapshot.chunks.get(&column) {
                    Some(shared) => {
                        assert!(inside, "{column:?} is outside the 3x3 neighbourhood");
                        assert!(Arc::ptr_eq(shared, live.chunks.get(&column).unwrap()));
                    }
                    None => assert!(!inside, "{column:?} is missing"),
                }
            }
        }
        assert!(snapshot.tile_entity(near_pot).is_some());
        assert!(snapshot.tile_entity(far_pot).is_none());
        // Everything a mesh reads that is not a column comes across unchanged.
        assert_eq!(snapshot.has_sky_light(), live.has_sky_light());
        assert_eq!(snapshot.is_debug_world(), live.is_debug_world());
        assert_eq!(
            snapshot.biome_at(BlockPos::new(-100, 64, 500)),
            live.biome_at(BlockPos::new(-100, 64, 500))
        );
        // An empty selection snapshots no columns at all.
        assert!(live.render_snapshot_for_sections(&[]).chunks.is_empty());
    }

    #[test]
    fn replacing_an_empty_chunk_drops_only_the_replaced_column_tile_entities() {
        let mut w = World::new();
        let replaced = ChunkPos::new(2, -3);
        let retained = ChunkPos::new(3, -3);
        let replaced_pos = BlockPos::new(replaced.x * 16 + 2, 70, replaced.z * 16 + 2);
        let retained_pos = BlockPos::new(retained.x * 16 + 2, 70, retained.z * 16 + 2);
        // Chest metadata 2 is the legal south-facing default; metadata 0 is an invalid packed
        // state in the 1.8.9 registry and would be exposed as air by `ChunkSection::block`.
        let chest = BlockState::new(BlockId(54), 2);

        w.load_empty_chunk(replaced);
        w.load_empty_chunk(retained);
        w.set_block(replaced_pos, chest);
        w.set_block(retained_pos, chest);
        assert_eq!(w.block_at(replaced_pos), chest);
        assert_eq!(w.block_at(retained_pos), chest);
        assert_eq!(w.tile_entity_count(), 2);
        assert!(w.tile_entity(replaced_pos).is_some());
        assert!(w.tile_entity(retained_pos).is_some());

        // A fresh full-column load has a new, empty Chunk tile-entity map in Java. The Rust
        // world keeps one flat map, so the replacement helper must perform that drop explicitly.
        w.load_empty_chunk(replaced);

        assert!(w.is_chunk_loaded(replaced));
        assert_eq!(w.block_at(replaced_pos), BlockState::AIR);
        assert!(w.tile_entity(replaced_pos).is_none());
        assert_eq!(w.tile_entity_count(), 1);
        assert_eq!(w.block_at(retained_pos), chest);
        assert!(w.tile_entity(retained_pos).is_some());
    }

    #[test]
    fn multi_block_change_position_decode() {
        // x=3, z=5, y=70 -> crammed = 3<<12 | 5<<8 | 70
        let crammed: u16 = (3 << 12) | (5 << 8) | 70;
        let pos = multi_block_change_pos(ChunkPos::new(2, -1), crammed);
        assert_eq!(pos, BlockPos::new(2 * 16 + 3, 70, -16 + 5));
    }

    #[test]
    fn apply_multi_block_change_writes_every_entry() {
        let mut w = World::new();
        w.apply_chunk_data(
            ChunkPos::new(0, 0),
            true,
            0xFFFF,
            &[0u8; 16 * (8192 + 2048 + 2048) + 256],
        )
        .unwrap();
        let stone = BlockState::new(BlockId(1), 0);
        let dirt = BlockState::new(block_by_name("dirt").unwrap(), 0);
        let c1: u16 = (1 << 12) | 5; // x=1, z=0, y=5
        let c2: u16 = (2 << 12) | 6; // x=2, z=0, y=6
        w.apply_multi_block_change(ChunkPos::new(0, 0), &[(c1, stone), (c2, dirt)]);
        assert_eq!(w.block_at(BlockPos::new(1, 5, 0)), stone);
        assert_eq!(w.block_at(BlockPos::new(2, 6, 0)), dirt);
    }

    #[test]
    fn unload_via_empty_bitmask_removes_the_chunk() {
        let mut w = World::new();
        let payload = vec![0u8; crate::expected_chunk_data_len(0b1, true, true)];
        w.apply_chunk_data(ChunkPos::new(0, 0), true, 0b1, &payload)
            .unwrap();
        assert!(w.is_chunk_loaded(ChunkPos::new(0, 0)));
        // Full column + empty bitmask is the wire's unload shape, not an empty chunk.
        w.apply_chunk_data(ChunkPos::new(0, 0), true, 0, &[])
            .unwrap();
        assert!(!w.is_chunk_loaded(ChunkPos::new(0, 0)));
    }

    #[test]
    fn bulk_zero_bitmask_loads_empty_but_single_zero_bitmask_unloads() {
        let mut w = World::new();
        let bulk_pos = ChunkPos::new(4, -7);
        let single_pos = ChunkPos::new(5, -7);
        let stone = BlockState::new(BlockId(1), 0);

        // Seed both columns with geometry so the test also proves that each path replaces the
        // previous state, not merely what it does from an initially absent world.
        w.load_empty_chunk(bulk_pos);
        w.load_empty_chunk(single_pos);
        w.set_block(bulk_pos.origin(20), stone);
        w.set_block(single_pos.origin(20), stone);

        let biomes = vec![7u8; 256];
        w.apply_chunk_bulk(
            true,
            &[BulkChunkEntry {
                chunk_x: bulk_pos.x,
                chunk_z: bulk_pos.z,
                primary_bitmask: 0,
                data: &biomes,
            }],
        )
        .unwrap();
        w.apply_chunk_data(single_pos, true, 0, &[]).unwrap();

        let bulk = w.chunk(bulk_pos).expect("S26 always loads every entry");
        assert!(bulk.sections().iter().all(Option::is_none));
        assert_eq!(bulk.biomes(), &[7u8; 256]);
        assert_eq!(w.block_at(bulk_pos.origin(20)), BlockState::AIR);
        assert!(
            !w.is_chunk_loaded(single_pos),
            "only the S21 full+zero shape is an unload"
        );
    }

    /// java: `handleChunkData` with `groundUp == false` calls `fillChunk` on whatever
    /// `getChunkFromChunkCoords` returns; for an absent column that is the provider's shared
    /// `EmptyChunk`, whose inherited `fillChunk` overwrites its sections. The states then read
    /// back (inherited `getBlockState`) in every absent column, while `EmptyChunk`'s hard-coded
    /// opacity and its `getBlock`-based precipitation scan do not change. (this
    /// test used to require the packet to vanish.)
    #[test]
    fn partial_chunk_for_an_absent_column_fills_the_shared_blank_chunk() {
        let mut w = World::new();
        let packet_pos = ChunkPos::new(8, -9);
        let alias_pos = ChunkPos::new(-2, 3);
        let stone = BlockState::new(BlockId(1), 0);
        let mut payload = vec![0u8; crate::expected_chunk_data_len(1, true, false)];
        payload[0] = stone.id as u8;
        payload[1] = (stone.id >> 8) as u8;

        w.apply_chunk_data(packet_pos, false, 1, &payload).unwrap();

        assert!(
            !w.is_chunk_loaded(packet_pos),
            "a partial chunk never loads a column"
        );
        assert_eq!(w.block_at(packet_pos.origin(0)), stone);
        assert_eq!(w.block_at(alias_pos.origin(0)), stone);
        assert_eq!(w.block_light_opacity_at(packet_pos.origin(0)), 255);
        assert_eq!(
            w.precipitation_height(packet_pos.origin(0).x, packet_pos.origin(0).z),
            -1
        );
        // A real load is a fresh chunk; the columns still absent keep reading the blank chunk.
        w.load_empty_chunk(packet_pos);
        assert_eq!(w.block_at(packet_pos.origin(0)), BlockState::AIR);
        assert_eq!(w.block_at(alias_pos.origin(0)), stone);
        // A short payload is still rejected before anything is written.
        let short = vec![0u8; 16];
        assert!(w.apply_chunk_data(alias_pos, false, 0b10, &short).is_err());
        assert_eq!(w.block_at(BlockPos::new(-32, 16, 48)), BlockState::AIR);
    }

    /// #15 / NET-1: unpacking a full column into a value and installing it must leave the world
    /// exactly as the in-place `apply_chunk_data` / `apply_chunk_bulk` paths do — every column
    /// field (via `Debug`), the dirty set, and the dropped tile entities — in both sky states.
    #[test]
    fn installing_an_unpacked_full_column_matches_applying_the_packet() {
        let pos = ChunkPos::new(-3, 5);
        for has_sky in [true, false] {
            // Sections 1 and 4: stone, a chest (tile provider), water and glass, plus light.
            let mask: u16 = 0b1_0010;
            let mut payload = Vec::new();
            for section in 0..2u16 {
                for cell in 0..4096u16 {
                    let raw: u16 = match (cell + section * 7) % 11 {
                        0 => 1 << 4,  // stone
                        1 => 54 << 4, // chest
                        2 => 9 << 4,  // water
                        3 => 20 << 4, // glass
                        _ => 0,
                    };
                    payload.extend_from_slice(&raw.to_le_bytes());
                }
            }
            payload.extend(std::iter::repeat_n(0x5Au8, 2 * 2048));
            if has_sky {
                payload.extend(std::iter::repeat_n(0xF3u8, 2 * 2048));
            }
            payload.extend((0..=255u8).map(|biome| biome % 23));

            let world = || {
                let mut world = World::new();
                world.set_has_sky_light(has_sky);
                world.load_empty_chunk(pos);
                world.set_block(
                    BlockPos::new(-48 + 3, 70, 80 + 3),
                    BlockState::new(BlockId(54), 2),
                );
                let _ = world.tile_entity_or_create(BlockPos::new(-48 + 3, 70, 80 + 3));
                let _ = world.drain_dirty_sections();
                world
            };

            let mut applied = world();
            applied
                .apply_chunk_data(pos, true, mask, &payload)
                .expect("well-formed payload");
            let mut bulk = world();
            bulk.apply_chunk_bulk(
                has_sky,
                &[BulkChunkEntry {
                    chunk_x: pos.x,
                    chunk_z: pos.z,
                    primary_bitmask: mask,
                    data: &payload,
                }],
            )
            .expect("well-formed payload");
            let mut installed = world();
            let column = Chunk::from_full_column_packet(mask, has_sky, &payload)
                .expect("well-formed payload");
            installed.install_full_column(pos, column);

            for other in [&mut bulk, &mut installed] {
                assert_eq!(
                    format!("{:?}", other.chunk(pos)),
                    format!("{:?}", applied.chunk(pos)),
                    "sky {has_sky}"
                );
                assert_eq!(other.tile_entity_count(), applied.tile_entity_count());
                assert_eq!(
                    other.drain_dirty_sections(),
                    applied.dirty_sections().collect::<BTreeSet<_>>()
                );
            }
            assert_eq!(
                applied.tile_entity_count(),
                0,
                "the column's tile entities are dropped"
            );
        }
        // A short payload fails the unpack exactly as `fill_from_packet` does.
        assert!(Chunk::from_full_column_packet(1, true, &[0; 16]).is_err());
    }

    /// Measurement aid for #15 / NET-1, not a gate: what one full-column unpack costs — the work
    /// that moves off the game thread — and what installing the ready column still costs on it.
    /// `cargo test -p mc-world --lib unpack_and_installing_cost_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe; run explicitly"]
    fn unpack_and_installing_cost_probe() {
        use std::time::Instant;
        // A void-map island column: sections 3 and 4, a stone floor in the lower one.
        let mut payload = Vec::new();
        for cell in 0..2 * 4096u32 {
            let raw: u16 = if cell % 4096 < 768 { 1 << 4 } else { 0 };
            payload.extend_from_slice(&raw.to_le_bytes());
        }
        payload.resize(payload.len() + 2 * 2 * 2048 + 256, 0);
        let air = vec![0u8; 256];
        let runs = 500u32;
        for (name, mask, data) in [("island", 0b1_1000u16, &payload), ("all-air", 0, &air)] {
            let start = Instant::now();
            for _ in 0..runs {
                let column = Chunk::from_full_column_packet(mask, true, data).unwrap();
                std::hint::black_box(&column);
            }
            let unpack = start.elapsed() / runs;
            let mut world = World::new();
            let columns: Vec<Chunk> = (0..runs)
                .map(|_| Chunk::from_full_column_packet(mask, true, data).unwrap())
                .collect();
            let start = Instant::now();
            for (i, column) in columns.into_iter().enumerate() {
                world.install_full_column(ChunkPos::new(i as i32 % 25, i as i32 / 25), column);
            }
            let install = start.elapsed() / runs;
            eprintln!(
                "unpack_and_installing_cost_probe {name}: unpack {unpack:?}, install {install:?}"
            );
        }
    }

    #[test]
    fn bulk_packet_sky_flag_shapes_payload_but_does_not_change_dimension() {
        let mut w = World::new();
        w.set_has_sky_light(false);
        let pos = ChunkPos::new(2, 3);

        // Packet says sky-light is present: blocks, block light, sky light, then biomes. The
        // distinct sentinels make parsing with the world's no-sky flag read the wrong biome byte.
        let mut payload = vec![0u8; 8192 + 2048];
        payload.extend(std::iter::repeat_n(0xEEu8, 2048));
        payload.extend(std::iter::repeat_n(9u8, 256));
        w.apply_chunk_bulk(
            true,
            &[BulkChunkEntry {
                chunk_x: pos.x,
                chunk_z: pos.z,
                primary_bitmask: 1,
                data: &payload,
            }],
        )
        .unwrap();

        assert!(!w.has_sky_light(), "join/respawn owns the dimension flag");
        assert_eq!(w.chunk(pos).unwrap().biomes(), &[9u8; 256]);
    }

    #[test]
    fn light_defaults_for_unloaded_space_follow_dimension_sky_flag() {
        let mut overworld = World::new();
        assert_eq!(overworld.light_at(BlockPos::new(0, 80, 0)), 15);
        overworld.set_has_sky_light(false);
        assert_eq!(overworld.light_at(BlockPos::new(0, 80, 0)), 0);
    }

    #[test]
    fn sky_light_at_distinguishes_missing_sections_empty_chunks_and_no_sky() {
        let mut world = World::new();
        let chunk = ChunkPos::new(0, 0);
        world.load_empty_chunk(chunk);
        world.set_block(BlockPos::new(1, 80, 1), BlockState::new(BlockId(1), 0));

        assert_eq!(
            world.sky_light_at(BlockPos::new(1, 32, 1)),
            0,
            "a missing section below an opaque height-map entry is not open sky"
        );
        assert_eq!(
            world.sky_light_at(BlockPos::new(1, 96, 1)),
            15,
            "a missing section above the height map is open sky"
        );
        assert_eq!(
            world.sky_light_at(BlockPos::new(32, 32, 32)),
            15,
            "an entirely unloaded chunk retains EmptyChunk's sky default"
        );

        // `Chunk.getLightFor` does not inspect the world provider in its null-storage arm, but
        // `sky_light_at` is the raw stored sky channel consumed by the client renderer.  A
        // no-sky dimension has no such channel at all, whether this is a real empty section or
        // the provider's shared EmptyChunk.
        world.set_has_sky_light(false);
        for p in [
            BlockPos::new(1, 32, 1),   // missing section below the height map
            BlockPos::new(1, 96, 1),   // missing section above the height map
            BlockPos::new(32, 32, 32), // absent column / EmptyChunk
        ] {
            assert_eq!(world.sky_light_at(p), 0, "no-sky provider at {p:?}");
        }

        // This accessor is a stored-nibble read, unlike `World.getLightFor`: positions outside
        // the 0..=255 storage range have no sky nibble to read.
        assert_eq!(world.sky_light_at(BlockPos::new(1, -1, 1)), 0);
        assert_eq!(world.sky_light_at(BlockPos::new(1, 256, 1)), 0);
    }

    #[test]
    fn spawn_and_query_round_trip() {
        let mut w = World::new();
        let e = w.spawn(EntityBundle {
            position: Position(DVec3::new(1.0, 2.0, 3.0)),
            velocity: Velocity::default(),
            rotation: Rotation {
                yaw: 90.0,
                pitch: 0.0,
            },
            kind: EntityKind::Type(mc_data::EntityType::Zombie),
            metadata: Default::default(),
        });
        let pos = w.ecs.get::<&Position>(e).unwrap();
        assert_eq!(pos.0, DVec3::new(1.0, 2.0, 3.0));
        drop(pos);
        assert!(w.despawn(e).is_ok());
        assert!(w.ecs.get::<&Position>(e).is_err());
        assert!(w.despawn(e).is_err());
    }

    /// java: S14 -> S01 §13 cross-section request, closed. Before `EntityKind::Player` existed,
    /// `EntityBundle::kind` could not name a player at all (no `EntityType` does), so a caller
    /// wanting "every drawable entity" needed a second, disjoint query for whatever marker
    /// component stood in for a player instead. A single `query::<&EntityKind>()` now returns
    /// both a mob and a player from one component set.
    #[test]
    fn a_remote_player_and_a_mob_both_resolve_through_one_entity_kind_query() {
        let mut w = World::new();
        let zombie = w.spawn(EntityBundle {
            position: Position(DVec3::new(1.0, 2.0, 3.0)),
            velocity: Velocity::default(),
            rotation: Rotation::default(),
            kind: EntityKind::Type(mc_data::EntityType::Zombie),
            metadata: Default::default(),
        });
        let player = w.spawn(EntityBundle {
            position: Position(DVec3::new(4.0, 5.0, 6.0)),
            velocity: Velocity::default(),
            rotation: Rotation::default(),
            kind: EntityKind::Player,
            metadata: Default::default(),
        });

        let found: Vec<(Entity, EntityKind)> = w
            .ecs
            .query::<(Entity, &EntityKind)>()
            .iter()
            .map(|(e, k)| (e, *k))
            .collect();
        assert_eq!(
            found.len(),
            2,
            "one EntityKind component per entity, no more"
        );
        assert!(found.contains(&(zombie, EntityKind::Type(mc_data::EntityType::Zombie))));
        assert!(found.contains(&(player, EntityKind::Player)));
    }

    #[test]
    fn dimension_provider_constants_win_over_unset_resident_biomes() {
        let mut world = World::new();
        let origin = BlockPos::new(3, 64, 5);

        world.set_dimension_id(-1);
        assert_eq!(world.biome_source(), BiomeSource::Nether);
        assert_eq!(world.biome_at(origin), 8);
        world.load_empty_chunk(origin.chunk_pos());
        // `Chunk::empty` carries the Java 0xFF biome sentinel; it must still consult the
        // selected provider instead of silently normalising to Plains.
        assert_eq!(world.biome_at(origin), 8);

        world.set_dimension_id(1);
        assert_eq!(world.biome_source(), BiomeSource::End);
        assert_eq!(world.biome_at(origin), 9);

        world.set_dimension_id(0);
        assert_eq!(world.biome_source(), BiomeSource::OverworldSeed0);
        // Java 1.8.9 `GenLayer.initializeAllBiomeGenerators(0L, DEFAULT, "")` resolves this
        // control cell to Forest (id 4), not the explicit Plains fallback used only when a
        // manager lookup returns no biome.
        assert_eq!(
            world.biome_at(origin),
            4,
            "the seed-0 control cell is Forest"
        );
    }

    #[test]
    fn rainfall_uses_the_installed_manager_and_matches_java_quantization() {
        let mut world = World::new();
        let origin = BlockPos::new(3, 64, 5);

        world.set_dimension_id(0);
        let generated_biome = world.biome_at(origin);
        assert_eq!(
            world.rainfall_at(origin),
            java_rainfall(generated_biome),
            "seed-0 absent columns use the provider's cached rainfall"
        );

        world.set_dimension_and_level_type(0, "flat");
        assert_eq!(world.rainfall_at(origin), 0.5);
        world.set_dimension_and_level_type(0, "debug_all_block_states");
        assert_eq!(world.rainfall_at(origin), 0.0);
        world.set_dimension_id(-1);
        assert_eq!(world.rainfall_at(origin), 0.0);
        world.set_dimension_id(1);
        assert_eq!(world.rainfall_at(origin), 0.0);
    }

    #[test]
    fn large_biomes_provider_uses_java_six_zoom_graph() {
        let mut world = World::new();
        let negative_control = BlockPos::new(-17, 64, -33);
        let far_control = BlockPos::new(4096, 64, -8193);

        world.set_dimension_and_level_type(0, "largeBiomes");
        assert_eq!(world.level_type(), "largeBiomes");
        assert_eq!(world.biome_source(), BiomeSource::OverworldSeed0);
        // Exact vanilla 1.8.9 `WorldChunkManager(0L, LARGE_BIOMES, "")` controls.
        assert_eq!(world.biome_at(negative_control), 4);
        assert_eq!(world.biome_at(far_control), 38);

        world.set_dimension_and_level_type(0, "amplified");
        // AMPLIFIED changes terrain height, not the biome GenLayer graph.
        assert_eq!(world.biome_at(negative_control), 1);
    }

    #[test]
    fn flat_and_debug_level_types_use_the_client_fixed_plains_manager() {
        let mut world = World::new();
        let absent = BlockPos::new(3, 64, 5);

        world.set_dimension_and_level_type(0, "flat");
        assert_eq!(world.biome_source(), BiomeSource::FlatPlains);
        assert_eq!(world.biome_at(absent), 1, "absent FLAT columns use Plains");
        world.load_empty_chunk(absent.chunk_pos());
        assert_eq!(
            world.biome_at(absent),
            1,
            "the 0xFF resident sentinel uses the same fixed manager"
        );

        world.set_dimension_and_level_type(0, "DEBUG_ALL_BLOCK_STATES");
        assert_eq!(world.biome_source(), BiomeSource::DebugPlains);
        assert_eq!(world.biome_at(absent), 1, "DEBUG columns use Plains");

        // Packet world-type parsing is case-insensitive and unknown names become DEFAULT.
        world.set_dimension_and_level_type(0, "not-a-world-type");
        assert_eq!(world.biome_source(), BiomeSource::OverworldSeed0);
        assert_eq!(
            world.biome_at(absent),
            4,
            "DEFAULT's seed-0 control cell is Forest"
        );

        // Dimension providers take precedence over the terrain type, just as WorldProvider's
        // Nether/End subclasses do. These are the normal-dimension negative controls.
        world.set_dimension_and_level_type(-1, "flat");
        assert_eq!(world.biome_source(), BiomeSource::Nether);
        assert_eq!(world.biome_at(absent), 8);
        world.set_dimension_and_level_type(1, "debug_all_block_states");
        assert_eq!(world.biome_source(), BiomeSource::End);
        assert_eq!(world.biome_at(absent), 9);
    }

    #[test]
    fn debug_world_block_lookup_matches_the_registry_grid_and_y_planes() {
        let mut world = World::new();
        assert_eq!(world.level_type(), "default");
        assert!(!world.is_debug_world());

        world.set_dimension_and_level_type(0, "DEBUG_ALL_BLOCK_STATES");
        assert_eq!(world.level_type(), "debug_all_block_states");
        assert!(world.is_debug_world());

        let barrier = BlockState::new(BlockId(166), 0);
        assert_eq!(world.block_at(BlockPos::new(-8, 60, 10)), barrier);
        assert_eq!(world.block_at(BlockPos::new(0, 60, 0)), barrier);

        // The first registry entry is air, followed by stone's seven valid metadata states. The
        // next row starts at index 89 (the Java ceil(sqrt(7806)) width) with lava metadata 5.
        assert_eq!(
            world.block_at(BlockPos::new(1, 70, 1)),
            BlockState::AIR,
            "registry index 0 is air"
        );
        assert_eq!(
            world.block_at(BlockPos::new(1, 70, 3)),
            BlockState::new(BlockId(1), 0),
            "registry index 1 is stone"
        );
        assert_eq!(
            world.block_at(BlockPos::new(1, 70, 15)),
            BlockState::new(BlockId(1), 6),
            "registry index 7 is stone metadata 6"
        );
        assert_eq!(
            world.block_at(BlockPos::new(3, 70, 1)),
            BlockState::new(BlockId(11), 5),
            "registry index 89 starts the second grid row"
        );
        assert_eq!(
            world.block_at(BlockPos::new(1, 70, 75)),
            BlockState::new(BlockId(8), 1),
            "the first row still exposes registry index 37"
        );

        // The packed representative intentionally collides for these DEBUG_WORLD entries, while
        // the sidecar retains the model tuple used by the renderer.
        let stair = world
            .debug_world_actual_state_at(BlockPos::new(77, 70, 59))
            .expect("registry index 3411 is a mapped stair tuple");
        assert_eq!(stair.stored(), BlockState::new(BlockId(53), 7));
        assert_eq!(stair.get("shape"), Some("inner_left"));
        let fence = world
            .debug_world_actual_state_at(BlockPos::new(113, 70, 169))
            .expect("registry index 5068 is a mapped fence tuple");
        assert_eq!(fence.stored(), BlockState::new(BlockId(85), 0));
        assert_eq!(fence.get("north"), Some("true"));

        // The final valid state is index 7805 at row 87/column 62. The following cells are
        // within Java's inclusive coordinate bounds but fail the list-size check and resolve to
        // air, as do non-positive/even coordinates.
        assert_eq!(
            world.block_at(BlockPos::new(175, 70, 125)),
            BlockState::new(BlockId(197), 0)
        );
        for pos in [
            BlockPos::new(175, 70, 127),
            BlockPos::new(177, 70, 1),
            BlockPos::new(1, 70, 179),
            BlockPos::new(0, 70, 1),
            BlockPos::new(2, 70, 1),
            BlockPos::new(-1, 70, 1),
            BlockPos::new(1, 70, -1),
        ] {
            assert_eq!(world.block_at(pos), BlockState::AIR, "{pos:?}");
        }

        for y in [59, 61, 71, -1, 256] {
            assert_eq!(
                world.block_at(BlockPos::new(1, y, 1)),
                BlockState::AIR,
                "DEBUG_WORLD non-plane y={y}"
            );
        }
    }

    #[test]
    fn debug_world_lookup_precedes_resident_storage_and_ordinary_worlds_do_not_change() {
        let mut world = World::new();
        let chunk = ChunkPos::new(0, 0);
        let stored = BlockState::new(BlockId(1), 0);
        let grid = BlockPos::new(1, 70, 1);
        let barrier_plane = BlockPos::new(1, 60, 1);
        let other_y = BlockPos::new(1, 61, 1);
        world.load_empty_chunk(chunk);
        world.set_block(grid, stored);
        world.set_block(barrier_plane, stored);
        world.set_block(other_y, stored);

        assert_eq!(world.block_at(grid), stored);
        assert_eq!(world.block_at(barrier_plane), stored);
        assert_eq!(world.block_at(other_y), stored);

        world.set_dimension_and_level_type(0, "debug_all_block_states");
        assert_eq!(world.block_at(grid), BlockState::AIR);
        assert_eq!(world.block_at(barrier_plane), DEBUG_WORLD_BARRIER);
        assert_eq!(world.block_at(other_y), BlockState::AIR);
        assert_eq!(world.loaded_block_at(grid), Some(BlockState::AIR));
        assert_eq!(
            world.loaded_block_at(barrier_plane),
            Some(DEBUG_WORLD_BARRIER)
        );
        assert_eq!(world.loaded_block_at(BlockPos::new(32, 70, 1)), None);

        // Switching back leaves the packet/resident storage untouched, matching the ordinary
        // client path after a later world-type lifecycle transition.
        world.set_dimension_and_level_type(0, "default");
        assert!(!world.is_debug_world());
        assert_eq!(world.block_at(grid), stored);
        assert_eq!(world.block_at(barrier_plane), stored);
        assert_eq!(world.block_at(other_y), stored);
    }

    #[test]
    fn debug_world_tuple_table_is_shared_by_render_snapshots() {
        let world = World::new();
        let snapshot = world.render_snapshot();
        assert_eq!(world.debug_world_states.len(), 7806);
        assert!(Arc::ptr_eq(
            &world.debug_world_states,
            &snapshot.debug_world_states
        ));
    }

    #[test]
    fn level_type_normalization_matches_join_and_respawn_packet_defaults() {
        assert_eq!(canonical_level_type_name("DEFAULT"), "default");
        assert_eq!(canonical_level_type_name("FLAT"), "flat");
        assert_eq!(
            canonical_level_type_name("debug_all_block_states"),
            "debug_all_block_states"
        );
        assert_eq!(canonical_level_type_name("server-custom"), "default");
    }

    #[test]
    fn resident_biome_bytes_remain_authoritative_after_provider_changes() {
        let mut world = World::new();
        let chunk_pos = ChunkPos::new(0, 0);
        let payload_len = crate::expected_chunk_data_len(1, true, true);
        let mut payload = vec![0_u8; payload_len];
        // One all-air section: block, block-light, sky-light, then the 256-byte biome array.
        payload[payload_len - 256..].fill(2); // desert
        world
            .apply_chunk_data(chunk_pos, true, 1, &payload)
            .expect("the one-section fixture is well formed");

        world.set_dimension_and_level_type(0, "flat");
        assert_eq!(
            world.biome_at(BlockPos::new(2, 64, 2)),
            2,
            "a real chunk biome is authoritative in FLAT too"
        );

        world.set_dimension_id(-1);
        assert_eq!(
            world.biome_at(BlockPos::new(2, 64, 2)),
            2,
            "a real chunk biome is authoritative even in a constant dimension"
        );
    }

    /// java: `doVoidFogParticles(floor_double(posX), ...)` sums with wrapping
    /// `int` arithmetic, and an S08 can place the player where `floor_double` saturates to
    /// `Integer.MAX_VALUE`/`MIN_VALUE`. The samples must wrap (the old `+` panicked in debug
    /// builds) and consume exactly the draws an ordinary origin does.
    #[test]
    fn void_fog_samples_wrap_at_saturated_origins_like_java_int() {
        let world = World::new();
        for origin in [
            BlockPos::new(i32::MAX, 64, i32::MAX),
            BlockPos::new(i32::MIN, i32::MAX, i32::MIN),
        ] {
            let mut sample = JavaRandom::new(7);
            let mut callback = JavaRandom::new(9);
            let effects = world.do_void_fog_particles(origin, &mut sample, &mut callback, true);
            assert!(effects.is_empty(), "nothing is loaded around {origin:?}");
            let mut ordinary = JavaRandom::new(7);
            let mut ordinary_callback = JavaRandom::new(9);
            let _ = world.do_void_fog_particles(
                BlockPos::new(0, 64, 0),
                &mut ordinary,
                &mut ordinary_callback,
                true,
            );
            assert_eq!(sample.next_i64(), ordinary.next_i64(), "same draw count");
        }
    }

    /// PART-6/BOVL-5 parity: the column-window sampler returns the same effects in the same
    /// order and leaves both RNG streams where the pristine per-sample `ambient_block_at` plus
    /// re-reading `random_display_tick_at_with_rain` loop left them. Random (including
    /// illegal) states fill a cube that straddles chunk borders around the origins, with one
    /// column unloaded; lit redstone ore exercises the world-RNG reach-back.
    #[test]
    fn void_fog_column_window_matches_the_per_sample_lookup_loop() {
        fn reference(
            world: &World,
            origin: BlockPos,
            sample_rng: &mut JavaRandom,
            callback_rng: &mut JavaRandom,
            reveal_barriers: bool,
            raining: bool,
        ) -> Vec<AmbientEffect> {
            let mut effects = Vec::new();
            for _ in 0..RANDOM_DISPLAY_TICK_SAMPLES {
                let pos = BlockPos::new(
                    origin.x + sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS)
                        - sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS),
                    origin.y + sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS)
                        - sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS),
                    origin.z + sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS)
                        - sample_rng.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS),
                );
                let state = world.ambient_block_at(pos);
                if state == BlockState::AIR {
                    continue;
                }
                effects.extend(world.random_display_tick_at_with_rain(
                    pos,
                    state,
                    callback_rng,
                    sample_rng,
                    raining,
                ));
                if reveal_barriers && state.props().block.0 == 166 {
                    effects.push(AmbientEffect::Particle(AmbientParticleEffect {
                        particle_id: ambient_particle::BARRIER,
                        position: mc_math::DVec3::new(
                            f64::from(pos.x) + 0.5,
                            f64::from(pos.y) + 0.5,
                            f64::from(pos.z) + 0.5,
                        ),
                        velocity: mc_math::DVec3::ZERO,
                    }));
                }
            }
            effects
        }

        let mut world = World::new();
        let unloaded = ChunkPos::new(1, -1);
        for cx in -2..=2 {
            for cz in -2..=2 {
                if ChunkPos::new(cx, cz) != unloaded {
                    world.load_empty_chunk(ChunkPos::new(cx, cz));
                }
            }
        }
        // Mostly the blocks with display-tick bodies, plus any id and meta at all.
        let ambient = [
            50, 51, 10, 11, 74, 76, 90, 18, 116, 47, 62, 117, 119, 122, 110, 8, 166,
        ];
        let mut fill = JavaRandom::new(0xd15_71c4);
        for _ in 0..12_000 {
            let p = BlockPos::new(
                fill.next_i32_bound(40) - 18,
                48 + fill.next_i32_bound(34),
                fill.next_i32_bound(40) - 22,
            );
            let id = if fill.next_bool() {
                ambient[fill.next_i32_bound(ambient.len() as i32) as usize]
            } else {
                1 + fill.next_i32_bound(197) as u16
            };
            let state = BlockState::new(BlockId(id), fill.next_i32_bound(16) as u8);
            if let Some(chunk) = world.chunks.get_mut(&p.chunk_pos()).map(Arc::make_mut) {
                let (x, z) = local_xz(p);
                chunk.set_block_at(x, p.y as u8, z, state);
            }
        }

        let mut sample = JavaRandom::new(0x5a_3b1e);
        let mut callback = JavaRandom::new(0xca11_bac4);
        let (mut want_sample, mut want_callback) = (sample.clone(), callback.clone());
        let mut produced = 0;
        for pass in 0..40 {
            let origin = BlockPos::new(
                [-1, 0, 7, 15, 16][pass % 5],
                64 + (pass as i32 % 3) * 5,
                [-17, -16, -9, 0][pass % 4],
            );
            let (reveal, raining) = (pass % 2 == 0, pass % 3 == 0);
            let got = world.do_void_fog_particles_with_rain(
                origin,
                &mut sample,
                &mut callback,
                reveal,
                raining,
            );
            let want = reference(
                &world,
                origin,
                &mut want_sample,
                &mut want_callback,
                reveal,
                raining,
            );
            assert_eq!(got, want, "pass {pass}");
            assert_eq!(sample.raw_seed(), want_sample.raw_seed(), "pass {pass}");
            assert_eq!(callback.raw_seed(), want_callback.raw_seed(), "pass {pass}");
            produced += got.len();
        }
        assert!(produced > 0, "the fixture exercises callback bodies");
    }

    /// Micro-bench (PART-6/BOVL-5): `cargo test -p mc-world --lib -- --ignored bench_void_fog
    /// --nocapture`. A void-map shape: 3×3 resident columns of air around the player.
    #[test]
    #[ignore = "timing bench; run explicitly"]
    fn bench_void_fog_window_vs_per_sample_lookups() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut world = World::new();
        for cx in -2..=2 {
            for cz in -2..=2 {
                world.load_empty_chunk(ChunkPos::new(cx, cz));
            }
        }
        let origin = BlockPos::new(5, 70, -3);
        let passes = 2_000;
        let (mut sample, mut callback) = (JavaRandom::new(1), JavaRandom::new(2));
        let start = Instant::now();
        for _ in 0..passes {
            // The pristine loop's lookups: `ambient_block_at` was `is_chunk_loaded` + `get`.
            for _ in 0..RANDOM_DISPLAY_TICK_SAMPLES {
                let pos = BlockPos::new(
                    origin.x + sample.next_i32_bound(16) - sample.next_i32_bound(16),
                    origin.y + sample.next_i32_bound(16) - sample.next_i32_bound(16),
                    origin.z + sample.next_i32_bound(16) - sample.next_i32_bound(16),
                );
                black_box(world.is_chunk_loaded(pos.chunk_pos()));
                black_box(world.loaded_block_at(pos));
            }
        }
        let old = start.elapsed();
        let start = Instant::now();
        for _ in 0..passes {
            black_box(world.do_void_fog_particles_with_rain(
                origin,
                &mut sample,
                &mut callback,
                false,
                false,
            ));
        }
        let new = start.elapsed();
        println!(
            "bench_void_fog old={:.1}us/tick new={:.1}us/tick",
            old.as_secs_f64() * 1e6 / f64::from(passes),
            new.as_secs_f64() * 1e6 / f64::from(passes),
        );
    }

    #[test]
    fn random_display_scheduler_consumes_only_sampling_rng_for_omitted_blocks() {
        let world = World::new();
        let origin = BlockPos::new(0, 0, 0);
        let mut sample = JavaRandom::new(0x1234_5678);
        let mut expected = sample.clone();
        let mut callback = JavaRandom::new(0xCAFE_BABE);
        let callback_before = callback.raw_seed();

        let effects = world.do_void_fog_particles(origin, &mut sample, &mut callback, false);
        assert!(effects.is_empty());
        for _ in 0..RANDOM_DISPLAY_TICK_SAMPLES {
            for _ in 0..3 {
                let _ = expected.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS);
                let _ = expected.next_i32_bound(RANDOM_DISPLAY_TICK_RADIUS);
            }
        }
        assert_eq!(
            sample.raw_seed(),
            expected.raw_seed(),
            "1000 samples still consume all six Java nextInt draws"
        );
        assert_eq!(callback.raw_seed(), callback_before);
    }

    #[test]
    fn ambient_callbacks_use_effective_resident_state_and_java_rng_streams() {
        let mut world = World::new();
        let pos = BlockPos::new(2, 64, 2);
        world.load_empty_chunk(pos.chunk_pos());
        let torch = BlockState::new(BlockId(50), 5);
        world.set_block(pos, torch);

        let mut callback = JavaRandom::new(7);
        let mut world_rng = JavaRandom::new(11);
        let callback_before = callback.raw_seed();
        let effects = world.random_display_tick_at(pos, torch, &mut callback, &mut world_rng);
        let ids: Vec<_> = effects
            .iter()
            .filter_map(|effect| match effect {
                AmbientEffect::Particle(particle) => Some(particle.particle_id),
                AmbientEffect::Sound(_) => None,
            })
            .collect();
        assert_eq!(
            ids,
            vec![ambient_particle::SMOKE_NORMAL, ambient_particle::FLAME]
        );
        assert_eq!(
            callback.raw_seed(),
            callback_before,
            "torch callback has no RNG draws"
        );

        let stale = BlockState::new(BlockId(50), 0);
        assert!(world
            .random_display_tick_at(pos, stale, &mut callback, &mut world_rng)
            .is_empty());
    }

    /// `torch_facing` transcribes `BlockTorch.getStateFromMeta`. The extractor's state table,
    /// generated from the real 1.8.9 classes, must agree for every legal torch and redstone torch
    /// metadata value.
    #[test]
    fn torch_facing_matches_the_extracted_java_facing_property() {
        let mut checked = 0;
        for id in [50, 75, 76] {
            for meta in 0..16u8 {
                let state = BlockState::new(BlockId(id), meta);
                if !state.is_legal() {
                    continue;
                }
                let facing = mc_data::block::state_properties(state)
                    .iter()
                    .find(|(name, _)| *name == "facing")
                    .map(|(_, value)| *value)
                    .expect("every legal torch state has a facing");
                assert_eq!(torch_facing(meta).name(), facing, "block {id} meta {meta}");
                checked += 1;
            }
        }
        assert_eq!(
            checked, 15,
            "five legal metas for each of the three torch blocks"
        );
        // Mutation control: the reading this replaced made a standing torch face east.
        assert_eq!(Facing::from_index(5), Facing::East);
        assert_eq!(torch_facing(5), Facing::Up);
    }

    /// java: `BlockTorch.randomDisplayTick` puts the smoke and flame at `(x+.5, y+.7, z+.5)` on a
    /// standing torch, and 0.27 toward the wall (the opposite of `FACING`) and 0.22 higher on a
    /// wall torch: on the flame of the model in both cases. `BlockRedstoneTorch` jitters the same
    /// anchor by three callback draws first.
    #[test]
    fn torch_particles_spawn_on_the_flame_for_every_facing() {
        fn particle_positions(effects: &[AmbientEffect]) -> Vec<DVec3> {
            effects
                .iter()
                .filter_map(|effect| match effect {
                    AmbientEffect::Particle(particle) => Some(particle.position),
                    AmbientEffect::Sound(_) => None,
                })
                .collect()
        }

        let mut world = World::new();
        let pos = BlockPos::new(2, 64, 2);
        world.load_empty_chunk(pos.chunk_pos());
        let (x, y, z) = (2.0, 64.0, 2.0);
        let wall_y = y + 0.7 + 0.22;
        for (meta, expected) in [
            (5, DVec3::new(x + 0.5, y + 0.7, z + 0.5)),
            // EAST hangs on the west wall, so the flame sits west of the block centre.
            (1, DVec3::new(x + 0.5 - 0.27, wall_y, z + 0.5)),
            (2, DVec3::new(x + 0.5 + 0.27, wall_y, z + 0.5)),
            (3, DVec3::new(x + 0.5, wall_y, z + 0.5 - 0.27)),
            (4, DVec3::new(x + 0.5, wall_y, z + 0.5 + 0.27)),
        ] {
            let torch = BlockState::new(BlockId(50), meta);
            world.set_block(pos, torch);
            let mut callback = JavaRandom::new(7);
            let mut world_rng = JavaRandom::new(11);
            let effects = world.random_display_tick_at(pos, torch, &mut callback, &mut world_rng);
            let positions = particle_positions(&effects);
            assert_eq!(positions.len(), 2, "smoke and flame for meta {meta}");
            for position in positions {
                assert!(
                    (position - expected).length() < 1e-12,
                    "torch meta {meta}: {position:?}, expected {expected:?}"
                );
            }
        }

        let redstone = BlockState::new(BlockId(76), 5);
        world.set_block(pos, redstone);
        let mut callback = JavaRandom::new(7);
        let mut draws = callback.clone();
        let mut world_rng = JavaRandom::new(11);
        let effects = world.random_display_tick_at(pos, redstone, &mut callback, &mut world_rng);
        let expected = DVec3::new(
            x + 0.5 + (draws.next_f64() - 0.5) * 0.2,
            y + 0.7 + (draws.next_f64() - 0.5) * 0.2,
            z + 0.5 + (draws.next_f64() - 0.5) * 0.2,
        );
        assert_eq!(particle_positions(&effects), vec![expected]);
    }

    #[test]
    fn temperature_noise_matches_java_seed_1234_at_high_altitude_boundaries() {
        // These are the f32 bits emitted by the exact 1.8.9 NoiseGeneratorSimplex/Perlin
        // calculation. The pair straddles the snow threshold for Extreme Hills at y=94;
        // keeping both signs also catches a dropped `nextDouble` or an ordinary (rather than
        // Java's strict-positive) floor for negative coordinates.
        assert_eq!(temperature_noise_at(0, 0).to_bits(), 0x0000_0000);
        assert_eq!(temperature_noise_at(0, 1).to_bits(), 0xBFF6_9B05);
        assert_eq!(temperature_noise_at(0, -1).to_bits(), 0x3FF6_9B9E);
        assert_eq!(temperature_noise_at(-1, -1).to_bits(), 0x4057_317F);
        assert_eq!(temperature_noise_at(1, 0).to_bits(), 0xBFF6_9B39);
        assert_eq!(temperature_noise_at(-1, 0).to_bits(), 0x3FF6_9B9E);
        assert_eq!(temperature_noise_at(1_234, -9_876).to_bits(), 0xBFEF_E1F4);
        assert_eq!(temperature_noise_at(-1_234, 9_876).to_bits(), 0xBFC0_3249);

        let hills = mc_data::biome(3);
        let warm_side = hills.temperature_at(94, temperature_noise_at(0, 1));
        let cold_side = hills.temperature_at(94, temperature_noise_at(0, -1));
        assert_eq!(warm_side.to_bits(), 0x3E1C_E359);
        assert_eq!(cold_side.to_bits(), 0x3E16_4FD9);
        assert!(warm_side > 0.15, "{warm_side}");
        assert!(cold_side < 0.15, "{cold_side}");
    }

    #[test]
    fn low_altitude_temperature_ignores_the_column_noise() {
        let plains = mc_data::biome(1);
        let noise = temperature_noise_at(-1, -1);
        assert_eq!(plains.temperature_at(0, noise), plains.temperature);
        assert_eq!(plains.temperature_at(64, noise), plains.temperature);
    }

    #[test]
    fn high_altitude_rain_gate_uses_negative_column_noise() {
        let mut world = World::new();
        let positive_z = BlockPos::new(0, 94, 1);
        let negative_z = BlockPos::new(0, 94, -1);
        world.load_empty_chunk(positive_z.chunk_pos());
        world.load_empty_chunk(negative_z.chunk_pos());

        // Extreme Hills is a rainy, non-snow biome with nominal temperature 0.2. At y=94,
        // the Java temperature noise above makes (0, 1) stay just over 0.15 while (0, -1)
        // falls just under it.
        world
            .chunks
            .get_mut(&positive_z.chunk_pos())
            .map(Arc::make_mut)
            .expect("positive-z test chunk")
            .set_biome((positive_z.x & 15) as u8, (positive_z.z & 15) as u8, 3);
        world
            .chunks
            .get_mut(&negative_z.chunk_pos())
            .map(Arc::make_mut)
            .expect("negative-z test chunk")
            .set_biome((negative_z.x & 15) as u8, (negative_z.z & 15) as u8, 3);
        assert!(world.is_raining_at(positive_z, true));
        assert!(!world.is_raining_at(negative_z, true));
    }

    #[test]
    fn liquid_callbacks_emit_lava_pop_and_audio_from_the_same_callback_stream() {
        let mut world = World::new();
        let pos = BlockPos::new(2, 64, 2);
        world.load_empty_chunk(pos.chunk_pos());
        let lava = BlockState::new(BlockId(11), 0);
        world.set_block(pos, lava);

        let (seed, effects) = (0_i64..2_000)
            .find_map(|seed| {
                let mut callback = JavaRandom::new(seed);
                let mut world_rng = JavaRandom::new(99);
                let effects =
                    world.random_display_tick_at(pos, lava, &mut callback, &mut world_rng);
                effects
                    .iter()
                    .any(|effect| {
                        matches!(
                            effect,
                            AmbientEffect::Particle(AmbientParticleEffect {
                                particle_id: ambient_particle::LAVA,
                                ..
                            })
                        )
                    })
                    .then_some((seed, effects))
            })
            .expect("a lava-pop callback must occur in the search window");
        assert!(seed >= 0);
        assert!(effects.iter().any(|effect| {
            matches!(
                effect,
                AmbientEffect::Sound(AmbientSoundEffect {
                    name: "liquid.lavapop",
                    ..
                })
            )
        }));
    }

    #[test]
    fn raining_is_loaded_only_and_flammability_reads_block_states() {
        let mut world = World::new();
        let feet = BlockPos::new(2, 64, 2);
        world.load_empty_chunk(feet.chunk_pos());
        assert!(world.is_raining_at(feet, true));
        assert!(world.is_raining_at(feet.up(), true));
        assert!(!world.is_raining_at(feet, false));

        let fire = BlockState::new(BlockId(51), 0);
        world.set_block(feet, fire);
        let box_around_feet = Aabb::from_bounds(1.9, 63.9, 1.9, 2.1, 64.1, 2.1);
        assert!(world.is_flammable_within(&box_around_feet));

        // java: `isFlammableWithin` reads `World.getBlockState`, so fire written into an absent
        // column (the shared blank chunk) burns there too, as it does in vanilla.
        let mut phantom = World::new();
        phantom.set_block(feet, fire);
        assert!(phantom.is_flammable_within(&box_around_feet));
        assert!(
            !World::new().is_flammable_within(&box_around_feet),
            "control: a fresh world's blank chunk is air"
        );
    }
}
