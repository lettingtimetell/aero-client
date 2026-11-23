//! Block states, block properties, and the packed-state-id lookup tables.
//!
//! # State packing
//!
//! 1.8.9 puts a block state on the wire as `block_id << 4 | meta` in a `u16`. That is what
//! chunk sections, `S23PacketBlockChange` and `S22PacketMultiBlockChange` all carry, and it
//! is what [`BlockState`] holds.
//!
//! Not every one of those combinations is legal: `minecraft:torch` has no `meta = 0`, most
//! blocks use only a handful of the 16 metas, and a hostile or simply buggy server can send
//! anything. Vanilla resolves an unknown packed id to air rather than failing
//! (`ExtendedBlockStorage.get` falls back to `Blocks.air.getDefaultState()`), and so do we -
//! [`block_props`] is total over the whole `u16` range and never panics.
//!
//! Note that `Block.getStateId`/`getStateById` in the Java use a *different* packing,
//! `block_id | meta << 12`. That one only shows up in a few packets (falling-block spawn,
//! block-break particles); [`BlockState::from_java_state_id`] converts it.

use crate::generated::blocks::{
    BLOCK_INFO, BLOCK_NAMES, BLOCK_PROPS, JAVA_STATE_FROM_ID, JAVA_STATE_VARIANT_IDX, STATE_PROPS,
    STATE_VARIANTS, STATE_VARIANT_IDX,
};
use crate::generated::debug_states::DEBUG_BLOCKS;
use crate::generated::materials::Material;
use crate::item::ItemId;

pub use crate::generated::blocks::{MAX_BLOCK_ID, STATE_COUNT};

/// A block registry id, `0..=4095` on the wire and `0..=197` in vanilla 1.8.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BlockId(pub u16);

impl BlockId {
    /// `minecraft:air`.
    pub const AIR: BlockId = BlockId(0);

    /// Registry metadata for this block, or the air entry for unregistered ids.
    #[inline]
    #[must_use]
    pub fn info(self) -> &'static BlockInfo {
        let i = self.0 as usize;
        if i < BLOCK_INFO.len() {
            &BLOCK_INFO[i]
        } else {
            &BLOCK_INFO[0]
        }
    }

    /// The state chosen by Java's `Block.getDefaultState()`.
    ///
    /// This is a constructor-selected property tuple, not necessarily storage metadata zero:
    /// for example, 1.8.9's torch default is meta 5 and every stair default is meta 3.
    #[inline]
    #[must_use]
    pub fn default_state(self) -> BlockState {
        BlockState::new(self, self.info().default_meta)
    }

    /// Whether vanilla 1.8.9 actually registers this id.
    #[inline]
    #[must_use]
    pub fn is_registered(self) -> bool {
        self.info().registered
    }
}

/// The two per-block values `BlockFire` uses while spreading and extinguishing.
///
/// These are deliberately keyed by [`BlockId`], rather than by material or packed state:
/// Java's `BlockFire.init` fills two `IdentityHashMap<Block, Integer>`s.  Consequently every
/// metadata variant of one registered block has the same values, while two wood blocks may
/// differ (and many otherwise plant/wood-material blocks are absent and therefore zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FireInfo {
    /// Java's `encouragements` entry (`getEncouragement`), or zero when no entry exists.
    pub encouragement: u8,
    /// Java's `flammabilities` entry (`getFlammability`), or zero when no entry exists.
    pub flammability: u8,
}

/// The `BlockFire.init` values for a registered block.
///
/// The source calls `setFireInfo` 35 times.  It does not infer either number from a block's
/// material, and an absent map entry reads as zero in `BlockFire.getEncouragement` and
/// `BlockFire.getFlammability`.
#[inline]
#[must_use]
pub const fn fire_info(block: BlockId) -> FireInfo {
    let (encouragement, flammability) = match block.0 {
        // planks, wooden slabs, all six fence gates, all six fences, and the four stairs
        // that existed when this table was written.  Acacia/dark-oak stairs intentionally do
        // not occur in 1.8.9's initializer.
        5 | 53 | 85 | 107 | 125 | 126 | 134 | 135 | 136 | 183..=192 => (5, 20),
        // Both log families and the coal block.
        17 | 162 | 173 => (5, 5),
        // Leaves and wool.
        18 | 35 | 161 => (30, 60),
        // Bookshelf.
        47 => (30, 20),
        // TNT.
        46 => (15, 100),
        // Tall grass, flowers, dead bush, and double plants.
        31 | 32 | 37 | 38 | 175 => (60, 100),
        // Vines.
        106 => (15, 100),
        // Hay and carpet.
        170 | 171 => (60, 20),
        _ => (0, 0),
    };
    FireInfo {
        encouragement,
        flammability,
    }
}

/// Java `BlockFire.getEncouragement(block)`: zero for blocks absent from its identity map.
#[inline]
#[must_use]
pub const fn fire_encouragement(block: BlockId) -> u8 {
    fire_info(block).encouragement
}

/// Java `BlockFire.getFlammability(block)`: zero for blocks absent from its identity map.
#[inline]
#[must_use]
pub const fn fire_flammability(block: BlockId) -> u8 {
    fire_info(block).flammability
}

/// Java `BlockFire.canCatchFire`: whether its identity-map encouragement is nonzero.
///
/// This deliberately derives from [`fire_info`] instead of restating the initializer's block-id
/// set.  `BlockFire.canCatchFire(IBlockAccess, BlockPos)` calls `getEncouragement` on the block
/// state at that position and compares it to zero; metadata has no effect because the Java maps
/// are keyed by the block instance.
#[inline]
#[must_use]
pub const fn can_catch_fire(block: BlockId) -> bool {
    fire_encouragement(block) > 0
}

/// Per-block (not per-state) registry data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockInfo {
    /// The block's numeric registry id.
    pub id: BlockId,
    /// Registry name, e.g. `minecraft:stone`.
    pub name: &'static str,
    /// Translation key stem, e.g. `tile.stone`.
    pub unlocalized: &'static str,
    /// `en_US` display name, e.g. `Stone`.
    pub display_name: &'static str,
    /// False for ids vanilla leaves empty; those resolve to air everywhere.
    pub registered: bool,
    /// The item form of this block, when one exists.
    pub item: Option<ItemId>,
    /// Step-sound family, e.g. `stone` (vanilla plays `step.stone`, `dig.stone`).
    pub step_sound: &'static str,
    /// Exact virtual `Block.SoundType.getBreakSound()` event, e.g. `dig.glass`.
    ///
    /// This is separate from [`Self::step_sound`]: glass, ladder, anvil and slime override the
    /// otherwise tempting `"dig." + soundName` derivation.
    pub break_sound: &'static str,
    /// Step-sound volume.
    pub step_sound_volume: f32,
    /// Step-sound pitch.
    pub step_sound_pitch: f32,
    /// Originating Java class, for tracing a port back to the source.
    pub java_class: &'static str,
    /// `Block.getMobilityFlag()` — `0` a piston moves it, `1` a piston destroys it, `2` a piston
    /// cannot move it at all.
    ///
    /// java: `Block.java:1091` returns `this.blockMaterial.getMaterialMobility()`, overridden to a
    /// constant by five classes (`BlockBasePressurePlate`, `BlockBed`, `BlockDoor`, `BlockIce`,
    /// `BlockRailBase`). The extractor calls the virtual, so this is the *effective* value and not
    /// the material's — the two differ for exactly those five.
    ///
    /// Read by `BlockPistonBase.canPush` and `BlockPistonStructureHelper`, which run on the client
    /// (`onBlockEventReceived` is only `!isRemote`-gated down to its `shouldBeExtended` pre-check).
    pub mobility_flag: u8,
    /// Storage metadata corresponding to Java's `Block.getDefaultState()`.
    pub default_meta: u8,
}

/// An axis-aligned box in block-local coordinates (`0.0..=1.0` for a normal cube).
///
/// `mc-math` (S05) owns the general-purpose `Aabb` used by physics; this one is a
/// dependency-free `const`-constructible mirror so `mc-data` stays a leaf crate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Aabb {
    /// Minimum X.
    pub min_x: f64,
    /// Minimum Y.
    pub min_y: f64,
    /// Minimum Z.
    pub min_z: f64,
    /// Maximum X.
    pub max_x: f64,
    /// Maximum Y.
    pub max_y: f64,
    /// Maximum Z.
    pub max_z: f64,
}

impl Aabb {
    /// The unit cube.
    pub const UNIT: Aabb = Aabb::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);

    /// Builds a box from its two corners. No normalisation is performed; the extractor
    /// always emits `min <= max`.
    #[inline]
    #[must_use]
    pub const fn new(
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Self {
        Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    /// `[min_x, min_y, min_z]`.
    #[inline]
    #[must_use]
    pub const fn min(&self) -> [f64; 3] {
        [self.min_x, self.min_y, self.min_z]
    }

    /// `[max_x, max_y, max_z]`.
    #[inline]
    #[must_use]
    pub const fn max(&self) -> [f64; 3] {
        [self.max_x, self.max_y, self.max_z]
    }

    /// True when the box encloses no volume at all. Vanilla emits several of these (a
    /// `snow_layer` with one layer has a zero-height collision box).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min_x >= self.max_x || self.min_y >= self.max_y || self.min_z >= self.max_z
    }

    /// Translates the box by a block position.
    #[inline]
    #[must_use]
    pub fn offset(&self, x: f64, y: f64, z: f64) -> Aabb {
        Aabb::new(
            self.min_x + x,
            self.min_y + y,
            self.min_z + z,
            self.max_x + x,
            self.max_y + y,
            self.max_z + z,
        )
    }
}

/// The collision volume of a block state.
///
/// Extracted by *running* `Block.addCollisionBoxesToList` against every legal state, which
/// is the only reliable way to get 1.8.9's imperatively-assigned bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollisionShape {
    /// No collision: air, plants, liquids, rails, torches, ...
    Empty,
    /// Exactly the unit cube. Broken out from [`CollisionShape::Boxes`] because it is by
    /// far the most common case and lets the sweep in S13 skip a slice read.
    Full,
    /// One or more explicit boxes in block-local coordinates.
    Boxes(&'static [Aabb]),
}

impl CollisionShape {
    /// The boxes making up this shape. `Empty` yields an empty slice; `Full` yields the
    /// unit cube. Never allocates.
    #[inline]
    #[must_use]
    pub fn boxes(&self) -> &'static [Aabb] {
        const FULL: [Aabb; 1] = [Aabb::UNIT];
        match self {
            CollisionShape::Empty => &[],
            CollisionShape::Full => &FULL,
            CollisionShape::Boxes(b) => b,
        }
    }

    /// True when nothing can collide with this block.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, CollisionShape::Empty)
    }
}

/// How vanilla draws a block, from `Block.getRenderType()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum RenderType {
    /// `-1`: nothing is drawn (air, barrier, structure voids).
    Invisible = -1,
    /// `1`: fluid, drawn by the special liquid renderer with interpolated corner heights.
    Liquid = 1,
    /// `2`: the **chest family, and only it**.
    ///
    /// java: `BlockContainer.getRenderType()` returns `-1` (`block/BlockContainer.java:37-40`);
    /// the only two classes in 1.8.9 that override it to `2` are `BlockChest` and
    /// `BlockEnderChest`, i.e. blocks 54, 130 and 146. A previous version of this comment listed
    /// "chest, sign, skull, banner"; signs (63/68), skulls (144) and banners (176/177) all
    /// report **`-1`**, and the extracted table agrees. Having a TESR and having render type 2
    /// are unrelated questions — most tile entities are `-1` (invisible terrain geometry, drawn
    /// entirely by the TESR) and some are `3` (a real block model *plus* a TESR: the mob
    /// spawner, the enchanting table, the beacon).
    ///
    /// What the `2` actually selects is the **item**-form renderer:
    /// `RenderItem.renderItemModelIntoGUI` → `TileEntityItemStackRenderer`.
    TileEntity = 2,
    /// `3`: the normal path - a baked JSON model from `assets/minecraft/models/block`.
    Model = 3,
}

/// Which of vanilla's four terrain render passes a block belongs to
/// (`Block.getBlockLayer()`). S11 keys its per-section vertex buffers off this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RenderLayer {
    /// Fully opaque geometry.
    Solid,
    /// Alpha-tested with mipmapping (leaves).
    CutoutMipped,
    /// Alpha-tested without mipmapping (glass, plants, rails).
    Cutout,
    /// Alpha-blended, needs back-to-front sorting (water, ice, stained glass).
    Translucent,
}

/// Everything the mesher, the physics sweep and the F3 overlay need about one block state.
///
/// Rows are deduplicated across the 4096-entry state table, so this struct is shared and
/// must stay `'static`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockProps {
    /// Collision volume in block-local coordinates.
    pub collision: CollisionShape,
    /// Light this block emits, `0..=15`.
    pub light_emission: u8,
    /// Light this block absorbs, `0..=255` (`255` means fully opaque).
    pub light_opacity: u8,
    /// Break hardness. `-1.0` means unbreakable (bedrock, barrier, end portal frame).
    pub hardness: f32,
    /// How vanilla draws this block.
    pub render_type: RenderType,
    /// The block's material.
    pub material: Material,

    /// The block id this state decodes to; `BlockId::AIR` for illegal states.
    ///
    /// Metadata is deliberately absent: it is `state.meta()`, and leaving it out lets rows
    /// that differ only by meta share a table slot.
    pub block: BlockId,
    /// False when the packed id is not a state vanilla registers. Illegal states get the
    /// air row, exactly as `ExtendedBlockStorage.get` does.
    pub legal: bool,
    /// Render/selection bounds (`Block.setBlockBoundsBasedOnState`). Often wider than
    /// [`BlockProps::collision`] - a cactus draws full-width but collides inset by 1/16.
    pub bounds: Aabb,
    /// Terrain render pass.
    pub render_layer: RenderLayer,
    /// Blast resistance.
    pub resistance: f32,
    /// Ground friction. `0.6` normally, `0.98` for ice, `0.8` for slime.
    pub slipperiness: f32,
    /// `Block.getAmbientOcclusionLightValue`, the AO base used by the smooth-lighting pass.
    pub ambient_occlusion_light: f32,
    /// Packed RGB map colour.
    pub map_color: u32,
    /// `Block.isOpaqueCube` - the face-culling test the mesher uses.
    pub opaque_cube: bool,
    /// `Block.isFullCube`.
    pub full_cube: bool,
    /// `Block.isFullBlock`.
    pub full_block: bool,
    /// `Block.isNormalCube` - `blockMaterial.isOpaque() && isFullCube() && !canProvidePower()`.
    ///
    /// java: `block/Block.java:356-359`. The old wording here said "and not a tile entity",
    /// which is a different predicate (`!isBlockContainer`, used by `canSilkHarvest` at
    /// `:1006-1009`). The two disagree in both directions: `canProvidePower` is overridden by 11
    /// classes, of which `BlockChest` is the only tile entity and `BlockCompressedPowered`
    /// (the redstone block, 152) is a full opaque cube that `isNormalCube` therefore rejects —
    /// while the furnace (61) is a tile entity that `isNormalCube` happily accepts. Both are
    /// visible in the extracted table: `redstone_block.normal_cube == false` with
    /// `opaque_cube == full_cube == true`, and `furnace.normal_cube == true`.
    pub normal_cube: bool,
    /// `Block.isVisuallyOpaque`.
    pub visually_opaque: bool,
    /// `Block.isTranslucent`, used by the AO pass.
    pub translucent: bool,
    /// `Block.getUseNeighborBrightness` - stairs and slabs sample a neighbour's light.
    pub use_neighbor_brightness: bool,
    /// `Block.hasTileEntity()` — literally `return this.isBlockContainer`
    /// (`block/Block.java:436-439`).
    ///
    /// **Also a faithful stand-in for `block instanceof ITileEntityProvider`, and that is a
    /// checked fact rather than an assumption.** Vanilla uses both spellings and uses them in
    /// different places — `hasTileEntity()` at `RenderChunk:158`, `EntityRenderer:1252`,
    /// `Chunk.createNewTileEntity:915` and `Chunk:1238`; the `instanceof` at
    /// `Chunk.setBlockState:705/744/759`, `Chunk:958`, `BlockPistonBase.canPush:363` and
    /// `EntityFallingBlock:138` — so it is natural to assume the sets differ. In 1.8.9 they do
    /// not. `isBlockContainer` is written in exactly two places: `BlockContainer`'s
    /// constructor (`BlockContainer.java:21`) **and** `BlockRedstoneComparator`'s
    /// (`BlockRedstoneComparator.java:38`), which is the one class that implements
    /// `ITileEntityProvider` without extending `BlockContainer` and sets the flag by hand to
    /// compensate. Blocks 149/150 are therefore `true` here, and a caller reproducing the
    /// `instanceof` needs no special case.
    ///
    /// `mc-data/tests/golden_extractor.rs` asserts the two sets are equal against the live
    /// registry, so this stops being true loudly rather than silently.
    pub has_tile_entity: bool,
    /// `Block.isPassable` with air neighbours.
    pub passable: bool,
    /// True when the real collision shape depends on surrounding blocks, so
    /// [`BlockProps::collision`] is only the isolated case.
    ///
    /// In 1.8.9 this is 33 blocks / 195 states: fences, cobblestone walls, glass panes and
    /// iron bars, doors, chests, and stair corners. S13 must recompute those from
    /// neighbours; everything else can trust the table as-is.
    ///
    /// Fence *gates* are deliberately not in that set. Their `in_wall` property is an
    /// actual-state/model-variant bit: `BlockFenceGate.getActualState` sets it when a wall
    /// touches the gate, so it selects a different baked model. Neither
    /// `getCollisionBoundingBox` nor `setBlockBoundsBasedOnState` reads that bit; both use
    /// only `open` and `facing` in 1.8.9.
    pub collision_depends_on_neighbors: bool,
}

impl BlockProps {
    /// True when this state is `minecraft:air` (including the illegal-state sentinel).
    #[inline]
    #[must_use]
    pub fn is_air(&self) -> bool {
        self.block.0 == 0
    }

    /// True when the block cannot be broken by hand or tool.
    #[inline]
    #[must_use]
    pub fn is_unbreakable(&self) -> bool {
        self.hardness < 0.0
    }

    /// True when an entity is stopped by this block's material (false for cobwebs, which
    /// slow rather than block).
    #[inline]
    #[must_use]
    pub fn blocks_movement(&self) -> bool {
        self.material.blocks_movement()
    }
}

/// A typed vanilla block-state property.
///
/// The generated registry stores property names and values as static strings because those are
/// what the wire/state-model tables need.  A [`Property`] adds the missing `IProperty<T>` layer:
/// callers provide the stable typed value set once, and [`BlockState`] validates reads and
/// derives sibling states without falling back to stringly-typed metadata edits.
///
/// Values are ordered exactly as supplied.  That order is significant to
/// [`BlockState::cycle_property`], just as `IProperty.getAllowedValues()` iteration order is in
/// 1.8.9.  The value slice must be `'static`, which makes descriptors suitable for constants and
/// prevents a state operation from borrowing a temporary value table.
#[derive(Debug, Clone, Copy)]
pub struct Property<T: 'static> {
    name: &'static str,
    values: &'static [(&'static str, T)],
}

/// The six 1.8.9 `EnumFacing` values, in Java declaration/`values()` order.
///
/// `mc-data` intentionally does not depend on `mc-math`, so this small registry-facing enum is
/// kept next to [`PropertyDirection`].  It is only the typed property value; movement and vector
/// operations remain owned by `mc_math::Facing`.  The discriminants still match Java's
/// `EnumFacing.getIndex()` order (`DOWN, UP, NORTH, SOUTH, WEST, EAST`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum EnumFacing {
    /// Negative Y.
    Down = 0,
    /// Positive Y.
    Up = 1,
    /// Negative Z.
    North = 2,
    /// Positive Z.
    South = 3,
    /// Negative X.
    West = 4,
    /// Positive X.
    East = 5,
}

/// Direction is a Rust-friendly alias for [`EnumFacing`].
pub type Direction = EnumFacing;

impl EnumFacing {
    /// All six directions in Java's `EnumFacing.values()` order.
    pub const VALUES: [Self; 6] = [
        Self::Down,
        Self::Up,
        Self::North,
        Self::South,
        Self::West,
        Self::East,
    ];

    /// Returns the immutable `EnumFacing.values()` declaration-order view.
    #[must_use]
    pub const fn values() -> &'static [Self] {
        &Self::VALUES
    }

    /// Returns the lower-case serialized name used by `IStringSerializable.getName()`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Down => "down",
            Self::Up => "up",
            Self::North => "north",
            Self::South => "south",
            Self::West => "west",
            Self::East => "east",
        }
    }

    /// Java-compatible spelling of [`Self::name`].
    #[must_use]
    pub const fn get_name(self) -> &'static str {
        self.name()
    }

    /// Looks up a direction by name, matching `EnumFacing.byName`'s case folding.
    ///
    /// This is deliberately distinct from [`PropertyDirection::parse_value`]: Java's
    /// `EnumFacing.byName` accepts uppercase input, while `PropertyEnum`'s serialized-name map
    /// is an exact (case-sensitive) lookup.
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "down" => Some(Self::Down),
            "up" => Some(Self::Up),
            "north" => Some(Self::North),
            "south" => Some(Self::South),
            "west" => Some(Self::West),
            "east" => Some(Self::East),
            _ => None,
        }
    }
}

const EMPTY_DIRECTION_VALUE: (&str, EnumFacing) = ("", EnumFacing::Down);

const fn direction_values_array() -> [(&'static str, EnumFacing); 6] {
    [
        ("down", EnumFacing::Down),
        ("up", EnumFacing::Up),
        ("north", EnumFacing::North),
        ("south", EnumFacing::South),
        ("west", EnumFacing::West),
        ("east", EnumFacing::East),
    ]
}

static ALL_DIRECTION_VALUES: [(&str, EnumFacing); 6] = direction_values_array();

/// A typed `PropertyDirection` descriptor.
///
/// Java's predicate factory filters a fresh `EnumFacing.values()` list, so filtered descriptors
/// retain declaration order.  This descriptor keeps its own fixed-size immutable value array;
/// that makes the collection factory useful for live `BlockState` reads/replacements/cycles
/// without allocating or leaking a temporary `'static` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropertyDirection {
    name: &'static str,
    values: [(&'static str, EnumFacing); 6],
    len: u8,
}

impl PropertyDirection {
    /// Creates a direction property containing all six directions.
    #[must_use]
    pub const fn create(name: &'static str) -> Self {
        Self {
            name,
            values: direction_values_array(),
            len: 6,
        }
    }

    /// Creates a property by applying a predicate in `EnumFacing.values()` order.
    #[must_use]
    pub fn create_with_predicate<F>(name: &'static str, filter: F) -> Self
    where
        F: Fn(EnumFacing) -> bool,
    {
        let mut property = Self {
            name,
            values: [EMPTY_DIRECTION_VALUE; 6],
            len: 0,
        };
        for facing in EnumFacing::VALUES {
            if filter(facing) {
                property.values[property.len as usize] = (facing.name(), facing);
                property.len += 1;
            }
        }
        property
    }

    /// Alias for [`Self::create_with_predicate`], named after Java's filtered factory.
    #[must_use]
    pub fn create_filtered<F>(name: &'static str, filter: F) -> Self
    where
        F: Fn(EnumFacing) -> bool,
    {
        Self::create_with_predicate(name, filter)
    }

    /// Alias for [`Self::create_with_predicate`].
    #[must_use]
    pub fn create_with_filter<F>(name: &'static str, filter: F) -> Self
    where
        F: Fn(EnumFacing) -> bool,
    {
        Self::create_with_predicate(name, filter)
    }

    /// Creates a property from a collection of direction values.
    ///
    /// As with Guava's `ImmutableSet.copyOf`, duplicate values are removed.  The supplied
    /// collection's order is retained, which matters to `cycle_property`.
    #[must_use]
    pub fn create_from_values<V>(name: &'static str, values: V) -> Self
    where
        V: AsRef<[EnumFacing]>,
    {
        let mut property = Self {
            name,
            values: [EMPTY_DIRECTION_VALUE; 6],
            len: 0,
        };
        for &facing in values.as_ref() {
            if property.values[..property.len as usize]
                .iter()
                .all(|(_, existing)| *existing != facing)
            {
                property.values[property.len as usize] = (facing.name(), facing);
                property.len += 1;
                if property.len == 6 {
                    break;
                }
            }
        }
        property
    }

    /// Alias for [`Self::create_from_values`].
    #[must_use]
    pub fn from_values<V>(name: &'static str, values: V) -> Self
    where
        V: AsRef<[EnumFacing]>,
    {
        Self::create_from_values(name, values)
    }

    /// Alias for the Java `create(name, Collection<EnumFacing>)` factory.
    #[must_use]
    pub fn create_from_collection<V>(name: &'static str, values: V) -> Self
    where
        V: AsRef<[EnumFacing]>,
    {
        Self::create_from_values(name, values)
    }

    /// The property registry name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// The immutable allowed values in declaration/collection order.
    #[must_use]
    pub fn values(&self) -> &[(&'static str, EnumFacing)] {
        &self.values[..self.len as usize]
    }

    /// Java-style name for [`Self::values`].
    #[must_use]
    pub fn get_allowed_values(&self) -> &[(&'static str, EnumFacing)] {
        self.values()
    }

    /// Returns the serialized name for a direction, matching `PropertyEnum.getName`.
    #[must_use]
    pub const fn get_name(self, value: EnumFacing) -> &'static str {
        value.name()
    }

    /// Parses an allowed serialized value using Java's exact property-name lookup.
    ///
    /// Unlike [`EnumFacing::by_name`], uppercase input is rejected.
    #[must_use]
    pub fn parse_value(&self, name: &str) -> Option<EnumFacing> {
        self.values()
            .iter()
            .find(|(serialized, _)| *serialized == name)
            .map(|(_, value)| *value)
    }

    /// Java-style alias for [`Self::parse_value`].
    #[must_use]
    pub fn parse(&self, name: &str) -> Option<EnumFacing> {
        self.parse_value(name)
    }

    /// Returns the descriptor's direction value as the next cycle value, wrapping at the end.
    #[must_use]
    pub fn cycle_value(&self, current: EnumFacing) -> Option<EnumFacing> {
        let index = self
            .values()
            .iter()
            .position(|(_, value)| *value == current)?;
        let values = self.values();
        Some(values[(index + 1) % values.len()].1)
    }
}

impl<T: 'static> Property<T> {
    /// Creates a typed property from its registry name and ordered `(serialized, typed)` values.
    ///
    /// A descriptor is normally declared as a `const`, for example:
    /// `const OPEN: Property<bool> = Property::boolean("open");`.
    #[must_use]
    pub const fn new(name: &'static str, values: &'static [(&'static str, T)]) -> Self {
        Self { name, values }
    }

    /// The registry name used by this property.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// The ordered typed values accepted by this property.
    #[must_use]
    pub const fn values(self) -> &'static [(&'static str, T)] {
        self.values
    }
}

/// The typed property surface consumed by [`BlockState`] operations.
///
/// [`Property`] is the zero-allocation static form used by generated-style descriptors;
/// [`PropertyDirection`] is the fixed-size form needed for Java's runtime predicate/collection
/// factories.  Keeping this trait small lets both forms share exact read/replacement/cycle
/// semantics without changing the generated registry tables.
pub trait PropertyDescriptor<T: Copy + PartialEq + 'static> {
    /// The registry property name.
    fn name(&self) -> &'static str;

    /// The immutable allowed values in cycle order.
    fn values(&self) -> &[(&'static str, T)];

    /// Parses an exact serialized value name.
    fn parse_value(&self, name: &str) -> Option<T> {
        self.values()
            .iter()
            .find(|(serialized, _)| *serialized == name)
            .map(|(_, value)| *value)
    }

    /// Serializes an allowed typed value.
    fn serialize_value(&self, value: T) -> Option<&'static str> {
        self.values()
            .iter()
            .find(|(_, candidate)| *candidate == value)
            .map(|(serialized, _)| *serialized)
    }

    /// Returns the cycle-order index of an allowed value.
    fn value_index(&self, value: T) -> Option<usize> {
        self.values()
            .iter()
            .position(|(_, candidate)| *candidate == value)
    }
}

impl<T: Copy + PartialEq + 'static> PropertyDescriptor<T> for Property<T> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn values(&self) -> &[(&'static str, T)] {
        self.values
    }
}

impl<T: Copy + PartialEq + 'static, P: PropertyDescriptor<T> + ?Sized> PropertyDescriptor<T>
    for &P
{
    fn name(&self) -> &'static str {
        (*self).name()
    }

    fn values(&self) -> &[(&'static str, T)] {
        (*self).values()
    }
}

impl PropertyDescriptor<EnumFacing> for PropertyDirection {
    fn name(&self) -> &'static str {
        self.name
    }

    fn values(&self) -> &[(&'static str, EnumFacing)] {
        self.values()
    }
}

impl Property<EnumFacing> {
    /// Creates the static all-direction descriptor in Java's declaration order.
    #[must_use]
    pub const fn direction(name: &'static str) -> Self {
        Self::new(name, &ALL_DIRECTION_VALUES)
    }

    /// Creates a runtime filtered direction descriptor.  Use the returned
    /// [`PropertyDirection`] directly with [`BlockState`] operations.
    #[must_use]
    pub fn direction_filtered<F>(name: &'static str, filter: F) -> PropertyDirection
    where
        F: Fn(EnumFacing) -> bool,
    {
        PropertyDirection::create_with_predicate(name, filter)
    }
}

impl Property<bool> {
    /// Creates the standard two-value `true`/`false` property used by vanilla boolean states.
    ///
    /// java: `PropertyBool.allowedValues` is `ImmutableSet.of(true, false)`, and that iteration
    /// order is part of the exposed `IProperty.getAllowedValues()` contract.
    #[must_use]
    pub const fn boolean(name: &'static str) -> Self {
        static VALUES: [(&str, bool); 2] = [("true", true), ("false", false)];
        Self::new(name, &VALUES)
    }
}

impl<T: Copy + 'static> From<&Property<T>> for Property<T> {
    fn from(property: &Property<T>) -> Self {
        *property
    }
}

impl<T: Copy + PartialEq + 'static> Property<T> {
    /// Finds the typed value encoded by a generated registry string.
    #[must_use]
    pub fn parse_value(self, name: &str) -> Option<T> {
        self.values
            .iter()
            .find(|(serialized, _)| *serialized == name)
            .map(|(_, value)| *value)
    }

    /// Finds the generated registry string for a typed value.
    #[must_use]
    pub fn serialize_value(self, value: T) -> Option<&'static str> {
        self.values
            .iter()
            .find(|(_, candidate)| *candidate == value)
            .map(|(serialized, _)| *serialized)
    }
}

/// A packed 1.8.9 block state: `block_id << 4 | meta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BlockState {
    /// The packed id, exactly as it appears in a chunk section.
    pub id: u16,
}

impl BlockState {
    /// `minecraft:air`.
    pub const AIR: BlockState = BlockState { id: 0 };

    /// Packs a block id and metadata. `meta` is masked to 4 bits, `block` to 12.
    #[inline]
    #[must_use]
    pub const fn new(block: BlockId, meta: u8) -> Self {
        Self {
            id: ((block.0 & 0x0FFF) << 4) | (meta as u16 & 0x0F),
        }
    }

    /// Wraps a raw packed id from the wire. Always succeeds; illegal ids resolve to air
    /// when looked up.
    #[inline]
    #[must_use]
    pub const fn from_raw(id: u16) -> Self {
        Self { id }
    }

    /// Decodes `Block.getStateById`'s packing (`block_id | meta << 12`), which a few 1.8.9
    /// packets use instead - `S0EPacketSpawnObject` for falling blocks and `S28PacketEffect`
    /// / block-break particles.
    ///
    /// This is intentionally *not* a bit-repacking operation. Java finds the block and calls
    /// its virtual `getStateFromMeta`, which normalizes unsupported metadata to each block's
    /// own canonical state (often metadata zero). Chunk storage instead uses
    /// `BLOCK_STATE_IDS`, represented by [`BlockState::from_raw`].
    ///
    /// # Example
    /// ```
    /// use mc_data::BlockState;
    /// // sand (id 12), meta 0
    /// assert_eq!(BlockState::from_java_state_id(12).id, 12 << 4);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_java_state_id(id: i32) -> Self {
        let block = (id & 4095) as u16;
        let meta = ((id >> 12) & 15) as u16;
        let index = (block << 4) | meta;
        if (index as usize) < JAVA_STATE_FROM_ID.len() {
            let state = JAVA_STATE_FROM_ID[index as usize];
            assert!(
                state != u16::MAX,
                "Block.getStateFromMeta rejected metadata"
            );
            Self { id: state }
        } else {
            Self::AIR
        }
    }

    /// [`BlockState::from_java_state_id`] for a server-controlled id: `None` exactly where
    /// vanilla's `Block.getStateById` throws because the block's `getStateFromMeta` rejects the
    /// metadata (a hopper's meta 1 is `withProperty(FACING, UP)`, an IllegalArgumentException).
    /// Vanilla catches that exception in its packet paths, so a packet must never reach the
    /// panicking form.
    #[inline]
    #[must_use]
    pub const fn try_from_java_state_id(id: i32) -> Option<Self> {
        let block = (id & 4095) as u16;
        let meta = ((id >> 12) & 15) as u16;
        let index = (block << 4) | meta;
        if (index as usize) < JAVA_STATE_FROM_ID.len() {
            let state = JAVA_STATE_FROM_ID[index as usize];
            if state == u16::MAX {
                None
            } else {
                Some(Self { id: state })
            }
        } else {
            Some(Self::AIR)
        }
    }

    /// Encodes this packed storage state using `Block.getStateId`'s bit layout.
    ///
    /// This preserves the two fields exactly, but is not generally an inverse of
    /// [`BlockState::from_java_state_id`]: Java's decoder goes through the block's virtual
    /// `getStateFromMeta`, while this accepts any packed storage state.
    #[inline]
    #[must_use]
    pub const fn to_java_state_id(self) -> i32 {
        ((self.id >> 4) as i32) | (((self.id & 0x0F) as i32) << 12)
    }

    /// The block id this state decodes to. Pure bit arithmetic - no legality check.
    #[inline]
    #[must_use]
    pub const fn block(self) -> BlockId {
        BlockId(self.id >> 4)
    }

    /// The metadata this state decodes to. Pure bit arithmetic - no legality check.
    #[inline]
    #[must_use]
    pub const fn meta(self) -> u8 {
        (self.id & 0x0F) as u8
    }

    /// Properties for this state. Total over every `u16`; illegal states yield air.
    #[inline]
    #[must_use]
    pub fn props(self) -> &'static BlockProps {
        block_props(self)
    }

    /// Whether vanilla registers this exact packed id.
    #[inline]
    #[must_use]
    pub fn is_legal(self) -> bool {
        self.props().legal
    }

    /// Reads a typed property value, mirroring `IBlockState.getValue`.
    ///
    /// `None` means the state does not declare the property or the generated string is not one
    /// of the descriptor's typed values.  Illegal packed states likewise have no properties,
    /// matching vanilla's use of a real registered state object for all `IBlockState` calls.
    #[must_use]
    pub fn get_value<T: Copy + PartialEq + 'static, P: PropertyDescriptor<T>>(
        self,
        property: P,
    ) -> Option<T> {
        state_properties(self)
            .iter()
            .find(|(name, _)| *name == property.name())
            .and_then(|(_, value)| property.parse_value(value))
    }

    /// Derives the sibling state with one typed property replaced, mirroring
    /// `StateImplementation.withProperty`.
    ///
    /// The packed representation has no object-level transition table, so this searches the
    /// block's sixteen metadata candidates and requires every unrelated generated property to
    /// remain identical.  There are at most sixteen candidates and this path is used for client
    /// interaction prediction, not a terrain hot loop.  `None` means the property is absent, the
    /// value is not allowed, or no legal sibling carries the requested tuple.
    #[must_use]
    pub fn with_property<T: Copy + PartialEq + 'static, P: PropertyDescriptor<T>>(
        self,
        property: P,
        value: T,
    ) -> Option<BlockState> {
        let current = state_properties(self);
        if !current.iter().any(|(name, _)| *name == property.name()) {
            return None;
        }
        let serialized = property.serialize_value(value)?;

        for meta in 0..16 {
            let candidate = BlockState::new(self.block(), meta);
            if !candidate.is_legal() {
                continue;
            }
            let candidate_properties = state_properties(candidate);
            if candidate_properties.len() != current.len()
                || !current.iter().all(|(name, current_value)| {
                    candidate_properties
                        .iter()
                        .find(|(candidate_name, _)| candidate_name == name)
                        .is_some_and(|(_, candidate_value)| {
                            if *name == property.name() {
                                *candidate_value == serialized
                            } else {
                                *candidate_value == *current_value
                            }
                        })
                })
            {
                continue;
            }
            return Some(candidate);
        }
        None
    }

    /// Cycles one typed property to its next allowed value, wrapping at the end, mirroring
    /// `BlockStateBase.cycleProperty`.
    ///
    /// `None` covers an absent/invalid property and an empty descriptor.  A descriptor's value
    /// order is the cycle order, so enum-like callers should declare it in vanilla's
    /// `getAllowedValues` order.
    #[must_use]
    pub fn cycle_property<T: Copy + PartialEq + 'static, P: PropertyDescriptor<T>>(
        self,
        property: P,
    ) -> Option<BlockState> {
        let current = self.get_value(&property)?;
        let index = property.value_index(current)?;
        let next = {
            let values = property.values();
            if values.is_empty() {
                return None;
            }
            values[(index + 1) % values.len()].1
        };
        self.with_property(property, next)
    }
}

/// A block state after vanilla's client-only `Block.getActualState` pass.
///
/// [`BlockState`] deliberately remains the 12-bit state stored in chunks and sent on the
/// wire.  Properties such as a fence's four connections or a stair's corner shape have no
/// metadata encoding, so renderers carry them in this separate, short-lived value instead.
/// Property names and values come from the generated vanilla registry and are therefore
/// `'static`; changing a derived property only copies the (normally three-to-eight element)
/// property slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActualBlockState {
    stored: BlockState,
    properties: Vec<(&'static str, &'static str)>,
}

/// One property in a generated DEBUG_WORLD state schema.
///
/// Vanilla's debug provider stores `IBlockState` objects, not packed metadata.  The packed
/// [`BlockState`] type remains the wire/storage identity, so DEBUG_WORLD carries the complete
/// property tuple beside the metadata representative selected by `Block.getMetaFromState`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DebugPropertySchema {
    pub(crate) name: &'static str,
    pub(crate) values: &'static [&'static str],
}

/// Compact generated description of one block's cartesian state space.
///
/// `metadata` is in the same order as the cartesian product of `properties`, and contains the
/// exact metadata returned by Java's `Block.getMetaFromState` for each tuple.  Keeping this
/// compact table instead of 7,806 duplicated property records makes the client-world cache
/// cheap while retaining every model-distinguishing value.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DebugBlockSchema {
    pub(crate) block: u16,
    pub(crate) properties: &'static [DebugPropertySchema],
    pub(crate) metadata: &'static [u8],
}

/// A complete property tuple from Java's `BlockState.getValidStates()` DEBUG_WORLD list.
///
/// `stored` is only the packed metadata representative used by collision/light/render property
/// tables.  It is deliberately kept separate from `properties`: multiple valid Java tuples can
/// (and do) share one packed id for fences, wire arms, stair shapes, flower-pot contents, and
/// other actual-state-only properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugWorldState {
    stored: BlockState,
    properties: Vec<(&'static str, &'static str)>,
}

impl DebugWorldState {
    /// The packed metadata representative chosen by Java's `getMetaFromState`.
    #[must_use]
    pub const fn stored(&self) -> BlockState {
        self.stored
    }

    /// The complete, name-sorted property tuple used for model lookup.
    #[must_use]
    pub fn properties(&self) -> &[(&'static str, &'static str)] {
        &self.properties
    }

    /// Converts this immutable grid entry into the renderer's short-lived actual-state value.
    #[must_use]
    pub fn actual_state(&self) -> ActualBlockState {
        ActualBlockState {
            stored: self.stored,
            properties: self.properties.clone(),
        }
    }
}

impl ActualBlockState {
    /// Starts with the exact property tuple chunk storage resolves for `stored`.
    #[must_use]
    pub fn from_stored(stored: BlockState) -> Self {
        Self {
            stored,
            properties: state_properties(stored).to_vec(),
        }
    }

    /// The packed state this actual state was derived from.
    #[must_use]
    pub const fn stored(&self) -> BlockState {
        self.stored
    }

    /// The sorted property tuple used by vanilla's `BlockStateMapper`.
    #[must_use]
    pub fn properties(&self) -> &[(&'static str, &'static str)] {
        &self.properties
    }

    /// Reads a property by its registry name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&'static str> {
        self.properties
            .binary_search_by_key(&name, |(property, _)| *property)
            .ok()
            .map(|index| self.properties[index].1)
    }

    /// Replaces one property while preserving the registry's name-sorted order.
    ///
    /// `getActualState` only writes properties declared by the block. Returning `false` for
    /// an absent property makes a bad hand port visible in tests instead of manufacturing an
    /// impossible state that no blockstate JSON can name.
    pub fn set(&mut self, name: &'static str, value: &'static str) -> bool {
        let Ok(index) = self
            .properties
            .binary_search_by_key(&name, |(property, _)| *property)
        else {
            return false;
        };
        self.properties[index].1 = value;
        true
    }
}

/// The block id half of a packed state. Frozen S01 API.
#[inline]
#[must_use]
pub fn block_of(state: BlockState) -> BlockId {
    state.block()
}

/// The metadata half of a packed state. Frozen S01 API.
#[inline]
#[must_use]
pub fn meta_of(state: BlockState) -> u8 {
    state.meta()
}

/// Properties for a packed block state. Frozen S01 API.
///
/// `O(1)`, allocation-free, and **total**: any `u16` resolves, with unregistered states
/// returning the air row rather than panicking. S08 and S11 call this per block in hot
/// loops.
///
/// # Example
/// ```
/// use mc_data::{block_props, BlockState};
/// let stone = BlockState::from_raw(1 << 4);
/// assert_eq!(block_props(stone).light_opacity, 255);
/// assert!(block_props(BlockState::from_raw(0xFFFF)).is_air()); // garbage -> air
/// ```
#[inline]
#[must_use]
pub fn block_props(state: BlockState) -> &'static BlockProps {
    let i = state.id as usize;
    if i < STATE_PROPS.len() {
        &BLOCK_PROPS[STATE_PROPS[i] as usize]
    } else {
        &BLOCK_PROPS[0]
    }
}

/// Registry data for a block id, or the air entry when the id is unregistered.
#[inline]
#[must_use]
pub fn block_info(id: BlockId) -> &'static BlockInfo {
    id.info()
}

/// Looks up a block by registry name, with or without the `minecraft:` namespace.
///
/// This deliberately consults only the block registry.  Vanilla has sixteen names whose block
/// and item registry entries are both present but have different numeric ids (for example,
/// `bed` is block 26 and item 355).  Call [`crate::item_by_name`] when the Java call site is
/// resolving an `Item`; a registry name alone is not enough to choose the identity.
///
/// # Example
/// ```
/// use mc_data::block_by_name;
/// assert_eq!(block_by_name("stone").unwrap().0, 1);
/// assert_eq!(block_by_name("minecraft:stone").unwrap().0, 1);
/// assert!(block_by_name("mod:widget").is_none());
/// ```
#[must_use]
pub fn block_by_name(name: &str) -> Option<BlockId> {
    BLOCK_NAMES
        .binary_search_by(|(n, _)| cmp_registry_name(n, name))
        .ok()
        .map(|i| BlockId(BLOCK_NAMES[i].1))
}

/// `entry.cmp(key)`, where `key` is `name` when it contains a `:` and `minecraft:{name}`
/// otherwise, without formatting that string.
///
/// The registry lookups sit on per-tick paths (every collision query resolved
/// `block_by_name("stone")`, R2-16/PHYS-1), so the implicit namespace is compared in place:
/// Rust's `str` ordering is bytewise lexicographic, so comparing the `minecraft:` prefix first
/// and then the remainder is the same ordering as comparing the concatenation.
#[inline]
pub(crate) fn cmp_registry_name(entry: &str, name: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    const NAMESPACE: &[u8] = b"minecraft:";
    if name.contains(':') {
        return entry.cmp(name);
    }
    let entry = entry.as_bytes();
    let shared = entry.len().min(NAMESPACE.len());
    match entry[..shared].cmp(&NAMESPACE[..shared]) {
        Ordering::Equal if entry.len() < NAMESPACE.len() => Ordering::Less,
        Ordering::Equal => entry[NAMESPACE.len()..].cmp(name.as_bytes()),
        unequal => unequal,
    }
}

/// The blockstate variant key for a state, in vanilla's
/// `assets/minecraft/blockstates/*.json` form: `facing=east,half=bottom`, or `normal` for a
/// block with no properties.
///
/// Properties are sorted by name, matching 1.8.9's `BlockState` constructor and therefore
/// the keys in the shipped JSON. S09 uses this to bind a state to a baked model.
///
/// # Example
/// ```
/// use mc_data::{state_variant_key, BlockState};
/// assert_eq!(state_variant_key(BlockState::from_raw(1 << 4)), "variant=stone");
/// ```
#[inline]
#[must_use]
pub fn state_variant_key(state: BlockState) -> &'static str {
    variant_entry(state).0
}

/// The state's properties as `(name, value)` pairs, sorted by name. Empty for illegal
/// states and for blocks with no properties.
#[inline]
#[must_use]
pub fn state_properties(state: BlockState) -> &'static [(&'static str, &'static str)] {
    variant_entry(state).1
}

/// The blockstate variant key produced by Java's `Block.getStateById` decoder.
///
/// Unlike [`state_variant_key`], this accepts Java's `block_id | meta << 12` packet packing
/// and keeps virtual-decoder defaults intact. This matters for doors: lower-half metadata has
/// no hinge bit, so `BlockDoor.getStateFromMeta` returns `hinge=left`, even though the raw
/// chunk-storage representative for the same packed id happens to carry `hinge=right`.
#[inline]
#[must_use]
pub fn java_state_variant_key(id: i32) -> &'static str {
    java_variant_entry(id).0
}

/// Properties produced by Java's `Block.getStateById` virtual metadata decoder.
///
/// See [`java_state_variant_key`] for why this may differ from [`state_properties`] of the
/// [`BlockState`] returned by [`BlockState::from_java_state_id`].
#[inline]
#[must_use]
pub fn java_state_properties(id: i32) -> &'static [(&'static str, &'static str)] {
    java_variant_entry(id).1
}

#[inline]
fn variant_entry(
    state: BlockState,
) -> &'static (&'static str, &'static [(&'static str, &'static str)]) {
    let i = state.id as usize;
    if i < STATE_VARIANT_IDX.len() {
        &STATE_VARIANTS[STATE_VARIANT_IDX[i] as usize]
    } else {
        &STATE_VARIANTS[0]
    }
}

#[inline]
fn java_variant_entry(id: i32) -> &'static (&'static str, &'static [(&'static str, &'static str)]) {
    let block = (id & 4095) as u16;
    let meta = ((id >> 12) & 15) as u16;
    let index = ((block << 4) | meta) as usize;
    if index < JAVA_STATE_VARIANT_IDX.len() {
        assert!(
            JAVA_STATE_FROM_ID[index] != u16::MAX,
            "Block.getStateFromMeta rejected metadata"
        );
        &STATE_VARIANTS[JAVA_STATE_VARIANT_IDX[index] as usize]
    } else {
        &STATE_VARIANTS[0]
    }
}

/// Every legal packed state id, ascending.
///
/// The ascending packed-id order is also the order used by Java's
/// `ChunkProviderDebug.field_177464_a`: `Block.blockRegistry` iterates blocks by their numeric
/// registry id, and each generated block state's valid metadata tuples are emitted in ascending
/// metadata order. Keeping this as a named adapter makes the DEBUG_WORLD caller's registry-order
/// dependency explicit instead of having it quietly rebuild a different state list.
pub fn legal_states() -> impl Iterator<Item = BlockState> {
    (0..STATE_PROPS.len() as u32)
        .map(|i| BlockState::from_raw(i as u16))
        .filter(|s| s.is_legal())
}

/// The valid states in the exact registry order used by Java's DEBUG_WORLD grid.
///
/// Java's `ChunkProviderDebug` walks each registered block's `BlockState.getValidStates()` list,
/// which is a cartesian product of the sorted property descriptors.  A packed metadata id cannot
/// represent that list: several tuples intentionally share one metadata value.  The generated
/// schema therefore stores one metadata representative per tuple and this adapter expands the
/// property values without changing [`legal_states`] or any ordinary wire-state API.
pub fn debug_world_states() -> impl Iterator<Item = DebugWorldState> {
    let mut states = Vec::new();
    for schema in DEBUG_BLOCKS {
        let tuple_count = schema
            .properties
            .iter()
            .map(|property| property.values.len())
            .product::<usize>();
        debug_assert_eq!(
            tuple_count,
            schema.metadata.len(),
            "generated DEBUG_WORLD schema metadata length must equal its cartesian product"
        );
        for tuple_index in 0..tuple_count {
            let mut properties = Vec::with_capacity(schema.properties.len());
            for (property_index, property) in schema.properties.iter().enumerate() {
                // Guava's CartesianList makes the last axis the fastest-changing one.  This is
                // the order produced by BlockState's `Cartesian.cartesianProduct` call.
                let stride = schema.properties[property_index + 1..]
                    .iter()
                    .map(|axis| axis.values.len())
                    .product::<usize>();
                let value_index = (tuple_index / stride) % property.values.len();
                properties.push((property.name, property.values[value_index]));
            }
            let metadata = schema
                .metadata
                .get(tuple_index)
                .copied()
                .unwrap_or_default();
            states.push(DebugWorldState {
                stored: BlockState::new(BlockId(schema.block), metadata),
                properties,
            });
        }
    }
    states.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R2-16/PHYS-1: the allocation-free comparison orders every registry entry against every
    /// probe exactly as the former `format!("minecraft:{name}")` key did, so both lookups
    /// return the same ids (or `None`) for every name.
    #[test]
    fn registry_name_lookups_match_the_formatted_namespace_key() {
        let entries: Vec<&str> = BLOCK_NAMES
            .iter()
            .chain(crate::generated::items::ITEM_NAMES.iter())
            .map(|(name, _)| *name)
            .collect();
        let mut probes: Vec<String> = entries.iter().map(|name| (*name).to_owned()).collect();
        probes.extend(
            entries
                .iter()
                .map(|name| name.trim_start_matches("minecraft:").to_owned()),
        );
        probes.extend(
            [
                "",
                "a",
                "zzz",
                "Stone",
                "stone ",
                "minecraft",
                "minecraft:",
                "minecraft:zzz",
                "mod:widget",
                "minecraf",
                "minecraftt",
                ":",
                "1",
                "256",
                "-1",
                "~",
            ]
            .map(str::to_owned),
        );
        for probe in &probes {
            let key = if probe.contains(':') {
                probe.clone()
            } else {
                format!("minecraft:{probe}")
            };
            for entry in &entries {
                assert_eq!(
                    cmp_registry_name(entry, probe),
                    (*entry).cmp(key.as_str()),
                    "{entry:?} vs {probe:?}"
                );
            }
            let block = BLOCK_NAMES
                .binary_search_by(|(n, _)| (*n).cmp(key.as_str()))
                .ok()
                .map(|i| BlockId(BLOCK_NAMES[i].1));
            assert_eq!(block_by_name(probe), block, "block {probe:?}");
        }
        assert_eq!(block_by_name("stone"), Some(BlockId(1)));
        assert_eq!(
            crate::item_by_name("minecraft:bow"),
            crate::item_by_name("bow")
        );
    }

    #[test]
    fn packed_storage_state_fields_round_trip_through_their_bit_layout() {
        for block in 0u16..=255 {
            for meta in 0u8..16 {
                let s = BlockState::new(BlockId(block), meta);
                assert_eq!(s.block().0, block);
                assert_eq!(s.meta(), meta);
                assert_eq!(block_of(s).0, block);
                assert_eq!(meta_of(s), meta);
                assert_eq!(
                    s.to_java_state_id(),
                    i32::from(block) | (i32::from(meta) << 12)
                );
            }
        }
    }

    #[test]
    fn debug_world_states_are_property_rich_and_registry_ordered() {
        let states: Vec<_> = debug_world_states().collect();
        assert_eq!(states.len(), 7806);
        assert!(states.iter().all(|state| state.stored().is_legal()));
        assert!(states
            .windows(2)
            .all(|pair| pair[0].stored().block() <= pair[1].stored().block()));

        // `Block.blockRegistry` includes air before stone, so the first DEBUG_WORLD cell is air;
        // the following entries are the seven legal stone metadata tuples.  The tuple sidecar is
        // empty for air and names stone's variant exactly as BlockState does.
        assert_eq!(states[0].stored(), BlockState::AIR);
        assert_eq!(states[1].stored(), BlockState::new(BlockId(1), 0));
        assert_eq!(states[7].stored(), BlockState::new(BlockId(1), 6));
        assert_eq!(states[8].stored(), BlockState::new(BlockId(2), 0));
        assert!(states[0].properties().is_empty());
        assert_eq!(states[1].properties(), &[("variant", "stone")]);

        // Fence and stair tuples prove why the packed id cannot be used as the DEBUG_WORLD key:
        // all four connection bits / five corner shapes survive even when metadata collides.
        let fence = states
            .iter()
            .find(|state| state.stored().block() == BlockId(85))
            .expect("fence states");
        assert_eq!(fence.properties().len(), 4);
        assert_eq!(fence.stored(), BlockState::new(BlockId(85), 0));
        assert_eq!(
            fence.properties(),
            &[
                ("east", "true"),
                ("north", "true"),
                ("south", "true"),
                ("west", "true"),
            ]
        );
        assert_eq!(
            states[3411].properties(),
            &[
                ("facing", "north"),
                ("half", "top"),
                ("shape", "inner_left"),
            ]
        );
        assert_eq!(states[3411].stored(), BlockState::new(BlockId(53), 7));
        assert_eq!(states[3410].stored(), states[3411].stored());
        assert!(states.iter().any(|state| {
            state.stored().block() == BlockId(53)
                && state
                    .properties()
                    .iter()
                    .any(|(name, value)| *name == "shape" && *value != "straight")
        }));
        let unique_raw = states
            .iter()
            .map(|state| state.stored())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique_raw.len(), 1394);
    }

    #[test]
    fn default_state_uses_the_blocks_constructor_selected_metadata() {
        // java: `Block.getDefaultState()` is populated by each block constructor, not by an
        // implicit metadata-zero rule. These span the distinct non-zero default families.
        for (name, meta) in [
            ("torch", 5),
            ("oak_stairs", 3),
            ("wooden_door", 3),
            ("leaves", 8),
            ("brown_mushroom_block", 14),
            ("portal", 1),
        ] {
            let block = block_by_name(name).expect(name);
            let state = block.default_state();
            assert_eq!(state, BlockState::new(block, meta), "{name}");
            assert!(
                state.is_legal(),
                "{name}'s default must be a registered state"
            );
        }

        // Negative control: meta zero is not a legal torch storage state, so the old hardcoded
        // implementation silently returned the air sentinel through `BlockState::props()`.
        let torch = block_by_name("torch").unwrap();
        assert!(!BlockState::new(torch, 0).is_legal());
        assert_eq!(torch.default_state().props().block, torch);

        // The extractor stores constructor metadata for every registered block. This catches a
        // missing generated field without merely pinning a few hand-selected examples.
        for id in 0..=MAX_BLOCK_ID {
            let block = BlockId(id);
            if block.is_registered() {
                assert!(block.default_state().is_legal(), "registered block id {id}");
            }
        }
    }

    #[test]
    fn break_sound_is_the_virtual_soundtype_result_not_a_step_family_guess() {
        // `SoundType.getBreakSound()` has four overrides in Block.java.  Glass is the decisive
        // negative control: it shares the `stone` step family but must not break as stone.
        for (name, expected) in [
            ("stone", "dig.stone"),
            ("glass", "dig.glass"),
            ("ladder", "dig.wood"),
            ("anvil", "dig.stone"),
            ("slime", "mob.slime.big"),
        ] {
            let block = block_by_name(name).expect(name);
            assert_eq!(block.info().break_sound, expected, "{name}");
        }
        let stone = block_by_name("stone").unwrap();
        let glass = block_by_name("glass").unwrap();
        assert_eq!(stone.info().step_sound, glass.info().step_sound);
        assert_ne!(stone.info().break_sound, glass.info().break_sound);

        // Every registered entry came from the virtual extractor call; gaps retain the exact
        // air sentinel rather than fabricating a per-id event.
        for id in 0..=MAX_BLOCK_ID {
            let info = BlockId(id).info();
            if info.registered {
                assert!(!info.break_sound.is_empty(), "registered block {id}");
            }
        }
        assert_eq!(
            BlockId(MAX_BLOCK_ID + 1).info().break_sound,
            BlockId::AIR.info().break_sound
        );
    }

    #[test]
    fn java_state_id_uses_each_block_virtual_metadata_decoder() {
        // BlockStone.EnumType.byMetadata clamps 7..15 to its meta-zero variant. A raw
        // repack would produce the illegal storage state `(stone, 15)`, which then looks like
        // air to mc-data's chunk-table lookup.
        let stone = block_by_name("stone").unwrap();
        let normalized = BlockState::from_java_state_id(i32::from(stone.0) | (15 << 12));
        assert_eq!(normalized, BlockState::new(stone, 0));
        assert!(normalized.is_legal());

        // BlockTorch maps the impossible down-facing meta zero to its canonical up-facing
        // state (storage meta 5), not to air or raw meta zero.
        let torch = block_by_name("torch").unwrap();
        assert_eq!(
            BlockState::from_java_state_id(i32::from(torch.0)),
            BlockState::new(torch, 5)
        );

        // This also has to be the *virtual* decoder for values that happen to be legal in
        // chunk storage. The registry table has all flower-pot metas, but its values describe
        // contents while Java's BlockFlowerPot.getStateFromMeta supplies its empty canonical
        // state (the tile entity carries the actual contents).
        let flower_pot = block_by_name("flower_pot").unwrap();
        let raw = BlockState::new(flower_pot, 1);
        assert!(raw.is_legal());
        let decoded = BlockState::from_java_state_id(i32::from(flower_pot.0) | (1 << 12));
        assert_ne!(decoded, raw);
        assert_eq!(decoded, BlockState::new(flower_pot, 0));
    }

    #[test]
    fn java_state_id_preserves_every_door_half_metadata_tuple() {
        // `BlockDoor.getStateFromMeta` is deliberately asymmetric. Lower metadata contains
        // facing/open and inherits right-hinge/unpowered defaults; upper metadata contains
        // hinge/powered and inherits north/closed defaults. Bits 2..0 are the only upper-half
        // payload, so 12..15 normalize to 8..11. Exercise the virtual decoder and the generated
        // variant tuple together for every one of the seven 1.8.9 door implementations.
        for name in [
            "wooden_door",
            "iron_door",
            "spruce_door",
            "birch_door",
            "jungle_door",
            "acacia_door",
            "dark_oak_door",
        ] {
            let door = block_by_name(name).unwrap();
            for meta in 0u8..16 {
                let decoded =
                    BlockState::from_java_state_id(i32::from(door.0) | (i32::from(meta) << 12));
                let (canonical_meta, expected) = if meta & 8 == 0 {
                    let facing = ["east", "south", "west", "north"][(meta & 3) as usize];
                    (
                        meta,
                        format!(
                            "facing={facing},half=lower,hinge=left,open={},powered=false",
                            meta & 4 != 0
                        ),
                    )
                } else {
                    let canonical = 8 | (meta & 3);
                    (
                        canonical,
                        format!(
                            "facing=north,half=upper,hinge={},open=false,powered={}",
                            if meta & 1 != 0 { "right" } else { "left" },
                            meta & 2 != 0
                        ),
                    )
                };
                assert_eq!(
                    decoded,
                    BlockState::new(door, canonical_meta),
                    "{name} meta {meta}"
                );
                assert_eq!(
                    java_state_variant_key(i32::from(door.0) | (i32::from(meta) << 12)),
                    expected,
                    "{name} meta {meta}"
                );
            }
        }

        // The raw chunk-storage lookup deliberately remains a different table. A door cell
        // received from an S22/S23 block-state packet preserves that representative; only
        // `Block.getStateById` callers need the virtual decoder tuple above.
        let door = block_by_name("wooden_door").unwrap();
        assert_eq!(
            state_variant_key(BlockState::new(door, 0)),
            "facing=east,half=lower,hinge=right,open=false,powered=false"
        );
        assert_eq!(
            java_state_variant_key(i32::from(door.0)),
            "facing=east,half=lower,hinge=left,open=false,powered=false"
        );
    }

    #[test]
    fn java_state_id_propagates_virtual_decoder_failures_and_defaults_unknown_blocks_to_air() {
        // Powered rails accept only six direction values. Their Java getStateFromMeta tries to
        // install `south_east` for input six, and StateImplementation rejects that property;
        // Block.getStateById propagates rather than treating it as a metadata-zero fallback.
        let golden_rail = block_by_name("golden_rail").unwrap();
        assert!(std::panic::catch_unwind(|| {
            BlockState::from_java_state_id(i32::from(golden_rail.0) | (6 << 12))
        })
        .is_err());

        // getBlockById uses Blocks.air as its default registry entry, even with signed/high
        // bits in the packet word. The upper bits do not participate in either Java field.
        assert_eq!(BlockState::from_java_state_id(-1), BlockState::AIR);
        assert_eq!(BlockState::from_java_state_id(1 << 20), BlockState::AIR);
    }

    /// The non-panicking form is `None` exactly where the panicking one panics
    /// (Java's `getStateFromMeta` throws) and identical everywhere else, over every value of the
    /// packet word's participating bits.
    #[test]
    fn try_from_java_state_id_is_none_exactly_where_get_state_from_meta_throws() {
        let golden_rail = block_by_name("golden_rail").unwrap();
        let rejected_id = i32::from(golden_rail.0) | (6 << 12);
        assert_eq!(BlockState::try_from_java_state_id(rejected_id), None);
        assert!(std::panic::catch_unwind(|| BlockState::from_java_state_id(rejected_id)).is_err());
        let mut rejected = 0;
        for id in 0..(1 << 16) {
            match BlockState::try_from_java_state_id(id) {
                Some(state) => assert_eq!(state, BlockState::from_java_state_id(id), "{id:#x}"),
                None => rejected += 1,
            }
        }
        assert!(rejected > 0);
        assert_eq!(
            BlockState::try_from_java_state_id(-1),
            Some(BlockState::AIR)
        );
    }

    #[test]
    fn the_two_shared_bounds_blocks_do_not_key_their_outline_on_metadata() {
        // `Block.minX..maxZ` is a mutable field on the block singleton and
        // `Block.setBlockBoundsBasedOnState` is an empty body, so for the two 1.8.9 classes that
        // write those fields from `addCollisionBoxesToList` without overriding the former —
        // `BlockBrewingStand` (117) and `BlockEndPortalFrame` (120) — the raytrace/outline box
        // is *not a function of the state*. The extractor used to read the fields cold, which
        // let state n inherit state n-1's leftover and produced meta 0 = unit cube with all the
        // other metas = the real shape. Metadata is not what vanilla keys this on, and that
        // split changed which `BlockPos` a click landed on for an empty brewing stand — meta 0,
        // by far the most common one.
        //
        // The extractor now latches the singleton first, so both blocks report their steady
        // state for every meta. See the comment at `McDataExtractor.emitState`.
        for (name, want, metas) in [
            (
                "brewing_stand",
                Aabb::new(0.0, 0.0, 0.0, 1.0, 0.125, 1.0),
                0u8..8,
            ),
            (
                "end_portal_frame",
                Aabb::new(0.0, 0.0, 0.0, 1.0, 0.8125, 1.0),
                0u8..8,
            ),
        ] {
            let id = block_by_name(name).unwrap();
            for meta in metas {
                let state = BlockState::new(id, meta);
                assert!(state.is_legal(), "{name} meta {meta}");
                assert_eq!(
                    state.props().bounds,
                    want,
                    "{name} meta {meta} must not differ from its siblings"
                );
            }
        }
    }

    #[test]
    fn the_mobility_flag_is_the_blocks_own_virtual_not_its_materials() {
        // java: `Block.getMobilityFlag()` returns `blockMaterial.getMaterialMobility()` unless
        // one of five classes overrides it. `BlockPistonBase.canPush` and
        // `BlockPistonStructureHelper` read the virtual, so the override is what matters:
        // a bed is `Material.wood` (mobility 0) but `BlockBed.getMobilityFlag()` is 1, and ice
        // is `Material.ice` (mobility 1) but `BlockIce.getMobilityFlag()` is 0.
        let mob = |name: &str| block_info(block_by_name(name).unwrap()).mobility_flag;
        assert_eq!(mob("bed"), 1, "BlockBed.java:232 overrides wood's 0");
        assert_eq!(
            mob("rail"),
            0,
            "BlockRailBase.java:156 overrides circuits' 1"
        );
        assert_eq!(mob("ice"), 0, "BlockIce.java:90 overrides ice's 1");
        assert_eq!(mob("stone_pressure_plate"), 1);
        assert_eq!(mob("wooden_door"), 1);
        // and the un-overridden majority comes straight off the material
        assert_eq!(mob("stone"), 0);
        assert_eq!(mob("anvil"), 2, "Material.anvil");
        assert_eq!(mob("piston"), 2, "Material.piston");
        assert_eq!(mob("torch"), 1, "Material.circuits");
    }

    #[test]
    fn known_blocks_have_the_right_shape() {
        let stone = BlockState::new(BlockId(1), 0);
        assert!(stone.is_legal());
        assert_eq!(stone.props().collision, CollisionShape::Full);
        assert_eq!(stone.props().light_opacity, 255);
        assert_eq!(stone.props().material, Material::Rock);
        assert_eq!(stone.props().render_layer, RenderLayer::Solid);

        let air = BlockState::AIR;
        assert!(air.props().is_air());
        assert!(air.props().collision.is_empty());

        // Bedrock is unbreakable.
        let bedrock = BlockState::new(block_by_name("bedrock").unwrap(), 0);
        assert!(bedrock.props().is_unbreakable());

        // Ice is slippery, stone is not.
        let ice = BlockState::new(block_by_name("ice").unwrap(), 0);
        assert!(ice.props().slipperiness > stone.props().slipperiness);
    }

    #[test]
    fn stairs_have_two_boxes_and_depend_on_neighbours() {
        let stairs = BlockState::new(block_by_name("oak_stairs").unwrap(), 0);
        assert_eq!(stairs.props().collision.boxes().len(), 2);
        assert!(stairs.props().collision_depends_on_neighbors);
    }

    /// Regression guard for the extractor's neighbour probe. A uniform neighbourhood is
    /// not enough to detect these: a glass pane collides identically with zero and with
    /// four connections, and a stair corner needs a *perpendicular* stair.
    #[test]
    fn neighbour_dependent_set_is_the_expected_33_blocks() {
        let mut names: Vec<&str> = legal_states()
            .filter(|s| s.props().collision_depends_on_neighbors)
            .map(|s| s.props().block.info().name)
            .collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 33, "{names:?}");
        for expect in [
            "minecraft:fence",
            "minecraft:cobblestone_wall",
            "minecraft:glass_pane",
            "minecraft:iron_bars",
            "minecraft:oak_stairs",
            "minecraft:wooden_door",
            "minecraft:chest",
        ] {
            assert!(names.contains(&expect), "{expect} missing from {names:?}");
        }
        // All six fence-gate classes expose `in_wall` in their actual state for model
        // selection. Java's BlockFenceGate.getCollisionBoundingBox instead reads only OPEN
        // and FACING, and setBlockBoundsBasedOnState reads only FACING. Thus a wall neighbour
        // is emphatically not a collision-dependency probe. Check every stored meta too:
        // bit 3 (POWERED) must likewise be irrelevant, while bit 2 (OPEN) removes the box.
        for gate in [
            "fence_gate",
            "spruce_fence_gate",
            "birch_fence_gate",
            "jungle_fence_gate",
            "dark_oak_fence_gate",
            "acacia_fence_gate",
        ] {
            let id = block_by_name(gate).unwrap();
            assert!(
                !names.contains(&id.info().name),
                "{gate} must not be a neighbour-dependent collision row"
            );
            for meta in 0u8..16 {
                let state = BlockState::new(id, meta);
                assert!(state.is_legal(), "{gate} meta {meta}");
                assert!(
                    !state.props().collision_depends_on_neighbors,
                    "{gate} meta {meta}"
                );
                let boxes = state.props().collision.boxes();
                if meta & 4 != 0 {
                    assert!(boxes.is_empty(), "{gate} open meta {meta}");
                    continue;
                }
                assert_eq!(boxes.len(), 1, "{gate} closed meta {meta}");
                let box_ = boxes[0];
                let (min_x, min_z, max_x, max_z) = if meta & 1 == 0 {
                    (0.0, 0.375, 1.0, 0.625)
                } else {
                    (0.375, 0.0, 0.625, 1.0)
                };
                assert_eq!(box_, Aabb::new(min_x, 0.0, min_z, max_x, 1.5, max_z));
            }
        }
    }

    #[test]
    fn fence_is_one_and_a_half_blocks_tall() {
        let fence = BlockState::new(block_by_name("fence").unwrap(), 0);
        let b = fence.props().collision.boxes();
        assert_eq!(b.len(), 1);
        assert!(
            (b[0].max_y - 1.5).abs() < 1e-9,
            "fence post height {}",
            b[0].max_y
        );
    }

    #[test]
    fn illegal_states_resolve_to_air_and_never_panic() {
        // Torch meta 0 is not a registered state in 1.8.9 (facing starts at 1).
        let torch = block_by_name("torch").unwrap();
        let bad = BlockState::new(torch, 0);
        assert!(!bad.is_legal());
        assert!(bad.props().is_air());
    }

    #[test]
    fn variant_keys_match_vanilla_blockstate_json() {
        assert_eq!(
            state_variant_key(BlockState::new(BlockId(1), 0)),
            "variant=stone"
        );
        assert_eq!(state_variant_key(BlockState::AIR), "normal");
        let stairs = BlockState::new(block_by_name("oak_stairs").unwrap(), 0);
        let key = state_variant_key(stairs);
        assert!(key.contains("facing="), "{key}");
        assert!(key.contains("half="), "{key}");
        // sorted by property name, as vanilla's BlockState constructor guarantees
        let names: Vec<_> = state_properties(stairs).iter().map(|(k, _)| *k).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn actual_state_changes_a_derived_property_without_changing_the_wire_state() {
        let stored = BlockState::new(block_by_name("fence").unwrap(), 0);
        let mut actual = ActualBlockState::from_stored(stored);
        assert_eq!(actual.get("east"), Some("false"));
        assert!(actual.set("east", "true"));
        assert_eq!(actual.get("east"), Some("true"));
        assert_eq!(
            actual.stored(),
            stored,
            "the packed chunk/wire state is invariant"
        );
        assert!(!actual.set("not_a_property", "true"));
    }

    /// **F-P1-block.state-01-3.** Typed state operations mirror `IProperty<T>` instead of
    /// exposing only the generated strings: reads parse a descriptor, replacement preserves the
    /// other tuple members, and cycling follows the descriptor order with wraparound.
    #[test]
    fn typed_state_properties_read_replace_cycle_and_reject_invalid_values() {
        const OPEN: Property<bool> = Property::boolean("open");
        const FACING: Property<u8> = Property::new(
            "facing",
            &[("north", 0), ("south", 1), ("west", 2), ("east", 3)],
        );
        const AGE: Property<u8> = Property::new(
            "age",
            &[
                ("0", 0),
                ("1", 1),
                ("2", 2),
                ("3", 3),
                ("4", 4),
                ("5", 5),
                ("6", 6),
                ("7", 7),
            ],
        );

        let closed = BlockState::new(block_by_name("trapdoor").unwrap(), 0);
        assert_eq!(closed.get_value(OPEN), Some(false));
        assert_eq!(
            closed.get_value(OPEN),
            Some(false),
            "borrowed descriptor is equivalent"
        );
        let opened = closed
            .with_property(OPEN, true)
            .expect("trapdoor declares the open property");
        assert_eq!(opened.get_value(OPEN), Some(true));
        assert_eq!(opened.get_value(FACING), closed.get_value(FACING));
        assert_ne!(opened, closed, "with_property cannot be a no-op");
        assert_eq!(
            opened.cycle_property(OPEN),
            Some(closed),
            "boolean cycle wraps"
        );

        let wheat = BlockState::new(block_by_name("wheat").unwrap(), 3);
        assert_eq!(wheat.get_value(AGE), Some(3));
        assert_eq!(wheat.cycle_property(AGE).unwrap().get_value(AGE), Some(4));
        let mature = BlockState::new(block_by_name("wheat").unwrap(), 7);
        assert_eq!(mature.cycle_property(AGE).unwrap().get_value(AGE), Some(0));

        let invalid_age = Property::new("age", &[("0", 0u8), ("1", 1u8)]);
        assert_eq!(wheat.with_property(invalid_age, 8), None);
        assert_eq!(wheat.get_value(Property::boolean("powered")), None);
        assert_eq!(wheat.cycle_property(Property::boolean("powered")), None);
    }

    #[test]
    fn property_bool_allowed_values_follow_java_true_then_false_order() {
        const OPEN: Property<bool> = Property::boolean("open");
        assert_eq!(OPEN.values(), &[("true", true), ("false", false)]);
        assert_eq!(OPEN.parse_value("true"), Some(true));
        assert_eq!(OPEN.parse_value("false"), Some(false));
    }

    #[test]
    fn property_direction_matches_java_order_lookup_and_filters() {
        let all = PropertyDirection::create("facing");
        let names: Vec<_> = all.values().iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["down", "up", "north", "south", "west", "east"]);
        assert_eq!(all.get_name(EnumFacing::North), "north");
        assert_eq!(all.parse_value("north"), Some(EnumFacing::North));
        assert_eq!(
            all.parse_value("NORTH"),
            None,
            "PropertyEnum lookup is exact"
        );
        assert_eq!(EnumFacing::by_name("NORTH"), Some(EnumFacing::North));
        assert_eq!(EnumFacing::by_name("sideways"), None);

        let horizontal = PropertyDirection::create_with_predicate("facing", |facing| {
            matches!(
                facing,
                EnumFacing::North | EnumFacing::South | EnumFacing::West | EnumFacing::East
            )
        });
        let names: Vec<_> = horizontal.values().iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["north", "south", "west", "east"]);

        let custom = PropertyDirection::create_from_values(
            "facing",
            [EnumFacing::East, EnumFacing::North, EnumFacing::East],
        );
        assert_eq!(
            custom.values(),
            &[("east", EnumFacing::East), ("north", EnumFacing::North),]
        );
    }

    #[test]
    fn property_direction_cycle_uses_filtered_declaration_order() {
        let property = PropertyDirection::create_filtered("facing", |facing| {
            !matches!(facing, EnumFacing::Down)
        });
        let torch = BlockState::new(block_by_name("torch").unwrap(), 5);
        assert_eq!(torch.get_value(property), Some(EnumFacing::Up));
        let next = torch
            .cycle_property(property)
            .expect("torch has the next facing");
        assert_eq!(next.get_value(property), Some(EnumFacing::North));

        // `state_variant_key` consumes the same lower-case names as the model mapper; adding a
        // typed descriptor must not alter the generated registry serialization.
        assert_eq!(state_variant_key(next), "facing=north");
    }

    #[test]
    fn name_lookup() {
        assert_eq!(block_by_name("air"), Some(BlockId::AIR));
        assert_eq!(
            block_by_name("minecraft:diamond_block").map(|b| b.0),
            Some(57)
        );
        assert_eq!(block_by_name("not_a_block"), None);
        assert_eq!(block_by_name("othermod:stone"), None);
    }

    #[test]
    fn virtual_block_localized_names_are_extracted_not_raw_language_keys() {
        // These classes override Block.getLocalizedName().  Calling the base translation
        // expression while extracting loses the override and leaves `tile.*.name` visible.
        for (id, expected) in [
            (64, "Oak Door"),
            (182, "Red Sandstone Slab"),
            (93, "Redstone Repeater"),
            (94, "Redstone Repeater"),
            (149, "Redstone Comparator"),
            (150, "Redstone Comparator"),
            (193, "Spruce Door"),
            (197, "Dark Oak Door"),
        ] {
            assert_eq!(BlockId(id).info().display_name, expected, "block id {id}");
        }
    }
}
