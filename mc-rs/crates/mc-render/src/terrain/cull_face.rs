//! `Block.shouldSideBeRendered` — the neighbour-face cull.
//!
//! This is one virtual method with nineteen overrides in 1.8.9, and it decides whether a
//! face exists at all. The base implementation is bounds-driven: a face that does not reach
//! the edge of its block is always drawn, and a face that does is drawn only when the
//! neighbour is not an opaque cube. The overrides are what make glass panes not z-fight,
//! adjacent stained glass of different colours keep their seam, and fences render their
//! insides.
//!
//! Every override is keyed on `mc_data`'s recorded originating Java class, since the class
//! hierarchy itself does not survive the port.
//
// java: block/Block.java:468, and the overrides in BlockLiquid, BlockBreakable, BlockPane,
//       BlockLeavesBase, BlockSlab, BlockSnow, BlockCarpet, BlockFarmland, BlockFence,
//       BlockFenceGate, BlockWall, BlockHopper, BlockRedstoneDiode, BlockAnvil,
//       BlockDragonEgg, BlockEndPortal, BlockPortal, BlockPistonExtension.

use mc_data::{block_props, BlockState};
use mc_math::{BlockPos, Facing};
use mc_world::World;

/// Graphics settings that change culling. java: `Block.fancyGraphics` (a static set from
/// `GameSettings.fancyGraphics`) and `Minecraft.isAmbientOcclusionEnabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshSettings {
    /// Fancy graphics: leaves become see-through and stop culling against each other.
    pub fancy_graphics: bool,
    /// Smooth lighting. `false` takes vanilla's flat `renderModelStandard` path.
    pub ambient_occlusion: bool,
    /// The Connected Textures mod. [`super::CtmMode::Off`] is vanilla
    /// meshing; Fast/Fancy apply the installed bake's CTM table, if it has one.
    pub connected_textures: super::ctm::CtmMode,
    /// The Tall Grass mod turned **off**: grass
    /// and ferns, one and two blocks tall ([`super::tall_grass`]), are not meshed. `false` is
    /// vanilla.
    pub hide_tall_grass: bool,
    /// What a running shader pack needs from the mesher: no vanilla per-face shade, material flags. [`PackMesh::VANILLA`] (the default)
    /// is vanilla meshing, byte for byte.
    ///
    /// [`PackMesh::VANILLA`]: crate::shaderpack::PackMesh::VANILLA
    pub pack: crate::shaderpack::PackMesh,
}

impl Default for MeshSettings {
    /// Vanilla's defaults: fancy graphics on, smooth lighting on, no connected textures, grass
    /// shown, no shader-pack meshing.
    fn default() -> Self {
        Self {
            fancy_graphics: true,
            ambient_occlusion: true,
            connected_textures: super::ctm::CtmMode::Off,
            hide_tall_grass: false,
            pack: crate::shaderpack::PackMesh::VANILLA,
        }
    }
}

/// The short Java class name `mc_data` recorded for a state's block.
#[must_use]
pub fn class_of(state: BlockState) -> &'static str {
    let class = mc_data::block_info(state.block()).java_class;
    class.rsplit('.').next().unwrap_or(class)
}

/// Which `shouldSideBeRendered` override a block uses: the arms of
/// [`should_side_be_rendered`]'s former string `match`, decided once per block id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CullClass {
    /// `BlockStaticLiquid`, `BlockDynamicLiquid`.
    Liquid,
    /// `BlockGlass`, `BlockStainedGlass`.
    Glass,
    /// `BlockIce`, `BlockSlime`.
    IceOrSlime,
    /// `BlockPane`, `BlockStainedGlassPane`.
    Pane,
    /// `BlockOldLeaf`, `BlockNewLeaf`.
    Leaves,
    /// The three single slabs (`BlockSlab.isSlab`).
    HalfSlab,
    /// The three double slabs.
    DoubleSlab,
    /// `BlockSnow`, `BlockCarpet`.
    SnowOrCarpet,
    /// `BlockFarmland`.
    Farmland,
    /// Fence, fence gate, hopper, anvil, dragon egg, piston extension: always drawn.
    AlwaysDrawn,
    /// `BlockWall`.
    Wall,
    /// `BlockRedstoneRepeater`, `BlockRedstoneComparator`.
    Diode,
    /// `BlockEndPortal`.
    EndPortal,
    /// `BlockPortal`.
    Portal,
    /// No override: [`base`].
    Base,
}

/// Per-block facts the hot mesher paths used to recompute from the Java class *string* on every
/// block and face (`rsplit('.')` plus a `match` of up to ~30 literals, about 40 times per block). They are pure functions of the block id, so they are decided once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockClassInfo {
    /// The `shouldSideBeRendered` override.
    pub(crate) cull: CullClass,
    /// `mesher::actual_state` has an arm for this class; when clear, the actual state is the
    /// stored state's own property tuple and the dense `models[state.id]` table is exact.
    pub(crate) actual_state: bool,
    /// `getOffsetType() == XYZ` (`BlockTallGrass`).
    pub(crate) offset_xyz: bool,
    /// `getOffsetType() == XZ` (the two flowers and `BlockDoublePlant`).
    pub(crate) offset_xz: bool,
    /// `BlockCactus`: full render bounds but an inset selection box.
    pub(crate) cactus: bool,
    /// `selected_bounding_box` takes a world-dependent or special branch for this class
    /// (door, fence, wall, pane, chest, vine, cactus) instead of the static block bounds.
    pub(crate) special_selection: bool,
}

/// The classes `mesher::actual_state` rewrites. **Keep in step with that `match`**;
/// `blocks_without_an_actual_state_arm_actualise_to_their_stored_tuple` guards it.
pub(crate) const ACTUAL_STATE_CLASSES: [&str; 20] = [
    "BlockBed",
    "BlockDirt",
    "BlockGrass",
    "BlockMycelium",
    "BlockDoor",
    "BlockDoublePlant",
    "BlockFence",
    "BlockFenceGate",
    "BlockFire",
    "BlockFlowerPot",
    "BlockPane",
    "BlockStainedGlassPane",
    "BlockRedstoneRepeater",
    "BlockRedstoneWire",
    "BlockStairs",
    "BlockStem",
    "BlockTripWire",
    "BlockTripWireHook",
    "BlockVine",
    "BlockWall",
];

fn classify(id: u16) -> BlockClassInfo {
    let class = mc_data::block_info(mc_data::BlockId(id)).java_class;
    let short = class.rsplit('.').next().unwrap_or(class);
    let cull = match short {
        "BlockStaticLiquid" | "BlockDynamicLiquid" => CullClass::Liquid,
        "BlockGlass" | "BlockStainedGlass" => CullClass::Glass,
        "BlockIce" | "BlockSlime" => CullClass::IceOrSlime,
        "BlockPane" | "BlockStainedGlassPane" => CullClass::Pane,
        "BlockOldLeaf" | "BlockNewLeaf" => CullClass::Leaves,
        "BlockHalfStoneSlab" | "BlockHalfStoneSlabNew" | "BlockHalfWoodSlab" => CullClass::HalfSlab,
        "BlockDoubleStoneSlab" | "BlockDoubleStoneSlabNew" | "BlockDoubleWoodSlab" => {
            CullClass::DoubleSlab
        }
        "BlockSnow" | "BlockCarpet" => CullClass::SnowOrCarpet,
        "BlockFarmland" => CullClass::Farmland,
        "BlockFence"
        | "BlockFenceGate"
        | "BlockHopper"
        | "BlockAnvil"
        | "BlockDragonEgg"
        | "BlockPistonExtension" => CullClass::AlwaysDrawn,
        "BlockWall" => CullClass::Wall,
        "BlockRedstoneRepeater" | "BlockRedstoneComparator" => CullClass::Diode,
        "BlockEndPortal" => CullClass::EndPortal,
        "BlockPortal" => CullClass::Portal,
        _ => CullClass::Base,
    };
    // Exactly the predicates `selected_bounding_box` and `static_selected_bounding_box` test.
    let special_selection = class.ends_with("BlockDoor")
        || class.ends_with("BlockCactus")
        || matches!(
            short,
            "BlockFence"
                | "BlockWall"
                | "BlockPane"
                | "BlockStainedGlassPane"
                | "BlockChest"
                | "BlockVine"
        );
    BlockClassInfo {
        cull,
        actual_state: ACTUAL_STATE_CLASSES.contains(&short),
        offset_xyz: short == "BlockTallGrass",
        offset_xz: matches!(
            short,
            "BlockYellowFlower" | "BlockRedFlower" | "BlockDoublePlant"
        ),
        cactus: short == "BlockCactus",
        special_selection,
    }
}

/// [`BlockClassInfo`] for a state's block: one table load after the first call.
#[inline]
#[must_use]
pub(crate) fn class_info(state: BlockState) -> BlockClassInfo {
    // `BlockState::block` is `id >> 4` of a `u16`, so 4096 entries cover every possible input,
    // including ids past the registry (which `block_info` maps to air, as `classify` does).
    static TABLE: std::sync::OnceLock<Vec<BlockClassInfo>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..4096u16).map(classify).collect())[usize::from(state.block().0)]
}

/// `Block.isOpaqueCube()`, with the one override that is not a constant.
///
/// `BlockLeaves.isOpaqueCube()` returns `!Blocks.leaves.fancyGraphics`, so it flips with a
/// graphics setting rather than being a property of the block. `mc_data` extracted it as a
/// constant (`true`, the fast-graphics value), so the fancy case has to be reapplied here —
/// otherwise a tree canopy culls its own interior and reads as a solid block to the visibility
/// graph.
#[must_use]
pub fn is_opaque_cube(state: BlockState, settings: MeshSettings) -> bool {
    if settings.fancy_graphics && class_info(state).cull == CullClass::Leaves {
        return false;
    }
    block_props(state).opaque_cube
}

/// `Block.getBlockLayer()`, with the one override that is not a constant.
///
/// `BlockLeaves.getBlockLayer()` returns `isTransparent ? CUTOUT_MIPPED : SOLID`, and
/// `isTransparent` is set from `fancyGraphics` — the same runtime flag [`is_opaque_cube`]
/// deals with. `mc_data` recorded the fast-graphics value (`SOLID`).
#[must_use]
pub fn render_layer_of(state: BlockState, settings: MeshSettings) -> mc_data::RenderLayer {
    if settings.fancy_graphics && class_info(state).cull == CullClass::Leaves {
        return mc_data::RenderLayer::CutoutMipped;
    }
    block_props(state).render_layer
}

/// java: `Block.shouldSideBeRendered` — the base, bounds-driven implementation.
///
/// `pos` is the **neighbour** position (vanilla passes `blockPos.offset(side)`), and the
/// bounds are the *rendered* block's.
///
/// `state_is_live` is the caller's promise that `world.block_at(rendered position) == state`
/// (the mesher's case). It lets a block whose class takes none of `selected_bounding_box`'s
/// special branches skip that function's live re-read of the same block (a hashed chunk lookup
/// per face, MESH-2) and its class-string tests; the bounds it computes are the ones
/// `static_selected_bounding_box` computes, with the same `f64` operations.
fn base(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    neighbour_state: BlockState,
    side: Facing,
    settings: MeshSettings,
    state_is_live: bool,
) -> bool {
    let rendered_pos = pos.relative(side.opposite());
    let info = class_info(state);
    // java: `Block.shouldSideBeRendered` compares against the mutable render bounds, not
    // `getSelectedBoundingBox`.  Cactus is the only block in this port whose selection box
    // differs from those bounds: its outline/collision column is inset, while the constructor
    // leaves its render bounds as the full unit cube.  Keep the selection path world-aware for
    // every other block (doors, fences, panes, etc.), and skip only cactus's inset comparisons.
    let b = if state_is_live && !info.special_selection {
        let origin = mc_math::DVec3::new(
            f64::from(rendered_pos.x),
            f64::from(rendered_pos.y),
            f64::from(rendered_pos.z),
        );
        let bounds = block_props(state).bounds;
        mc_math::Aabb::new(
            origin + mc_math::DVec3::new(bounds.min_x, bounds.min_y, bounds.min_z),
            origin + mc_math::DVec3::new(bounds.max_x, bounds.max_y, bounds.max_z),
        )
    } else {
        crate::terrain::selected_bounding_box(world, rendered_pos, state)
    };
    let cactus_uses_full_render_bounds = info.cactus;
    match side {
        Facing::Down if !cactus_uses_full_render_bounds && b.min.y > f64::from(rendered_pos.y) => {
            true
        }
        Facing::Up
            if !cactus_uses_full_render_bounds && b.max.y < f64::from(rendered_pos.y + 1) =>
        {
            true
        }
        Facing::North if !cactus_uses_full_render_bounds && b.min.z > f64::from(rendered_pos.z) => {
            true
        }
        Facing::South
            if !cactus_uses_full_render_bounds && b.max.z < f64::from(rendered_pos.z + 1) =>
        {
            true
        }
        Facing::West if !cactus_uses_full_render_bounds && b.min.x > f64::from(rendered_pos.x) => {
            true
        }
        Facing::East
            if !cactus_uses_full_render_bounds && b.max.x < f64::from(rendered_pos.x + 1) =>
        {
            true
        }
        // `neighbour_state` is the caller's `world.block_at(pos)`, read once per face.
        _ => !is_opaque_cube(neighbour_state, settings),
    }
}

/// Whether the face of `state` at `pos` pointing along `side` should be drawn.
///
/// `pos` is the position of the block being rendered; the neighbour is derived here (unlike
/// vanilla, which makes every call site do `pos.offset(side)` and is easy to get wrong).
#[must_use]
pub fn should_side_be_rendered(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    side: Facing,
    settings: MeshSettings,
) -> bool {
    side_rendered(world, state, pos, side, settings, false)
}

/// [`should_side_be_rendered`] for a caller whose `state` **is** `world.block_at(pos)` — the
/// mesher, which read it from the world itself. Same answer; it lets [`base`] skip re-reading
/// the block for its bounds.
#[must_use]
pub(crate) fn should_side_be_rendered_for_world_block(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    side: Facing,
    settings: MeshSettings,
) -> bool {
    debug_assert_eq!(
        world.block_at(pos),
        state,
        "the mesher renders the world's own block"
    );
    side_rendered(world, state, pos, side, settings, true)
}

fn side_rendered(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    side: Facing,
    settings: MeshSettings,
    state_is_live: bool,
) -> bool {
    let neighbour = pos.relative(side);
    let neighbour_state = world.block_at(neighbour);
    let props = block_props(state);
    let base_cull = |side: Facing| {
        base(
            world,
            state,
            neighbour,
            neighbour_state,
            side,
            settings,
            state_is_live,
        )
    };

    // The arms are the Java overrides, keyed by the block's class through `class_info` rather
    // than a per-call string match; `CullClass` names which classes each arm covers.
    match class_info(state).cull {
        // java: BlockLiquid — a liquid never draws against its own material, and its top face
        // is always drawn so the surface exists even under another liquid cell.
        CullClass::Liquid => {
            if block_props(neighbour_state).material == props.material {
                false
            } else if side == Facing::Up {
                true
            } else {
                base_cull(side)
            }
        }

        // java: BlockBreakable.shouldSideBeRendered's special state-sensitive branch for the
        // two glass blocks. Its non-glass same-block tail is handled for ice/slime below;
        // portal overrides the method outright.
        CullClass::Glass => {
            // Different *states* of glass keep their seam, identical ones cull.
            if neighbour_state != state {
                true
            } else if neighbour_state.block() == state.block() {
                false
            } else {
                base_cull(side)
            }
        }
        // java: BlockIce and BlockSlime both extend BlockBreakable with
        // `ignoreSimilarity == false`, so adjacent blocks of the same kind cull their shared
        // face. BlockPackedIce extends Block outright and therefore takes the base
        // implementation.
        CullClass::IceOrSlime => {
            if neighbour_state.block() == state.block() {
                false
            } else {
                base_cull(side)
            }
        }

        // java: BlockPane (and BlockStainedGlassPane, which extends it).
        CullClass::Pane => {
            if neighbour_state.block() == state.block() {
                false
            } else {
                base_cull(side)
            }
        }

        // java: BlockLeavesBase — fast graphics culls leaves against leaves of the same block.
        CullClass::Leaves => {
            if !settings.fancy_graphics && neighbour_state.block() == state.block() {
                false
            } else {
                base_cull(side)
            }
        }

        // java: BlockSlab — the long ternary, split out for legibility.
        CullClass::HalfSlab => slab(world, state, pos, neighbour, side, &base_cull),
        CullClass::DoubleSlab => base_cull(side),

        // java: BlockSnow, BlockCarpet — always draw the top.
        CullClass::SnowOrCarpet => side == Facing::Up || base_cull(side),

        // java: BlockFarmland
        CullClass::Farmland => match side {
            Facing::Up => true,
            Facing::North | Facing::South | Facing::West | Facing::East => {
                // `isOpaqueCube()` here is the *virtual* call, so it has to go through
                // [`is_opaque_cube`] and pick up the fancy-graphics leaves override; reading
                // `block_props().opaque_cube` straight would use the frozen fast-graphics
                // value and cull the dirt side against a leaf block.
                !is_opaque_cube(neighbour_state, settings)
                    && class_info(neighbour_state).cull != CullClass::Farmland
            }
            Facing::Down => base_cull(side),
        },

        // java: BlockFence, BlockFenceGate, BlockHopper, BlockAnvil, BlockDragonEgg,
        //       BlockPistonExtension — unconditionally true.
        CullClass::AlwaysDrawn => true,

        // java: BlockWall
        CullClass::Wall => side != Facing::Down || base_cull(side),

        // java: BlockRedstoneDiode (repeater, comparator)
        CullClass::Diode => !matches!(side, Facing::Down | Facing::Up),

        // java: BlockEndPortal
        CullClass::EndPortal => side == Facing::Down && base_cull(side),

        // java: BlockPortal — the axis it reads is the *neighbour's*, not its own, and the
        // side is only drawn at the edge of the portal slab. AXIS: `(meta & 3) == 2` is Z,
        // anything else is X.
        CullClass::Portal => {
            let neighbour_axis_z = if neighbour_state.block() == state.block() {
                let z = neighbour_state.meta() & 3 == 2;
                if z && !matches!(side, Facing::East | Facing::West) {
                    return false;
                }
                if !z && !matches!(side, Facing::South | Facing::North) {
                    return false;
                }
                Some(z)
            } else {
                None
            };
            let same = |p: BlockPos| world.block_at(p).block() == state.block();
            let edge_x = (same(neighbour.west()) && !same(neighbour.west().west()))
                || (same(neighbour.east()) && !same(neighbour.east().east()))
                || neighbour_axis_z == Some(false);
            let edge_z = (same(neighbour.north()) && !same(neighbour.north().north()))
                || (same(neighbour.south()) && !same(neighbour.south().south()))
                || neighbour_axis_z == Some(true);
            match side {
                Facing::West | Facing::East => edge_x,
                Facing::North | Facing::South => edge_z,
                Facing::Down | Facing::Up => false,
            }
        }

        CullClass::Base => base_cull(side),
    }
}

/// java: `BlockSlab.shouldSideBeRendered`, for a single (non-double) slab.
///
/// The `if`/`else if` chain is deliberately shaped like the Java ternary it comes from — two
/// of its arms happen to both yield `true`, which clippy would rather see merged, but merging
/// them would make the port unreadable against the source.
#[allow(clippy::if_same_then_else)]
fn slab(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    neighbour: BlockPos,
    side: Facing,
    base_cull: &dyn Fn(Facing) -> bool,
) -> bool {
    // java: the **static** `BlockSlab.isSlab(Block)` (`BlockSlab.java:145-148`), which is
    // `blockIn == Blocks.stone_slab || Blocks.wooden_slab || Blocks.stone_slab2` — the three
    // *single* slabs only. That is a narrower test than `super::light::is_slab`, which is
    // `getMixedBrightnessForBlock`'s `instanceof BlockSlab` and does include the doubles.
    // Reusing the wide one here makes a neighbouring `double_stone_slab` look like a slab: its
    // meta bit 3 is SEAMLESS, not HALF=TOP, so it is also read as a top slab and the
    // `!isSlab(here) || flag` arms flip.
    let is_slab = |s: BlockState| class_info(s).cull == CullClass::HalfSlab;
    // java: HALF == TOP is meta bit 3.
    let is_top = |s: BlockState| is_slab(s) && s.meta() & 8 != 0;

    if !matches!(side, Facing::Up | Facing::Down) && !base_cull(side) {
        return false;
    }
    // vanilla reads `pos` (the neighbour) and `pos.offset(side.opposite)` (the block itself).
    let here = world.block_at(neighbour);
    let behind = world.block_at(pos);
    let flag = is_top(here);
    let flag1 = is_top(behind);
    let _ = state;

    if flag1 {
        if side == Facing::Down {
            true
        } else if side == Facing::Up && base_cull(side) {
            true
        } else {
            !is_slab(here) || !flag
        }
    } else if side == Facing::Up {
        true
    } else if side == Facing::Down && base_cull(side) {
        true
    } else {
        !is_slab(here) || flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::fixture;

    fn world_with(blocks: &[(BlockPos, &str)]) -> World {
        let mut w = fixture::blank_world(1);
        for (p, name) in blocks {
            fixture::place(&mut w, *p, name);
        }
        w
    }

    #[test]
    fn an_opaque_neighbour_culls_a_full_cube_face() {
        let a = BlockPos::new(1, 1, 1);
        let b = a.up();
        let w = world_with(&[(a, "minecraft:stone"), (b, "minecraft:stone")]);
        let stone = w.block_at(a);
        assert!(!should_side_be_rendered(
            &w,
            stone,
            a,
            Facing::Up,
            MeshSettings::default()
        ));
        assert!(should_side_be_rendered(
            &w,
            stone,
            a,
            Facing::Down,
            MeshSettings::default()
        ));
    }

    #[test]
    fn identical_glass_culls_but_different_colours_do_not() {
        let a = BlockPos::new(1, 1, 1);
        let b = a.up();
        let w = world_with(&[(a, "minecraft:glass"), (b, "minecraft:glass")]);
        let glass = w.block_at(a);
        assert!(!should_side_be_rendered(
            &w,
            glass,
            a,
            Facing::Up,
            MeshSettings::default()
        ));

        let mut w2 = world_with(&[(a, "minecraft:stained_glass")]);
        let sg = mc_data::block_by_name("minecraft:stained_glass").expect("stained_glass");
        w2.set_block(a, BlockState::new(sg, 0));
        w2.set_block(b, BlockState::new(sg, 5));
        assert!(should_side_be_rendered(
            &w2,
            BlockState::new(sg, 0),
            a,
            Facing::Up,
            MeshSettings::default()
        ));
    }

    #[test]
    fn leaves_cull_against_leaves_only_in_fast_graphics() {
        let a = BlockPos::new(1, 1, 1);
        let b = a.up();
        let w = world_with(&[(a, "minecraft:leaves"), (b, "minecraft:leaves")]);
        let leaves = w.block_at(a);
        let fancy = MeshSettings {
            fancy_graphics: true,
            ..MeshSettings::default()
        };
        let fast = MeshSettings {
            fancy_graphics: false,
            ..MeshSettings::default()
        };
        assert!(should_side_be_rendered(&w, leaves, a, Facing::Up, fancy));
        assert!(!should_side_be_rendered(&w, leaves, a, Facing::Up, fast));
    }

    #[test]
    fn slime_uses_blockbreakables_same_block_cull() {
        // java: BlockSlime -> BlockBreakable(Material.clay, false), then
        // BlockBreakable.shouldSideBeRendered returns false when the neighbour block is `this`.
        // The base implementation cannot stand in for it because slime is not opaque.
        let a = BlockPos::new(1, 1, 1);
        let w = world_with(&[(a, "minecraft:slime"), (a.up(), "minecraft:slime")]);
        let slime = w.block_at(a);
        assert!(
            !is_opaque_cube(slime, MeshSettings::default()),
            "the control must exercise BlockBreakable rather than base opaque-cube culling"
        );
        assert!(!should_side_be_rendered(
            &w,
            slime,
            a,
            Facing::Up,
            MeshSettings::default(),
        ));

        // Negative control: the override is identity-based, not a blanket rule for every
        // translucent neighbour. A slime face against glass remains visible.
        let other = world_with(&[(a, "minecraft:slime"), (a.up(), "minecraft:glass")]);
        assert!(should_side_be_rendered(
            &other,
            other.block_at(a),
            a,
            Facing::Up,
            MeshSettings::default(),
        ));
    }

    #[test]
    fn farmland_sides_follow_the_fancy_graphics_leaves_override() {
        // java: BlockFarmland.shouldSideBeRendered:136 — `block.isOpaqueCube()` is the virtual
        // call, and `BlockLeaves` overrides it to `!fancyGraphics`. Under fancy graphics the
        // canopy is not opaque, so the dirt side next to it must still be drawn.
        let a = BlockPos::new(1, 1, 1);
        let w = world_with(&[(a, "minecraft:farmland"), (a.north(), "minecraft:leaves")]);
        let farmland = w.block_at(a);
        let fancy = MeshSettings {
            fancy_graphics: true,
            ..MeshSettings::default()
        };
        let fast = MeshSettings {
            fancy_graphics: false,
            ..MeshSettings::default()
        };
        assert!(
            should_side_be_rendered(&w, farmland, a, Facing::North, fancy),
            "fancy graphics: leaves are not an opaque cube, so the side is drawn"
        );
        assert!(
            !should_side_be_rendered(&w, farmland, a, Facing::North, fast),
            "fast graphics: leaves are an opaque cube, so the side culls"
        );
    }

    #[test]
    fn farmland_never_draws_a_side_against_farmland() {
        // java: BlockFarmland.shouldSideBeRendered:137 — `&& block != Blocks.farmland`.
        let a = BlockPos::new(1, 1, 1);
        let w = world_with(&[(a, "minecraft:farmland"), (a.north(), "minecraft:farmland")]);
        let farmland = w.block_at(a);
        assert!(!should_side_be_rendered(
            &w,
            farmland,
            a,
            Facing::North,
            MeshSettings::default()
        ));
        // ...but a non-opaque, non-farmland neighbour does draw it.
        let w2 = world_with(&[(a, "minecraft:farmland"), (a.north(), "minecraft:glass")]);
        assert!(should_side_be_rendered(
            &w2,
            farmland,
            a,
            Facing::North,
            MeshSettings::default()
        ));
    }

    #[test]
    fn a_fence_always_renders_every_side() {
        let a = BlockPos::new(1, 1, 1);
        let w = world_with(&[(a, "minecraft:fence"), (a.up(), "minecraft:stone")]);
        let fence = w.block_at(a);
        for side in Facing::VALUES {
            assert!(
                should_side_be_rendered(&w, fence, a, side, MeshSettings::default()),
                "fence culled {side:?}"
            );
        }
    }

    #[test]
    fn a_partial_block_always_renders_the_faces_it_does_not_fill() {
        // A snow layer's max_y is 1/8, so its top face is drawn even under solid stone.
        let a = BlockPos::new(1, 1, 1);
        let w = world_with(&[(a, "minecraft:snow_layer"), (a.up(), "minecraft:stone")]);
        let snow = w.block_at(a);
        assert!(should_side_be_rendered(
            &w,
            snow,
            a,
            Facing::Up,
            MeshSettings::default()
        ));
    }

    #[test]
    fn cactus_uses_full_render_bounds_for_face_culling() {
        let a = BlockPos::new(1, 1, 1);
        let w = world_with(&[(a, "minecraft:cactus"), (a.east(), "minecraft:stone")]);
        let cactus = w.block_at(a);

        // java: BlockCactus.getSelectedBoundingBox is inset on X/Z, but its constructor
        // render bounds remain 0..1.  Therefore an opaque neighbour culls the covered side,
        // while the three exposed horizontal sides still render against air.
        assert!(!should_side_be_rendered(
            &w,
            cactus,
            a,
            Facing::East,
            MeshSettings::default(),
        ));
        for side in [Facing::North, Facing::South, Facing::West] {
            assert!(
                should_side_be_rendered(&w, cactus, a, side, MeshSettings::default()),
                "cactus exposed side {side:?} was culled"
            );
        }

        // The render fix must not change the separate selection box used by the outline.
        let selected = crate::terrain::selected_bounding_box(&w, a, cactus);
        assert_eq!(selected.min.x, f64::from(a.x) + 0.0625);
        assert_eq!(selected.max.x, f64::from(a.x + 1) - 0.0625);
        assert_eq!(selected.min.z, f64::from(a.z) + 0.0625);
        assert_eq!(selected.max.z, f64::from(a.z + 1) - 0.0625);
    }

    #[test]
    fn cactus_cull_and_boxes_hold_at_negative_coordinates() {
        let a = BlockPos::new(-1, 3, -1);
        let mut w = fixture::blank_world(1);
        fixture::place(&mut w, a, "minecraft:cactus");
        let cactus = w.block_at(a);

        // Java's BlockCactus never changes the constructor render bounds, so every face is
        // culled when the corresponding neighbour is an opaque full cube.  This explicitly
        // exercises the negative-coordinate path as well as vertical faces.
        for side in Facing::VALUES {
            fixture::place(&mut w, a.relative(side), "minecraft:stone");
            assert!(
                !should_side_be_rendered(&w, cactus, a, side, MeshSettings::default()),
                "opaque neighbour must cull cactus {side:?} face"
            );
            w.set_block(a.relative(side), BlockState::AIR);
        }

        // Selection is inset only horizontally and collision is inset on all three axes, as in
        // BlockCactus.getSelectedBoundingBox/getCollisionBoundingBox.  These are distinct from
        // the full render bounds used above.
        let selected = crate::terrain::selected_bounding_box(&w, a, cactus);
        assert_eq!(selected.min, mc_math::DVec3::new(-0.9375, 3.0, -0.9375));
        assert_eq!(selected.max, mc_math::DVec3::new(-0.0625, 4.0, -0.0625));
        let boxes: Vec<_> = mc_world::collision_boxes_at(cactus, a).collect();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].min, mc_math::DVec3::new(-0.9375, 3.0, -0.9375));
        assert_eq!(boxes[0].max, mc_math::DVec3::new(-0.0625, 3.9375, -0.0625));

        // With air around it, every full render face is emitted, including up/down.
        for side in Facing::VALUES {
            assert!(
                should_side_be_rendered(&w, cactus, a, side, MeshSettings::default()),
                "air neighbour must keep cactus {side:?} face"
            );
        }
    }

    #[test]
    fn a_live_door_slab_keeps_its_inset_face_against_opaque_stone() {
        let lower = BlockPos::new(1, 1, 1);
        let door = mc_data::block_by_name("minecraft:wooden_door").expect("wooden door");
        let mut discriminating_cases = 0;

        for facing in 0u8..4 {
            for open in [false, true] {
                for hinge_right in [false, true] {
                    for side in [Facing::North, Facing::South, Facing::West, Facing::East] {
                        let mut world = fixture::blank_world(1);
                        world.set_block(
                            lower,
                            BlockState::new(door, facing | if open { 4 } else { 0 }),
                        );
                        world.set_block(
                            lower.up(),
                            BlockState::new(door, 8 | if hinge_right { 1 } else { 0 }),
                        );

                        for pos in [lower, lower.up()] {
                            let state = world.block_at(pos);
                            let live = crate::terrain::selected_bounding_box(&world, pos, state);
                            let static_bounds = block_props(state).bounds;
                            let live_inset = match side {
                                Facing::North => live.min.z > f64::from(pos.z),
                                Facing::South => live.max.z < f64::from(pos.z + 1),
                                Facing::West => live.min.x > f64::from(pos.x),
                                Facing::East => live.max.x < f64::from(pos.x + 1),
                                Facing::Down | Facing::Up => unreachable!(),
                            };
                            let static_reaches_edge = match side {
                                Facing::North => static_bounds.min_z <= 0.0,
                                Facing::South => static_bounds.max_z >= 1.0,
                                Facing::West => static_bounds.min_x <= 0.0,
                                Facing::East => static_bounds.max_x >= 1.0,
                                Facing::Down | Facing::Up => unreachable!(),
                            };
                            if !live_inset || !static_reaches_edge {
                                continue;
                            }
                            discriminating_cases += 1;
                            world.set_block(
                                pos.relative(side),
                                fixture::state_of("minecraft:stone"),
                            );
                            assert!(
                                should_side_be_rendered(
                                    &world,
                                    state,
                                    pos,
                                    side,
                                    MeshSettings::default(),
                                ),
                                "live combined door bounds keep the inset face: facing={facing} open={open} hinge={hinge_right} pos={pos:?} side={side:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            discriminating_cases > 0,
            "fixture must contain a combined-state door case that static state bounds cannot represent"
        );
        // `an_opaque_neighbour_culls_a_full_cube_face` is the paired full-cube negative control.
    }

    #[test]
    fn double_slabs_are_not_single_slabs_in_the_side_render_predicate() {
        // java: BlockSlab.isSlab(Block) names only stone_slab, wooden_slab, and
        // stone_slab2.  In particular, bit 3 on a double slab means SEAMLESS rather than
        // HALF=TOP.  Reusing light::is_slab's wider `instanceof BlockSlab` predicate drops
        // both of these faces (the two mismatches in the auditor's 84-case sweep).
        let a = BlockPos::new(1, 2, 1);
        let single = mc_data::block_by_name("minecraft:stone_slab").expect("stone_slab");
        let double =
            mc_data::block_by_name("minecraft:double_stone_slab").expect("double_stone_slab");

        let mut below = fixture::blank_world(1);
        below.set_block(a, BlockState::new(single, 0)); // bottom half
        below.set_block(a.down(), BlockState::new(double, 0)); // non-seamless
        assert!(
            should_side_be_rendered(
                &below,
                below.block_at(a),
                a,
                Facing::Down,
                MeshSettings::default(),
            ),
            "a non-seamless double slab below must not be read as a bottom single slab"
        );

        let mut above = fixture::blank_world(1);
        above.set_block(a, BlockState::new(single, 8)); // top half
        above.set_block(a.up(), BlockState::new(double, 8)); // seamless, not top
        assert!(
            should_side_be_rendered(
                &above,
                above.block_at(a),
                a,
                Facing::Up,
                MeshSettings::default(),
            ),
            "a seamless double slab above must not be read as a top single slab"
        );
    }
}
