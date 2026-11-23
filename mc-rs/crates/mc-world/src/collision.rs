//! The `mc_data::Aabb` -> `mc_math::Aabb` boundary conversion.
//!
//! `mc-data` (S01) deliberately stores collision shapes as its own dependency-free, block-local
//! `Aabb` (six `f64`s) rather than depending on `mc-math`, so that `mc-data` stays a leaf crate.
//! Per S01's handoff note, `mc-world` is the natural place to cross that boundary: this module
//! offsets a block-local shape into world space and produces the `mc_math::Aabb` that
//! `mc-physics` (S13) sweeps against.
//!
//! **S13 must check [`mc_data::BlockProps::collision_depends_on_neighbors`] before trusting the
//! boxes returned by the state-only [`collision_boxes_at`] helper.** 33 blocks / 195 states
//! (fences, walls, panes, iron bars, doors, chests, stair corners) only store their *isolated*
//! shape. The pane-, wall-, and chest-specific world-aware helpers below establish reusable
//! selection/collision source seams; other families remain the physics crate's responsibility
//! until they acquire an equivalent world-aware API.

use crate::World;
use mc_data::{block_info, block_props, BlockState};
use mc_math::{Aabb, BlockPos, DVec3};

/// Converts one block-local collision box (as extracted by S01, `0.0..=1.0` for a full cube)
/// into a world-space [`mc_math::Aabb`] anchored at `pos`.
#[inline]
#[must_use]
pub fn to_world_aabb(local: mc_data::Aabb, pos: BlockPos) -> mc_math::Aabb {
    let origin = DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));
    mc_math::Aabb::new(
        DVec3::new(local.min_x, local.min_y, local.min_z) + origin,
        DVec3::new(local.max_x, local.max_y, local.max_z) + origin,
    )
}

/// The world-space collision boxes for the block state at `pos`.
///
/// Empty for air and other non-colliding blocks. For the 195 neighbour-dependent states this
/// is only the *isolated* shape — see the module docs before using it for a real sweep.
pub fn collision_boxes_at(state: BlockState, pos: BlockPos) -> impl Iterator<Item = mc_math::Aabb> {
    block_props(state)
        .collision
        .boxes()
        .iter()
        .map(move |&b| to_world_aabb(b, pos))
}

/// The world-space collision boxes for a glass pane or iron-bar state at `pos`.
///
/// This is the world-aware counterpart to [`collision_boxes_at`].  Java
/// `BlockPane.addCollisionBoxesToList` reads the four neighbouring blocks every time it is
/// called; the extracted `BlockProps::collision` row only contains the isolated cross.  The
/// returned boxes are already filtered with Java's strict `AxisAlignedBB.intersectsWith` test.
/// `None` means that the centre state is not a `BlockPane` subclass; `Some(empty)` means that it
/// is a pane but the supplied mask misses its live boxes.
///
/// java: `BlockPane.addCollisionBoxesToList` (`BlockPane.java:75-126`).
#[must_use]
pub fn pane_collision_boxes_at(world: &World, pos: BlockPos, mask: &Aabb) -> Option<Vec<Aabb>> {
    let state = world.block_at(pos);
    if !is_pane(state) {
        return None;
    }

    let (north, east, south, west) = pane_connections(world, pos, state.block());
    let any = north || east || south || west;
    let mut out = Vec::with_capacity(2);

    // Java emits the X-axis pass first.  When exactly one horizontal side is connected, the
    // other side of that pass is deliberately absent; this is why a north-only pane has only a
    // Z half-bar instead of a central post plus a Z half-bar.
    if (!west || !east) && any {
        if west {
            push_pane_box(&mut out, pos, mask, 0.0, 0.0, 0.4375, 0.5, 1.0, 0.5625);
        } else if east {
            push_pane_box(&mut out, pos, mask, 0.5, 0.0, 0.4375, 1.0, 1.0, 0.5625);
        }
    } else {
        push_pane_box(&mut out, pos, mask, 0.0, 0.0, 0.4375, 1.0, 1.0, 0.5625);
    }

    // The Z-axis pass follows the X pass, matching the two super.addCollisionBoxesToList calls
    // in BlockPane.java and therefore preserving collision-list order for equal configurations.
    if (!north || !south) && any {
        if north {
            push_pane_box(&mut out, pos, mask, 0.4375, 0.0, 0.0, 0.5625, 1.0, 0.5);
        } else if south {
            push_pane_box(&mut out, pos, mask, 0.4375, 0.0, 0.5, 0.5625, 1.0, 1.0);
        }
    } else {
        push_pane_box(&mut out, pos, mask, 0.4375, 0.0, 0.0, 0.5625, 1.0, 1.0);
    }

    Some(out)
}

/// The world-space selection bounds left by `BlockPane.setBlockBoundsBasedOnState`.
///
/// This is the one connected envelope used by Java's `Block.collisionRayTrace` and
/// `getSelectedBoundingBox`; it intentionally differs from [`pane_collision_boxes_at`], whose
/// collision path can return two independent bars.  A caller that draws the outline can apply
/// its own Java `0.002` expansion after this value is returned.
///
/// java: `BlockPane.setBlockBoundsBasedOnState` (`BlockPane.java:129-173`).
#[must_use]
pub fn pane_selection_bounds_at(world: &World, pos: BlockPos) -> Option<Aabb> {
    let state = world.block_at(pos);
    if !is_pane(state) {
        return None;
    }

    let (north, east, south, west) = pane_connections(world, pos, state.block());
    let any = north || east || south || west;

    let (min_x, max_x) = if (!west || !east) && any {
        if west {
            (0.0, 0.5625)
        } else if east {
            (0.4375, 1.0)
        } else {
            (0.4375, 0.5625)
        }
    } else {
        (0.0, 1.0)
    };

    let (min_z, max_z) = if (!north || !south) && any {
        if north {
            (0.0, 0.5625)
        } else if south {
            (0.4375, 1.0)
        } else {
            (0.4375, 0.5625)
        }
    } else {
        (0.0, 1.0)
    };

    Some(Aabb::new(
        DVec3::new(
            f64::from(pos.x) + min_x,
            f64::from(pos.y),
            f64::from(pos.z) + min_z,
        ),
        DVec3::new(
            f64::from(pos.x) + max_x,
            f64::from(pos.y + 1),
            f64::from(pos.z) + max_z,
        ),
    ))
}

/// The world-space selection bounds left by `BlockFence.setBlockBoundsBasedOnState`.
///
/// This is the world-aware counterpart to the isolated `BlockProps::bounds` row. Java reads
/// each horizontal neighbour before every `collisionRayTrace` and selection-outline query, so a
/// connected arm must be part of the one envelope returned here. The bounds are intentionally
/// one block high; a fence's movement collision boxes are 1.5 blocks high and belong to a
/// separate collision path.
///
/// java: `BlockFence.setBlockBoundsBasedOnState` (`BlockFence.java:109-141`).
#[must_use]
pub fn fence_selection_bounds_at(world: &World, pos: BlockPos) -> Option<Aabb> {
    let state = world.block_at(pos);
    if !is_fence(state) {
        return None;
    }

    let (north, east, south, west) = fence_connections(world, pos, block_props(state).material);
    let min_x = if west { 0.0 } else { 0.375 };
    let max_x = if east { 1.0 } else { 0.625 };
    let min_z = if north { 0.0 } else { 0.375 };
    let max_z = if south { 1.0 } else { 0.625 };

    Some(Aabb::new(
        DVec3::new(
            f64::from(pos.x) + min_x,
            f64::from(pos.y),
            f64::from(pos.z) + min_z,
        ),
        DVec3::new(
            f64::from(pos.x) + max_x,
            f64::from(pos.y + 1),
            f64::from(pos.z) + max_z,
        ),
    ))
}

/// The world-space selection bounds left by `BlockChest.setBlockBoundsBasedOnState`.
///
/// This is the world-aware counterpart to the isolated `BlockProps::bounds` row. Java's chest
/// singleton checks the four horizontal neighbours in source order (north, south, west, east)
/// and widens exactly the first matching side. The same mutable bounds are consumed by both
/// `Block.collisionRayTrace` and the client selection outline, so this helper deliberately
/// returns the ordered source box before either caller applies its own ray/outline handling.
/// Ordinary and trapped chests are distinct Java singleton instances even though both use the
/// `BlockChest` class; [`World::chest_neighbours`] preserves that same-type-only rule.
///
/// java: `BlockChest.setBlockBoundsBasedOnState` (`BlockChest.java:66-87`), called from
/// `Block.collisionRayTrace` (`Block.java:681`) and `RenderGlobal.drawSelectionBox`.
#[must_use]
pub fn chest_selection_bounds_at(world: &World, pos: BlockPos) -> Option<Aabb> {
    let state = world.block_at(pos);
    if !is_chest(state) {
        return None;
    }

    let neighbours = world.chest_neighbours(pos);
    let (min_x, min_z, max_x, max_z) = if neighbours.z_neg {
        // java: `pos.north()` is tested first; it wins when malformed/multi-adjacent layouts
        // have more than one same-type neighbour.
        (0.0625, 0.0, 0.9375, 0.9375)
    } else if neighbours.z_pos {
        (0.0625, 0.0625, 0.9375, 1.0)
    } else if neighbours.x_neg {
        (0.0, 0.0625, 0.9375, 0.9375)
    } else if neighbours.x_pos {
        (0.0625, 0.0625, 1.0, 0.9375)
    } else {
        (0.0625, 0.0625, 0.9375, 0.9375)
    };

    Some(Aabb::new(
        DVec3::new(
            f64::from(pos.x) + min_x,
            f64::from(pos.y),
            f64::from(pos.z) + min_z,
        ),
        DVec3::new(
            f64::from(pos.x) + max_x,
            f64::from(pos.y) + 0.875,
            f64::from(pos.z) + max_z,
        ),
    ))
}

/// The world-space selection bounds left by `BlockVine.setBlockBoundsBasedOnState`.
///
/// Vine bounds are a particularly important exception to the ordinary `Aabb` construction rule:
/// Java's mutable `Block.minX..maxZ` fields start at `(1, 1, 1, 0, 0, 0)`, and an unattached
/// metadata-0 vine leaves those corners inverted.  `Block.collisionRayTrace` consumes those six
/// fields directly, so this source oracle deliberately returns a raw [`Aabb`] rather than calling
/// [`Aabb::new`], which normalises each axis.  The selected-outline caller is responsible for
/// applying Java's later `AxisAlignedBB` normalisation and expansion.
///
/// The four horizontal faces come from the stored metadata (`SOUTH=1`, `WEST=2`, `NORTH=4`,
/// `EAST=8`).  Only when none is stored does Java read the block above and use
/// `isFullCube() && blockMaterial.blocksMovement()` to make the 1/16 ceiling slab.
///
/// java: `BlockVine.setBlockBoundsBasedOnState` (`BlockVine.java:81-146`).
#[must_use]
pub fn vine_selection_bounds_at(world: &World, pos: BlockPos) -> Option<Aabb> {
    let state = world.block_at(pos);
    if !is_vine(state) {
        return None;
    }

    // java: BlockVine.getStateFromMeta (`BlockVine.java:435-438`).  UP is an actual-state
    // property and is intentionally not part of this bounds method; the fallback below reads the
    // supporting block directly, just as the Java method does.
    let meta = state.meta();
    let south = meta & 1 != 0;
    let west = meta & 2 != 0;
    let north = meta & 4 != 0;
    let east = meta & 8 != 0;
    let any = south || west || north || east;

    // Java initialises min fields to 1 and max fields to 0.  Preserve that ordering until the
    // caller chooses a Java operation which normalises it (e.g. getSelectedBoundingBox).
    let (mut min_x, mut min_y, mut min_z): (f64, f64, f64) = (1.0, 1.0, 1.0);
    let (mut max_x, mut max_y, mut max_z): (f64, f64, f64) = (0.0, 0.0, 0.0);

    // Keep the source order: later faces deliberately widen/reset fields set by earlier faces.
    if west {
        max_x = max_x.max(0.0625);
        min_x = 0.0;
        min_y = 0.0;
        max_y = 1.0;
        min_z = 0.0;
        max_z = 1.0;
    }
    if east {
        min_x = min_x.min(0.9375);
        max_x = 1.0;
        min_y = 0.0;
        max_y = 1.0;
        min_z = 0.0;
        max_z = 1.0;
    }
    if north {
        max_z = max_z.max(0.0625);
        min_z = 0.0;
        min_x = 0.0;
        max_x = 1.0;
        min_y = 0.0;
        max_y = 1.0;
    }
    if south {
        min_z = min_z.min(0.9375);
        max_z = 1.0;
        min_x = 0.0;
        max_x = 1.0;
        min_y = 0.0;
        max_y = 1.0;
    }

    // java: BlockVine.canPlaceOn (`BlockVine.java:175-178`) and the final branch at :136-144.
    // This is deliberately `full_cube`, not `full_block` or `normal_cube`: those are distinct
    // virtual predicates in 1.8.9 (for example, a powered redstone block is full and movement
    // blocking but is not a normal cube).
    if !any && vine_can_place_on(world.block_at(pos.up())) {
        min_y = min_y.min(0.9375);
        max_y = 1.0;
        min_x = 0.0;
        max_x = 1.0;
        min_z = 0.0;
        max_z = 1.0;
    }

    Some(raw_world_aabb(
        pos,
        (min_x, min_y, min_z),
        (max_x, max_y, max_z),
    ))
}

/// The world-space collision box for a cobblestone wall state at `pos`.
///
/// This is the world-aware counterpart to [`collision_boxes_at`]. Java
/// `BlockWall.getCollisionBoundingBox` recomputes the four horizontal connections through
/// `setBlockBoundsBasedOnState`, then raises only `maxY` to `1.5`; the extracted
/// `BlockProps::collision` row contains the isolated post instead. `None` means that the centre
/// state is not a `BlockWall`; `Some(empty)` means that the wall box does not strictly intersect
/// the supplied query mask.
///
/// java: `BlockWall.getCollisionBoundingBox` / `setBlockBoundsBasedOnState`
/// (`BlockWall.java:67-113`).
#[must_use]
pub fn wall_collision_boxes_at(world: &World, pos: BlockPos, mask: &Aabb) -> Option<Vec<Aabb>> {
    let state = world.block_at(pos);
    if !is_wall(state) {
        return None;
    }

    let bounds = wall_bounds_at(world, pos, state.block(), true);
    Some(if mask.intersects_with(&bounds) {
        vec![bounds]
    } else {
        Vec::new()
    })
}

/// The single carried wall box used by `BlockPistonMoving.getBoundingBox`.
///
/// Unlike the ordinary world collision query, the moving carrier asks the wall block for its
/// singular `getCollisionBoundingBox` while the carrier occupies `pos`; the wall implementation
/// still reads the four neighbours at that position and then forces `maxY` to `1.5`. The centre
/// block need not itself still be a wall because the world contains `piston_extension` there.
pub(crate) fn wall_carried_collision_bounds(
    world: &World,
    pos: BlockPos,
    self_block: mc_data::BlockId,
) -> Aabb {
    wall_bounds_at(world, pos, self_block, true)
}

/// The world-space selection bounds left by `BlockWall.setBlockBoundsBasedOnState`.
///
/// This is the one connected envelope used by Java's `Block.collisionRayTrace` and
/// `getSelectedBoundingBox`; it intentionally differs from the wall collision height, which is
/// forced to `1.5` by `getCollisionBoundingBox`. A caller that draws the outline can apply its
/// own Java `0.002` expansion after this value is returned.
///
/// java: `BlockWall.setBlockBoundsBasedOnState` (`BlockWall.java:67-110`).
#[must_use]
pub fn wall_selection_bounds_at(world: &World, pos: BlockPos) -> Option<Aabb> {
    let state = world.block_at(pos);
    if !is_wall(state) {
        return None;
    }

    Some(wall_bounds_at(world, pos, state.block(), false))
}

/// The block-local box `BlockTorch.collisionRayTrace` writes right before it
/// intersects the ray (`BlockRedstoneTorch` inherits the method), or `None` for any other block.
///
/// `Block.getSelectedBoundingBox` reads the same bound fields back for the outline, so this is
/// both the pick shape and the outline. The extracted registry `bounds` row is the constructor's
/// full cube (torches never override `setBlockBoundsBasedOnState`), which made the outline a
/// whole block. Every component is a `float` widened to
/// `double`, as `setBlockBounds` does, so an edge-of-box pick stays bit-exact.
///
/// java: `BlockTorch.java:184-211` (meta: 1 east, 2 west, 3 south, 4 north, 5 floor).
#[must_use]
pub fn torch_bounds(state: BlockState) -> Option<Aabb> {
    if !matches!(
        block_info(state.block()).java_class.rsplit('.').next(),
        Some("BlockTorch" | "BlockRedstoneTorch")
    ) {
        return None;
    }
    let f = 0.15_f32;
    let (min, max): ([f32; 3], [f32; 3]) = match state.meta() {
        1 => ([0.0, 0.2, 0.5 - f], [f * 2.0, 0.8, 0.5 + f]),
        2 => ([1.0 - f * 2.0, 0.2, 0.5 - f], [1.0, 0.8, 0.5 + f]),
        3 => ([0.5 - f, 0.2, 0.0], [0.5 + f, 0.8, f * 2.0]),
        4 => ([0.5 - f, 0.2, 1.0 - f * 2.0], [0.5 + f, 0.8, 1.0]),
        _ => {
            let f = 0.1_f32;
            ([0.5 - f, 0.0, 0.5 - f], [0.5 + f, 0.6, 0.5 + f])
        }
    };
    let widen = |v: [f32; 3]| DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
    Some(Aabb::new(widen(min), widen(max)))
}

fn is_pane(state: BlockState) -> bool {
    matches!(
        block_info(state.block()).java_class.rsplit('.').next(),
        Some("BlockPane" | "BlockStainedGlassPane")
    )
}

fn is_fence(state: BlockState) -> bool {
    matches!(
        block_info(state.block()).java_class.rsplit('.').next(),
        Some("BlockFence")
    )
}

fn is_chest(state: BlockState) -> bool {
    matches!(
        block_info(state.block()).java_class.rsplit('.').next(),
        Some("BlockChest")
    )
}

fn is_wall(state: BlockState) -> bool {
    matches!(
        block_info(state.block()).java_class.rsplit('.').next(),
        Some("BlockWall")
    )
}

fn is_vine(state: BlockState) -> bool {
    matches!(
        block_info(state.block()).java_class.rsplit('.').next(),
        Some("BlockVine")
    )
}

/// Java's `BlockVine.canPlaceOn(Block)` support predicate.
fn vine_can_place_on(state: BlockState) -> bool {
    let props = block_props(state);
    props.full_cube && props.material.blocks_movement()
}

/// Offsets a pair of corners without normalising them.
///
/// Most world collision helpers call [`Aabb::new`] because their Java source supplies ordered
/// corners.  Vine's unattached state is the intentional exception, so keep this conversion local
/// to the vine selection seam instead of weakening the normalised [`to_world_aabb`] contract.
#[inline]
fn raw_world_aabb(pos: BlockPos, min: (f64, f64, f64), max: (f64, f64, f64)) -> Aabb {
    let origin = pos.to_vec3();
    Aabb {
        min: origin + DVec3::new(min.0, min.1, min.2),
        max: origin + DVec3::new(max.0, max.1, max.2),
    }
}

/// Returns `(north, east, south, west)`, the four live connection bits read by
/// `BlockFence.setBlockBoundsBasedOnState` and `getActualState`.
fn fence_connections(
    world: &World,
    pos: BlockPos,
    self_material: mc_data::Material,
) -> (bool, bool, bool, bool) {
    (
        fence_can_connect(world.block_at(pos.north()), self_material),
        fence_can_connect(world.block_at(pos.east()), self_material),
        fence_can_connect(world.block_at(pos.south()), self_material),
        fence_can_connect(world.block_at(pos.west()), self_material),
    )
}

/// Java `BlockFence.canConnectTo`: same-material fences and all fence gates connect; barriers,
/// different-material fences, gourds, and non-full/non-opaque blocks do not.
fn fence_can_connect(neighbour: BlockState, self_material: mc_data::Material) -> bool {
    let props = block_props(neighbour);
    let info = block_info(neighbour.block());
    let class = info.java_class.rsplit('.').next();
    if class == Some("BlockBarrier") {
        return false;
    }
    if class == Some("BlockFence") && props.material == self_material {
        return true;
    }
    if class == Some("BlockFenceGate") {
        return true;
    }
    props.material.is_opaque() && props.full_cube && props.material != mc_data::Material::Gourd
}

/// Returns `(north, east, south, west)`, the four live connection bits read by
/// `BlockWall.setBlockBoundsBasedOnState`.
fn wall_connections(
    world: &World,
    pos: BlockPos,
    self_block: mc_data::BlockId,
) -> (bool, bool, bool, bool) {
    (
        wall_can_connect(world.block_at(pos.north()), self_block),
        wall_can_connect(world.block_at(pos.east()), self_block),
        wall_can_connect(world.block_at(pos.south()), self_block),
        wall_can_connect(world.block_at(pos.west()), self_block),
    )
}

/// Builds either the selection envelope (`collision = false`) or the collision box
/// (`collision = true`) after the live connection branches have set its horizontal bounds.
fn wall_bounds_at(
    world: &World,
    pos: BlockPos,
    self_block: mc_data::BlockId,
    collision: bool,
) -> Aabb {
    let (north, east, south, west) = wall_connections(world, pos, self_block);
    let mut max_y = if collision { 1.5 } else { 1.0 };

    let mut min_x = 0.25;
    let mut max_x = 0.75;
    let mut min_z = 0.25;
    let mut max_z = 0.75;

    if north {
        min_z = 0.0;
    }
    if south {
        max_z = 1.0;
    }
    if west {
        min_x = 0.0;
    }
    if east {
        max_x = 1.0;
    }

    // Java lowers a straight run and narrows its post away entirely. The collision path calls
    // this same bounds setup and then overwrites maxY with 1.5, hence max_y is supplied by the
    // caller rather than folded into this branch.
    if north && south && !west && !east {
        if !collision {
            max_y = 0.8125;
        }
        min_x = 0.3125;
        max_x = 0.6875;
    } else if !north && !south && west && east {
        if !collision {
            max_y = 0.8125;
        }
        min_z = 0.3125;
        max_z = 0.6875;
    }

    Aabb::new(
        DVec3::new(
            f64::from(pos.x) + min_x,
            f64::from(pos.y),
            f64::from(pos.z) + min_z,
        ),
        DVec3::new(
            f64::from(pos.x) + max_x,
            f64::from(pos.y) + max_y,
            f64::from(pos.z) + max_z,
        ),
    )
}

/// Java `BlockWall.canConnectTo`: barriers are an explicit exclusion, while the same wall
/// singleton, every fence gate, and opaque full cubes other than gourds connect. Wall metadata
/// (normal versus mossy) does not change this identity check.
///
/// "Full cube" is `Block.isFullCube()` (`BlockWall.java:125`), not `isFullBlock()`: the two
/// differ for slime, mob spawners and End Portal Frames, which a wall joins.
fn wall_can_connect(neighbour: BlockState, self_block: mc_data::BlockId) -> bool {
    let props = block_props(neighbour);
    let info = block_info(neighbour.block());
    if info.java_class.rsplit('.').next() == Some("BlockBarrier") {
        return false;
    }
    props.block == self_block
        || info.java_class.rsplit('.').next() == Some("BlockFenceGate")
        || (props.material.is_opaque()
            && props.full_cube
            && props.material != mc_data::Material::Gourd)
}

/// Returns `(north, east, south, west)`, grouped by the two collision axes while retaining the
/// same four directions that `BlockPane.getActualState` derives.
fn pane_connections(
    world: &World,
    pos: BlockPos,
    self_block: mc_data::BlockId,
) -> (bool, bool, bool, bool) {
    (
        pane_can_connect(world.block_at(pos.north()), self_block),
        pane_can_connect(world.block_at(pos.east()), self_block),
        pane_can_connect(world.block_at(pos.south()), self_block),
        pane_can_connect(world.block_at(pos.west()), self_block),
    )
}

/// Java `BlockPane.canPaneConnectToBlock`: full blocks, the same singleton, vanilla glass
/// blocks, and every `BlockPane` subclass connect.  In particular, barriers connect because
/// `isFullBlock()` is true; unlike fences and walls, panes have no barrier exclusion.
fn pane_can_connect(neighbour: BlockState, self_block: mc_data::BlockId) -> bool {
    let props = block_props(neighbour);
    let info = block_info(neighbour.block());
    props.full_block
        || neighbour.block() == self_block
        || matches!(
            info.name,
            "minecraft:glass" | "minecraft:stained_glass" | "minecraft:stained_glass_pane"
        )
        || matches!(
            info.java_class.rsplit('.').next(),
            Some("BlockPane" | "BlockStainedGlassPane")
        )
}

// Keep the Java block-bounds call shape explicit: each coordinate is a separate
// bound in the source algorithm, and packing them into a temporary would make
// the parity correspondence harder to audit.
#[allow(clippy::too_many_arguments)]
fn push_pane_box(
    out: &mut Vec<Aabb>,
    pos: BlockPos,
    mask: &Aabb,
    min_x: f64,
    min_y: f64,
    min_z: f64,
    max_x: f64,
    max_y: f64,
    max_z: f64,
) {
    // The six values are Java float bounds widened to double by Block.setBlockBounds.  All pane
    // constants are exact sixteenths, so f64 preserves the stored values and Java's subsequent
    // integer-plus-double world offset.
    let local = mc_data::Aabb::new(min_x, min_y, min_z, max_x, max_y, max_z);
    let world_box = to_world_aabb(local, pos);
    if mask.intersects_with(&world_box) {
        out.push(world_box);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::BlockId;

    #[test]
    fn full_cube_offsets_to_the_unit_box_at_its_block_pos() {
        let stone = BlockState::new(BlockId(1), 0);
        let pos = BlockPos::new(5, 10, -3);
        let boxes: Vec<_> = collision_boxes_at(stone, pos).collect();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].min, DVec3::new(5.0, 10.0, -3.0));
        assert_eq!(boxes[0].max, DVec3::new(6.0, 11.0, -2.0));
    }

    #[test]
    fn air_has_no_collision_boxes() {
        let boxes: Vec<_> = collision_boxes_at(BlockState::AIR, BlockPos::ORIGIN).collect();
        assert!(boxes.is_empty());
    }

    #[test]
    fn neighbour_dependent_flag_survives_the_lookup() {
        let fence = BlockState::new(mc_data::block_by_name("fence").unwrap(), 0);
        assert!(block_props(fence).collision_depends_on_neighbors);
        // The isolated shape is still returned — S13's job, not ours, to widen it.
        assert!(!collision_boxes_at(fence, BlockPos::ORIGIN)
            .collect::<Vec<_>>()
            .is_empty());
    }

    fn expected_raw_world_box(pos: BlockPos, min: (f64, f64, f64), max: (f64, f64, f64)) -> Aabb {
        let origin = pos.to_vec3();
        Aabb {
            min: origin + DVec3::new(min.0, min.1, min.2),
            max: origin + DVec3::new(max.0, max.1, max.2),
        }
    }

    #[test]
    fn vine_metadata_faces_cover_all_java_masks_without_normalising_raw_fields() {
        // Java's getStateFromMeta packs SOUTH=1, WEST=2, NORTH=4, EAST=8.  The expected table
        // is independently enumerated from BlockVine.setBlockBoundsBasedOnState's assignment
        // order, rather than derived by calling the implementation under test.  `meta=0` is the
        // deliberately inverted no-face/no-support state; every multi-face mask widens to a full
        // cube after the later branch resets the accumulated fields.
        const EXPECTED: [(f64, f64, f64, f64, f64, f64); 16] = [
            (1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.9375, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0625, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 0.0625),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.9375, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        ];
        let pos = BlockPos::new(-7, 64, 11);
        let vine = mc_data::block_by_name("vine").expect("vine block");

        for (meta, &(min_x, min_y, min_z, max_x, max_y, max_z)) in EXPECTED.iter().enumerate() {
            let mut world = World::new();
            world.load_empty_chunk(pos.chunk_pos());
            world.set_block(pos, BlockState::new(vine, meta as u8));
            assert_eq!(
                vine_selection_bounds_at(&world, pos),
                Some(expected_raw_world_box(
                    pos,
                    (min_x, min_y, min_z),
                    (max_x, max_y, max_z),
                )),
                "vine metadata {meta} face mask"
            );
        }
    }

    #[test]
    fn vine_meta_zero_reads_live_ceiling_support_and_bites_normalising_mutant() {
        let pos = BlockPos::new(-3, 64, 5);
        let vine = mc_data::block_by_name("vine").expect("vine block");
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        world.set_block(pos, BlockState::new(vine, 0));

        // With no support, Java leaves all six mutable fields inverted.  A normalising
        // Aabb::new mutant would turn this into a unit cube, so assert both the raw ordering and
        // that the two representations are observably different.
        let unattached = vine_selection_bounds_at(&world, pos).expect("vine source seam");
        assert!(
            unattached.min.x > unattached.max.x
                && unattached.min.y > unattached.max.y
                && unattached.min.z > unattached.max.z,
            "metadata-0 vine without support must retain Java's inverted corners: {unattached:?}"
        );
        let normalised_mutant = Aabb::new(unattached.min, unattached.max);
        assert_ne!(
            unattached, normalised_mutant,
            "normalising the raw vine corners is a biting mutant"
        );

        // The fallback uses full_cube && blocks_movement, not the extracted normal_cube bit.  A
        // redstone block is a useful control because it is full and movement-blocking while
        // canProvidePower makes Block.isNormalCube() false in 1.8.9.
        let redstone = mc_data::block_by_name("redstone_block").expect("redstone block");
        let redstone_state = BlockState::new(redstone, 0);
        assert!(redstone_state.props().full_cube);
        assert!(redstone_state.props().material.blocks_movement());
        assert!(!redstone_state.props().normal_cube);
        world.set_block(pos.up(), redstone_state);
        assert_eq!(
            vine_selection_bounds_at(&world, pos),
            Some(expected_raw_world_box(
                pos,
                (0.0, 0.9375, 0.0),
                (1.0, 1.0, 1.0),
            )),
            "a supported meta-0 vine is the 1/16 ceiling slab"
        );

        // Removing the support must refresh the live source back to the inverted empty shape;
        // this catches a cached-UP or cached-bounds implementation.
        world.set_block(pos.up(), BlockState::AIR);
        let refreshed = vine_selection_bounds_at(&world, pos).expect("vine source seam");
        assert!(refreshed.min.y > refreshed.max.y);

        // The ceiling fallback is gated by *no* horizontal face.  A stored SOUTH face remains a
        // wall sheet even when a solid block is above it.
        world.set_block(pos, BlockState::new(vine, 1));
        world.set_block(pos.up(), redstone_state);
        assert_eq!(
            vine_selection_bounds_at(&world, pos),
            Some(expected_raw_world_box(
                pos,
                (0.0, 0.0, 0.9375),
                (1.0, 1.0, 1.0),
            )),
            "a stored face wins over the ceiling fallback"
        );
    }

    #[test]
    fn vine_selection_source_rejects_non_vine_centres() {
        let pos = BlockPos::new(2, 64, -4);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        world.set_block(
            pos,
            BlockState::new(mc_data::block_by_name("stone").expect("stone block"), 0),
        );
        assert!(vine_selection_bounds_at(&world, pos).is_none());
    }

    fn pane_world(mask: u8, centre_name: &str, neighbour_name: &str) -> (World, BlockPos) {
        let centre = BlockPos::new(0, 64, 0);
        let mut world = World::new();
        load_fixture_columns(&mut world, centre);
        let centre_id = mc_data::block_by_name(centre_name)
            .unwrap_or_else(|| panic!("no centre block {centre_name}"));
        let neighbour_id = mc_data::block_by_name(neighbour_name)
            .unwrap_or_else(|| panic!("no neighbour block {neighbour_name}"));
        world.set_block(centre, BlockState::new(centre_id, 0));
        let neighbour = BlockState::new(neighbour_id, 0);

        // The mask is N=1, E=2, S=4, W=8.  Keeping the fixture in world coordinates rather
        // than passing four booleans exercises the same live World::block_at seam a caller uses.
        if mask & 1 != 0 {
            world.set_block(centre.north(), neighbour);
        }
        if mask & 2 != 0 {
            world.set_block(centre.east(), neighbour);
        }
        if mask & 4 != 0 {
            world.set_block(centre.south(), neighbour);
        }
        if mask & 8 != 0 {
            world.set_block(centre.west(), neighbour);
        }
        (world, centre)
    }

    fn load_fixture_columns(world: &mut World, centre: BlockPos) {
        for position in [
            centre,
            centre.north(),
            centre.east(),
            centre.south(),
            centre.west(),
        ] {
            world.load_empty_chunk(position.chunk_pos());
        }
    }

    fn fence_world(mask: u8, centre_name: &str, neighbour_name: &str) -> (World, BlockPos) {
        let centre = BlockPos::new(0, 64, 0);
        let mut world = World::new();
        load_fixture_columns(&mut world, centre);
        let centre_id = mc_data::block_by_name(centre_name)
            .unwrap_or_else(|| panic!("no centre block {centre_name}"));
        let neighbour_id = mc_data::block_by_name(neighbour_name)
            .unwrap_or_else(|| panic!("no neighbour block {neighbour_name}"));
        world.set_block(centre, BlockState::new(centre_id, 0));
        let neighbour = BlockState::new(neighbour_id, 0);

        // The mask is N=1, E=2, S=4, W=8, matching BlockFence's four live calls. Keeping the
        // neighbours in World makes the fixture exercise the same source seam as a real caller.
        if mask & 1 != 0 {
            world.set_block(centre.north(), neighbour);
        }
        if mask & 2 != 0 {
            world.set_block(centre.east(), neighbour);
        }
        if mask & 4 != 0 {
            world.set_block(centre.south(), neighbour);
        }
        if mask & 8 != 0 {
            world.set_block(centre.west(), neighbour);
        }
        (world, centre)
    }

    fn world_box(min_x: f64, min_z: f64, max_x: f64, max_z: f64) -> Aabb {
        Aabb::from_bounds(min_x, 64.0, min_z, max_x, 65.0, max_z)
    }

    fn wall_world(mask: u8, centre_name: &str, neighbour_name: &str) -> (World, BlockPos) {
        let centre = BlockPos::new(0, 64, 0);
        let mut world = World::new();
        load_fixture_columns(&mut world, centre);
        let centre_id = mc_data::block_by_name(centre_name)
            .unwrap_or_else(|| panic!("no centre block {centre_name}"));
        let neighbour_id = mc_data::block_by_name(neighbour_name)
            .unwrap_or_else(|| panic!("no neighbour block {neighbour_name}"));
        world.set_block(centre, BlockState::new(centre_id, 0));
        let neighbour = BlockState::new(neighbour_id, 0);

        // The mask is N=1, E=2, S=4, W=8, matching the four live calls in BlockWall's
        // setBlockBoundsBasedOnState. Keeping the neighbours in World makes this test exercise
        // the same source seam used by a future raytrace or outline consumer.
        if mask & 1 != 0 {
            world.set_block(centre.north(), neighbour);
        }
        if mask & 2 != 0 {
            world.set_block(centre.east(), neighbour);
        }
        if mask & 4 != 0 {
            world.set_block(centre.south(), neighbour);
        }
        if mask & 8 != 0 {
            world.set_block(centre.west(), neighbour);
        }
        (world, centre)
    }

    fn chest_world(
        mask: u8,
        centre_name: &str,
        neighbour_name: &str,
        centre_meta: u8,
        neighbour_meta: u8,
    ) -> (World, BlockPos) {
        let centre = BlockPos::new(0, 64, 0);
        let mut world = World::new();
        load_fixture_columns(&mut world, centre);
        let centre_id = mc_data::block_by_name(centre_name)
            .unwrap_or_else(|| panic!("no centre block {centre_name}"));
        let neighbour_id = mc_data::block_by_name(neighbour_name)
            .unwrap_or_else(|| panic!("no neighbour block {neighbour_name}"));
        world.set_block(centre, BlockState::new(centre_id, centre_meta));
        let neighbour = BlockState::new(neighbour_id, neighbour_meta);

        // The mask is N=1, E=2, S=4, W=8.  The helper must retain Java's source order when more
        // than one same-type neighbour is present, so this fixture intentionally allows every
        // combination rather than modelling only legal player placement layouts.
        if mask & 1 != 0 {
            world.set_block(centre.north(), neighbour);
        }
        if mask & 2 != 0 {
            world.set_block(centre.east(), neighbour);
        }
        if mask & 4 != 0 {
            world.set_block(centre.south(), neighbour);
        }
        if mask & 8 != 0 {
            world.set_block(centre.west(), neighbour);
        }
        (world, centre)
    }

    fn expected_fence_selection(mask: u8) -> Aabb {
        // `(min_x, min_z, max_x, max_z)`, independently enumerated from
        // BlockFence.setBlockBoundsBasedOnState. The mask is N=1, E=2, S=4, W=8.
        const EXPECTED: [(f64, f64, f64, f64); 16] = [
            (0.375, 0.375, 0.625, 0.625),
            (0.375, 0.0, 0.625, 0.625),
            (0.375, 0.375, 1.0, 0.625),
            (0.375, 0.0, 1.0, 0.625),
            (0.375, 0.375, 0.625, 1.0),
            (0.375, 0.0, 0.625, 1.0),
            (0.375, 0.375, 1.0, 1.0),
            (0.375, 0.0, 1.0, 1.0),
            (0.0, 0.375, 0.625, 0.625),
            (0.0, 0.0, 0.625, 0.625),
            (0.0, 0.375, 1.0, 0.625),
            (0.0, 0.0, 1.0, 0.625),
            (0.0, 0.375, 0.625, 1.0),
            (0.0, 0.0, 0.625, 1.0),
            (0.0, 0.375, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ];
        let (min_x, min_z, max_x, max_z) = EXPECTED[mask as usize];
        world_box(min_x, min_z, max_x, max_z)
    }

    fn expected_chest_selection(mask: u8) -> Aabb {
        // `(min_x, min_z, max_x, max_z)`, independently enumerated from
        // BlockChest.setBlockBoundsBasedOnState's north/south/west/east else-if chain.  N=1,
        // E=2, S=4, W=8; the table therefore also pins precedence for malformed multi-neighbour
        // layouts instead of merely checking the four one-neighbour cases.
        const EXPECTED: [(f64, f64, f64, f64); 16] = [
            (0.0625, 0.0625, 0.9375, 0.9375),
            (0.0625, 0.0, 0.9375, 0.9375),
            (0.0625, 0.0625, 1.0, 0.9375),
            (0.0625, 0.0, 0.9375, 0.9375),
            (0.0625, 0.0625, 0.9375, 1.0),
            (0.0625, 0.0, 0.9375, 0.9375),
            (0.0625, 0.0625, 0.9375, 1.0),
            (0.0625, 0.0, 0.9375, 0.9375),
            (0.0, 0.0625, 0.9375, 0.9375),
            (0.0625, 0.0, 0.9375, 0.9375),
            (0.0, 0.0625, 0.9375, 0.9375),
            (0.0625, 0.0, 0.9375, 0.9375),
            (0.0625, 0.0625, 0.9375, 1.0),
            (0.0625, 0.0, 0.9375, 0.9375),
            (0.0625, 0.0625, 0.9375, 1.0),
            (0.0625, 0.0, 0.9375, 0.9375),
        ];
        let (min_x, min_z, max_x, max_z) = EXPECTED[mask as usize];
        Aabb::from_bounds(min_x, 64.0, min_z, max_x, 64.875, max_z)
    }

    #[test]
    fn chest_selection_bounds_cover_all_masks_and_all_horizontal_facings() {
        // FACING is not read by BlockChest.setBlockBoundsBasedOnState, but all four legal state
        // values must remain usable as centre states.  The neighbour uses a different legal
        // facing to catch an implementation that accidentally compares complete states rather
        // than Java's block-singleton identity.
        for centre_meta in 2..=5u8 {
            for mask in 0..16u8 {
                let (world, centre) = chest_world(mask, "chest", "chest", centre_meta, 5);
                assert_eq!(
                    chest_selection_bounds_at(&world, centre),
                    Some(expected_chest_selection(mask)),
                    "chest bounds for facing meta {centre_meta} and neighbour mask {mask:04b}"
                );
            }
        }

        // A seam probe lies outside the isolated constructor box but inside an east-joined
        // chest.  This is the geometric negative control for the tempting static-bounds mutant;
        // it is also the region where Java's raytrace can otherwise miss the double chest.
        let (world, centre) = chest_world(2, "chest", "chest", 2, 5);
        let joined = chest_selection_bounds_at(&world, centre).expect("chest source seam");
        let isolated = expected_chest_selection(0);
        let seam_probe = Aabb::from_bounds(0.95, 64.1, 0.2, 0.99, 64.7, 0.8);
        assert!(joined.intersects_with(&seam_probe));
        assert!(!isolated.intersects_with(&seam_probe));
    }

    #[test]
    fn chest_selection_bounds_require_the_same_singleton_and_live_legal_neighbour() {
        // Ordinary and trapped chests share a Java class but are separate BlockChest singleton
        // instances (`this` identity in setBlockBoundsBasedOnState).  Ender chests are a
        // different class and must never enter this source seam at all.
        for (centre_name, neighbour_name, connects) in [
            ("chest", "chest", true),
            ("trapped_chest", "trapped_chest", true),
            ("chest", "trapped_chest", false),
            ("trapped_chest", "chest", false),
            ("chest", "ender_chest", false),
            ("chest", "stone", false),
            ("chest", "air", false),
        ] {
            let (world, centre) = chest_world(2, centre_name, neighbour_name, 2, 5);
            assert_eq!(
                chest_selection_bounds_at(&world, centre),
                Some(expected_chest_selection(if connects { 2 } else { 0 })),
                "same-singleton rule for {centre_name} next to {neighbour_name}"
            );
        }

        // Illegal metadata is exposed as air by the World/Chunk lookup, just as
        // ExtendedBlockStorage.get substitutes the air state for an unregistered state.  A raw
        // block-id comparison would incorrectly create a double chest here.
        let (world, centre) = chest_world(2, "chest", "chest", 2, 0);
        assert_eq!(
            world.block_at(centre.east()),
            BlockState::AIR,
            "an illegal chest metadata neighbour must resolve to air"
        );
        assert_eq!(
            chest_selection_bounds_at(&world, centre),
            Some(expected_chest_selection(0))
        );

        // A diagonal same-type block is not one of BlockChest's four source checks.
        let (mut world, centre) = chest_world(0, "chest", "chest", 2, 5);
        world.set_block(
            BlockPos::new(centre.x + 1, centre.y, centre.z + 1),
            BlockState::new(mc_data::block_by_name("chest").unwrap(), 5),
        );
        assert_eq!(
            chest_selection_bounds_at(&world, centre),
            Some(expected_chest_selection(0)),
            "diagonal neighbours do not widen chest bounds"
        );

        // The source reads the live world each call.  Removing the east half must immediately
        // retract the source box rather than retain a cached double-chest side.
        let (mut world, centre) = chest_world(2, "chest", "chest", 2, 5);
        assert_eq!(
            chest_selection_bounds_at(&world, centre),
            Some(expected_chest_selection(2))
        );
        world.set_block(centre.east(), BlockState::AIR);
        assert_eq!(
            chest_selection_bounds_at(&world, centre),
            Some(expected_chest_selection(0)),
            "removing the live neighbour retracts the east extension"
        );
    }

    #[test]
    fn chest_selection_source_rejects_non_chest_centres() {
        let (world, centre) = chest_world(2, "stone", "chest", 0, 5);
        assert!(chest_selection_bounds_at(&world, centre).is_none());

        let (world, centre) = chest_world(2, "ender_chest", "ender_chest", 0, 0);
        assert!(chest_selection_bounds_at(&world, centre).is_none());
    }

    #[test]
    fn fence_selection_bounds_cover_all_sixteen_neighbour_masks() {
        for mask in 0..16u8 {
            let (world, centre) = fence_world(mask, "fence", "fence");
            assert_eq!(
                fence_selection_bounds_at(&world, centre),
                Some(expected_fence_selection(mask)),
                "selection shape for fence neighbour mask {mask:04b}"
            );
        }

        let (world, centre) = fence_world(2, "stone", "fence");
        assert!(
            fence_selection_bounds_at(&world, centre).is_none(),
            "non-fence centres must not claim the fence source seam"
        );
    }

    #[test]
    fn fence_connection_predicate_matches_java_identity_material_and_block_rules() {
        // Java's canConnectTo is an identity/material predicate, not simply "has a fence-like
        // collision box". All cases below use the east neighbour so the expected envelope differs
        // visibly from the isolated post when a connection is accepted.
        for (neighbour, connects) in [
            ("spruce_fence", true),
            ("nether_brick_fence", false),
            ("fence_gate", true),
            ("stone", true),
            ("pumpkin", false),
            ("barrier", false),
            ("glass", false),
        ] {
            let (world, centre) = fence_world(2, "fence", neighbour);
            assert_eq!(
                fence_selection_bounds_at(&world, centre),
                Some(expected_fence_selection(if connects { 2 } else { 0 })),
                "connection predicate for {neighbour}"
            );
        }
    }

    #[test]
    fn fence_selection_bounds_refresh_when_a_connected_neighbour_is_removed() {
        let (mut world, centre) = fence_world(2, "fence", "fence");
        assert_eq!(
            fence_selection_bounds_at(&world, centre),
            Some(expected_fence_selection(2))
        );
        world.set_block(centre.east(), BlockState::AIR);
        assert_eq!(
            fence_selection_bounds_at(&world, centre),
            Some(expected_fence_selection(0)),
            "the live world read must not retain a stale east arm"
        );
    }

    fn expected_wall_collision(mask: u8) -> Aabb {
        // `(min_x, min_z, max_x, max_z, max_y)`, independently enumerated from
        // BlockWall.getCollisionBoundingBox. Unlike selection bounds, a straight run keeps
        // maxY=1.5 after Java overwrites the bounds field.
        const EXPECTED: [(f64, f64, f64, f64, f64); 16] = [
            (0.25, 0.25, 0.75, 0.75, 1.5),
            (0.25, 0.0, 0.75, 0.75, 1.5),
            (0.25, 0.25, 1.0, 0.75, 1.5),
            (0.25, 0.0, 1.0, 0.75, 1.5),
            (0.25, 0.25, 0.75, 1.0, 1.5),
            (0.3125, 0.0, 0.6875, 1.0, 1.5),
            (0.25, 0.25, 1.0, 1.0, 1.5),
            (0.25, 0.0, 1.0, 1.0, 1.5),
            (0.0, 0.25, 0.75, 0.75, 1.5),
            (0.0, 0.0, 0.75, 0.75, 1.5),
            (0.0, 0.3125, 1.0, 0.6875, 1.5),
            (0.0, 0.0, 1.0, 0.75, 1.5),
            (0.0, 0.25, 0.75, 1.0, 1.5),
            (0.0, 0.0, 0.75, 1.0, 1.5),
            (0.0, 0.25, 1.0, 1.0, 1.5),
            (0.0, 0.0, 1.0, 1.0, 1.5),
        ];
        let (min_x, min_z, max_x, max_z, max_y) = EXPECTED[mask as usize];
        Aabb::from_bounds(min_x, 64.0, min_z, max_x, 64.0 + max_y, max_z)
    }

    fn expected_wall_selection(mask: u8) -> Aabb {
        // `(min_x, min_z, max_x, max_z, max_y)`, independently enumerated from
        // BlockWall.setBlockBoundsBasedOnState. The two straight runs are the only masks whose
        // envelope is lower and narrower than the ordinary connected box.
        const EXPECTED: [(f64, f64, f64, f64, f64); 16] = [
            (0.25, 0.25, 0.75, 0.75, 1.0),
            (0.25, 0.0, 0.75, 0.75, 1.0),
            (0.25, 0.25, 1.0, 0.75, 1.0),
            (0.25, 0.0, 1.0, 0.75, 1.0),
            (0.25, 0.25, 0.75, 1.0, 1.0),
            (0.3125, 0.0, 0.6875, 1.0, 0.8125),
            (0.25, 0.25, 1.0, 1.0, 1.0),
            (0.25, 0.0, 1.0, 1.0, 1.0),
            (0.0, 0.25, 0.75, 0.75, 1.0),
            (0.0, 0.0, 0.75, 0.75, 1.0),
            (0.0, 0.3125, 1.0, 0.6875, 0.8125),
            (0.0, 0.0, 1.0, 0.75, 1.0),
            (0.0, 0.25, 0.75, 1.0, 1.0),
            (0.0, 0.0, 0.75, 1.0, 1.0),
            (0.0, 0.25, 1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0, 1.0),
        ];
        let (min_x, min_z, max_x, max_z, max_y) = EXPECTED[mask as usize];
        Aabb::from_bounds(min_x, 64.0, min_z, max_x, 64.0 + max_y, max_z)
    }

    #[test]
    fn wall_collision_and_selection_bounds_cover_all_sixteen_neighbour_masks() {
        let full_mask = Aabb::from_bounds(-1.0, 63.0, -1.0, 2.0, 67.0, 2.0);
        for mask in 0..16u8 {
            let (world, centre) = wall_world(mask, "cobblestone_wall", "cobblestone_wall");
            assert_eq!(
                wall_collision_boxes_at(&world, centre, &full_mask),
                Some(vec![expected_wall_collision(mask)]),
                "collision shape for wall neighbour mask {mask:04b}"
            );
            assert_eq!(
                wall_selection_bounds_at(&world, centre),
                Some(expected_wall_selection(mask)),
                "selection shape for wall neighbour mask {mask:04b}"
            );
        }

        // The source helper retains Java's strict intersectsWith filter: a valid wall can still
        // contribute no collision box when the caller's query mask is disjoint.
        let (world, centre) = wall_world(2, "cobblestone_wall", "cobblestone_wall");
        let disjoint = Aabb::from_bounds(3.0, 64.0, 3.0, 4.0, 65.0, 4.0);
        assert_eq!(
            wall_collision_boxes_at(&world, centre, &disjoint),
            Some(Vec::new())
        );
    }

    #[test]
    fn wall_connection_predicate_matches_java_identity_material_and_barrier_rules() {
        let (world, centre) = wall_world(2, "cobblestone_wall", "stone");
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(2)),
            "opaque full cubes connect"
        );

        let (world, centre) = wall_world(2, "cobblestone_wall", "fence_gate");
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(2)),
            "all BlockFenceGate instances connect"
        );

        let (world, centre) = wall_world(2, "cobblestone_wall", "cobblestone_wall");
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(2)),
            "the same wall singleton connects"
        );

        let (world, centre) = wall_world(2, "cobblestone_wall", "fence");
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(0)),
            "a non-opaque fence is not a wall connection"
        );

        let (world, centre) = wall_world(2, "cobblestone_wall", "pumpkin");
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(0)),
            "opaque gourd blocks are explicitly excluded"
        );

        let (world, centre) = wall_world(2, "cobblestone_wall", "barrier");
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(0)),
            "barriers are explicitly excluded before the full-cube test"
        );

        // java: the test is `isFullCube()`, not `isFullBlock()`. These three are full cubes of
        // an opaque material without being opaque cubes, so a wall joins them (the 1.8.9
        // bytecode oracle in mc-render's tests/actual_state_java_oracle.rs found the slime case).
        for neighbour in ["slime", "mob_spawner", "end_portal_frame"] {
            let (world, centre) = wall_world(2, "cobblestone_wall", neighbour);
            assert_eq!(
                wall_selection_bounds_at(&world, centre),
                Some(expected_wall_selection(2)),
                "a wall joins {neighbour}"
            );
        }

        let (world, centre) = wall_world(2, "stone", "cobblestone_wall");
        assert!(wall_collision_boxes_at(
            &world,
            centre,
            &Aabb::from_bounds(-1.0, 63.0, -1.0, 2.0, 67.0, 2.0)
        )
        .is_none());
        assert!(wall_selection_bounds_at(&world, centre).is_none());
    }

    #[test]
    fn wall_bounds_refresh_when_a_connected_neighbour_is_removed() {
        let (mut world, centre) = wall_world(2, "cobblestone_wall", "cobblestone_wall");
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(2))
        );
        world.set_block(centre.east(), BlockState::AIR);
        assert_eq!(
            wall_selection_bounds_at(&world, centre),
            Some(expected_wall_selection(0)),
            "the live world read must not retain a stale east arm"
        );
    }

    /// Independent expected table for Java's two collision passes.  The first tuple in each
    /// row is the X-axis pass and the second (when present) is the Z-axis pass.  Listing all 16
    /// masks catches the tempting but wrong "always return the isolated cross" implementation.
    fn expected_pane_collision(mask: u8) -> Vec<Aabb> {
        let local = match mask {
            0 => vec![(0.0, 0.4375, 1.0, 0.5625), (0.4375, 0.0, 0.5625, 1.0)],
            1 => vec![(0.4375, 0.0, 0.5625, 0.5)],
            2 => vec![(0.5, 0.4375, 1.0, 0.5625)],
            3 => vec![(0.5, 0.4375, 1.0, 0.5625), (0.4375, 0.0, 0.5625, 0.5)],
            4 => vec![(0.4375, 0.5, 0.5625, 1.0)],
            5 => vec![(0.4375, 0.0, 0.5625, 1.0)],
            6 => vec![(0.5, 0.4375, 1.0, 0.5625), (0.4375, 0.5, 0.5625, 1.0)],
            7 => vec![(0.5, 0.4375, 1.0, 0.5625), (0.4375, 0.0, 0.5625, 1.0)],
            8 => vec![(0.0, 0.4375, 0.5, 0.5625)],
            9 => vec![(0.0, 0.4375, 0.5, 0.5625), (0.4375, 0.0, 0.5625, 0.5)],
            10 => vec![(0.0, 0.4375, 1.0, 0.5625)],
            11 => vec![(0.0, 0.4375, 1.0, 0.5625), (0.4375, 0.0, 0.5625, 0.5)],
            12 => vec![(0.0, 0.4375, 0.5, 0.5625), (0.4375, 0.5, 0.5625, 1.0)],
            13 => vec![(0.0, 0.4375, 0.5, 0.5625), (0.4375, 0.0, 0.5625, 1.0)],
            14 => vec![(0.0, 0.4375, 1.0, 0.5625), (0.4375, 0.5, 0.5625, 1.0)],
            15 => vec![(0.0, 0.4375, 1.0, 0.5625), (0.4375, 0.0, 0.5625, 1.0)],
            _ => unreachable!("pane mask is four bits"),
        };
        local
            .into_iter()
            .map(|(min_x, min_z, max_x, max_z)| world_box(min_x, min_z, max_x, max_z))
            .collect()
    }

    fn expected_pane_selection(mask: u8) -> Aabb {
        // `(min_x, max_x, min_z, max_z)`, independently enumerated from
        // BlockPane.setBlockBoundsBasedOnState's two axis branches.
        const EXPECTED: [(f64, f64, f64, f64); 16] = [
            (0.0, 1.0, 0.0, 1.0),
            (0.4375, 0.5625, 0.0, 0.5625),
            (0.4375, 1.0, 0.4375, 0.5625),
            (0.4375, 1.0, 0.0, 0.5625),
            (0.4375, 0.5625, 0.4375, 1.0),
            (0.4375, 0.5625, 0.0, 1.0),
            (0.4375, 1.0, 0.4375, 1.0),
            (0.4375, 1.0, 0.0, 1.0),
            (0.0, 0.5625, 0.4375, 0.5625),
            (0.0, 0.5625, 0.0, 0.5625),
            (0.0, 1.0, 0.4375, 0.5625),
            (0.0, 1.0, 0.0, 0.5625),
            (0.0, 0.5625, 0.4375, 1.0),
            (0.0, 0.5625, 0.0, 1.0),
            (0.0, 1.0, 0.4375, 1.0),
            (0.0, 1.0, 0.0, 1.0),
        ];
        let (min_x, max_x, min_z, max_z) = EXPECTED[mask as usize];
        world_box(min_x, min_z, max_x, max_z)
    }

    #[test]
    fn pane_collision_and_selection_bounds_cover_all_sixteen_neighbour_masks() {
        let full_mask = Aabb::from_bounds(-1.0, 63.0, -1.0, 2.0, 66.0, 2.0);
        for mask in 0..16u8 {
            let (world, centre) = pane_world(mask, "glass_pane", "glass_pane");
            assert_eq!(
                pane_collision_boxes_at(&world, centre, &full_mask),
                Some(expected_pane_collision(mask)),
                "collision shape for pane neighbour mask {mask:04b}"
            );
            assert_eq!(
                pane_selection_bounds_at(&world, centre),
                Some(expected_pane_selection(mask)),
                "selection shape for pane neighbour mask {mask:04b}"
            );
        }

        // The filter is part of Java's `super.addCollisionBoxesToList` contract: a valid pane
        // can still contribute no boxes when the query mask is disjoint.
        let (world, centre) = pane_world(1, "glass_pane", "glass_pane");
        let disjoint = Aabb::from_bounds(3.0, 64.0, 3.0, 4.0, 65.0, 4.0);
        assert_eq!(
            pane_collision_boxes_at(&world, centre, &disjoint),
            Some(Vec::new())
        );
    }

    #[test]
    fn pane_connection_predicate_matches_java_full_block_and_pane_identity_rules() {
        let (world, centre) = pane_world(2, "glass_pane", "stone");
        assert!(world.block_at(centre.east()).props().full_block);
        assert_eq!(
            pane_collision_boxes_at(
                &world,
                centre,
                &Aabb::from_bounds(-1.0, 63.0, -1.0, 2.0, 66.0, 2.0)
            ),
            Some(expected_pane_collision(2)),
            "a full block connects even when it is not a pane"
        );

        let (world, centre) = pane_world(2, "glass_pane", "fence");
        assert!(!world.block_at(centre.east()).props().full_block);
        assert_eq!(
            pane_collision_boxes_at(
                &world,
                centre,
                &Aabb::from_bounds(-1.0, 63.0, -1.0, 2.0, 66.0, 2.0)
            ),
            Some(expected_pane_collision(0)),
            "an opaque-looking but non-full fence must not connect"
        );

        let (world, centre) = pane_world(2, "stained_glass_pane", "glass_pane");
        assert_eq!(
            pane_selection_bounds_at(&world, centre),
            Some(expected_pane_selection(2)),
            "BlockStainedGlassPane and BlockPane connect through instanceof"
        );

        let (world, centre) = pane_world(2, "stone", "glass_pane");
        assert!(pane_collision_boxes_at(
            &world,
            centre,
            &Aabb::from_bounds(-1.0, 63.0, -1.0, 2.0, 66.0, 2.0)
        )
        .is_none());
        assert!(pane_selection_bounds_at(&world, centre).is_none());
    }
}
