//! Gathering the world collision boxes an entity box sweeps against.
//!
//! java: `World.getCollidingBoundingBoxes` (`World.java:1262`), plus the
//! `addCollisionBoxesToList`/`getCollisionBoundingBox` overrides of the seven block classes
//! whose collision shape reads its neighbours.
//!
//! # Why this is not just [`mc_world::collision_boxes_at`]
//!
//! S01 extracted collision shapes by *calling* `Block.addCollisionBoxesToList` on an isolated
//! block, so the table is exact for every block whose shape comes from its own metadata — the
//! overwhelming majority. For 195 states across 7 classes it is only the isolated case, and S01
//! flags those with `BlockProps::collision_depends_on_neighbors`. This module recomputes them.
//!
//! One of the seven is a false positive for **movement collision** worth writing down.
//! `BlockChest` overrides `setBlockBoundsBasedOnState` to widen the mutable bounds used by the
//! separate selection/raytrace path, but it does not widen the movement-collision boxes because
//! neither `addCollisionBoxesToList` nor `getCollisionBoundingBox`, and
//! `World.getCollidingBoundingBoxes` never calls `setBlockBoundsBasedOnState` — so a chest
//! collides with whatever is left in `Block`'s mutable `minX..maxZ` singleton fields, which on a
//! server that has not raytraced anything is the constructor's
//! `(0.0625, 0.0, 0.0625)..(0.9375, 0.875, 0.9375)`. That is exactly what S01's table holds, so
//! chests fall through to the isolated shape deliberately, not by omission.
//!
//! **Latched shapes (2026-11-09).** The `BoxRefGen` oracle proves that exactly two
//! classes collide with the singleton's *latched* bounds: chests/trapped chests and anvils. On a
//! vanilla client those fields are rewritten by the chunk mesher
//! (`BlockModelRenderer.renderModel:26` calls `setBlockBoundsBasedOnState` for every meshed
//! anvil) and by the mouse-over ray (`Block.collisionRayTrace`), so the real shape depends on
//! what was last meshed or looked at, and the server's copy latches independently. mc-physics
//! keeps the deterministic model: a chest collides as a single chest, an anvil with its own
//! facing's bounds (what meshing or looking at that anvil leaves). Every other block's shape is
//! a pure function of the world, and `tests/block_boxes.rs` checks all of them bit-for-bit.

use mc_data::{block_info, block_props, BlockId, BlockState, Material};
use mc_math::{floor_f64, Aabb, BlockPos, ChunkPos, DVec3};
use mc_world::{Chunk, World};

/// The relationship of a supplied entity to the entity currently moving.
///
/// `World.getCollidingBoundingBoxes` only asks the world to exclude the mover itself. It does
/// *not* discard a vehicle or passenger here; riding skips the normal local-player movement path
/// before this method is reached. Keeping the relation in the provider value makes that otherwise
/// surprising rule explicit without coupling physics to the client's ECS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EntityCollisionRelation {
    /// An unrelated nearby entity.
    #[default]
    Other,
    /// The local mover. This is always excluded, matching `...ExcludingEntity`.
    SelfEntity,
    /// The entity the mover is riding. It is not filtered by this world method.
    Riding,
    /// An entity riding the mover. It is not filtered by this world method.
    RiddenBy,
}

/// One entity candidate supplied by the client/tracker to a collision sweep.
///
/// `query_box` is that entity's ordinary bounding box, used by
/// `getEntitiesWithinAABBExcludingEntity` to select nearby entities. `collision_box` is the
/// result of that entity's `getCollisionBoundingBox()` and `mover_collision_box` is the result
/// of the mover's `getCollisionBox(other)`. Both may be `None`: an entity can be near enough to
/// query but contribute no obstacle. This deliberately does **not** use `canBeCollidedWith()`;
/// that predicate is for interaction/ray picking, not `World.getCollidingBoundingBoxes`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityCollisionCandidate {
    /// The other entity's normal box, used for the expanded nearby-entity query.
    pub query_box: Aabb,
    /// `other.getCollisionBoundingBox()`; `None` means this side contributes no obstacle.
    pub collision_box: Option<Aabb>,
    /// `mover.getCollisionBox(other)`; normally `None` for a player, but part of the Java API.
    pub mover_collision_box: Option<Aabb>,
    /// Whether the candidate is self, a rider, or a vehicle.
    pub relation: EntityCollisionRelation,
    /// A caller-side exclusion (dead/despawned/not in the tracker query) which wins over boxes.
    pub excluded: bool,
}

impl EntityCollisionCandidate {
    /// Builds an ordinary nearby candidate with an optional collision bounding box.
    #[must_use]
    pub const fn new(query_box: Aabb, collision_box: Option<Aabb>) -> Self {
        Self {
            query_box,
            collision_box,
            mover_collision_box: None,
            relation: EntityCollisionRelation::Other,
            excluded: false,
        }
    }

    /// Adds the mover-owned collision box returned by `Entity.getCollisionBox(other)`.
    #[must_use]
    pub const fn with_mover_collision_box(mut self, collision_box: Option<Aabb>) -> Self {
        self.mover_collision_box = collision_box;
        self
    }

    /// Records the relation without changing World.java's entity-collision eligibility.
    #[must_use]
    pub const fn with_relation(mut self, relation: EntityCollisionRelation) -> Self {
        self.relation = relation;
        self
    }

    /// Marks a candidate absent from the caller's `...ExcludingEntity` query result.
    #[must_use]
    pub const fn excluded(mut self) -> Self {
        self.excluded = true;
        self
    }
}

/// Java class names whose collision shape depends on neighbouring blocks.
mod class {
    pub const FENCE: &str = "net.minecraft.block.BlockFence";
    pub const WALL: &str = "net.minecraft.block.BlockWall";
    pub const PANE: &str = "net.minecraft.block.BlockPane";
    pub const STAINED_PANE: &str = "net.minecraft.block.BlockStainedGlassPane";
    pub const STAIRS: &str = "net.minecraft.block.BlockStairs";
    pub const DOOR: &str = "net.minecraft.block.BlockDoor";
    pub const FENCE_GATE: &str = "net.minecraft.block.BlockFenceGate";
    pub const BARRIER: &str = "net.minecraft.block.BlockBarrier";
}

/// The collision-facing projection of a client world border.
///
/// `WorldBorder` itself belongs to the client environment and cannot be stored in
/// `mc_world::World` without coupling packet state to world storage. The client projects its
/// current wall-clock diameter into this value, then supplies it to the world query. The
/// `entity_inside_border` argument on [`collision_boxes_in_with_border`] is the Java
/// `World.isInsideBorder`/`Entity.outsideBorder` hysteresis decision made by that caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionBorder {
    /// Border centre X.
    pub center_x: f64,
    /// Border centre Z.
    pub center_z: f64,
    /// Current interpolated diameter.
    pub diameter: f64,
    /// Java `WorldBorder.worldSize` clamp.
    pub world_size: i32,
}

impl CollisionBorder {
    /// Builds a collision projection from the current border geometry.
    #[must_use]
    pub const fn new(center_x: f64, center_z: f64, diameter: f64, world_size: i32) -> Self {
        Self {
            center_x,
            center_z,
            diameter,
            world_size,
        }
    }

    fn min_x(self) -> f64 {
        (self.center_x - self.diameter / 2.0).max(-f64::from(self.world_size))
    }

    fn max_x(self) -> f64 {
        (self.center_x + self.diameter / 2.0).min(f64::from(self.world_size))
    }

    fn min_z(self) -> f64 {
        (self.center_z - self.diameter / 2.0).max(-f64::from(self.world_size))
    }

    fn max_z(self) -> f64 {
        (self.center_z + self.diameter / 2.0).min(f64::from(self.world_size))
    }

    /// Whether an entity position is inside the border's hysteresis region.
    ///
    /// java: `World.isInsideBorder(WorldBorder, Entity)` (`World.java:1335`). Vanilla expands
    /// the allowed rectangle by one block while an entity is currently inside, and contracts it
    /// by one block while the entity is already outside. That asymmetric test is what prevents a
    /// player walking along the wall from toggling the physical stone substitution every tick.
    /// The caller owns the entity's latched `outsideBorder` bit and must pass the same bit back
    /// on the next collision query.
    #[must_use]
    pub fn entity_inside(self, x: f64, z: f64, outside_border: bool) -> bool {
        let (mut min_x, mut min_z, mut max_x, mut max_z) =
            (self.min_x(), self.min_z(), self.max_x(), self.max_z());
        if outside_border {
            min_x += 1.0;
            min_z += 1.0;
            max_x -= 1.0;
            max_z -= 1.0;
        } else {
            min_x -= 1.0;
            min_z -= 1.0;
            max_x += 1.0;
            max_z += 1.0;
        }
        x > min_x && x < max_x && z > min_z && z < max_z
    }

    /// Updates the caller-owned `outsideBorder` latch at a collision-query boundary.
    ///
    /// java: the two assignments in `World.getCollidingBoundingBoxes` (`World.java:1288-1295`):
    /// an inside entity latches outside only after leaving the expanded rectangle, while an
    /// outside entity latches back inside only after entering the contracted rectangle.
    #[must_use]
    pub fn update_entity_outside(self, x: f64, z: f64, outside_border: bool) -> bool {
        let inside = self.entity_inside(x, z, outside_border);
        if outside_border && inside {
            false
        } else if !outside_border && !inside {
            true
        } else {
            outside_border
        }
    }

    /// Whether a block has any portion inside the border.
    ///
    /// java: `WorldBorder.contains(BlockPos)` uses strict comparisons and widens the minimum
    /// coordinate by one block before testing it.
    #[must_use]
    pub fn contains_block(self, pos: BlockPos) -> bool {
        f64::from(pos.x.wrapping_add(1)) > self.min_x()
            && f64::from(pos.x) < self.max_x()
            && f64::from(pos.z.wrapping_add(1)) > self.min_z()
            && f64::from(pos.z) < self.max_z()
    }
}

/// Every collision box overlapping `mask`, in vanilla's iteration order.
///
/// java: `World.getCollidingBoundingBoxes(entity, bb)`. Note the scan range — `x` and `z` run
/// `floor(min)..floor(max + 1.0)` but `y` starts one block **lower**, at `floor(minY) - 1`,
/// which is what lets a 1.5-tall fence under your feet still be found.
///
/// This compatibility wrapper omits entity boxes. Use [`collision_boxes_in_with_entities`] when
/// a caller has entity-tracker candidates.
pub fn collision_boxes_in(w: &World, mask: &Aabb, out: &mut Vec<Aabb>) {
    collision_boxes_in_impl(w, mask, None, None, out);
}

/// [`collision_boxes_in`] with Java's physical world-border substitution.
///
/// When `entity_inside_border` is true, every scanned block position outside `border` is
/// queried as a full stone cube, regardless of the real world block. This is the invisible wall
/// in `World.getCollidingBoundingBoxes`; when the entity is outside the hysteresis region the
/// real blocks are queried instead. The caller owns the entity's latched outside-border state,
/// because this crate does not own client world-border packet state.
pub fn collision_boxes_in_with_border(
    w: &World,
    mask: &Aabb,
    border: CollisionBorder,
    entity_inside_border: bool,
    out: &mut Vec<Aabb>,
) {
    collision_boxes_in_impl(w, mask, Some((border, entity_inside_border)), None, out);
}

/// [`collision_boxes_in_with_entities`] with Java's physical world-border substitution and the
/// caller-owned `Entity.outsideBorder` latch.
///
/// `entity_x`/`entity_z` are the entity position fields, not the centre reconstructed from its
/// AABB. Vanilla computes `World.isInsideBorder` once at the beginning of each
/// `getCollidingBoundingBoxes` call, then updates the latch only when at least one scanned column
/// is loaded. The movement caller therefore passes the same position and mutable latch to every
/// sneak probe, primary sweep, and step-up query.
// This is the public parity seam for the Java border/entity query. Keeping each Java input
// explicit makes the caller-owned `outside_border` latch and entity position impossible to
// accidentally conflate; wrapping them would add churn at every movement call site.
#[allow(clippy::too_many_arguments)]
pub fn collision_boxes_in_with_border_and_entities(
    w: &World,
    mask: &Aabb,
    border: CollisionBorder,
    entity_x: f64,
    entity_z: f64,
    outside_border: &mut bool,
    entities: &[EntityCollisionCandidate],
    out: &mut Vec<Aabb>,
) {
    let entity_inside_border = border.entity_inside(entity_x, entity_z, *outside_border);
    collision_boxes_in_impl(
        w,
        mask,
        Some((border, entity_inside_border)),
        Some((entity_x, entity_z, outside_border)),
        out,
    );
    append_entity_collision_boxes(mask, entities, out);
}

/// java: `World.isValid`'s horizontal bound, `-30000000 <= x < 30000000` (same for z).
const WORLD_LIMIT: i32 = 30_000_000;

/// Whether the scan range `a0..a1` contains at least one `World.isValid` coordinate.
fn valid_span(a0: i32, a1: i32) -> bool {
    a0 < a1 && a0 < WORLD_LIMIT && a1 > -WORLD_LIMIT
}

/// java: `Blocks.stone.getDefaultState()`, the world border's substitute cube.
///
/// A constant because `block_by_name("stone")` formatted `minecraft:stone` and binary-searched
/// the registry on **every** collision query (every particle, item, XP orb and movement sweep
/// probe; R2-16/PHYS-1). `stone_constant_is_the_registry_default_state` pins the id.
const STONE: BlockState = BlockState::new(BlockId(1), 0);

/// The last two chunk columns a collision scan resolved.
///
/// The scan used to ask `is_chunk_loaded` per column and then `World::block_at` per cell, each
/// one SipHash of the same `ChunkPos`; a resolved column answers both. Two slots cover a mask
/// that straddles a chunk border on the inner (z) loop. Only lookups are cached, never block
/// data, and the world is borrowed immutably for the whole query.
struct ColumnCache<'w> {
    world: &'w World,
    slots: [Option<(ChunkPos, Option<&'w Chunk>)>; 2],
    next: usize,
}

impl<'w> ColumnCache<'w> {
    fn new(world: &'w World) -> Self {
        Self {
            world,
            slots: [None, None],
            next: 0,
        }
    }

    /// `world.chunk(pos)`; `is_some()` is `world.is_chunk_loaded(pos)`.
    #[inline]
    fn column(&mut self, pos: ChunkPos) -> Option<&'w Chunk> {
        for &(cached, chunk) in self.slots.iter().flatten() {
            if cached == pos {
                return chunk;
            }
        }
        let chunk = self.world.chunk(pos);
        self.slots[self.next] = Some((pos, chunk));
        self.next ^= 1;
        chunk
    }
}

fn collision_boxes_in_impl(
    w: &World,
    mask: &Aabb,
    border: Option<(CollisionBorder, bool)>,
    entity_border: Option<(f64, f64, &mut bool)>,
    out: &mut Vec<Aabb>,
) {
    let x0 = floor_f64(mask.min.x);
    let x1 = floor_f64(mask.max.x + 1.0);
    let y0 = floor_f64(mask.min.y);
    let y1 = floor_f64(mask.max.y + 1.0);
    let z0 = floor_f64(mask.min.z);
    let z1 = floor_f64(mask.max.z + 1.0);
    let mut columns = ColumnCache::new(w);
    // `World::block_at` answers DEBUG_WORLD before consulting the column; keep that path whole.
    let debug_world = w.is_debug_world();

    // java: World.getCollidingBoundingBoxes captures `flag1` before entering the block loops,
    // but the entity's outside-border bit is changed only from inside an `isBlockLoaded`
    // column branch. Preserve both details: the current query's substitution uses the old
    // latch, while the updated latch is visible to the next query in the same movement sweep.
    //
    // On a CLIENT `isBlockLoaded(x, 64, z)` is `World.isValid`
    // alone: `ChunkProviderClient.chunkExists` is a hard-coded `true`
    // (`ChunkProviderClient.java:42`), so every column within +-30,000,000 is scanned whether
    // the server sent it or not. Gating this on resident columns left the latch frozen, and
    // the border's stone missing, wherever the server had sent an all-air chunk (an S21 with
    // bitmask 0 is an unload): exactly the void around a minigame map.
    if let (Some((entity_x, entity_z, outside_border)), Some((border, _))) = (entity_border, border)
    {
        if valid_span(x0, x1) && valid_span(z0, z1) {
            *outside_border = border.update_entity_outside(entity_x, entity_z, *outside_border);
        }
    }

    for x in x0..x1 {
        if !(-WORLD_LIMIT..WORLD_LIMIT).contains(&x) {
            continue;
        }
        for z in z0..z1 {
            // java: World.getCollidingBoundingBoxes:1281, the column guard sampled at y = 64.
            if !(-WORLD_LIMIT..WORLD_LIMIT).contains(&z) {
                continue;
            }
            // `None` is a column the server never sent (or unloaded). Java still scans it: the
            // border substitution below applies, and every other cell is
            // `World.getBlockState`, i.e. `ChunkProviderClient.blankChunk`, which `World::block_at`
            // answers for an absent column.
            let chunk = columns.column(BlockPos::new(x, 64, z).chunk_pos());
            let (local_x, local_z) = ((x & 15) as u8, (z & 15) as u8);
            // java: `for (int i2 = k - 1; i2 < l; ++i2)`: `k - 1` is wrapping `int` arithmetic, so
            // a box whose `floor_double(minY)` is `Integer.MIN_VALUE` (a long-distance particle
            // at an extreme S2A position, say) scans nothing.
            for y in y0.wrapping_sub(1)..y1 {
                let pos = BlockPos::new(x, y, z);
                let substitute = border.is_some_and(|(border, entity_inside)| {
                    entity_inside && !border.contains_block(pos)
                });
                if substitute {
                    for b in block_props(STONE).collision.boxes() {
                        let a = mc_world::to_world_aabb(*b, pos);
                        if mask.intersects_with(&a) {
                            out.push(a);
                        }
                    }
                } else {
                    // `w.block_at(pos)` from the already-resolved column: out-of-world y and
                    // absent sections are air there too.
                    let state = match chunk {
                        _ if debug_world => w.block_at(pos),
                        Some(chunk) if (0..256).contains(&y) => {
                            chunk.block_at(local_x, y as u8, local_z)
                        }
                        Some(_) => BlockState::AIR,
                        None => w.block_at(pos),
                    };
                    // Air has no collision shape, no neighbour dependence and no tile entity
                    // (`air_has_no_collision_boxes` pins it); on void maps it is nearly every
                    // cell.
                    if state != BlockState::AIR {
                        add_block_boxes_for(w, pos, state, mask, out);
                    }
                }
            }
        }
    }
}

/// [`collision_boxes_in`] plus World.java's nearby-entity collision lane.
///
/// java: `World.getCollidingBoundingBoxes:1309-1329`. Candidates are selected through the
/// mover-expanded (`0.25` on every axis) query box, then each non-null collision box is appended
/// only when it strictly intersects the original `mask`. The caller supplies tracker-owned
/// candidates, so physics remains independent of any ECS or network entity representation.
pub fn collision_boxes_in_with_entities(
    w: &World,
    mask: &Aabb,
    entities: &[EntityCollisionCandidate],
    out: &mut Vec<Aabb>,
) {
    collision_boxes_in_impl(w, mask, None, None, out);

    append_entity_collision_boxes(mask, entities, out);
}

pub(crate) fn append_entity_collision_boxes(
    mask: &Aabb,
    entities: &[EntityCollisionCandidate],
    out: &mut Vec<Aabb>,
) {
    let query = mask.expand(0.25, 0.25, 0.25);
    for entity in entities {
        if entity.excluded
            || entity.relation == EntityCollisionRelation::SelfEntity
            || !entity.query_box.intersects_with(&query)
        {
            continue;
        }
        if let Some(collision_box) = entity.collision_box {
            if collision_box.intersects_with(mask) {
                out.push(collision_box);
            }
        }
        if let Some(collision_box) = entity.mover_collision_box {
            if collision_box.intersects_with(mask) {
                out.push(collision_box);
            }
        }
    }
}

/// The collision boxes of the single block at `pos`, neighbour-aware.
///
/// Unlike [`mc_world::collision_boxes_at`] this consults the world for the seven classes that
/// need it. `mask` is only used for vanilla's `intersectsWith` filter.
pub fn collision_boxes_at(w: &World, pos: BlockPos, mask: &Aabb) -> Vec<Aabb> {
    let mut out = Vec::new();
    add_block_boxes(w, pos, mask, &mut out);
    out
}

/// The single bounds box that `BlockFence.setBlockBoundsBasedOnState` leaves in the mutable
/// `Block.minX..maxZ` fields before selection/ray tracing.
///
/// This is deliberately distinct from [`collision_boxes_at`]: a fence's collision arms are
/// `1.5` blocks high (and may be split into two boxes), while its selected/raytrace box is one
/// connected envelope, only one block high.  The renderer and client interaction paths share
/// this oracle so their visible outline and targetable shape cannot drift apart.
///
/// java: `BlockFence.setBlockBoundsBasedOnState` (`BlockFence.java:127-159`).
#[must_use]
pub fn fence_selection_bounds(w: &World, pos: BlockPos, state: BlockState) -> Option<Aabb> {
    if block_info(state.block()).java_class != class::FENCE {
        return None;
    }

    let material = block_props(state).material;
    let north = fence_can_connect(w, pos.north(), material);
    let south = fence_can_connect(w, pos.south(), material);
    let west = fence_can_connect(w, pos.west(), material);
    let east = fence_can_connect(w, pos.east(), material);

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

fn push(out: &mut Vec<Aabb>, mask: &Aabb, pos: BlockPos, b: LocalBox) {
    // java: Block.getCollisionBoundingBox — the block-local bound is widened to f64 *before*
    // being added to the integer coordinate.
    let a = Aabb::new(
        DVec3::new(
            f64::from(pos.x) + b.0,
            f64::from(pos.y) + b.1,
            f64::from(pos.z) + b.2,
        ),
        DVec3::new(
            f64::from(pos.x) + b.3,
            f64::from(pos.y) + b.4,
            f64::from(pos.z) + b.5,
        ),
    );
    if mask.intersects_with(&a) {
        out.push(a);
    }
}

/// java block ids of the two absolute-coordinate `float` shapes below.
const CACTUS_ID: u16 = 81;
const CAKE_ID: u16 = 92;

fn push_box(out: &mut Vec<Aabb>, mask: &Aabb, a: Aabb) {
    if mask.intersects_with(&a) {
        out.push(a);
    }
}

/// java: `BlockCactus.getCollisionBoundingBox` (`BlockCactus.java:63-67`).
///
/// `(double)((float)pos.getX() + f)`: the inset is added in `float` to the *absolute*
/// coordinate. Near the origin that is exact; from |x| = 2^20 the 1/16 inset rounds away and
/// vanilla's cactus is a full-width box there. `minY` is the only plain-double bound.
fn cactus_box(pos: BlockPos, mask: &Aabb, out: &mut Vec<Aabb>) {
    let f = 0.0625f32;
    push_box(
        out,
        mask,
        Aabb::from_bounds(
            f64::from(pos.x as f32 + f),
            f64::from(pos.y),
            f64::from(pos.z as f32 + f),
            f64::from(pos.x.wrapping_add(1) as f32 - f),
            f64::from(pos.y.wrapping_add(1) as f32 - f),
            f64::from(pos.z.wrapping_add(1) as f32 - f),
        ),
    );
}

/// java: `BlockCake.getCollisionBoundingBox` (`BlockCake.java:49-55`): the bitten edge
/// `(1 + bites * 2) / 16F` and the `1/16` insets added in `float` to absolute coordinates.
/// Only legal states (`bites` = meta 0..=6) reach here; the illegal metas resolve to air.
fn cake_box(pos: BlockPos, bites: u8, mask: &Aabb, out: &mut Vec<Aabb>) {
    let f = 0.0625f32;
    let f1 = (1 + i32::from(bites) * 2) as f32 / 16.0;
    let f2 = 0.5f32;
    push_box(
        out,
        mask,
        Aabb::from_bounds(
            f64::from(pos.x as f32 + f1),
            f64::from(pos.y),
            f64::from(pos.z as f32 + f),
            f64::from(pos.x.wrapping_add(1) as f32 - f),
            f64::from(pos.y as f32 + f2),
            f64::from(pos.z.wrapping_add(1) as f32 - f),
        ),
    );
}

/// A block-local box, in the argument order `Block.setBlockBounds` takes.
type LocalBox = (f64, f64, f64, f64, f64, f64);

fn local(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) -> LocalBox {
    // java: setBlockBounds takes floats and stores them as doubles.
    (
        f64::from(min_x),
        f64::from(min_y),
        f64::from(min_z),
        f64::from(max_x),
        f64::from(max_y),
        f64::from(max_z),
    )
}

fn add_block_boxes(w: &World, pos: BlockPos, mask: &Aabb, out: &mut Vec<Aabb>) {
    add_block_boxes_for(w, pos, w.block_at(pos), mask, out);
}

/// [`add_block_boxes`] for a `state` the caller already read as `w.block_at(pos)`.
pub(crate) fn add_block_boxes_for(
    w: &World,
    pos: BlockPos,
    state: BlockState,
    mask: &Aabb,
    out: &mut Vec<Aabb>,
) {
    let props = block_props(state);
    // java: `BlockPistonMoving.getCollisionBoundingBox` (`block/BlockPistonMoving.java:159-177`)
    // is the one shape that depends on a *tile entity* rather than on neighbouring blocks, so
    // S01 could not flag it and the extracted `collision: Empty` is only the no-tile-entity
    // case. That is exactly the state a moving block is *never* in, so before this a player
    // stood on air for the two ticks a piston takes to fire. See
    // `mc_world::tileent::World::piston_collision_box`. That world-side helper follows
    // BlockPistonMoving.getBoundingBox's singular carried `getCollisionBoundingBox` result;
    // ordinary block collision below still appends every `addCollisionBoxesToList` box.
    if state.block().0 == mc_world::tileent::PISTON_EXTENSION_ID {
        if let Some(a) = w.piston_collision_box(pos) {
            if mask.intersects_with(&a) {
                out.push(a);
            }
        }
        return;
    }
    // Cactus and cake build their box with `float` arithmetic on the
    // *absolute* coordinates, so beyond |x| = 2^20 the extracted f64 table (a block-local offset
    // added in f64) is not what vanilla collides with. Ids, not class names: this runs for every
    // non-air cell of every sweep. `float_shape_ids_are_the_registry_blocks` pins both.
    match props.block.0 {
        CACTUS_ID => {
            cactus_box(pos, mask, out);
            return;
        }
        CAKE_ID => {
            cake_box(pos, state.meta(), mask, out);
            return;
        }
        _ => {}
    }
    // `BlockFenceGate` builds its box in `getCollisionBoundingBox` from the state alone, with
    // the same absolute-coordinate float arithmetic. Do not route it through the extracted,
    // already-widened f64 table.
    if block_info(props.block).java_class == class::FENCE_GATE {
        fence_gate_boxes(pos, state, mask, out);
        return;
    }
    if !props.collision_depends_on_neighbors {
        for b in props.collision.boxes() {
            let a = mc_world::to_world_aabb(*b, pos);
            if mask.intersects_with(&a) {
                out.push(a);
            }
        }
        return;
    }

    match block_info(props.block).java_class {
        class::FENCE => fence_boxes(w, pos, state, mask, out),
        class::WALL => wall_boxes(w, pos, mask, out),
        class::PANE | class::STAINED_PANE => pane_boxes(w, pos, state, mask, out),
        class::STAIRS => stairs_boxes(w, pos, state, mask, out),
        class::DOOR => door_boxes(w, pos, mask, out),
        // BlockChest and anything else flagged: the isolated shape is the movement-collision
        // shape. Connected chest selection/raytrace bounds are resolved separately in mc-world.
        _ => {
            for b in props.collision.boxes() {
                let a = mc_world::to_world_aabb(*b, pos);
                if mask.intersects_with(&a) {
                    out.push(a);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------- fence gates

/// java: `BlockFenceGate.getCollisionBoundingBox` (`BlockFenceGate.java:52-63`).
///
/// The open bit removes the collision box. Closed gates facing north/south span the X axis;
/// east/west gates span Z. The in-wall property only changes rendering and the selected bounds,
/// not this collision volume in 1.8.9.
///
/// Each bound is transcribed with its own Java arithmetic: the fractional ones are
/// `(double)((float)pos + 0.375F)`, but the whole-block ones are `(double)pos.getX()` and
/// `(double)(pos.getX() + 1)`, an int add widened to double. Routing those through `float` too
/// (as this did before) rounds an odd coordinate beyond 2^24 to its even neighbour.
fn fence_gate_boxes(pos: BlockPos, state: BlockState, mask: &Aabb, out: &mut Vec<Aabb>) {
    let meta = state.meta();
    if meta & 4 != 0 {
        return;
    }

    // java: `getHorizontal(meta & 3)` is SOUTH, WEST, NORTH, EAST, so an even meta is Z-axis.
    let a = if meta & 1 == 0 {
        Aabb::from_bounds(
            f64::from(pos.x),
            f64::from(pos.y),
            f64::from(pos.z as f32 + 0.375),
            f64::from(pos.x.wrapping_add(1)),
            f64::from(pos.y as f32 + 1.5),
            f64::from(pos.z as f32 + 0.625),
        )
    } else {
        Aabb::from_bounds(
            f64::from(pos.x as f32 + 0.375),
            f64::from(pos.y),
            f64::from(pos.z),
            f64::from(pos.x as f32 + 0.625),
            f64::from(pos.y as f32 + 1.5),
            f64::from(pos.z.wrapping_add(1)),
        )
    };
    push_box(out, mask, a);
}

// ---------------------------------------------------------------------------- fences

/// java: `BlockFence.canConnectTo` (`BlockFence.java:161`).
fn fence_can_connect(w: &World, pos: BlockPos, self_material: Material) -> bool {
    let state = w.block_at(pos);
    let props = block_props(state);
    let info = block_info(props.block);
    if info.java_class == class::BARRIER {
        return false;
    }
    if info.java_class == class::FENCE && props.material == self_material {
        return true;
    }
    if info.java_class == class::FENCE_GATE {
        return true;
    }
    props.material.is_opaque() && props.full_cube && props.material != Material::Gourd
}

/// java: `BlockFence.addCollisionBoxesToList` (`BlockFence.java:50`).
///
/// At most two boxes, both `1.5` tall. The Z-axis bar is emitted when the fence connects
/// north or south; the X-axis bar when it connects east or west **or when it connects to
/// nothing at all** — that last clause is what makes an isolated post exist.
fn fence_boxes(w: &World, pos: BlockPos, state: BlockState, mask: &Aabb, out: &mut Vec<Aabb>) {
    let material = block_props(state).material;
    let north = fence_can_connect(w, pos.north(), material);
    let south = fence_can_connect(w, pos.south(), material);
    let west = fence_can_connect(w, pos.west(), material);
    let east = fence_can_connect(w, pos.east(), material);

    let f = 0.375f32;
    let f1 = 0.625f32;
    let mut f2 = 0.375f32;
    let mut f3 = 0.625f32;

    if north {
        f2 = 0.0;
    }
    if south {
        f3 = 1.0;
    }
    if north || south {
        push(out, mask, pos, local(f, 0.0, f2, f1, 1.5, f3));
    }

    // vanilla resets the Z pair here, then widens the X pair.
    let f2 = 0.375f32;
    let f3 = 0.625f32;
    let f = if west { 0.0f32 } else { f };
    let f1 = if east { 1.0f32 } else { f1 };
    if west || east || (!north && !south) {
        push(out, mask, pos, local(f, 0.0, f2, f1, 1.5, f3));
    }
}

// ---------------------------------------------------------------------------- walls

/// java: `BlockWall.canConnectTo` (`BlockWall.java:122`). Note `block != this`: identity with
/// the same `BlockWall` singleton, so both cobblestone-wall variants connect to each other.
fn wall_can_connect(w: &World, pos: BlockPos, self_block: mc_data::BlockId) -> bool {
    let state = w.block_at(pos);
    let props = block_props(state);
    let info = block_info(props.block);
    if info.java_class == class::BARRIER {
        return false;
    }
    if props.block == self_block || info.java_class == class::FENCE_GATE {
        return true;
    }
    props.material.is_opaque() && props.full_cube && props.material != Material::Gourd
}

/// java: `BlockWall.getCollisionBoundingBox` (`BlockWall.java:115`) — one box, from
/// `setBlockBoundsBasedOnState` with `maxY` forced to `1.5`.
///
/// The straight-run narrowing to `[0.3125, 0.6875]` still applies to collision; only the
/// `0.8125` height it comes with is overwritten.
fn wall_boxes(w: &World, pos: BlockPos, mask: &Aabb, out: &mut Vec<Aabb>) {
    let self_block = block_props(w.block_at(pos)).block;
    let north = wall_can_connect(w, pos.north(), self_block);
    let south = wall_can_connect(w, pos.south(), self_block);
    let west = wall_can_connect(w, pos.west(), self_block);
    let east = wall_can_connect(w, pos.east(), self_block);

    let mut f = 0.25f32;
    let mut f1 = 0.75f32;
    let mut f2 = 0.25f32;
    let mut f3 = 0.75f32;

    if north {
        f2 = 0.0;
    }
    if south {
        f3 = 1.0;
    }
    if west {
        f = 0.0;
    }
    if east {
        f1 = 1.0;
    }

    if north && south && !west && !east {
        f = 0.3125;
        f1 = 0.6875;
    } else if !north && !south && west && east {
        f2 = 0.3125;
        f3 = 0.6875;
    }

    push(out, mask, pos, local(f, 0.0, f2, f1, 1.5, f3));
}

// ---------------------------------------------------------------------------- panes

/// java: `BlockPane.canPaneConnectToBlock` (`BlockPane.java:177`). Takes a block, not a
/// position — no metadata is involved.
fn pane_can_connect(w: &World, pos: BlockPos, self_block: mc_data::BlockId) -> bool {
    let props = block_props(w.block_at(pos));
    let info = block_info(props.block);
    props.full_block
        || props.block == self_block
        || info.name == "minecraft:glass"
        || info.name == "minecraft:stained_glass"
        || info.name == "minecraft:stained_glass_pane"
        || info.java_class == class::PANE
        || info.java_class == class::STAINED_PANE
}

/// java: `BlockPane.addCollisionBoxesToList` (`BlockPane.java:75`). Two independent axis
/// passes; a fully isolated pane emits a full-length cross, and a pane that only connects
/// on one axis emits nothing on the other.
fn pane_boxes(w: &World, pos: BlockPos, state: BlockState, mask: &Aabb, out: &mut Vec<Aabb>) {
    let self_block = block_props(state).block;
    let north = pane_can_connect(w, pos.north(), self_block);
    let south = pane_can_connect(w, pos.south(), self_block);
    let west = pane_can_connect(w, pos.west(), self_block);
    let east = pane_can_connect(w, pos.east(), self_block);
    let any = north || south || west || east;

    if (!west || !east) && any {
        if west {
            push(out, mask, pos, local(0.0, 0.0, 0.4375, 0.5, 1.0, 0.5625));
        } else if east {
            push(out, mask, pos, local(0.5, 0.0, 0.4375, 1.0, 1.0, 0.5625));
        }
    } else {
        push(out, mask, pos, local(0.0, 0.0, 0.4375, 1.0, 1.0, 0.5625));
    }

    if (!north || !south) && any {
        if north {
            push(out, mask, pos, local(0.4375, 0.0, 0.0, 0.5625, 1.0, 0.5));
        } else if south {
            push(out, mask, pos, local(0.4375, 0.0, 0.5, 0.5625, 1.0, 1.0));
        }
    } else {
        push(out, mask, pos, local(0.4375, 0.0, 0.0, 0.5625, 1.0, 1.0));
    }
}

// ---------------------------------------------------------------------------- stairs

/// The four horizontal facings, in `EnumFacing`'s own index order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    North,
    South,
    West,
    East,
}

impl Dir {
    fn offset(self, pos: BlockPos) -> BlockPos {
        match self {
            Dir::North => pos.north(),
            Dir::South => pos.south(),
            Dir::West => pos.west(),
            Dir::East => pos.east(),
        }
    }
}

/// java: `BlockStairs.getStateFromMeta` — `facing = EnumFacing.getFront(5 - (meta & 3))`,
/// `half = (meta & 4) != 0 ? TOP : BOTTOM`.
fn stairs_facing(meta: u8) -> Dir {
    match 5 - (meta & 3) {
        2 => Dir::North,
        3 => Dir::South,
        4 => Dir::West,
        _ => Dir::East,
    }
}

fn stairs_is_top(meta: u8) -> bool {
    meta & 4 != 0
}

/// java: `BlockStairs.isBlockStairs` — `instanceof`, so any stairs block of any material.
fn is_stairs(w: &World, pos: BlockPos) -> Option<u8> {
    let state = w.block_at(pos);
    let props = block_props(state);
    if block_info(props.block).java_class == class::STAIRS {
        Some(state.meta())
    } else {
        None
    }
}

/// java: `BlockStairs.isSameStair` — same class, same `HALF`, same `FACING`.
fn is_same_stair(w: &World, pos: BlockPos, meta: u8) -> bool {
    match is_stairs(w, pos) {
        Some(m) => {
            stairs_is_top(m) == stairs_is_top(meta) && stairs_facing(m) == stairs_facing(meta)
        }
        None => false,
    }
}

/// java: `BlockStairs.addCollisionBoxesToList` (`BlockStairs.java:533`).
///
/// Three `super.addCollisionBoxesToList` calls: the base half-slab, then the upper step (always
/// added — `func_176306_h` sets the bounds whatever it returns), then an optional third box for
/// inner corners.
fn stairs_boxes(w: &World, pos: BlockPos, state: BlockState, mask: &Aabb, out: &mut Vec<Aabb>) {
    let meta = state.meta();
    let top = stairs_is_top(meta);

    // setBaseCollisionBounds
    if top {
        push(out, mask, pos, local(0.0, 0.5, 0.0, 1.0, 1.0, 1.0));
    } else {
        push(out, mask, pos, local(0.0, 0.0, 0.0, 1.0, 0.5, 1.0));
    }

    let (flag, upper) = stairs_upper_box(w, pos, meta);
    push(out, mask, pos, upper);

    if flag {
        if let Some(extra) = stairs_extra_box(w, pos, meta) {
            push(out, mask, pos, extra);
        }
    }
}

/// java: `BlockStairs.func_176306_h` — returns `flag1` and, as a side effect, the bounds of
/// the second box (which the caller adds unconditionally).
fn stairs_upper_box(w: &World, pos: BlockPos, meta: u8) -> (bool, LocalBox) {
    let top = stairs_is_top(meta);
    let facing = stairs_facing(meta);
    let (f, f1) = if top {
        (0.0f32, 0.5f32)
    } else {
        (0.5f32, 1.0f32)
    };
    let mut f2 = 0.0f32;
    let mut f3 = 1.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.5f32;
    let mut flag1 = true;

    let neighbour =
        |d: Dir| -> Option<u8> { is_stairs(w, d.offset(pos)).filter(|m| stairs_is_top(*m) == top) };

    match facing {
        Dir::East => {
            f2 = 0.5;
            f5 = 1.0;
            if let Some(m) = neighbour(Dir::East) {
                match stairs_facing(m) {
                    Dir::North if !is_same_stair(w, pos.south(), meta) => {
                        f5 = 0.5;
                        flag1 = false;
                    }
                    Dir::South if !is_same_stair(w, pos.north(), meta) => {
                        f4 = 0.5;
                        flag1 = false;
                    }
                    _ => {}
                }
            }
        }
        Dir::West => {
            f3 = 0.5;
            f5 = 1.0;
            if let Some(m) = neighbour(Dir::West) {
                match stairs_facing(m) {
                    Dir::North if !is_same_stair(w, pos.south(), meta) => {
                        f5 = 0.5;
                        flag1 = false;
                    }
                    Dir::South if !is_same_stair(w, pos.north(), meta) => {
                        f4 = 0.5;
                        flag1 = false;
                    }
                    _ => {}
                }
            }
        }
        Dir::South => {
            f4 = 0.5;
            f5 = 1.0;
            if let Some(m) = neighbour(Dir::South) {
                match stairs_facing(m) {
                    Dir::West if !is_same_stair(w, pos.east(), meta) => {
                        f3 = 0.5;
                        flag1 = false;
                    }
                    Dir::East if !is_same_stair(w, pos.west(), meta) => {
                        f2 = 0.5;
                        flag1 = false;
                    }
                    _ => {}
                }
            }
        }
        Dir::North => {
            if let Some(m) = neighbour(Dir::North) {
                match stairs_facing(m) {
                    Dir::West if !is_same_stair(w, pos.east(), meta) => {
                        f3 = 0.5;
                        flag1 = false;
                    }
                    Dir::East if !is_same_stair(w, pos.west(), meta) => {
                        f2 = 0.5;
                        flag1 = false;
                    }
                    _ => {}
                }
            }
        }
    }

    (flag1, local(f2, f, f4, f3, f1, f5))
}

/// java: `BlockStairs.func_176304_i` — the inner-corner filler. The `f2`/`f3` pre-set on WEST
/// but not EAST, and `f4`/`f5` on SOUTH but not NORTH, are asymmetries in the original; they
/// are reproduced verbatim rather than normalised.
fn stairs_extra_box(w: &World, pos: BlockPos, meta: u8) -> Option<LocalBox> {
    let top = stairs_is_top(meta);
    let facing = stairs_facing(meta);
    let (f, f1) = if top {
        (0.0f32, 0.5f32)
    } else {
        (0.5f32, 1.0f32)
    };
    let mut f2 = 0.0f32;
    let mut f3 = 0.5f32;
    let mut f4 = 0.5f32;
    let mut f5 = 1.0f32;
    let mut flag1 = false;

    let neighbour =
        |d: Dir| -> Option<u8> { is_stairs(w, d.offset(pos)).filter(|m| stairs_is_top(*m) == top) };

    match facing {
        Dir::East => {
            if let Some(m) = neighbour(Dir::West) {
                match stairs_facing(m) {
                    Dir::North if !is_same_stair(w, pos.north(), meta) => {
                        f4 = 0.0;
                        f5 = 0.5;
                        flag1 = true;
                    }
                    Dir::South if !is_same_stair(w, pos.south(), meta) => {
                        f4 = 0.5;
                        f5 = 1.0;
                        flag1 = true;
                    }
                    _ => {}
                }
            }
        }
        Dir::West => {
            if let Some(m) = neighbour(Dir::East) {
                f2 = 0.5;
                f3 = 1.0;
                match stairs_facing(m) {
                    Dir::North if !is_same_stair(w, pos.north(), meta) => {
                        f4 = 0.0;
                        f5 = 0.5;
                        flag1 = true;
                    }
                    Dir::South if !is_same_stair(w, pos.south(), meta) => {
                        f4 = 0.5;
                        f5 = 1.0;
                        flag1 = true;
                    }
                    _ => {}
                }
            }
        }
        Dir::South => {
            if let Some(m) = neighbour(Dir::North) {
                f4 = 0.0;
                f5 = 0.5;
                match stairs_facing(m) {
                    Dir::West if !is_same_stair(w, pos.west(), meta) => {
                        flag1 = true;
                    }
                    Dir::East if !is_same_stair(w, pos.east(), meta) => {
                        f2 = 0.5;
                        f3 = 1.0;
                        flag1 = true;
                    }
                    _ => {}
                }
            }
        }
        Dir::North => {
            if let Some(m) = neighbour(Dir::South) {
                match stairs_facing(m) {
                    Dir::West if !is_same_stair(w, pos.west(), meta) => {
                        flag1 = true;
                    }
                    Dir::East if !is_same_stair(w, pos.east(), meta) => {
                        f2 = 0.5;
                        f3 = 1.0;
                        flag1 = true;
                    }
                    _ => {}
                }
            }
        }
    }

    if flag1 {
        Some(local(f2, f, f4, f3, f1, f5))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------- doors

/// java: `BlockDoor.combineMetadata` (`BlockDoor.java:295`).
///
/// The upper half carries the hinge and powered bits; the lower half carries facing and open.
/// Vanilla reads `getMetaFromState` on the blocks above and below **without checking they are
/// doors**.  `World#getBlockState` has already resolved an illegal packed chunk state to the
/// air default before that call, though, so its metadata is zero rather than the raw low nibble.
/// A legal non-door neighbour still contributes its metadata exactly as Java does.
fn door_state_meta(w: &World, pos: BlockPos) -> u8 {
    let state = w.block_at(pos);
    if state.is_legal() {
        state.meta()
    } else {
        // `ExtendedBlockStorage.get` returns `Blocks.air.getDefaultState()` for an unregistered
        // packed state; `BlockDoor.getMetaFromState` then sees that state's metadata zero.
        0
    }
}

fn door_combined_meta(w: &World, pos: BlockPos) -> u8 {
    let i = door_state_meta(w, pos);
    let is_top = i & 8 != 0;
    let j = door_state_meta(w, pos.down());
    let l = door_state_meta(w, pos.up());
    let k = if is_top { j } else { i };
    let i1 = if is_top { i } else { l };
    let hinge = i1 & 1 != 0;
    let powered = i1 & 2 != 0;
    (k & 7) | if is_top { 8 } else { 0 } | if hinge { 16 } else { 0 } | if powered { 32 } else { 0 }
}

/// java: `BlockDoor.getFacing` — `EnumFacing.getHorizontal(combined & 3).rotateYCCW()`.
/// `getHorizontal` is `[SOUTH, WEST, NORTH, EAST]`; `rotateYCCW` maps
/// `SOUTH->EAST, WEST->SOUTH, NORTH->WEST, EAST->NORTH`.
fn door_facing(combined: u8) -> Dir {
    match combined & 3 {
        0 => Dir::East,
        1 => Dir::South,
        2 => Dir::West,
        _ => Dir::North,
    }
}

/// java: `BlockDoor.getCollisionBoundingBox` -> `setBoundBasedOnMeta` (`BlockDoor.java:83`).
///
/// `isHingeLeft` is a deobfuscation misnomer — bit 16 is set when the hinge is on the *right*.
/// It is treated here as the opaque flag it is.
fn door_boxes(w: &World, pos: BlockPos, mask: &Aabb, out: &mut Vec<Aabb>) {
    let combined = door_combined_meta(w, pos);
    let facing = door_facing(combined);
    let open = combined & 4 != 0;
    let hinge = combined & 16 != 0;
    const F: f32 = 0.1875;
    const G: f32 = 1.0 - F;

    let b = if open {
        match (facing, hinge) {
            (Dir::East, false) => local(0.0, 0.0, 0.0, 1.0, 1.0, F),
            (Dir::East, true) => local(0.0, 0.0, G, 1.0, 1.0, 1.0),
            (Dir::South, false) => local(G, 0.0, 0.0, 1.0, 1.0, 1.0),
            (Dir::South, true) => local(0.0, 0.0, 0.0, F, 1.0, 1.0),
            (Dir::West, false) => local(0.0, 0.0, G, 1.0, 1.0, 1.0),
            (Dir::West, true) => local(0.0, 0.0, 0.0, 1.0, 1.0, F),
            (Dir::North, false) => local(0.0, 0.0, 0.0, F, 1.0, 1.0),
            (Dir::North, true) => local(G, 0.0, 0.0, 1.0, 1.0, 1.0),
        }
    } else {
        match facing {
            Dir::East => local(0.0, 0.0, 0.0, F, 1.0, 1.0),
            Dir::South => local(0.0, 0.0, 0.0, 1.0, 1.0, F),
            Dir::West => local(G, 0.0, 0.0, 1.0, 1.0, 1.0),
            Dir::North => local(0.0, 0.0, G, 1.0, 1.0, 1.0),
        }
    };
    push(out, mask, pos, b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{block_by_name, BlockState};

    fn world_with(blocks: &[(BlockPos, &str, u8)]) -> World {
        let mut w = World::new();
        for (pos, name, meta) in blocks {
            let id = block_by_name(name).unwrap_or_else(|| panic!("no block {name}"));
            crate::scene::set_block(&mut w, *pos, BlockState::new(id, *meta));
        }
        w
    }

    fn boxes(w: &World, pos: BlockPos) -> Vec<Aabb> {
        let huge = Aabb::from_bounds(-64.0, -64.0, -64.0, 320.0, 320.0, 320.0);
        collision_boxes_at(w, pos, &huge)
    }

    /// java: `getCollidingBoundingBoxes` scans `y` from `floor_double(minY) - 1`
    /// in wrapping `int` arithmetic. A box at `minY` in `[-2^31, -2^31 + 1)` (reachable through
    /// a long-distance S2A particle) starts at `Integer.MIN_VALUE`, so `- 1` wraps and the scan
    /// is empty. The old `y0 - 1` panicked in debug builds; the packet fuzz found it.
    #[test]
    fn a_box_at_the_int_floor_scans_nothing_like_javas_wrapped_start() {
        let w = World::new();
        let low = -2_147_483_648.0;
        let mask = Aabb::from_bounds(0.25, low, 0.25, 0.75, low + 0.5, 0.75);
        let mut out = Vec::new();
        collision_boxes_in(&w, &mask, &mut out);
        assert!(out.is_empty());
    }

    /// The pristine gather: `is_chunk_loaded` per column, then `World::block_at` for every
    /// cell through `add_block_boxes`. The parity oracle for the column cache (#19/PART-4).
    fn reference_gather(w: &World, mask: &Aabb) -> Vec<Aabb> {
        let mut out = Vec::new();
        for x in floor_f64(mask.min.x)..floor_f64(mask.max.x + 1.0) {
            for z in floor_f64(mask.min.z)..floor_f64(mask.max.z + 1.0) {
                if !w.is_chunk_loaded(BlockPos::new(x, 64, z).chunk_pos()) {
                    continue;
                }
                for y in (floor_f64(mask.min.y) - 1)..floor_f64(mask.max.y + 1.0) {
                    add_block_boxes(w, BlockPos::new(x, y, z), mask, &mut out);
                }
            }
        }
        out
    }

    /// The column-cached gather returns the same boxes in the same order (the sweep's parity
    /// contract) for neighbour-dependent shapes straddling chunk borders, at the world floor
    /// and ceiling, and across an unloaded column inside the scanned range.
    #[test]
    fn column_cached_gather_matches_the_per_cell_gather_box_for_box() {
        let names = [
            "stone",
            "fence",
            "oak_stairs",
            "glass_pane",
            "cobblestone_wall",
            "stone_slab",
            "fence_gate",
            "iron_bars",
            "chest",
            "trapdoor",
            "carpet",
            "snow_layer",
            "ladder",
            "nether_brick_fence",
        ];
        let heights = [0, 1, 2, 63, 64, 65, 254, 255];
        let mut rng = mc_math::JavaRandom::new(0x0c01_11de);
        let mut blocks = Vec::new();
        for _ in 0..900 {
            let pos = BlockPos::new(
                rng.next_i32_bound(40) - 20,
                heights[rng.next_i32_bound(heights.len() as i32) as usize],
                rng.next_i32_bound(40) - 20,
            );
            // Leave column (1, -1) unloaded.
            if pos.chunk_pos() == mc_math::ChunkPos::new(1, -1) {
                continue;
            }
            let name = names[rng.next_i32_bound(names.len() as i32) as usize];
            blocks.push((pos, name, rng.next_i32_bound(16) as u8));
        }
        let w = world_with(&blocks);
        assert!(!w.is_chunk_loaded(mc_math::ChunkPos::new(1, -1)));
        for case in 0..4_000 {
            let min = DVec3::new(
                f64::from(rng.next_i32_bound(48) - 24) + rng.next_f64(),
                f64::from(rng.next_i32_bound(262) - 3) + rng.next_f64(),
                f64::from(rng.next_i32_bound(48) - 24) + rng.next_f64(),
            );
            let mask = Aabb::from_bounds(
                min.x,
                min.y,
                min.z,
                min.x + 0.05 + 3.5 * rng.next_f64(),
                min.y + 0.05 + 3.5 * rng.next_f64(),
                min.z + 0.05 + 3.5 * rng.next_f64(),
            );
            let mut got = Vec::new();
            collision_boxes_in(&w, &mask, &mut got);
            assert_eq!(got, reference_gather(&w, &mask), "case {case}: {mask:?}");
        }
    }

    /// Micro-bench (#19/PART-4): `cargo test -p mc-physics --lib -- --ignored bench_gather
    /// --nocapture`. The old side adds the per-query `format!` + registry search the pristine
    /// gather paid for its stone substitute.
    #[test]
    #[ignore = "timing bench; run explicitly"]
    fn bench_gather_column_cache_vs_per_cell() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut blocks = Vec::new();
        for x in -8..8 {
            for z in -8..8 {
                blocks.push((BlockPos::new(x, 64, z), "stone", 0));
            }
        }
        let w = world_with(&blocks);
        let masks = [
            (
                "particle in void air",
                Aabb::from_bounds(3.4, 79.8, 3.4, 3.6, 80.0, 3.6),
            ),
            (
                "item resting",
                Aabb::from_bounds(2.375, 65.0, 2.375, 2.625, 65.25, 2.625),
            ),
            (
                "player sweep",
                Aabb::from_bounds(-0.3, 64.9, -0.3, 0.6, 66.8, 0.6),
            ),
        ];
        for (label, mask) in masks {
            let n = 200_000;
            let start = Instant::now();
            for _ in 0..n {
                black_box(format!("minecraft:{}", black_box("stone")));
                black_box(block_by_name("minecraft:stone"));
                black_box(reference_gather(&w, black_box(&mask)));
            }
            let old = start.elapsed();
            let mut out = Vec::new();
            let start = Instant::now();
            for _ in 0..n {
                out.clear();
                collision_boxes_in(&w, black_box(&mask), &mut out);
                black_box(&out);
            }
            let new = start.elapsed();
            println!(
                "bench_gather [{label}] boxes={} old={:.0}ns new={:.0}ns",
                out.len(),
                old.as_secs_f64() * 1e9 / f64::from(n),
                new.as_secs_f64() * 1e9 / f64::from(n),
            );
        }
    }

    #[test]
    fn stone_constant_is_the_registry_default_state() {
        assert_eq!(
            STONE,
            BlockState::new(block_by_name("stone").expect("stone"), 0)
        );
    }

    /// The gather skips air cells without consulting `add_block_boxes_for`.
    #[test]
    fn air_has_no_collision_boxes() {
        let props = block_props(BlockState::AIR);
        assert!(props.collision.boxes().is_empty());
        assert!(!props.collision_depends_on_neighbors);
        assert_ne!(
            BlockState::AIR.block().0,
            mc_world::tileent::PISTON_EXTENSION_ID
        );
        assert_ne!(block_info(props.block).java_class, class::FENCE_GATE);
        let huge = Aabb::from_bounds(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0);
        let mut out = Vec::new();
        add_block_boxes_for(
            &World::new(),
            BlockPos::new(0, 0, 0),
            BlockState::AIR,
            &huge,
            &mut out,
        );
        assert!(out.is_empty());
    }

    fn boxes_at_far_position(w: &World, pos: BlockPos) -> Vec<Aabb> {
        let x = f64::from(pos.x);
        let y = f64::from(pos.y);
        let z = f64::from(pos.z);
        collision_boxes_at(
            w,
            pos,
            &Aabb::from_bounds(x - 2.0, y - 2.0, z - 2.0, x + 3.0, y + 3.0, z + 3.0),
        )
    }

    #[test]
    fn border_substitutes_stone_outside_only_for_an_inside_entity() {
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let border = CollisionBorder::new(0.0, 0.0, 4.0, 29_999_984);
        let mask = Aabb::from_bounds(2.1, 64.1, 0.1, 2.9, 64.9, 0.9);

        let mut wall = Vec::new();
        collision_boxes_in_with_border(&w, &mask, border, true, &mut wall);
        assert_eq!(
            wall.len(),
            1,
            "outside block positions become stone wall cubes"
        );
        assert_eq!(wall[0].min, DVec3::new(2.0, 64.0, 0.0));
        assert_eq!(wall[0].max, DVec3::new(3.0, 65.0, 1.0));

        // Java's `!isInsideBorder(entity)` branch stops substituting wall blocks for an entity
        // that is already outside the hysteresis region.
        let mut escaped = Vec::new();
        collision_boxes_in_with_border(&w, &mask, border, false, &mut escaped);
        assert!(
            escaped.is_empty(),
            "outside entity sees the real empty world"
        );
    }

    #[test]
    fn border_entity_latch_uses_vanillas_one_block_hysteresis() {
        let border = CollisionBorder::new(0.0, 0.0, 10.0, 29_999_984);

        // An inside entity may cross the physical edge at x=5 while it remains in the expanded
        // [-6, 6] hysteresis rectangle. It latches outside only after leaving that rectangle.
        assert!(border.entity_inside(5.5, 0.0, false));
        assert!(!border.entity_inside(6.0, 0.0, false), "the edge is strict");
        assert!(!border.update_entity_outside(5.5, 0.0, false));
        assert!(border.update_entity_outside(6.1, 0.0, false));

        // Once outside, re-entry needs the contracted (-4, 4) rectangle. Crossing x=4 from the
        // outside side must not immediately clear the latch, while x=3.9 does.
        assert!(!border.entity_inside(4.5, 0.0, true));
        assert!(border.update_entity_outside(4.5, 0.0, true));
        assert!(border.entity_inside(3.9, 0.0, true));
        assert!(!border.update_entity_outside(3.9, 0.0, true));
    }

    #[test]
    fn border_entity_latch_tracks_a_shifted_center() {
        let border = CollisionBorder::new(10.0, -4.0, 8.0, 29_999_984);
        // The current edges are x=6/14 and z=-8/0; the inside latch expands each by one.
        assert!(border.entity_inside(14.5, -0.5, false));
        assert!(border.update_entity_outside(15.1, -0.5, false));
    }

    #[test]
    fn static_no_border_query_is_unchanged_and_missing_wall_mutant_bites() {
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let mask = Aabb::from_bounds(2.1, 64.1, 0.1, 2.9, 64.9, 0.9);
        let mut out = Vec::new();
        collision_boxes_in(&w, &mask, &mut out);
        assert!(out.is_empty(), "the no-border compatibility path reads air");
    }

    #[test]
    fn moving_border_changes_the_wall_at_the_interpolated_edge() {
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let mask = Aabb::from_bounds(4.1, 64.1, 0.1, 4.9, 64.9, 0.9);
        let shrinking = CollisionBorder::new(0.0, 0.0, 8.0, 29_999_984);
        let growing = CollisionBorder::new(0.0, 0.0, 12.0, 29_999_984);

        let mut before = Vec::new();
        collision_boxes_in_with_border(&w, &mask, shrinking, true, &mut before);
        assert_eq!(before.len(), 1, "edge x=4 is outside an 8-wide border");

        let mut after = Vec::new();
        collision_boxes_in_with_border(&w, &mask, growing, true, &mut after);
        assert!(after.is_empty(), "the same edge is inside a 12-wide border");
    }

    #[test]
    fn a_moving_piston_block_has_no_box_without_its_tile_entity_and_one_with_it() {
        // java: `BlockPistonMoving.getCollisionBoundingBox` returns null when
        // `getTileEntity(pos)` is not a `TileEntityPiston`, and the carried block's own box,
        // shifted by `getProgress(0.0F)`, when it is. mc-data's extracted `collision: Empty` is
        // only the first half, and a moving block is never in that state — so before this a
        // player fell through the block a piston was pushing under them.
        //
        // Negative control: the `is_empty()` assertion below is what the *old* code produced in
        // both halves of this test, so the second half fails on it.
        let pos = BlockPos::new(0, 0, 0);
        let bare = world_with(&[(pos, "piston_extension", 5)]);
        assert!(boxes(&bare, pos).is_empty(), "no tile entity, no box");

        let mut w = world_with(&[(pos, "piston_extension", 5)]);
        w.set_tile_entity(
            pos,
            mc_world::TileEntity::new_piston(
                BlockState::new(block_by_name("stone").unwrap(), 0),
                mc_math::Facing::East,
                false,
                false,
            ),
        );
        let b = boxes(&w, pos);
        assert_eq!(b.len(), 1, "a retracting stone has a floor: {b:?}");
        // Retracting at progress 0: `f = getProgress(0) = 0`, so the box is the full cube.
        assert_eq!(b[0].min, DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(b[0].max, DVec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn moving_piston_collision_uses_one_carried_bounds_box() {
        // Java's BlockPistonMoving.getBoundingBox asks the carried block for one
        // getCollisionBoundingBox. It does not union the fence's 1.5-high addCollisionBoxes
        // result. This is also the physics-facing control that keeps the local moving carrier
        // path separate from ordinary neighbour-aware collision enumeration.
        let pos = BlockPos::ORIGIN;
        let mut w = world_with(&[(pos, "piston_extension", 5)]);
        w.set_tile_entity(
            pos,
            mc_world::TileEntity::new_piston(
                BlockState::new(block_by_name("fence").unwrap(), 0),
                mc_math::Facing::East,
                false,
                false,
            ),
        );
        if let Some(te) = w.tile_entity_mut(pos) {
            if let mc_world::TileData::Piston(p) = &mut te.data {
                p.progress = 0.5;
                p.last_progress = 0.5;
            }
        }

        let b = boxes(&w, pos);
        assert_eq!(b.len(), 1, "the moving carrier is one AABB: {b:?}");
        assert_eq!(b[0].min, DVec3::new(0.125, 0.0, 0.375));
        assert_eq!(b[0].max, DVec3::new(0.375, 1.0, 0.625));
    }

    #[test]
    fn isolated_fence_is_a_single_tall_post() {
        let w = world_with(&[(BlockPos::new(0, 0, 0), "fence", 0)]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1, "the `!north && !south` clause emits the post");
        assert_eq!(b[0].min, DVec3::new(0.375, 0.0, 0.375));
        assert_eq!(b[0].max, DVec3::new(0.625, 1.5, 0.625));
    }

    #[test]
    fn closed_fence_gate_world_offsets_narrow_through_java_f32_at_positive_boundary() {
        // 2^21 is the first positive coordinate where an f32 has a 0.25 ULP.  Java's
        // `int + 0.375F` therefore rounds to 2^21 + 0.5; adding the widened local f64 would
        // incorrectly retain 2^21 + 0.375.
        let pos = BlockPos::new(2_097_152, 64, 0);
        let w = world_with(&[(pos, "fence_gate", 1)]); // west: thin along X
        let b = boxes_at_far_position(&w, pos);
        assert_eq!(b.len(), 1);

        let java_min = f64::from((pos.x as f32) + 0.375f32);
        let java_max = f64::from((pos.x as f32) + 0.625f32);
        assert_eq!(b[0].min.x, java_min);
        assert_eq!(b[0].max.x, java_max);
        assert_ne!(
            b[0].min.x,
            f64::from(pos.x) + f64::from(0.375f32),
            "the old static-f64 offset must fail at the rounding boundary"
        );
    }

    #[test]
    fn closed_fence_gate_world_offsets_narrow_through_java_f32_at_negative_boundary() {
        // Just below -2^21 the negative-side f32 binade also has 0.25 spacing.  Exercise the
        // other thin axis so both world-coordinate additions stay in Java's float domain.
        let pos = BlockPos::new(0, 64, -2_097_153);
        let w = world_with(&[(pos, "fence_gate", 0)]); // south: thin along Z
        let b = boxes_at_far_position(&w, pos);
        assert_eq!(b.len(), 1);

        let java_min = f64::from((pos.z as f32) + 0.375f32);
        let java_max = f64::from((pos.z as f32) + 0.625f32);
        assert_eq!(b[0].min.z, java_min);
        assert_eq!(b[0].max.z, java_max);
        assert_ne!(
            b[0].max.z,
            f64::from(pos.z) + f64::from(0.625f32),
            "the old static-f64 offset must fail on the negative-side boundary too"
        );
    }

    #[test]
    fn closed_fence_gate_keeps_ordinary_coordinate_shape() {
        let pos = BlockPos::new(17, 64, -23);
        let w = world_with(&[(pos, "fence_gate", 1)]);
        let b = boxes_at_far_position(&w, pos);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].min.x, 17.375);
        assert_eq!(b[0].max.x, 17.625);
        assert_eq!(b[0].min.z, -23.0);
        assert_eq!(b[0].max.z, -22.0);
    }

    #[test]
    fn fence_row_widens_along_its_run() {
        let w = world_with(&[
            (BlockPos::new(0, 0, -1), "fence", 0),
            (BlockPos::new(0, 0, 0), "fence", 0),
            (BlockPos::new(0, 0, 1), "fence", 0),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1, "N+S connected: only the Z bar is emitted");
        assert_eq!(b[0].min, DVec3::new(0.375, 0.0, 0.0));
        assert_eq!(b[0].max, DVec3::new(0.625, 1.5, 1.0));
    }

    #[test]
    fn fences_do_not_connect_across_materials() {
        // Wood fence next to a nether brick fence: different Material, so no connection.
        let w = world_with(&[
            (BlockPos::new(0, 0, 0), "fence", 0),
            (BlockPos::new(0, 0, 1), "nether_brick_fence", 0),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1);
        assert_eq!(
            b[0].max.z, 0.625,
            "no widening toward the nether brick fence"
        );
    }

    #[test]
    fn fence_connects_to_a_solid_neighbour() {
        let w = world_with(&[
            (BlockPos::new(0, 0, 0), "fence", 0),
            (BlockPos::new(1, 0, 0), "stone", 0),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].max.x, 1.0, "the X bar reaches the stone");
    }

    #[test]
    fn wall_is_one_and_a_half_blocks_tall_and_narrows_on_a_straight_run() {
        let w = world_with(&[
            (BlockPos::new(0, 0, -1), "cobblestone_wall", 0),
            (BlockPos::new(0, 0, 0), "cobblestone_wall", 0),
            (BlockPos::new(0, 0, 1), "cobblestone_wall", 0),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].min, DVec3::new(0.3125, 0.0, 0.0));
        assert_eq!(b[0].max, DVec3::new(0.6875, 1.5, 1.0));
    }

    #[test]
    fn isolated_pane_is_a_full_cross() {
        let w = world_with(&[(BlockPos::new(0, 0, 0), "glass_pane", 0)]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(
            b.len(),
            2,
            "both `else` branches fire when nothing connects"
        );
        assert_eq!(b[0].min.x, 0.0);
        assert_eq!(b[0].max.x, 1.0);
        assert_eq!(b[1].min.z, 0.0);
        assert_eq!(b[1].max.z, 1.0);
    }

    #[test]
    fn pane_connected_only_north_emits_one_half_bar() {
        let w = world_with(&[
            (BlockPos::new(0, 0, 0), "glass_pane", 0),
            (BlockPos::new(0, 0, -1), "glass_pane", 0),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1, "the X pass emits nothing when only N connects");
        assert_eq!(b[0].min.z, 0.0);
        assert_eq!(b[0].max.z, 0.5);
        assert_eq!(b[0].min.x, 0.4375);
    }

    #[test]
    fn straight_stairs_are_two_boxes() {
        // meta 0 -> facing EAST, bottom half.
        let w = world_with(&[(BlockPos::new(0, 0, 0), "stone_stairs", 0)]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].min, DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(b[0].max, DVec3::new(1.0, 0.5, 1.0));
        assert_eq!(b[1].min, DVec3::new(0.5, 0.5, 0.0));
        assert_eq!(b[1].max, DVec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn top_stairs_flip_the_base_slab() {
        let w = world_with(&[(BlockPos::new(0, 0, 0), "stone_stairs", 4)]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b[0].min.y, 0.5);
        assert_eq!(b[0].max.y, 1.0);
        assert_eq!(b[1].min.y, 0.0, "the step is on the bottom for a top stair");
    }

    #[test]
    fn closed_door_is_a_three_sixteenths_slab_on_its_facing_side() {
        // meta 0 -> facing EAST (getHorizontal(0)=SOUTH, rotateYCCW -> EAST), closed.
        let w = world_with(&[
            (BlockPos::new(0, 0, 0), "wooden_door", 0),
            (BlockPos::new(0, 1, 0), "wooden_door", 8),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].max.x, f64::from(0.1875f32));
        assert_eq!(b[0].max.y, 1.0);
        let upper = boxes(&w, BlockPos::new(0, 1, 0));
        assert_eq!(
            upper[0].max.x,
            f64::from(0.1875f32),
            "the upper half reads the lower half's facing"
        );
    }

    #[test]
    fn open_door_swings_out_of_the_way() {
        let w = world_with(&[
            (BlockPos::new(0, 0, 0), "wooden_door", 4), // open
            (BlockPos::new(0, 1, 0), "wooden_door", 8),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].max.z, f64::from(0.1875f32));
        assert_eq!(b[0].max.x, 1.0);
    }

    #[test]
    fn door_combine_metadata_normalizes_an_illegal_upper_neighbour_before_reading_hinge() {
        // `BlockDoor.combineMetadata` reads the upper neighbour even when it is not a door.
        // Chunk storage's illegal stone/meta-7 state resolves to air first, so Java reads meta
        // zero here.  Its raw low bit must not flip the open lower half's hinge and collision.
        let door = BlockPos::ORIGIN;
        let legal_control = world_with(&[(door, "wooden_door", 4), (door.up(), "stone", 0)]);
        let legal_hinge = world_with(&[(door, "wooden_door", 4), (door.up(), "stone", 1)]);
        let illegal_upper = world_with(&[(door, "wooden_door", 4), (door.up(), "stone", 7)]);

        assert!(!BlockState::new(block_by_name("stone").unwrap(), 7).is_legal());
        assert_eq!(boxes(&legal_hinge, door)[0].min.z, f64::from(0.8125f32));
        assert_eq!(
            boxes(&illegal_upper, door),
            boxes(&legal_control, door),
            "an illegal upper packed state has air/default metadata, not raw metadata"
        );
        assert_eq!(boxes(&illegal_upper, door)[0].max.z, f64::from(0.1875f32));
    }

    #[test]
    fn door_combine_metadata_normalizes_an_illegal_lower_neighbour_before_reading_facing() {
        // The upper door half obtains facing/open from the lower neighbour.  Raw stone/meta-7
        // would turn this into an open north-facing door; normalized air/default metadata is a
        // closed east-facing door, matching the legal stone/meta-0 control.
        let upper = BlockPos::new(0, 1, 0);
        let legal_control = world_with(&[(upper.down(), "stone", 0), (upper, "wooden_door", 8)]);
        let legal_open = world_with(&[(upper.down(), "stone", 4), (upper, "wooden_door", 8)]);
        let illegal_lower = world_with(&[(upper.down(), "stone", 7), (upper, "wooden_door", 8)]);

        assert_eq!(boxes(&legal_open, upper)[0].max.z, f64::from(0.1875f32));
        assert_eq!(
            boxes(&illegal_lower, upper),
            boxes(&legal_control, upper),
            "an illegal lower packed state has air/default metadata, not raw metadata"
        );
        assert_eq!(boxes(&illegal_lower, upper)[0].max.x, f64::from(0.1875f32));
    }

    #[test]
    fn chest_uses_the_isolated_shape_even_when_doubled() {
        let w = world_with(&[
            (BlockPos::new(0, 0, 0), "chest", 2),
            (BlockPos::new(0, 0, 1), "chest", 2),
        ]);
        let b = boxes(&w, BlockPos::ORIGIN);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].min, DVec3::new(0.0625, 0.0, 0.0625));
        assert_eq!(b[0].max, DVec3::new(0.9375, 0.875, 0.9375));
    }

    #[test]
    fn the_scan_reaches_one_block_below_the_mask() {
        // A fence at y=0 is 1.5 tall, so it must be found by a box that starts at y=1.
        let w = world_with(&[(BlockPos::new(0, 0, 0), "fence", 0)]);
        let mut out = Vec::new();
        collision_boxes_in(
            &w,
            &Aabb::from_bounds(0.3, 1.0, 0.3, 0.7, 2.8, 0.7),
            &mut out,
        );
        assert_eq!(out.len(), 1, "the y loop starts at floor(minY) - 1");
    }

    #[test]
    fn air_and_plants_contribute_nothing() {
        let w = world_with(&[(BlockPos::new(0, 0, 0), "tallgrass", 1)]);
        let mut out = Vec::new();
        collision_boxes_in(
            &w,
            &Aabb::from_bounds(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0),
            &mut out,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn unloaded_blank_chunk_writes_become_collision_boxes_in_every_absent_column() {
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let pos = BlockPos::new(32, 64, 32);
        let mask = Aabb::from_bounds(32.1, 64.1, 32.1, 32.9, 64.9, 32.9);
        let mut w = World::new();
        let mut out = Vec::new();
        collision_boxes_in(&w, &mask, &mut out);
        assert!(out.is_empty(), "control: a fresh absent column is air");

        // `World::set_block` into a column that has not arrived writes the shared
        // blank chunk (`ChunkProviderClient.blankChunk`, an `EmptyChunk` that inherits
        // `Chunk.setBlockState`/`getBlockState`). The collision scan reads an absent column
        // through `World::block_at`, as Java's reads `World.getBlockState`: on a client
        // `isBlockLoaded` is true for every valid column (`ChunkProviderClient.chunkExists`).
        w.set_block(pos, stone);
        assert!(!w.is_chunk_loaded(pos.chunk_pos()));
        assert_eq!(w.block_at(pos), stone);
        collision_boxes_in(&w, &mask, &mut out);
        assert_eq!(
            out.len(),
            1,
            "an absent column collides exactly as World::block_at reads it"
        );

        // The same local cell of another absent column is the same blank-chunk cell.
        out.clear();
        let alias = Aabb::from_bounds(-79.9, 64.1, 48.1, -79.1, 64.9, 48.9);
        collision_boxes_in(&w, &alias, &mut out);
        assert_eq!(
            out.len(),
            1,
            "(-80, 64, 48) shares local (0, 64, 0) with (32, 64, 32)"
        );

        // A column that arrives is a fresh chunk: air until its own write.
        out.clear();
        w.load_empty_chunk(pos.chunk_pos());
        collision_boxes_in(&w, &mask, &mut out);
        assert!(out.is_empty());
        w.set_block(pos, stone);
        collision_boxes_in(&w, &mask, &mut out);
        assert_eq!(out.len(), 1, "a resident stone cell collides");
    }

    /// java: on a client `isBlockLoaded(x, 64, z)` is `World.isValid` alone, so the border's
    /// stone and the `outsideBorder` latch also happen over columns the server never sent
    /// (an S21 with bitmask 0 is an unload, which is what an all-air void chunk arrives as).
    #[test]
    fn border_stone_and_latch_apply_over_unsent_columns() {
        let w = World::new();
        let border = CollisionBorder::new(8.5, 8.5, 10.0, 29_999_984); // walls at 3.5 / 13.5
                                                                       // A player box straddling the -X wall at feet y = 64, with nothing loaded anywhere.
        let mask = Aabb::from_bounds(2.9, 64.0, 8.2, 3.5, 65.8, 8.8);
        let mut outside = false;
        let mut out = Vec::new();
        collision_boxes_in_with_border_and_entities(
            &w,
            &mask,
            border,
            3.2,
            8.5,
            &mut outside,
            &[],
            &mut out,
        );
        // Column x = 2 is outside (2 + 1 > 3.5 fails); y scans 63..66.
        assert_eq!(
            out,
            vec![
                Aabb::from_bounds(2.0, 64.0, 8.0, 3.0, 65.0, 9.0),
                Aabb::from_bounds(2.0, 65.0, 8.0, 3.0, 66.0, 9.0),
            ],
            "the invisible wall stands over an unsent column"
        );
        assert!(
            !outside,
            "an entity inside the hysteresis band stays latched inside"
        );

        // The latch updates although no column is resident: 15.0 is past 13.5 + 1.
        let far = Aabb::from_bounds(14.7, 64.0, 8.2, 15.3, 65.8, 8.8);
        let mut out = Vec::new();
        collision_boxes_in_with_border_and_entities(
            &w,
            &far,
            border,
            15.0,
            8.5,
            &mut outside,
            &[],
            &mut out,
        );
        assert!(
            outside,
            "the outsideBorder latch is set from an unsent column"
        );

        // Beyond World.isValid nothing is scanned and the latch is untouched.
        let mut latch = false;
        let mut out = Vec::new();
        let invalid = Aabb::from_bounds(30_000_001.0, 64.0, 0.2, 30_000_001.6, 65.8, 0.8);
        collision_boxes_in_with_border_and_entities(
            &w,
            &invalid,
            CollisionBorder::new(0.0, 0.0, 6.0e7, 29_999_984),
            30_000_001.3,
            0.5,
            &mut latch,
            &[],
            &mut out,
        );
        assert!(out.is_empty() && !latch);
    }

    /// The far-coordinate shapes are `float` arithmetic on absolute coordinates; the oracle
    /// (`tests/block_boxes.rs`, scenes `far_all_states`/`edge_all_states`) found these values.
    #[test]
    fn float_shapes_round_like_java_far_from_the_origin() {
        assert_eq!(block_by_name("cactus").map(|b| b.0), Some(CACTUS_ID));
        assert_eq!(block_by_name("cake").map(|b| b.0), Some(CAKE_ID));
        let huge = |p: BlockPos| {
            Aabb::from_bounds(
                f64::from(p.x - 2),
                f64::from(p.y - 2),
                f64::from(p.z - 2),
                f64::from(p.x + 3),
                f64::from(p.y + 4),
                f64::from(p.z + 3),
            )
        };

        // java: cactus at (8388616, 4, -8388602) is a full-width box; the inset rounds away.
        let p = BlockPos::new(8_388_616, 4, -8_388_602);
        let w = world_with(&[(p, "cactus", 0)]);
        assert_eq!(
            collision_boxes_at(&w, p, &huge(p)),
            vec![Aabb::from_bounds(
                8_388_616.0,
                4.0,
                -8_388_602.0,
                8_388_617.0,
                4.9375,
                -8_388_601.0
            )]
        );

        // java: cake bites 4 at (8388649, 16, -8388602): minX rounds up to the next block.
        let p = BlockPos::new(8_388_649, 16, -8_388_602);
        let w = world_with(&[(p, "cake", 4)]);
        assert_eq!(
            collision_boxes_at(&w, p, &huge(p)),
            vec![Aabb::from_bounds(
                8_388_650.0,
                16.0,
                -8_388_602.0,
                8_388_650.0,
                16.5,
                -8_388_601.0
            )]
        );

        // java: a Z-axis gate at x = 29999933 keeps its odd whole-block X bounds (int + 1 in
        // double) while the 0.375 inset rounds in float.
        let p = BlockPos::new(29_999_933, 4, -29_999_932);
        let w = world_with(&[(p, "fence_gate", 0)]);
        assert_eq!(
            collision_boxes_at(&w, p, &huge(p)),
            vec![Aabb::from_bounds(
                29_999_933.0,
                4.0,
                -29_999_932.0,
                29_999_934.0,
                5.5,
                -29_999_932.0
            )]
        );
    }

    #[test]
    fn entity_tail_uses_java_query_expansion_and_strict_box_intersection() {
        // Java: World.getCollidingBoundingBoxes first queries the entity's ordinary AABB in
        // bb.expand(0.25D, 0.25D, 0.25D), then appends each non-null collision box only when it
        // strictly intersects the original bb.  Keep the query box and the returned boxes
        // distinct here so neither half can accidentally stand in for the other.
        let w = World::new();
        let mask = Aabb::from_bounds(0.0, 64.0, 0.0, 1.0, 65.0, 1.0);
        let near_query = Aabb::from_bounds(1.2, 64.0, 0.0, 1.3, 65.0, 1.0);
        let collision = Aabb::from_bounds(0.5, 64.0, 0.5, 1.0, 65.0, 1.0);
        let mover_collision = Aabb::from_bounds(0.25, 64.0, 0.25, 0.75, 65.0, 0.75);

        let mut out = Vec::new();
        collision_boxes_in_with_entities(
            &w,
            &mask,
            &[EntityCollisionCandidate::new(near_query, Some(collision))
                .with_mover_collision_box(Some(mover_collision))],
            &mut out,
        );
        assert_eq!(out, vec![collision, mover_collision]);

        // Negative control 1: touching the expanded query face is not an entity hit.
        let query_touching = Aabb::from_bounds(1.25, 64.0, 0.0, 2.0, 65.0, 1.0);
        let mut out = Vec::new();
        collision_boxes_in_with_entities(
            &w,
            &mask,
            &[EntityCollisionCandidate::new(
                query_touching,
                Some(collision),
            )],
            &mut out,
        );
        assert!(out.is_empty(), "query faces use strict intersectsWith");

        // Negative control 2: an in-range candidate whose collision box only touches bb is not
        // appended.  This is the second strict-intersection gate in World.java.
        let touching_collision = Aabb::from_bounds(1.0, 64.0, 0.25, 2.0, 65.0, 0.75);
        let mut out = Vec::new();
        collision_boxes_in_with_entities(
            &w,
            &mask,
            &[EntityCollisionCandidate::new(
                near_query,
                Some(touching_collision),
            )],
            &mut out,
        );
        assert!(out.is_empty(), "collision faces use strict intersectsWith");
    }
}
