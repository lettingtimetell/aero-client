//! What the player is looking at, and what happens when they click it.
//!
//! java: `World.rayTraceBlocks` (`World.java:888`), `Block.collisionRayTrace` (`Block.java:1073`),
//! `Entity.rayTrace` (`Entity.java:1500`), `client/renderer/EntityRenderer.getMouseOver` (`:400`)
//! and `client/multiplayer/PlayerControllerMP.java`.
//!
//! # Why this lives in `mc-client`
//!
//! `rayTraceBlocks` is a `World` method in Java, so `mc-world` would be its natural home — but
//! `mc-world` is S08's and this section owns only `mc-client/src/**`. `getMouseOver` and
//! `PlayerControllerMP` are client classes either way, and the three are useless apart, so all
//! three are here. A later pass that wants the raytrace in `mc-world` can lift
//! [`raytrace_blocks`] wholesale: it takes `&World` and returns plain data.
//!
//! # What this is *for*
//!
//! Nothing in this client had a raytrace at all, which is why left-click and right-click did
//! nothing: `Minecraft.clickMouse` and `rightClickMouse` are both driven entirely by
//! `objectMouseOver`, and with no hit result there is no block to dig and no face to place
//! against. Found by a human play-testing against a live server.

use mc_data::{BlockState, EntityType};
use mc_math::{floor_f64, Aabb, BlockPos, DVec3, Facing};
use mc_protocol::{play::s2c::GameMode, ItemStack};
use mc_world::world::LocalBlockClick;
use mc_world::World;

// 1.7.10 support: the 1.7.10 pick, ray trace and dig rules, a child so they share
// `PlayerController`'s fields.
#[path = "interact_rules_v5.rs"]
pub(crate) mod v5;

/// A block the look vector intersects. java: `MovingObjectPosition` with `typeOfHit == BLOCK`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockHit {
    /// The block that was hit.
    pub pos: BlockPos,
    /// Which face the ray entered through. java: `sideHit`
    pub face: Facing,
    /// The exact intersection point, in world coordinates. java: `hitVec`
    pub hit_vec: DVec3,
}

impl BlockHit {
    /// The hit point relative to the block's own corner, which is what
    /// `C08PacketPlayerBlockPlacement` carries as three sixteenths bytes.
    #[must_use]
    pub fn cursor(&self) -> DVec3 {
        // java: `Minecraft.rightClickMouse` narrows each `hitVec - blockPos` component to
        // `float` before `C08PacketPlayerBlockPlacement` quantises it to a sixteenths byte.
        // Keeping the subtraction in `f64` changes the byte immediately below a 1/16 boundary.
        let cursor = self.hit_vec - self.pos.to_vec3();
        DVec3::new(
            f64::from(cursor.x as f32),
            f64::from(cursor.y as f32),
            f64::from(cursor.z as f32),
        )
    }
}

/// What a block raytrace found.
///
/// **The two arms are not interchangeable and conflating them is a real bug.**
/// `Entity.rayTrace` passes `returnLastUncollidableBlock = true`, so vanilla hands back a
/// `MovingObjectPosition` even when the ray hit nothing solid — but that one carries
/// `MovingObjectType.MISS`, and `Minecraft.clickMouse`'s `switch (typeOfHit)` drops it on the
/// floor. Its only job is to clamp the *entity* search distance in `getMouseOver`. A port that
/// returned it as a block hit would have the player digging air.
// java: World.java:1050 — `new MovingObjectPosition(MovingObjectType.MISS, vec31, enumfacing, blockpos)`
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RayTrace {
    /// A block whose selection box the ray really entered.
    Block(BlockHit),
    /// The last block that refused to collide — air, or a fluid. java: `typeOfHit == MISS`.
    Miss(BlockHit),
}

impl RayTrace {
    /// The geometry, whichever arm this is. Vanilla reads `hitVec` without checking the type
    /// when it computes the entity-search clamp.
    #[must_use]
    pub fn hit(self) -> BlockHit {
        match self {
            RayTrace::Block(hit) | RayTrace::Miss(hit) => hit,
        }
    }

    /// The block, and only when one was really hit.
    #[must_use]
    pub fn as_block(self) -> Option<BlockHit> {
        match self {
            RayTrace::Block(hit) => Some(hit),
            RayTrace::Miss(_) => None,
        }
    }
}

/// An entity the look vector intersects. java: `MovingObjectPosition` with `typeOfHit == ENTITY`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityHit {
    /// The server's id for the entity.
    pub network_id: i32,
    /// java: `MovingObjectPosition.hitVec` for an ENTITY hit — `new
    /// MovingObjectPosition(pointedEntity, vec33)`, where `vec33` is the intercept point in
    /// **world** coordinates.
    ///
    /// `getMouseOver` computed it and this port dropped it, which is the whole reason
    /// `C02PacketUseEntity(INTERACT_AT)` could not be built: its payload is
    /// `hitVec - entity.pos`. `F-P1-util-12-2`.
    pub hit_vec: DVec3,
}

/// What the crosshair is on. java: `Minecraft.objectMouseOver`
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum HitResult {
    /// Nothing within reach. java: `MovingObjectType.MISS`
    #[default]
    Miss,
    /// A block.
    Block(BlockHit),
    /// An entity.
    Entity(EntityHit),
}

impl HitResult {
    /// The block, if this is a block hit.
    #[must_use]
    pub fn block(self) -> Option<BlockHit> {
        match self {
            HitResult::Block(hit) => Some(hit),
            _ => None,
        }
    }

    /// The entity, if this is an entity hit.
    #[must_use]
    pub fn entity(self) -> Option<EntityHit> {
        match self {
            HitResult::Entity(hit) => Some(hit),
            _ => None,
        }
    }
}

/// java: `Entity.getVectorForRotation(pitch, yaw)` — the unit vector for angles already selected
/// by the caller. `Entity.getLook(partialTicks)` performs interpolation before delegating here.
///
/// Vanilla uses `MathHelper.cos`/`sin`, the 65536-entry lookup table (hazard **H1**), not libm.
/// Using `f32::sin` here would put the crosshair a fraction of a degree off vanilla's, which is
/// enough to pick a different block at the edge of a face.
// java: Entity.java:1478 (getVectorForRotation)
#[must_use]
pub fn look_vector(yaw: f32, pitch: f32) -> DVec3 {
    let f = mc_math::cos(-yaw * 0.017_453_292_f32 - std::f32::consts::PI);
    let f1 = mc_math::sin(-yaw * 0.017_453_292_f32 - std::f32::consts::PI);
    let f2 = -mc_math::cos(-pitch * 0.017_453_292_f32);
    let f3 = mc_math::sin(-pitch * 0.017_453_292_f32);
    mc_math::vec::vec3(f64::from(f1 * f2), f64::from(f3), f64::from(f * f2))
}

/// `Block.collisionRayTrace`: intersect a segment with one block's selection bounds.
///
/// Returns the nearest of the six face intersections that actually lies inside the box, exactly
/// as vanilla's chain of `getIntermediateWith*Value` + `isVecInside**Bounds` does — including the
/// tie-break, which is *last assignment wins* over the six `if`s rather than a strict minimum.
// java: Block.java:1073 (collisionRayTrace)
#[must_use]
pub fn collision_ray_trace(
    bounds: Aabb,
    pos: BlockPos,
    start: DVec3,
    end: DVec3,
) -> Option<BlockHit> {
    let origin = pos.to_vec3();
    let local_start = start - origin;
    let local_end = end - origin;
    bounds
        .calculate_intercept(local_start, local_end)
        .map(|(point, face)| BlockHit {
            pos,
            face,
            hit_vec: point + origin,
        })
}

/// The state-aware `Block.collisionRayTrace` dispatch used by `World.rayTraceBlocks`.
///
/// `BlockPistonMoving` deliberately returns `null` unconditionally.  Its visual/moving
/// collision shape belongs to `TileEntityPiston`; it must not make the carrier block itself
/// targetable.  Keep this separate from `can_collide_check`: Java still reaches
/// `collisionRayTrace` for block 36 and only then receives `null`.
// java: BlockPistonMoving.java:143-146 (`collisionRayTrace` -> null)
#[must_use]
fn collision_ray_trace_for_state(
    _world: &World,
    state: BlockState,
    bounds: Aabb,
    pos: BlockPos,
    start: DVec3,
    end: DVec3,
) -> Option<BlockHit> {
    let class = mc_data::block_info(state.block()).java_class;
    if state.props().block.0 == 36 {
        None
    } else if class.ends_with("BlockStairs") {
        stairs_collision_ray_trace(state, pos, start, end)
    } else if let Some(torch) = mc_world::collision::torch_bounds(state) {
        // The one shared torch box (the outline draws it too).
        debug_assert_eq!(torch, torch_raytrace_bounds(state.meta()));
        collision_ray_trace(torch, pos, start, end)
    } else {
        collision_ray_trace(bounds, pos, start, end)
    }
}

/// The five boxes `BlockTorch.collisionRayTrace` writes immediately before calling the base
/// implementation.  `BlockRedstoneTorch` inherits this body unchanged.
///
/// Every component is a `float` widened to `double` by `setBlockBounds`, the heights included
/// (`0.2F` is 0.20000000298..., not 0.2): the F5 camera's distance and an edge-of-box crosshair
/// pick are bit-exact with vanilla only that way (`javaref/CameraRayRef.java`).
// java: BlockTorch.java:184-211; BlockRedstoneTorch.java:18
#[must_use]
fn torch_raytrace_bounds(meta: u8) -> Aabb {
    let f = 0.15_f32;
    let (min, max): ([f32; 3], [f32; 3]) = match meta {
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
    Aabb::new(widen(min), widen(max))
}

/// `BlockStairs.collisionRayTrace` is not a trace against the block's one mutable selection
/// box.  It scans the six occupied half-cubes of the stair silhouette and returns the candidate
/// furthest from `end` (therefore nearest `start`), preserving Java's pass-order tie break.
///
/// The neighbour-aware physics collision boxes are intentionally not used here: Java's
/// `BlockStairs.collisionRayTrace` only reads this stair's facing and half, then masks two of
/// its eight half-cubes.  Corners affect `addCollisionBoxesToList`, not this method.
// java: BlockStairs.java:33, 52-61, 679-720
#[must_use]
fn stairs_collision_ray_trace(
    state: BlockState,
    pos: BlockPos,
    start: DVec3,
    end: DVec3,
) -> Option<BlockHit> {
    // `field_150150_a`, indexed by EnumFacing#getHorizontalIndex plus four for top stairs.
    const SKIPPED_PASSES: [[usize; 2]; 8] = [
        [4, 5],
        [5, 7],
        [6, 7],
        [4, 6],
        [0, 1],
        [1, 3],
        [2, 3],
        [0, 2],
    ];
    // BlockStairs#getStateFromMeta: `EnumFacing.getFront(5 - (meta & 3))`.
    const HORIZONTAL_INDEX_BY_META: [usize; 4] = [3, 1, 0, 2];

    let meta = state.meta();
    let skipped = SKIPPED_PASSES
        [HORIZONTAL_INDEX_BY_META[usize::from(meta & 3)] + if meta & 4 != 0 { 4 } else { 0 }];
    let mut best: Option<(BlockHit, f64)> = None;

    for pass in 0..8 {
        if skipped.contains(&pass) {
            continue;
        }
        let min = DVec3::new(
            0.5 * f64::from((pass % 2) as u8),
            0.5 * f64::from(((pass / 4) % 2) as u8),
            0.5 * f64::from(((pass / 2) % 2) as u8),
        );
        let hit = collision_ray_trace(Aabb::new(min, min + DVec3::splat(0.5)), pos, start, end);
        if let Some(hit) = hit {
            let distance_to_end = (hit.hit_vec - end).length_squared();
            // Java starts at zero and accepts only a strictly greater distance.  This retains
            // its earliest-pass result when a ray lies on a shared half-cube boundary.
            if best.is_none_or(|(_, distance)| distance_to_end > distance) {
                best = Some((hit, distance_to_end));
            }
        }
    }
    best.map(|(hit, _)| hit)
}

/// `Block.canCollideCheck(state, hitIfLiquid)`.
///
/// `canCollideCheck` itself is overridden by three classes: `BlockAir:37` (always `false`),
/// `BlockLiquid:82` (`hitIfLiquid && level == 0`) and `BlockStairs:610` (defers to the model
/// block, which for every vanilla stair is `true`).
///
/// **The base body is `return this.isCollidable();`, and `isCollidable` has its own override:
/// `BlockFire:353` returns `false`.** The comment that used to sit here said "the whole rule is
/// not air and not a fluid", missed fire, and was wrong — the crosshair targeted fire, so
/// left-clicking a burning block dug the *fire* (a `C07` on the fire's own position) instead of
/// what is behind it. `F-P1-block-SWEEP-MOVE-1`.
// java: Block.java:512/520, BlockAir.java:37, BlockLiquid.java:82, BlockFire.java:353
#[must_use]
fn can_collide_check(state: BlockState, hit_if_liquid: bool) -> bool {
    let props = state.props();
    if props.is_air() {
        return false;
    }
    if props.material.is_liquid() {
        // java: `BlockLiquid.canCollideCheck` (`BlockLiquid.java:82`) only accepts level 0
        // source states when `hitIfLiquid` is true. Flowing levels and every liquid when the
        // flag is false remain the uncollidable/miss arm of World.rayTraceBlocks. `LEVEL` is
        // the whole 0..15 meta (`getStateFromMeta:363`), so a falling liquid (8..15) is not a
        // source even when its low three bits are zero: `meta & 7` stopped a bucket on it.
        return hit_if_liquid && state.meta() == 0;
    }
    // java: `BlockFire.isCollidable()` — the javadoc on `Block.isCollidable` even says "only used
    // by Fire".
    props.material != mc_data::Material::Fire
}

/// The selection box `Block.setBlockBoundsBasedOnState` leaves behind, in block-local space.
///
/// Walls, panes, fences, and chests use the world-owned live-neighbour oracles here because Java's
/// `collisionRayTrace` implementations call `setBlockBoundsBasedOnState` before intersecting the
/// ray. Other stateful families continue through the shared renderer helper; this is essential for
/// doors, whose lower cell owns facing/open and whose upper cell owns hinge.
#[must_use]
fn selection_bounds(world: &World, pos: BlockPos, state: BlockState) -> Aabb {
    // `Block.collisionRayTrace` reads the block's bound fields, never
    // `getSelectedBoundingBox`. The two differ for cactus alone: its selected box is the outline
    // inset by 1/16 (`BlockCactus.java:69-73`), but it never sets its bound fields, so rays meet
    // the constructor's full cube. Checked against the real classes (`javaref/CameraRayRef.java`);
    // the outline itself still draws the inset box.
    if state.block() == CACTUS {
        let b = state.props().bounds;
        return Aabb {
            min: DVec3::new(b.min_x, b.min_y, b.min_z),
            max: DVec3::new(b.max_x, b.max_y, b.max_z),
        };
    }
    // Java's stateful `collisionRayTrace` implementations first call
    // `setBlockBoundsBasedOnState`, so each raytrace shape is the live connected envelope rather
    // than an isolated registry shape or a collision rail. The renderer's outline path has its
    // own expansion and remains outside this seam.
    let b = mc_world::collision::wall_selection_bounds_at(world, pos)
        .or_else(|| mc_world::collision::pane_selection_bounds_at(world, pos))
        .or_else(|| mc_world::collision::fence_selection_bounds_at(world, pos))
        .or_else(|| mc_world::collision::chest_selection_bounds_at(world, pos))
        .or_else(|| mc_world::collision::vine_selection_bounds_at(world, pos))
        .unwrap_or_else(|| mc_render::terrain::selected_bounding_box(world, pos, state));
    let origin = pos.to_vec3();
    // Do not use `Aabb::from_bounds`: it sorts each axis, whereas
    // `Block.collisionRayTrace` reads the block's six mutable bound fields verbatim. Vanilla vine
    // metadata 0 has `min = 1`, `max = 0`; retaining that inversion makes the ray pass through.
    Aabb {
        min: b.min - origin,
        max: b.max - origin,
    }
}

/// `minecraft:cactus`, whose ray bounds are not its selected box (see [`selection_bounds`]).
const CACTUS: mc_data::BlockId = mc_data::BlockId(81);

/// The maximum number of block steps one raytrace takes. java: `int k1 = 200;`
const RAYTRACE_STEPS: i32 = 200;

/// Walk a segment through the block grid and return the first block it hits using the exact
/// flags from the `Entity.rayTrace` call (`false, false, true`).
///
/// Item helpers use [`raytrace_blocks_with_options`] because Java's
/// `Item.getMovingObjectPositionFromPlayer` calls the five-argument World overload with
/// `stopOnLiquid = useLiquids`, `ignoreBlockWithoutBoundingBox = !useLiquids`, and
/// `returnLastUncollidableBlock = false`.
///
/// **Unloaded chunks read as air**, which is `ChunkProviderClient`'s own behaviour: the ray
/// simply passes through a column that has not arrived.
// java: World.java:888
#[must_use]
pub fn raytrace_blocks(world: &World, start: DVec3, end: DVec3) -> Option<RayTrace> {
    raytrace_blocks_with_options(world, start, end, false, false, true)
}

/// Walk a segment through the block grid with Java's complete `World.rayTraceBlocks` flags.
///
/// `stop_on_liquid` is the `hitIfLiquid` value passed to `Block.canCollideCheck`; source water
/// and lava are targetable only when it is true. `ignore_block_without_bounding_box` skips the
/// block entirely when its collision shape is empty, including the otherwise-targetable
/// selection-only plants and rails. `return_last_uncollidable_block` controls whether the last
/// MISS geometry is returned when no block is hit. These three booleans are deliberately kept at
/// the call boundary instead of baking one item's behaviour into the crosshair ray.
// java: World.java:874-906, 1039-1050
#[must_use]
pub fn raytrace_blocks_with_options(
    world: &World,
    start: DVec3,
    end: DVec3,
    stop_on_liquid: bool,
    ignore_block_without_bounding_box: bool,
    return_last_uncollidable_block: bool,
) -> Option<RayTrace> {
    raytrace_walk(
        world,
        start,
        end,
        stop_on_liquid,
        ignore_block_without_bounding_box,
        return_last_uncollidable_block,
        |_, _, _| false,
    )
}

/// java: `World.rayTraceBlocks(start, end)`, the two-argument overload (`World.java:874`:
/// `stopOnLiquid`, `ignoreBlockWithoutBoundingBox` and `returnLastUncollidableBlock` all
/// `false`) that `EntityRenderer.orientCamera` casts its F5 rays with. It returns block hits
/// only: with `returnLastUncollidableBlock` false a miss is `null`.
///
/// `except` lets the ray pass through blocks as if they were air (the Tall Grass mod's hidden
/// grass). With `|_, _, _| false` the test compiles away and this is vanilla.
#[must_use]
pub fn raytrace_blocks_except(
    world: &World,
    start: DVec3,
    end: DVec3,
    except: impl Fn(&World, BlockState, BlockPos) -> bool,
) -> Option<BlockHit> {
    raytrace_walk(world, start, end, false, false, false, except).and_then(RayTrace::as_block)
}

/// The walk behind [`raytrace_blocks_with_options`] and [`raytrace_blocks_except`]. `except`
/// marks blocks that never collide, exactly like air; every vanilla caller passes a constant
/// `false`, which the generic instance folds away.
#[inline]
fn raytrace_walk(
    world: &World,
    start: DVec3,
    end: DVec3,
    stop_on_liquid: bool,
    ignore_block_without_bounding_box: bool,
    return_last_uncollidable_block: bool,
    except: impl Fn(&World, BlockState, BlockPos) -> bool,
) -> Option<RayTrace> {
    if !start.is_finite() || !end.is_finite() {
        return None;
    }
    let (mut lx, mut ly, mut lz) = (
        mc_math::floor_f64(start.x),
        mc_math::floor_f64(start.y),
        mc_math::floor_f64(start.z),
    );
    let (tx, ty, tz) = (
        mc_math::floor_f64(end.x),
        mc_math::floor_f64(end.y),
        mc_math::floor_f64(end.z),
    );

    let mut cursor = start;
    let mut pos = BlockPos::new(lx, ly, lz);
    let state = world.block_at(pos);
    let has_collision_box = !state.props().collision.is_empty();
    if (!ignore_block_without_bounding_box || has_collision_box)
        && can_collide_check(state, stop_on_liquid)
        && !except(world, state, pos)
    {
        if let Some(hit) = collision_ray_trace_for_state(
            world,
            state,
            selection_bounds(world, pos, state),
            pos,
            cursor,
            end,
        ) {
            return Some(RayTrace::Block(hit));
        }
    }

    // java: `movingobjectposition2` — the last block the ray passed through that refused to
    // collide, handed back when `returnLastUncollidableBlock` is set. `getMouseOver` sets it, and
    // it is what makes the crosshair report a face while standing inside water or air.
    let mut last_uncollidable: Option<RayTrace> = None;
    let mut budget = RAYTRACE_STEPS;
    while budget >= 0 {
        budget -= 1;
        if !cursor.is_finite() {
            return None;
        }
        if lx == tx && ly == ty && lz == tz {
            return return_last_uncollidable_block
                .then_some(last_uncollidable)
                .flatten();
        }

        let mut step_x = true;
        let mut step_y = true;
        let mut step_z = true;
        let mut plane_x = 999.0;
        let mut plane_y = 999.0;
        let mut plane_z = 999.0;

        if tx > lx {
            plane_x = f64::from(lx) + 1.0;
        } else if tx < lx {
            plane_x = f64::from(lx);
        } else {
            step_x = false;
        }
        if ty > ly {
            plane_y = f64::from(ly) + 1.0;
        } else if ty < ly {
            plane_y = f64::from(ly);
        } else {
            step_y = false;
        }
        if tz > lz {
            plane_z = f64::from(lz) + 1.0;
        } else if tz < lz {
            plane_z = f64::from(lz);
        } else {
            step_z = false;
        }

        let delta = end - cursor;
        let mut t_x = 999.0;
        let mut t_y = 999.0;
        let mut t_z = 999.0;
        if step_x {
            t_x = (plane_x - cursor.x) / delta.x;
        }
        if step_y {
            t_y = (plane_y - cursor.y) / delta.y;
        }
        if step_z {
            t_z = (plane_z - cursor.z) / delta.z;
        }
        // java: `world/World.java:997` — `if (d3 == -0.0D) { d3 = -1.0E-4D; }`.
        //
        // corrected. The literal is written `-0.0D`, but Java's `==` on doubles is
        // IEEE comparison and `+0.0 == -0.0` is **true**, so the guard fires on a zero of either
        // sign. The previous transcription added `&& t_*.is_sign_negative()`, which narrowed it
        // to `-0.0` only: for a ray starting exactly on a block corner and travelling negatively
        // on one axis and positively on another, Java nudges *both* parameters to `-1.0E-4` and
        // the tie resolves to the later axis, while the narrowed form nudged one and resolved to
        // the earlier one — a dig or place landing one block off. `== 0.0` is the whole test.
        if t_x == 0.0 {
            t_x = -1.0E-4;
        }
        if t_y == 0.0 {
            t_y = -1.0E-4;
        }
        if t_z == 0.0 {
            t_z = -1.0E-4;
        }

        let face;
        if t_x < t_y && t_x < t_z {
            face = if tx > lx { Facing::West } else { Facing::East };
            cursor = DVec3::new(plane_x, cursor.y + delta.y * t_x, cursor.z + delta.z * t_x);
        } else if t_y < t_z {
            face = if ty > ly { Facing::Down } else { Facing::Up };
            cursor = DVec3::new(cursor.x + delta.x * t_y, plane_y, cursor.z + delta.z * t_y);
        } else {
            face = if tz > lz {
                Facing::North
            } else {
                Facing::South
            };
            cursor = DVec3::new(cursor.x + delta.x * t_z, cursor.y + delta.y * t_z, plane_z);
        }

        // java: the `- 1` corrections put the cursor back inside the block it just entered when
        // the crossing landed exactly on the far plane.
        lx = mc_math::floor_f64(cursor.x) - i32::from(face == Facing::East);
        ly = mc_math::floor_f64(cursor.y) - i32::from(face == Facing::Up);
        lz = mc_math::floor_f64(cursor.z) - i32::from(face == Facing::South);
        pos = BlockPos::new(lx, ly, lz);
        let state = world.block_at(pos);
        let has_collision_box = !state.props().collision.is_empty();
        if (!ignore_block_without_bounding_box || has_collision_box)
            && can_collide_check(state, stop_on_liquid)
            && !except(world, state, pos)
        {
            if let Some(hit) = collision_ray_trace_for_state(
                world,
                state,
                selection_bounds(world, pos, state),
                pos,
                cursor,
                end,
            ) {
                return Some(RayTrace::Block(hit));
            }
        } else if !ignore_block_without_bounding_box || has_collision_box {
            last_uncollidable = Some(RayTrace::Miss(BlockHit {
                pos,
                face,
                hit_vec: cursor,
            }));
        }
    }
    return_last_uncollidable_block
        .then_some(last_uncollidable)
        .flatten()
}

/// Classify the client-side body of the two item overrides that raytrace liquids.
///
/// java: `Item.getMovingObjectPositionFromPlayer` (`Item.java:437-453`) always uses a flat
/// `5.0D` reach. `ItemGlassBottle` accepts any water material hit, while `ItemLilyPad` requires a
/// level-0 water state and an air cell above it. Both run `isBlockModifiable` (always true on the
/// client) and then `EntityPlayer.canPlayerEdit` before mutating anything. This helper only
/// selects a reachable local prediction; it does not alter the world or inventory.
#[must_use]
pub fn liquid_item_use(
    world: &World,
    eye: DVec3,
    yaw: f32,
    pitch: f32,
    held: &ItemStack,
    allow_edit: bool,
    creative: bool,
) -> Option<LiquidUsePrediction> {
    liquid_item_use_in_dimension(world, eye, yaw, pitch, held, allow_edit, creative, 0)
}

/// Resolve the client-visible half of `ItemBoat.onItemRightClick`.
///
/// The returned block is only the server spawn anchor. Java never spawns a boat entity in a
/// remote world, but it still performs the liquid ray, eye-inside-entity obstruction test,
/// contracted boat-vs-block collision test, and survival stack decrement. `collidable` contains
/// live entity boxes already expanded by each entity's collision border.
// The argument order mirrors `ItemBoat.onItemRightClick`'s resolved live inputs and keeps the
// collision probe before the local stack mutation.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn boat_item_use(
    world: &World,
    eye: DVec3,
    yaw: f32,
    pitch: f32,
    held: &ItemStack,
    collidable: &[Aabb],
    spawn_entities: &[Aabb],
    border: Option<mc_physics::collision::CollisionBorder>,
) -> Option<BlockPos> {
    if held.item_id != 333 || held.count <= 0 {
        return None;
    }
    let end = eye + look_vector(yaw, pitch) * 5.0;
    let hit = raytrace_blocks_with_options(world, eye, end, true, false, false)?.as_block()?;
    if collidable.iter().any(|bounds| bounds.is_vec_inside(eye)) {
        return None;
    }

    let spawn_block = if world.block_at(hit.pos).block().0 == 78 {
        hit.pos.down()
    } else {
        hit.pos
    };
    // EntityBoat(World, x+.5, y+1, z+.5) after setSize(1.5F, .6F), followed by Java's
    // `expand(-.1, -.1, -.1)` collision query.
    let x = f64::from(spawn_block.x) + 0.5;
    let y = f64::from(spawn_block.y) + 1.0;
    let z = f64::from(spawn_block.z) + 0.5;
    let boat = Aabb::from_bounds(x - 0.75, y, z - 0.75, x + 0.75, y + 0.6, z + 0.75)
        .expand(-0.1, -0.1, -0.1);
    // `World.getCollidingBoundingBoxes(newBoat, boatBox)` calls the boat's
    // `getCollisionBox(other)`, which returns every nearby entity's exact bounding box.
    if spawn_entities
        .iter()
        .any(|bounds| bounds.intersects_with(&boat))
    {
        return None;
    }
    // `BlockLilyPad.addCollisionBoxesToList` deliberately contributes no box to an
    // `EntityBoat`; every other block uses the ordinary world collision query.
    let collisions =
        collision_boxes_in_for_entity(world, &boat, Some(EntityType::Boat), border, x, z);
    if !collisions.is_empty() {
        return None;
    }
    Some(spawn_block)
}

/// The single-block collision query with the entity-sensitive overrides from vanilla blocks.
///
/// java: `BlockLilyPad.addCollisionBoxesToList` returns immediately for `EntityBoat`, while
/// `EntityPlayer` and every other entity receive the shallow pad box. The physics crate keeps
/// the generic world query deliberately entity-agnostic, so the client interaction caller owns
/// this narrow item-specific dispatch.
#[must_use]
pub fn collision_boxes_at_for_entity(
    world: &World,
    pos: BlockPos,
    mask: &Aabb,
    colliding_entity: Option<EntityType>,
) -> Vec<Aabb> {
    if colliding_entity == Some(EntityType::Boat) && world.block_at(pos).block() == WATERLILY_BLOCK
    {
        Vec::new()
    } else {
        mc_physics::collision::collision_boxes_at(world, pos, mask)
    }
}

fn collision_boxes_in_for_entity(
    world: &World,
    mask: &Aabb,
    colliding_entity: Option<EntityType>,
    border: Option<mc_physics::collision::CollisionBorder>,
    entity_x: f64,
    entity_z: f64,
) -> Vec<Aabb> {
    let x0 = floor_f64(mask.min.x);
    let x1 = floor_f64(mask.max.x + 1.0);
    let y0 = floor_f64(mask.min.y);
    let y1 = floor_f64(mask.max.y + 1.0);
    let z0 = floor_f64(mask.min.z);
    let z1 = floor_f64(mask.max.z + 1.0);
    let entity_inside_border = border.map(|border| border.entity_inside(entity_x, entity_z, false));
    let mut collisions = Vec::new();

    for x in x0..x1 {
        for z in z0..z1 {
            // Match `World.getCollidingBoundingBoxes`, whose `isBlockLoaded(x, 64, z)` column gate
            // is `World.isValid` alone on a client (`ChunkProviderClient.chunkExists` is `true`):
            // an absent column is scanned too, reading the shared blank chunk through
            // `World::block_at` and getting the border's stone. The y scan starts one cell below
            // the AABB floor.
            if !(-30_000_000..30_000_000).contains(&x) || !(-30_000_000..30_000_000).contains(&z) {
                continue;
            }
            for y in (y0 - 1)..y1 {
                let pos = BlockPos::new(x, y, z);
                let substitute = border.is_some_and(|border| {
                    entity_inside_border == Some(true) && !border.contains_block(pos)
                });
                if substitute {
                    let stone = mc_data::BlockState::new(
                        mc_data::block_by_name("stone").expect("vanilla stone is registered"),
                        0,
                    );
                    for bounds in stone.props().collision.boxes() {
                        let bounds =
                            bounds.offset(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));
                        let aabb = Aabb::from_bounds(
                            bounds.min_x,
                            bounds.min_y,
                            bounds.min_z,
                            bounds.max_x,
                            bounds.max_y,
                            bounds.max_z,
                        );
                        if mask.intersects_with(&aabb) {
                            collisions.push(aabb);
                        }
                    }
                } else {
                    collisions.extend(collision_boxes_at_for_entity(
                        world,
                        pos,
                        mask,
                        colliding_entity,
                    ));
                }
            }
        }
    }
    collisions
}

/// Dimension-aware form used by `ItemBucket`'s Nether water-vaporization branch.
// Dimension and creative gates are intentionally explicit at this Java call-site boundary.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn liquid_item_use_in_dimension(
    world: &World,
    eye: DVec3,
    yaw: f32,
    pitch: f32,
    held: &ItemStack,
    allow_edit: bool,
    creative: bool,
    dimension: i32,
) -> Option<LiquidUsePrediction> {
    if held.count <= 0
        || !matches!(
            held.item_id,
            GLASS_BOTTLE_ITEM
                | WATERLILY_ITEM
                | BUCKET_ITEM_ID
                | WATER_BUCKET_ITEM_ID
                | LAVA_BUCKET_ITEM_ID
        )
    {
        return None;
    }
    let end = eye + look_vector(yaw, pitch) * 5.0;
    let empty_bucket = held.item_id == BUCKET_ITEM_ID;
    let hit = raytrace_blocks_with_options(
        world,
        eye,
        end,
        !matches!(held.item_id, WATER_BUCKET_ITEM_ID | LAVA_BUCKET_ITEM_ID),
        matches!(held.item_id, WATER_BUCKET_ITEM_ID | LAVA_BUCKET_ITEM_ID),
        false,
    )?
    .as_block()?;
    let target = hit.pos.relative(hit.face);
    if !can_player_edit(world, target, hit.face, Some(held), allow_edit) {
        return None;
    }

    let state = world.block_at(hit.pos);
    if empty_bucket {
        // java: `ItemBucket.onItemRightClick` (`:62`, `:69`) and `ItemLilyPad` (`:51`) test
        // `BlockLiquid.LEVEL == 0`, and `LEVEL` is the whole 0..=15 metadata: a falling
        // column's level 8 is not a source, so it is neither emptied into the bucket nor
        // planted on (the old `meta & 7` test emptied a falling block to air locally).
        let filled_item = match state.props().material {
            mc_data::Material::Water if state.meta() == 0 => WATER_BUCKET_ITEM_ID,
            mc_data::Material::Lava if state.meta() == 0 => LAVA_BUCKET_ITEM_ID,
            _ => return None,
        };
        Some(LiquidUsePrediction::BucketFill {
            source: hit.pos,
            filled_item,
            preserve_held: creative,
        })
    } else if matches!(held.item_id, WATER_BUCKET_ITEM_ID | LAVA_BUCKET_ITEM_ID) {
        let destination = world.block_at(target);
        if !destination.props().is_air() && destination.props().material.info().solid {
            return None;
        }
        let vaporizes = held.item_id == WATER_BUCKET_ITEM_ID && dimension == -1;
        let state = (!vaporizes).then(|| {
            let block = if held.item_id == WATER_BUCKET_ITEM_ID {
                mc_data::block_by_name("flowing_water").expect("registered flowing water")
            } else {
                mc_data::block_by_name("flowing_lava").expect("registered flowing lava")
            };
            BlockState::new(block, 0)
        });
        Some(LiquidUsePrediction::BucketPour {
            target,
            state,
            replace_with_empty: !creative,
        })
    } else if held.item_id == GLASS_BOTTLE_ITEM {
        // java: `getMaterial() == Material.water`; the ray's `stopOnLiquid` flag already
        // limits this to level-0 states before this material check is reached.
        (state.props().material == mc_data::Material::Water)
            .then_some(LiquidUsePrediction::GlassBottle { water: hit.pos })
    } else if state.props().material == mc_data::Material::Water
        && state.meta() == 0
        && world.block_at(hit.pos.up()).props().is_air()
    {
        Some(LiquidUsePrediction::LilyPad {
            water: hit.pos,
            place: hit.pos.up(),
            decrement_stack: !creative,
        })
    } else {
        None
    }
}

/// java: `EntityLivingBase.onDeathUpdate` (`:402`) — `if (this.deathTime == 20) { ...
/// this.setDead(); }`, and that branch is **not** fenced behind `!worldObj.isRemote`. So a client
/// entity is `isDead` exactly once its death animation has run out, and hittable up to then.
pub const DEATH_ANIMATION_TICKS: u32 = 20;

/// The two per-entity facts [`can_be_collided_with`] needs on top of the type.
///
/// Both are things a *client* can see: `isDead` is what `EntityLivingBase.onDeathUpdate` latches
/// at `deathTime == 20`, and `hasMarker` is watched index 10 bit `16`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollisionState {
    /// java: `Entity.isDead`.
    pub dead: bool,
    /// java: `EntityArmorStand.hasMarker()` (`:920`). Ignored for every other type.
    pub marker: bool,
}

/// java: `Entity.canBeCollidedWith()` (`Entity.java:1511`) and its thirteen overrides.
///
/// **This is the predicate whose absence gets the player kicked.** `EntityRenderer.getMouseOver`
/// (`:452`) filters every pick candidate through
/// `Predicates.and(EntitySelectors.NOT_SPECTATING, e -> e.canBeCollidedWith())`, and
/// `NetHandlerPlayServer.processUseEntity` (`:928-933`) answers a `C02PacketUseEntity(ATTACK)`
/// naming an `EntityItem`, `EntityXPOrb` or `EntityArrow` with
/// `kickPlayerFromServer("Attempting to attack an invalid entity")`. A vanilla client cannot
/// produce that packet.
///
/// `kind` is `None` for a remote player, which is an `EntityLivingBase` and so `!isDead`.
///
/// The base `Entity` answer is **false**, so this is a whitelist, not a blacklist. The overrides,
/// each verified against 1.8.9 source:
///
/// | class | body | types here |
/// |---|---|---|
/// | `EntityLivingBase:2211` | `!isDead` | every `living` row, minus the dragon |
/// | `EntityArmorStand:992` | `super && !hasMarker()` | `ArmorStand` |
/// | `EntityDragon:748` | `false` | `EnderDragon` (`EntityDragonPart:43` is `true`, and parts are not `EntityList` types) |
/// | `EntityHanging:163` | `true` | `LeashKnot`, `Painting`, `ItemFrame` |
/// | `EntityBoat:166` | `!isDead` | `Boat` |
/// | `EntityMinecart:225` | `!isDead` | every minecart |
/// | `EntityTNTPrimed:53` | `!isDead` | `PrimedTnt` |
/// | `EntityFallingBlock:72` | `!isDead` | `FallingSand` |
/// | `EntityEnderCrystal:84` | `true` | `EnderCrystal` |
/// | `EntityFireball:292` | `true` | `Fireball` (`EntityLargeFireball`) |
/// | `EntitySmallFireball:79` | `false` | `SmallFireball` |
/// | `EntityWitherSkull:124` | `false` | `WitherSkull` |
///
/// Everything else inherits `Entity`'s `false`: dropped items, XP orbs, arrows, snowballs, eggs,
/// ender pearls, ender eyes, potions, XP bottles and firework rockets.
// java: Entity.java:1511 (canBeCollidedWith)
#[must_use]
pub fn can_be_collided_with(kind: Option<EntityType>, state: CollisionState) -> bool {
    use EntityType as E;
    let Some(kind) = kind else {
        // A remote player: `EntityPlayer` -> `EntityLivingBase.canBeCollidedWith`.
        return !state.dead;
    };
    match kind {
        // java: `EntityDragon.canBeCollidedWith` returns false even though the dragon is an
        // `EntityLiving`; only its (non-networked) `EntityDragonPart`s are targetable.
        E::EnderDragon => false,
        E::ArmorStand => !state.dead && !state.marker,
        E::LeashKnot | E::Painting | E::ItemFrame | E::EnderCrystal | E::Fireball => true,
        E::SmallFireball | E::WitherSkull => false,
        // `EntityBoat`, every `EntityMinecart`, `EntityTNTPrimed` and `EntityFallingBlock` all
        // spell out the same `!isDead` that `EntityLivingBase` uses.
        E::Boat
        | E::MinecartCommandBlock
        | E::MinecartRideable
        | E::MinecartChest
        | E::MinecartFurnace
        | E::MinecartTNT
        | E::MinecartHopper
        | E::MinecartSpawner
        | E::PrimedTnt
        | E::FallingSand => !state.dead,
        other => other.info().living && !state.dead,
    }
}

/// The client-local outcome of `EntityPlayer.attackTargetEntityWithCurrentItem` for a known
/// remote entity type.
///
/// This deliberately models the *whole* pre-damage gate, not merely whether an entity's
/// `attackEntityFrom` override returns `true`. `EntityHanging.hitByEntity` calls its own damage
/// override and returns that result, so a painting, leash knot, or item frame consumes the hit
/// before `attackTargetEntityWithCurrentItem` reaches the sprint/knockback block. That subtle
/// distinction is why those three types are not sprint-clearing despite their client-side
/// `attackEntityFrom` success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalAttackResult {
    /// No client-side target mutation or sprint-clearing attack branch.
    NoClientDamage,
    /// The target's remote `attackEntityFrom` result reaches the player's knockback branch.
    TargetKnockback,
    /// `EntityFireball.attackEntityFrom` reflects motion and acceleration before the common
    /// sprint knockback branch. This is only the large-fireball `Fireball` registration.
    ReflectLargeFireball,
    /// `EntityOtherPlayerMP.attackEntityFrom` returns `true` (`EntityOtherPlayerMP.java:34-37`)
    /// and a player is an `EntityLivingBase`, so the whole successful-hit branch runs: knockback,
    /// the critical and enchantment-critical emitters, thorns and `ItemStack.hitEntity`.
    RemotePlayer,
}

/// Classify the result of the client half of `EntityPlayer.attackTargetEntityWithCurrentItem`.
///
/// java: `EntityMinecart.attackEntityFrom` and `EntityBoat.attackEntityFrom` both return `true`
/// on the client (`else { return true; }`), as do `EntityEnderCrystal` and `EntityFireball`.
/// `EntityHanging.hitByEntity` is an earlier successful hook for paintings, leash knots, and
/// item frames, so the common body (including sprint clearing) is skipped for those types.
///
/// `None` is "no `EntityList` type": an untracked id or an `EntityFishHook`. A remote player has
/// no `EntityList` type either, so it is classified from its tracked kind by
/// [`tracked_attack_result`] instead.
#[must_use]
pub(crate) fn local_attack_result(kind: Option<EntityType>) -> LocalAttackResult {
    use EntityType as E;
    match kind {
        Some(
            E::Boat
            | E::MinecartCommandBlock
            | E::MinecartRideable
            | E::MinecartChest
            | E::MinecartFurnace
            | E::MinecartTNT
            | E::MinecartHopper
            | E::MinecartSpawner
            | E::EnderCrystal,
        ) => LocalAttackResult::TargetKnockback,
        Some(E::Fireball) => LocalAttackResult::ReflectLargeFireball,
        Some(E::LeashKnot | E::Painting | E::ItemFrame)
        | Some(E::SmallFireball | E::WitherSkull)
        | Some(E::PrimedTnt | E::FallingSand)
        | None => LocalAttackResult::NoClientDamage,
        Some(_) => LocalAttackResult::NoClientDamage,
    }
}

/// Whether a client-side `EntityPlayer.attackTargetEntityWithCurrentItem` call reaches the
/// successful-return sprint/knockback edge.
///
/// The player controller sends `C02PacketUseEntity` before running this local half; therefore
/// [`crate::session::Session`] uses this predicate only after the attack packet is queued.
#[must_use]
pub(crate) fn attack_cancels_sprint(kind: Option<EntityType>) -> bool {
    matches!(
        local_attack_result(kind),
        LocalAttackResult::TargetKnockback | LocalAttackResult::ReflectLargeFireball
    )
}

/// [`local_attack_result`] for a tracked target, including the classes without an `EntityList`
/// type.
///
/// java: `EntityOtherPlayerMP` overrides `attackEntityFrom` to return `true`
/// (`EntityOtherPlayerMP.java:34-37`), unlike `EntityLivingBase`'s remote-world `false`.
/// `EntityFishHook` keeps `Entity.attackEntityFrom`'s `false`.
#[must_use]
pub(crate) fn tracked_attack_result(kind: crate::sync::TrackedKind) -> LocalAttackResult {
    match kind {
        crate::sync::TrackedKind::Player => LocalAttackResult::RemotePlayer,
        crate::sync::TrackedKind::Type(kind) => local_attack_result(Some(kind)),
        crate::sync::TrackedKind::FishingBobber => LocalAttackResult::NoClientDamage,
    }
}

/// `Enchantment.sharpness.effectId`.
const SHARPNESS_ENCHANTMENT_ID: i16 = 16;

/// `EnchantmentHelper.getModifierForCreature(heldItem, EnumCreatureAttribute.UNDEFINED)`.
///
/// java: `EnchantmentHelper.java:215-221` sums `calcDamageByCreature(lvl, creature)` over every
/// registered entry of the held stack's `ench` list, in float. Only `EnchantmentDamage` overrides
/// that method (`EnchantmentDamage.java:72-75`), and for an `UNDEFINED` creature only its
/// `damageType 0` row, Sharpness, is non-zero: `level * 1.25F`. A player's creature attribute is
/// `UNDEFINED`, so this is the `f1` that gates `onEnchantmentCritical`
/// (`EntityPlayer.java:1379-1382`) for a remote-player target.
#[must_use]
pub(crate) fn undefined_creature_damage_modifier(held: Option<&ItemStack>) -> f32 {
    let Some(list) = held
        .and_then(|stack| stack.nbt.as_ref())
        .and_then(|nbt| nbt.get_list("ench"))
    else {
        return 0.0;
    };
    let mut modifier = 0.0_f32;
    for entry in list {
        let mc_nbt::Nbt::Compound(entry) = entry else {
            continue;
        };
        if entry.get_i16("id").unwrap_or(0) == SHARPNESS_ENCHANTMENT_ID {
            modifier += f32::from(entry.get_i16("lvl").unwrap_or(0)) * 1.25;
        }
    }
    modifier
}

/// The durability `ItemStack.hitEntity` asks of the held stack after a successful hit on an
/// `EntityLivingBase` (`EntityPlayer.java:1410-1418`).
///
/// java: `ItemSword.hitEntity` calls `damageItem(1, attacker)` (`ItemSword.java:54-58`) and
/// `ItemTool.hitEntity`, inherited by the pickaxe, axe and shovel, calls `damageItem(2, attacker)`
/// (`ItemTool.java:44-48`). `ItemSaddle.hitEntity` only saddles a pig, and every other item
/// keeps `Item.hitEntity`'s no-op `false`.
#[must_use]
pub(crate) fn hit_entity_item_damage(held: Option<&ItemStack>) -> u8 {
    let Some(item_id) = held.and_then(|stack| u16::try_from(stack.item_id).ok()) else {
        return 0;
    };
    let props = mc_data::item_props(mc_data::ItemId(item_id));
    if !props.registered {
        return 0;
    }
    match props.kind {
        mc_data::ItemKind::Sword => 1,
        mc_data::ItemKind::Pickaxe | mc_data::ItemKind::Axe | mc_data::ItemKind::Shovel => 2,
        _ => 0,
    }
}

/// The client-visible facts needed by the small, wire-facing part of
/// `EntityPlayer.interactWith` that the input driver can decide without running an entity world.
///
/// This is deliberately not an entity simulation. `Entity.interactFirst` has several branches
/// whose answer depends on state the multiplayer client does not receive (a leash target, a
/// horse's temper, a villager's open trade, and so on). The compatibility wrapper below keeps
/// the old four-field input stable; callers that own more of the live state use
/// [`EntityInteractionFacts`] and [`entity_interaction_consumed_with_facts`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EntityInteractionState {
    /// java: `EntityAgeable.isChild()` / the subclass equivalent.
    pub child: bool,
    /// java: `EntityPlayer.capabilities.isCreativeMode`.
    pub creative: bool,
    /// WAVE8RCLICK: java: `EntityPig.getSaddled()` (index 16, bit 0).
    pub pig_saddled: bool,
    /// WAVE8RCLICK: java: `EntitySheep.getSheared()` (index 16, high bit).
    pub sheep_sheared: bool,
}

/// Extra client-visible state for the entity half of `Minecraft.rightClickMouse`.
///
/// Every `Option` is intentional: a watcher bag can be partial, and several Java predicates have
/// no wire owner at all. `None` means "do not manufacture the untargeted C08" rather than
/// pretending an absent value is the entity's default. The right-click driver fills the fields
/// that are truly available from `Watched`/the riding tracker; unit callers can provide precise
/// values for the remaining branches.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EntityInteractionFacts {
    /// The original WAVE8 state, kept separate so existing callers remain source-compatible.
    pub base: EntityInteractionState,
    /// `EntitySheep.getFleeceColor()` at the click edge.  The live driver supplies the
    /// generation's low-nibble metadata (defaulting to Java's white `0` when the spawn watcher
    /// is omitted); keeping the snapshot here lets the application reject a same-generation
    /// server correction that arrived after classification instead of recolouring it again.
    pub sheep_fleece_metadata: Option<u8>,
    /// `EntityMinecartEmpty.riddenByEntity` (and the horse's rider) as known by S1B/local riding.
    pub rideable_occupancy: RideableOccupancy,
    /// `EntityPlayer.isSneaking()` at the click edge.
    pub sneaking: Option<bool>,
    /// The local player's network id.  The session-owned player is not in the remote entity
    /// tracker, so a lead prediction needs this id explicitly when it installs the local leash
    /// holder edge.
    pub local_player_entity_id: Option<i32>,
    /// The clicked living entity's raw `EntityLiving.leashedToEntity` id, as received from S1B or
    /// a client-local lead prediction.  `Some` means the target is already leashed, even when the
    /// holder is not currently loaded.
    pub leashed_to: Option<i32>,
    /// `EntityTameable.isOwner(player)` at the click edge.  It is derived from the watched owner
    /// UUID and the session-owned local UUID by the driver; a missing/invalid owner is false.
    pub tame_owner: Option<bool>,
    /// `PlayerCapabilities.isCreativeMode` for the lead's copied-stack rule.  This is separate
    /// from `base.creative`, whose legacy callers still derive from the game-mode label.
    pub lead_creative: Option<bool>,
    /// `EntityHorse.isTame()` (watcher index 16, bit 2).
    pub horse_tame: Option<bool>,
    /// `EntityHorse.getHorseType()` (watcher index 19).
    pub horse_type: Option<i32>,
    /// `EntityHorse.isChested()` (watcher index 16, bit 8).
    pub horse_chested: Option<bool>,
    /// `EntityWolf.isTamed()` (watcher index 16, bit 4).
    pub wolf_tamed: Option<bool>,
    /// `EntityWolf.isAngry()` (watcher index 16, bit 2).
    pub wolf_angry: Option<bool>,
    /// Whether the watched wolf health is below 20.0, the client branch's heal gate.
    pub wolf_hurt: Option<bool>,
    /// `EntityWolf.getCollarColor()` in the dye-damage numbering scheme (watcher index 20).
    pub wolf_collar_dye_damage: Option<i32>,
    /// `EntityOcelot.isTamed()` (watcher index 16, bit 4).
    pub ocelot_tamed: Option<bool>,
    /// `EntityLivingBase.isEntityAlive()` as proved by watched health.
    pub villager_alive: Option<bool>,
    /// `EntityVillager.isTrading()`, which is client-local/server-owned and normally unknown.
    pub villager_trading: Option<bool>,
    /// `EntityAnimal`'s adult `growingAge == 0 && inLove <= 0` result when a caller owns it.
    pub animal_breeding_ready: Option<bool>,
    /// Whether a horse's watched health is below its client-side max-health attribute. Java's
    /// horse food branch only consumes ordinary food on a tame adult when this is true (golden
    /// carrot/apple are the breeding exception). The attribute is not an entity watcher, so the
    /// driver supplies this as a derived fact and partial fixtures may leave it unknown.
    pub horse_hurt: Option<bool>,
    /// `EntityHorse`'s client-local temper field. It is not on the wire; the tracker keeps the
    /// generation-owned value so untamed horse feed can reproduce the local temper gate without
    /// borrowing server authority.
    pub horse_temper: Option<i32>,
    /// `EntityMinecartFurnace.posX/Y/Z` at the interaction edge. The furnace cart's client
    /// `interactFirst` body always records the cart-minus-player horizontal vector, even when the
    /// held item is not coal; keeping the coordinates optional avoids manufacturing that private
    /// state for a partial unit fixture.
    pub furnace_minecart_cart_position: Option<DVec3>,
    /// `EntityPlayer.posX/Y/Z` at the furnace-cart interaction edge.
    pub furnace_minecart_player_position: Option<DVec3>,
    /// `PlayerCapabilities.isCreativeMode` at the furnace-cart interaction edge.  This is kept
    /// separate from the legacy `base.creative` compatibility field because the live controller
    /// owns its capability bit independently of the game-mode label after an S39 correction.
    pub furnace_minecart_creative: Option<bool>,
}

/// The rider relationship needed by the client half of an empty minecart and an adult horse.
///
/// A player rider is distinct from another entity for `EntityMinecartEmpty`: the former returns
/// `true` to `interactFirst`, while the latter returns `false`. Horses only care whether *any*
/// rider is present, but sharing this small state keeps the driver scan in one place.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RideableOccupancy {
    /// No rider snapshot is available.
    #[default]
    Unknown,
    /// The vehicle has no rider.
    Empty,
    /// A player (remote or the local session player) is riding it.
    Player,
    /// A non-player entity is riding it.
    Other,
}

/// The client-side result of `ItemSaddle.itemInteractionForEntity` for a pig.
///
/// Java returns `true` for **every** pig that reaches the item override, including a child or
/// already-saddled pig.  Only an adult, unsaddled pig gets the two visible local mutations.  The
/// creative path passes a copy of the held stack into the override, so its decrement is likewise
/// suppressed even though the pig's watched saddle bit is still set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SaddleInteraction {
    /// `ItemSaddle.itemInteractionForEntity` consumed the entity click.
    pub consumed: bool,
    /// `EntityPig.setSaddled(true)` must be reflected in the local watched state.
    pub set_saddled: bool,
    /// The selected survival stack loses one saddle.
    pub decrement_stack: bool,
}

/// The client-visible result of `ItemNameTag.itemInteractionForEntity`.
///
/// Java writes the exact `display.Name` string into watched index 2, enables the target's
/// persistence flag, and decrements the stack passed to the item override. Creative mode passes
/// a copy of the selected stack, so the target mutations still happen while the inventory count
/// remains unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameTagInteraction {
    /// The exact custom name copied from `display.Name`.
    pub name: String,
    /// The selected survival stack loses one name tag.
    pub decrement_stack: bool,
}

/// The client-visible half of `EntityLiving.interactFirst` for a held lead.
///
/// The relation itself is still a local shadow: the server's S1B attach packet remains the
/// authority, and a generation check at the tracker boundary prevents a click captured for an
/// old network-id allocation from mutating its replacement.  Clearing a leash never decrements
/// the selected stack on a remote world; attaching does so only for the survival copy of the
/// stack passed to `interactFirst`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LeadInteraction {
    /// `EntityLiving.setLeashedToEntity(player, true)` reached the client object.
    pub attach_to_player: bool,
    /// `EntityLiving.clearLeashed(true, !creative)` reached the client object.
    pub clear_from_player: bool,
    /// The selected survival lead is decremented after a successful attach.
    pub decrement_stack: bool,
}

/// Whether a type implements `IMob` for `EntityLiving.allowLeashing` purposes.
///
/// 1.8.9's default `allowLeashing()` is `!getLeashed() && !(this instanceof IMob)`.  The
/// generated entity table records living-ness but not Java interfaces, so keep this exact small
/// roster here.  `EntityHorse` has its own undead override and is handled by
/// [`lead_interaction`].
fn is_immob_leash_target(kind: EntityType) -> bool {
    matches!(
        kind,
        EntityType::Creeper
            | EntityType::Skeleton
            | EntityType::Spider
            | EntityType::Giant
            | EntityType::Zombie
            | EntityType::Slime
            | EntityType::Ghast
            | EntityType::PigZombie
            | EntityType::Enderman
            | EntityType::CaveSpider
            | EntityType::Silverfish
            | EntityType::Blaze
            | EntityType::LavaSlime
            | EntityType::EnderDragon
            | EntityType::WitherBoss
            | EntityType::Witch
            | EntityType::Endermite
            | EntityType::Guardian
    )
}

/// Classify the boolean returned by the client-reachable `EntityLiving.interactFirst` lead branch.
///
/// `None` means the held stack is not a lead, the target is not an `EntityLiving`, or a subtype
/// (currently Villager) owns the post-allowLeashing interaction branch; `Some(false)` is a known
/// invalid lead click and must therefore preserve Java's untargeted C08 fall-through; `Some(true)`
/// is a consumed clear/attach branch. The local player id is not required to decide an ordinary
/// attach, but is used when an already-leashed target can be recognised as held by this player.
fn lead_consumed(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<bool> {
    let kind = kind?;
    if !kind.info().living || held.map(|stack| stack.item_id) != Some(LEAD_ITEM_ID) {
        return None;
    }

    // Java checks this before both `allowLeashing()` and every subtype `interact` body.  A
    // plugin/server leash can therefore still be cleared even on an otherwise non-leashable
    // mob, provided this client knows the holder is the local player.
    if facts
        .leashed_to
        .zip(facts.local_player_entity_id)
        .is_some_and(|(holder, player)| holder == player)
    {
        return Some(true);
    }

    // `allowLeashing()` rejects an already-leashed target, regardless of who holds it.  An
    // unknown target relation is deliberately not manufactured as an attach: the driver always
    // supplies the raw S1B edge, but partial unit fixtures retain the conservative no-prediction
    // answer when they leave it absent.
    if facts.leashed_to.is_some() {
        return Some(false);
    }

    // EntityAmbientCreature (the Bat) overrides allowLeashing() to false, while EntityWolf adds
    // an angry gate before the base IMob check. Keep those subtype gates here rather than
    // accidentally predicting an attach for every living watcher row. Villager also overrides
    // allowLeashing(), but its own live/trading interaction still runs after that decline, so it
    // deliberately falls through to the subtype classifier below.
    if kind == EntityType::Bat || (kind == EntityType::Wolf && facts.wolf_angry == Some(true)) {
        return Some(false);
    }
    if kind == EntityType::Villager {
        return None;
    }

    if is_immob_leash_target(kind) {
        return Some(false);
    }
    if kind == EntityType::EntityHorse
        && facts
            .horse_type
            .is_some_and(|horse_type| matches!(horse_type, 3 | 4))
    {
        // java: `EntityHorse.allowLeashing()` -> `!isUndead() && super.allowLeashing()`.
        return Some(false);
    }

    // Tamed wolves and ocelots retain the base allow-leashing gate but can only be attached by
    // their owner.  The driver resolves the watched owner UUID against the local UUID; a tamed
    // entity without a valid owner is therefore a known decline rather than a guessed attach.
    if matches!(kind, EntityType::Wolf | EntityType::Ozelot)
        && (kind == EntityType::Wolf && facts.wolf_tamed == Some(true)
            || kind == EntityType::Ozelot && facts.ocelot_tamed == Some(true))
        && facts.tame_owner != Some(true)
    {
        return Some(false);
    }

    Some(true)
}

/// Classify the local mutations following a successful client-side lead interaction.
///
/// Invalid lead clicks intentionally return `None` here even though [`lead_consumed`] returns
/// `Some(false)`: the driver still needs the boolean to choose Java's C08 fall-through, but must
/// not install a local leash shadow or decrement inventory for the decline.
#[must_use]
pub fn lead_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<LeadInteraction> {
    if lead_consumed(kind, held, facts) != Some(true) {
        return None;
    }
    let clear_from_player = facts
        .leashed_to
        .zip(facts.local_player_entity_id)
        .is_some_and(|(holder, player)| holder == player);
    Some(LeadInteraction {
        attach_to_player: !clear_from_player,
        clear_from_player,
        decrement_stack: !clear_from_player && !facts.lead_creative.unwrap_or(facts.base.creative),
    })
}

const SADDLE_ITEM_ID: i16 = 329;

/// Classify the client-reachable `ItemNameTag.itemInteractionForEntity` override.
#[must_use]
pub fn name_tag_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    creative: bool,
) -> Option<NameTagInteraction> {
    let kind = kind?;
    if !kind.info().living || kind == EntityType::ArmorStand {
        return None;
    }
    let stack = held.filter(|stack| stack.item_id == NAME_TAG_ITEM_ID)?;
    let name = stack
        .nbt
        .as_ref()?
        .get_compound("display")?
        .get_str("Name")?;
    Some(NameTagInteraction {
        name: name.to_owned(),
        decrement_stack: !creative,
    })
}

/// Whether `EntityLiving.interactFirst` reaches the held `ItemNameTag` override.
///
/// Most living subclasses decline a named tag and let `EntityPlayer.interactWith` invoke the
/// item. These are the three earlier consuming paths in 1.8.9: clearing a leash held by this
/// player, opening an adult live non-trading Villager, and opening a tame adult horse while
/// sneaking.
#[must_use]
pub fn name_tag_override_reached(
    kind: Option<EntityType>,
    facts: EntityInteractionFacts,
    leashed_to_player: bool,
) -> bool {
    if leashed_to_player {
        return false;
    }
    match kind {
        Some(EntityType::Villager) => {
            facts.base.child
                || facts.villager_alive != Some(true)
                || facts.villager_trading == Some(true)
        }
        Some(EntityType::EntityHorse) => {
            facts.base.child || facts.horse_tame != Some(true) || facts.sneaking != Some(true)
        }
        _ => true,
    }
}

/// Classify the client-reachable `ItemSaddle.itemInteractionForEntity` override.
///
/// java: `ItemStack.interactWithEntity` (`ItemStack.java:406-409`) calls
/// `ItemSaddle.itemInteractionForEntity` after `EntityPig.interactFirst` declines on the remote
/// world (`EntityPig.java:122-137`, where `worldObj.isRemote` is an explicit decline branch).
/// The local sound call is intentionally not represented: `World.playSoundAtEntity` routes to
/// the silent `IWorldAccess.playSound` path on this client, while the server's S29 supplies the
/// audible copy.
#[must_use]
pub fn saddle_item_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    state: EntityInteractionState,
) -> Option<SaddleInteraction> {
    if kind != Some(EntityType::Pig) || held.map(|stack| stack.item_id) != Some(SADDLE_ITEM_ID) {
        return None;
    }

    let mut result = SaddleInteraction {
        // ItemSaddle returns true for a pig even when its mutation gate below is false.
        consumed: true,
        ..SaddleInteraction::default()
    };
    if !state.pig_saddled && !state.child {
        result.set_saddled = true;
        // EntityPlayer.interactWith gives the item override a copy in creative mode, so only
        // survival mutates the selected inventory stack.
        result.decrement_stack = !state.creative;
    }
    Some(result)
}

/// Classifies the result of the client-side `EntityPlayer.interactWith` chain for an entity.
///
/// The return value is an `Option` because `None` is meaningful: it means the entity's
/// `interactFirst` body still needs live state this client does not own, so the caller must keep
/// its conservative no-fallback policy instead of manufacturing an untargeted use packet.
/// `Some(true)` is a consumed interaction (no `sendUseItem` fall-through); `Some(false)` is a
/// known client-side decline (the untargeted `C08` follows when a stack is held).
// java: EntityPlayer.interactWith (`EntityPlayer.java:1236`), EntityLiving.interactFirst
// (`EntityLiving.java:1091`), and the concrete item/entity overrides in 1.8.9.
#[must_use]
pub fn entity_interaction_consumed(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    state: EntityInteractionState,
) -> Option<bool> {
    entity_interaction_consumed_with_facts(
        kind,
        held,
        EntityInteractionFacts {
            base: state,
            ..EntityInteractionFacts::default()
        },
    )
}

const BOWL_ITEM_ID: i16 = 281;
/// `minecraft:mushroom_stew`.
pub const MUSHROOM_STEW_ITEM_ID: i16 = 282;
/// `minecraft:bucket`.
pub const BUCKET_ITEM_ID: i16 = 325;
/// `minecraft:milk_bucket`.
pub const MILK_BUCKET_ITEM_ID: i16 = 335;
/// `minecraft:water_bucket`.
pub const WATER_BUCKET_ITEM_ID: i16 = 326;
/// `minecraft:lava_bucket`.
pub const LAVA_BUCKET_ITEM_ID: i16 = 327;
const DYE_ITEM_ID: i16 = 351;
const GOLDEN_APPLE_ITEM_ID: i16 = 322;
const NAME_TAG_ITEM_ID: i16 = 421;
const SHEARS_ITEM_ID: i16 = 359;
const SPAWN_EGG_ITEM_ID: i16 = 383;
/// `minecraft:lead` / `Items.leash`.
pub const LEAD_ITEM_ID: i16 = 420;
const FISH_ITEM_ID: i16 = 349;
/// `minecraft:fishing_rod`.
pub const FISHING_ROD_ITEM_ID: i16 = 346;
const BONE_ITEM_ID: i16 = 352;
const WHEAT_SEEDS_ITEM_ID: i16 = 295;
const WHEAT_ITEM_ID: i16 = 296;
const SUGAR_ITEM_ID: i16 = 353;
const APPLE_ITEM_ID: i16 = 260;
const GOLDEN_CARROT_ITEM_ID: i16 = 396;
const CARROT_ITEM_ID: i16 = 391;
const YELLOW_FLOWER_ITEM_ID: i16 = 37;
const HAY_BLOCK_ITEM_ID: i16 = 170;
const CHEST_ITEM_ID: i16 = 54;
/// `minecraft:carrot_on_a_stick`.
pub const CARROT_ON_A_STICK_ITEM_ID: i16 = 398;

fn is_ageable(kind: Option<EntityType>) -> bool {
    matches!(
        kind,
        Some(
            EntityType::Pig
                | EntityType::Sheep
                | EntityType::Cow
                | EntityType::Chicken
                | EntityType::Wolf
                | EntityType::MushroomCow
                | EntityType::Ozelot
                | EntityType::EntityHorse
                | EntityType::Rabbit
                | EntityType::Villager
        )
    )
}

fn is_wolf_favorite_meat(item_id: Option<i16>) -> bool {
    matches!(
        item_id,
        Some(
            319 // porkchop
            | 320 // cooked_porkchop
            | 363 // beef
            | 364 // cooked_beef
            | 365 // chicken
            | 366 // cooked_chicken
            | 367 // rotten_flesh
            | 411 // rabbit
            | 412 // cooked_rabbit
            | 423 // mutton
            | 424 // cooked_mutton
        )
    )
}

fn is_horse_feed(item_id: Option<i16>) -> bool {
    matches!(
        item_id,
        Some(
            WHEAT_ITEM_ID
                | SUGAR_ITEM_ID
                | HAY_BLOCK_ITEM_ID
                | APPLE_ITEM_ID
                | GOLDEN_CARROT_ITEM_ID
                | GOLDEN_APPLE_ITEM_ID
        )
    )
}

fn is_horse_armor(item_id: Option<i16>) -> bool {
    matches!(item_id, Some(417..=419))
}

fn animal_breeding_item(kind: Option<EntityType>, item_id: Option<i16>) -> bool {
    match kind {
        Some(EntityType::Pig) => item_id == Some(CARROT_ITEM_ID),
        Some(EntityType::Chicken) => item_id == Some(WHEAT_SEEDS_ITEM_ID),
        Some(EntityType::Rabbit) => matches!(
            item_id,
            Some(CARROT_ITEM_ID | GOLDEN_CARROT_ITEM_ID | YELLOW_FLOWER_ITEM_ID)
        ),
        Some(EntityType::Ozelot) => item_id == Some(FISH_ITEM_ID),
        Some(EntityType::Wolf) => is_wolf_favorite_meat(item_id),
        Some(EntityType::Sheep | EntityType::Cow | EntityType::MushroomCow) => {
            item_id == Some(WHEAT_ITEM_ID)
        }
        _ => false,
    }
}

/// The client-owned portion of `EntityAnimal.interact` after a breeding item has passed the
/// entity's age/love gate. `EntityPlayer.interactWith` invokes this method before any server
/// response, so the selected survival stack really does change locally. The eventual child,
/// love state and status-18 packet remain server-authoritative and are deliberately not faked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnimalFeedInteraction {
    /// `EntityAnimal.setInLove(player)`'s local 600-tick timer.
    pub starts_in_love: bool,
    /// `EntityAgeable.field_175503_c = 40`, which drives ten client happy-particle emissions for
    /// an ordinary baby feed. The age byte itself stays clamped at -1 on a multiplayer client.
    pub happy_particles: bool,
    /// `EntityPlayer.interactWith` passes the real stack in survival and a copy in creative.
    pub decrement_stack: bool,
}

/// The client-visible result of `ItemDye.itemInteractionForEntity` for a sheep.
///
/// `EntityPlayer.interactWith` queues the targeted `INTERACT_AT` and `INTERACT` C02 packets
/// before it reaches this item override.  The caller must apply this result only after that
/// pair: `ItemDye` returns `true` for every sheep click, so the untargeted C08 fallback is
/// suppressed even when the sheep is already the requested colour or has been sheared.
///
/// The sheep's fleece byte is the **metadata** numbering (`white == 0`, `black == 15`), while
/// the held dye's damage is the reversed `EnumDyeColor.byDyeDamage` numbering.  Keeping the
/// conversion here prevents the collar/dye damage numbering from leaking into the watcher
/// mutation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SheepDyeInteraction {
    /// `ItemDye.itemInteractionForEntity` always returns `true` for a sheep.
    pub consumed: bool,
    /// The fleece metadata selected by `EnumDyeColor.byDyeDamage(stack.getMetadata())`.
    ///
    /// This is present even for a no-op so a caller can retain the exact Java callback result;
    /// it must write the watcher only when [`Self::recolor`] is true.
    pub new_fleece_metadata: u8,
    /// Whether Java's remote `setFleeceColor` call changes the watched byte.
    pub recolor: bool,
    /// Whether the selected survival stack is decremented by the item override.
    pub decrement_stack: bool,
}

/// Classify the client-local half of `ItemDye.itemInteractionForEntity` for a sheep.
///
/// `fleece_metadata` is the low nibble from `EntitySheep`'s watcher index 16.  The low-nibble
/// mask is repeated here because Java's `getFleeceColor` applies `& 15` before
/// `EnumDyeColor.byMetadata`; callers that have already decoded the watcher can pass its
/// metadata unchanged.  `child` is intentionally not consulted: unlike wheat feeding and
/// shearing, Java's dye override has no age gate and recolours both adult and child sheep.
///
/// `None` means the target/held item is not this override.  `Some` means the callback consumed
/// the click, including matching-colour and sheared no-ops.  Therefore a live caller should
/// queue the targeted C02 pair first, apply `recolor`/`decrement_stack` second, and never append
/// the untargeted C08 for this result.
#[must_use]
pub fn sheep_dye_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    state: EntityInteractionState,
    fleece_metadata: u8,
) -> Option<SheepDyeInteraction> {
    if kind != Some(EntityType::Sheep) {
        return None;
    }
    let stack = held.filter(|stack| stack.item_id == DYE_ITEM_ID)?;

    let new_fleece_metadata =
        mc_data::item::dye_color_by_dye_damage(i32::from(stack.damage)).metadata;
    let recolor = !state.sheep_sheared && (fleece_metadata & 0x0f) != new_fleece_metadata;
    Some(SheepDyeInteraction {
        consumed: true,
        new_fleece_metadata,
        recolor,
        decrement_stack: recolor && !state.creative,
    })
}

/// Classify the local half of an animal breeding-item click.
///
/// `animal_breeding_ready == None` intentionally produces no prediction: `inLove` is a private
/// client field, so an absent owner must not turn a partial watcher fixture into a guessed adult
/// breed. A baby still consumes a valid item and starts the 40-tick happy-particle timer even
/// though the Java client cannot advance its ordinary age past the clamped watcher byte.
#[must_use]
pub fn animal_feed_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<AnimalFeedInteraction> {
    if !animal_breeding_item(kind, held.map(|stack| stack.item_id)) {
        return None;
    }
    if facts.base.child {
        return Some(AnimalFeedInteraction {
            starts_in_love: false,
            happy_particles: true,
            decrement_stack: !facts.base.creative,
        });
    }
    (facts.animal_breeding_ready == Some(true)).then_some(AnimalFeedInteraction {
        starts_in_love: true,
        happy_particles: false,
        decrement_stack: !facts.base.creative,
    })
}

/// The client-local result of an adult cow/mooshroom bucket click. A count-one bucket is
/// replaced in the selected slot; a larger stack is decremented and the returned milk bucket is
/// inserted by `InventoryPlayer.addItemStackToInventory` (or dropped server-side when full).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CowMilkInteraction {
    /// Replace the selected bucket with one milk bucket (the count-one branch).
    pub replace_selected: bool,
    /// Add one milk bucket to the remaining inventory (the count-greater-than-one branch).
    pub add_to_inventory: bool,
    /// Survival-only selected-stack decrement. Creative cow buckets fall through to super.
    pub decrement_stack: bool,
}

/// Classify `EntityCow.interact`, including its inherited mooshroom bucket path.
#[must_use]
pub fn cow_milk_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<CowMilkInteraction> {
    if !matches!(kind, Some(EntityType::Cow | EntityType::MushroomCow))
        || held.map(|stack| stack.item_id) != Some(BUCKET_ITEM_ID)
        || facts.base.child
        || facts.base.creative
    {
        return None;
    }
    let count_one = held.is_some_and(|stack| stack.count == 1);
    Some(CowMilkInteraction {
        replace_selected: count_one,
        add_to_inventory: !count_one,
        decrement_stack: true,
    })
}

/// The client-visible mooshroom bowl result. For count > 1, the driver must first ask its flat
/// inventory model whether `addItemStackToInventory` has room; a full inventory means Java falls
/// through without decrementing. Creative mode still inserts the stew but leaves the bowl stack.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MooshroomBowlInteraction {
    /// Replace a count-one selected bowl directly with mushroom stew.
    pub replace_selected: bool,
    /// Try to add one stew to the remaining inventory (count > 1).
    pub add_to_inventory: bool,
    /// Survival count > 1 decrement, only after the add succeeds.
    pub decrement_stack: bool,
}

/// Classify an adult mooshroom's bowl interaction, including the count-one replacement and the
/// count-greater-than-one inventory/decrement split.
#[must_use]
pub fn mooshroom_bowl_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<MooshroomBowlInteraction> {
    if kind != Some(EntityType::MushroomCow)
        || held.map(|stack| stack.item_id) != Some(BOWL_ITEM_ID)
        || facts.base.child
    {
        return None;
    }
    let count_one = held.is_some_and(|stack| stack.count == 1);
    Some(MooshroomBowlInteraction {
        replace_selected: count_one,
        add_to_inventory: !count_one,
        decrement_stack: !count_one && !facts.base.creative,
    })
}

/// Shearing an adult mooshroom is unusual: `setDead()` and the large explosion particle are
/// outside the server guard, while the replacement cow, mushrooms, shears damage and sound are
/// server-only. The tracker therefore removes only this exact client generation and queues the
/// one local particle; the server's replacement spawn/correction owns the rest.
#[must_use]
pub fn mooshroom_shears_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> bool {
    kind == Some(EntityType::MushroomCow)
        && held.map(|stack| stack.item_id) == Some(SHEARS_ITEM_ID)
        && !facts.base.child
}

/// Java's horse feed table. The tuple is `(heal, growth, temper increase)`; the values are the
/// 1.8.9 `EntityHorse.interact` literals, not server attribute snapshots.
fn horse_feed_values(item_id: Option<i16>) -> Option<(f32, i32, i32)> {
    match item_id {
        Some(WHEAT_ITEM_ID) => Some((2.0, 20, 3)),
        Some(SUGAR_ITEM_ID) => Some((1.0, 30, 3)),
        Some(HAY_BLOCK_ITEM_ID) => Some((20.0, 180, 0)),
        Some(APPLE_ITEM_ID) => Some((3.0, 60, 3)),
        Some(GOLDEN_CARROT_ITEM_ID) => Some((4.0, 60, 5)),
        Some(GOLDEN_APPLE_ITEM_ID) => Some((10.0, 240, 10)),
        _ => None,
    }
}

/// The client-local part of `EntityHorse.interact` for feed/chest. Armor, saddle and mounting
/// retain their existing C02/S1B paths. `growth` is a positive `addGrowth` amount for a child;
/// `heal` is applied against the local watched health when the horse is hurt; `temper_increase`
/// advances the generation-owned private temper field.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HorseItemInteraction {
    /// `EntityHorse.addGrowth(j)` amount; zero for adults.
    pub growth: i32,
    /// The local `heal(f)` amount when the hurt gate was true.
    pub heal: f32,
    /// The client-local `setInLove` edge for tame adults fed golden carrot/apple.
    pub starts_in_love: bool,
    /// The client-local `increaseTemper(k)` amount.
    pub temper_increase: i32,
    /// `setChested(true)` for an adult donkey/mule with an empty chest slot.
    pub chest: bool,
    /// Survival decrement after any local horse mutation.
    pub decrement_stack: bool,
}

/// The client-local half of `EntityMinecartFurnace.interactFirst`.
///
/// Java does not guard this method with `!worldObj.isRemote`: every targeted C02 interaction is
/// consumed, coal adds 3600 ticks to the cart's private fuel counter, survival decrements the
/// selected coal stack, and the cart always overwrites `pushX/pushZ` from the two entity
/// positions. Rail motion remains server-owned; these fields only retain the state that the
/// remote client can observe through its later watcher/smoke tick.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FurnaceMinecartInteraction {
    /// Whether the held stack is `Items.coal` (coal and charcoal share item id 263).
    pub coal: bool,
    /// `EntityMinecartFurnace.fuel += 3600` when `coal` is true.
    pub fuel_added: i32,
    /// `pushX = cart.posX - player.posX`.
    pub push_x: f64,
    /// `pushZ = cart.posZ - player.posZ`.
    pub push_z: f64,
    /// Survival's `--itemstack.stackSize == 0` branch.
    pub decrement_stack: bool,
}

const COAL_ITEM_ID: i16 = 263;

/// Classify the client-reachable portion of `EntityMinecartFurnace.interactFirst`.
///
/// The return is `Some` for every furnace-cart click, including an empty hand and non-coal
/// items, because vanilla always returns `true` and always refreshes the push vector. A missing
/// coordinate snapshot is deliberately unresolved: no private local push state is invented for a
/// partial entity fixture. A zero-count coal object is retained as coal for the fuel branch, but
/// the driver’s normal stack decrement gate refuses to underflow it.
#[must_use]
pub fn furnace_minecart_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    creative: bool,
    cart_position: Option<DVec3>,
    player_position: Option<DVec3>,
) -> Option<FurnaceMinecartInteraction> {
    if kind != Some(EntityType::MinecartFurnace) {
        return None;
    }
    let cart_position = cart_position?;
    let player_position = player_position?;
    let coal = held.is_some_and(|stack| stack.item_id == COAL_ITEM_ID);
    Some(FurnaceMinecartInteraction {
        coal,
        fuel_added: if coal { 3600 } else { 0 },
        push_x: cart_position.x - player_position.x,
        push_z: cart_position.z - player_position.z,
        decrement_stack: coal && !creative && held.is_some_and(|stack| stack.count > 0),
    })
}

/// Classify horse food/chest mutations that happen before the server correction. A tame adult's
/// ordinary food only consumes when the local health is below max; otherwise Java falls through
/// to the mount branch. Untamed horses have the client constructor's temper-zero field, so their
/// wheat/sugar/apple/golden-food temper path is locally available even without a wire field.
#[must_use]
pub fn horse_item_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<HorseItemInteraction> {
    if kind != Some(EntityType::EntityHorse)
        || facts.horse_tame != Some(true) && facts.horse_tame != Some(false)
    {
        return None;
    }
    let horse_type = facts.horse_type?;
    if matches!(horse_type, 3 | 4) || facts.rideable_occupancy != RideableOccupancy::Empty {
        return None;
    }
    let item_id = held.map(|stack| stack.item_id);
    if let Some((heal, growth, temper)) = horse_feed_values(item_id) {
        let adult = !facts.base.child;
        let tame = facts.horse_tame == Some(true);
        let golden_breeding_food =
            matches!(item_id, Some(GOLDEN_CARROT_ITEM_ID | GOLDEN_APPLE_ITEM_ID));
        let mut active = false;
        let mut applied_heal = 0.0;
        if facts.horse_hurt == Some(true) && heal > 0.0 {
            active = true;
            applied_heal = heal;
        }
        if !adult && growth > 0 {
            active = true;
        }
        if adult && tame && golden_breeding_food {
            active = true;
        }
        // `k > 0 && (flag || !isTame()) && k < getMaxTemper()`. The private temper field starts
        // at zero on a network-created horse and is retained in the tracker after local clicks.
        let temper_now = facts.horse_temper.unwrap_or(0).clamp(0, 100);
        let temper_delta = if temper > 0 && (active || !tame) && temper_now < 100 {
            temper.min(100 - temper_now)
        } else {
            0
        };
        active |= temper_delta > 0;
        if active {
            return Some(HorseItemInteraction {
                growth: if !adult { growth } else { 0 },
                heal: applied_heal,
                starts_in_love: adult && tame && golden_breeding_food,
                temper_increase: temper_delta,
                chest: false,
                decrement_stack: !facts.base.creative,
            });
        }
    }

    if !facts.base.child
        && facts.horse_tame == Some(true)
        && matches!(horse_type, 1 | 2)
        && item_id == Some(CHEST_ITEM_ID)
        && facts.horse_chested == Some(false)
    {
        return Some(HorseItemInteraction {
            chest: true,
            decrement_stack: !facts.base.creative,
            ..HorseItemInteraction::default()
        });
    }
    None
}

/// Item-level client prediction for `ItemCarrotOnAStick.onItemRightClick`. `controlled` is a
/// rider-control shadow supplied by the session: vanilla's AI object owns it privately, so a
/// remote client cannot reconstruct the server's current-speed value from packets alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CarrotOnAStickInteraction {
    /// `EntityAIControlledByPlayer.boostSpeed()` was reached.
    pub boost: bool,
    /// Durability points requested from `ItemStack.damageItem`.
    pub damage: u8,
}

/// Classify the mounted, saddled pig boost and the seven durability points consumed by
/// `ItemCarrotOnAStick.onItemRightClick`.
#[must_use]
pub fn carrot_on_a_stick_interaction(
    held: Option<&ItemStack>,
    pig_saddled: bool,
    mounted_pig: bool,
    controlled: bool,
    _creative: bool,
) -> Option<CarrotOnAStickInteraction> {
    let stack = held.filter(|stack| stack.item_id == CARROT_ON_A_STICK_ITEM_ID)?;
    if !pig_saddled || !mounted_pig || !controlled || 25_i32 - i32::from(stack.damage) < 7 {
        return None;
    }
    Some(CarrotOnAStickInteraction {
        boost: true,
        damage: 7,
    })
}

/// All entity-side client mutations that can follow the targeted C02 pair. Keeping the
/// precedence in one classifier mirrors `EntityLiving.interactFirst`: mooshroom's bowl/shears
/// override runs before inherited cow milk, while ordinary breeding items reach EntityAnimal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntityLocalInteraction {
    /// `ItemDye.itemInteractionForEntity`'s sheep fleece mutation.
    ///
    /// The expected metadata is part of the value rather than re-read by the driver: an
    /// authoritative watcher update between classification and application must invalidate the
    /// prediction just as an ECS generation replacement does.
    SheepDye {
        /// The client-local callback result, including the selected fleece and stack decrement.
        prediction: SheepDyeInteraction,
        /// The low-nibble fleece value captured before the targeted C02 pair was queued.
        expected_fleece_metadata: u8,
    },
    /// Client-side lead attach/clear relation.
    Lead(LeadInteraction),
    /// Animal feed: adult love or baby happy-particle timer.
    AnimalFeed(AnimalFeedInteraction),
    /// Cow or mooshroom bucket-to-milk mutation.
    CowMilk(CowMilkInteraction),
    /// Adult mooshroom bowl-to-stew mutation.
    MooshroomBowl(MooshroomBowlInteraction),
    /// Adult mooshroom client despawn + explosion particle.
    MooshroomShears,
    /// `EntitySheep.interact`'s client-side `itemstack.damageItem(1,
    /// player)` on an unsheared adult (outside its `!isRemote` block, unlike the mooshroom's).
    SheepShears,
    /// Horse feed/chest mutation.
    Horse(HorseItemInteraction),
    /// Furnace-cart fuel/push mutation.
    FurnaceMinecart(FurnaceMinecartInteraction),
}

/// Resolve the local entity mutation, if any, after the driver has queued the targeted C02
/// packet(s). A `None` result is deliberately not a decline: it means that this click either has
/// no client-local state or still needs a server-owned/private predicate.
#[must_use]
pub fn entity_local_interaction(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<EntityLocalInteraction> {
    if let Some(expected_fleece_metadata) = facts.sheep_fleece_metadata {
        if let Some(prediction) =
            sheep_dye_interaction(kind, held, facts.base, expected_fleece_metadata)
        {
            return Some(EntityLocalInteraction::SheepDye {
                prediction,
                expected_fleece_metadata,
            });
        }
    }
    if let Some(prediction) = furnace_minecart_interaction(
        kind,
        held,
        facts
            .furnace_minecart_creative
            .unwrap_or(facts.base.creative),
        facts.furnace_minecart_cart_position,
        facts.furnace_minecart_player_position,
    ) {
        return Some(EntityLocalInteraction::FurnaceMinecart(prediction));
    }
    if let Some(prediction) = lead_interaction(kind, held, facts) {
        return Some(EntityLocalInteraction::Lead(prediction));
    }
    if mooshroom_shears_interaction(kind, held, facts) {
        return Some(EntityLocalInteraction::MooshroomShears);
    }
    // java: `if (itemstack != null && itemstack.getItem() ==
    // Items.shears && !this.getSheared() && !this.isChild())`, then `itemstack.damageItem(1,
    // player)` on the client too; the interaction still returns `super.interact` (not consumed).
    if kind == Some(EntityType::Sheep)
        && held.map(|stack| stack.item_id) == Some(SHEARS_ITEM_ID)
        && !facts.base.sheep_sheared
        && !facts.base.child
    {
        return Some(EntityLocalInteraction::SheepShears);
    }
    if let Some(prediction) = mooshroom_bowl_interaction(kind, held, facts) {
        return Some(EntityLocalInteraction::MooshroomBowl(prediction));
    }
    if let Some(prediction) = cow_milk_interaction(kind, held, facts) {
        return Some(EntityLocalInteraction::CowMilk(prediction));
    }
    if let Some(prediction) = horse_item_interaction(kind, held, facts) {
        return Some(EntityLocalInteraction::Horse(prediction));
    }
    animal_feed_interaction(kind, held, facts).map(EntityLocalInteraction::AnimalFeed)
}

fn wolf_consumed(item_id: Option<i16>, facts: EntityInteractionFacts) -> Option<bool> {
    let tamed = facts.wolf_tamed?;
    if tamed {
        if is_wolf_favorite_meat(item_id) {
            if facts.wolf_hurt == Some(true) || facts.base.child {
                return Some(true);
            }
            return facts.animal_breeding_ready;
        }
        return Some(false);
    }

    if item_id == Some(BONE_ITEM_ID) {
        return facts.wolf_angry.map(|angry| !angry);
    }
    if is_wolf_favorite_meat(item_id) {
        if facts.base.child {
            return Some(true);
        }
        return facts.animal_breeding_ready;
    }
    Some(false)
}

fn ocelot_consumed(item_id: Option<i16>, facts: EntityInteractionFacts) -> Option<bool> {
    if item_id != Some(FISH_ITEM_ID) {
        return Some(false);
    }
    if facts.base.child {
        return Some(true);
    }
    let tamed = facts.ocelot_tamed?;
    if tamed {
        return facts.animal_breeding_ready;
    }
    // `EntityAITempt` only runs in `EntityLivingBase`'s server-world AI branch, so the wild
    // ocelot's client-side tempt gate is always false. The remote call falls through to
    // EntityAnimal.interact, whose fish result is exactly the local in-love predicate.
    facts.animal_breeding_ready
}

fn horse_consumed(item_id: Option<i16>, facts: EntityInteractionFacts) -> Option<bool> {
    let tame = facts.horse_tame?;
    let horse_type = facts.horse_type?;
    let undead = matches!(horse_type, 3 | 4);

    // `EntityHorse.interact` checks this before occupancy. Spawn eggs and named tags are handled
    // by the public classifier, so no subtype needs to leak through here.
    if !tame && undead {
        return Some(false);
    }
    if !facts.base.child && tame && facts.sneaking == Some(true) {
        return Some(true);
    }
    // The adult occupied branch delegates to EntityAnimal before armor/feed/chest/saddle.
    if !facts.base.child {
        return match facts.rideable_occupancy {
            RideableOccupancy::Empty => Some(true),
            RideableOccupancy::Player | RideableOccupancy::Other => Some(false),
            RideableOccupancy::Unknown => None,
        };
    }
    // An untamed, non-undead horse rears for every held, unconsumed item, including a child
    // donkey. With an empty hand the item body is skipped and a child falls through to the
    // EntityAnimal return instead.
    if !tame {
        return Some(item_id.is_some());
    }
    if !undead && ((is_horse_armor(item_id) && horse_type == 0) || is_horse_feed(item_id)) {
        return Some(true);
    }
    if (horse_type == 1 || horse_type == 2) && item_id == Some(CHEST_ITEM_ID) {
        return facts.horse_chested.map(|chested| !chested);
    }
    Some(false)
}

/// Whether the client-side `EntityHorse.interact` tail calls `mountTo(player)` for this click.
///
/// This is intentionally narrower than [`horse_consumed`]: feed, inventory, chest, spawn-egg,
/// name-tag and unresolved lead/saddle branches can consume an adult empty horse without
/// mounting it. The local driver retains this one-shot intent until the matching S1B confirms
/// the vehicle, at which point Java's client-predicted yaw/pitch and seat snap are observable.
#[must_use]
pub fn horse_mount_intent(held: Option<&ItemStack>, facts: EntityInteractionFacts) -> bool {
    if facts.base.child || facts.rideable_occupancy != RideableOccupancy::Empty {
        return false;
    }
    let (Some(tame), Some(horse_type)) = (facts.horse_tame, facts.horse_type) else {
        return false;
    };
    // java: an untamed horse reaches the `mountTo` tail only with an
    // empty hand (`!isTame() && isUndead()` returns false first; any held item ends in the item
    // block's rear, feed or `interactWithEntity` return). The sneak test opens the GUI of a tame
    // horse only (`isTame() && isAdultHorse() && player.isSneaking()`).
    if !tame {
        return held.is_none() && !matches!(horse_type, 3 | 4);
    }
    if facts.sneaking != Some(false) {
        return false;
    }
    let Some(stack) = held else {
        return true;
    };
    let item_id = Some(stack.item_id);
    let named_tag = stack.item_id == NAME_TAG_ITEM_ID
        && stack
            .nbt
            .as_ref()
            .and_then(|tag| tag.get_compound("display"))
            .and_then(|display| display.get_str("Name"))
            .is_some();
    if named_tag
        || stack.item_id == SPAWN_EGG_ITEM_ID
        || stack.item_id == LEAD_ITEM_ID
        || stack.item_id == SADDLE_ITEM_ID
    {
        return false;
    }
    let undead = matches!(horse_type, 3 | 4);
    // A tame adult at known full health falls through the feed block and mounts.  Keep the
    // legacy conservative answer for an absent health attribute: the client cannot prove that
    // ordinary food is a no-op there, while a hurt horse (or a breeding/growth food) is consumed
    // by `horse_item_interaction` and must not reach `mountTo`.
    let ordinary_feed_consumed = is_horse_feed(item_id)
        && (facts.horse_hurt != Some(false)
            || horse_item_interaction(Some(EntityType::EntityHorse), held, facts).is_some());
    if !undead && (ordinary_feed_consumed || (horse_type == 0 && is_horse_armor(item_id))) {
        return false;
    }
    if matches!(horse_type, 1 | 2)
        && stack.item_id == CHEST_ITEM_ID
        && facts.horse_chested == Some(false)
    {
        return false;
    }
    true
}

fn passive_consumed(
    kind: Option<EntityType>,
    facts: EntityInteractionFacts,
    item_id: Option<i16>,
) -> Option<bool> {
    if !matches!(
        kind,
        Some(
            EntityType::Pig
                | EntityType::Sheep
                | EntityType::Cow
                | EntityType::Chicken
                | EntityType::MushroomCow
                | EntityType::Rabbit
        )
    ) {
        return None;
    }
    if animal_breeding_item(kind, item_id) {
        if facts.base.child {
            Some(true)
        } else {
            facts.animal_breeding_ready
        }
    } else {
        Some(false)
    }
}

/// The stateful, Java-backed classifier used by the live right-click driver.
///
/// The driver supplies watcher defaults explicitly (for example `wolf_tamed: Some(false)`),
/// while a partial unit fixture can leave a field `None` and retain the safe no-fallback result.
#[must_use]
pub fn entity_interaction_consumed_with_facts(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: EntityInteractionFacts,
) -> Option<bool> {
    let item_id = held.map(|stack| stack.item_id);

    // `ItemNameTag.itemInteractionForEntity` runs after `interactFirst` and is true for every
    // EntityLiving (but not EntityLivingBase-only stands or players) when display.Name exists.
    let named = name_tag_interaction(kind, held, facts.base.creative).is_some();

    if kind == Some(EntityType::ArmorStand) {
        // ArmorStand is EntityLivingBase, not EntityLiving, so ItemNameTag's entity override does
        // not accept it. A non-marker stand is consumed by interactAt in the driver; a marker
        // stand reaches this generic body and correctly declines.
        return Some(false);
    }
    if kind == Some(EntityType::Sheep) && item_id == Some(DYE_ITEM_ID) {
        // ItemDye returns true even for a matching colour or an already-sheared sheep.
        return Some(true);
    }
    if kind == Some(EntityType::Sheep) && item_id == Some(SHEARS_ITEM_ID) {
        // EntitySheep mutates its local client copy, then returns `super.interact`; shears are not
        // a breeding item, so the generic result is false (the old classifier called this true).
        return Some(false);
    }
    if let Some(saddle) = saddle_item_interaction(kind, held, facts.base) {
        return Some(saddle.consumed);
    }
    if matches!(kind, Some(EntityType::Cow | EntityType::MushroomCow))
        && item_id == Some(BUCKET_ITEM_ID)
    {
        return Some(!facts.base.creative && !facts.base.child);
    }
    if kind == Some(EntityType::MushroomCow) && item_id == Some(SHEARS_ITEM_ID) {
        return Some(!facts.base.child);
    }
    if kind == Some(EntityType::MushroomCow) && item_id == Some(BOWL_ITEM_ID) {
        if facts.base.child {
            // EntityMooshroom falls through to EntityCow/EntityAnimal for a child, and neither
            // accepts a bowl.
            return Some(false);
        }
        // A one-bowl stack is unconditional. For a larger stack the result depends on whether
        // InventoryPlayer can add the stew, which this seam does not own.
        return held.filter(|stack| stack.count == 1).map(|_| true);
    }
    if named && kind.is_some_and(|kind| kind.info().living && kind != EntityType::ArmorStand) {
        return Some(true);
    }

    // EntityAgeable's spawn-egg branch returns true on the remote world even when it cannot spawn
    // the requested subtype; this covers horse/villager and all ordinary passive classes.
    if item_id == Some(SPAWN_EGG_ITEM_ID) && is_ageable(kind) {
        return Some(true);
    }

    // These object entities either always consume `interactFirst` on the client, or always
    // decline it. The empty minecart's occupant is represented by the released W-RIDE state.
    match kind {
        Some(
            EntityType::Boat
            | EntityType::MinecartChest
            | EntityType::MinecartFurnace
            | EntityType::MinecartHopper
            | EntityType::ItemFrame
            | EntityType::LeashKnot,
        ) => return Some(true),
        Some(EntityType::MinecartRideable) => {
            return match facts.rideable_occupancy {
                RideableOccupancy::Empty | RideableOccupancy::Player => Some(true),
                RideableOccupancy::Other => Some(false),
                RideableOccupancy::Unknown => None,
            };
        }
        Some(
            EntityType::MinecartCommandBlock
            | EntityType::MinecartTNT
            | EntityType::MinecartSpawner
            | EntityType::Painting
            | EntityType::EnderCrystal
            | EntityType::Fireball
            | EntityType::SmallFireball
            | EntityType::WitherSkull
            | EntityType::PrimedTnt
            | EntityType::FallingSand
            | EntityType::Item
            | EntityType::XPOrb
            | EntityType::ThrownEgg
            | EntityType::Arrow
            | EntityType::Snowball
            | EntityType::ThrownEnderpearl
            | EntityType::EyeOfEnderSignal
            | EntityType::ThrownPotion
            | EntityType::ThrownExpBottle
            | EntityType::FireworksRocketEntity,
        ) => return Some(false),
        None => {
            // `None` is a remote player or the directly-constructed fishing bobber in this
            // tracker. Neither overrides `interactFirst`, and none of the item overrides accepts
            // either target.
            return Some(false);
        }
        _ => {}
    }

    // `EntityLiving.interactFirst` checks the leash/lead branch before every concrete living
    // interaction. Keep a known invalid lead as `Some(false)` so `rightClickMouse` appends the
    // untargeted C08 exactly like Java; subtype branches such as Villager's own interaction are
    // allowed to run when `lead_consumed` returns `None`.
    if let Some(consumed) = lead_consumed(kind, held, facts) {
        return Some(consumed);
    }

    // IMob has `allowLeashing() == false`, so the base EntityLiving interactFirst cannot consume
    // a lead for these rows. Keep Zombie's golden-apple conversion unresolved: villager,
    // weakness, and conversion state are not all watched here.
    match kind {
        Some(EntityType::Creeper) => return Some(false),
        Some(EntityType::Zombie) if item_id != Some(GOLDEN_APPLE_ITEM_ID) => return Some(false),
        Some(
            EntityType::Skeleton
            | EntityType::Spider
            | EntityType::Giant
            | EntityType::PigZombie
            | EntityType::CaveSpider
            | EntityType::Silverfish
            | EntityType::Blaze
            | EntityType::LavaSlime
            | EntityType::Enderman
            | EntityType::Ghast
            | EntityType::Witch
            | EntityType::Endermite
            | EntityType::Guardian
            | EntityType::WitherBoss
            | EntityType::EnderDragon,
        ) => return Some(false),
        _ => {}
    }

    match kind {
        Some(EntityType::EntityHorse) => return horse_consumed(item_id, facts),
        Some(EntityType::Villager) => {
            if facts.base.child {
                return Some(false);
            }
            let alive = facts.villager_alive?;
            let trading = facts.villager_trading?;
            return Some(alive && !trading);
        }
        Some(EntityType::Wolf) => {
            if item_id == Some(DYE_ITEM_ID) {
                let tamed = facts.wolf_tamed?;
                if !tamed {
                    return Some(false);
                }
                let collar = facts.wolf_collar_dye_damage?;
                let held_damage = held
                    .map(|stack| {
                        // java: EnumDyeColor.byDyeDamage clamps negative and >=16 metadata to
                        // WHITE (index 0); it does not mask or take a modulo.
                        let damage = i32::from(stack.damage);
                        if (0..16).contains(&damage) {
                            damage
                        } else {
                            0
                        }
                    })
                    .expect("dye branch has a held stack");
                return Some(held_damage != collar);
            }
            return wolf_consumed(item_id, facts);
        }
        Some(EntityType::Ozelot) => return ocelot_consumed(item_id, facts),
        _ => {}
    }

    if let Some(result) = passive_consumed(kind, facts, item_id) {
        return Some(result);
    }

    // The remaining EntityLiving classes (bat, squid, golems, etc.) have no client-consuming
    // override.  Their lead branch was decided above; all other held items remain unresolved
    // because the corresponding private subtype state is not represented here.
    None
}

/// java: `Entity.getCollisionBorderSize()` (`:2036`) and its two overrides.
///
/// `0.1F` for everything, `0.0F` for `EntityItemFrame` (`:39`) and `1.0F` for `EntityFireball`
/// (`:297`) — which in 1.8.9 means the large fireball, the small fireball and the wither skull,
/// since all three extend it. `getMouseOver` expands each candidate box by this before the
/// intercept, so leaving it out shrinks every hitbox by 0.1 on all six sides.
///
/// The comment this replaces claimed the value "is 0 for every entity except an `EntityBoat`",
/// which is wrong twice: `EntityBoat` has no override, and the default is not zero.
// java: Entity.java:2036 (getCollisionBorderSize)
#[must_use]
pub fn collision_border_size(kind: Option<EntityType>) -> f32 {
    match kind {
        Some(EntityType::ItemFrame) => 0.0,
        Some(EntityType::Fireball | EntityType::SmallFireball | EntityType::WitherSkull) => 1.0,
        _ => 0.1,
    }
}

/// Whether `bounds` grown by `border` (`getCollisionBorderSize`) comes within
/// `reach` of `eye`: the exact condition for `getMouseOver`'s entity ray, which is `reach` long,
/// to meet the box (or start inside it). The pick's cheap candidate reject must never drop an
/// entity vanilla could hit, and this is a superset of that by construction: a hit point lies on
/// the ray, so within `reach` of the eye, and inside the grown box.
#[must_use]
pub fn box_within_reach(bounds: Aabb, border: f32, eye: DVec3, reach: f64) -> bool {
    let grown = f64::from(border);
    let gap = |min: f64, max: f64, at: f64| (min - grown - at).max(at - max - grown).max(0.0);
    let dx = gap(bounds.min.x, bounds.max.x, eye.x);
    let dy = gap(bounds.min.y, bounds.max.y, eye.y);
    let dz = gap(bounds.min.z, bounds.max.z, eye.z);
    // A hair of slack: the intercept is computed in doubles from the same inputs.
    dx * dx + dy * dy + dz * dz <= reach * reach + 1e-9
}

/// java: the box `Entity.setSize` (`:1826`) leaves behind -- centred on x/z, **resting** on y.
///
/// This is the default only. Three of the pickable types replace it outright in their
/// constructor; see [`hanging_bounding_box`] and [`leash_knot_bounding_box`].
#[must_use]
pub fn size_bounding_box(pos: DVec3, width: f32, height: f32) -> Aabb {
    let half = f64::from(width) / 2.0;
    Aabb::from_bounds(
        pos.x - half,
        pos.y,
        pos.z - half,
        pos.x + half,
        pos.y + f64::from(height),
        pos.z + half,
    )
}

/// java: `EntityHanging.updateBoundingBox` (`EntityHanging.java:53-90`), the tail that builds the
/// box once `posX/posY/posZ` are known.
///
/// **It replaces `setSize(0.5F, 0.5F)`'s box entirely**, and the replacement is a thin plate, not
/// a cube: `d6 = d7 = d8 = widthPixels/heightPixels/widthPixels`, then whichever of `d6`/`d8` is
/// on the facing axis is forced to `1.0`, then all three are divided by 32 and used as **half**
/// extents about the entity position. So an item frame (12x12) is a `0.75 x 0.75 x 0.0625` plate
/// flush with its wall, and `EntityPainting` is `art.sizeX/16` by `art.sizeY/16` blocks.
///
/// `F-P1-entity.item-06-2`: the port used `mc_data`'s `0.5 x 0.5` row, which is
/// `EntityHanging(World)`'s `setSize` -- the box vanilla throws away. The result was a cube
/// resting on the entity rather than a plate centred on it: 0.44 too thick through the wall and
/// short by 0.25 in every other direction, so a click aimed at the top of a frame landed on the
/// block behind it.
///
/// The vertical half-extent is `heightPixels/32` **about** `pos.y`, not `pos.y .. pos.y + h`;
/// `updateBoundingBox` has already lifted `posY` by `func_174858_a(heightPixels)` for art whose
/// height is a multiple of 32, which is what [`mc_render::entity::objects::hanging_position`]
/// reproduces at spawn.
#[must_use]
pub fn hanging_bounding_box(
    pos: DVec3,
    facing: Facing,
    width_pixels: i32,
    height_pixels: i32,
) -> Aabb {
    let mut d6 = f64::from(width_pixels);
    let d7 = f64::from(height_pixels);
    let mut d8 = f64::from(width_pixels);
    // java: `if (facingDirection.getAxis() == Axis.Z) d8 = 1.0; else d6 = 1.0;`
    if matches!(facing, Facing::North | Facing::South) {
        d8 = 1.0;
    } else {
        d6 = 1.0;
    }
    let (d6, d7, d8) = (d6 / 32.0, d7 / 32.0, d8 / 32.0);
    Aabb::from_bounds(
        pos.x - d6,
        pos.y - d7,
        pos.z - d8,
        pos.x + d6,
        pos.y + d7,
        pos.z + d8,
    )
}

/// java: `EntityLeashKnot`'s constructor (`EntityLeashKnot.java:20-28`).
///
/// A leash knot is an `EntityHanging` that **opts out** of the hanging box: its
/// `updateFacingWithBoundingBox` override (`:38-40`) has an empty body, so `facingDirection`
/// stays null, `updateBoundingBox`'s `if (this.facingDirection != null)` never fires, and the
/// box is the literal one the constructor sets -- `+-0.1875` horizontally and
/// `-0.125 .. +0.375` vertically about a position pinned to the block centre. Its
/// `getWidthPixels()`/`getHeightPixels()` of `9` are therefore never used for a box at all.
#[must_use]
pub fn leash_knot_bounding_box(pos: DVec3) -> Aabb {
    Aabb::from_bounds(
        pos.x - 0.187_5,
        pos.y - 0.25 + 0.125,
        pos.z - 0.187_5,
        pos.x + 0.187_5,
        pos.y + 0.25 + 0.125,
        pos.z + 0.187_5,
    )
}

/// One entity a pick can land on, as [`pick`] needs it.
#[derive(Debug, Clone, Copy)]
pub struct Pickable {
    /// The server's id.
    pub network_id: i32,
    /// java: `Entity.getEntityBoundingBox()`, which is what `getMouseOver` reads (`:459`).
    ///
    /// An explicit box rather than a `(pos, width, height)` triple because three of the types a
    /// pick can land on do not have a `setSize` box at all: `EntityPainting` and
    /// `EntityItemFrame` replace it in `EntityHanging.updateBoundingBox`, and `EntityLeashKnot`
    /// writes its own in its constructor. See [`size_bounding_box`], [`hanging_bounding_box`]
    /// and [`leash_knot_bounding_box`].
    pub bounds: Aabb,
    /// java: `getCollisionBorderSize()` — how far the box is expanded before the intercept.
    /// See [`collision_border_size`].
    pub border_size: f32,
    /// java: `Predicates.and(EntitySelectors.NOT_SPECTATING, e -> e.canBeCollidedWith())`.
    ///
    /// False keeps the candidate out of the list entirely, exactly as
    /// `getEntitiesInAABBexcluding`'s predicate does. See [`can_be_collided_with`].
    pub collidable: bool,
}

/// What the crosshair is on, blocks and entities together.
///
/// A line-for-line port of `EntityRenderer.getMouseOver` (`:409-514`):
///
/// ```text
/// double d0 = playerController.getBlockReachDistance();      // 5.0 creative, 4.5 otherwise
/// objectMouseOver = entity.rayTrace(d0, partialTicks);
/// double d1 = d0;
/// boolean flag = false;
/// if (playerController.extendedReach()) { d0 = 6.0; d1 = 6.0; }
/// else if (d0 > 3.0) flag = true;
/// if (objectMouseOver != null) d1 = objectMouseOver.hitVec.distanceTo(vec3);
/// ... entity loop, d2 starting at d1 ...
/// if (pointedEntity != null && flag && vec3.distanceTo(vec33) > 3.0) {
///     pointedEntity = null;
///     objectMouseOver = new MovingObjectPosition(MISS, vec33, null, new BlockPos(vec33));
/// }
/// if (pointedEntity != null && (d2 < d1 || objectMouseOver == null))
///     objectMouseOver = new MovingObjectPosition(pointedEntity, vec33);
/// ```
///
/// `extended_reach` is `PlayerControllerMP.extendedReach()`, i.e. creative. Two things it decides
/// are easy to get wrong and both change the wire:
///
/// * The **entity ray is not clamped to 3 blocks in survival.** It runs the full 4.5, and the
///   3-block limit is a *post-hoc rejection* that also throws the block hit away — so a player
///   with a mob 4 blocks off in front of a wall gets a MISS, and left-clicking sends no `C07` at
///   all. A port that shortened the ray instead would have sent one.
/// * In creative both the entity ray and the initial clamp are **6.0**, longer than the 5.0
///   block reach.
///
/// Candidates must already be filtered by [`can_be_collided_with`] — see [`Pickable::collidable`],
/// which is checked here so the filter cannot be skipped at a call site.
// java: EntityRenderer.java:409 (getMouseOver)
#[must_use]
pub fn pick(
    world: &World,
    eye: DVec3,
    look: DVec3,
    block_reach: f64,
    extended_reach: bool,
    entities: impl IntoIterator<Item = Pickable>,
) -> HitResult {
    pick_with_riding(
        world,
        eye,
        look,
        block_reach,
        extended_reach,
        None,
        entities,
    )
}

/// [`pick`] with the local player's `ridingEntity`, as its network id.
///
/// Vanilla excludes that vehicle from ordinary intercepts so looking through a boat or minecart
/// does not make the crosshair target the mount the player is already controlling. The
/// eye-inside-box branch still selects it, and the peculiar `d2 == 0` sentinel also permits it
/// after an earlier containing candidate. `None` is the on-foot behaviour exposed by [`pick`].
#[must_use]
pub fn pick_with_riding(
    world: &World,
    eye: DVec3,
    look: DVec3,
    block_reach: f64,
    extended_reach: bool,
    riding_entity: Option<i32>,
    entities: impl IntoIterator<Item = Pickable>,
) -> HitResult {
    // java: `objectMouseOver = entity.rayTrace(d0, partialTicks)` runs at the *block* reach,
    // before `d0` is widened for creative.
    let trace = raytrace_blocks(world, eye, eye + look * block_reach);

    // java: `d0`/`d1`/`flag`, in that order.
    let (entity_ray, mut d1) = if extended_reach {
        (6.0, 6.0)
    } else {
        (block_reach, block_reach)
    };
    let survival_clamp = !extended_reach && block_reach > 3.0;
    // java: `if (objectMouseOver != null) d1 = objectMouseOver.hitVec.distanceTo(vec3);` — read
    // off the raytrace *whatever its type*, so a MISS still shortens the entity search.
    if let Some(t) = trace {
        d1 = mc_math::vec::distance_to(t.hit().hit_vec, eye);
    }

    let entity_end = eye + look * entity_ray;
    let mut d2 = d1;
    let mut pointed: Option<(i32, DVec3)> = None;
    for candidate in entities {
        if !candidate.collidable {
            continue;
        }
        // java: `axisalignedbb = entity1.getEntityBoundingBox().expand(f1, f1, f1)` with
        // `f1 = entity1.getCollisionBorderSize()`.
        let f1 = f64::from(candidate.border_size);
        let box_ = candidate.bounds.expand(f1, f1, f1);
        let point = intercept(box_, eye, entity_end);
        if box_.is_vec_inside(eye) {
            // java: the `d2 >= 0.0D` guard, which is unconditionally true — `d2` only ever holds
            // a distance or the literal `0.0`. The eye being inside the box wins outright.
            pointed = Some((candidate.network_id, point.unwrap_or(eye)));
            d2 = 0.0;
        } else if let Some(point) = point {
            let d3 = mc_math::vec::distance_to(point, eye);
            // java: `if (d3 < d2 || d2 == 0.0D)`. The `|| d2 == 0.0` arm is why an eye-inside hit
            // is beaten by any later intercepting entity, subject to the ridden-vehicle fork.
            if d3 < d2 || d2 == 0.0 {
                // java: `if (entity1 == entity.ridingEntity) { if (d2 == 0.0D) ... } else ...`.
                // Contrary to later-version ports, 1.8.9 has no `canRiderInteract` call here.
                if riding_entity == Some(candidate.network_id) {
                    if d2 == 0.0 {
                        pointed = Some((candidate.network_id, point));
                    }
                } else {
                    pointed = Some((candidate.network_id, point));
                    d2 = d3;
                }
            }
        }
    }

    if let Some((_, hit_vec)) = pointed {
        // java: `if (pointedEntity != null && flag && vec3.distanceTo(vec33) > 3.0D)` — the
        // entity is dropped *and* `objectMouseOver` is replaced with a MISS, so the block behind
        // it is discarded too.
        if survival_clamp && mc_math::vec::distance_to(hit_vec, eye) > 3.0 {
            return HitResult::Miss;
        }
    }

    // java: `if (pointedEntity != null && (d2 < d1 || objectMouseOver == null))`.
    if let Some((network_id, hit_vec)) = pointed {
        if d2 < d1 || trace.is_none() {
            return HitResult::Entity(EntityHit {
                network_id,
                hit_vec,
            });
        }
    }
    match trace.and_then(RayTrace::as_block) {
        Some(hit) => HitResult::Block(hit),
        None => HitResult::Miss,
    }
}

/// java: `AxisAlignedBB.calculateIntercept`, reduced to the intersection point.
fn intercept(box_: Aabb, start: DVec3, end: DVec3) -> Option<DVec3> {
    collision_ray_trace(box_, BlockPos::new(0, 0, 0), start, end).map(|hit| hit.hit_vec)
}

// ---------------------------------------------------------------------------------------------
// The client-side placement gate — W-RCLICK (1)
// ---------------------------------------------------------------------------------------------

/// The block an `ItemBlock` places, or `None` when the held item is not an `ItemBlock`.
///
/// java: `heldStack.getItem() instanceof ItemBlock`, then `((ItemBlock)item).block`.
/// `ItemKind::Block` is `mc_data`'s name for that `instanceof`, and there are exactly 151 of
/// them — the same 151 the audit measured suppression across.
///
/// Every `ItemBlock` but one is registered under its block's own numeric id
/// (`Item.registerItemBlock` -> `registerItem(Block.getIdFromBlock(block), ...)`), so the two ids
/// are equal by construction; `every_item_block_shares_its_blocks_id` is the oracle. The
/// exception is `ItemBanner` (`Item.registerItems:935`), which `registerItem`s at **425** while
/// its `block` field is `Blocks.standing_banner` (`ItemBanner:22`).
#[must_use]
pub fn placed_block(stack: &ItemStack) -> Option<mc_data::BlockId> {
    let id = u16::try_from(stack.item_id).ok()?;
    let props = mc_data::ItemId(id).props();
    if !props.registered || props.kind != mc_data::ItemKind::Block {
        return None;
    }
    if id == BANNER_ITEM_ID {
        return Some(STANDING_BANNER);
    }
    (id < 256).then_some(mc_data::BlockId(id))
}

/// The `ItemReed` registrations whose blocks have a client-visible placement predicate.
///
/// Repeater and comparator items are intentionally `ItemKind::Generic` in the data table: Java
/// registers them as `new ItemReed(Blocks.unpowered_*)`, not as `ItemBlock`. `ItemReed.onItemUse`
/// runs the `World.canBlockBePlaced` chain itself, *after* `onPlayerRightClick` has sent the
/// targeted C08 (the pre-C08 `canPlaceBlockOnSide` gate is `instanceof ItemBlock` only), so this
/// resolves their backing block for that post-C08 placement prediction.
#[must_use]
pub fn item_reed_block(stack: &ItemStack) -> Option<mc_data::BlockId> {
    match stack.item_id {
        287 => Some(mc_data::BlockId(132)), // ItemReed(Blocks.tripwire)
        338 => Some(mc_data::BlockId(83)),  // ItemReed(Blocks.reeds)
        354 => Some(mc_data::BlockId(92)),  // ItemReed(Blocks.cake)
        356 => Some(mc_data::BlockId(93)),  // ItemReed(Blocks.unpowered_repeater)
        379 => Some(mc_data::BlockId(117)), // ItemReed(Blocks.brewing_stand)
        380 => Some(mc_data::BlockId(118)), // ItemReed(Blocks.cauldron)
        390 => Some(mc_data::BlockId(140)), // ItemReed(Blocks.flower_pot)
        404 => Some(mc_data::BlockId(149)), // ItemReed(Blocks.unpowered_comparator)
        _ => None,
    }
}

/// The item `Minecraft.middleClickMouse` asks `Block.getItem`/`getDamageValue` for.
///
/// The inventory mutation is deliberately left to the app driver: Java's
/// `InventoryPlayer.setCurrentItem` has different survival and creative branches, while this
/// value is also useful to tests without a live session. `meta_specific` is the exact
/// `Item.getHasSubtypes()` bit used by that inventory method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PickedItem {
    /// The registry id of the picked item.
    pub item_id: i16,
    /// The metadata passed to `new ItemStack(item, 1, metadata)`.
    pub metadata: i16,
    /// Whether inventory matching includes [`Self::metadata`].
    pub meta_specific: bool,
}

impl PickedItem {
    /// The one-item stack Java creates before `setCurrentItem` consumes it.
    #[must_use]
    pub fn stack(self) -> ItemStack {
        ItemStack {
            item_id: self.item_id,
            count: 1,
            damage: self.metadata,
            nbt: None,
        }
    }
}

/// Resolve the item and metadata produced by Java's `Block.getItem` and `getDamageValue` for a
/// block pick. The generated block table supplies `Item.getItemFromBlock`; the small explicit
/// dispatch below is for the 1.8.9 virtual overrides (crops, pots, skulls, etc.).
///
/// Tile entities in mc-rs intentionally retain typed client-visible state rather than their raw
/// NBT compounds. The ordinary pick path uses the typed flower-pot/skull values; the creative
/// Ctrl path reconstructs the fields that the live client state still exposes.
#[must_use]
pub fn pick_block_item(world: &World, pos: BlockPos) -> Option<PickedItem> {
    let state = world.block_at(pos);
    let block = state.block().0;
    if state.props().is_air() {
        return None;
    }

    // java: the virtual `Block.getItem(world, pos)` overrides. Values are vanilla 1.8.9 item
    // registry ids; keeping these here avoids making a second item-name lookup in the hot path.
    let item_id = match block {
        26 => 355, // BlockBed
        34 => {
            // BlockPistonExtension.getItem chooses Blocks.sticky_piston for the sticky TYPE.
            if state.meta() & 8 != 0 {
                29
            } else {
                33
            }
        }
        36 | 52 => return None, // moving piston / mob spawner have getItem() == null
        43 => 44,               // double stone slab -> single slab item
        125 => 126,             // double wooden slab -> single slab item
        181 => 182,             // double red-sandstone slab -> single slab item
        59 => 295,              // BlockCrops -> wheat seeds
        141 => 391,             // BlockCarrot (BlockCrops) -> carrots
        142 => 392,             // BlockPotato (BlockCrops) -> potatoes
        60 => 3,                // BlockFarmland -> dirt
        63 | 68 => 323,         // BlockStandingSign / BlockWallSign
        64 => 324,              // BlockDoor -> wooden door
        71 => 330,              // BlockDoor -> iron door
        193 => 427,             // BlockDoor -> spruce door
        194 => 428,             // BlockDoor -> birch door
        195 => 429,             // BlockDoor -> jungle door
        196 => 430,             // BlockDoor -> acacia door
        197 => 431,             // BlockDoor -> dark-oak door
        83 => 338,              // BlockReed -> reeds
        90 | 119 | 122 => return None, // portal, end portal, dragon egg
        92 => 354,              // BlockCake
        93 | 94 => 356,         // BlockRedstoneRepeater
        55 => 331,              // BlockRedstoneWire -> redstone dust
        62 => 61,               // lit furnace -> furnace
        74 | 124 => 123,        // lit redstone ore/lamp share their unlit item
        75 => 76,               // unlit redstone torch -> redstone torch
        99 => 39,               // BlockHugeMushroom -> brown mushroom
        100 => 40,              // BlockHugeMushroom -> red mushroom
        104 => 361,             // pumpkin stem -> pumpkin seeds
        105 => 362,             // melon stem -> melon seeds
        115 => 372,             // BlockNetherWart
        117 => 379,             // BlockBrewingStand
        118 => 380,             // BlockCauldron
        127 => 351,             // BlockCocoa -> brown dye
        132 => 287,             // BlockTripWire -> string
        140 => world
            .tile_entity(pos)
            .and_then(|tile| tile.flower_pot())
            .and_then(|pot| {
                let id = u16::try_from(pot.item_id).ok()?;
                mc_data::ItemId(id).try_props()?;
                i16::try_from(id).ok()
            })
            .unwrap_or(390),
        144 => 397,       // BlockSkull
        149 | 150 => 404, // BlockRedstoneComparator
        176 | 177 => 425, // BlockBanner
        178 => 151,       // inverted daylight detector shares the item
        _ => i16::try_from(mc_data::block_info(mc_data::BlockId(block)).item?.0).ok()?,
    };

    // java: `Block.getDamageValue` defaults to `damageDropped`, which returns zero. Only blocks
    // whose class overrides that virtual (or whose getDamageValue itself is overridden) carry a
    // state-derived metadata value here. This distinction is observable for, e.g., a lit furnace
    // and a snow layer: their block-state metadata must not leak into the picked item.
    let metadata = match block {
        // BlockStone / BlockDirt / BlockPlanks / BlockSand / BlockSandStone and the other enum
        // variant blocks use their state metadata directly.
        1 | 3 | 5 | 6 | 12 | 24 | 31 | 35 | 37 | 38 | 95 | 97 | 98 | 139 | 159 | 160 | 168
        | 171 | 179 => i16::from(state.meta()),
        // Log/leaf state metadata carries an axis/decay bit alongside the item variant.
        17 | 18 | 161 | 162 => i16::from(state.meta() & 3),
        // Sponge's only item subtype is the wet bit.
        19 => i16::from(state.meta() & 1),
        // BlockOre.getDamageValue is explicitly hard-zero in 1.8.9. Its lapis override is only
        // damageDropped (the break/drop path), not the middle-click pick path.
        // BlockSlab.getDamageValue masks the half/double flag off.
        43 | 44 | 125 | 126 | 181 | 182 => i16::from(state.meta() & 7),
        // BlockAnvil stores damage in bits 2..3; facing is not item metadata.
        145 => i16::from((state.meta() >> 2) & 3),
        // BlockQuartz's two directional line states drop as the vertical line variant.
        155 => match state.meta() {
            3 | 4 => 2,
            meta => i16::from(meta),
        },
        // BlockDoublePlant reads the actual variant, including from the lower half when the
        // crosshair is on the upper cell. Its storage metadata puts the variant in bits 0..2.
        175 if state.meta() & 8 != 0 => {
            let lower = world.block_at(pos.relative(Facing::Down));
            if lower.block().0 == 175 {
                i16::from(lower.meta() & 7)
            } else {
                // BlockDoublePlant.getVariant returns FERN when the lower half is absent.
                3
            }
        }
        175 => i16::from(state.meta() & 7),
        // BlockCocoa.getDamageValue is always the brown dye value.
        127 => 3,
        // BlockFlowerPot and BlockSkull read their tile entity state.
        140 => world
            .tile_entity(pos)
            .and_then(|tile| tile.flower_pot())
            .filter(|pot| pot.item_id != 0)
            .map_or(0, |pot| pot.data as i16),
        144 => world
            .tile_entity(pos)
            .and_then(|tile| tile.skull())
            .map_or(0, |skull| i16::from(skull.skull_type)),
        // BlockPistonExtension.getItem picks the piston item; its own damage value is zero.
        34 => 0,
        _ => 0,
    };

    // BlockPistonExtension's item choice depends on its TYPE property rather than its item
    // registry id. This is the only getItem override above whose id is state-dependent.
    let item_id = if block == 34 {
        if state.meta() & 8 != 0 {
            29
        } else {
            33
        }
    } else {
        item_id
    };
    // Flower-pot contents already supply their own item metadata; all other picked stacks use
    // the selected item's actual `hasSubtypes` bit, including the banner and dye exceptions.
    let props = mc_data::ItemId(u16::try_from(item_id).ok()?).try_props()?;
    Some(PickedItem {
        item_id,
        metadata,
        meta_specific: props.has_subtypes,
    })
}

/// Build the stack used by Java's Ctrl-middle-click tile-entity branch when the client still has
/// enough typed state to serialize it. The ordinary block-pick resolver remains separate because
/// Java only asks for this compound in creative mode while Ctrl is held.
///
/// `World` intentionally discards the original tile NBT after decoding S35, so this is a typed
/// reconstruction: it preserves the fields mc-world exposes (sign text, banner patterns, skull
/// owner, spawner timing, beacon levels and flower-pot contents) and leaves unavailable server
/// fields absent. That is preferable to manufacturing a stale raw compound or rejecting a pick
/// that vanilla accepts.
#[must_use]
pub fn pick_block_with_nbt(world: &World, pos: BlockPos) -> Option<ItemStack> {
    let picked = pick_block_item(world, pos)?;
    let tile = world.tile_entity(pos)?;
    Some(pick_block_with_tile_nbt(picked, tile))
}

/// Build the Ctrl-middle-click stack while mirroring Java's immediate tile lookup. Unlike the
/// read-only helper above, this path creates the typed default tile that `World.getTileEntity`
/// would cache when a loaded tile block has not received its description packet yet.
#[must_use]
pub fn pick_block_with_nbt_or_create(world: &mut World, pos: BlockPos) -> Option<ItemStack> {
    let picked = pick_block_item(world, pos)?;
    let tile = world.tile_entity_or_create(pos)?;
    Some(pick_block_with_tile_nbt(picked, tile))
}

fn pick_block_with_tile_nbt(picked: PickedItem, tile: &mc_world::TileEntity) -> ItemStack {
    let tile_nbt = tile_entity_nbt(tile);
    let mut stack = picked.stack();

    // java: `if (itemIn == Items.skull && nbttagcompound.hasKey("Owner"))` promotes the
    // tile's Owner compound to the stack's SkullOwner root tag rather than nesting it under
    // BlockEntityTag.
    if picked.item_id == 397 {
        if let Some(owner) = tile_nbt.get_compound("Owner") {
            let mut root = mc_nbt::NbtCompound::new();
            root.insert("SkullOwner", mc_nbt::Nbt::Compound(owner.clone()));
            stack.nbt = Some(root);
            return stack;
        }
    }

    // java: every non-skull tile pick receives BlockEntityTag plus a one-line (+NBT) tooltip.
    let mut root = mc_nbt::NbtCompound::new();
    root.insert("BlockEntityTag", mc_nbt::Nbt::Compound(tile_nbt));
    let mut display = mc_nbt::NbtCompound::new();
    display.insert(
        "Lore",
        mc_nbt::Nbt::list(vec![mc_nbt::Nbt::String("(+NBT)".to_owned())]),
    );
    root.insert("display", mc_nbt::Nbt::Compound(display));
    stack.nbt = Some(root);
    stack
}

/// Serialize the subset of a tile entity retained by `mc-world` after its S35 packet was parsed.
fn tile_entity_nbt(tile: &mc_world::TileEntity) -> mc_nbt::NbtCompound {
    use mc_world::TileData;

    let mut nbt = mc_nbt::NbtCompound::new();
    match &tile.data {
        TileData::Chest(_) => {
            // The client-side chest state intentionally does not retain contents; the Java
            // writer still emits an empty Items list for the state we can represent.
            nbt.insert("Items", mc_nbt::Nbt::list(Vec::new()));
        }
        TileData::Sign(sign) => {
            for (index, text) in sign.components.iter().enumerate() {
                nbt.insert(
                    format!("Text{}", index + 1),
                    mc_nbt::Nbt::String(text.clone()),
                );
            }
        }
        TileData::Banner(banner) => {
            nbt.insert("Base", mc_nbt::Nbt::Int(banner.base_color));
            let patterns = banner
                .patterns
                .iter()
                .map(|(pattern, color)| {
                    let mut entry = mc_nbt::NbtCompound::new();
                    entry.insert("Pattern", mc_nbt::Nbt::String(pattern.clone()));
                    entry.insert("Color", mc_nbt::Nbt::Int(*color));
                    mc_nbt::Nbt::Compound(entry)
                })
                .collect();
            nbt.insert("Patterns", mc_nbt::Nbt::list(patterns));
        }
        TileData::Skull(skull) => {
            nbt.insert("SkullType", mc_nbt::Nbt::Byte(skull.skull_type as i8));
            nbt.insert("Rot", mc_nbt::Nbt::Byte(skull.rotation as i8));
            if let Some(profile) = &skull.owner_profile {
                nbt.insert("Owner", mc_nbt::Nbt::Compound(game_profile_nbt(profile)));
            } else if let Some(name) = &skull.owner_name {
                // `ExtraType` is read into a GameProfile(null, name) by TileEntitySkull; the
                // write path therefore emits an Owner compound containing just Name.
                let mut owner = mc_nbt::NbtCompound::new();
                owner.insert("Name", mc_nbt::Nbt::String(name.clone()));
                nbt.insert("Owner", mc_nbt::Nbt::Compound(owner));
            }
        }
        TileData::Beacon(beacon) => {
            // Primary/Secondary are not retained in BeaconState; Levels is the client-visible
            // field that survives the S35 action-3 decode.
            nbt.insert("Levels", mc_nbt::Nbt::Int(beacon.levels));
        }
        TileData::Spawner(spawner) => {
            nbt.insert("EntityId", mc_nbt::Nbt::String(spawner.entity_id.clone()));
            nbt.insert(
                "Delay",
                mc_nbt::Nbt::Short(i16::try_from(spawner.spawn_delay).unwrap_or_else(|_| {
                    if spawner.spawn_delay.is_negative() {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                })),
            );
            nbt.insert(
                "MinSpawnDelay",
                mc_nbt::Nbt::Short(i16::try_from(spawner.min_spawn_delay).unwrap_or_else(|_| {
                    if spawner.min_spawn_delay.is_negative() {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                })),
            );
            nbt.insert(
                "RequiredPlayerRange",
                mc_nbt::Nbt::Short(i16::try_from(spawner.activating_range).unwrap_or_else(|_| {
                    if spawner.activating_range.is_negative() {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                })),
            );
        }
        TileData::FlowerPot(pot) => {
            let item = u16::try_from(pot.item_id)
                .ok()
                .and_then(|id| mc_data::try_item_props(mc_data::ItemId(id)))
                .map_or_else(String::new, |props| props.name.to_owned());
            nbt.insert("Item", mc_nbt::Nbt::String(item));
            nbt.insert("Data", mc_nbt::Nbt::Int(pot.data));
        }
        // The retained states below carry no fields written by their Java TileEntity classes.
        TileData::Enchant(_) | TileData::Piston(_) | TileData::None | _ => {}
    }
    nbt
}

/// `NBTUtil.writeGameProfile`, retaining every signed and unsigned property the auth layer
/// decoded from a player skull.
fn game_profile_nbt(profile: &mc_auth::GameProfile) -> mc_nbt::NbtCompound {
    let mut owner = mc_nbt::NbtCompound::new();
    // java: NBTUtil.writeGameProfile omits an empty name and a null UUID.  In particular, the
    // legacy `SkullOwner` promotion must remain Name-only until a local profile cache resolves it.
    if !profile.name.is_empty() {
        owner.insert("Name", mc_nbt::Nbt::String(profile.name.clone()));
    }
    if let Some(id) = profile.id {
        owner.insert("Id", mc_nbt::Nbt::String(id.to_string()));
    }

    if !profile.properties.is_empty() {
        let mut properties = mc_nbt::NbtCompound::new();
        for property in &profile.properties {
            let mut entry = mc_nbt::NbtCompound::new();
            entry.insert("Value", mc_nbt::Nbt::String(property.value.clone()));
            if let Some(signature) = &property.signature {
                entry.insert("Signature", mc_nbt::Nbt::String(signature.clone()));
            }
            let tag = mc_nbt::Nbt::Compound(entry);
            if let Some(mc_nbt::Nbt::List(values)) = properties.get_mut(&property.name) {
                values.push(tag);
            } else {
                properties.insert(property.name.clone(), mc_nbt::Nbt::list(vec![tag]));
            }
        }
        owner.insert("Properties", mc_nbt::Nbt::Compound(properties));
    }
    owner
}

/// Resolve the creative-only entity half of `Minecraft.middleClickMouse`.
///
/// `frame_item` is the live `EntityItemFrame` watched stack, if present. The entity tracker does
/// not expose Java's concrete class hierarchy, so `EntityType` plus the generated spawn-egg
/// columns are the faithful equivalent for the remaining branches.
#[must_use]
pub fn pick_entity_item(
    entity_type: Option<EntityType>,
    frame_item: Option<&ItemStack>,
    creative: bool,
) -> Option<PickedItem> {
    if !creative {
        return None;
    }
    let entity_type = entity_type?;
    let (item_id, metadata, meta_specific) = match entity_type {
        EntityType::Painting => (321, 0, false),
        EntityType::LeashKnot => (420, 0, false),
        EntityType::ItemFrame => match frame_item {
            Some(stack) => (stack.item_id, stack.damage, true),
            None => (389, 0, false),
        },
        EntityType::MinecartFurnace => (343, 0, false),
        EntityType::MinecartChest => (342, 0, false),
        EntityType::MinecartTNT => (407, 0, false),
        EntityType::MinecartHopper => (408, 0, false),
        EntityType::MinecartCommandBlock => (422, 0, false),
        EntityType::MinecartRideable | EntityType::MinecartSpawner => (328, 0, false),
        EntityType::Boat => (333, 0, false),
        EntityType::ArmorStand => (416, 0, false),
        _ if entity_type.info().egg_primary.is_some() => {
            (383, i16::from(entity_type.net_id()), true)
        }
        _ => return None,
    };
    Some(PickedItem {
        item_id,
        metadata,
        meta_specific,
    })
}

/// `Item.registerItems:935` — the one `ItemBlock` whose item id is not its block id.
const BANNER_ITEM_ID: u16 = 425;
/// `Blocks.standing_banner`, the block `ItemBanner`'s constructor passes to `super`.
const STANDING_BANNER: mc_data::BlockId = mc_data::BlockId(176);

/// Does this item override `Item.onItemUse`?
///
/// `Item.onItemUse` (`Item.java`) is `return false`, and that answer is what leaves `flag` true in
/// `Minecraft.rightClickMouse` so the untargeted `sendUseItem` `C08` goes out. **That default is
/// the whole reason a player can eat, drink, draw a bow or block with a sword while looking at a
/// block**, and mc-rs never took it.
///
/// The classes below *do* override it, and their bodies remain unported. Rather than guess, this
/// client keeps its existing behaviour for them (treat the click as consumed), so the fall-through
/// fix introduces no new divergence for an item it cannot evaluate. `ItemSign` and `ItemSkull`
/// are dispatched to [`sign_item_use`] and [`skull_item_use`] by the live controller because
/// their client-side validation is small and packet-visible; so are `ItemEnderEye`
/// ([`ender_eye_item_use`]), `ItemFirework` (always `false` on a client) and the `ItemBlock`
/// subclass `ItemBanner` ([`banner_item_use`]). The list is every `onItemUse`
/// override in `net/minecraft/item/` that is not an `ItemBlock` subclass, read off
/// `Item.registerItems`:
///
/// `ItemFlintAndSteel` 259; `ItemReed` 287/338/354/356/379/380/390/404; `ItemHoe` 290-294;
/// `ItemSeeds` 295/361/362/372; `ItemHangingEntity` 321/389; `ItemSign` 323; `ItemDoor`
/// 324/330/427-431; `ItemMinecart` 328/342/343/407/408/422; `ItemRedstone` 331; `ItemDye` 351;
/// `ItemBed` 355; `ItemEnderEye` 381; `ItemMonsterPlacer` 383; `ItemFireball` 385;
/// `ItemSeedFood` 391/392; `ItemSkull` 397; `ItemFirework` 401; `ItemArmorStand` 416; `ItemLead`
/// 420; `ItemRecord` 2256-2267.
#[must_use]
pub fn has_on_item_use_override(stack: &ItemStack) -> bool {
    matches!(
        stack.item_id,
        259 | 287
            | 290..=295
            | 321
            | 323
            | 324
            | 328
            | 330
            | 331
            | 338
            | 342
            | 343
            | 351
            | 354..=356
            | 361
            | 362
            | 372
            | 379..=381
            | 383
            | 385
            | 389..=392
            | 397
            | 401
            | 404
            | 407
            | 408
            | 416
            | 420
            | 422
            | 427..=431
            | 2256..=2267
    )
}

/// java: `Block.isReplaceable(World, BlockPos)` (`Block.java:387`).
///
/// **The default is `return false`, not `blockMaterial.isReplaceable()`** — the two predicates
/// are different and both appear in the placement chain. This one decides whether a placement
/// lands *in* the clicked cell or one cell along `side`; `Material.isReplaceable` decides whether
/// the destination cell may be overwritten. Six blocks override it.
fn is_replaceable_in_place(world: &World, pos: BlockPos) -> bool {
    let state = world.block_at(pos);
    let Some(name) = block_name(state) else {
        return false;
    };
    match name {
        // `BlockAir:52`, `BlockDeadBush:45`, `BlockTallGrass:49`, `BlockVine:76` -> true.
        "minecraft:air" | "minecraft:deadbush" | "minecraft:tallgrass" | "minecraft:vine" => true,
        // `BlockSnow:159` -> `getValue(LAYERS) == 1`, and `getMetaFromState` stores `layers - 1`.
        "minecraft:snow_layer" => state.meta() == 0,
        // `BlockDoublePlant:69` -> the *actual* state's variant is FERN(3) or GRASS(2); the upper
        // half (`meta & 8`) carries no variant of its own and reads the lower half's.  The
        // `getActualState` call only copies that variant when the lower cell is another double
        // plant; a malformed/orphan upper state must therefore stay non-replaceable even when
        // the unrelated block below happens to use metadata 2 or 3.
        "minecraft:double_plant" => {
            let variant = if state.meta() & 8 != 0 {
                let lower = world.block_at(pos.down());
                if lower.block() != state.block() {
                    return false;
                }
                lower.meta() & 7
            } else {
                state.meta() & 7
            };
            variant == 2 || variant == 3
        }
        _ => false,
    }
}

fn block_name(state: BlockState) -> Option<&'static str> {
    let props = state.props();
    props.legal.then(|| props.block.info().name)
}

/// The `getCollisionBoundingBox` overrides that `World.canBlockBePlaced`
/// reaches for a placed block's default state and whose box is not the block's bounds.
/// `Some(None)` is Java's null (no entity test), `Some(Some(box))` Java's box, and `None` means
/// the block does not override it (the caller's bounds stand-in applies).
///
/// - `BlockSnow:43`: `(layers - 1) * 0.125` high, so the one-layer default state is a
///   **zero-height** box, which Java's strict `intersectsWith` never finds a player standing on
///   the cell's floor inside: a snow layer can be placed in the cell the player stands in. Its
///   0.125-high bounds refused that click, so the targeted C08 and the swing went missing.
/// - `BlockFenceGate:52`: closed, 1.5 high and 1/4 thick along its facing's axis; open, null.
/// - `BlockSoulSand:20`: 0.875 high.
/// - `BlockCactus:63`: inset by 1/16 on x and z, and 15/16 high.
/// - `BlockFarmland:33`: the full cube (its bounds are 15/16 high).
///
/// The arithmetic is Java's: `int` sums wrap, `float` sums are done in `f32` and then widened.
fn overridden_collision_box(state: BlockState, pos: BlockPos) -> Option<Option<Aabb>> {
    let (x, y, z) = (pos.x, pos.y, pos.z);
    let (fx, fy, fz) = (x as f32, y as f32, z as f32);
    let (x1, y1, z1) = (x.wrapping_add(1), y.wrapping_add(1), z.wrapping_add(1));
    // Java's widenings: `(double)` of an `int` and of a `float`.
    let di = <f64 as From<i32>>::from;
    let df = <f64 as From<f32>>::from;
    let bb = |min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64| {
        Some(Some(Aabb::from_bounds(
            min_x, min_y, min_z, max_x, max_y, max_z,
        )))
    };
    match state.block().0 {
        // java: `int i = LAYERS - 1; ... (double)((float)pos.getY() + (float)i * 0.125F)`, with
        // `minX..maxX` and `minZ..maxZ` the full width (`BlockSnow.getBoundsForLayers`).
        78 => {
            let i = i32::from(state.meta() & 7);
            bb(
                di(x),
                di(y),
                di(z),
                di(x) + 1.0,
                df(fy + i as f32 * 0.125),
                di(z) + 1.0,
            )
        }
        // java: `if (OPEN) return null;` then the axis of `EnumFacing.getHorizontal(meta)`:
        // SOUTH (0) and NORTH (2) are the Z axis.
        107 | 183..=187 => {
            if state.meta() & 4 != 0 {
                return Some(None);
            }
            if state.meta() & 1 == 0 {
                bb(
                    di(x),
                    di(y),
                    df(fz + 0.375),
                    di(x1),
                    df(fy + 1.5),
                    df(fz + 0.625),
                )
            } else {
                bb(
                    df(fx + 0.375),
                    di(y),
                    di(z),
                    df(fx + 0.625),
                    df(fy + 1.5),
                    di(z1),
                )
            }
        }
        // java: `(double)((float)(pos.getY() + 1) - 0.125F)`.
        88 => bb(di(x), di(y), di(z), di(x1), df(y1 as f32 - 0.125), di(z1)),
        // java: `(float)pos.getX() + 0.0625F` ... `(float)(pos.getY() + 1) - 0.0625F`.
        81 => bb(
            df(fx + 0.0625),
            di(y),
            df(fz + 0.0625),
            df(x1 as f32 - 0.0625),
            df(y1 as f32 - 0.0625),
            df(z1 as f32 - 0.0625),
        ),
        60 => bb(di(x), di(y), di(z), di(x1), di(y1), di(z1)),
        _ => None,
    }
}

/// java: `World.canBlockBePlaced` (`World.java:3153`).
///
/// ```text
/// AxisAlignedBB bb = p_175716_3_ ? null : blockIn.getCollisionBoundingBox(this, pos, blockIn.getDefaultState());
/// return bb != null && !this.checkNoEntityCollision(bb, entityIn) ? false
///      : (block.getMaterial() == Material.circuits && blockIn == Blocks.anvil ? true
///         : block.getMaterial().isReplaceable() && blockIn.canReplace(this, pos, side, itemStackIn));
/// ```
///
/// `blocking` is what `checkNoEntityCollision(bb, null)` would iterate: the boxes of every entity
/// with `preventEntitySpawning` set — `EntityLivingBase` (so the local player too), `EntityBoat`,
/// `EntityEnderCrystal`, `EntityFallingBlock`, `EntityMinecart` and `EntityTNTPrimed`.
fn can_block_be_placed(
    world: &World,
    placing: mc_data::BlockId,
    pos: BlockPos,
    side: Facing,
    blocking: &[Aabb],
) -> bool {
    // java: `blockIn.getDefaultState()`, which is not metadata zero for every block. A ladder,
    // furnace, chest or ender chest has no legal metadata-zero state, and an illegal state reads
    // as air with no collision volume, which used to skip the entity test for them entirely: a
    // ladder could be predicted inside the player where Java refuses it (a ghost ladder).
    let placing_state = placing.default_state();
    let placing_props = placing_state.props();
    // java: `getCollisionBoundingBox` of the default state. Five
    // placeable blocks override it with a box that is not their bounds (`overridden_collision_box`);
    // for every other block the bounds below are the stand-in.
    let java_box = if let Some(overridden) = overridden_collision_box(placing_state, pos) {
        overridden
    } else if !placing_props.collision.is_empty() {
        // java: `getCollisionBoundingBox` returns null for every block with no collision volume,
        // which is exactly the `Empty` shape here.
        //
        // java: `BlockLadder.getCollisionBoundingBox` ignores the state argument and calls
        // `setBlockBoundsBasedOnState(world, pos)`, which changes nothing for the air cell being
        // tested, so Java tests the ladder's *mutable* `Block` bounds: whichever facing the last
        // ladder touched by a chunk rebuild, a collision sweep or a raytrace left there (the
        // full cube before any). That history is not reproducible here; the ladder being placed
        // is the stand-in, which is exactly Java's value while building up one wall.
        let bounds = if placing.0 == LADDER_BLOCK_ID {
            BlockState::new(placing, ladder_placed_facing(world, pos, side).index())
                .props()
                .bounds
        } else {
            placing_props.bounds
        };
        let b = bounds.offset(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));
        Some(Aabb::from_bounds(
            b.min_x, b.min_y, b.min_z, b.max_x, b.max_y, b.max_z,
        ))
    } else {
        None
    };
    if let Some(bb) = java_box {
        if blocking.iter().any(|other| bb.intersects_with(other)) {
            return false;
        }
    }
    let existing = world.block_at(pos);
    let existing_material = existing.props().material;
    // java: the anvil-over-redstone special case, the one exception to `isReplaceable`.
    if existing_material == mc_data::Material::Circuits
        && block_name(placing_state) == Some("minecraft:anvil")
    {
        return true;
    }
    // java: `block.getMaterial().isReplaceable() && blockIn.canReplace(...)`. `canReplace`
    // (`Block:833`) delegates to `canPlaceBlockOnSide` -> `canPlaceBlockAt` (`:846`).
    //
    // **W-RCLICK's last piece.** The 28 `canPlaceBlockAt` and 7 `canPlaceBlockOnSide` overrides
    // used to be absent, which made this whole chain collapse to the one `isReplaceable` test
    // and left the gate suppressing a *strict subset* of the `C08`s vanilla suppresses — a
    // torch on air, a rail on a slab, a vine on a floor all reached the wire. They live in
    // `mc-world` rather than `mc-data` because 25 of the 28 read neighbouring cells.
    existing_material.is_replaceable() && world.can_place_block_on_side(placing, pos, side)
}

/// java: `ItemBlock.canPlaceBlockOnSide` (`ItemBlock.java:126`), and `ItemSlab`'s override of it
/// (`ItemSlab.java:85`), which is the one item override that changes *which cell* is tested.
///
/// This is the gate `PlayerControllerMP.onPlayerRightClick` (`:412-419`) applies before it sends
/// anything:
///
/// ```text
/// if (!flag && heldStack != null && heldStack.getItem() instanceof ItemBlock) {
///     ItemBlock itemblock = (ItemBlock)heldStack.getItem();
///     if (!itemblock.canPlaceBlockOnSide(worldIn, hitPos, side, player, heldStack)) return false;
/// }
/// ```
///
/// A `false` here means **vanilla sends no packet at all** — no `C08`, and no `C0A` swing either,
/// because `onPlayerRightClick` returned false. **That gate is only for `ItemBlock`**
/// ([`placed_block`]): the live caller checks `placed_block(stack).is_some()` first (L6). For an `ItemReed` registration this still evaluates the analogous
/// `ItemReed.onItemUse` placement predicate, but Java runs that *after* the targeted C08, so it
/// must never gate the packet. Returns `true` for every other item class.
#[must_use]
pub fn can_place_block_on_side(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    stack: &ItemStack,
    blocking: &[Aabb],
) -> bool {
    let Some(placing) = placed_block(stack).or_else(|| item_reed_block(stack)) else {
        return true;
    };

    // java: `ItemSlab.canPlaceBlockOnSide` runs *instead of* `ItemBlock`'s and only calls `super`
    // at the end — with the **original** `blockpos`, not the offset one.
    if let Some(single) = slab_block(stack) {
        let variant = i32::from(stack.damage) & 7;
        let existing = world.block_at(hit_pos);
        if existing.props().block == single && i32::from(existing.meta() & 7) == variant {
            // java: `boolean flag = getValue(HALF) == TOP` — meta bit 3.
            let top = existing.meta() & 8 != 0;
            if (side == Facing::Up && !top) || (side == Facing::Down && top) {
                return true;
            }
        }
        let neighbour = world.block_at(hit_pos.relative(side));
        if neighbour.props().block == single && i32::from(neighbour.meta() & 7) == variant {
            return true;
        }
        // Falls through to `super.canPlaceBlockOnSide(worldIn, blockpos, side, player, stack)`.
    }

    // java: `ItemBlock.canPlaceBlockOnSide` — snow layers keep the clicked cell and force the
    // side to UP; anything else that is not `isReplaceable` steps one cell along `side`.
    let existing = world.block_at(hit_pos);
    // java: `if (block == Blocks.snow_layer) { side = EnumFacing.UP; } else if
    // (!block.isReplaceable(worldIn, pos)) { pos = pos.offset(side); }`. The snow arm reassigns
    // `side` rather than `pos`, so a snow layer of *any* depth is replaced in place where
    // `isReplaceable` would only allow depth 1 — **and the reassigned `side` is now live**: it
    // is passed on to `canPlaceBlockOnSide`, where the button, lever, trapdoor, tripwire-hook
    // and vine overrides all read it. A torch clicked onto a snow layer is therefore asked
    // whether it can attach *upward*, which is the answer vanilla gives.
    let snow = block_name(existing) == Some("minecraft:snow_layer");
    let (target, side) = if snow {
        (hit_pos, Facing::Up)
    } else if is_replaceable_in_place(world, hit_pos) {
        (hit_pos, side)
    } else {
        (hit_pos.relative(side), side)
    };
    can_block_be_placed(world, placing, target, side, blocking)
}

/// The single-slab block for the three `ItemSlab` registrations in 1.8.9
/// (`Item.registerItems:616/697/764`), or `None` for every other item.
fn slab_block(stack: &ItemStack) -> Option<mc_data::BlockId> {
    match stack.item_id {
        44 | 126 | 182 => u16::try_from(stack.item_id).ok().map(mc_data::BlockId),
        _ => None,
    }
}

/// java: `WorldBorder.contains(BlockPos)` (`WorldBorder.java:34`).
///
/// `PlayerControllerMP.onPlayerRightClick` (`:397`) and `clickBlock` (`:223`) both return early
/// when this is false, so a block outside the border gets neither a `C07` nor a `C08`. The
/// creative `onPlayerDamageBlock` continuation has the same guard in its own branch (`:294`).
/// `now_millis` must be the same wall-clock sample used by the live interaction caller: Java's
/// `WorldBorder.getDiameter()` lerps between the packet endpoints during a resize.
#[must_use]
pub fn border_contains(
    border: &crate::sync::env::BorderState,
    pos: BlockPos,
    now_millis: i64,
) -> bool {
    border.contains_block(pos, border.current_diameter(now_millis))
}

/// The player inputs consumed by `EntityPlayer.getToolDigEfficiency`.
///
/// Vanilla reads these from four different objects (`InventoryPlayer`, `activePotionsMap`, the
/// player's material query, and the armour inventory). Keeping the resolved inputs together makes
/// the order of the floating-point operations explicit and lets [`PlayerController`] retain the
/// context between the initial click and subsequent damage ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DigContext {
    held_item: Option<mc_data::ItemId>,
    efficiency_level: i16,
    haste_amplifier: Option<i8>,
    fatigue_amplifier: Option<i8>,
    underwater: bool,
    aqua_affinity: bool,
}

impl DigContext {
    /// Resolves the held stack and the remaining player state into a dig context.
    ///
    /// Potion amplifiers are zero-based. `aqua_affinity` is true when any armour stack carries
    /// enchantment id 6, matching `EnchantmentHelper.getAquaAffinityModifier`.
    #[must_use]
    pub fn from_player(
        held: Option<&ItemStack>,
        haste_amplifier: Option<i8>,
        fatigue_amplifier: Option<i8>,
        underwater: bool,
        aqua_affinity: bool,
    ) -> Self {
        Self {
            held_item: held.and_then(|stack| {
                let id = u16::try_from(stack.item_id).ok()?;
                mc_data::item_props(mc_data::ItemId(id))
                    .registered
                    .then_some(mc_data::ItemId(id))
            }),
            efficiency_level: held.map_or(0, |stack| enchantment_level(stack, 32)),
            haste_amplifier,
            fatigue_amplifier,
            underwater,
            aqua_affinity,
        }
    }
}

/// `EnchantmentHelper.getEnchantmentLevel`: the first matching `{id, lvl}` in `ench`.
fn enchantment_level(stack: &ItemStack, enchantment_id: i16) -> i16 {
    stack
        .nbt
        .as_ref()
        .and_then(|nbt| nbt.get_list("ench"))
        .and_then(|list| {
            list.iter().find_map(|entry| {
                let mc_nbt::Nbt::Compound(entry) = entry else {
                    return None;
                };
                (entry.get_i16("id").unwrap_or(0) == enchantment_id)
                    .then(|| entry.get_i16("lvl").unwrap_or(0))
            })
        })
        .unwrap_or(0)
}

/// `InventoryPlayer.getStrVsBlock` followed by the item subclass' virtual dispatch.
fn strength_vs_block(state: BlockState, held_item: Option<mc_data::ItemId>) -> f32 {
    use mc_data::{ItemKind, Material};

    let Some(item) = held_item else { return 1.0 };
    let props = mc_data::item_props(item);
    let block = state.block().0;
    let material = state.props().material;
    match props.kind {
        // ItemPickaxe recognizes these three materials before delegating to ItemTool's set. The
        // set is still load-bearing for rails and both kinds of ice.
        ItemKind::Pickaxe => {
            const EFFECTIVE_ON: &[u16] = &[
                157, 16, 4, 28, 57, 56, 43, 27, 41, 14, 79, 42, 15, 22, 21, 74, 48, 87, 174, 66,
                73, 24, 179, 1, 44,
            ];
            if matches!(material, Material::Iron | Material::Anvil | Material::Rock)
                || EFFECTIVE_ON.contains(&block)
            {
                props.efficiency
            } else {
                1.0
            }
        }
        // Pumpkin, melon and ladder are the important set members outside the material branch.
        ItemKind::Axe => {
            const EFFECTIVE_ON: &[u16] = &[5, 47, 17, 162, 54, 86, 91, 103, 65];
            if matches!(material, Material::Wood | Material::Plants | Material::Vine)
                || EFFECTIVE_ON.contains(&block)
            {
                props.efficiency
            } else {
                1.0
            }
        }
        ItemKind::Shovel => {
            const EFFECTIVE_ON: &[u16] = &[82, 3, 60, 2, 13, 110, 12, 80, 78, 88];
            if EFFECTIVE_ON.contains(&block) {
                props.efficiency
            } else {
                1.0
            }
        }
        // ItemSword and ItemShears use constants, not ToolMaterial efficiency.
        ItemKind::Sword if block == 30 => 15.0,
        ItemKind::Sword
            if matches!(
                material,
                Material::Plants
                    | Material::Vine
                    | Material::Coral
                    | Material::Leaves
                    | Material::Gourd
            ) =>
        {
            1.5
        }
        ItemKind::Shears if block == 30 || material == Material::Leaves => 15.0,
        ItemKind::Shears if block == 35 => 5.0,
        _ => 1.0,
    }
}

/// `InventoryPlayer.canHeldItemHarvest` including every item subclass override in 1.8.9.
fn can_held_item_harvest(state: BlockState, held_item: Option<mc_data::ItemId>) -> bool {
    use mc_data::{ItemKind, Material};

    if state.props().material.tool_not_required() {
        return true;
    }
    let Some(item) = held_item else { return false };
    let props = mc_data::item_props(item);
    let block = state.block().0;
    match props.kind {
        ItemKind::Pickaxe => {
            let level = props
                .tool_material
                .map_or(0, |material| material.info().harvest_level);
            match block {
                49 => level == 3,                                      // obsidian
                56 | 57 | 129 | 133 | 14 | 41 | 73 | 74 => level >= 2, // diamond, emerald, gold, redstone
                15 | 42 | 21 | 22 => level >= 1,                       // iron, lapis
                _ => matches!(
                    state.props().material,
                    Material::Rock | Material::Iron | Material::Anvil
                ),
            }
        }
        ItemKind::Shovel => matches!(block, 78 | 80),
        ItemKind::Sword => block == 30,
        ItemKind::Shears => matches!(block, 30 | 55 | 132),
        _ => false,
    }
}

/// `Block.getPlayerRelativeBlockHardness` with the complete 1.8.9 player/tool formula.
///
/// The operation order matches `EntityPlayer.getToolDigEfficiency`: item strength, Efficiency,
/// Haste, Mining Fatigue, underwater, airborne; only then does `Block` choose the harvestable
/// `/30` or non-harvestable `/100` divisor.
// java: Block.java:1030, EntityPlayer.java:900
#[must_use]
pub fn player_relative_hardness_with_context(
    state: BlockState,
    context: DigContext,
    on_ground: bool,
) -> f32 {
    let hardness = state.props().hardness;
    if hardness < 0.0 {
        return 0.0;
    }

    let mut speed = strength_vs_block(state, context.held_item);
    if speed > 1.0 && context.efficiency_level > 0 && context.held_item.is_some() {
        let level = i32::from(context.efficiency_level);
        speed += (level * level + 1) as f32;
    }
    if let Some(amplifier) = context.haste_amplifier {
        speed *= 1.0 + (f32::from(amplifier) + 1.0) * 0.2;
    }
    if let Some(amplifier) = context.fatigue_amplifier {
        speed *= match amplifier {
            0 => 0.3,
            1 => 0.09,
            2 => 0.0027,
            _ => 8.1E-4,
        };
    }
    if context.underwater && !context.aqua_affinity {
        speed /= 5.0;
    }
    if !on_ground {
        speed /= 5.0;
    }

    let divisor = if can_held_item_harvest(state, context.held_item) {
        30.0
    } else {
        100.0
    };
    speed / hardness / divisor
}

/// Backwards-compatible bare-hand form of [`player_relative_hardness_with_context`].
#[must_use]
pub fn player_relative_hardness(state: BlockState, on_ground: bool) -> f32 {
    player_relative_hardness_with_context(state, DigContext::default(), on_ground)
}

/// `EntityLivingBase.getArmSwingAnimationEnd`.
///
/// Haste wins when both effects are present, matching the Java conditional expression.
#[must_use]
pub fn arm_swing_animation_end(haste_amplifier: Option<i8>, fatigue_amplifier: Option<i8>) -> i32 {
    if let Some(amplifier) = haste_amplifier {
        6 - (i32::from(amplifier) + 1)
    } else if let Some(amplifier) = fatigue_amplifier {
        6 + (i32::from(amplifier) + 1) * 2
    } else {
        6
    }
}

/// What the dig loop wants sent this tick.
///
/// Returned rather than performed, exactly as [`crate::app::state::StateMachine`] returns its
/// effects: the packets themselves are `OutboundState`'s (S14's) and the state machine here is
/// then testable with no socket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DigStep {
    /// Nothing to send.
    None,
    /// java: `C07PacketPlayerDigging(START_DESTROY_BLOCK, ..)`
    Start(BlockPos, Facing),
    /// A START whose block was locally destroyed in the same controller call. Java sends no
    /// STOP here, so the outbound dig state must be cleared before a subsequent START.
    StartAndDestroy(BlockPos, Facing),
    /// java: `C07PacketPlayerDigging(STOP_DESTROY_BLOCK, ..)`
    Finish(BlockPos, Facing),
    /// java: `resetBlockRemoving` → `ABORT_DESTROY_BLOCK`
    Abort,
}

/// One local block-hit sound requested by `PlayerControllerMP.onPlayerDamageBlock`.
///
/// This is deliberately separate from [`mc_physics::StepSound`].  Both start from a block's
/// `SoundType`, but a walking step is `step.*` at `volume * 0.15`, whereas mining uses that
/// same step-event name at `(volume + 1) / 8` and `frequency * 0.5`.  Java builds the
/// `PositionedSoundRecord` from the block position, whose overload centres it at `+0.5` on all
/// three axes.
// java: PlayerControllerMP.onPlayerDamageBlock:304
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHitSound {
    /// The block being damaged.  This is also the sound's centred position.
    pub pos: BlockPos,
    /// The state whose `SoundType` Java reads on this exact tick.
    pub state: BlockState,
}

impl BlockHitSound {
    /// The fully resolved live-audio request for this mining hit.
    ///
    /// `StepSound::sound_name` preserves the one 1.8.9 override: slime is
    /// `mob.slime.small`, not the non-existent `step.slime`.
    #[must_use]
    pub fn play_request(self) -> mc_audio::PlayRequest {
        let step = mc_physics::StepSound {
            pos: self.pos,
            state: self.state,
        };
        let info = mc_data::block_info(mc_data::block_props(self.state).block);
        mc_audio::PlayRequest {
            event: mc_assets::ResourceLocation::minecraft(step.sound_name()),
            volume: (info.step_sound_volume + 1.0) / 8.0,
            pitch: info.step_sound_pitch * 0.5,
            position: Some(self.pos.to_vec3() + DVec3::new(0.5, 0.5, 0.5)),
            attenuation: mc_audio::AttenuationType::Linear,
            repeat: false,
            repeat_delay_ticks: 0,
        }
    }
}

/// One tick of [`PlayerController::damage_block`], i.e. one call to `onPlayerDamageBlock`.
///
/// The two halves are independent and conflating them is the `C0APacketAnimation` cadence bug:
/// vanilla sends a `C07` only on a state change but swings **every tick** the method returns
/// true, which is every tick of a break.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DamageTick {
    /// The digging packet this tick, if any.
    pub step: DigStep,
    /// java: `onPlayerDamageBlock`'s return value. `Minecraft.sendClickBlockToController`
    /// (`:1496`) runs `this.thePlayer.swingItem()` exactly when it is true.
    ///
    /// **Adjudicated, do not re-litigate:** vanilla really does send one `C0APacketAnimation` per
    /// tick for the whole break. `EntityPlayerSP.swingItem` sends unconditionally; the
    /// `swingProgressInt` guard lives in `EntityLivingBase.swingItem` and gates only the
    /// animation plus a broadcast fenced behind `worldObj instanceof WorldServer`, which is never
    /// true on a client.
    pub progressed: bool,
    /// The locally audible mining hit, if this is one of Java's every-four-damage-tick beats.
    ///
    /// This is independent of [`Self::step`]: the final damage tick plays it before it discovers
    /// that the accumulated damage reached one, and ordinary accumulating ticks can request it
    /// while sending no C07 at all.
    pub hit_sound: Option<BlockHitSound>,
}

impl DamageTick {
    /// A tick that sends `step` and returns true.
    #[must_use]
    pub fn sending(step: DigStep) -> Self {
        DamageTick {
            step,
            progressed: true,
            hit_sound: None,
        }
    }

    /// A tick that sends nothing but still counts as progress, so it still swings.
    #[must_use]
    pub fn silent_progress() -> Self {
        DamageTick {
            step: DigStep::None,
            progressed: true,
            hit_sound: None,
        }
    }

    /// A tick that sends nothing and does not swing.
    #[must_use]
    pub fn stalled() -> Self {
        DamageTick {
            step: DigStep::None,
            progressed: false,
            hit_sound: None,
        }
    }

    /// A progressing tick with a block-hit sound.  Java's sound branch is independent of the
    /// later START/STOP decision, so this accepts every `DigStep` including `None` and `Finish`.
    #[must_use]
    pub fn with_hit_sound(step: DigStep, hit_sound: BlockHitSound) -> Self {
        DamageTick {
            step,
            progressed: true,
            hit_sound: Some(hit_sound),
        }
    }
}

/// `minecraft:golden_apple`. java: `Item.java:831` is the only `registerItem` call in 1.8.9 that
/// chains `.setAlwaysEdible()`, so it is the only food a full player can eat.
const GOLDEN_APPLE: i16 = 322;
/// `minecraft:arrow`. java: `ItemBow.onItemRightClick`'s `inventory.hasItem(Items.arrow)`.
pub const ARROW_ITEM_ID: i16 = 262;
/// `ItemBow.getMaxItemUseDuration`.
pub const BOW_MAX_USE_DURATION: i32 = 72_000;
/// `minecraft:milk_bucket` — the one `ItemBucket` subclass that is an `ItemBucketMilk`.
const MILK_BUCKET: i16 = 335;
/// `minecraft:snowball`, whose right-click body decrements on the client in survival.
const SNOWBALL_ITEM: i16 = 332;
/// `minecraft:egg`, whose right-click body decrements on the client in survival
/// (`ItemEgg.onItemRightClick`, the decrement is outside the `!worldIn.isRemote` spawn guard).
const EGG_ITEM: i16 = 344;
/// `minecraft:ender_pearl`, whose right-click body decrements on the client in survival.
const ENDER_PEARL_ITEM: i16 = 368;
/// `minecraft:experience_bottle`, whose right-click body decrements on the client in survival.
const EXPERIENCE_BOTTLE_ITEM: i16 = 384;
/// `minecraft:glass_bottle`, whose remote-world right-click body fills from a water hit.
const GLASS_BOTTLE_ITEM: i16 = 374;
/// `minecraft:potion`, the replacement stack made by `ItemGlassBottle`.
pub const POTION_ITEM: i16 = 373;
/// `minecraft:waterlily`, both the item and block registry id in 1.8.9.
const WATERLILY_ITEM: i16 = 111;
/// `minecraft:waterlily` block registry id used by the local lily-pad prediction.
pub const WATERLILY_BLOCK: mc_data::BlockId = mc_data::BlockId(111);
/// `minecraft:sign`, the item whose `ItemSign.onItemUse` callback places standing or wall signs.
pub const SIGN_ITEM_ID: i16 = 323;
/// `minecraft:skull`, whose remote-world callback validates placement without mutating locally.
pub const SKULL_ITEM_ID: i16 = 397;
/// java: `ItemPotion.isSplash(meta)` is `(meta & 16384) != 0`.
const SPLASH_POTION_BIT: i16 = 16384;

/// The five 1.8.9 `ItemHoe` registrations.  Unlike an `ItemBlock`, a hoe's `onItemUse` does not
/// place the held item: it changes the clicked grass/dirt state on the server only.
const HOE_ITEM_IDS: std::ops::RangeInclusive<i16> = 290..=294;

/// Block ids used by `ItemHoe.onItemUse` and its two `useHoe` states.
const GRASS_BLOCK_ID: mc_data::BlockId = mc_data::BlockId(2);
const DIRT_BLOCK_ID: mc_data::BlockId = mc_data::BlockId(3);
const FARMLAND_BLOCK_ID: mc_data::BlockId = mc_data::BlockId(60);

/// The local world mutation selected by an item that performs a five-block liquid-aware ray.
///
/// The packet is still the server's authority. This value only carries the client-side body of
/// `ItemGlassBottle`, `ItemLilyPad`, or `ItemBucket`, which Java runs after
/// `PlayerControllerMP.sendUseItem` has queued its untargeted C08.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidUsePrediction {
    /// A water block was hit and the glass bottle can edit the clicked face.
    GlassBottle {
        /// The water source block hit by the item ray.
        water: BlockPos,
    },
    /// A source-water block was hit and its cell above is air.
    LilyPad {
        /// The source-water block hit by the item ray.
        water: BlockPos,
        /// The air cell above `water` where the pad is placed.
        place: BlockPos,
        /// `ItemLilyPad` decrements only outside creative mode.
        decrement_stack: bool,
    },
    /// An empty bucket removes one source block and produces the matching filled bucket.
    BucketFill {
        /// The source-water or source-lava block removed locally.
        source: BlockPos,
        /// Item ID of the returned water/lava bucket.
        filled_item: i16,
        /// Creative mode keeps the held empty bucket even though the source still disappears.
        preserve_held: bool,
    },
    /// A filled bucket successfully places its contents, or vaporizes water in the Nether.
    BucketPour {
        /// Face-adjacent destination selected by the non-liquid ray.
        target: BlockPos,
        /// The flowing liquid state to install, or `None` for Nether water vaporization.
        state: Option<BlockState>,
        /// Survival replaces the filled bucket with one empty bucket; creative preserves it.
        replace_with_empty: bool,
    },
}

/// The action an item exposes while it is in use.
///
/// This is the client-visible `EnumAction` table from 1.8.9, kept in the interaction module so
/// first-person rendering, local blocking, and any future use-effects consumer can all ask the
/// same item-class question.  It is deliberately not the renderer's compressed action enum:
/// Java distinguishes eating from drinking even though a renderer may choose to share their
/// transform.
///
/// java: `item/EnumAction.java:3` and `Item.getItemUseAction(ItemStack)` (`Item.java:365`), with
/// overrides in `ItemFood:87`, `ItemPotion:137`, `ItemBucketMilk:48`, `ItemSword:84`, and
/// `ItemBow:113`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ItemUseAction {
    /// The item has no use animation. java: `EnumAction.NONE`.
    #[default]
    None,
    /// An `ItemFood` stack is being eaten. java: `EnumAction.EAT`.
    Eat,
    /// A potion or milk bucket is being drunk. java: `EnumAction.DRINK`.
    Drink,
    /// A sword is being used to block. java: `EnumAction.BLOCK`.
    Block,
    /// A bow is being drawn. java: `EnumAction.BOW`.
    Bow,
}

/// The model variant selected by `RenderItem.renderItemModel` while a bow is in use.
///
/// java: `RenderItem.java:290-303` computes
/// `item.getMaxItemUseDuration() - entityplayer.getItemInUseCount()` and selects
/// `bow_pulling_0` for `1..=13`, `bow_pulling_1` for `14..=17`, and `bow_pulling_2` for
/// `>=18`.  `Base` is the ordinary `bow` model used at elapsed zero (and by the defensive
/// negative-duration case); callers must still gate this oracle on `ItemUseAction::Bow`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BowPullStage {
    /// The un-drawn `bow` model.
    #[default]
    Base,
    /// `bow_pulling_0`, selected after the first use tick.
    Pulling0,
    /// `bow_pulling_1`, selected after fourteen use ticks.
    Pulling1,
    /// `bow_pulling_2`, selected after eighteen use ticks.
    Pulling2,
}

/// Java's complete first-person use snapshot, assembled from one latched `itemInUse`.
///
/// Keeping action, remaining count, maximum duration, and elapsed duration together prevents a
/// caller from accidentally mixing the current held stack with the stack that actually started
/// the use.  The renderer may use `elapsed_duration` for bow model/FOV decisions and
/// `remaining_count` for the first-person transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemUseRenderState {
    /// `ItemStack.getItemUseAction()` for the latched item.
    pub action: ItemUseAction,
    /// `EntityLivingBase.getItemInUseCount()`.
    pub remaining_count: i32,
    /// `ItemStack.getMaxItemUseDuration()`.
    pub max_duration: i32,
    /// `EntityPlayer.getItemInUseDuration()` (`max_duration - remaining_count`).
    pub elapsed_duration: i32,
}

/// Java's `RenderItem` bow model threshold oracle.
#[must_use]
pub fn bow_pull_stage(elapsed_duration: i32) -> BowPullStage {
    match elapsed_duration {
        1..=13 => BowPullStage::Pulling0,
        14..=17 => BowPullStage::Pulling1,
        18.. => BowPullStage::Pulling2,
        _ => BowPullStage::Base,
    }
}

/// Java's bow branch in `AbstractClientPlayer.getFovModifier` (`:126-140`).
///
/// This is a pure target multiplier; `EntityRenderer.updateFovModifierHand` owns the separate
/// half-distance interpolation.  It intentionally retains Java's behavior for a negative input
/// instead of clamping it: normal client state never supplies one, but matching the arithmetic
/// makes malformed-state tests useful.
#[must_use]
pub fn bow_fov_multiplier(elapsed_duration: i32) -> f32 {
    let mut pull = elapsed_duration as f32 / 20.0;
    if pull > 1.0 {
        pull = 1.0;
    } else {
        pull *= pull;
    }
    1.0 - pull * 0.15
}

/// Java `ItemBow.onPlayerStoppedUsing` charge curve, or `None` when the release is too weak to
/// fire. `time_left` is the live `EntityPlayer.itemInUseCount` passed through
/// `ItemStack.onPlayerStoppedUsing`; keeping the arithmetic in `f32` preserves Java's float
/// rounding before the `0.1D` gate.
#[must_use]
pub fn bow_release_strength(time_left: i32) -> Option<f32> {
    let elapsed = BOW_MAX_USE_DURATION.wrapping_sub(time_left);
    let mut strength = elapsed as f32 / 20.0;
    strength = (strength * strength + strength * 2.0) / 3.0;
    if f64::from(strength) < 0.1 {
        return None;
    }
    Some(strength.min(1.0))
}

/// The exact `ItemStack.getItemUseAction()` result for a registered 1.8.9 item.
///
/// The action does not depend on stack count, metadata, NBT, hunger, or whether the item has
/// actually started a use; those facts are checked by [`item_use_duration`] and the player state
/// machine.  Splash potions still report `Drink` here because Java's `ItemPotion` override is
/// unconditional; the splash check belongs only to `onItemRightClick`.
#[must_use]
pub fn item_use_action(stack: &ItemStack) -> ItemUseAction {
    use mc_data::ItemKind;

    let Ok(id) = u16::try_from(stack.item_id) else {
        return ItemUseAction::None;
    };
    let props = mc_data::item_props(mc_data::ItemId(id));
    if !props.registered {
        return ItemUseAction::None;
    }

    if stack.item_id == MILK_BUCKET {
        return ItemUseAction::Drink;
    }
    match props.kind {
        ItemKind::Bow => ItemUseAction::Bow,
        ItemKind::Sword => ItemUseAction::Block,
        ItemKind::Food => ItemUseAction::Eat,
        ItemKind::Potion => ItemUseAction::Drink,
        _ => ItemUseAction::None,
    }
}

/// Window-0 armor slot selected by `ItemArmor.onItemRightClick`.
///
/// Java's `EntityLiving.getArmorPosition(stack) - 1` produces armor indices `0..=3` in
/// helmet/chest/legs/feet order. `ContainerPlayer` stores those same four entries at slots
/// `5..=8`.
#[must_use]
pub fn armor_inventory_slot(stack: &ItemStack) -> Option<usize> {
    let item_id = u16::try_from(stack.item_id).ok()?;
    let props = mc_data::try_item_props(mc_data::ItemId(item_id))?;
    (props.kind == mc_data::ItemKind::Armor)
        .then_some(props.armor_slot)
        .flatten()
        .map(|slot| 5 + usize::from(slot))
}

/// Whether a right-click item decrements its held stack on the client before the server reply.
///
/// This is intentionally narrower than "the item eventually gets consumed": the four
/// projectile items below and splash potions mutate their `ItemStack` outside the
/// `!world.isRemote` spawn guard,
/// while `ItemFireball` only mutates on the server, `ItemEnderEye` decrements only inside its
/// `!worldIn.isRemote` block, and timed-use items are consumed by the server's release/finish
/// path. The caller still owns the inventory write-back and the equipped-progress reset from
/// `PlayerControllerMP.sendUseItem`.
///
/// java: `ItemEnderPearl.onItemRightClick` (`:29`), `ItemExpBottle.onItemRightClick` (`:30`),
/// `ItemSnowball.onItemRightClick` (`:24`), `ItemEgg.onItemRightClick` (`:24`), and splash
/// `ItemPotion.onItemRightClick` (`:147-162`), consumed by
/// `client/multiplayer/PlayerControllerMP.sendUseItem` (`:466-479`). The egg was missing (H3): a thrown egg kept its count and skipped `resetEquippedProgress2` until
/// the server's S2F.
#[must_use]
pub fn client_right_click_decrements_stack(stack: &ItemStack, creative: bool) -> bool {
    if creative {
        return false;
    }
    matches!(
        stack.item_id,
        SNOWBALL_ITEM | EGG_ITEM | ENDER_PEARL_ITEM | EXPERIENCE_BOTTLE_ITEM
    ) || (stack.item_id == POTION_ITEM && stack.damage & 16_384 != 0)
}

/// The player state the five `Item.onItemRightClick` overrides consult before they call
/// `setItemInUse`.
///
/// Vanilla reaches these off `EntityPlayer` directly; the port has to be handed them because
/// [`item_use_duration`] is a free function over an `ItemStack`, not a method on a player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UseGuards {
    /// java: `capabilities.isCreativeMode`. `ItemBow` draws without arrows in creative.
    pub creative: bool,
    /// java: `capabilities.disableDamage` — set by *both* `CREATIVE` and `SPECTATOR` in
    /// `WorldSettings.GameType.configurePlayerCapabilities`. It is the second half of
    /// `EntityPlayer.canEat` (`:2086`), and is why a creative player cannot eat.
    pub disable_damage: bool,
    /// java: `foodStats.foodLevel`; `FoodStats.needFood()` (`:132`) is `foodLevel < 20`.
    pub food_level: i32,
    /// java: `InventoryPlayer.hasItem(Items.arrow)`, which scans `mainInventory` only.
    pub has_arrow: bool,
}

/// The virtual `ItemStack.getMaxItemUseDuration()` result, independent of whether a new use is
/// currently permitted.  This is separate from [`item_use_duration`]'s right-click guard so a
/// renderer reading an already-latched use cannot accidentally substitute a caller-provided
/// default (for example, `32` for a bow).
#[must_use]
pub fn item_max_use_duration(stack: &ItemStack) -> Option<i32> {
    use mc_data::ItemKind;

    let id = u16::try_from(stack.item_id).ok()?;
    let props = mc_data::item_props(mc_data::ItemId(id));
    if !props.registered {
        return None;
    }
    if stack.item_id == MILK_BUCKET {
        return Some(32);
    }
    match props.kind {
        ItemKind::Bow | ItemKind::Sword => Some(BOW_MAX_USE_DURATION),
        ItemKind::Food | ItemKind::Potion => Some(32),
        _ => None,
    }
}

/// How long a right-click with `stack` keeps the item in use, or `None` when the click does not
/// begin a timed use at all.
///
/// java: `Item.getMaxItemUseDuration` (`Item.java:373`) returns `0` on the base class and is
/// overridden by exactly five classes in 1.8.9 — `ItemBow` (72000), `ItemSword` (72000),
/// `ItemFood` (32), `ItemPotion` (32) and `ItemBucketMilk` (32) — and those same five are the
/// only `setItemInUse` call sites in `item/`. Each override's `onItemRightClick` carries its own
/// guard, reproduced here:
///
/// | class | guard | java |
/// |---|---|---|
/// | `ItemBow` | `isCreativeMode \|\| inventory.hasItem(Items.arrow)` | `ItemBow.java:123` |
/// | `ItemSword` | none | `ItemSword.java:100` |
/// | `ItemFood` | `canEat(alwaysEdible)` | `ItemFood.java:97` |
/// | `ItemPotion` | not a splash potion (splashes are *thrown*, not drunk) | `ItemPotion.java:147` |
/// | `ItemBucketMilk` | none | `ItemBucketMilk.java:56` |
///
/// `ItemSword`'s 72000 is not a typo: right-clicking a sword in 1.8.9 is *blocking*, and it goes
/// through the identical `itemInUse` machinery — including the `RELEASE_USE_ITEM` on let-go.
#[must_use]
pub fn item_use_duration(stack: &ItemStack, guards: UseGuards) -> Option<i32> {
    use mc_data::ItemKind;
    let id = u16::try_from(stack.item_id).ok()?;
    let props = mc_data::item_props(mc_data::ItemId(id));
    if !props.registered {
        return None;
    }
    // Milk is matched by id, not by [`mc_data::ItemKind`]. The extractor follows `ItemFood`'s
    // subclasses (`ItemAppleGold`, `ItemSoup`, `ItemFishFood`, `ItemSeedFood` all report
    // `ItemKind::Food`) but *not* `ItemBucket`'s: `minecraft:milk_bucket` reports
    // `ItemKind::Generic` while its `java_class` is `net.minecraft.item.ItemBucketMilk`. Keying
    // this one on the id keeps the rule right without reaching into `mc-data`, which is another
    // section's crate. The three real `ItemKind::Bucket` items — empty, water and lava — have no
    // use duration, so nothing is lost by not matching that kind at all.
    let max_duration = item_max_use_duration(stack)?;
    if stack.item_id == MILK_BUCKET {
        return Some(max_duration);
    }
    match props.kind {
        ItemKind::Bow => (guards.creative || guards.has_arrow).then_some(max_duration),
        ItemKind::Sword => Some(max_duration),
        // java: `canEat(alwaysEdible)` = `(alwaysEdible || foodStats.needFood()) &&
        // !capabilities.disableDamage`.
        ItemKind::Food => {
            let always_edible = stack.item_id == GOLDEN_APPLE;
            ((always_edible || guards.food_level < 20) && !guards.disable_damage)
                .then_some(max_duration)
        }
        ItemKind::Potion => (stack.damage & SPLASH_POTION_BIT == 0).then_some(max_duration),
        _ => None,
    }
}

/// What a right-click is aimed at, as far as reaching
/// `sendUseItem` goes (see [`right_click_starts_use`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseAim {
    /// Nothing, or an air block: `rightClickMouse` goes straight to `sendUseItem`.
    Nothing,
    /// A solid block. `consumed` when `onPlayerRightClick` returns `true` for it (an activated
    /// block, or the item's own `onItemUse`), which ends the click before `sendUseItem`
    /// ([`block_right_click_consumes`]).
    Block {
        /// `onPlayerRightClick`'s result.
        consumed: bool,
    },
    /// An entity. Never counted (see [`right_click_starts_use`]).
    Entity,
}

/// 1.7 Animations' "Using items stops mining" (it extends the
/// earlier "Eating stops mining"):
/// whether a right-click with `stack` aimed at `aim` would start using the item right now, that
/// is, reach `sendUseItem` and have the item's own `onItemRightClick` call `setItemInUse`.
///
/// The item half is [`item_use_duration`], the five 1.8.9 use items with their guards: a sword
/// always blocks, a bow draws only with an arrow in the main inventory or in creative, food needs
/// `canEat` (hungry, or a golden apple; never with `disableDamage`), a potion is drunk unless it
/// is a splash potion, milk always.
///
/// The aim half is `Minecraft.rightClickMouse`'s `flag`: a click consumed by the block (a chest,
/// a bed, a fence, a door...; [`block_right_click_consumes`]) never reaches `sendUseItem`. An
/// entity is never counted: while a block is being broken the crosshair is on an entity for one
/// tick at most (`sendClickBlockToController` resets the dig at the end of that tick, and the
/// held use button then starts the use on the next one, as in vanilla), and whether an entity
/// takes the click needs its whole interaction table.
///
/// A press that would not start a use must leave a dig alone: vanilla keeps mining then, and
/// cancelling would only throw the progress away (a sword aimed at the BedWars bed you are
/// breaking would lose the bed's progress *and* not block).
#[must_use]
pub fn right_click_starts_use(stack: &ItemStack, guards: UseGuards, aim: UseAim) -> bool {
    let reaches_use = match aim {
        UseAim::Nothing => true,
        UseAim::Block { consumed } => !consumed,
        UseAim::Entity => false,
    };
    reaches_use && item_use_duration(stack, guards).is_some()
}

/// Whether `PlayerControllerMP.onPlayerRightClick` would return
/// `true` for a right-click with the use item `held` on the solid block at `pos`, read-only: the
/// driver's `on_player_right_click` cascade without any of its bodies, sounds or packets.
///
/// * `(!player.isSneaking() || player.getHeldItem() == null) && block.onBlockActivated(...)`:
///   the held stack is never null here, so sneaking skips activation. The activated set is the
///   cascade's: [`activate_remote_block`] (chests, bed, fences, workbench, furnace, ...), the
///   command block in creative, buttons, signs, cake, TNT with flint and steel or a fire charge,
///   doors (iron too), fence gates, trapdoors (iron too), repeaters and comparators with
///   `allowEdit`, a jukebox holding a record, a flower pot taking a plant, the dragon egg.
///   Redstone ore runs its body but returns `false`.
/// * `heldStack.onItemUse(...)`: of the five use items only `ItemSeedFood` (carrot, potato)
///   overrides it, planting on farmland ([`seeds_item_use`]).
///
/// `x4_input`'s `the_dry_run_matches_the_real_right_click_for_every_block` holds this to the
/// real cascade, block by block. The world border is not checked: outside it vanilla skips the
/// activation and *does* reach `sendUseItem`, so this errs towards leaving the dig alone.
#[must_use]
pub fn block_right_click_consumes(
    world: &World,
    pos: BlockPos,
    face: Facing,
    held: &ItemStack,
    sneaking: bool,
    allow_edit: bool,
    creative: bool,
) -> bool {
    let activated = !sneaking && block_activation_consumes(world, pos, held, allow_edit, creative);
    activated
        || (matches!(held.item_id, 391 | 392)
            && matches!(
                seeds_item_use(world, pos, face, held, allow_edit),
                BlockUseResult::Consumed { .. }
            ))
}

/// The `consumed` result of the driver's `onBlockActivated` cascade for `held` at `pos`, without
/// running it (see [`block_right_click_consumes`]).
fn block_activation_consumes(
    world: &World,
    pos: BlockPos,
    held: &ItemStack,
    allow_edit: bool,
    creative: bool,
) -> bool {
    if let Some(activation) = activate_remote_block(world, pos, allow_edit) {
        return activation.consumed;
    }
    let state = world.block_at(pos);
    match state.block().0 {
        // `CommandBlockLogic.tryOpenEditCommandBlock`: creative only.
        137 => creative,
        // Buttons (pressed or not), signs, cake (even when `canEat` refuses the bite), fence
        // gates, trapdoors (iron returns true unchanged), the dragon egg.
        77 | 143 | 63 | 68 | 92 | 107 | 183..=187 | 96 | 167 | 122 => true,
        // `BlockTNT`: flint and steel or a fire charge.
        46 => matches!(held.item_id, 259 | 385),
        // `BlockDoor`: iron consumes unchanged; a wooden upper half resolves its lower half,
        // and a malformed one returns false (`World::activate_door`).
        64 | 71 | 193..=197 => {
            if state.block().0 == 71 || state.meta() & 8 == 0 {
                true
            } else {
                let lower = world.block_at(pos.down());
                lower.block() == state.block() && lower.meta() & 8 == 0
            }
        }
        // Repeaters and comparators cycle only with `allowEdit`.
        93 | 94 | 149 | 150 => allow_edit,
        // A jukebox with a record ejects it.
        84 => state.meta() != 0,
        // A flower pot takes a plant `ItemBlock` into an empty pot (`activate_flower_pot`).
        140 => placed_block(held).is_some_and(|block| {
            matches!(block.0, 6 | 31 | 32 | 37 | 38 | 39 | 40 | 81)
                && (block.0 != 31 || held.damage as u8 == 2)
                && world
                    .tile_entity(pos)
                    .and_then(|tile| tile.flower_pot())
                    .is_some_and(|pot| pot.item_id == 0)
        }),
        _ => false,
    }
}

/// The client-local `FoodStats` values after a completed `ItemFood` use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinishedFoodStats {
    /// Hunger after Java's upper cap of 20.
    pub food_level: i32,
    /// Saturation after adding `heal * modifier * 2` and capping at the new hunger value.
    pub saturation: f32,
}

/// Apply `ItemFood.onItemUseFinish`'s `FoodStats.addStats` arithmetic.
///
/// This consumes the generated virtual `getHealAmount(stack)` and
/// `getSaturationModifier(stack)` values, including `ItemFishFood`'s metadata variants. Stack
/// replacement/decrement remains with the live controller because it owns the selected slot.
#[must_use]
pub fn finished_food_stats(
    stack: &ItemStack,
    food_level: i32,
    saturation: f32,
) -> Option<FinishedFoodStats> {
    let id = u16::try_from(stack.item_id).ok()?;
    let props = mc_data::item_props(mc_data::ItemId(id));
    if !props.registered || props.kind != mc_data::ItemKind::Food {
        return None;
    }
    let food = props.food_stats(stack.damage);
    let food_level = food_level.saturating_add(i32::from(food.heal)).min(20);
    let saturation =
        (saturation + f32::from(food.heal) * food.saturation * 2.0).min(food_level as f32);
    Some(FinishedFoodStats {
        food_level,
        saturation,
    })
}

/// `PlayerControllerMP`'s block-breaking half.
///
/// java: `client/multiplayer/PlayerControllerMP.java` — `clickBlock` (`:198`),
/// `onPlayerDamageBlock` (`:285`), `resetBlockRemoving` (`:274`) — plus `Minecraft`'s
/// `leftClickCounter` and `rightClickDelayTimer`, which are the two counters that stop a held
/// mouse button flooding the server.
///
/// The packets are S14's [`crate::outbound::OutboundState`]; this owns only the *decision*.
#[derive(Debug, Clone)]
pub struct PlayerController {
    /// java: `isHittingBlock`
    hitting: bool,
    /// java: `currentBlock`, initially `new BlockPos(-1, -1, -1)`. It keeps its X/Z with
    /// `Y = -1` after a local destroy and is *not* cleared by `resetBlockRemoving`, which is why
    /// the held-button continuation can resume a block without a new START.
    current_block: BlockPos,
    /// The `currentBlock` that `clickBlock`'s or `resetBlockRemoving`'s
    /// `ABORT_DESTROY_BLOCK` names, recorded when Java's `if (this.isHittingBlock)` guard holds
    /// and handed to the packet layer once by [`Self::take_pending_abort`]. The controller, not
    /// the packet layer, knows `isHittingBlock`/`currentBlock` (H3).
    pending_abort: Option<BlockPos>,
    /// The arguments of the controller's latest `World.sendBlockBreakProgress(entityId,
    /// currentBlock, progress)` call, handed once to the crack overlay by
    /// [`Self::take_break_progress`]. Java's local crack follows these calls only.
    pending_break_progress: Option<(BlockPos, i32)>,
    /// java: `currentItemHittingBlock`.  `isHittingPosition` deliberately compares the item
    /// type, tag and (only for non-damageable items) metadata rather than the whole stack:
    /// changing a tool restarts the dig, while durability loss and stack-count changes do not.
    current_item_hitting_block: Option<ItemStack>,
    /// The first-click `Block.onBlockClicked` result awaiting the application-layer effect
    /// consumer. It is intentionally separate from [`DigStep`]: the callback is local feedback,
    /// not another digging packet.
    pending_block_click: Option<LocalBlockClick>,
    /// java: `curBlockDamageMP`, `0.0..1.0`
    damage: f32,
    /// java: `blockHitDelay`, the five-tick cooldown after a break or a creative click.
    hit_delay: i32,
    /// java: `stepSoundTickCounter`
    step_sound_counter: f32,
    /// java: `Minecraft.leftClickCounter` — ten ticks of silence after a swing that hit nothing.
    left_click_counter: i32,
    /// java: `Minecraft.rightClickDelayTimer` — four ticks between held right-click repeats.
    right_click_delay: i32,
    /// java: `EntityPlayer.itemInUse` (`:2102`). Vanilla stores the *same object* the hand
    /// holds and later compares it by reference (`itemstack != this.itemInUse`). The production
    /// hotbar path preserves that distinction with [`Self::tick_item_in_use_at_slot`].
    item_in_use: Option<ItemStack>,
    /// Hotbar slot that owned `item_in_use` when the use began. Java retains the same
    /// `ItemStack` object through `onItemUseFinish`; preserving its slot prevents a status-9
    /// completion from consuming whichever different slot happens to be selected later.
    item_in_use_slot: Option<u8>,
    /// Client-local identity of the inventory `ItemStack` object latched by the production
    /// hotbar path. Equal-valued packet replacements still receive a different identity.
    item_in_use_identity: Option<u64>,
    /// java: `EntityPlayer.itemInUseCount`, counted **down** from
    /// [`item_use_duration`]. `getItemInUseDuration` is `max - count`, which is what a bow's
    /// draw animation and its damage are computed from.
    item_in_use_count: i32,
    /// Resolved inputs to `EntityPlayer.getToolDigEfficiency` for the local player.
    dig_context: DigContext,
}

impl Default for PlayerController {
    fn default() -> Self {
        Self {
            hitting: false,
            // java: `private BlockPos currentBlock = new BlockPos(-1, -1, -1);`. A `(0, 0, 0)`
            // default would let an empty-handed first hold at the origin "continue" a dig that
            // never started, now that the continuation (correctly) ignores `isHittingBlock`.
            current_block: BlockPos::new(-1, -1, -1),
            current_item_hitting_block: None,
            pending_abort: None,
            pending_break_progress: None,
            pending_block_click: None,
            damage: 0.0,
            hit_delay: 0,
            step_sound_counter: 0.0,
            left_click_counter: 0,
            right_click_delay: 0,
            item_in_use: None,
            item_in_use_slot: None,
            item_in_use_identity: None,
            item_in_use_count: 0,
            dig_context: DigContext::default(),
        }
    }
}

/// java: `ItemStack.canDestroy`. `CanDestroy` is a list of block registry names, never state
/// names: every metadata variant of a block compares equal to Java's one `Block` instance.
fn held_nbt_allows_block(held: Option<&ItemStack>, key: &str, target: BlockState) -> bool {
    let target = target.block().info().id;
    held.and_then(|stack| stack.nbt.as_ref())
        .and_then(|nbt| nbt.get(key))
        .and_then(|tag| {
            let entries = tag.as_list()?;
            // `NBTTagCompound.getTagList("CanDestroy", 8)` returns a fresh empty list when
            // the stored list has a non-string element type. Empty lists are allowed regardless
            // of their stored type, but cannot grant permission either.
            (entries.is_empty() || tag.list_tag_type() == Some(mc_nbt::TagType::String))
                .then_some(entries)
        })
        .is_some_and(|entries| {
            entries.iter().any(|entry| {
                entry
                    .as_str()
                    .and_then(block_from_name)
                    .is_some_and(|block| block == target)
            })
        })
}

/// java: `Block.getBlockFromName`: the registry name, else `Integer.parseInt(name)` as a numeric
/// id (`getObjectById`, whose unknown ids fall back to air, which no outline or dig targets).
/// The numeric fallback was missing.
fn block_from_name(name: &str) -> Option<mc_data::BlockId> {
    mc_data::block_by_name(name).or_else(|| {
        let id = mc_data::BlockId(u16::try_from(name.parse::<i32>().ok()?).ok()?);
        (id.0 != 0 && mc_data::block_info(id).registered).then_some(id)
    })
}

/// java: `ItemStack.canDestroy`, used by the adventure dig gate and the adventure
/// block outline (`EntityRenderer.isDrawBlockOutline`).
pub(crate) fn held_can_destroy(held: Option<&ItemStack>, target: BlockState) -> bool {
    held_nbt_allows_block(held, "CanDestroy", target)
}

/// java: `ItemStack.canPlaceOn`, used by `EntityPlayer.canPlayerEdit` when `allowEdit` is false,
/// and by the adventure block outline.
/// The list stores block registry names, not state names, so metadata never changes the result.
pub(crate) fn held_can_place_on(held: Option<&ItemStack>, target: BlockState) -> bool {
    held_nbt_allows_block(held, "CanPlaceOn", target)
}

/// The client-visible edit gate shared by `ItemGlassBottle` and `ItemLilyPad`.
///
/// `WorldClient.isBlockModifiable` inherits the client `World` implementation, which returns
/// `true` in 1.8.9. `EntityPlayer.canPlayerEdit` is therefore the only permission branch that
/// can decline here: `allowEdit`, or an adventure `CanPlaceOn` entry for the block behind the
/// clicked face. Both items inherit `Item.canItemEditBlocks() == false`.
#[must_use]
pub fn can_player_edit(
    world: &World,
    target: BlockPos,
    face: Facing,
    held: Option<&ItemStack>,
    allow_edit: bool,
) -> bool {
    if allow_edit {
        return true;
    }
    let clicked = target.relative(face.opposite());
    held_can_place_on(held, world.block_at(clicked))
}

/// Dispatch the client-visible `BlockButton.onBlockActivated` body for the right-click caller.
///
/// `PlayerControllerMP.onPlayerRightClick` owns the spectator/sneaking guard and must invoke
/// this before queueing the targeted `C08`; this bridge owns neither of those caller policies nor
/// button placement validation. The world helper cycles an unpressed wood/stone button, consumes
/// an already-pressed button without a second write, and leaves the server's later block update
/// free to reset the prediction.
#[must_use]
pub fn activate_button(
    world: &mut World,
    pos: mc_math::BlockPos,
) -> Option<mc_world::world::LocalBlockActivation> {
    world.activate_button(pos)
}

/// Dispatch the remote-world-only `Block.onBlockActivated` family whose server arm opens a GUI
/// or otherwise owns the interaction, but whose client arm returns `true` without a local write.
///
/// `PlayerControllerMP.onPlayerRightClick` still sends the targeted C08 for these blocks, while
/// the consumed result prevents the caller from treating the held stack as a second, adjacent
/// placement.  The server remains authoritative for every GUI, tile inventory, sleep state,
/// redstone transition, and note event.  `BlockFenceGate` is intentionally not in this table:
/// its callback mutates the client state and is already dispatched through `World::activate_fence_gate`.
///
/// The daylight detector is the one listed override whose client return is conditional: with
/// `allowEdit == false`, Java delegates to `Block.onBlockActivated` and returns `false`.
///
/// java: `BlockAnvil`, `BlockBeacon`, `BlockBed`, `BlockBrewingStand`, `BlockCauldron`,
/// `BlockChest`, `BlockDispenser`/`BlockDropper`, `BlockEnchantmentTable`, `BlockEnderChest`,
/// `BlockFence`/`BlockNewFence`, `BlockFurnace`, `BlockHopper`, `BlockLever`, `BlockNote`, and
/// `BlockWorkbench` `onBlockActivated` implementations.
#[must_use]
pub fn activate_remote_block(
    world: &World,
    pos: BlockPos,
    allow_edit: bool,
) -> Option<mc_world::world::LocalBlockActivation> {
    let state = world.block_at(pos);
    if !state.props().legal {
        return None;
    }
    let block = state.block().0;
    let consumed = match block {
        // GUI-only remote returns.  These are deliberately a single shared gate: no client-side
        // container/tile mutation is representable at this call site, and each Java body returns
        // before doing any such work when `world.isRemote` is true.
        25 // note block
        | 26 // bed
        | 54 // chest
        | 58 // workbench
        | 61 | 62 // furnace (unlit/lit)
        // Every `BlockFence` instance: `return worldIn.isRemote ? true : attachToFence(...)`
        // (`BlockFence.java:172-175`), and the nether brick fence (113) is one of them.
        | 85 | 113 | 188..=192 // oak, nether brick and new wood fences
        | 116 // enchantment table
        | 117 // brewing stand
        | 118 // cauldron
        | 130 // ender chest
        | 138 // beacon
        | 145 // anvil
        | 146 // trapped chest
        | 154 // hopper
        | 23 | 158 // dispenser/dropper
        | 69 // lever
        => true,
        // BlockDaylightDetector returns true remotely only when the player may edit; otherwise
        // its explicit `super.onBlockActivated` call returns false.
        151 | 178 => allow_edit,
        _ => return None,
    };

    Some(mc_world::world::LocalBlockActivation {
        consumed,
        effect: None,
    })
}

/// The client-side result of `ItemSign.onItemUse` after its validation chain succeeds.
///
/// `ItemSign` returns from the `worldIn.isRemote` branch before mutating the world, consuming the
/// stack, or opening the newly-created tile entity.  The driver only needs the successful result
/// to suppress `sendUseItem`; retaining the target and face makes the prediction explicit for a
/// future local placement bridge without pretending the client has already placed a sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignUsePrediction {
    /// The cell one step from the clicked block where the server will place the sign.
    pub target: BlockPos,
    /// The face passed to `ItemSign.onItemUse`; `UP` selects a standing sign and a horizontal
    /// face selects a wall sign.
    pub face: Facing,
}

impl SignUsePrediction {
    /// Whether the successful callback selects `Blocks.standing_sign`.
    #[must_use]
    pub fn is_standing(self) -> bool {
        self.face == Facing::Up
    }
}

/// Classify the client-visible body of `ItemSign.onItemUse`.
///
/// java: `ItemSign.onItemUse` (`ItemSign.java:31-79`).  The callback rejects DOWN and any
/// clicked block whose material is not solid, offsets to the destination, applies
/// `EntityPlayer.canPlayerEdit`, then asks `Blocks.standing_sign.canPlaceBlockAt`.  On the client
/// `worldIn.isRemote` is true, so that last successful branch returns `true` without local block,
/// tile, inventory, or GUI mutation.  `World::can_place_sign_at` owns the shared cactus-neighbor
/// and replaceable-destination rules.
#[must_use]
pub fn sign_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> Option<SignUsePrediction> {
    if held.item_id != SIGN_ITEM_ID || side == Facing::Down {
        return None;
    }
    // java: `worldIn.getBlockState(pos).getBlock().getMaterial().isSolid()`.
    if !world.block_at(hit_pos).props().material.info().solid {
        return None;
    }

    let target = hit_pos.relative(side);
    if !can_player_edit(world, target, side, Some(held), allow_edit)
        || !world.can_place_sign_at(target)
    {
        return None;
    }

    Some(SignUsePrediction { target, face: side })
}

/// The client-side result of `ItemSkull.onItemUse` after its placement guards succeed.
///
/// Like [`SignUsePrediction`], this is deliberately read-only: Java's remote-world branch
/// returns `true` before placing the skull block, creating its tile entity, or consuming the
/// stack. The server remains authoritative for those changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkullUsePrediction {
    /// The replaceable cell where the server may place the skull.
    pub target: BlockPos,
    /// The face stored in `BlockSkull.FACING` by the server-side branch.
    pub face: Facing,
}

/// Classify the client-visible body of `ItemSkull.onItemUse`.
///
/// java: `ItemSkull.onItemUse` rejects DOWN, keeps the target in place when the clicked block's
/// `Block.isReplaceable` override accepts it, and otherwise requires a solid clicked material
/// before offsetting. It then applies `canPlayerEdit` and `Blocks.skull.canPlaceBlockAt`; the
/// latter inherits `Block`'s replaceable-material destination check. A successful remote-world
/// callback consumes the click without any local world or inventory mutation.
#[must_use]
pub fn skull_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> Option<SkullUsePrediction> {
    if held.item_id != SKULL_ITEM_ID || side == Facing::Down {
        return None;
    }

    let target = if is_replaceable_in_place(world, hit_pos) {
        hit_pos
    } else {
        if !world.block_at(hit_pos).props().material.info().solid {
            return None;
        }
        hit_pos.relative(side)
    };

    if !can_player_edit(world, target, side, Some(held), allow_edit)
        || !world.block_at(target).props().material.is_replaceable()
    {
        return None;
    }

    Some(SkullUsePrediction { target, face: side })
}

/// `Items.ender_eye` (`Item.registerItems`).
pub const ENDER_EYE_ITEM_ID: i16 = 381;
/// `Items.fireworks` (`Item.registerItems`).
pub const FIREWORK_ITEM_ID: i16 = 401;
/// `Blocks.end_portal_frame`.
const END_PORTAL_FRAME: u16 = 120;

/// The client result of `ItemBanner.onItemUse`.
///
/// java: `ItemBanner.onItemUse` (`ItemBanner.java:33-57`) returns false for a DOWN click or a
/// clicked block whose material is not solid, offsets to the destination, then returns false
/// unless `EntityPlayer.canPlayerEdit` and `Blocks.standing_banner.canPlaceBlockAt` both accept
/// it; the remote-world branch returns true without placing anything. `ItemBanner` is an
/// `ItemBlock`, so `PlayerControllerMP`'s `canPlaceBlockOnSide` gate has already run (and sent
/// nothing on a refusal), but that gate does not test the clicked face or material: a click on
/// the underside of a block, or on a replaceable plant, reaches this callback and is refused here,
/// which lets `Minecraft.rightClickMouse` send the untargeted `C08` and skip the swing.
#[must_use]
pub fn banner_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> bool {
    if i32::from(held.item_id) != i32::from(BANNER_ITEM_ID) || side == Facing::Down {
        return false;
    }
    // java: `worldIn.getBlockState(pos).getBlock().getMaterial().isSolid()`.
    if !world.block_at(hit_pos).props().material.info().solid {
        return false;
    }
    let target = hit_pos.relative(side);
    can_player_edit(world, target, side, Some(held), allow_edit)
        && world.can_place_block_at(STANDING_BANNER, target)
}

/// The client result of `ItemEnderEye.onItemUse`.
///
/// java: `ItemEnderEye.onItemUse` (`ItemEnderEye.java:26-43`): true only when
/// `canPlayerEdit(pos.offset(side), side, stack)` passes and the clicked block is an end portal
/// frame without an eye (`BlockEndPortalFrame.EYE`, meta bit 4); the remote-world branch then
/// returns true before filling the frame. Every other click returns false, so the untargeted
/// `C08` follows and there is no swing.
#[must_use]
pub fn ender_eye_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> bool {
    if held.item_id != ENDER_EYE_ITEM_ID {
        return false;
    }
    let state = world.block_at(hit_pos);
    can_player_edit(world, hit_pos.relative(side), side, Some(held), allow_edit)
        && state.block().0 == END_PORTAL_FRAME
        && state.meta() & 4 == 0
}

/// The client-visible result of `ItemHoe.onItemUse`.
///
/// `target` and `state` are carried only so the caller can reproduce `useHoe`'s step sound.  On
/// a remote `World`, Java returns from `useHoe` immediately after `playSoundEffect`; it does not
/// set the block or damage the held stack.  The server's later block/durability update therefore
/// remains authoritative even for the successful `Consumed` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoeUseResult {
    /// The held item is not one of the five vanilla hoe registrations.
    NotApplicable,
    /// `ItemHoe.onItemUse` returned `false`, allowing the untargeted C08 fallback.
    Rejected,
    /// `useHoe` returned `true` after selecting this server-authoritative state and sound.
    Consumed {
        /// The originally clicked block, where `useHoe` centres its sound.
        target: BlockPos,
        /// The state passed to `useHoe`; this selects the step sound and is not written locally.
        state: BlockState,
    },
}

/// Reproduce `ItemHoe.onItemUse`'s remote-world-visible validation and state selection.
///
/// Java's order is significant: `canPlayerEdit(pos.offset(side), side, stack)` runs before the
/// side/air/block checks; then DOWN, a non-air block above, grass, and dirt variants are rejected
/// in that order.  The remote `useHoe` branch only plays the new state's step sound and returns
/// `true`, so this helper deliberately does not mutate `world`, decrement the stack, or consume
/// durability.  A caller that owns the click packets can use [`hoe_item_use_sound`] after
/// receiving `Consumed`.
///
/// java: `ItemHoe.onItemUse` (`ItemHoe.java:46-77`) and `useHoe` (`:79-95`).
#[must_use]
pub fn hoe_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> HoeUseResult {
    if !HOE_ITEM_IDS.contains(&held.item_id) {
        return HoeUseResult::NotApplicable;
    }

    // Java checks this before reading the clicked block, and the target is always one block along
    // the clicked face (including DOWN). `can_player_edit` maps that target back to `hit_pos` for
    // the adventure `CanPlaceOn` predicate.
    let target = hit_pos.relative(side);
    if !can_player_edit(world, target, side, Some(held), allow_edit) {
        return HoeUseResult::Rejected;
    }
    if side == Facing::Down
        || world.block_at(hit_pos.up()).props().material != mc_data::Material::Air
    {
        return HoeUseResult::Rejected;
    }

    let clicked = world.block_at(hit_pos);
    let state = match clicked.block() {
        GRASS_BLOCK_ID => BlockState::new(FARMLAND_BLOCK_ID, 0),
        DIRT_BLOCK_ID => match clicked.meta() {
            // BlockDirt.DirtType.byMetadata maps malformed metadata to DIRT. Metadata 2 is
            // PODZOL and is the only registered dirt variant that ItemHoe leaves untouched.
            2 => return HoeUseResult::Rejected,
            1 => BlockState::new(DIRT_BLOCK_ID, 0),
            _ => BlockState::new(FARMLAND_BLOCK_ID, 0),
        },
        _ => return HoeUseResult::Rejected,
    };

    HoeUseResult::Consumed {
        target: hit_pos,
        state,
    }
}

/// Build the `useHoe` step sound for a successful remote callback.
///
/// Java selects `newState.getBlock().stepSound`, centres the sound at the clicked block, and
/// applies `(volume + 1) / 2` and `frequency * .8` before returning `true`.  The result is kept
/// separate from [`HoeUseResult`] so no caller can mistake the prediction for a local block write.
/// The request goes through `World.playSoundEffect`, whose client sink is empty, so the driver
/// hands it to `Client::render_global_play_sound` and never plays it (the server's S29 does).
#[must_use]
pub fn hoe_item_use_sound(result: HoeUseResult) -> Option<mc_audio::PlayRequest> {
    let HoeUseResult::Consumed { target, state } = result else {
        return None;
    };
    Some(item_block_place_sound(BlockUsePlacement {
        pos: target,
        state,
    }))
}

/// A state write selected by a client-side `Item.onItemUse` body.
///
/// The server remains authoritative for the eventual block update.  This value carries only the
/// remote-world mutation that Java performs immediately on the client, so the driver can apply it
/// after queueing the targeted `C08`.  Keeping the write separate from the callback's boolean is
/// important: `ItemSlab` and `ItemSnow` return `true` even when their collision/set-state branch
/// fails, which still suppresses the untargeted `sendUseItem` fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockUsePlacement {
    /// The destination cell.
    pub pos: BlockPos,
    /// The exact state passed to the client's `World.setBlockState`.
    pub state: BlockState,
}

/// The part of an item callback that is observable by `Minecraft.rightClickMouse`.
///
/// `NotApplicable` means the item is not the special subclass being queried. `Delegate` means
/// the subclass reached `super.onItemUse` (the ordinary `ItemBlock` body remains the caller's
/// generic path) and carries the position passed to that superclass. `Rejected` is a callback
/// `false` before that delegation. `Consumed` is the callback `true`; its optional placement is
/// absent when Java's collision/set-state branch did not mutate the remote world but still claimed
/// the click.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockUseResult {
    /// The held item is not this helper's item subclass.
    NotApplicable,
    /// The subclass returned `false` before calling its superclass.
    Rejected,
    /// The subclass deferred to `ItemBlock.onItemUse` at this position.
    Delegate {
        /// The position passed to the superclass callback.  `ItemSnow` may already have stepped
        /// once, and `ItemBlock` may therefore perform its own second offset.
        pos: BlockPos,
    },
    /// The subclass returned `true`, optionally with a client-side state write.
    Consumed {
        /// The state write, if collision and `setBlockState` both succeeded.
        placement: Option<BlockUsePlacement>,
        /// Whether Java decremented the held stack in the successful state-write branch.
        decrement_stack: bool,
    },
    /// The callback returned `true` after selecting two ordered client-side state writes.
    ///
    /// `ItemDoor.placeDoor` and `BlockDoublePlant.onBlockPlacedBy` are the 1.8.9 callbacks that
    /// need this shape: each writes the lower half first and the upper half second, after the
    /// targeted C08 has been queued.
    ConsumedPair {
        /// Lower then upper, matching Java's two `setBlockState(..., 2)` calls.
        placements: [BlockUsePlacement; 2],
        /// Whether Java decremented the held stack after the lower write succeeded.  The upper
        /// write is allowed to fail at the inclusive world-height boundary without undoing that
        /// lower success.
        decrement_stack: bool,
    },
    /// The callback returned `true` after selecting an ordered set of client-side state writes.
    ///
    /// `BlockFire.onBlockAdded` can turn a newly placed fire into a whole portal.  Unlike the
    /// two-cell door/double-plant callbacks, the Java portal interior is variable-sized (up to
    /// 21x21), so a compact fixed-size result cannot represent it without inventing writes.
    ConsumedMany {
        /// Interior cells in Java's `BlockPortal.Size.func_150859_c` order: width first, then
        /// height, starting at the anchor's negative-axis edge.
        placements: Vec<BlockUsePlacement>,
        /// Whether Java decremented the held stack after the writes.  Flint-and-steel owns tool
        /// durability instead, so its portal result leaves this false.
        decrement_stack: bool,
    },
}

/// Classify `ItemLead.onItemUse` for the targeted block callback.
///
/// Java only claims a fence click.  The remote-world fence branch returns `true` without
/// creating the knot locally; a non-fence returns `false`, so the controller still sends its
/// untargeted `(-1,-1,-1)` C08 in `Minecraft.rightClickMouse` after the targeted placement.
#[must_use]
pub fn lead_item_use(world: &World, hit_pos: BlockPos, held: &ItemStack) -> BlockUseResult {
    if held.item_id != LEAD_ITEM_ID {
        return BlockUseResult::NotApplicable;
    }
    match world.block_at(hit_pos).block().0 {
        85 | 113 | 188..=192 => BlockUseResult::Consumed {
            placement: None,
            decrement_stack: false,
        },
        _ => BlockUseResult::Rejected,
    }
}

/// Reproduce `ItemSeeds.onItemUse`'s client-visible crop placement.
///
/// Java sends the targeted C08 before invoking this callback. A valid UP click writes the crop
/// into the air cell above the exact registered soil and decrements the stack; every rejected
/// branch returns `false`, allowing `Minecraft.rightClickMouse` to send its untargeted fallback.
#[must_use]
pub fn seeds_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> BlockUseResult {
    let (crop, soil) = match held.item_id {
        295 => (mc_data::BlockId(59), mc_data::BlockId(60)),
        361 => (mc_data::BlockId(104), mc_data::BlockId(60)),
        362 => (mc_data::BlockId(105), mc_data::BlockId(60)),
        372 => (mc_data::BlockId(115), mc_data::BlockId(88)),
        391 => (mc_data::BlockId(141), mc_data::BlockId(60)),
        392 => (mc_data::BlockId(142), mc_data::BlockId(60)),
        _ => return BlockUseResult::NotApplicable,
    };
    let target = hit_pos.up();
    if held.count == 0
        || side != Facing::Up
        || !can_player_edit(world, target, side, Some(held), allow_edit)
        || world.block_at(hit_pos).block() != soil
        || !world.block_at(target).props().is_air()
    {
        return BlockUseResult::Rejected;
    }

    BlockUseResult::Consumed {
        placement: Some(BlockUsePlacement {
            pos: target,
            state: BlockState::new(crop, 0),
        }),
        decrement_stack: true,
    }
}

/// Reproduce `ItemRedstone.onItemUse`'s client-visible wire placement.
///
/// Unlike an `ItemBlock`, redstone reaches this callback after the targeted C08 has already been
/// queued. Rejection therefore lets `Minecraft.rightClickMouse` append its untargeted fallback;
/// success writes wire metadata zero and decrements the survival stack locally.
#[must_use]
pub fn redstone_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
    blocking: &[Aabb],
) -> BlockUseResult {
    if held.item_id != 331 {
        return BlockUseResult::NotApplicable;
    }
    let target = if is_replaceable_in_place(world, hit_pos) {
        hit_pos
    } else {
        hit_pos.relative(side)
    };
    let wire = mc_data::BlockId(55);
    if held.count == 0
        || !can_player_edit(world, target, side, Some(held), allow_edit)
        || !can_block_be_placed(world, wire, target, side, blocking)
    {
        return BlockUseResult::Rejected;
    }
    BlockUseResult::Consumed {
        placement: Some(BlockUsePlacement {
            pos: target,
            state: BlockState::new(wire, 0),
        }),
        decrement_stack: true,
    }
}

/// Reproduce the remote half of `ItemRecord.onItemUse`.
///
/// An empty jukebox consumes the click, but insertion, aux effect, statistic, and decrement all
/// live exclusively in Java's server arm. Every other target returns false after the targeted
/// packet so the outer click path may send its untargeted fallback.
#[must_use]
pub fn record_item_use(world: &World, hit_pos: BlockPos, held: &ItemStack) -> BlockUseResult {
    if !(2256..=2267).contains(&held.item_id) {
        return BlockUseResult::NotApplicable;
    }
    let jukebox = world.block_at(hit_pos);
    if held.count == 0 || jukebox.block().0 != 84 || jukebox.meta() != 0 {
        return BlockUseResult::Rejected;
    }
    BlockUseResult::Consumed {
        placement: None,
        decrement_stack: false,
    }
}

/// Run the remote-visible half of `BlockRedstoneRepeater.onBlockActivated`.
///
/// The server owns lock scheduling and output propagation.  The client callback only cycles the
/// four delay values when editing is allowed; a refused edit still returns `false`, allowing the
/// ordinary ItemReed callback (and its packet fallback) to continue.
#[must_use]
pub fn activate_repeater(
    world: &mut World,
    pos: BlockPos,
    allow_edit: bool,
) -> Option<mc_world::world::LocalBlockActivation> {
    let state = world.block_at(pos);
    if !matches!(state.block().0, 93 | 94) {
        return None;
    }
    if !allow_edit {
        return Some(mc_world::world::LocalBlockActivation {
            consumed: false,
            effect: None,
        });
    }
    // `DELAY` is stored as `(delay - 1) << 2`; cycleProperty moves 1..4 then wraps to 1.
    let meta = (state.meta() & 3) | (((state.meta() >> 2).wrapping_add(1) & 3) << 2);
    world.set_block(pos, BlockState::new(state.block(), meta));
    Some(mc_world::world::LocalBlockActivation {
        consumed: true,
        effect: None,
    })
}

/// Run the remote-visible half of `BlockRedstoneComparator.onBlockActivated`.
///
/// Comparator mode is the metadata bit four (`MODE`), while the powered bit eight and facing
/// low bits survive the cycle.  The click sound and output recomputation are server-owned here;
/// only the state write and consumed boolean are predicted.
#[must_use]
pub fn activate_comparator(
    world: &mut World,
    pos: BlockPos,
    allow_edit: bool,
) -> Option<mc_world::world::LocalBlockActivation> {
    let state = world.block_at(pos);
    if !matches!(state.block().0, 149 | 150) {
        return None;
    }
    if !allow_edit {
        return Some(mc_world::world::LocalBlockActivation {
            consumed: false,
            effect: None,
        });
    }
    world.set_block(pos, BlockState::new(state.block(), state.meta() ^ 4));
    Some(mc_world::world::LocalBlockActivation {
        consumed: true,
        effect: None,
    })
}

/// Run the client half of `BlockJukebox.onBlockActivated`.
///
/// A recorded jukebox clears only its `HAS_RECORD` state on the remote world.  Record extraction,
/// the item entity, and the auxiliary effect all stay server-authoritative; an empty jukebox
/// returns `false` so a held record can reach `ItemRecord.onItemUse` after C08.
#[must_use]
pub fn activate_jukebox(
    world: &mut World,
    pos: BlockPos,
) -> Option<mc_world::world::LocalBlockActivation> {
    let state = world.block_at(pos);
    if state.block().0 != 84 {
        return None;
    }
    if state.meta() == 0 {
        return Some(mc_world::world::LocalBlockActivation {
            consumed: false,
            effect: None,
        });
    }
    world.set_block(pos, BlockState::new(state.block(), 0));
    Some(mc_world::world::LocalBlockActivation {
        consumed: true,
        effect: None,
    })
}

/// Run the remote half of `BlockFlowerPot.onBlockActivated`.
///
/// The flower-pot tile keeps the contained item and metadata, while the caller owns the selected
/// stack's creative/survival decrement.  `Some(false)` is an identified pot whose callback
/// declined (occupied pot, non-plant item, or missing tile); it must not be confused with an
/// unrelated block because the outer dispatcher still proceeds to the ordinary item callback.
#[must_use]
pub fn activate_flower_pot(
    world: &mut World,
    pos: BlockPos,
    held: Option<&ItemStack>,
) -> Option<mc_world::world::LocalBlockActivation> {
    if world.block_at(pos).block().0 != 140 {
        return None;
    }
    let Some(held) = held else {
        return Some(mc_world::world::LocalBlockActivation {
            consumed: false,
            effect: None,
        });
    };
    let Some(block) = placed_block(held) else {
        return Some(mc_world::world::LocalBlockActivation {
            consumed: false,
            effect: None,
        });
    };
    let allowed = matches!(block.0, 6 | 31 | 32 | 37 | 38 | 39 | 40 | 81)
        && (block.0 != 31 || held.damage as u8 == 2);
    let empty = world
        .tile_entity(pos)
        .and_then(|tile| tile.flower_pot())
        .is_some_and(|pot| pot.item_id == 0);
    if !allowed || !empty {
        return Some(mc_world::world::LocalBlockActivation {
            consumed: false,
            effect: None,
        });
    }
    if let Some(tile) = world.tile_entity_mut(pos) {
        if let mc_world::TileData::FlowerPot(pot) = &mut tile.data {
            pot.item_id = i32::from(block.0);
            pot.data = i32::from(held.damage as u8);
            // java: `TileEntityFlowerPot.markDirty(); worldIn.markBlockForUpdate(pos)`.
            // The block state is unchanged, so `World::set_block` cannot invalidate the
            // section for the model's tile-entity-backed `contents=` variant.  Mark the exact
            // vanilla one-block update box only after the activation has really succeeded;
            // declined/occupied/missing-tile callbacks must not manufacture render work.
            world.mark_dirty_box(pos);
            return Some(mc_world::world::LocalBlockActivation {
                consumed: true,
                effect: None,
            });
        }
    }
    Some(mc_world::world::LocalBlockActivation {
        consumed: false,
        effect: None,
    })
}

/// `BlockDragonEgg.onBlockActivated` always consumes a click.  The random destination and portal
/// particle fan-out are application/render work; this marker lets the driver run that body while
/// preserving the egg's local state (the remote Java branch never moves it).
#[must_use]
pub fn activate_dragon_egg(
    world: &World,
    pos: BlockPos,
) -> Option<mc_world::world::LocalBlockActivation> {
    (world.block_at(pos).block().0 == 122).then_some(mc_world::world::LocalBlockActivation {
        consumed: true,
        effect: None,
    })
}

/// FoodStats arithmetic shared by cake's block callback and ordinary `ItemFood` finish paths.
///
/// `BlockCake.eatCake` passes `(heal=2, saturationModifier=0.1)`, so saturation increases by
/// `2 * 0.1 * 2 = 0.4`, then both fields cap at their Java limits.
#[must_use]
pub fn cake_food_stats(food_level: i32, saturation: f32) -> FinishedFoodStats {
    let food_level = food_level.saturating_add(2).min(20);
    FinishedFoodStats {
        food_level,
        saturation: (saturation + 0.4).min(food_level as f32),
    }
}

/// Apply the client-visible Peaceful food refill from `EntityPlayer.onLivingUpdate`.
///
/// java: `EntityPlayer.java:606-613`.  This is deliberately separate from cake's
/// `FoodStats.addStats`: on a client `WorldClient` starts with the default
/// `naturalRegeneration=true`, and the local player still runs this branch every tenth
/// simulated tick in Peaceful.  The server's later `S06PacketUpdateHealth` remains the
/// authority for any disagreement.
#[must_use]
pub fn peaceful_food_regeneration(
    food_level: i32,
    peaceful: bool,
    natural_regeneration: bool,
    ticks_existed: u32,
) -> Option<i32> {
    (peaceful && natural_regeneration && food_level < 20 && ticks_existed.is_multiple_of(10))
        .then_some(food_level.saturating_add(1))
}

const OBSIDIAN_BLOCK_ID: mc_data::BlockId = mc_data::BlockId(49);
const FIRE_BLOCK_ID: mc_data::BlockId = mc_data::BlockId(51);
const PORTAL_BLOCK_ID: mc_data::BlockId = mc_data::BlockId(90);

/// Whether a cell is one of the three materials accepted by Java's `BlockPortal.Size` scan.
///
/// `func_150857_a` compares the block material, not the block id.  In the 1.8.9 registry the
/// only relevant non-air materials are fire and an already-existing portal; keeping the air
/// predicate on the decoded state also preserves the registry's malformed-state-to-air rule.
#[inline]
fn portal_interior_cell(state: BlockState) -> bool {
    state.props().is_air() || state.block() == FIRE_BLOCK_ID || state.block() == PORTAL_BLOCK_ID
}

/// `BlockPortal.Size.func_180120_a`, including its somewhat unusual endpoint convention.
///
/// The returned number is the count of interior cells before the obsidian endpoint.  Every
/// candidate interior cell must be air/fire/portal and must have obsidian immediately below it;
/// the first non-candidate cell must itself be obsidian.
fn portal_span(world: &World, start: BlockPos, direction: Facing) -> i32 {
    let mut index = 0;
    while index < 22 {
        let pos = start.offset(direction, index);
        if !portal_interior_cell(world.block_at(pos))
            || world.block_at(pos.down()).block() != OBSIDIAN_BLOCK_ID
        {
            break;
        }
        index += 1;
    }
    if world.block_at(start.offset(direction, index)).block() == OBSIDIAN_BLOCK_ID {
        index
    } else {
        0
    }
}

/// Find a Java-valid obsidian frame around a newly ignited cell and return the complete portal
/// interior write list.
///
/// This is the client-visible half of `BlockFire.onBlockAdded` ->
/// `BlockPortal.func_176548_d`.  Java tries the X portal axis first and then Z, walks down at most
/// 21 cells to the bottom interior row, requires a width of 2..=21 and a height of 3..=21, and
/// rejects a candidate containing an existing portal.  The returned state metadata is the portal
/// block's canonical axis metadata (`1` for X and `2` for Z).
fn portal_fill_for_ignition(world: &World, ignition: BlockPos) -> Option<Vec<BlockUsePlacement>> {
    for (positive, negative, metadata) in [
        (Facing::East, Facing::West, 1u8),
        (Facing::North, Facing::South, 2u8),
    ] {
        // java: `for (blockpos = p; p.getY() > blockpos.getY() - 21 && p.getY() > 0
        // && isReplaceable(p.down()); p = p.down())`.  The original Y is fixed by `blockpos`.
        let mut bottom = ignition;
        let original_y = ignition.y;
        while original_y - bottom.y < 21
            && bottom.y > 0
            && portal_interior_cell(world.block_at(bottom.down()))
        {
            bottom = bottom.down();
        }

        let span = portal_span(world, bottom, positive);
        if span < 1 {
            continue;
        }
        // java: `field_150861_f = p.offset(positive, i - 1)` and then the opposite-direction
        // scan determines the actual interior width.
        let anchor = bottom.offset(positive, span - 1);
        let width = portal_span(world, anchor, negative);
        if !(2..=21).contains(&width) {
            continue;
        }

        let mut existing_portals = 0;
        let mut height = 0;
        'height: for y in 0..21 {
            for lateral in 0..width {
                let pos = anchor.offset(negative, lateral).offset(Facing::Up, y);
                let state = world.block_at(pos);
                if !portal_interior_cell(state) {
                    break 'height;
                }
                if state.block() == PORTAL_BLOCK_ID {
                    existing_portals += 1;
                }

                // java: the left/right side frames are checked for every interior row, even
                // though the horizontal span check already validated the bottom row.
                let frame = if lateral == 0 {
                    pos.offset(positive, 1)
                } else if lateral == width - 1 {
                    pos.offset(negative, 1)
                } else {
                    continue;
                };
                if world.block_at(frame).block() != OBSIDIAN_BLOCK_ID {
                    break 'height;
                }
            }
            height += 1;
        }

        // The first non-interior row is the top frame.  Java checks every top cell and then
        // applies the inclusive 3..=21 height bounds.
        if !(3..=21).contains(&height)
            || (0..width).any(|lateral| {
                world
                    .block_at(anchor.offset(negative, lateral).offset(Facing::Up, height))
                    .block()
                    != OBSIDIAN_BLOCK_ID
            })
            || existing_portals != 0
        {
            continue;
        }

        let mut placements = Vec::with_capacity((width * height) as usize);
        for lateral in 0..width {
            let column = anchor.offset(negative, lateral);
            for y in 0..height {
                placements.push(BlockUsePlacement {
                    pos: column.offset(Facing::Up, y),
                    state: BlockState::new(PORTAL_BLOCK_ID, metadata),
                });
            }
        }
        debug_assert_eq!(placements.len(), (width * height) as usize);
        return Some(placements);
    }
    None
}

/// Reproduce `ItemFlintAndSteel.onItemUse` after the targeted C08.
///
/// Permission failure returns false. Every editable target consumes and damages the tool. An air
/// target normally receives a local fire state; when `BlockFire.onBlockAdded` recognizes a valid
/// obsidian frame, the fire is immediately replaced by the complete portal interior instead.
#[must_use]
pub fn flint_and_steel_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> BlockUseResult {
    if held.item_id != 259 {
        return BlockUseResult::NotApplicable;
    }
    let target = hit_pos.relative(side);
    if held.count == 0 || !can_player_edit(world, target, side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }
    if world.block_at(target).props().is_air() {
        if let Some(placements) = portal_fill_for_ignition(world, target) {
            return BlockUseResult::ConsumedMany {
                placements,
                decrement_stack: false,
            };
        }
    }
    let placement = world
        .block_at(target)
        .props()
        .is_air()
        .then_some(BlockUsePlacement {
            pos: target,
            state: BlockState::new(FIRE_BLOCK_ID, 0),
        });
    BlockUseResult::Consumed {
        placement,
        decrement_stack: false,
    }
}

fn hanging_position(
    anchor: BlockPos,
    facing: Facing,
    width_pixels: i32,
    height_pixels: i32,
) -> DVec3 {
    let centered = |pixels: i32| if pixels % 32 == 0 { 0.5 } else { 0.0 };
    let (fx, _, fz) = facing.offset();
    let (sx, _, sz) = facing.rotate_y_ccw().offset();
    let lateral = centered(width_pixels);
    DVec3::new(
        f64::from(anchor.x) + 0.5 - f64::from(fx) * 0.468_75 + lateral * f64::from(sx),
        f64::from(anchor.y) + 0.5 + centered(height_pixels),
        f64::from(anchor.z) + 0.5 - f64::from(fz) * 0.468_75 + lateral * f64::from(sz),
    )
}

fn hanging_surface_valid(
    world: &World,
    anchor: BlockPos,
    facing: Facing,
    width_pixels: i32,
    height_pixels: i32,
    hanging_entities: &[Aabb],
) -> bool {
    let position = hanging_position(anchor, facing, width_pixels, height_pixels);
    let bounds = hanging_bounding_box(position, facing, width_pixels, height_pixels);
    let mut collisions = Vec::new();
    mc_physics::collision::collision_boxes_in(world, &bounds, &mut collisions);
    if !collisions.is_empty() {
        return false;
    }

    let support = anchor.relative(facing.opposite());
    let across = facing.rotate_y_ccw();
    let (ax, _, az) = across.offset();
    for horizontal in 0..(width_pixels / 16).max(1) {
        for vertical in 0..(height_pixels / 16).max(1) {
            let pos = BlockPos::new(
                support.x + ax * horizontal,
                support.y + vertical,
                support.z + az * horizontal,
            );
            let state = world.block_at(pos);
            if !state.props().material.info().solid
                && !matches!(state.block().0, 93 | 94 | 149 | 150)
            {
                return false;
            }
        }
    }

    !hanging_entities
        .iter()
        .any(|existing| existing.intersects_with(&bounds))
}

/// Reproduce `ItemHangingEntity.onItemUse` after the targeted `C08`.
///
/// The remote client constructs a prospective painting or item frame solely to run
/// `EntityHanging.onValidSurface`. It never spawns that entity locally, but a valid surface still
/// decrements the survival stack. Invalid support/collision is a consumed callback with no
/// decrement; vertical faces and edit-permission failures return false.
#[must_use]
pub fn hanging_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
    hanging_entities: &[Aabb],
) -> BlockUseResult {
    if !matches!(held.item_id, 321 | 389) {
        return BlockUseResult::NotApplicable;
    }
    if held.count == 0 || !side.axis().is_horizontal() {
        return BlockUseResult::Rejected;
    }
    let anchor = hit_pos.relative(side);
    if !can_player_edit(world, anchor, side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }

    let valid = if held.item_id == 389 {
        hanging_surface_valid(world, anchor, side, 12, 12, hanging_entities)
    } else {
        mc_render::entity::objects::ART.iter().any(|art| {
            hanging_surface_valid(
                world,
                anchor,
                side,
                art.size_x,
                art.size_y,
                hanging_entities,
            )
        })
    };
    BlockUseResult::Consumed {
        placement: None,
        decrement_stack: valid,
    }
}

/// Java `IGrowable.canGrow(..., isClient=true)` for the eight vanilla implementations.
fn bonemeal_can_grow(world: &World, pos: BlockPos) -> bool {
    let state = world.block_at(pos);
    match state.block().0 {
        2 | 6 | 39 | 40 => true, // grass, sapling, mushrooms
        31 => state.meta() != 0, // tall grass rejects its DEAD_BUSH variant
        59 | 104 | 105 | 141 | 142 => state.meta() < 7,
        127 => state.meta() >> 2 < 2,
        175 => {
            let lower = if state.meta() & 8 != 0 {
                world.block_at(pos.down())
            } else {
                state
            };
            lower.block().0 == 175 && !matches!(lower.meta() & 7, 2 | 3)
        }
        _ => false,
    }
}

/// Reproduce the remote-visible half of `ItemDye.onItemUse`.
///
/// White dye validates `IGrowable.canGrow` but leaves growth, particles, and decrement to the
/// server. Brown dye predicts age-zero cocoa on a horizontal face of a jungle `Blocks.log` and
/// decrements survival; an occupied target is still consumed without mutation.
#[must_use]
pub fn dye_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
) -> BlockUseResult {
    if held.item_id != 351 {
        return BlockUseResult::NotApplicable;
    }
    if held.count == 0
        || !can_player_edit(world, hit_pos.relative(side), side, Some(held), allow_edit)
    {
        return BlockUseResult::Rejected;
    }
    match held.damage & 15 {
        15 if bonemeal_can_grow(world, hit_pos) => BlockUseResult::Consumed {
            placement: None,
            decrement_stack: false,
        },
        3 if world.block_at(hit_pos).block().0 == 17
            && world.block_at(hit_pos).meta() & 3 == 3
            && side.axis().is_horizontal() =>
        {
            let target = hit_pos.relative(side);
            if !world.block_at(target).props().is_air() {
                return BlockUseResult::Consumed {
                    placement: None,
                    decrement_stack: false,
                };
            }
            let meta = side
                .opposite()
                .horizontal_index()
                .expect("a horizontal cocoa support face has a horizontal opposite");
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: target,
                    state: BlockState::new(mc_data::BlockId(127), meta),
                }),
                decrement_stack: true,
            }
        }
        _ => BlockUseResult::Rejected,
    }
}

/// Reproduce `ItemMinecart.onItemUse`'s client-visible rail gate and stack decrement.
///
/// The remote world never predicts the minecart entity: Java only spawns it in the
/// `!worldIn.isRemote` arm. It does, however, decrement the client stack and return true on any
/// rail. A non-rail click returns false after the targeted C08, allowing `rightClickMouse` to add
/// its untargeted sentinel.
#[must_use]
pub fn minecart_item_use(world: &World, hit_pos: BlockPos, held: &ItemStack) -> BlockUseResult {
    if !matches!(held.item_id, 328 | 342 | 343 | 407 | 408 | 422) {
        return BlockUseResult::NotApplicable;
    }
    if held.count == 0 || !matches!(world.block_at(hit_pos).block().0, 27 | 28 | 66 | 157) {
        return BlockUseResult::Rejected;
    }
    BlockUseResult::Consumed {
        placement: None,
        decrement_stack: true,
    }
}

/// Reproduce the `ItemPiston` -> `ItemBlock` -> `BlockPistonBase.onBlockPlaced` chain.
///
/// `ItemPiston.getMetadata` returns the sentinel metadata 7 for both piston items, but
/// `BlockPistonBase.onBlockPlaced` deliberately replaces it with a facing derived from the
/// placer's feet/eyes/yaw and clears EXTENDED. The caller applies the returned state after the
/// targeted C08 and owns creative restoration exactly like the other ItemBlock callbacks.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn piston_item_use(
    world: &World,
    pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
    blocking: &[Aabb],
    placer_pos: DVec3,
    placer_eye_y: f64,
    yaw: f64,
) -> BlockUseResult {
    let placing = match held.item_id {
        29 => mc_data::BlockId(29),
        33 => mc_data::BlockId(33),
        _ => return BlockUseResult::NotApplicable,
    };
    let target = if is_replaceable_in_place(world, pos) {
        pos
    } else {
        pos.relative(side)
    };
    if held.count == 0 || !can_player_edit(world, target, side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }
    if !can_block_be_placed(world, placing, target, side, blocking) {
        return BlockUseResult::Rejected;
    }

    // java: ItemPiston.getMetadata returns 7, then BlockPistonBase.onBlockPlaced ignores that
    // value and creates `FACING=getFacingFromEntity, EXTENDED=false`. Keep the static helper shared
    // with BlockDispenser so the float-coordinate and yaw rounding controls cannot drift apart.
    let _item_metadata = 7_u8;
    let facing = facing_from_entity(target, placer_pos, placer_eye_y, yaw);
    BlockUseResult::Consumed {
        placement: Some(BlockUsePlacement {
            pos: target,
            state: BlockState::new(placing, facing.index()),
        }),
        decrement_stack: true,
    }
}

/// Reproduce `ItemAnvilBlock.getMetadata` and `BlockAnvil.onBlockPlaced`.
///
/// The item damage becomes the anvil damage property in metadata bits 2-3. The low two bits are
/// replaced by the placer's horizontal facing rotated clockwise. The caller owns the targeted
/// C08 and applies this remote-world write afterwards, preserving server correction authority.
#[must_use]
pub fn anvil_item_use(
    world: &World,
    pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
    blocking: &[Aabb],
    yaw: f64,
) -> BlockUseResult {
    if held.item_id != 145 {
        return BlockUseResult::NotApplicable;
    }
    let placing = mc_data::BlockId(145);
    let target = if is_replaceable_in_place(world, pos) {
        pos
    } else {
        pos.relative(side)
    };
    if held.count == 0 || !can_player_edit(world, target, side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }
    if !can_block_be_placed(world, placing, target, side, blocking) {
        return BlockUseResult::Rejected;
    }

    let facing = java_horizontal_facing(yaw).rotate_y();
    let facing_meta = facing
        .horizontal_index()
        .expect("rotating a horizontal yaw remains horizontal");
    let meta = ((held.damage as u8) << 2) | facing_meta;
    BlockUseResult::Consumed {
        placement: Some(BlockUsePlacement {
            pos: target,
            state: BlockState::new(placing, meta),
        }),
        decrement_stack: true,
    }
}

/// Resolve the remote-world half of `ItemArmorStand.onItemUse`.
///
/// Vanilla performs every validation and decrements the client stack, but creates the stand only
/// inside `!world.isRemote`. Consequently a valid result has no local entity or block placement;
/// the caller writes only the survival inventory prediction after the targeted C08.
#[must_use]
pub fn armor_stand_item_use(
    world: &World,
    pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
    entities: &[Aabb],
) -> BlockUseResult {
    if held.item_id != 416 {
        return BlockUseResult::NotApplicable;
    }
    if held.count == 0 || side == Facing::Down {
        return BlockUseResult::Rejected;
    }

    let target = if is_replaceable_in_place(world, pos) {
        pos
    } else {
        pos.relative(side)
    };
    if !can_player_edit(world, target, side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }
    let blocks_placement = |check: BlockPos| {
        let state = world.block_at(check);
        state.block().0 != 0 && !is_replaceable_in_place(world, check)
    };
    if blocks_placement(target) || blocks_placement(target.up()) {
        return BlockUseResult::Rejected;
    }

    let occupied = Aabb::from_bounds(
        f64::from(target.x),
        f64::from(target.y),
        f64::from(target.z),
        f64::from(target.x) + 1.0,
        f64::from(target.y) + 2.0,
        f64::from(target.z) + 1.0,
    );
    if entities
        .iter()
        .any(|bounds| bounds.intersects_with(&occupied))
    {
        return BlockUseResult::Rejected;
    }

    BlockUseResult::Consumed {
        placement: None,
        decrement_stack: true,
    }
}

/// Resolve the seven independently-registered 1.8.9 `ItemDoor` items to their blocks.
///
/// These cannot use `ItemProps.block`: `ItemDoor` is not an `ItemBlock`, and Java registers the
/// item and block under different numeric ids.
#[must_use]
fn door_item_block(held: &ItemStack) -> Option<mc_data::BlockId> {
    Some(mc_data::BlockId(match held.item_id {
        324 => 64,
        330 => 71,
        427 => 193,
        428 => 194,
        429 => 195,
        430 => 196,
        431 => 197,
        _ => return None,
    }))
}

/// Reproduce `ItemDoor.onItemUse` and `ItemDoor.placeDoor`'s client-visible prediction.
///
/// Java runs this body on both sides. The caller owns the targeted C08 and applies the returned
/// lower/upper writes afterwards; a rejected callback therefore still leaves the targeted packet
/// queued and permits `Minecraft.rightClickMouse` to take its untargeted-use fallback.
#[must_use]
pub fn door_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    yaw: f64,
    held: &ItemStack,
    allow_edit: bool,
) -> BlockUseResult {
    let Some(door) = door_item_block(held) else {
        return BlockUseResult::NotApplicable;
    };
    if side != Facing::Up {
        return BlockUseResult::Rejected;
    }

    let target = if is_replaceable_in_place(world, hit_pos) {
        hit_pos
    } else {
        hit_pos.up()
    };
    if !can_player_edit(world, target, side, Some(held), allow_edit)
        || !world.can_place_block_at(door, target)
    {
        return BlockUseResult::Rejected;
    }

    // java: `placeDoor(..., EnumFacing.fromAngle((double)playerIn.rotationYaw), ...)` - the
    // double `yaw / 90` form, not `getHorizontalFacing`'s float `yaw * 4 / 360`.
    let facing = Facing::from_angle(yaw);
    let left = target.relative(facing.rotate_y_ccw());
    let right = target.relative(facing.rotate_y());
    let left_solids =
        u8::from(world.is_normal_cube_at(left)) + u8::from(world.is_normal_cube_at(left.up()));
    let right_solids =
        u8::from(world.is_normal_cube_at(right)) + u8::from(world.is_normal_cube_at(right.up()));
    // java: `getBlock() == door` - the same door block, so an oak door does not mirror the
    // hinge of a spruce door beside it.
    let is_door = |pos| world.block_at(pos).block() == door;
    let door_on_left = is_door(left) || is_door(left.up());
    let door_on_right = is_door(right) || is_door(right.up());
    let hinge_right = (door_on_left && !door_on_right) || right_solids > left_solids;

    let lower_meta = facing
        .rotate_y()
        .horizontal_index()
        .expect("a yaw always selects a horizontal door facing");
    let upper_meta = 8 | u8::from(hinge_right);
    BlockUseResult::ConsumedPair {
        placements: [
            BlockUsePlacement {
                pos: target,
                state: BlockState::new(door, lower_meta),
            },
            BlockUsePlacement {
                pos: target.up(),
                state: BlockState::new(door, upper_meta),
            },
        ],
        decrement_stack: true,
    }
}

/// `ItemSlab.onItemUse`'s client-visible double-slab branches.
///
/// java: `ItemSlab.java:47-83`.  The callback first checks `canPlayerEdit(pos.offset(side),
/// side, stack)`, then tries to merge the matching half slab at the clicked cell and at the cell
/// one step along the clicked face.  Both merge helpers return `true` even if collision checking
/// or `setBlockState` fails; only a successful state write decrements the stack.  A non-merge
/// falls through to `ItemBlock.onItemUse`, represented by [`BlockUseResult::Delegate`].
#[must_use]
pub fn slab_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
    blocking: &[Aabb],
) -> BlockUseResult {
    let Some(single) = slab_block(held) else {
        return BlockUseResult::NotApplicable;
    };
    if held.count == 0 {
        return BlockUseResult::Rejected;
    }

    // Java's ItemSlab call uses `pos.offset(side)` as the canPlayerEdit argument, while the
    // first merge lookup still uses the original hit position.  `can_player_edit` performs the
    // same opposite-face read internally, so passing this destination preserves that split.
    if !can_player_edit(world, hit_pos.relative(side), side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }

    let variant = held.damage as u8 & 7;
    let existing = world.block_at(hit_pos);
    if existing.block() == single
        && existing.meta() & 7 == variant
        && ((side == Facing::Up && existing.meta() & 8 == 0)
            || (side == Facing::Down && existing.meta() & 8 != 0))
    {
        return slab_merge_result(single, hit_pos, variant, blocking);
    }

    let adjacent = hit_pos.relative(side);
    let adjacent_state = world.block_at(adjacent);
    if adjacent_state.block() == single && adjacent_state.meta() & 7 == variant {
        return slab_merge_result(single, adjacent, variant, blocking);
    }

    // Java calls `super.onItemUse` with the original hit position here, not the adjacent cell.
    // The generic ItemBlock callback is intentionally left to the caller, which already owns
    // the common placement gate and metadata/shape dispatch.
    BlockUseResult::Delegate { pos: hit_pos }
}

fn slab_merge_result(
    single: mc_data::BlockId,
    pos: BlockPos,
    variant: u8,
    blocking: &[Aabb],
) -> BlockUseResult {
    let double = match single.0 {
        44 => mc_data::BlockId(43),
        126 => mc_data::BlockId(125),
        182 => mc_data::BlockId(181),
        _ => return BlockUseResult::NotApplicable,
    };
    let state = BlockState::new(double, variant);
    let placement = (!state_collision_intersects(state, pos, blocking))
        .then_some(BlockUsePlacement { pos, state });
    BlockUseResult::Consumed {
        decrement_stack: placement.is_some(),
        placement,
    }
}

/// `ItemSnow.onItemUse`'s layered-snow client branch.
///
/// java: `ItemSnow.java:24-67`.  The permission check intentionally uses the original hit cell,
/// then the callback keeps a snow layer in place only for an UP click; every other non-replaceable
/// hit is offset before it checks for a layer.  A layer below eight returns `true` and mutates the
/// remote world when its new collision box is clear.  An eighth layer or a blocked increment
/// reaches `ItemBlock` and is reported as [`BlockUseResult::Delegate`] so the caller can run that
/// ordinary path.
#[must_use]
pub fn snow_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    held: &ItemStack,
    allow_edit: bool,
    blocking: &[Aabb],
) -> BlockUseResult {
    // `Items.snow` is the ItemSnow/BlockSnow registration at id 78; snowballs are 332 and must
    // not accidentally enter this block callback.
    if held.item_id != 78 {
        return BlockUseResult::NotApplicable;
    }
    if held.count == 0 || !can_player_edit(world, hit_pos, side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }

    let hit_state = world.block_at(hit_pos);
    let hit_block = hit_state.block();
    let target =
        if (side != Facing::Up || hit_block.0 != 78) && !is_replaceable_in_place(world, hit_pos) {
            hit_pos.relative(side)
        } else {
            hit_pos
        };
    let target_state = world.block_at(target);
    if target_state.block().0 != 78 {
        return BlockUseResult::Delegate { pos: target };
    }

    // BlockSnow stores layers 1..=8 as metadata 0..=7.  ItemSnow's `i <= 7` therefore accepts
    // old metadata 0..=6 and writes exactly one higher metadata value.
    if target_state.meta() >= 7 {
        return BlockUseResult::Delegate { pos: target };
    }
    let state = BlockState::new(mc_data::BlockId(78), target_state.meta() + 1);
    if state_collision_intersects(state, target, blocking) {
        return BlockUseResult::Delegate { pos: target };
    }
    BlockUseResult::Consumed {
        decrement_stack: true,
        placement: Some(BlockUsePlacement { pos: target, state }),
    }
}

/// Run the ordinary `ItemBlock.onItemUse` body after a special block item has declined its own
/// branch.
///
/// `ItemSnow` passes its already-offset `blockpos` to `super.onItemUse`; the superclass then
/// applies its own `!isReplaceable` offset a second time.  Keeping that position explicit is
/// what preserves the otherwise very easy to lose snow double-offset.  This narrow bridge is
/// used by the live controller for `ItemSlab`/`ItemSnow` delegates, ItemReed subclasses, and the
/// ordinary ItemBlock family. The per-block `onBlockPlaced` hook table below derives every
/// metadata-bearing orientation that can differ from raw item damage.
///
/// java: `ItemBlock.onItemUse` (`ItemBlock.java:39-78`).
// This bridge carries the exact Java callback inputs so metadata and collision checks stay in
// their original order instead of being hidden behind a reordered context object.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn item_block_delegate_use(
    world: &World,
    pos: BlockPos,
    side: Facing,
    hit_y: f64,
    yaw: f64,
    held: &ItemStack,
    allow_edit: bool,
    blocking: &[Aabb],
) -> BlockUseResult {
    // ItemBanner is an ItemBlock subclass only in the Java type hierarchy. Its own
    // `onItemUse` validates a banner destination and then returns from the client branch
    // without a local block write or decrement. Keep it out of this generic ItemBlock bridge so
    // a future caller cannot accidentally turn its `BlockId(176)` association into a prediction.
    if held.item_id == i16::try_from(BANNER_ITEM_ID).expect("banner item id fits") {
        return BlockUseResult::NotApplicable;
    }
    let reed_placing = item_reed_block(held);
    let Some(placing) = placed_block(held).or(reed_placing) else {
        return BlockUseResult::NotApplicable;
    };

    // java: `if (!block.isReplaceable(worldIn, pos)) pos = pos.offset(side);`.
    // ItemReed has the narrower snow-layer special case and forces its callback side to UP.
    let shallow_snow = reed_placing.is_some()
        && world.block_at(pos).block().0 == 78
        && world.block_at(pos).meta() < 1;
    let effective_side = if shallow_snow { Facing::Up } else { side };
    let target = if shallow_snow || is_replaceable_in_place(world, pos) {
        pos
    } else {
        pos.relative(effective_side)
    };
    if held.count == 0 || !can_player_edit(world, target, effective_side, Some(held), allow_edit) {
        return BlockUseResult::Rejected;
    }
    if !can_block_be_placed(world, placing, target, effective_side, blocking) {
        return BlockUseResult::Rejected;
    }

    // `ItemSlab` starts as BOTTOM and flips to TOP for a downward click or an upper-half
    // horizontal hit. `BlockStairs.onBlockPlaced` derives facing from the player's horizontal
    // yaw and uses the same face/hit-height half rule. `BlockLog.onBlockPlaced` preserves the low
    // species bits and derives its pillar axis from the clicked face. Snow has no orientation.
    let meta = if reed_placing.is_some() && matches!(placing.0, 93 | 149) {
        java_horizontal_facing(yaw)
            .opposite()
            .horizontal_index()
            .expect("a yaw always selects a horizontal diode facing")
    } else if reed_placing.is_some() {
        0
    } else if matches!(placing.0, 44 | 126 | 182) {
        let mut meta = held.damage as u8 & 7;
        if side == Facing::Down || (side != Facing::Up && hit_y > 0.5) {
            meta |= 8;
        }
        meta
    } else if is_stair_block(placing) {
        // java: BlockStairs#getMetaFromState writes `5 - facing.getIndex()` into bits 0..1.
        let mut meta = 5 - java_horizontal_facing(yaw).index();
        if side == Facing::Down || (side != Facing::Up && hit_y > 0.5) {
            meta |= 4;
        }
        meta
    } else if matches!(placing.0, 17 | 162) {
        let axis = match side {
            Facing::Up | Facing::Down => 0,
            Facing::East | Facing::West => 4,
            Facing::North | Facing::South => 8,
        };
        (held.damage as u8 & 3) | axis
    } else if matches!(placing.0, 18 | 161) {
        // `ItemLeaves#getMetadata` is the one ordinary ItemBlock subtype whose callback does not
        // pass raw damage through: it ORs bit two so a newly placed leaf block starts persistent
        // (the decay/check bits are cleared by the block's own state codec).  Java's value is
        // otherwise the unmodified item damage, so do not mask the low species bits here.
        (held.damage as u8) | 4
    } else {
        item_block_placed_metadata(world, target, effective_side, hit_y, yaw, held, placing)
    };
    let lower = BlockUsePlacement {
        pos: target,
        // `Block.onBlockPlaced` receives the ItemStack metadata and decodes it.  The common
        // ItemBlock path preserves that metadata; specialized Java callbacks above handle
        // the families whose metadata is derived from pose or hit data.  Double plants are the
        // notable virtual decoder here: `EnumPlantType.byMetadata` maps item values 6/7 back to
        // SUNFLOWER rather than leaving an illegal lower-half storage state behind.
        state: if placing.0 == 175 {
            BlockState::from_java_state_id(i32::from(placing.0) | (i32::from(meta) << 12))
        } else {
            BlockState::new(placing, meta)
        },
    };
    if placing.0 == 175 {
        // java: `BlockDoublePlant.onBlockPlacedBy` writes the default upper state.  The default
        // facing is NORTH, whose horizontal index is 2, so the serialized upper metadata is 10;
        // the lower item variant is the virtual-decoded stack damage in `meta`.
        return BlockUseResult::ConsumedPair {
            placements: [
                lower,
                BlockUsePlacement {
                    pos: target.up(),
                    state: BlockState::new(
                        placing,
                        8 | Facing::North
                            .horizontal_index()
                            .expect("north is horizontal"),
                    ),
                },
            ],
            // `World.setBlockState` accepts y=0..=255 only.  BlockDoublePlant's placement gate
            // deliberately checks the upper cell through `getBlockState` (which reads air at
            // y=256), so a target at y=256 reaches ItemBlock but its lower write fails and the
            // stack/sound must remain untouched.  At y=255 the lower write succeeds; the upper
            // y=256 write is simply the second no-op and the stack is still consumed.
            decrement_stack: (0..256).contains(&target.y),
        };
    }
    BlockUseResult::Consumed {
        placement: Some(lower),
        decrement_stack: true,
    }
}

/// Run [`item_block_delegate_use`] with the live player pose available to
/// `BlockPistonBase.getFacingFromEntity`/`BlockDispenser.onBlockPlaced`.
///
/// The compatibility entry point above predates the pose snapshot and remains useful to small
/// callers that only have yaw.  The actual right-click controller has the feet and eye positions,
/// so it uses this bridge: all ordinary `ItemBlock` validation stays byte-for-byte shared while
/// the two pose-derived block families receive Java's steep-pitch UP/DOWN result.  A server
/// `S22`/`S23` correction still wins over this remote prediction in the normal packet path.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn item_block_delegate_use_with_pose(
    world: &World,
    pos: BlockPos,
    side: Facing,
    hit_y: f64,
    yaw: f64,
    held: &ItemStack,
    allow_edit: bool,
    blocking: &[Aabb],
    placer_pos: DVec3,
    placer_eye_y: f64,
) -> BlockUseResult {
    let result = item_block_delegate_use(world, pos, side, hit_y, yaw, held, allow_edit, blocking);
    let replace = |placement: BlockUsePlacement| {
        if matches!(placement.state.block().0, 23 | 158) {
            let facing = facing_from_entity(placement.pos, placer_pos, placer_eye_y, yaw);
            BlockUsePlacement {
                state: BlockState::new(placement.state.block(), facing.index()),
                ..placement
            }
        } else {
            placement
        }
    };
    match result {
        BlockUseResult::Consumed {
            placement: Some(placement),
            decrement_stack,
        } => BlockUseResult::Consumed {
            placement: Some(replace(placement)),
            decrement_stack,
        },
        other => other,
    }
}

/// Java's `BlockPistonBase.getFacingFromEntity` (also used by dispensers/droppers).
///
/// The float casts are intentional: 1.8.9 narrows both horizontal coordinates before its
/// two-block proximity comparison, while the vertical branch uses the entity's double eye Y.
#[must_use]
pub fn facing_from_entity(
    clicked: BlockPos,
    placer_pos: DVec3,
    placer_eye_y: f64,
    yaw: f64,
) -> Facing {
    let close_x = ((placer_pos.x as f32) - (clicked.x as f32)).abs() < 2.0;
    let close_z = ((placer_pos.z as f32) - (clicked.z as f32)).abs() < 2.0;
    if close_x && close_z && placer_eye_y - f64::from(clicked.y) > 2.0 {
        Facing::Up
    } else if close_x && close_z && f64::from(clicked.y) - placer_eye_y > 0.0 {
        Facing::Down
    } else {
        java_horizontal_facing(yaw).opposite()
    }
}

/// Java 1.8.9's `EntityLivingBase.getHorizontalFacing` calculation.
///
/// `rotationYaw` is a Java `float`, and the multiply/divide are also `float` operations before
/// `MathHelper.floor_double` receives the promoted value.  Taking the `f32` round here matters at
/// quadrant boundaries for callers that provide a wider `f64` pose snapshot.
#[must_use]
fn java_horizontal_facing(yaw: f64) -> Facing {
    let quarter_turns = (yaw as f32) * 4.0_f32 / 360.0_f32;
    Facing::from_horizontal((f64::from(quarter_turns) + 0.5).floor() as i32 & 3)
}

/// java: the iteration order of `EnumFacing.Plane.HORIZONTAL` (`EnumFacing.java:454`):
/// NORTH, EAST, SOUTH, WEST.
///
/// This is **not** [`Facing::HORIZONTALS`], which is `EnumFacing.HORIZONTALS` indexed by
/// `getHorizontalIndex` (SOUTH, WEST, NORTH, EAST). The `for (EnumFacing f :
/// EnumFacing.Plane.HORIZONTAL)` fallbacks in `BlockLadder.onBlockPlaced` and
/// `BlockTorch.onBlockPlaced` return the *first* supported face, so in a corner the order is
/// the placed state: a ladder on the wrong wall collides on the wrong side of its cell.
const PLANE_HORIZONTAL: [Facing; 4] = [Facing::North, Facing::East, Facing::South, Facing::West];

/// `Blocks.ladder`.
const LADDER_BLOCK_ID: u16 = 65;

/// The `FACING` that `BlockLadder.onBlockPlaced` (`BlockLadder.java:91-108`) gives a ladder
/// placed at `target` from a click on `side`.
///
/// ```text
/// if (facing.getAxis().isHorizontal() && canBlockStay(world, pos, facing)) return FACING=facing;
/// for (EnumFacing f : EnumFacing.Plane.HORIZONTAL) if (canBlockStay(world, pos, f)) return f;
/// return getDefaultState();   // FACING=NORTH
/// ```
///
/// `canBlockStay` is `getBlockState(pos.offset(facing.getOpposite())).getBlock().isNormalCube()`.
fn ladder_placed_facing(world: &World, target: BlockPos, side: Facing) -> Facing {
    if side.axis().is_horizontal() && world.is_normal_cube_at(target.relative(side.opposite())) {
        return side;
    }
    PLANE_HORIZONTAL
        .into_iter()
        .find(|&f| world.is_normal_cube_at(target.relative(f.opposite())))
        .unwrap_or(Facing::North)
}

/// Resolve the metadata produced by `Block.onBlockPlaced` for the ordinary `ItemBlock` family.
///
/// The item callback is client-executed: Java computes this state before `WorldClient.setBlockState`
/// and then runs the block's `onBlockPlacedBy` hook (also client-visible for the orientation
/// rewrites).  Keep this as a total function over the registry so a newly covered ItemBlock does
/// not silently fall back to raw item damage.  The common/default arm is exactly
/// `Block.onBlockPlaced`, which feeds the item metadata through the block's state codec.
///
/// java: `Block.onBlockPlaced` and the 1.8.9 overrides in `Block*.java`.
#[must_use]
fn item_block_placed_metadata(
    world: &World,
    target: BlockPos,
    side: Facing,
    hit_y: f64,
    yaw: f64,
    held: &ItemStack,
    placing: mc_data::BlockId,
) -> u8 {
    let horizontal = java_horizontal_facing(yaw);
    let horizontal_opposite = horizontal.opposite();

    match placing.0 {
        // BlockDispenser.onBlockPlaced and BlockPistonBase use the same pose-derived helper.
        // Piston items are dispatched through piston_item_use, but keeping the class here makes
        // this helper safe for callers that already have the final destination state.
        23 | 158 => horizontal_opposite.index(),
        // Button metadata is not `EnumFacing.index()`: its state codec uses DOWN=0, EAST=1,
        // WEST=2, SOUTH=3, NORTH=4, UP=5.  `canPlaceBlockOnSide` already selected a supported
        // face, so the onBlockPlaced callback retains that face.
        77 | 143 => match side {
            Facing::Down => 0,
            Facing::East => 1,
            Facing::West => 2,
            Facing::South => 3,
            Facing::North => 4,
            Facing::Up => 5,
        },
        // Command blocks and huge mushrooms both discard the ItemBlock metadata in their
        // onBlockPlaced callback.  Command-block UI/permission activation is dispatched by the
        // controller; this arm is the ordinary placement result.
        99 | 100 | 137 => 0,
        // Lever's orientation enum stores eight mount combinations, with the horizontal values
        // matching the vanilla metadata codec and vertical mounts using the placer's yaw axis.
        69 => match side {
            Facing::Down => {
                if matches!(horizontal.axis(), mc_math::Axis::X) {
                    0
                } else {
                    7
                }
            }
            Facing::Up => {
                if matches!(horizontal.axis(), mc_math::Axis::X) {
                    6
                } else {
                    5
                }
            }
            Facing::North => 4,
            Facing::South => 3,
            Facing::West => 2,
            Facing::East => 1,
        },
        // BlockChest/BlockFurnace/BlockEnderChest onBlockPlacedBy rewrites FACING to the
        // placer's opposite horizontal direction on the client as well.
        54 | 61 | 130 | 146 => horizontal_opposite.index(),
        // BlockHopper.onBlockPlaced stores the clicked face's opposite, except that a top click
        // is forced down.  ENABLED defaults true, so there is no bit eight here.
        154 => {
            let facing = match side.opposite() {
                Facing::Up => Facing::Down,
                other => other,
            };
            facing.index()
        }
        // BlockFenceGate stores horizontalIndex and starts closed/unpowered.
        107 | 183..=187 => horizontal.horizontal_index().unwrap_or(0),
        // BlockPumpkin (the lit form has no item) stores horizontalIndex directly.
        86 => horizontal_opposite.horizontal_index().unwrap_or(0),
        // BlockQuartz changes only the pillar variant (metadata 2) according to the clicked
        // face's axis; chiseled/default variants collapse to their corresponding default state.
        155 => match held.damage {
            2 => match side.axis() {
                mc_math::Axis::X => 3,
                mc_math::Axis::Z => 4,
                mc_math::Axis::Y => 2,
            },
            1 => 1,
            _ => 0,
        },
        // BlockEndPortalFrame stores the placer's opposite horizontal facing and clears EYE.
        120 => horizontal_opposite.horizontal_index().unwrap_or(0),
        // BlockLadder retains the clicked horizontal face when it is supported. Otherwise (a top
        // or bottom click, or an unsupporting clicked block) Java scans `EnumFacing.Plane
        // .HORIZONTAL` - N-E-S-W - and picks the first supported face; default NORTH.
        65 => ladder_placed_facing(world, target, side).index(),
        // BlockTripWireHook stores horizontalIndex and all three state booleans false.
        131 => side.horizontal_index().unwrap_or(0),
        // BlockTrapDoor stores N/S/W/E as 0/1/2/3 and the top-half bit eight. It is a generic
        // ItemBlock callback even though its activation body is dispatched before this path.
        96 | 167 => {
            // `BlockTrapDoor.onBlockPlaced` only consumes face/hit data for horizontal clicks;
            // the placement gate rejects vertical faces. Keep the unreachable vertical callback
            // exact anyway: its default state is metadata zero, not a synthetic top-half flag.
            if !side.axis().is_horizontal() {
                return 0;
            }
            let facing_meta = match side {
                Facing::North => 0,
                Facing::South => 1,
                Facing::West => 2,
                Facing::East => 3,
                Facing::Up | Facing::Down => 0,
            };
            facing_meta | u8::from(hit_y > 0.5) << 3
        }
        // BlockVine starts with every attachment false, then sets the property for the opposite
        // horizontal face.  Its four booleans are SOUTH=1, WEST=2, NORTH=4, EAST=8.
        106 => match side {
            Facing::North => 1,
            Facing::South => 4,
            Facing::West => 8,
            Facing::East => 2,
            Facing::Up | Facing::Down => 0,
        },
        // BlockHay.onBlockPlaced uses the clicked face axis while preserving no subtype bits.
        170 => match side.axis() {
            mc_math::Axis::Y => 0,
            mc_math::Axis::X => 4,
            mc_math::Axis::Z => 8,
        },
        // BlockTorch's state codec maps EAST/WEST/SOUTH/NORTH to 1..4 and both vertical values
        // to the standing torch metadata 5.  The support gate normally makes the clicked face
        // valid; the fallback scan is Java's `EnumFacing.Plane.HORIZONTAL` order (N-E-S-W).
        50 => {
            let top_support = target.down();
            // java: `BlockTorch.canPlaceOn` - `block instanceof BlockFence` covers every fence
            // (85, 113 and the 1.8 wood fences 188..=192), plus glass, wall and stained glass.
            let top_valid = world.does_block_have_solid_top_surface(top_support)
                || matches!(
                    world.block_at(top_support).block().0,
                    20 | 85 | 95 | 113 | 139 | 188..=192
                );
            let clicked_valid = match side {
                Facing::Up => top_valid,
                Facing::Down => false,
                horizontal => {
                    world.is_block_normal_cube(target.relative(horizontal.opposite()), true)
                }
            };
            let facing = if clicked_valid {
                side
            } else {
                PLANE_HORIZONTAL
                    .iter()
                    .find(|&&f| world.is_block_normal_cube(target.relative(f.opposite()), true))
                    .copied()
                    .unwrap_or(Facing::Up)
            };
            match facing {
                Facing::East => 1,
                Facing::West => 2,
                Facing::South => 3,
                Facing::North => 4,
                Facing::Down | Facing::Up => 5,
            }
        }
        // BlockFlower.getStateFromMeta canonicalizes through EnumFlowerType.getType: the
        // yellow flower has only DANDELION (metadata 0), while the red flower accepts POPPY
        // through OXEYE_DAISY (metadata 0..=8).  Do this before packing the predicted state so
        // malformed creative stacks match Java's local ItemBlock callback rather than creating
        // an illegal state which the server immediately corrects.  Keep the signed comparison
        // here: a malformed negative ItemStack damage also maps to the default flower variant;
        // casting first would turn -1 into 255 and accidentally obscure that Java behavior.
        37 => 0,
        38 => {
            if (0..=8).contains(&held.damage) {
                held.damage as u8
            } else {
                0
            }
        }
        // BlockFarmland is a plain ItemBlock, so Item.getMetadata always returns zero before
        // BlockFarmland.getStateFromMeta masks its moisture property.  Malformed creative/NBT
        // damage must therefore still predict the dry farmland state rather than leaking the
        // raw item value into the local block state.
        // java: Item.getMetadata -> 0; BlockFarmland.getStateFromMeta(meta) -> meta & 7.
        60 => 0,
        // The ordinary callback is `return meta`, including every simple ItemBlock subtype.
        _ => held.damage as u8,
    }
}

/// The placement sound request `ItemBlock.onItemUse` makes after a successful state write.
///
/// java: `ItemBlock.java:68-70`: `getPlaceSound`, `(volume + 1) / 2`, and `frequency * .8`.
/// The returned request is intentionally separate from [`BlockUsePlacement`] so callers can
/// preserve the targeted-C08-before-local-prediction order and can test sound independently.
/// It is a `World.playSoundEffect` call, whose client sink (`RenderGlobal.playSound`) is empty:
/// the driver hands it to `Client::render_global_play_sound` and never plays it, because the
/// placer already hears the server's S29. Playing it locally made every placement sound twice.
#[must_use]
pub fn item_block_place_sound(placement: BlockUsePlacement) -> mc_audio::PlayRequest {
    let info = mc_data::block_info(placement.state.block());
    let sound = mc_physics::StepSound {
        pos: placement.pos,
        state: placement.state,
    }
    .sound_name();
    mc_audio::PlayRequest {
        event: mc_assets::ResourceLocation::minecraft(sound),
        volume: (info.step_sound_volume + 1.0) / 2.0,
        pitch: info.step_sound_pitch * 0.8,
        position: Some(placement.pos.to_vec3() + DVec3::splat(0.5)),
        attenuation: mc_audio::AttenuationType::Linear,
        repeat: false,
        repeat_delay_ticks: 0,
    }
}

fn is_stair_block(block: mc_data::BlockId) -> bool {
    matches!(
        block.0,
        53 | 67 | 108 | 109 | 114 | 128 | 134 | 135 | 136 | 156 | 163 | 164 | 180
    )
}

/// Whether the stack is one of vanilla 1.8.9's thirteen `BlockStairs` ItemBlocks.
#[must_use]
pub fn is_stair_item(held: &ItemStack) -> bool {
    placed_block(held).is_some_and(is_stair_block)
}

fn state_collision_intersects(state: BlockState, pos: BlockPos, blocking: &[Aabb]) -> bool {
    state.props().collision.boxes().iter().any(|bounds| {
        let translated = Aabb::from_bounds(
            bounds.min_x + f64::from(pos.x),
            bounds.min_y + f64::from(pos.y),
            bounds.min_z + f64::from(pos.z),
            bounds.max_x + f64::from(pos.x),
            bounds.max_y + f64::from(pos.y),
            bounds.max_z + f64::from(pos.z),
        );
        blocking
            .iter()
            .any(|other| translated.intersects_with(other))
    })
}

/// The common `PlayerControllerMP` adventure gate for block digs. It is intentionally based on
/// the controller game mode rather than `creative_mode`: an S39 packet can alter the latter
/// independently, while `clickBlock` reads `currentGameType`.
fn may_edit_block(
    game_mode: GameMode,
    allow_edit: bool,
    held: Option<&ItemStack>,
    target: BlockState,
) -> bool {
    match game_mode {
        GameMode::Spectator => false,
        GameMode::Adventure if !allow_edit => held_can_destroy(held, target),
        GameMode::NotSet | GameMode::Survival | GameMode::Creative | GameMode::Adventure => true,
    }
}

/// The client-side item callback that follows `PlayerControllerMP.onPlayerDestroyBlock`.
///
/// java: `ItemStack.onBlockDestroyed`, dispatching to `ItemTool`, `ItemSword`, or `ItemShears`.
/// The returned damage is the *attempted* durability loss before Unbreaking's random negation;
/// inventory code applies the damage only when the stack is actually damageable.  Keeping this
/// decision here avoids making the driver guess which broad `ItemKind` owns the callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDestroyItemEffect {
    /// Durability points requested by the item's callback.
    pub damage: u8,
    /// Whether the callback returns true and would trigger the item-use statistic.
    pub triggers_use_stat: bool,
}

/// The `onPlayerDestroyBlock` gates that are not covered by the packet-producing dig loop.
///
/// java: `PlayerControllerMP.onPlayerDestroyBlock` first delegates to the adventure edit gate,
/// then refuses a creative destruction when the held item is an `ItemSword`.  The latter is easy
/// to miss because creative `clickBlock` still sends its `START_DESTROY_BLOCK` packet and enters
/// the five-tick cooldown.
#[must_use]
pub fn can_destroy_block(
    game_mode: GameMode,
    allow_edit: bool,
    held: Option<&ItemStack>,
    target: BlockState,
) -> bool {
    if !may_edit_block(game_mode, allow_edit, held, target) {
        return false;
    }
    if matches!(game_mode, GameMode::Creative) {
        let sword = held.is_some_and(|stack| {
            mc_data::item_props(mc_data::ItemId(stack.item_id as u16)).kind
                == mc_data::ItemKind::Sword
        });
        if sword {
            return false;
        }
    }
    true
}

/// Resolves the held item's `onBlockDestroyed` callback for one broken block.
///
/// `ItemHoe` deliberately does not appear: it inherits `Item.onBlockDestroyed`, which returns
/// false and does not consume durability.  The same is true for every generic item.  Shears only
/// take damage for their six vanilla special cases; otherwise they delegate to that base method.
#[must_use]
pub fn block_destroy_item_effect(
    state: BlockState,
    held: Option<&ItemStack>,
) -> Option<BlockDestroyItemEffect> {
    let held = held?;
    let props = mc_data::item_props(mc_data::ItemId(held.item_id as u16));
    let block = state.block().0;
    let hardness_nonzero = state.props().hardness != 0.0;
    let (damage, triggers_use_stat) = match props.kind {
        mc_data::ItemKind::Pickaxe | mc_data::ItemKind::Axe | mc_data::ItemKind::Shovel => {
            (u8::from(hardness_nonzero), true)
        }
        mc_data::ItemKind::Sword => (if hardness_nonzero { 2 } else { 0 }, true),
        mc_data::ItemKind::Shears
            if state.props().material == mc_data::Material::Leaves
                || matches!(block, 30 | 31 | 106 | 132 | 35) =>
        {
            (1, true)
        }
        // ItemShears delegates to Item.onBlockDestroyed here, whose return is false.
        _ => return None,
    };
    Some(BlockDestroyItemEffect {
        damage,
        triggers_use_stat,
    })
}

/// Resolves the one client-visible `Block.onBlockDestroyedByPlayer` override that mutates a
/// second block. `BlockPistonMoving` removes an extended piston base behind the moving extension;
/// TNT's explosion and redstone diode neighbour notifications are server-only in a `WorldClient`.
#[must_use]
pub fn block_destroyed_callback(
    world: &World,
    pos: BlockPos,
    state: BlockState,
) -> Option<BlockPos> {
    if state.block().0 != 36 {
        return None;
    }
    let facing = match state.meta() & 7 {
        0 => Facing::Down,
        1 => Facing::Up,
        2 => Facing::North,
        3 => Facing::South,
        4 => Facing::West,
        5 => Facing::East,
        _ => return None,
    };
    let base_pos = pos.relative(facing.opposite());
    let base = world.block_at(base_pos);
    (matches!(base.block().0, 29 | 33) && base.meta() & 8 != 0).then_some(base_pos)
}

impl PlayerController {
    /// A controller that has never dug anything.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the player/tool inputs used by the block-break accumulator.
    pub fn set_dig_context(&mut self, context: DigContext) {
        self.dig_context = context;
    }

    /// The block being broken, if any. java: `getIsHittingBlock`
    #[must_use]
    pub fn hitting_block(&self) -> Option<BlockPos> {
        self.hitting.then_some(self.current_block)
    }

    /// java: `PlayerControllerMP.isHittingPosition` (`:363`).
    ///
    /// `ItemStack.areItemStackTagsEqual` excludes stack count; damageable items also exclude
    /// their damage/durability.  This is intentionally not `current == held`.
    fn is_hitting_position(&self, pos: BlockPos, held: Option<&ItemStack>) -> bool {
        if pos != self.current_block {
            return false;
        }
        match (&self.current_item_hitting_block, held) {
            (None, None) => true,
            (Some(previous), Some(current)) => {
                previous.item_id == current.item_id
                    && previous.nbt == current.nbt
                    && (mc_data::item_props(mc_data::ItemId(previous.item_id as u16)).damageable
                        || previous.damage == current.damage)
            }
            _ => false,
        }
    }

    /// Break progress, `0.0..1.0`. java: `curBlockDamageMP` — the crack overlay's input.
    #[must_use]
    pub fn damage(&self) -> f32 {
        self.damage
    }

    /// How many ticks the current dig has damaged its block: java's
    /// `stepSoundTickCounter`, which `clickBlock`'s START zeroes (the only place `hitting` turns
    /// on) and each damaging `onPlayerDamageBlock` tick increments.
    #[must_use]
    pub fn dig_ticks(&self) -> u32 {
        self.step_sound_counter as u32
    }

    /// java: `PlayerControllerMP.getBlockReachDistance` — 5 in creative, 4.5 otherwise.
    #[must_use]
    pub fn block_reach(creative: bool) -> f64 {
        if creative {
            5.0
        } else {
            4.5
        }
    }

    /// How far the **entity** ray in `getMouseOver` runs. java: `d0` after
    /// `if (this.mc.playerController.extendedReach()) { d0 = 6.0D; d1 = 6.0D; }`, i.e. `6.0` in
    /// creative and the plain block reach otherwise.
    ///
    /// This used to return `3.0` in survival, which is the wrong shape: vanilla's 3-block limit
    /// is applied *after* the loop, as a rejection that also discards the block hit. [`pick`]
    /// carries it now. Getting this wrong either lets a survival player hit someone 4.5 blocks
    /// away or, as here, digs a block vanilla would not have touched.
    #[must_use]
    pub fn entity_reach(creative: bool) -> f64 {
        if creative {
            6.0
        } else {
            Self::block_reach(false)
        }
    }

    /// Count the two per-tick timers down. java: the top of `Minecraft.runTick` (`rightClickDelayTimer`)
    /// and the `if (this.leftClickCounter > 0) --this.leftClickCounter;` inside the mouse block.
    pub fn tick(&mut self) {
        if self.right_click_delay > 0 {
            self.right_click_delay -= 1;
        }
        if self.left_click_counter > 0 {
            self.left_click_counter -= 1;
        }
    }

    /// Whether a held right-click may repeat this tick. java: `rightClickDelayTimer == 0`
    #[must_use]
    pub fn right_click_ready(&self) -> bool {
        self.right_click_delay == 0
    }

    /// java: `rightClickMouse` sets `this.rightClickDelayTimer = 4` on entry.
    pub fn note_right_click(&mut self) {
        self.right_click_delay = 4;
    }

    /// java: `clickMouse`'s `if (this.playerController.isNotCreative()) this.leftClickCounter = 10;`
    /// — swinging at nothing locks the dig loop out for half a second.
    pub fn note_missed_swing(&mut self, creative: bool) {
        if !creative {
            self.left_click_counter = 10;
        }
    }

    /// Applies the post-GUI/focus lockout from `Minecraft.runTick`/`setIngameFocus`.
    ///
    /// java: `leftClickCounter = 10000` when a screen is displayed and when focus is regained.
    /// This is intentionally distinct from [`Self::release_attack`], whose `= 0` models
    /// `sendClickBlockToController(false)` when the attack button is released.
    pub fn lock_out(&mut self) {
        self.left_click_counter = 10_000;
    }

    /// Whether `sendClickBlockToController` is allowed to progress a dig. java:
    /// `Minecraft.sendClickBlockToController` (`:1496`) — `leftClickCounter <= 0 &&
    /// !thePlayer.isUsingItem()`.
    ///
    /// The `isUsingItem()` half was missing until the item-use state machine existed to answer
    /// it (the first half). Without it a player who holds both buttons keeps
    /// mining while drawing a bow, which vanilla does not.
    #[must_use]
    pub fn may_damage_block(&self) -> bool {
        self.left_click_counter <= 0 && !self.is_using_item()
    }

    /// java: `leftClickCounter <= 0`, the half of [`Self::may_damage_block`] that is not the
    /// item-use gate. Read by "1.7 Animations"' packet-free punch-during-use swing.
    #[must_use]
    pub fn left_click_ready(&self) -> bool {
        self.left_click_counter <= 0
    }

    /// java: `EntityPlayer.isUsingItem()` (`:224`) — literally `itemInUse != null`.
    #[must_use]
    pub fn is_using_item(&self) -> bool {
        self.item_in_use.is_some()
    }

    /// The stack currently being drawn/eaten/drunk, if any. java: `EntityPlayer.itemInUse`.
    #[must_use]
    pub fn item_in_use(&self) -> Option<&ItemStack> {
        self.item_in_use.as_ref()
    }

    /// Hotbar slot that owns the latched item-use stack, when the live driver supplied one.
    #[must_use]
    pub fn item_in_use_slot(&self) -> Option<u8> {
        self.item_in_use_slot
    }

    /// Client-local identity of the inventory `ItemStack` latched by the production driver.
    ///
    /// Java's `EntityPlayer.itemInUse` callback mutates that object, so an equal-valued packet
    /// replacement in the owning slot must not receive the old bow's durability wear.
    #[must_use]
    pub fn item_in_use_identity(&self) -> Option<u64> {
        self.item_in_use_identity
    }

    /// The `EnumAction` exposed by the current use, if one is active.
    ///
    /// java: `EntityPlayer.itemInUse.getItemUseAction()` as consumed by `isBlocking`,
    /// `EntityPlayer.updateItemUse`, `ItemRenderer`, and `RenderPlayer`. Keeping this accessor
    /// on the controller prevents each consumer from reconstructing a subtly different item-id
    /// table from the duration alone.
    #[must_use]
    pub fn item_in_use_action(&self) -> Option<ItemUseAction> {
        self.item_in_use.as_ref().map(item_use_action)
    }

    /// The render-facing snapshot for the one latched `itemInUse` object.
    ///
    /// Java's renderer reads all four values from that object/player pair. Returning them as a
    /// single value gives the held-item caller one coherent source for `EnumAction.BOW`, the
    /// countdown, and the max-duration-derived model/FOV calculations.
    #[must_use]
    pub fn item_in_use_render_state(&self) -> Option<ItemUseRenderState> {
        let stack = self.item_in_use.as_ref()?;
        let max_duration = item_max_use_duration(stack)?;
        Some(ItemUseRenderState {
            action: item_use_action(stack),
            remaining_count: self.item_in_use_count,
            max_duration,
            elapsed_duration: self.item_in_use_duration(max_duration),
        })
    }

    /// How many ticks the current use has been held. java: `getItemInUseDuration` (`:232`) —
    /// `itemInUse.getMaxItemUseDuration() - itemInUseCount`, and `0` when nothing is in use.
    /// This is the number a bow's charge (and therefore its arrow velocity) is derived from.
    #[must_use]
    pub fn item_in_use_duration(&self, max_duration: i32) -> i32 {
        let Some(stack) = self.item_in_use.as_ref() else {
            return 0;
        };
        // The argument is retained for compatibility with existing callers that already know
        // the virtual max duration. Prefer the latched stack's own Java virtual result so a
        // stale/default caller value cannot disconnect bow model/FOV state from its countdown.
        let max_duration = item_max_use_duration(stack).unwrap_or(max_duration);
        max_duration - self.item_in_use_count
    }

    /// Remaining `EntityLivingBase.itemInUseCount`, consumed directly by first-person item
    /// transforms. Zero also covers the not-using state.
    #[must_use]
    pub fn item_in_use_count(&self) -> i32 {
        self.item_in_use_count
    }

    /// java: `EntityPlayer.setItemInUse` (`:2102`). The `stack != itemInUse` guard means
    /// re-starting the *same* use does not restart the counter, which is what stops a held
    /// right-click (which fires `rightClickMouse` every four ticks) from resetting a bow's draw
    /// on every repeat.
    pub fn set_item_in_use(&mut self, stack: ItemStack, duration: i32) {
        self.set_item_in_use_with_owner(stack, duration, None, None);
    }

    /// Start using the stack owned by `slot`, retaining that ownership through status-9
    /// completion. Repeating the same held use does not restart its countdown.
    pub fn set_item_in_use_at_slot(
        &mut self,
        stack: ItemStack,
        duration: i32,
        slot: u8,
        identity: u64,
    ) {
        self.set_item_in_use_with_owner(stack, duration, Some(slot.min(8)), Some(identity));
    }

    fn set_item_in_use_with_owner(
        &mut self,
        stack: ItemStack,
        duration: i32,
        slot: Option<u8>,
        identity: Option<u64>,
    ) {
        let same_stack = match (identity, self.item_in_use_identity) {
            (Some(current), Some(latched)) => current == latched,
            (None, None) => self.item_in_use.as_ref() == Some(&stack),
            _ => false,
        };
        if !same_stack || self.item_in_use_slot != slot {
            self.item_in_use = Some(stack);
            self.item_in_use_slot = slot;
            self.item_in_use_identity = identity;
            self.item_in_use_count = duration;
        }
    }

    /// java: `EntityPlayer.stopUsingItem` (`:237`) → `clearItemInUse` (`:247`). On a client
    /// world `setEating(false)` is skipped (`if (!this.worldObj.isRemote)`), and
    /// `onPlayerStoppedUsing` is the item's own server-side effect, so clearing the two fields
    /// is the whole of it here.
    pub fn stop_using_item(&mut self) {
        self.item_in_use = None;
        self.item_in_use_slot = None;
        self.item_in_use_identity = None;
        self.item_in_use_count = 0;
    }

    /// Advance the use timer one tick. java: the `if (this.itemInUse != null)` block at the top
    /// of `EntityPlayer.onUpdate`.
    ///
    /// ```text
    /// ItemStack itemstack = this.inventory.getCurrentItem();
    /// if (itemstack != this.itemInUse) { this.clearItemInUse(); }
    /// else { ...; if (--this.itemInUseCount == 0 && !this.worldObj.isRemote) onItemUseFinish(); }
    /// ```
    ///
    /// One deliberate translation:
    ///
    /// * the counter reaching `0` **finishes nothing**: `onItemUseFinish` is guarded by
    ///   `!worldObj.isRemote`, and the multiplayer-only rule means this client's world is always remote. The
    ///   eaten food, the fired arrow and the drunk potion all come back from the server. The
    ///   counter is still decremented (vanilla decrements it on both sides, and lets it go
    ///   negative on the client) because `getItemInUseDuration` reads it.
    pub fn tick_item_in_use(&mut self, held: Option<&ItemStack>) {
        if self.item_in_use.is_none() {
            return;
        }
        if held != self.item_in_use.as_ref() {
            self.stop_using_item();
            return;
        }
        self.item_in_use_count -= 1;
    }

    /// Production hotbar variant of [`Self::tick_item_in_use`]. Java compares the current
    /// inventory object with `itemInUse` by reference; the slot plus client-local identity make
    /// that comparison exact even when a replacement stack has identical serialized fields.
    pub fn tick_item_in_use_at_slot(
        &mut self,
        held: Option<&ItemStack>,
        slot: u8,
        identity: Option<u64>,
    ) {
        if self.item_in_use.is_none() {
            return;
        }
        if !self.item_in_use_matches_owner(held, slot, identity) {
            self.stop_using_item();
            return;
        }
        self.item_in_use_count -= 1;
    }

    /// Whether the current hotbar stack is the exact inventory object latched for item use.
    ///
    /// Java performs this reference comparison before `updateItemUse`, so presentation callers
    /// must share the predicate rather than emitting one stale cadence beat before the decrement
    /// path notices an equal-valued packet replacement.
    #[must_use]
    pub fn item_in_use_matches_owner(
        &self,
        held: Option<&ItemStack>,
        slot: u8,
        identity: Option<u64>,
    ) -> bool {
        self.item_in_use.is_some()
            && held.is_some()
            && self.item_in_use_slot == Some(slot.min(8))
            && self.item_in_use_identity.is_some()
            && self.item_in_use_identity == identity
    }

    /// java: `PlayerControllerMP.clickBlock` (`:198`)
    pub fn click_block(
        &mut self,
        world: &World,
        hit: BlockHit,
        game_mode: GameMode,
        allow_edit: bool,
        held: Option<&ItemStack>,
        on_ground: bool,
    ) -> DigStep {
        // The callback is one-shot: the application must drain it during this click dispatch.
        // Clearing first also prevents a stale local effect from surviving a denied/creative
        // click or being replayed after a missed driver drain.
        self.pending_block_click = None;
        // Same for the ABORT target: only the survival branch below can name one.
        self.pending_abort = None;
        if !may_edit_block(game_mode, allow_edit, held, world.block_at(hit.pos)) {
            return DigStep::None;
        }
        if matches!(game_mode, GameMode::Creative) {
            // java: START, then `clickBlockCreative` breaks it outright, then a five-tick gap.
            // The creative branch sends no ABORT and leaves `isHittingBlock` alone.
            self.hit_delay = 5;
            return DigStep::StartAndDestroy(hit.pos, hit.face);
        }
        if self.hitting && self.is_hitting_position(hit.pos, held) {
            return DigStep::None;
        }
        // java: `if (this.isHittingBlock) addToSendQueue(ABORT_DESTROY_BLOCK, this.currentBlock,
        // face)`, decided here from the controller's own fields. After an instant break
        // interrupted a dig, `currentBlock` is `onPlayerDestroyBlock`'s `(x, -1, z)` and vanilla
        // really does put that `y = -1` position on the wire (H3).
        self.pending_abort = self.hitting.then_some(self.current_block);
        let state = world.block_at(hit.pos);
        let solid = !state.props().is_air();
        // java: `if (flag && curBlockDamageMP == 0.0F) block1.onBlockClicked(...)`.
        // This is before the instant-break hardness branch, so a fresh click on an instant-break
        // block still receives the virtual callback exactly once.
        if solid && self.damage == 0.0 {
            self.pending_block_click = world.on_block_clicked(hit.pos);
        }
        // java: `if (flag && getPlayerRelativeBlockHardness(..) >= 1.0F) onPlayerDestroyBlock(..)`
        // — an instant-break block never becomes `isHittingBlock`, so no STOP packet follows and
        // the server acts on the START alone.
        //
        // That branch assigns **nothing** else: `isHittingBlock`, `currentBlock`,
        // `currentItemHittingBlock` and `curBlockDamageMP` keep the interrupted dig's values,
        // and `onPlayerDestroyBlock` then moves `currentBlock` to `(x, -1, z)`. So a held dig
        // swept onto tall grass stays "hitting": `rightClickMouse` stays refused and the next
        // target change or release sends `ABORT_DESTROY_BLOCK` at `y = -1`. This used to clear
        // `hitting` and the damage here (H3).
        if solid && player_relative_hardness_with_context(state, self.dig_context, on_ground) >= 1.0
        {
            return DigStep::StartAndDestroy(hit.pos, hit.face);
        }
        self.hitting = true;
        self.current_block = hit.pos;
        self.current_item_hitting_block = held.cloned();
        self.damage = 0.0;
        self.step_sound_counter = 0.0;
        // java: `sendBlockBreakProgress(id, currentBlock, (int)(curBlockDamageMP * 10.0F) - 1)`
        // with the damage just zeroed: `-1`, which clears any local crack.
        self.pending_break_progress = Some((hit.pos, -1));
        DigStep::Start(hit.pos, hit.face)
    }

    /// java: `PlayerControllerMP.onPlayerDamageBlock` (`:285`), called once per tick while the
    /// attack button is held and the crosshair is on a block.
    ///
    /// The **return value** is as load-bearing as the packet: `Minecraft.sendClickBlockToController`
    /// (`:1496`) swings the arm on every tick this returns true, and it returns true on *every*
    /// tick of a break — including the ticks that send nothing. See [`DamageTick::progressed`].
    pub fn damage_block(
        &mut self,
        world: &World,
        hit: BlockHit,
        game_mode: GameMode,
        allow_edit: bool,
        held: Option<&ItemStack>,
        on_ground: bool,
    ) -> DamageTick {
        self.damage_block_with_border(world, hit, game_mode, allow_edit, held, on_ground, true)
    }

    /// [`Self::damage_block`] with the live world-border result supplied by the caller.
    ///
    /// Java's `onPlayerDamageBlock` does not apply the border check uniformly: the creative
    /// branch checks it after consuming `blockHitDelay`, while a same-target continuation skips
    /// `clickBlock` (and therefore skips its adventure/border gates) entirely.  The caller passes
    /// the border result so a target change can take the `clickBlock` refusal path without making
    /// an outside-border hold consume neither its cooldown nor its swing.
    // Keep the explicit border flag at this controller boundary; collapsing it into context would
    // risk changing the creative/survival branch order.
    #[allow(clippy::too_many_arguments)]
    pub fn damage_block_with_border(
        &mut self,
        world: &World,
        hit: BlockHit,
        game_mode: GameMode,
        allow_edit: bool,
        held: Option<&ItemStack>,
        on_ground: bool,
        within_world_border: bool,
    ) -> DamageTick {
        if self.hit_delay > 0 {
            self.hit_delay -= 1;
            // java: `if (this.blockHitDelay > 0) { --this.blockHitDelay; return true; }` — the
            // cooldown ticks still swing.
            return DamageTick::silent_progress();
        }
        if matches!(game_mode, GameMode::Creative) && within_world_border {
            self.hit_delay = 5;
            // java: this creative START (`:296-298`) is never preceded by an ABORT.
            self.pending_abort = None;
            return DamageTick::sending(DigStep::StartAndDestroy(hit.pos, hit.face));
        }
        // java: `else if (this.isHittingPosition(posBlock))` — **without** `isHittingBlock`.
        // `resetBlockRemoving` clears `isHittingBlock` and the damage but keeps `currentBlock`
        // and `currentItemHittingBlock`, so when the crosshair slips onto an entity or the sky
        // for a tick and comes back to the same block with the same item, vanilla resumes the
        // dig from zero with **no new START** (and finishes it with a STOP). This used to require
        // `self.hitting` and re-sent START (H3).
        if self.is_hitting_position(hit.pos, held) {
            // java: the `isHittingPosition` continuation is deliberately before `clickBlock`'s
            // adventure/CanDestroy and world-border gates.  A mode or capability change during a
            // held dig therefore does not clear the accumulated client-side damage; only a new
            // target/item identity re-enters those gates.
            let state = world.block_at(hit.pos);
            if state.props().is_air() {
                // java: the block went away under us — stop, send nothing, and **return false**, so
                // this is the one tick of a dig that does not swing.
                self.hitting = false;
                return DamageTick::stalled();
            }
            self.damage +=
                player_relative_hardness_with_context(state, self.dig_context, on_ground);
            // java: this check is intentionally before `++stepSoundTickCounter`.  A newly started
            // survival dig has counter zero, so its *first accumulating* tick plays a hit; then the
            // following beats are 4, 8, ...  Moving this below the increment delays every sound by
            // a tick, while resetting it on a no-progress path invents sounds after a cooldown.
            let hit_sound = (self.step_sound_counter % 4.0 == 0.0 && !state.props().is_air())
                .then_some(BlockHitSound {
                    pos: hit.pos,
                    state,
                });
            self.step_sound_counter += 1.0;
            if self.damage >= 1.0 {
                self.hitting = false;
                self.damage = 0.0;
                self.step_sound_counter = 0.0;
                self.hit_delay = 5;
                // java: the tail `sendBlockBreakProgress(.., (int)(0.0F * 10.0F) - 1)` clears it.
                self.pending_break_progress = Some((self.current_block, -1));
                return hit_sound.map_or_else(
                    || DamageTick::sending(DigStep::Finish(hit.pos, hit.face)),
                    |sound| DamageTick::with_hit_sound(DigStep::Finish(hit.pos, hit.face), sound),
                );
            }
            // java: `sendBlockBreakProgress(id, currentBlock, (int)(curBlockDamageMP * 10.0F) -
            // 1)` after every accumulating tick; `as i32` truncates toward zero like `(int)`.
            self.pending_break_progress =
                Some((self.current_block, (self.damage * 10.0) as i32 - 1));
            // java: the accumulating tick still `return true`s, so it still swings. mc-rs used to
            // swing only on a state change, which is the `C0A` under-send half of W-SWING.
            return hit_sound.map_or_else(DamageTick::silent_progress, |sound| {
                DamageTick::with_hit_sound(DigStep::None, sound)
            });
        }

        // A target change is the only continuation path that re-enters clickBlock.  Its Java
        // border/permission gates are represented here before delegating to the existing helper;
        // the initial mouse-down path performs the same gate in the driver.
        if !within_world_border {
            return DamageTick::stalled();
        }
        // java: `else return this.clickBlock(posBlock, directionFacing);` — a refused
        // adventure/border gate returns false and therefore must not cause the caller to swing.
        let step = self.click_block(world, hit, game_mode, allow_edit, held, on_ground);
        if matches!(step, DigStep::None) {
            DamageTick::stalled()
        } else {
            DamageTick::sending(step)
        }
    }

    /// java: `resetBlockRemoving` (`:274`) — a no-op unless a dig was actually in progress.
    /// Takes the one-shot client-local `Block.onBlockClicked` result from the most recent fresh
    /// survival click. The application layer owns player food, particles, and local effects.
    pub fn take_block_click(&mut self) -> Option<LocalBlockClick> {
        self.pending_block_click.take()
    }

    /// java: `resetBlockRemoving`: a no-op unless a dig was actually in progress.
    ///
    /// ```text
    /// if (this.isHittingBlock) {
    ///     addToSendQueue(new C07PacketPlayerDigging(ABORT_DESTROY_BLOCK, currentBlock, DOWN));
    ///     this.isHittingBlock = false;
    ///     this.curBlockDamageMP = 0.0F;
    ///     this.mc.theWorld.sendBlockBreakProgress(entityId, this.currentBlock, -1);
    /// }
    /// ```
    ///
    /// `currentBlock` and `currentItemHittingBlock` survive, which is what lets
    /// [`Self::damage_block_with_border`] resume the same block without a START.
    pub fn reset_block_removing(&mut self) -> DigStep {
        // The application normally drains this synchronously with the click's C07, but a modal
        // transition can reset an instant-break click before that drain. Never let its local
        // callback leak into a later, unrelated click.
        self.pending_block_click = None;
        if !self.hitting {
            return DigStep::None;
        }
        self.hitting = false;
        self.damage = 0.0;
        self.pending_abort = Some(self.current_block);
        self.pending_break_progress = Some((self.current_block, -1));
        DigStep::Abort
    }

    /// The position the ABORT of the [`DigStep::Start`], [`DigStep::StartAndDestroy`] or
    /// [`DigStep::Abort`] just returned must carry, if Java's `isHittingBlock` guard sent one.
    /// One-shot: the packet layer drains it while emitting that step.
    pub fn take_pending_abort(&mut self) -> Option<BlockPos> {
        self.pending_abort.take()
    }

    /// The latest `sendBlockBreakProgress(entityId, currentBlock, progress)` the controller made
    /// since the last call, if any. Java's local crack changes only on these calls (and on
    /// `RenderGlobal`'s 400-tick expiry), so a tick without one leaves the crack alone.
    pub fn take_break_progress(&mut self) -> Option<(BlockPos, i32)> {
        self.pending_break_progress.take()
    }

    /// Marks the controller's `currentBlock` as no longer active after local destruction.
    ///
    /// java: `onPlayerDestroyBlock` keeps the previous X/Z and writes `Y = -1`, even though the
    /// world mutation and packet have already happened.  The driver calls this only after the
    /// packet-preserving local prediction path has accepted the destruction.
    pub fn note_block_destroyed(&mut self) {
        self.current_block = BlockPos::new(self.current_block.x, -1, self.current_block.z);
    }

    /// java: `Minecraft.sendClickBlockToController(false)` sets `leftClickCounter = 0`, so
    /// releasing the button cancels the ten-tick lockout immediately.
    pub fn release_attack(&mut self) {
        self.pending_block_click = None;
        self.left_click_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stone() -> BlockState {
        BlockState::new(mc_data::block_by_name("stone").expect("stone"), 0)
    }

    /// A world with one solid block at `pos`, in a column that has actually been loaded.
    fn world_with(pos: BlockPos) -> World {
        let mut world = World::new();
        let bitmask = 1u16 << ((pos.y >> 4) as u16);
        let data = vec![0u8; mc_world::expected_chunk_data_len(bitmask, true, true)];
        world
            .apply_chunk_data(pos.chunk_pos(), true, bitmask, &data)
            .expect("an empty column decodes");
        world.set_block(pos, stone());
        world
    }

    fn world_with_state(pos: BlockPos, state: BlockState) -> World {
        let mut world = world_with(pos);
        world.set_block(pos, state);
        world
    }

    fn can_destroy_stack(names: &[&str]) -> ItemStack {
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert(
            "CanDestroy",
            mc_nbt::Nbt::list(
                names
                    .iter()
                    .map(|name| mc_nbt::Nbt::String((*name).to_owned()))
                    .collect(),
            ),
        );
        ItemStack {
            item_id: mc_data::item_by_name("stick").expect("stick").0 as i16,
            count: 1,
            damage: 0,
            nbt: Some(nbt),
        }
    }

    /// `ItemStack.canDestroy` compares the registry's block object, not the stack's item type
    /// or the target's state metadata. A non-string `CanDestroy` list is rejected by Java's
    /// `getTagList(key, 8)` and is the negative control here.
    #[test]
    fn can_destroy_uses_the_block_registry_and_string_nbt_list() {
        let stone = BlockState::new(mc_data::block_by_name("stone").unwrap(), 7);
        let dirt = BlockState::new(mc_data::block_by_name("dirt").unwrap(), 0);
        let allowed = can_destroy_stack(&["minecraft:stone", "dirt"]);
        assert!(held_can_destroy(Some(&allowed), stone));
        assert!(held_can_destroy(Some(&allowed), dirt));
        assert!(!held_can_destroy(None, stone));

        let mut malformed_nbt = mc_nbt::NbtCompound::new();
        malformed_nbt.insert("CanDestroy", mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Int(1)]));
        let malformed = ItemStack {
            nbt: Some(malformed_nbt),
            ..allowed
        };
        assert!(
            !held_can_destroy(Some(&malformed), stone),
            "a non-string list must behave like Java's empty replacement list"
        );
    }

    /// `clickBlock` must deny before it creates a START packet, while a continuing damage tick
    /// deliberately skips that gate after the controller becomes Spectator.
    #[test]
    fn block_digging_honours_spectator_adventure_and_can_destroy() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = hit_at(target, Facing::Up);
        let allowed = can_destroy_stack(&["stone"]);
        let mut controller = PlayerController::new();

        assert_eq!(
            controller.click_block(&world, hit, GameMode::Spectator, true, Some(&allowed), true),
            DigStep::None,
            "Spectator is denied even with CanDestroy"
        );
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Adventure, false, None, true),
            DigStep::None,
            "Adventure without a held CanDestroy stack is denied"
        );
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Adventure, true, None, true),
            DigStep::Start(target, Facing::Up),
            "the gate is specifically allowEdit, not a blanket Adventure denial"
        );
        assert_eq!(controller.reset_block_removing(), DigStep::Abort);
        assert_eq!(
            controller.damage_block(
                &world,
                hit,
                GameMode::Adventure,
                false,
                Some(&allowed),
                true
            ),
            DamageTick::sending(DigStep::Start(target, Facing::Up)),
            "an exact CanDestroy name admits Adventure's initial START"
        );
        let continued = controller.damage_block(
            &world,
            hit,
            GameMode::Spectator,
            false,
            Some(&allowed),
            true,
        );
        assert!(
            continued.progressed,
            "Java's matching-target continuation bypasses clickBlock's mode gate"
        );
        assert_eq!(controller.hitting_block(), Some(target));
    }

    #[test]
    fn first_left_click_captures_each_client_block_callback_once() {
        let pos = BlockPos::new(0, 64, 0);
        let hit = hit_at(pos, Facing::Up);

        for (state, expected) in [
            (
                BlockState::new(mc_data::BlockId(73), 0),
                LocalBlockClick::RedstoneOre { pos },
            ),
            (
                BlockState::new(mc_data::BlockId(92), 0),
                LocalBlockClick::Cake { pos, bites: 0 },
            ),
            (
                BlockState::new(mc_data::BlockId(122), 0),
                LocalBlockClick::DragonEgg { pos },
            ),
        ] {
            let world = world_with_state(pos, state);
            let mut controller = PlayerController::new();
            assert_eq!(
                controller.click_block(&world, hit, GameMode::Survival, true, None, true),
                DigStep::Start(pos, Facing::Up)
            );
            assert_eq!(controller.take_block_click(), Some(expected));
            assert_eq!(
                controller.take_block_click(),
                None,
                "the local callback is one-shot"
            );

            // Holding the same target does not re-enter clickBlock/onBlockClicked. This is the
            // repeated-damage-tick negative control for the tempting `damage == 0` mutant.
            assert_eq!(
                controller.click_block(&world, hit, GameMode::Survival, true, None, true),
                DigStep::None
            );
            assert_eq!(controller.take_block_click(), None);
        }
    }

    #[test]
    fn reset_and_release_discard_an_undrained_block_click_callback() {
        let pos = BlockPos::new(0, 64, 0);
        let world = world_with_state(pos, BlockState::new(mc_data::BlockId(73), 0));
        let hit = hit_at(pos, Facing::Up);

        let mut reset = PlayerController::new();
        assert_eq!(
            reset.click_block(&world, hit, GameMode::Survival, true, None, true),
            DigStep::Start(pos, Facing::Up)
        );
        assert_eq!(reset.reset_block_removing(), DigStep::Abort);
        assert_eq!(reset.take_block_click(), None);

        let mut released = PlayerController::new();
        assert_eq!(
            released.click_block(&world, hit, GameMode::Survival, true, None, true),
            DigStep::Start(pos, Facing::Up)
        );
        released.release_attack();
        assert_eq!(released.take_block_click(), None);

        // Reset must invalidate the marker before its `!hitting` early return. Instant-break is
        // the production route to this state; setting the private controller bit directly keeps
        // this unit test independent of item/tool hardness tables.
        let mut controller = PlayerController::new();
        let _ = controller.click_block(&world, hit, GameMode::Survival, true, None, true);
        controller.hitting = false;
        assert_eq!(controller.reset_block_removing(), DigStep::None);
        assert_eq!(controller.take_block_click(), None);
    }

    #[test]
    fn damage_block_new_target_routes_through_first_click_hook_and_nonzero_damage_skips_it() {
        let first = BlockPos::new(0, 64, 0);
        let second = BlockPos::new(1, 64, 0);
        let mut world = world_with_state(first, BlockState::new(mc_data::BlockId(73), 0));
        world.set_block(second, BlockState::new(mc_data::BlockId(92), 0));
        let mut controller = PlayerController::new();

        // The live held-button path calls damageBlock on a fresh target; its delegated clickBlock
        // must expose the same one-shot callback before the caller handles DigStep::Start.
        let tick = controller.damage_block(
            &world,
            hit_at(first, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        assert_eq!(tick.step, DigStep::Start(first, Facing::Up));
        assert_eq!(
            controller.take_block_click(),
            Some(LocalBlockClick::RedstoneOre { pos: first })
        );

        // A nonzero accumulated damage means Java's first-click gate is false for the new target;
        // this catches a callback implementation that keys only on `hitting` or target changes.
        controller.damage = 0.25;
        assert_eq!(
            controller.click_block(
                &world,
                hit_at(second, Facing::Up),
                GameMode::Survival,
                true,
                None,
                true,
            ),
            DigStep::Start(second, Facing::Up)
        );
        assert_eq!(controller.take_block_click(), None);
    }

    #[test]
    fn local_destroy_gate_keeps_creative_swords_and_spectators_from_mutating_world() {
        let stone = stone();
        let sword = item("diamond_sword");
        assert!(!can_destroy_block(
            GameMode::Creative,
            true,
            Some(&sword),
            stone
        ));
        assert!(!can_destroy_block(GameMode::Spectator, true, None, stone));
        assert!(!can_destroy_block(GameMode::Adventure, false, None, stone));
        let allowed = can_destroy_stack(&["stone"]);
        assert!(can_destroy_block(
            GameMode::Adventure,
            false,
            Some(&allowed),
            stone
        ));
        assert!(can_destroy_block(GameMode::Creative, true, None, stone));
    }

    #[test]
    fn local_destroy_item_callback_matches_tool_sword_shears_and_base_item_dispatch() {
        let stone = stone();
        let pick = item("iron_pickaxe");
        let sword = item("diamond_sword");
        let hoe = item("diamond_hoe");
        let shears = item("shears");
        let leaves = block("leaves");
        let wool = block("wool");

        assert_eq!(
            block_destroy_item_effect(stone, Some(&pick)),
            Some(BlockDestroyItemEffect {
                damage: 1,
                triggers_use_stat: true,
            })
        );
        assert_eq!(
            block_destroy_item_effect(stone, Some(&sword)),
            Some(BlockDestroyItemEffect {
                damage: 2,
                triggers_use_stat: true,
            })
        );
        assert_eq!(
            block_destroy_item_effect(stone, Some(&hoe)),
            None,
            "ItemHoe inherits Item.onBlockDestroyed"
        );
        assert_eq!(
            block_destroy_item_effect(leaves, Some(&shears)),
            Some(BlockDestroyItemEffect {
                damage: 1,
                triggers_use_stat: true,
            })
        );
        assert_eq!(
            block_destroy_item_effect(wool, Some(&shears)),
            Some(BlockDestroyItemEffect {
                damage: 1,
                triggers_use_stat: true,
            })
        );
        assert_eq!(
            block_destroy_item_effect(stone, Some(&shears)),
            None,
            "ItemShears delegates to the base callback for ordinary blocks"
        );
    }

    #[test]
    fn moving_piston_destroy_callback_removes_only_an_extended_base() {
        let pos = BlockPos::new(0, 64, 0);
        let base_pos = pos.relative(Facing::West);
        // The moving extension and its base straddle x=0, so load both client columns just like
        // the real chunk stream would before exercising the cross-block callback.
        let mut world = world_of(&[(pos, "stone"), (base_pos, "stone")]);
        world.set_block(pos, BlockState::new(mc_data::BlockId(36), 5));
        world.set_block(base_pos, BlockState::new(mc_data::BlockId(33), 13));
        assert_eq!(
            block_destroyed_callback(&world, pos, BlockState::new(mc_data::BlockId(36), 5)),
            Some(base_pos)
        );

        world.set_block(base_pos, BlockState::new(mc_data::BlockId(33), 5));
        assert_eq!(
            block_destroyed_callback(&world, pos, BlockState::new(mc_data::BlockId(36), 5)),
            None,
            "the callback only clears a piston base whose EXTENDED bit is set"
        );
    }

    fn door_world(lower_meta: u8, upper_meta: u8) -> (World, BlockPos) {
        let lower = BlockPos::new(0, 64, 0);
        let mut world = world_with(lower);
        let door = mc_data::block_by_name("wooden_door").expect("wooden door");
        world.set_block(lower, BlockState::new(door, lower_meta));
        world.set_block(lower.up(), BlockState::new(door, upper_meta));
        (world, lower)
    }

    fn expected_door_bounds(facing: u8, open: bool, hinge_right: bool) -> Aabb {
        const F: f64 = 0.1875;
        const G: f64 = 1.0 - F;
        let (min_x, min_z, max_x, max_z) = if open {
            match (facing, hinge_right) {
                (0, false) => (0.0, 0.0, 1.0, F),
                (0, true) => (0.0, G, 1.0, 1.0),
                (1, false) => (G, 0.0, 1.0, 1.0),
                (1, true) => (0.0, 0.0, F, 1.0),
                (2, false) => (0.0, G, 1.0, 1.0),
                (2, true) => (0.0, 0.0, 1.0, F),
                (3, false) => (0.0, 0.0, F, 1.0),
                (3, true) => (G, 0.0, 1.0, 1.0),
                _ => unreachable!("door metadata has four horizontal facings"),
            }
        } else {
            match facing {
                0 => (0.0, 0.0, F, 1.0),
                1 => (0.0, 0.0, 1.0, F),
                2 => (G, 0.0, 1.0, 1.0),
                3 => (0.0, G, 1.0, 1.0),
                _ => unreachable!("door metadata has four horizontal facings"),
            }
        };
        Aabb::new(DVec3::new(min_x, 0.0, min_z), DVec3::new(max_x, 1.0, max_z))
    }

    /// A door's bounds are live state, not its lower cell's static registry bounds.  Exercise
    /// all 32 lower/upper observations: facing/open arrive from the lower cell, hinge arrives
    /// from the upper one, and both cells must expose exactly the same X/Z slab.
    #[test]
    fn door_raytrace_uses_the_combined_lower_upper_bounds() {
        for facing in 0u8..4 {
            for open in [false, true] {
                for hinge_right in [false, true] {
                    let lower_meta = facing | if open { 4 } else { 0 };
                    let upper_meta = 8 | if hinge_right { 1 } else { 0 };
                    let (world, lower) = door_world(lower_meta, upper_meta);
                    let expected = expected_door_bounds(facing, open, hinge_right);

                    for pos in [lower, lower.up()] {
                        let actual = selection_bounds(&world, pos, world.block_at(pos));
                        assert_eq!(
                            actual, expected,
                            "facing={facing}, open={open}, hinge_right={hinge_right}, pos={pos:?}"
                        );
                    }

                    // A static full cube would hit this ray.  It travels through the empty side
                    // of this live slab, so it is an isolated negative control for the raytrace
                    // (not merely for the helper's returned numbers).
                    let (start, end) = if expected.max.x - expected.min.x < 0.5 {
                        let x = if expected.min.x == 0.0 { 0.9 } else { 0.1 };
                        (DVec3::new(x, 64.5, -1.0), DVec3::new(x, 64.5, 2.0))
                    } else {
                        let z = if expected.min.z == 0.0 { 0.9 } else { 0.1 };
                        (DVec3::new(-1.0, 64.5, z), DVec3::new(2.0, 64.5, z))
                    };
                    assert!(
                        raytrace_blocks(&world, start, end)
                            .and_then(RayTrace::as_block)
                            .is_none(),
                        "static bounds falsely hit facing={facing}, open={open}, hinge_right={hinge_right}"
                    );
                }
            }
        }
    }

    /// Fences do not raytrace as their isolated registry post once a live neighbour supplies an
    /// arm.  The ray travels through the part that an isolated post leaves empty, so this checks
    /// the actual click path rather than only the renderer's shared bounds helper.
    #[test]
    fn fence_raytrace_uses_live_connections_and_rejects_other_materials() {
        let fence = BlockPos::new(0, 64, 0);
        let behind = BlockPos::new(0, 64, 2);
        let start = DVec3::new(0.9, 64.5, -1.0);
        let end = DVec3::new(0.9, 64.5, 3.0);

        let connected = world_of(&[
            (fence, "minecraft:fence"),
            (fence.east(), "minecraft:fence"),
            (behind, "minecraft:stone"),
        ]);
        let hit = raytrace_blocks(&connected, start, end)
            .and_then(RayTrace::as_block)
            .expect("the east arm is targetable");
        assert_eq!(hit.pos, fence);
        assert!((hit.hit_vec.z - 0.375).abs() < 1e-9, "{hit:?}");

        let mismatched = world_of(&[
            (fence, "minecraft:fence"),
            (fence.east(), "minecraft:nether_brick_fence"),
            (behind, "minecraft:stone"),
        ]);
        let hit = raytrace_blocks(&mismatched, start, end)
            .and_then(RayTrace::as_block)
            .expect("the stone behind the isolated post");
        assert_eq!(
            hit.pos, behind,
            "wood and nether-brick fences must not connect"
        );
    }

    /// Java's `BlockFence.canConnectTo` also accepts a gate, and the selection box is rebuilt
    /// from the live world on every raytrace.  Removing the gate must therefore turn the same
    /// arm probe from a centre hit into the stone behind it.  This exercises both the gate branch
    /// and the stale-neighbour failure mode of a cached/static selection box.
    #[test]
    fn fence_raytrace_accepts_gates_and_refreshes_after_gate_removal() {
        let fence = BlockPos::new(0, 64, 0);
        let gate = fence.east();
        let behind = BlockPos::new(0, 64, 2);
        let start = DVec3::new(0.9, 64.5, -1.0);
        let end = DVec3::new(0.9, 64.5, 3.0);
        let mut world = world_of(&[
            (fence, "minecraft:fence"),
            (gate, "minecraft:fence_gate"),
            (behind, "minecraft:stone"),
        ]);

        assert_eq!(
            raytrace_blocks(&world, start, end)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(fence),
            "a fence gate supplies the east connection"
        );

        world.set_block(gate, BlockState::AIR);
        assert_eq!(
            raytrace_blocks(&world, start, end)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(behind),
            "removing the gate must refresh the isolated fence selection box"
        );
    }

    /// The interaction seam owns a world-state read, just like Java's
    /// `Block.collisionRayTrace` caller.  A stale state argument must not send a fence through
    /// the static renderer fallback (or make the caller's material authoritative); the live
    /// centre and its east neighbour still define the one-block selection envelope.
    #[test]
    fn fence_selection_consumer_prefers_live_world_state_over_stale_argument() {
        let fence = BlockPos::new(0, 64, 0);
        let world = world_of(&[
            (fence, "minecraft:fence"),
            (fence.east(), "minecraft:fence"),
        ]);
        let expected = Aabb::new(DVec3::new(0.375, 0.0, 0.375), DVec3::new(1.0, 1.0, 0.625));

        assert_eq!(
            selection_bounds(&world, fence, stone()),
            expected,
            "the live centre must win over a stale non-fence state argument"
        );
        assert_ne!(
            expected,
            Aabb::new(DVec3::ZERO, DVec3::ONE),
            "negative control: the stale static stone path is a different shape"
        );
    }

    /// `BlockChest.setBlockBoundsBasedOnState` is read at the raytrace call site, so each live
    /// same-singleton pair must widen the centre in exactly its own direction.  The probe is
    /// outside the isolated chest on that axis; after removing the neighbour it must pass on to
    /// the stone behind it.  The ordinary and trapped variants use different legal facings to
    /// prove that Java compares the block singleton, not the complete metadata state.
    #[test]
    fn chest_raytrace_uses_live_pairs_in_all_directions_and_types() {
        let pos = BlockPos::new(0, 64, 0);
        let behind_x = BlockPos::new(2, 64, 0);
        let behind_z = BlockPos::new(0, 64, 2);
        let cases = [
            (
                "north",
                pos.north(),
                DVec3::new(-1.0, 64.5, 0.05),
                DVec3::new(3.0, 64.5, 0.05),
                behind_x,
                Aabb::new(
                    DVec3::new(0.0625, 0.0, 0.0),
                    DVec3::new(0.9375, 0.875, 0.9375),
                ),
            ),
            (
                "east",
                pos.east(),
                DVec3::new(0.95, 64.5, -1.0),
                DVec3::new(0.95, 64.5, 3.0),
                behind_z,
                Aabb::new(
                    DVec3::new(0.0625, 0.0, 0.0625),
                    DVec3::new(1.0, 0.875, 0.9375),
                ),
            ),
            (
                "south",
                pos.south(),
                DVec3::new(-1.0, 64.5, 0.95),
                DVec3::new(3.0, 64.5, 0.95),
                behind_x,
                Aabb::new(
                    DVec3::new(0.0625, 0.0, 0.0625),
                    DVec3::new(0.9375, 0.875, 1.0),
                ),
            ),
            (
                "west",
                pos.west(),
                DVec3::new(0.05, 64.5, -1.0),
                DVec3::new(0.05, 64.5, 3.0),
                behind_z,
                Aabb::new(
                    DVec3::new(0.0, 0.0, 0.0625),
                    DVec3::new(0.9375, 0.875, 0.9375),
                ),
            ),
        ];

        for centre_name in ["minecraft:chest", "minecraft:trapped_chest"] {
            let centre_id = mc_data::block_by_name(centre_name).expect(centre_name);
            for (direction, neighbour, start, end, behind, expected) in cases {
                let mut world = world_of(&[
                    (pos, "minecraft:stone"),
                    (neighbour, "minecraft:stone"),
                    (behind, "minecraft:stone"),
                ]);
                world.set_block(pos, BlockState::new(centre_id, 2));
                world.set_block(neighbour, BlockState::new(centre_id, 5));

                assert_eq!(
                    selection_bounds(&world, pos, world.block_at(pos)),
                    expected,
                    "{centre_name} {direction} pair must widen the live selection box"
                );
                let hit = raytrace_blocks(&world, start, end)
                    .and_then(RayTrace::as_block)
                    .expect("the connected chest seam must be targetable");
                assert_eq!(hit.pos, pos, "{centre_name} {direction} pair");

                world.set_block(neighbour, BlockState::AIR);
                assert_eq!(
                    selection_bounds(&world, pos, world.block_at(pos)),
                    Aabb::new(
                        DVec3::new(0.0625, 0.0, 0.0625),
                        DVec3::new(0.9375, 0.875, 0.9375),
                    ),
                    "removing the {direction} neighbour must retract the chest box"
                );
                let hit = raytrace_blocks(&world, start, end)
                    .and_then(RayTrace::as_block)
                    .expect("the removal probe must reach the stone behind the chest");
                assert_eq!(hit.pos, behind, "{centre_name} {direction} removal");
            }
        }
    }

    /// The chest source checks north, south, west, east with an `else-if` chain, and ordinary
    /// and trapped chests are distinct singleton blocks despite sharing `BlockChest`.  This is
    /// also a caller seam control: a stale state argument cannot route a live paired chest to a
    /// static renderer box, while a mixed pair must remain isolated.
    #[test]
    fn chest_selection_preserves_precedence_type_identity_and_live_caller_state() {
        let pos = BlockPos::new(0, 64, 0);
        let behind = BlockPos::new(0, 64, 2);
        let chest = mc_data::block_by_name("minecraft:chest").expect("chest");
        let trapped = mc_data::block_by_name("minecraft:trapped_chest").expect("trapped chest");

        let mut world = world_of(&[
            (pos, "minecraft:stone"),
            (pos.north(), "minecraft:stone"),
            (pos.east(), "minecraft:stone"),
            (pos.south(), "minecraft:stone"),
            (pos.west(), "minecraft:stone"),
            (behind, "minecraft:stone"),
        ]);
        world.set_block(pos, BlockState::new(chest, 2));
        for neighbour in [pos.north(), pos.east(), pos.south(), pos.west()] {
            world.set_block(neighbour, BlockState::new(chest, 5));
        }

        let north = Aabb::new(
            DVec3::new(0.0625, 0.0, 0.0),
            DVec3::new(0.9375, 0.875, 0.9375),
        );
        assert_eq!(
            selection_bounds(&world, pos, stone()),
            north,
            "the live paired chest must win over a stale state argument, with north precedence"
        );
        let hit = raytrace_blocks(
            &world,
            DVec3::new(-1.0, 64.5, 0.05),
            DVec3::new(3.0, 64.5, 0.05),
        )
        .and_then(RayTrace::as_block)
        .expect("the north-precedence seam must be targetable");
        assert_eq!(hit.pos, pos);

        let mut mixed = world_of(&[
            (pos, "minecraft:stone"),
            (pos.east(), "minecraft:stone"),
            (behind, "minecraft:stone"),
        ]);
        mixed.set_block(pos, BlockState::new(chest, 2));
        mixed.set_block(pos.east(), BlockState::new(trapped, 5));
        let isolated = Aabb::new(
            DVec3::new(0.0625, 0.0, 0.0625),
            DVec3::new(0.9375, 0.875, 0.9375),
        );
        assert_eq!(
            selection_bounds(&mixed, pos, mixed.block_at(pos)),
            isolated,
            "ordinary and trapped chests must not form a pair"
        );
        let hit = raytrace_blocks(
            &mixed,
            DVec3::new(0.95, 64.5, -1.0),
            DVec3::new(0.95, 64.5, 3.0),
        )
        .and_then(RayTrace::as_block)
        .expect("the mixed-pair probe must reach the stone behind the isolated chest");
        assert_eq!(hit.pos, behind);
    }

    /// The joined seam is deliberately outside the constructor/static chest box.  This negative
    /// control makes the caller test mutation-sensitive: replacing the live world oracle with a
    /// static isolated bounds path must miss the centre and expose the stone behind it instead.
    #[test]
    fn chest_raytrace_seam_is_not_satisfied_by_static_isolated_bounds() {
        let pos = BlockPos::new(0, 64, 0);
        let behind = BlockPos::new(0, 64, 2);
        let chest = mc_data::block_by_name("minecraft:chest").expect("chest");
        let mut world = world_of(&[
            (pos, "minecraft:stone"),
            (pos.east(), "minecraft:stone"),
            (behind, "minecraft:stone"),
        ]);
        world.set_block(pos, BlockState::new(chest, 2));
        world.set_block(pos.east(), BlockState::new(chest, 5));

        let isolated = Aabb::new(
            DVec3::new(0.0625, 0.0, 0.0625),
            DVec3::new(0.9375, 0.875, 0.9375),
        );
        let seam = Aabb::from_bounds(0.95, 0.1, 0.2, 0.99, 0.7, 0.8);
        assert!(!isolated.intersects_with(&seam));
        assert!(selection_bounds(&world, pos, world.block_at(pos)).intersects_with(&seam));

        let hit = raytrace_blocks(
            &world,
            DVec3::new(0.95, 64.5, -1.0),
            DVec3::new(0.95, 64.5, 3.0),
        )
        .and_then(RayTrace::as_block)
        .expect("the live east seam must target the centre chest");
        assert_eq!(hit.pos, pos);
        assert_ne!(
            hit.pos, behind,
            "static isolated bounds would hit the stone behind"
        );
    }

    /// The ray path must consume the same one-block `setBlockBoundsBasedOnState` envelope as
    /// the outline, not the 1.5-block collision rails.  Each probe travels through a point that
    /// is inside exactly one possible arm; the four neighbour positions themselves cannot
    /// intercept that probe.  Cover all masks for both `BlockFence` material families.
    #[test]
    fn fence_raytrace_uses_every_live_connection_mask_for_both_materials() {
        let pos = BlockPos::new(0, 64, 0);
        let arms = [
            (
                1_u8,
                pos.north(),
                DVec3::new(-1.0, 64.5, 0.1),
                DVec3::new(2.0, 64.5, 0.1),
                "north",
            ),
            (
                2_u8,
                pos.east(),
                DVec3::new(0.9, 64.5, -1.0),
                DVec3::new(0.9, 64.5, 2.0),
                "east",
            ),
            (
                4_u8,
                pos.south(),
                DVec3::new(-1.0, 64.5, 0.9),
                DVec3::new(2.0, 64.5, 0.9),
                "south",
            ),
            (
                8_u8,
                pos.west(),
                DVec3::new(0.1, 64.5, -1.0),
                DVec3::new(0.1, 64.5, 2.0),
                "west",
            ),
        ];

        for fence_name in ["minecraft:fence", "minecraft:nether_brick_fence"] {
            for mask in 0_u8..16 {
                let mut blocks = vec![(pos, fence_name)];
                for (bit, neighbour, ..) in arms {
                    if mask & bit != 0 {
                        blocks.push((neighbour, fence_name));
                    }
                }
                let world = world_of(&blocks);

                for (bit, _, start, end, direction) in arms {
                    let hit = raytrace_blocks(&world, start, end).and_then(RayTrace::as_block);
                    if mask & bit != 0 {
                        assert_eq!(
                            hit.map(|hit| hit.pos),
                            Some(pos),
                            "{fence_name}, mask={mask:04b}, {direction} arm must target centre"
                        );
                    } else {
                        assert!(
                            hit.is_none(),
                            "{fence_name}, mask={mask:04b}, {direction} absent arm must not target a block: {hit:?}"
                        );
                    }
                }
            }
        }
    }

    /// `BlockWall.collisionRayTrace` must consume the live one-block selection envelope, not the
    /// 1.5-block collision box or the isolated registry post.  The first control connects the
    /// east arm, then removes it and repeats the same ray so a stale neighbour read would remain
    /// visible.  The two straight runs separately exercise Java's unusual 13/16 selection height
    /// and transverse 3/8 narrowing; the stones behind them prove each high/outside probe really
    /// passed through the wall rather than merely matching a returned helper value.
    #[test]
    fn wall_raytrace_uses_live_connections_removal_and_straight_run_bounds() {
        let pos = BlockPos::new(0, 64, 0);
        let wall = "minecraft:cobblestone_wall";

        let east = pos.east();
        let behind_z = BlockPos::new(0, 64, 2);
        let mut world = world_of(&[(pos, wall), (east, wall), (behind_z, "minecraft:stone")]);
        let state = world.block_at(pos);
        assert_eq!(
            selection_bounds(&world, pos, state),
            Aabb::new(DVec3::new(0.25, 0.0, 0.25), DVec3::new(1.0, 1.0, 0.75)),
            "the east neighbour must extend only the live wall selection envelope"
        );

        let start = DVec3::new(0.9, 64.5, -1.0);
        let end = DVec3::new(0.9, 64.5, 3.0);
        let hit = raytrace_blocks(&world, start, end)
            .and_then(RayTrace::as_block)
            .expect("the connected east arm is targetable");
        assert_eq!(hit.pos, pos);

        world.set_block(east, BlockState::AIR);
        assert_eq!(
            selection_bounds(&world, pos, world.block_at(pos)),
            Aabb::new(DVec3::new(0.25, 0.0, 0.25), DVec3::new(0.75, 1.0, 0.75)),
            "removing the neighbour must refresh the wall to its isolated post"
        );
        let hit = raytrace_blocks(&world, start, end)
            .and_then(RayTrace::as_block)
            .expect("the removal control must reach the stone behind the absent arm");
        assert_eq!(hit.pos, behind_z);

        let north_south = world_of(&[
            (pos, wall),
            (pos.north(), wall),
            (pos.south(), wall),
            (BlockPos::new(2, 64, 0), "minecraft:stone"),
        ]);
        assert_eq!(
            selection_bounds(&north_south, pos, north_south.block_at(pos)),
            Aabb::new(
                DVec3::new(0.3125, 0.0, 0.0),
                DVec3::new(0.6875, 0.8125, 1.0)
            ),
            "north/south straight runs lower and narrow the selection envelope"
        );
        let hit = raytrace_blocks(
            &north_south,
            DVec3::new(-1.0, 64.9, 0.5),
            DVec3::new(3.0, 64.9, 0.5),
        )
        .and_then(RayTrace::as_block)
        .expect("the high probe must pass through the lower straight-run wall");
        assert_eq!(hit.pos, BlockPos::new(2, 64, 0));

        let west_east = world_of(&[
            (pos, wall),
            (pos.west(), wall),
            (pos.east(), wall),
            (BlockPos::new(0, 64, 2), "minecraft:stone"),
        ]);
        assert_eq!(
            selection_bounds(&west_east, pos, west_east.block_at(pos)),
            Aabb::new(
                DVec3::new(0.0, 0.0, 0.3125),
                DVec3::new(1.0, 0.8125, 0.6875)
            ),
            "west/east straight runs lower and narrow the selection envelope"
        );
        let hit = raytrace_blocks(
            &west_east,
            DVec3::new(0.5, 64.9, -1.0),
            DVec3::new(0.5, 64.9, 3.0),
        )
        .and_then(RayTrace::as_block)
        .expect("the high probe must pass through the lower straight-run wall");
        assert_eq!(hit.pos, BlockPos::new(0, 64, 2));
    }

    /// `BlockPane.collisionRayTrace` uses the one connected envelope from
    /// `setBlockBoundsBasedOnState`, not the separate bars emitted by
    /// `addCollisionBoxesToList`.  The two probes below deliberately land in their divergent
    /// regions.  Replacing the envelope with a stale full cube, the isolated registry cross, or
    /// the live collision bar makes at least one assertion target the wrong block.
    ///
    /// The second pass removes the east neighbour and repeats the outside-arm probe.  Java reads
    /// the four neighbours on every call, so the same ray must change from the stone behind the
    /// pane to the now-unconnected pane itself.
    #[test]
    fn pane_raytrace_uses_live_selection_bounds_and_refreshes_after_neighbour_removal() {
        let pane = BlockPos::new(0, 64, 0);
        let east = pane.east();
        let behind_z = BlockPos::new(0, 64, 2);
        let behind_x = BlockPos::new(2, 64, 0);
        let selection_probe_start = DVec3::new(0.46, 64.5, -1.0);
        let selection_probe_end = DVec3::new(0.46, 64.5, 3.0);
        let outside_arm_start = DVec3::new(-1.0, 64.5, 0.8);
        let outside_arm_end = DVec3::new(3.0, 64.5, 0.8);
        let mask = Aabb::from_bounds(-1.0, 63.0, -1.0, 3.0, 66.0, 3.0);

        for centre_name in ["minecraft:glass_pane", "minecraft:stained_glass_pane"] {
            let mut world = world_of(&[
                (pane, centre_name),
                (east, "minecraft:glass_pane"),
                (behind_z, "minecraft:stone"),
                (behind_x, "minecraft:stone"),
            ]);
            let state = world.block_at(pane);

            assert_eq!(
                selection_bounds(&world, pane, state),
                Aabb::new(
                    DVec3::new(0.4375, 0.0, 0.4375),
                    DVec3::new(1.0, 1.0, 0.5625)
                ),
                "east-connected {centre_name} must use Java's live selection envelope"
            );

            let collision_boxes = mc_world::collision::pane_collision_boxes_at(&world, pane, &mask)
                .expect("pane collision oracle must recognise the centre");
            assert_eq!(
                collision_boxes,
                vec![Aabb::from_bounds(0.5, 64.0, 0.4375, 1.0, 65.0, 0.5625)],
                "east-connected {centre_name} has one X collision rail, not the selection box"
            );
            let collision_local = Aabb {
                min: collision_boxes[0].min - pane.to_vec3(),
                max: collision_boxes[0].max - pane.to_vec3(),
            };
            assert!(
                collision_ray_trace(
                    collision_local,
                    pane,
                    selection_probe_start,
                    selection_probe_end
                )
                .is_none(),
                "the selection-only probe must miss Java's live collision rail"
            );

            let hit = raytrace_blocks(&world, selection_probe_start, selection_probe_end)
                .and_then(RayTrace::as_block)
                .expect("Block.collisionRayTrace must use the connected selection envelope");
            assert_eq!(hit.pos, pane, "selection-only probe for {centre_name}");

            let hit = raytrace_blocks(&world, outside_arm_start, outside_arm_end)
                .and_then(RayTrace::as_block)
                .expect("the stone behind the disconnected selection arm");
            assert_eq!(
                hit.pos, behind_x,
                "east-connected {centre_name} must not use an isolated full-cross collision shape"
            );

            world.set_block(east, BlockState::AIR);
            assert_eq!(
                selection_bounds(&world, pane, world.block_at(pane)),
                Aabb::new(DVec3::ZERO, DVec3::ONE),
                "removing the neighbour must refresh the pane envelope to Java's no-neighbour cube"
            );
            let hit = raytrace_blocks(&world, outside_arm_start, outside_arm_end)
                .and_then(RayTrace::as_block)
                .expect("the removed neighbour must make the pane's full selection targetable");
            assert_eq!(hit.pos, pane, "removal control for {centre_name}");
        }
    }

    /// `Item.getMovingObjectPositionFromPlayer(..., true)` must stop at a level-0 liquid, while
    /// the crosshair's `Entity.rayTrace(..., false, false, true)` keeps walking through it. The
    /// same sample also exercises the explicit no-last-MISS overload used by item bodies.
    #[test]
    fn raytrace_options_distinguish_liquid_hits_and_last_misses() {
        let water = BlockPos::new(0, 64, 3);
        let behind = BlockPos::new(0, 64, 5);
        let world = world_of(&[(water, "minecraft:water"), (behind, "minecraft:stone")]);
        let start = DVec3::new(0.5, 64.5, 0.5);
        let end = DVec3::new(0.5, 64.5, 8.0);

        assert_eq!(
            raytrace_blocks(&world, start, end)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(behind),
            "the crosshair ray ignores source water"
        );
        assert_eq!(
            raytrace_blocks_with_options(&world, start, end, true, false, false)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(water),
            "useLiquids=true stops on the water source"
        );

        let no_hit_end = DVec3::new(0.5, 64.5, 2.5);
        assert!(
            raytrace_blocks_with_options(&world, start, no_hit_end, true, false, false).is_none(),
            "the item overload must not return the crosshair's last MISS geometry"
        );
        assert!(
            raytrace_blocks_with_options(&world, start, no_hit_end, true, false, true).is_some(),
            "the negative control must reach returnLastUncollidableBlock"
        );
    }

    /// `ignoreBlockWithoutBoundingBox=true` skips a selection-only plant even though its base
    /// `canCollideCheck` answer is true. With the flag false the same ray stops on the plant,
    /// giving a negative control that demonstrably reaches the divergent branch.
    #[test]
    fn raytrace_options_can_ignore_a_block_without_a_collision_box() {
        let plant = BlockPos::new(0, 64, 3);
        let behind = BlockPos::new(0, 64, 5);
        let world = world_of(&[(plant, "minecraft:tallgrass"), (behind, "minecraft:stone")]);
        let start = DVec3::new(0.5, 64.5, 0.5);
        let end = DVec3::new(0.5, 64.5, 8.0);
        let state = world.block_at(plant);
        assert!(
            can_collide_check(state, false),
            "plant uses Block's base canCollideCheck"
        );
        assert!(
            state.props().collision.is_empty(),
            "the fixture must have Java's null collision bounding box"
        );
        assert_eq!(
            raytrace_blocks_with_options(&world, start, end, false, false, false)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(plant),
            "without the ignore flag the selection-only plant is targetable"
        );
        assert_eq!(
            raytrace_blocks_with_options(&world, start, end, false, true, false)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(behind),
            "the ignore-no-box overload must pass through the plant"
        );
    }

    /// java: `Block.canCollideCheck` -> `isCollidable()`, whose one override is
    /// `BlockFire:353` -> `false`. Fire is not air and is not a liquid, so the old
    /// "not air and not a fluid" rule targeted it and the crosshair dug the flames.
    /// L7 (L8's finding): `BlockLiquid.canCollideCheck` is
    /// `hitIfLiquid && LEVEL == 0`, and `LEVEL` is the whole 0..15 meta. Falling water (level 8)
    /// is not a source, so a liquid ray (bucket, bottle, lily pad, boat) passes through it to the
    /// source behind. The old `meta & 7 == 0` stopped on the falling column (negative control:
    /// level 8's low bits are zero).
    #[test]
    fn a_liquid_ray_passes_falling_water_and_stops_on_a_source() {
        let falling = BlockPos::new(0, 64, 3);
        let source = BlockPos::new(0, 64, 5);
        let mut world = world_of(&[(falling, "minecraft:stone"), (source, "minecraft:water")]);
        world.set_block(falling, BlockState::new(mc_data::BlockId(9), 8));
        assert_eq!(
            world.block_at(falling).meta() & 7,
            0,
            "the fixture is the & 7 trap"
        );
        assert_eq!(world.block_at(source).meta(), 0);
        assert!(!can_collide_check(world.block_at(falling), true));
        assert!(can_collide_check(world.block_at(source), true));
        assert!(!can_collide_check(world.block_at(source), false));
        for level in 1..16 {
            let flowing = BlockState::new(mc_data::BlockId(8), level);
            assert!(!can_collide_check(flowing, true), "flowing level {level}");
        }
        let start = DVec3::new(0.5, 64.5, 0.5);
        let end = DVec3::new(0.5, 64.5, 8.0);
        assert_eq!(
            raytrace_blocks_with_options(&world, start, end, true, false, false)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(source),
            "the liquid ray must pass the falling column and stop on the source"
        );
    }

    #[test]
    fn fire_is_the_one_non_air_non_liquid_block_the_ray_walks_through() {
        let fire = BlockPos::new(0, 64, 3);
        let behind = BlockPos::new(0, 64, 5);
        let world = world_of(&[(fire, "minecraft:fire"), (behind, "minecraft:stone")]);
        assert_eq!(
            world.block_at(fire).props().material,
            mc_data::Material::Fire,
            "the fixture must really be fire, or this proves nothing"
        );
        assert!(!can_collide_check(world.block_at(fire), false));
        // The negative control: a block whose material is not fire still stops the ray.
        assert!(can_collide_check(world.block_at(behind), false));
        let hit = raytrace_blocks(
            &world,
            DVec3::new(0.5, 64.5, 0.5),
            DVec3::new(0.5, 64.5, 8.0),
        )
        .and_then(RayTrace::as_block)
        .expect("the stone behind the fire");
        assert_eq!(hit.pos, behind);
    }

    /// java: `Block.collisionRayTrace` reads raw `minX..maxZ` fields rather than constructing an
    /// `AxisAlignedBB`. An unattached metadata-0 vine is the shipped state whose bounds are
    /// inverted (`1..0`) and therefore has no ray hit; sorting those corners makes it a full cube.
    #[test]
    fn an_unattached_vines_inverted_bounds_are_not_normalised_into_a_full_cube() {
        let vine = BlockPos::new(0, 64, 3);
        let behind = BlockPos::new(0, 64, 5);
        let world = world_of(&[(vine, "minecraft:vine"), (behind, "minecraft:stone")]);
        let bounds = selection_bounds(&world, vine, world.block_at(vine));
        assert!(
            bounds.min.x > bounds.max.x
                && bounds.min.y > bounds.max.y
                && bounds.min.z > bounds.max.z,
            "negative control: the fixture must retain metadata-0 vine's inverted bounds: {bounds:?}"
        );
        let hit = raytrace_blocks(
            &world,
            DVec3::new(0.5, 64.5, 0.5),
            DVec3::new(0.5, 64.5, 8.0),
        )
        .and_then(RayTrace::as_block)
        .expect("the stone behind the untargetable vine");
        assert_eq!(hit.pos, behind);
    }

    /// `BlockPistonMoving.collisionRayTrace` is an unconditional `null`, independent of the
    /// moving tile entity's separate collision box.  A registry-sized block 36 would otherwise
    /// intercept this ray before the stone behind it, proving the test exercises dispatch rather
    /// than merely an empty/unloaded-cell path.
    #[test]
    fn a_moving_piston_carrier_is_not_raytraceable() {
        let carrier = BlockPos::new(0, 64, 3);
        let behind = BlockPos::new(0, 64, 5);
        let start = DVec3::new(0.5, 64.5, 0.5);
        let end = DVec3::new(0.5, 64.5, 8.0);

        // Positive control: an ordinary piston uses Block.collisionRayTrace and stops the same
        // ray at the carrier.  This prevents the moving-piston exception from widening to the
        // whole piston material/class family.
        let ordinary = world_of(&[(carrier, "minecraft:piston"), (behind, "minecraft:stone")]);
        let ordinary_state = ordinary.block_at(carrier);
        assert_eq!(
            ordinary_state.props().block.0,
            33,
            "fixture must be BlockPistonBase"
        );
        assert!(!ordinary_state.props().collision.is_empty());
        assert_eq!(
            raytrace_blocks(&ordinary, start, end)
                .and_then(RayTrace::as_block)
                .map(|hit| hit.pos),
            Some(carrier),
            "an ordinary piston must still intercept the ray"
        );

        let mut world = world_of(&[(behind, "minecraft:stone")]);
        world.set_block(carrier, BlockState::new(mc_data::BlockId(36), 0));

        let state = world.block_at(carrier);
        assert_eq!(
            state.props().block.0,
            36,
            "fixture must be BlockPistonMoving"
        );
        assert!(
            can_collide_check(state, false),
            "the Java null comes from collisionRayTrace, not canCollideCheck"
        );
        assert!(
            selection_bounds(&world, carrier, state).min.x
                < selection_bounds(&world, carrier, state).max.x,
            "negative control: static extracted bounds would be raytraceable without the override"
        );

        let hit = raytrace_blocks(&world, start, end)
            .and_then(RayTrace::as_block)
            .expect("the stone behind the untargetable carrier");
        assert_eq!(hit.pos, behind);
    }

    /// `BlockStairs.collisionRayTrace` masks two of eight half-cubes for every metadata state.
    /// These rays stay in one masked row all the way through the block, so a full-cube selection
    /// box would falsely target the stair before the stone behind it.  This covers all four
    /// horizontal facings and both halves; the normal corridor is 0.25..0.75 on the two static
    /// axes and runs from -1 to 3 on the remaining axis.
    #[test]
    fn stairs_raytrace_uses_its_six_half_cube_dispatch_shape() {
        const SKIPPED_BY_STAIR_META: [[usize; 2]; 8] = [
            [4, 6],
            [5, 7],
            [4, 5],
            [6, 7],
            [0, 2],
            [1, 3],
            [0, 1],
            [2, 3],
        ];
        let stair_pos = BlockPos::new(0, 64, 0);
        let stair = mc_data::block_by_name("stone_stairs").expect("stone stairs");

        for meta in 0_u8..8 {
            let [first, second] = SKIPPED_BY_STAIR_META[usize::from(meta)];
            let centre = |pass: usize| {
                DVec3::new(
                    0.25 + 0.5 * f64::from((pass % 2) as u8),
                    0.25 + 0.5 * f64::from(((pass / 4) % 2) as u8),
                    0.25 + 0.5 * f64::from(((pass / 2) % 2) as u8),
                )
            };
            let a = centre(first);
            let b = centre(second);
            let (start, end, behind) = if a.x != b.x {
                (
                    DVec3::new(-1.0, 64.0 + a.y, a.z),
                    DVec3::new(3.0, 64.0 + a.y, a.z),
                    BlockPos::new(2, 64, 0),
                )
            } else {
                (
                    DVec3::new(a.x, 64.0 + a.y, -1.0),
                    DVec3::new(a.x, 64.0 + a.y, 3.0),
                    BlockPos::new(0, 64, 2),
                )
            };

            let mut world = world_with(stair_pos);
            world.set_block(stair_pos, BlockState::new(stair, meta));
            world.set_block(behind, stone());
            let state = world.block_at(stair_pos);
            assert!(
                collision_ray_trace(
                    selection_bounds(&world, stair_pos, state),
                    stair_pos,
                    start,
                    end,
                )
                .is_some(),
                "negative control: the shared selection box must falsely hit stair meta {meta}"
            );
            let hit = raytrace_blocks(&world, start, end)
                .and_then(RayTrace::as_block)
                .expect("the stone behind the missing stair half-cubes");
            assert_eq!(hit.pos, behind, "stair metadata {meta}");
        }
    }

    /// All five `BlockTorch.collisionRayTrace` orientations write a small box before delegating
    /// to `Block`; redstone torches inherit that same body.  The high corner ray is outside all
    /// five boxes but inside the stale unit cube, while the exact bounds assertions cover every
    /// facing-specific axis placement and retain Java's `float` arithmetic.
    #[test]
    fn torch_raytrace_dispatches_the_five_facing_boxes() {
        let f = 0.15_f32;
        // `setBlockBounds` widens float heights too: `0.2F` is not the double 0.2 (checked against the real classes by `javaref/CameraRayRef.java`).
        let (low, high, top) = (f64::from(0.2_f32), f64::from(0.8_f32), f64::from(0.6_f32));
        assert_ne!(low, 0.2, "the float height is not the double literal");
        let cases = [
            (
                1,
                DVec3::new(0.0, low, f64::from(0.5_f32 - f)),
                DVec3::new(f64::from(f * 2.0), high, f64::from(0.5_f32 + f)),
            ),
            (
                2,
                DVec3::new(f64::from(1.0_f32 - f * 2.0), low, f64::from(0.5_f32 - f)),
                DVec3::new(1.0, high, f64::from(0.5_f32 + f)),
            ),
            (
                3,
                DVec3::new(f64::from(0.5_f32 - f), low, 0.0),
                DVec3::new(f64::from(0.5_f32 + f), high, f64::from(f * 2.0)),
            ),
            (
                4,
                DVec3::new(f64::from(0.5_f32 - f), low, f64::from(1.0_f32 - f * 2.0)),
                DVec3::new(f64::from(0.5_f32 + f), high, 1.0),
            ),
            (
                5,
                DVec3::new(
                    f64::from(0.5_f32 - 0.1_f32),
                    0.0,
                    f64::from(0.5_f32 - 0.1_f32),
                ),
                DVec3::new(
                    f64::from(0.5_f32 + 0.1_f32),
                    top,
                    f64::from(0.5_f32 + 0.1_f32),
                ),
            ),
        ];
        let torch_pos = BlockPos::new(0, 64, 0);
        let behind = BlockPos::new(0, 64, 2);
        let torch = mc_data::block_by_name("torch").expect("torch");

        for (meta, min, max) in cases {
            assert_eq!(
                torch_raytrace_bounds(meta),
                Aabb::new(min, max),
                "meta {meta}"
            );
            let mut world = world_with(torch_pos);
            world.set_block(torch_pos, BlockState::new(torch, meta));
            world.set_block(behind, stone());
            let state = world.block_at(torch_pos);
            let start = DVec3::new(0.9, 64.9, -1.0);
            let end = DVec3::new(0.9, 64.9, 3.0);
            let registry = state.props().bounds;
            let registry = Aabb::new(
                DVec3::new(registry.min_x, registry.min_y, registry.min_z),
                DVec3::new(registry.max_x, registry.max_y, registry.max_z),
            );
            assert!(
                collision_ray_trace(registry, torch_pos, start, end).is_some(),
                "negative control: the registry's unit cube must falsely hit torch meta {meta}"
            );
            // The selection (outline) bounds are the torch box too now.
            assert_eq!(
                selection_bounds(&world, torch_pos, state),
                Aabb::new(min, max),
                "selection bounds for meta {meta}"
            );
            let hit = raytrace_blocks(&world, start, end)
                .and_then(RayTrace::as_block)
                .expect("the stone behind the torch");
            assert_eq!(hit.pos, behind, "torch metadata {meta}");
        }
    }

    #[test]
    fn a_straight_ray_hits_the_near_face_of_a_block() {
        let target = BlockPos::new(4, 64, 0);
        let world = world_with(target);
        let hit = raytrace_blocks(
            &world,
            DVec3::new(0.5, 64.5, 0.5),
            DVec3::new(8.0, 64.5, 0.5),
        )
        .and_then(RayTrace::as_block)
        .expect("the ray must find the block");
        assert_eq!(hit.pos, target);
        assert_eq!(hit.face, Facing::West, "entered through the -X face");
        assert!((hit.hit_vec.x - 4.0).abs() < 1e-9, "{:?}", hit.hit_vec);
        // `pos.offset(sideHit)` is deliberately an operation at the ItemBlock caller, not
        // cached in MovingObjectPosition. Keep the two Java-owned inputs so placement prediction
        // can make that offset when ItemBlock is ported.
        assert_eq!(hit.pos.relative(hit.face), BlockPos::new(3, 64, 0));
    }

    /// java: `Minecraft.rightClickMouse` casts the relative hit coordinates to `float` before
    /// the packet constructor multiplies by 16. The representable `f64` immediately below 1/16
    /// therefore rounds up to 1/16 and encodes as byte 1; retaining the double encodes byte 0.
    #[test]
    fn block_hit_cursor_narrows_to_float_before_sixteenths_quantisation() {
        let boundary = 0.0625_f64;
        let immediately_below = f64::from_bits(boundary.to_bits() - 1);
        let hit = BlockHit {
            pos: BlockPos::new(0, 0, 0),
            face: Facing::West,
            hit_vec: DVec3::new(immediately_below, 0.0, 0.0),
        };
        let narrowed = hit.cursor().x;
        assert_eq!(narrowed, boundary);
        assert_eq!((narrowed * 16.0) as i32, 1);
        assert_eq!(
            (immediately_below * 16.0) as i32,
            0,
            "negative control: keeping the subtraction in f64 must change the packet byte"
        );
    }

    /// The face decides where a placed block goes, so getting it wrong builds into the wall.
    #[test]
    fn each_axis_reports_the_face_the_ray_entered_through() {
        for (target, from, face) in [
            (
                BlockPos::new(0, 64, 0),
                DVec3::new(0.5, 70.0, 0.5),
                Facing::Up,
            ),
            (
                BlockPos::new(0, 64, 0),
                DVec3::new(0.5, 60.0, 0.5),
                Facing::Down,
            ),
            (
                BlockPos::new(0, 64, 4),
                DVec3::new(0.5, 64.5, 0.5),
                Facing::North,
            ),
            (
                BlockPos::new(0, 64, -4),
                DVec3::new(0.5, 64.5, 0.5),
                Facing::South,
            ),
        ] {
            let world = world_with(target);
            let centre = target.to_vec3() + DVec3::new(0.5, 0.5, 0.5);
            let direction = (centre - from).normalize();
            let hit = raytrace_blocks(&world, from, from + direction * 16.0)
                .and_then(RayTrace::as_block)
                .unwrap_or_else(|| panic!("no hit from {from:?} towards {target:?}"));
            assert_eq!(hit.pos, target, "from {from:?}");
            assert_eq!(hit.face, face, "from {from:?}");
        }
    }

    /// java: with `returnLastUncollidableBlock = true` the trace still *returns* something —
    /// but it is a `MISS`, and reading it as a block is what would make the client dig air.
    #[test]
    fn air_and_unloaded_chunks_produce_a_miss_and_never_a_block() {
        let world = World::new();
        let trace = raytrace_blocks(
            &world,
            DVec3::new(0.5, 64.5, 0.5),
            DVec3::new(0.5, 64.5, 40.5),
        );
        assert!(matches!(trace, Some(RayTrace::Miss(_))));
        assert_eq!(trace.and_then(RayTrace::as_block), None);
        assert_eq!(
            pick(
                &world,
                DVec3::new(0.5, 64.5, 0.5),
                DVec3::new(0.0, 0.0, 1.0),
                4.5,
                false,
                []
            ),
            HitResult::Miss
        );
    }

    /// java: the loop runs at most 200 iterations, so a very long ray gives up rather than
    /// hanging. The reach clamp means a real pick never gets close, which is exactly why an
    /// unbounded loop here would never be noticed until it was.
    #[test]
    fn the_walk_is_bounded_at_two_hundred_steps() {
        let world = World::new();
        let start = DVec3::new(0.5, 64.5, 0.5);
        let trace = raytrace_blocks(&world, start, start + DVec3::new(10_000.0, 0.0, 0.0));
        // It gives up rather than hanging, and what it gives up with is never a block.
        assert_eq!(trace.and_then(RayTrace::as_block), None);
        let travelled = trace.map_or(0.0, |t| (t.hit().hit_vec - start).length());
        assert!(
            travelled < 210.0,
            "the walk ran past its 200-step budget: {travelled}"
        );
    }

    /// java: `Entity.rayTrace` starts at the eyes, so a block at the player's feet is *behind*
    /// the ray when they look forward.
    #[test]
    fn a_ray_that_starts_inside_a_block_hits_it_immediately() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = raytrace_blocks(
            &world,
            DVec3::new(0.5, 64.5, 0.5),
            DVec3::new(0.5, 64.5, 4.5),
        )
        .and_then(RayTrace::as_block)
        .expect("standing inside the block hits it");
        assert_eq!(hit.pos, target);
    }

    #[test]
    fn the_look_vector_matches_vanillas_axes() {
        // Yaw 0 faces +Z, yaw 90 faces -X, pitch -90 faces straight up.
        let south = look_vector(0.0, 0.0);
        assert!(
            south.z > 0.99 && south.x.abs() < 1e-6 && south.y.abs() < 1e-6,
            "{south:?}"
        );
        let west = look_vector(90.0, 0.0);
        assert!(west.x < -0.99 && west.z.abs() < 1e-6, "{west:?}");
        let up = look_vector(0.0, -90.0);
        assert!(up.y > 0.99, "{up:?}");
        let down = look_vector(0.0, 90.0);
        assert!(down.y < -0.99, "{down:?}");
    }

    /// java: every `Vec3` constructor rewrites `-0.0` to `+0.0`. Looking straight down makes
    /// the sine-table formula produce signed zero for many ordinary yaw values, so this is a
    /// live constructor rule rather than an artificial bit-pattern test.
    #[test]
    fn look_vectors_normalise_signed_zero_like_the_vec3_constructor() {
        let mut saw_zero = false;
        for yaw in -180..=180 {
            let look = look_vector(yaw as f32, 90.0);
            for component in look.to_array() {
                if component == 0.0 {
                    saw_zero = true;
                    assert!(!component.is_sign_negative(), "yaw {yaw}: {look:?}");
                }
            }
        }
        assert!(saw_zero, "the pitch-90 sample must exercise the zero fold");
    }

    /// A player-shaped candidate that the `canBeCollidedWith` filter has already passed.
    fn pickable(network_id: i32, pos: DVec3) -> Pickable {
        Pickable {
            network_id,
            bounds: size_bounding_box(pos, 0.6, 1.8),
            border_size: 0.1,
            collidable: true,
        }
    }

    #[test]
    fn an_entity_in_front_of_a_block_wins_the_pick() {
        let target = BlockPos::new(0, 64, 5);
        let world = world_with(target);
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        // With no entity in the way the block is what the crosshair reports.
        assert_eq!(
            pick(&world, eye, look, 5.0, true, [])
                .block()
                .map(|b| b.pos),
            Some(target)
        );

        let near = pickable(7, DVec3::new(0.5, 64.0, 3.0));
        let hit = pick(&world, eye, look, 5.0, true, [near]);
        let entity = hit.entity().expect("the nearer entity must win");
        assert_eq!(entity.network_id, 7);
        assert_eq!(
            entity.hit_vec,
            DVec3::new(
                0.5,
                64.5,
                3.0 - f64::from(0.6_f32) / 2.0 - f64::from(0.1_f32),
            )
        );
        // The selection distance stays local to EntityRenderer.getMouseOver (`d2`); Java's
        // MovingObjectPosition stores the entity and hitVec, never that temporary scalar. The
        // stored hitVec is the load-bearing data for the later INTERACT_AT packet.

        // java: `d1 = objectMouseOver.hitVec.distanceTo(vec3)` — an entity *behind* the block
        // the ray already stopped at loses, even though it is within the entity reach.
        let behind = Pickable {
            bounds: size_bounding_box(DVec3::new(0.5, 64.0, 5.6), 0.6, 1.8),
            ..near
        };
        let hit = pick(&world, eye, look, 5.0, true, [behind]);
        assert_eq!(hit.block().map(|b| b.pos), Some(target));
    }

    /// java: `Vec3.distanceTo` returns a `double`, but its square root goes through
    /// `MathHelper.sqrt_double`, whose return type is `float`. Consequently these two distances
    /// tie at 4.5 in vanilla and the strict `<` keeps the block. Full-f64 distance picks entity 7.
    #[test]
    fn float_rounded_distance_ties_keep_the_block() {
        let target = BlockPos::new(0, 64, 5);
        let world = world_with(target);
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        let candidate = |front: f64| Pickable {
            network_id: 7,
            bounds: Aabb::from_bounds(0.0, 64.0, front, 1.0, 65.0, 5.5),
            border_size: 0.0,
            collidable: true,
        };

        let float_tie = candidate(5.0 - 1.0e-7);
        assert_eq!(
            pick(&world, eye, look, 5.0, true, [float_tie])
                .block()
                .map(|hit| hit.pos),
            Some(target)
        );
        let clearly_closer = candidate(5.0 - 1.0e-5);
        assert_eq!(
            pick(&world, eye, look, 5.0, true, [clearly_closer])
                .entity()
                .map(|hit| hit.network_id),
            Some(7),
            "negative control: a difference wider than one f32 rounding bin must still pick the entity"
        );
    }

    /// The same float round-trip is load-bearing at survival's post-hoc three-block rejection.
    /// 3.00000005 rounds to exactly 3.0 and remains hittable; a visibly larger excess is a miss.
    #[test]
    fn survival_reach_clamp_uses_vanillas_float_rounded_distance() {
        let world = World::new();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        let candidate = |distance: f64| Pickable {
            network_id: 7,
            bounds: Aabb::from_bounds(
                0.0,
                64.0,
                eye.z + distance,
                1.0,
                65.0,
                eye.z + distance + 0.5,
            ),
            border_size: 0.0,
            collidable: true,
        };

        assert_eq!(
            pick(&world, eye, look, 4.5, false, [candidate(3.0 + 5.0e-8)])
                .entity()
                .map(|hit| hit.network_id),
            Some(7)
        );
        assert_eq!(
            pick(&world, eye, look, 4.5, false, [candidate(3.0 + 1.0e-5)]),
            HitResult::Miss,
            "negative control: a distance outside the float tie must still be rejected"
        );
    }

    /// java: an eye inside an expanded entity box assigns `d2 = 0` and wins immediately, even
    /// over an earlier candidate whose front face is closer than the containing box's exit face.
    /// Treating the exit intercept as an ordinary distance leaves entity 1 selected instead.
    #[test]
    fn a_later_entity_whose_box_contains_the_eye_wins_immediately() {
        let world = World::new();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        let nearer_intercept = Pickable {
            network_id: 1,
            bounds: Aabb::from_bounds(0.0, 64.0, 0.75, 1.0, 65.0, 1.0),
            border_size: 0.0,
            collidable: true,
        };
        let containing = Pickable {
            network_id: 2,
            bounds: Aabb::from_bounds(0.0, 64.0, 0.0, 1.0, 65.0, 2.0),
            border_size: 0.0,
            collidable: true,
        };
        assert_eq!(
            pick(
                &world,
                eye,
                look,
                4.5,
                false,
                [nearer_intercept, containing]
            )
            .entity()
            .map(|hit| hit.network_id),
            Some(2)
        );
    }

    /// java: `EntityRenderer.getMouseOver`'s `entity1 == entity.ridingEntity` fork. A mount whose
    /// ordinary front-face intercept would win is ignored, while the eye-inside branch remains
    /// selectable exactly as the Java loop orders the checks.
    #[test]
    fn the_ridden_vehicle_is_skipped_unless_its_box_contains_the_eye() {
        let world = World::new();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        let vehicle = Pickable {
            network_id: 42,
            bounds: Aabb::from_bounds(0.0, 64.0, 2.0, 1.0, 65.0, 3.0),
            border_size: 0.0,
            collidable: true,
        };

        assert_eq!(
            pick_with_riding(&world, eye, look, 4.5, false, None, [vehicle])
                .entity()
                .map(|hit| hit.network_id),
            Some(42),
            "negative control: the same entity is targetable while not riding it"
        );
        assert_eq!(
            pick_with_riding(&world, eye, look, 4.5, false, Some(42), [vehicle]),
            HitResult::Miss
        );

        let containing_vehicle = Pickable {
            bounds: Aabb::from_bounds(0.0, 64.0, 0.0, 1.0, 65.0, 1.0),
            ..vehicle
        };
        assert_eq!(
            pick_with_riding(
                &world,
                eye,
                look,
                4.5,
                false,
                Some(42),
                [containing_vehicle]
            )
            .entity()
            .map(|hit| hit.network_id),
            Some(42),
            "the eye-inside branch precedes the riding exception"
        );
    }

    /// java: `if (pointedEntity != null && flag && vec3.distanceTo(vec33) > 3.0D)`.
    ///
    /// The survival 3-block limit is **not** a shortened ray. Vanilla runs the entity ray the
    /// full 4.5 blocks, picks the entity, and then throws away *both* the entity and the block
    /// hit behind it. Shortening the ray instead leaves the block hit standing, which is a `C07`
    /// vanilla never sends.
    #[test]
    fn the_survival_three_block_limit_discards_the_block_hit_too() {
        let world = World::new();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        // Front face at z = 3.6, minus the 0.1 border expansion: 3.0 blocks from the eye, just
        // inside the limit.
        let inside = pickable(7, DVec3::new(0.5, 64.0, 3.9));
        assert!(pick(&world, eye, look, 4.5, false, [inside])
            .entity()
            .is_some());

        let far = pickable(7, DVec3::new(0.5, 64.0, 4.2));
        assert_eq!(pick(&world, eye, look, 4.5, false, [far]), HitResult::Miss);
        // Creative reaches 6.0 and applies no post-hoc limit.
        assert!(pick(&world, eye, look, 5.0, true, [far]).entity().is_some());

        // The block behind the rejected entity is discarded with it: the crosshair reports a
        // MISS, so `clickMouse` takes the `MISS` arm and sends nothing.
        let world = world_with(BlockPos::new(0, 64, 5));
        assert_eq!(pick(&world, eye, look, 4.5, false, [far]), HitResult::Miss);
    }

    /// java: the `canBeCollidedWith` table in `can_be_collided_with`, at the point where
    /// `getMouseOver` applies it. A dropped item in front of a block must leave the *block* as
    /// the hit, because that is the difference between a `C07` dig and a kick.
    #[test]
    fn a_non_collidable_candidate_is_skipped_and_the_block_behind_it_survives() {
        let target = BlockPos::new(0, 64, 5);
        let world = world_with(target);
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        let dropped = Pickable {
            collidable: false,
            ..pickable(7, DVec3::new(0.5, 64.0, 3.0))
        };
        assert_eq!(
            pick(&world, eye, look, 5.0, true, [dropped])
                .block()
                .map(|b| b.pos),
            Some(target)
        );
    }

    /// **`F-P1-entity.item-06-2`.** java: `EntityHanging.updateBoundingBox` (`:53-90`) replaces
    /// `setSize(0.5F, 0.5F)`'s box outright. For a 12x12 item frame the result is a
    /// `0.75 x 0.75 x 0.0625` plate **centred** on the entity and flush with its wall, not a
    /// half-block cube resting on it.
    ///
    /// Worked example from the report, for a NORTH-facing frame in block `(0, 64, 5)`: the
    /// entity sits at `(0.5, 64.5, 5.96875)` (block centre backed `0.46875` off the wall, no
    /// `func_174858_a` offset because `12 % 32 != 0`).
    #[test]
    fn an_item_frames_box_is_a_plate_flush_with_its_wall_not_a_half_block_cube() {
        let pos = DVec3::new(0.5, 64.5, 5.968_75);
        let plate = hanging_bounding_box(pos, Facing::North, 12, 12);
        for (got, want) in [
            (plate.min.x, 0.125),
            (plate.max.x, 0.875),
            (plate.min.y, 64.125),
            (plate.max.y, 64.875),
            (plate.min.z, 5.937_5),
            (plate.max.z, 6.0),
        ] {
            assert!((got - want).abs() < 1e-9, "{got} != {want} in {plate:?}");
        }

        // The box this replaces, and every way it was wrong: 0.21875 of it stuck out through
        // the supporting block, it was a quarter-block short in x, and it sat *above* the
        // entity instead of straddling it.
        let cube = size_bounding_box(pos, 0.5, 0.5);
        assert!(cube.max.z > 6.0, "the old box poked through the wall");
        assert!(cube.min.x > plate.min.x && cube.max.x < plate.max.x);
        assert!(
            (cube.min.y - pos.y).abs() < 1e-9 && plate.min.y < pos.y,
            "the old box rested on the entity; the real one straddles it"
        );

        // The thin axis follows the facing. An EAST-facing frame is thin in x.
        let east = hanging_bounding_box(pos, Facing::East, 12, 12);
        assert!((east.max.x - east.min.x - 0.0625).abs() < 1e-9, "{east:?}");
        assert!((east.max.z - east.min.z - 0.75).abs() < 1e-9, "{east:?}");
        // ...and a SOUTH-facing one is thin in z, like north: the axis, not the sign, decides.
        let south = hanging_bounding_box(pos, Facing::South, 12, 12);
        assert!(
            (south.max.z - south.min.z - 0.0625).abs() < 1e-9,
            "{south:?}"
        );

        // Negative control for the pixel arithmetic: a 4x4 painting (64x48 texels) is four
        // blocks wide and three tall, so the same function has to scale, not clamp.
        let art = hanging_bounding_box(pos, Facing::North, 64, 48);
        assert!((art.max.x - art.min.x - 4.0).abs() < 1e-9, "{art:?}");
        assert!((art.max.y - art.min.y - 3.0).abs() < 1e-9, "{art:?}");
        assert!((art.max.z - art.min.z - 0.0625).abs() < 1e-9, "{art:?}");
    }

    /// java: `EntityLeashKnot` is the hanging entity that is not hung — its
    /// `updateFacingWithBoundingBox` override (`:38-40`) is empty, so `facingDirection` stays
    /// null, `updateBoundingBox`'s `if (this.facingDirection != null)` never fires, and the box
    /// is the literal one the constructor wrote. Feeding it through
    /// [`hanging_bounding_box`] with its `getWidthPixels()` of `9` would be the plausible wrong
    /// answer, so this pins the difference.
    #[test]
    fn a_leash_knot_keeps_the_box_its_constructor_wrote() {
        let pos = DVec3::new(4.5, 64.5, 9.5);
        let knot = leash_knot_bounding_box(pos);
        for (got, want) in [
            (knot.min.x, 4.312_5),
            (knot.max.x, 4.687_5),
            (knot.min.y, 64.375),
            (knot.max.y, 64.875),
            (knot.min.z, 9.312_5),
            (knot.max.z, 9.687_5),
        ] {
            assert!((got - want).abs() < 1e-9, "{got} != {want} in {knot:?}");
        }
        let wrong = hanging_bounding_box(pos, Facing::North, 9, 9);
        assert!(
            knot != wrong,
            "the 9-pixel hanging box is not the knot's box"
        );
    }

    /// java: `getCollisionBorderSize()` — `0.1F` by default, so the pickable box is 0.1 wider on
    /// every side than the entity's own bounding box.
    #[test]
    fn the_collision_border_expands_the_pick_box() {
        let world = World::new();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let look = DVec3::new(0.0, 0.0, 1.0);
        // The ray is horizontal at y = 64.5. Feet at 62.65 put the bare box top at 64.45, just
        // under it; the 0.1 border lifts the top to 64.55, just over. So the expansion is the
        // only thing that can turn this miss into a hit.
        let stand = pickable(7, DVec3::new(0.5, 62.65, 3.0));
        let bare = Pickable {
            border_size: 0.0,
            ..stand
        };
        assert_eq!(pick(&world, eye, look, 4.5, false, [bare]), HitResult::Miss);
        assert!(pick(&world, eye, look, 4.5, false, [stand])
            .entity()
            .is_some());
    }

    /// java: `Entity.canBeCollidedWith` is `return false`, so the predicate is a whitelist. The
    /// three types in `NetHandlerPlayServer.processUseEntity`'s kick list must all be out.
    #[test]
    fn the_kick_list_types_are_never_collidable() {
        use mc_data::EntityType as E;
        let alive = CollisionState::default();
        for kind in [E::Item, E::XPOrb, E::Arrow] {
            assert!(
                !can_be_collided_with(Some(kind), alive),
                "{kind:?} is in processUseEntity's kick list"
            );
        }
    }

    /// **W-SPRINTSTATE.** Boat, every minecart, ender crystals, and large fireballs reach the
    /// common successful-return branch on the remote client. Living entities and armour stands
    /// return false, while hanging objects are consumed by `hitByEntity` before that branch.
    #[test]
    fn only_client_successful_attack_results_cancel_sprint() {
        use mc_data::EntityType as E;
        let carts = [
            E::MinecartCommandBlock,
            E::MinecartRideable,
            E::MinecartChest,
            E::MinecartFurnace,
            E::MinecartTNT,
            E::MinecartHopper,
            E::MinecartSpawner,
        ];
        assert!(attack_cancels_sprint(Some(E::Boat)));
        assert!(carts
            .into_iter()
            .all(|kind| attack_cancels_sprint(Some(kind))));
        assert_eq!(
            local_attack_result(Some(E::EnderCrystal)),
            LocalAttackResult::TargetKnockback
        );
        assert_eq!(
            local_attack_result(Some(E::Fireball)),
            LocalAttackResult::ReflectLargeFireball
        );
        assert!(attack_cancels_sprint(Some(E::EnderCrystal)));
        assert!(attack_cancels_sprint(Some(E::Fireball)));

        // `EntityHanging.hitByEntity` calls `attackEntityFrom(..., 0)` and returns true on the
        // remote world. The enclosing `if (!hitByEntity)` therefore does not run, even for a
        // non-empty frame whose own damage override would also return true.
        for kind in [E::Painting, E::LeashKnot, E::ItemFrame] {
            assert_eq!(
                local_attack_result(Some(kind)),
                LocalAttackResult::NoClientDamage,
                "{kind:?} is consumed by EntityHanging.hitByEntity"
            );
            assert!(!attack_cancels_sprint(Some(kind)));
        }
        // `None` has no EntityList type (an untracked id or a fishing hook). Remote players are
        // classified from their tracked kind: `EntityOtherPlayerMP.attackEntityFrom` returns
        // true, unlike every remote `EntityLivingBase` mob.
        assert!(!attack_cancels_sprint(None));
        assert_eq!(
            tracked_attack_result(crate::sync::TrackedKind::Player),
            LocalAttackResult::RemotePlayer
        );
        assert_eq!(
            tracked_attack_result(crate::sync::TrackedKind::FishingBobber),
            LocalAttackResult::NoClientDamage
        );
        assert_eq!(
            tracked_attack_result(crate::sync::TrackedKind::Type(E::Zombie)),
            LocalAttackResult::NoClientDamage
        );
        assert_eq!(
            tracked_attack_result(crate::sync::TrackedKind::Type(E::Boat)),
            LocalAttackResult::TargetKnockback
        );
        assert!(!attack_cancels_sprint(Some(E::Zombie)));
        assert!(!attack_cancels_sprint(Some(E::ArmorStand)));
        assert!(!attack_cancels_sprint(Some(E::PrimedTnt)));
        assert!(!attack_cancels_sprint(Some(E::SmallFireball)));
        assert!(!attack_cancels_sprint(Some(E::WitherSkull)));
        assert_eq!(
            local_attack_result(Some(E::SmallFireball)),
            LocalAttackResult::NoClientDamage
        );
    }

    /// java: for a player target `EntityPlayer.attackTargetEntityWithCurrentItem` takes `f1`
    /// from `EnchantmentHelper.getModifierForCreature(held, UNDEFINED)` (`:1315-1318`), and its
    /// `ItemStack.hitEntity` reaches `ItemSword` (one point) or `ItemTool` (two points).
    #[test]
    fn remote_player_hits_read_sharpness_and_dispatch_hit_entity_durability() {
        assert_eq!(undefined_creature_damage_modifier(None), 0.0);
        assert_eq!(
            undefined_creature_damage_modifier(Some(&item("diamond_sword"))),
            0.0
        );
        assert_eq!(
            undefined_creature_damage_modifier(Some(&enchanted_item("diamond_sword", 16, 3))),
            3.75
        );
        // Smite and Bane of Arthropods only apply to UNDEAD and ARTHROPOD creatures.
        for id in [17, 18] {
            assert_eq!(
                undefined_creature_damage_modifier(Some(&enchanted_item("diamond_sword", id, 5))),
                0.0
            );
        }
        // `applyEnchantmentModifier` visits every entry, so a negative duplicate cancels the
        // positive level and `f1 > 0` fails, even though `getEnchantmentLevel` would report 2.
        let entry = |id: i16, level: i16| {
            let mut entry = mc_nbt::NbtCompound::new();
            entry.insert("id", mc_nbt::Nbt::Short(id));
            entry.insert("lvl", mc_nbt::Nbt::Short(level));
            mc_nbt::Nbt::Compound(entry)
        };
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("ench", mc_nbt::Nbt::list(vec![entry(16, 2), entry(16, -2)]));
        let cancelled = ItemStack {
            nbt: Some(nbt),
            ..item("diamond_sword")
        };
        assert_eq!(undefined_creature_damage_modifier(Some(&cancelled)), 0.0);

        assert_eq!(hit_entity_item_damage(Some(&item("diamond_sword"))), 1);
        for tool in ["iron_pickaxe", "stone_axe", "wooden_shovel"] {
            assert_eq!(hit_entity_item_damage(Some(&item(tool))), 2, "{tool}");
        }
        assert_eq!(
            hit_entity_item_damage(Some(&item("golden_hoe"))),
            0,
            "ItemHoe extends Item, not ItemTool"
        );
        assert_eq!(hit_entity_item_damage(Some(&item("saddle"))), 0);
        assert_eq!(hit_entity_item_damage(Some(&item("stick"))), 0);
        assert_eq!(hit_entity_item_damage(None), 0);
    }

    /// The rest of the table, each row traceable to a `canBeCollidedWith` body in 1.8.9.
    #[test]
    fn the_can_be_collided_with_table_matches_the_overrides() {
        use mc_data::EntityType as E;
        let alive = CollisionState::default();
        let dead = CollisionState {
            dead: true,
            marker: false,
        };
        let marker = CollisionState {
            dead: false,
            marker: true,
        };

        // `Entity.canBeCollidedWith` -> false, inherited.
        for kind in [
            E::ThrownEgg,
            E::Snowball,
            E::ThrownEnderpearl,
            E::EyeOfEnderSignal,
            E::ThrownPotion,
            E::ThrownExpBottle,
            E::FireworksRocketEntity,
        ] {
            assert!(!can_be_collided_with(Some(kind), alive), "{kind:?}");
        }
        // `EntityHanging:163`, `EntityEnderCrystal:84`, `EntityFireball:292` -> true.
        for kind in [
            E::LeashKnot,
            E::Painting,
            E::ItemFrame,
            E::EnderCrystal,
            E::Fireball,
        ] {
            assert!(can_be_collided_with(Some(kind), alive), "{kind:?}");
            assert!(
                can_be_collided_with(Some(kind), dead),
                "{kind:?} ignores isDead"
            );
        }
        // `EntitySmallFireball:79` / `EntityWitherSkull:124` -> false, even though their shared
        // superclass says true.
        assert!(!can_be_collided_with(Some(E::SmallFireball), alive));
        assert!(!can_be_collided_with(Some(E::WitherSkull), alive));
        // `EntityLivingBase:2211` -> !isDead.
        for kind in [E::Zombie, E::Villager, E::EntityHorse, E::Guardian] {
            assert!(can_be_collided_with(Some(kind), alive), "{kind:?}");
            assert!(!can_be_collided_with(Some(kind), dead), "{kind:?}");
        }
        // A remote player has no `EntityList` row.
        assert!(can_be_collided_with(None, alive));
        assert!(!can_be_collided_with(None, dead));
        // `EntityDragon:748` -> false, despite being an `EntityLiving`.
        assert!(!can_be_collided_with(Some(E::EnderDragon), alive));
        assert!(
            can_be_collided_with(Some(E::WitherBoss), alive),
            "the other boss is normal"
        );
        // `EntityArmorStand:992` -> super && !hasMarker().
        assert!(can_be_collided_with(Some(E::ArmorStand), alive));
        assert!(!can_be_collided_with(Some(E::ArmorStand), marker));
        assert!(!can_be_collided_with(Some(E::ArmorStand), dead));
        // The four `!isDead` non-living overrides.
        for kind in [
            E::Boat,
            E::MinecartRideable,
            E::MinecartHopper,
            E::PrimedTnt,
            E::FallingSand,
        ] {
            assert!(can_be_collided_with(Some(kind), alive), "{kind:?}");
            assert!(!can_be_collided_with(Some(kind), dead), "{kind:?}");
        }
    }

    /// WAVE8RCLICK: `rightClickMouse`'s second entity predicate is not equivalent to blindly
    /// treating every C02 as consumed.  These controls pin the exact object/monster bodies that
    /// can be decided on a client, while the rideable minecart and ordinary passive mob remain
    /// unresolved until their live rider/animal state has an owner.  The pig saddle rows are the
    /// item override's return value, not merely its visible mutation gate: Java returns true for
    /// child and already-saddled pigs too.
    #[test]
    fn entity_interaction_classifier_preserves_known_consumption_and_declines() {
        use mc_data::EntityType as E;

        let survival = EntityInteractionState::default();
        assert_eq!(
            entity_interaction_consumed(Some(E::Boat), None, survival),
            Some(true),
            "EntityBoat.interactFirst returns true even on a remote world"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::Painting), None, survival),
            Some(false),
            "the base Entity.interactFirst declines paintings"
        );
        assert_eq!(
            entity_interaction_consumed(
                Some(E::Creeper),
                Some(&stack("flint_and_steel", 1)),
                survival
            ),
            Some(false),
            "Creeper's client flint body swings but does not consume"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::Zombie), Some(&stack("bread", 1)), survival),
            Some(false),
            "ordinary food has no zombie entity override"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::Sheep), Some(&stack("dye", 1)), survival),
            Some(true),
            "ItemDye returns true even when the sheep needs no visible change"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::Sheep), Some(&stack("shears", 1)), survival),
            Some(false),
            "EntitySheep mutates its client copy but returns super.interact"
        );
        assert_eq!(
            entity_interaction_consumed(
                Some(E::Sheep),
                Some(&stack("shears", 1)),
                EntityInteractionState {
                    sheep_sheared: true,
                    ..survival
                }
            ),
            Some(false),
            "a sheared sheep declines ItemShears"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::Pig), Some(&stack("saddle", 1)), survival),
            Some(true),
            "an adult unsaddled pig accepts ItemSaddle"
        );
        assert_eq!(
            entity_interaction_consumed(
                Some(E::Pig),
                Some(&stack("saddle", 1)),
                EntityInteractionState {
                    pig_saddled: true,
                    ..survival
                }
            ),
            Some(true),
            "ItemSaddle consumes an already-saddled pig without changing it"
        );
        assert_eq!(
            entity_interaction_consumed(
                Some(E::Pig),
                Some(&stack("saddle", 1)),
                EntityInteractionState {
                    child: true,
                    ..survival
                }
            ),
            Some(true),
            "ItemSaddle consumes a child pig without changing it"
        );
        assert_eq!(
            entity_interaction_consumed(
                Some(E::MushroomCow),
                Some(&stack("bowl", 1)),
                EntityInteractionState {
                    child: true,
                    ..survival
                }
            ),
            Some(false),
            "a child mooshroom declines a bowl before the inventory fallback"
        );
        let mut display = mc_nbt::NbtCompound::new();
        display.insert("Name", mc_nbt::Nbt::String("Dinnerbone".to_owned()));
        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert("display", mc_nbt::Nbt::Compound(display));
        let named_name_tag = ItemStack {
            nbt: Some(tag),
            ..stack("name_tag", 1)
        };
        assert_eq!(
            entity_interaction_consumed(Some(E::Creeper), Some(&named_name_tag), survival),
            Some(true),
            "ItemNameTag consumes a tag with display.Name on EntityLiving"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::Cow), Some(&stack("bucket", 1)), survival),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed(
                Some(E::Cow),
                Some(&stack("bucket", 1)),
                EntityInteractionState {
                    creative: true,
                    ..survival
                }
            ),
            Some(false),
            "ItemBucket's cow branch is disabled in creative"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::MinecartRideable), None, survival),
            None,
            "the empty minecart's occupant check is not represented here"
        );
        assert_eq!(
            entity_interaction_consumed(Some(E::Wolf), None, survival),
            None,
            "wolf tame/sitting state must not be guessed"
        );
    }

    /// FW2-LEAD-RELATION-PREDICTION: the client reaches `EntityLiving.interactFirst`'s lead
    /// branch before subtype/item overrides.  A valid passive target attaches to the local
    /// player, a local-held leash clears without consuming another lead, and invalid/tamed-owner
    /// IMob branches remain known declines so the caller can emit Java's fallback C08.
    #[test]
    fn lead_classifier_matches_attach_clear_owner_and_invalid_gates() {
        use mc_data::EntityType as E;

        let lead = stack("lead", 2);
        let ordinary = EntityInteractionFacts {
            local_player_entity_id: Some(7),
            lead_creative: Some(false),
            ..EntityInteractionFacts::default()
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Cow), Some(&lead), ordinary),
            Some(true)
        );
        assert_eq!(
            lead_interaction(Some(E::Cow), Some(&lead), ordinary),
            Some(LeadInteraction {
                attach_to_player: true,
                clear_from_player: false,
                decrement_stack: true,
            })
        );

        let held_by_player = EntityInteractionFacts {
            leashed_to: Some(7),
            ..ordinary
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Cow), Some(&lead), held_by_player),
            Some(true)
        );
        assert_eq!(
            lead_interaction(Some(E::Cow), Some(&lead), held_by_player),
            Some(LeadInteraction {
                attach_to_player: false,
                clear_from_player: true,
                decrement_stack: false,
            })
        );

        let held_by_other = EntityInteractionFacts {
            leashed_to: Some(8),
            ..ordinary
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Cow), Some(&lead), held_by_other),
            Some(false),
            "allowLeashing rejects an already-leashed target"
        );
        assert!(lead_interaction(Some(E::Cow), Some(&lead), held_by_other).is_none());

        let tamed_not_owner = EntityInteractionFacts {
            wolf_tamed: Some(true),
            tame_owner: Some(false),
            ..ordinary
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Wolf), Some(&lead), tamed_not_owner),
            Some(false),
            "EntityTameable.isOwner is required for a tamed wolf"
        );
        let tamed_owner = EntityInteractionFacts {
            tame_owner: Some(true),
            ..tamed_not_owner
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Wolf), Some(&lead), tamed_owner),
            Some(true)
        );
        let tamed_ocelot_not_owner = EntityInteractionFacts {
            ocelot_tamed: Some(true),
            tame_owner: Some(false),
            ..ordinary
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Ozelot),
                Some(&lead),
                tamed_ocelot_not_owner,
            ),
            Some(false),
            "EntityTameable.isOwner is required for a tamed ocelot"
        );

        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Enderman), Some(&lead), ordinary),
            Some(false),
            "IMob.allowLeashing is false"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Bat), Some(&lead), ordinary),
            Some(false),
            "EntityAmbientCreature.allowLeashing is false"
        );
        let angry_wolf = EntityInteractionFacts {
            wolf_angry: Some(true),
            ..ordinary
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Wolf), Some(&lead), angry_wolf),
            Some(false),
            "an angry wolf refuses EntityWolf.allowLeashing"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Villager),
                Some(&lead),
                EntityInteractionFacts {
                    villager_alive: Some(true),
                    villager_trading: Some(false),
                    ..ordinary
                },
            ),
            Some(true),
            "a live non-trading villager still consumes its own interact branch"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&lead),
                EntityInteractionFacts {
                    horse_type: Some(3),
                    ..ordinary
                }
            ),
            Some(false),
            "undead horses override allowLeashing"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::LeashKnot), Some(&lead), ordinary),
            Some(true),
            "the remote EntityLeashKnot interactFirst body returns true"
        );
        assert!(lead_interaction(Some(E::LeashKnot), Some(&lead), ordinary).is_none());

        let creative = EntityInteractionFacts {
            lead_creative: Some(true),
            ..ordinary
        };
        assert!(
            !lead_interaction(Some(E::Cow), Some(&lead), creative)
                .expect("creative attach")
                .decrement_stack
        );
    }

    /// `ItemLead.onItemUse` returns true only for the fence family.  Invalid blocks return false
    /// after the targeted placement packet, which is what lets the driver append the sentinel.
    #[test]
    fn lead_block_use_consumes_fences_and_rejects_non_fences() {
        let pos = BlockPos::new(0, 64, 0);
        let lead = stack("lead", 1);
        for block in [85, 113, 188, 192] {
            let world = world_with_state(pos, BlockState::new(mc_data::BlockId(block), 0));
            assert_eq!(
                lead_item_use(&world, pos, &lead),
                BlockUseResult::Consumed {
                    placement: None,
                    decrement_stack: false,
                },
                "fence block {block}"
            );
        }
        let world = world_with(pos);
        assert_eq!(lead_item_use(&world, pos, &lead), BlockUseResult::Rejected);
    }

    /// FW2-FURNACE-MINECART-PREDICTION: `EntityMinecartFurnace.interactFirst` consumes every
    /// targeted click, only coal adds 3600 fuel, and every held-item branch overwrites the
    /// cart-minus-player push vector.  The survival decrement is a copied-stack effect, while
    /// the fuel/push mutation still happens for creative, empty, and zero-count fixtures.
    #[test]
    fn furnace_minecart_interaction_matches_coal_survival_and_push_edges() {
        use mc_data::EntityType as E;

        let cart = DVec3::new(12.5, 70.0, -3.25);
        let player = DVec3::new(8.25, 68.0, -9.5);
        let coal = stack("coal", 1);
        assert_eq!(
            furnace_minecart_interaction(
                Some(E::MinecartFurnace),
                Some(&coal),
                false,
                Some(cart),
                Some(player),
            ),
            Some(FurnaceMinecartInteraction {
                coal: true,
                fuel_added: 3600,
                push_x: 4.25,
                push_z: 6.25,
                decrement_stack: true,
            })
        );

        let creative_coal = stack("coal", 4);
        assert_eq!(
            furnace_minecart_interaction(
                Some(E::MinecartFurnace),
                Some(&creative_coal),
                true,
                Some(cart),
                Some(player),
            )
            .map(|prediction| (prediction.fuel_added, prediction.decrement_stack)),
            Some((3600, false)),
            "creative passes a copied coal stack to the item body"
        );

        let zero_coal = stack("coal", 0);
        assert_eq!(
            furnace_minecart_interaction(
                Some(E::MinecartFurnace),
                Some(&zero_coal),
                false,
                Some(cart),
                Some(player),
            )
            .map(|prediction| (
                prediction.coal,
                prediction.fuel_added,
                prediction.decrement_stack
            )),
            Some((true, 3600, false)),
            "the Java fuel branch is reached before the selected-stack underflow edge"
        );

        let bread = stack("bread", 1);
        let non_coal = furnace_minecart_interaction(
            Some(E::MinecartFurnace),
            Some(&bread),
            false,
            Some(cart),
            Some(player),
        )
        .expect("furnace interactFirst always consumes a targeted click");
        assert!(!non_coal.coal);
        assert_eq!(non_coal.fuel_added, 0);
        assert!(!non_coal.decrement_stack);
        assert_eq!((non_coal.push_x, non_coal.push_z), (4.25, 6.25));

        let empty = furnace_minecart_interaction(
            Some(E::MinecartFurnace),
            None,
            false,
            Some(cart),
            Some(player),
        )
        .expect("empty hand still reaches EntityMinecartFurnace.interactFirst");
        assert_eq!(
            (empty.fuel_added, empty.push_x, empty.push_z),
            (0, 4.25, 6.25)
        );
        assert_eq!(
            entity_interaction_consumed(
                Some(E::MinecartFurnace),
                None,
                EntityInteractionState::default()
            ),
            Some(true),
            "the targeted C02 path remains consumed, so no fallback C08 is invented"
        );
        let capability_override = entity_local_interaction(
            Some(E::MinecartFurnace),
            Some(&coal),
            EntityInteractionFacts {
                base: EntityInteractionState::default(),
                furnace_minecart_cart_position: Some(cart),
                furnace_minecart_player_position: Some(player),
                furnace_minecart_creative: Some(true),
                ..EntityInteractionFacts::default()
            },
        )
        .expect("live furnace click has a local prediction");
        assert!(matches!(
            capability_override,
            EntityLocalInteraction::FurnaceMinecart(FurnaceMinecartInteraction {
                decrement_stack: false,
                ..
            })
        ));
        assert!(furnace_minecart_interaction(
            Some(E::MinecartRideable),
            Some(&coal),
            false,
            Some(cart),
            Some(player),
        )
        .is_none());
        assert!(furnace_minecart_interaction(
            Some(E::MinecartFurnace),
            Some(&coal),
            false,
            None,
            Some(player),
        )
        .is_none());
    }

    /// WAVE10RCLICK: the released riding edge closes the empty-minecart fork without changing
    /// the old compatibility wrapper's deliberately unknown result.
    #[test]
    fn entity_interaction_facts_split_empty_minecart_by_occupant_kind() {
        use mc_data::EntityType as E;

        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::MinecartRideable),
                None,
                EntityInteractionFacts {
                    rideable_occupancy: RideableOccupancy::Empty,
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::MinecartRideable),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    rideable_occupancy: RideableOccupancy::Player,
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::MinecartRideable),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    rideable_occupancy: RideableOccupancy::Other,
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(false)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::MinecartRideable),
                None,
                EntityInteractionFacts::default()
            ),
            None
        );
    }

    /// java: `EntitySheep.interact` damages the shears on the client for
    /// an unsheared adult (its `damageItem` is outside `!isRemote`), and the click is still not
    /// consumed; the mooshroom's `damageItem` is server-only.
    #[test]
    fn shears_on_an_unsheared_adult_sheep_are_damaged_on_the_client() {
        use mc_data::EntityType as E;
        let facts = EntityInteractionFacts::default();
        let shears = stack("shears", 1);
        assert!(matches!(
            entity_local_interaction(Some(E::Sheep), Some(&shears), facts),
            Some(EntityLocalInteraction::SheepShears)
        ));
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Sheep), Some(&shears), facts),
            Some(false)
        );
        let sheared = EntityInteractionFacts {
            base: EntityInteractionState {
                sheep_sheared: true,
                ..facts.base
            },
            ..facts
        };
        let lamb = EntityInteractionFacts {
            base: EntityInteractionState {
                child: true,
                ..facts.base
            },
            ..facts
        };
        for no_damage in [sheared, lamb] {
            assert!(entity_local_interaction(Some(E::Sheep), Some(&shears), no_damage).is_none());
        }
        assert!(
            entity_local_interaction(Some(E::Sheep), Some(&stack("stick", 1)), facts).is_none()
        );
        assert!(matches!(
            entity_local_interaction(Some(E::MushroomCow), Some(&shears), facts),
            Some(EntityLocalInteraction::MooshroomShears)
        ));
    }

    /// java: `EntityHorse.interact` reaches `mountTo` (whose first two
    /// statements turn the player to the horse on the client) for an untamed, living, adult,
    /// rider-less horse clicked with an empty hand, sneaking or not; any held item ends earlier.
    #[test]
    fn an_untamed_horse_reaches_mount_to_only_with_an_empty_hand() {
        let untamed = EntityInteractionFacts {
            rideable_occupancy: RideableOccupancy::Empty,
            sneaking: Some(false),
            horse_tame: Some(false),
            horse_type: Some(0),
            horse_chested: Some(false),
            ..EntityInteractionFacts::default()
        };
        assert!(horse_mount_intent(None, untamed));
        assert!(horse_mount_intent(
            None,
            EntityInteractionFacts {
                sneaking: Some(true),
                ..untamed
            }
        ));
        for item in ["wheat", "stick", "saddle", "iron_horse_armor"] {
            assert!(
                !horse_mount_intent(Some(&stack(item, 1)), untamed),
                "{item}"
            );
        }
        // `!isTame() && isUndead()` returns false before the tail.
        for undead in [3, 4] {
            assert!(!horse_mount_intent(
                None,
                EntityInteractionFacts {
                    horse_type: Some(undead),
                    ..untamed
                }
            ));
        }
        // A foal and an occupied horse go to `super.interact`.
        assert!(!horse_mount_intent(
            None,
            EntityInteractionFacts {
                base: EntityInteractionState {
                    child: true,
                    ..untamed.base
                },
                ..untamed
            }
        ));
        assert!(!horse_mount_intent(
            None,
            EntityInteractionFacts {
                rideable_occupancy: RideableOccupancy::Player,
                ..untamed
            }
        ));
    }

    /// WAVE10RCLICK: EntityHorse's order is spawn/undead/sneak/occupied before its item body;
    /// the child controls bite both the untamed rear return and tame armor/feed/chest branches.
    #[test]
    fn entity_interaction_facts_cover_horse_occupancy_and_child_branches() {
        use mc_data::EntityType as E;

        let adult_tame_empty = EntityInteractionFacts {
            rideable_occupancy: RideableOccupancy::Empty,
            sneaking: Some(false),
            horse_tame: Some(true),
            horse_type: Some(0),
            horse_chested: Some(false),
            ..EntityInteractionFacts::default()
        };
        assert!(horse_mount_intent(None, adult_tame_empty));
        assert!(horse_mount_intent(
            Some(&stack("bread", 1)),
            adult_tame_empty
        ));
        assert!(!horse_mount_intent(
            Some(&stack("wheat", 1)),
            adult_tame_empty
        ));
        assert!(!horse_mount_intent(
            Some(&stack("saddle", 1)),
            adult_tame_empty
        ));
        assert!(!horse_mount_intent(
            None,
            EntityInteractionFacts {
                rideable_occupancy: RideableOccupancy::Other,
                ..adult_tame_empty
            }
        ));
        assert!(!horse_mount_intent(
            None,
            EntityInteractionFacts {
                sneaking: Some(true),
                ..adult_tame_empty
            }
        ));
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&stack("bread", 1)),
                adult_tame_empty
            ),
            Some(true),
            "an empty adult horse mounts after its item body declines"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    rideable_occupancy: RideableOccupancy::Other,
                    ..adult_tame_empty
                }
            ),
            Some(false),
            "an occupied adult horse delegates to EntityAnimal"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                None,
                EntityInteractionFacts {
                    rideable_occupancy: RideableOccupancy::Other,
                    sneaking: Some(true),
                    ..adult_tame_empty
                }
            ),
            Some(true),
            "a sneaking tame adult opens its horse GUI before occupancy"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    horse_tame: Some(false),
                    horse_type: Some(3),
                    rideable_occupancy: RideableOccupancy::Empty,
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(false),
            "an untamed skeleton horse declines before its item body"
        );

        let child_tame = EntityInteractionFacts {
            base: EntityInteractionState {
                child: true,
                ..EntityInteractionState::default()
            },
            horse_tame: Some(true),
            horse_type: Some(0),
            horse_chested: Some(false),
            ..EntityInteractionFacts::default()
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&stack("wheat", 1)),
                child_tame
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&stack("chest", 1)),
                EntityInteractionFacts {
                    horse_type: Some(2),
                    horse_chested: Some(false),
                    ..child_tame
                }
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&stack("chest", 1)),
                EntityInteractionFacts {
                    horse_type: Some(2),
                    horse_chested: Some(true),
                    ..child_tame
                }
            ),
            Some(false)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    horse_tame: Some(false),
                    horse_type: Some(1),
                    ..child_tame
                }
            ),
            Some(true),
            "an untamed child donkey still takes the rear-return path"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::EntityHorse),
                None,
                EntityInteractionFacts {
                    horse_tame: Some(false),
                    horse_type: Some(0),
                    ..child_tame
                }
            ),
            Some(false),
            "an untamed child horse with an empty hand skips the item/rear branch"
        );
    }

    /// WAVE10RCLICK: wolf and ocelot item branches are exact where watcher/animal state proves
    /// them; the wild ocelot's server-only temptation gate falls through to EntityAnimal.
    #[test]
    fn entity_interaction_facts_cover_wolf_ocelot_and_passive_breeding() {
        use mc_data::EntityType as E;

        let wild_wolf = EntityInteractionFacts {
            wolf_tamed: Some(false),
            wolf_angry: Some(false),
            ..EntityInteractionFacts::default()
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Wolf),
                Some(&stack("bone", 1)),
                wild_wolf
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Wolf),
                Some(&stack("bone", 1)),
                EntityInteractionFacts {
                    wolf_angry: Some(true),
                    ..wild_wolf
                }
            ),
            Some(false)
        );
        let meat = stack("beef", 1);
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Wolf),
                Some(&meat),
                EntityInteractionFacts {
                    wolf_tamed: Some(true),
                    wolf_hurt: Some(true),
                    ..wild_wolf
                }
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Wolf),
                Some(&meat),
                EntityInteractionFacts {
                    wolf_tamed: Some(true),
                    wolf_hurt: Some(false),
                    ..wild_wolf
                }
            ),
            None,
            "adult full-health meat still needs EntityAnimal's inLove state"
        );
        let mut dye = stack("dye", 1);
        dye.damage = 1;
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Wolf),
                Some(&dye),
                EntityInteractionFacts {
                    wolf_tamed: Some(true),
                    wolf_collar_dye_damage: Some(0),
                    ..wild_wolf
                }
            ),
            Some(true)
        );
        dye.damage = 0;
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Wolf),
                Some(&dye),
                EntityInteractionFacts {
                    wolf_tamed: Some(true),
                    wolf_collar_dye_damage: Some(0),
                    ..wild_wolf
                }
            ),
            Some(false)
        );

        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Ozelot),
                Some(&stack("fish", 1)),
                EntityInteractionFacts {
                    base: EntityInteractionState {
                        child: true,
                        ..EntityInteractionState::default()
                    },
                    ocelot_tamed: Some(false),
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Ozelot),
                Some(&stack("fish", 1)),
                EntityInteractionFacts {
                    ocelot_tamed: Some(false),
                    animal_breeding_ready: None,
                    ..EntityInteractionFacts::default()
                }
            ),
            None
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Ozelot),
                Some(&stack("fish", 1)),
                EntityInteractionFacts {
                    ocelot_tamed: Some(false),
                    animal_breeding_ready: Some(false),
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(false),
            "the client cannot run EntityAITempt, so a non-breeding wild ocelot declines"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Ozelot),
                Some(&stack("fish", 1)),
                EntityInteractionFacts {
                    ocelot_tamed: Some(true),
                    animal_breeding_ready: Some(false),
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(false)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Pig),
                Some(&stack("carrot", 1)),
                EntityInteractionFacts {
                    base: EntityInteractionState {
                        child: true,
                        ..EntityInteractionState::default()
                    },
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Cow),
                Some(&stack("bread", 1)),
                EntityInteractionFacts::default()
            ),
            Some(false)
        );
    }

    /// WAVE10RCLICK: villager's client GUI branch is only deterministic with alive + trading
    /// state; the child and spawn-egg branches do not need server-only state.
    #[test]
    fn entity_interaction_facts_cover_villager_trade_and_ageable_spawn_egg() {
        use mc_data::EntityType as E;

        let adult = EntityInteractionFacts {
            villager_alive: Some(true),
            villager_trading: Some(false),
            ..EntityInteractionFacts::default()
        };
        assert_eq!(
            entity_interaction_consumed_with_facts(Some(E::Villager), None, adult),
            Some(true)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Villager),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    villager_alive: Some(true),
                    villager_trading: Some(true),
                    ..adult
                }
            ),
            Some(false)
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Villager),
                None,
                EntityInteractionFacts {
                    villager_alive: Some(false),
                    villager_trading: Some(false),
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(false),
            "a dead villager cannot enter the client trade branch"
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Villager),
                Some(&stack("bread", 1)),
                EntityInteractionFacts::default()
            ),
            None
        );
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Villager),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    base: EntityInteractionState {
                        child: true,
                        ..EntityInteractionState::default()
                    },
                    ..EntityInteractionFacts::default()
                }
            ),
            Some(false)
        );
        for kind in [
            E::Pig,
            E::Sheep,
            E::Cow,
            E::Chicken,
            E::Wolf,
            E::MushroomCow,
            E::Ozelot,
            E::EntityHorse,
            E::Rabbit,
            E::Villager,
        ] {
            assert_eq!(
                entity_interaction_consumed_with_facts(
                    Some(kind),
                    Some(&stack("spawn_egg", 1)),
                    EntityInteractionFacts::default()
                ),
                Some(true),
                "EntityAgeable spawn egg must consume for {kind:?}"
            );
        }
    }

    /// java: `ItemSaddle.itemInteractionForEntity` (`ItemSaddle.java:19-38`) returns true for
    /// every pig, but only an adult unsaddled pig sets index 16 and decrements the stack. Creative
    /// receives a copied stack in `EntityPlayer.interactWith`, so it keeps its inventory count.
    #[test]
    fn saddle_item_interaction_separates_consumption_from_mutation() {
        use mc_data::EntityType as E;

        let saddle = stack("saddle", 2);
        assert_eq!(
            saddle_item_interaction(
                Some(E::Pig),
                Some(&saddle),
                EntityInteractionState::default()
            ),
            Some(SaddleInteraction {
                consumed: true,
                set_saddled: true,
                decrement_stack: true,
            })
        );
        assert_eq!(
            saddle_item_interaction(
                Some(E::Pig),
                Some(&saddle),
                EntityInteractionState {
                    creative: true,
                    ..EntityInteractionState::default()
                }
            ),
            Some(SaddleInteraction {
                consumed: true,
                set_saddled: true,
                decrement_stack: false,
            })
        );
        for state in [
            EntityInteractionState {
                child: true,
                ..EntityInteractionState::default()
            },
            EntityInteractionState {
                pig_saddled: true,
                ..EntityInteractionState::default()
            },
        ] {
            assert_eq!(
                saddle_item_interaction(Some(E::Pig), Some(&saddle), state),
                Some(SaddleInteraction {
                    consumed: true,
                    set_saddled: false,
                    decrement_stack: false,
                })
            );
        }
        assert_eq!(
            saddle_item_interaction(
                Some(E::Cow),
                Some(&saddle),
                EntityInteractionState::default()
            ),
            None,
            "ItemSaddle is pig-only"
        );
    }

    /// `ItemDye.itemInteractionForEntity` is consumed for every sheep click, but only a
    /// non-sheared colour change mutates the remote watcher and decrements the survival stack.
    /// The item damage is the reversed dyeDamage numbering: damage zero is black metadata 15.
    #[test]
    fn sheep_dye_prediction_preserves_color_noop_sheared_child_and_creative_gates() {
        use mc_data::EntityType as E;

        let mut black = stack("dye", 1);
        black.damage = 0;
        let adult_white = EntityInteractionState::default();
        assert_eq!(
            sheep_dye_interaction(Some(E::Sheep), Some(&black), adult_white, 0),
            Some(SheepDyeInteraction {
                consumed: true,
                new_fleece_metadata: 15,
                recolor: true,
                decrement_stack: true,
            })
        );

        let already_black = sheep_dye_interaction(Some(E::Sheep), Some(&black), adult_white, 15)
            .expect("ItemDye still consumes a matching-colour sheep click");
        assert_eq!(
            already_black,
            SheepDyeInteraction {
                consumed: true,
                new_fleece_metadata: 15,
                recolor: false,
                decrement_stack: false,
            }
        );

        let sheared = sheep_dye_interaction(
            Some(E::Sheep),
            Some(&black),
            EntityInteractionState {
                sheep_sheared: true,
                ..adult_white
            },
            0,
        )
        .expect("a sheared sheep still reaches ItemDye and consumes the click");
        assert!(!sheared.recolor);
        assert!(!sheared.decrement_stack);

        // EntitySheep's dye override has no `isChild()` branch.  A child is recoloured and the
        // survival copy is decremented exactly like an adult.
        let child = sheep_dye_interaction(
            Some(E::Sheep),
            Some(&black),
            EntityInteractionState {
                child: true,
                ..adult_white
            },
            0,
        )
        .expect("child sheep use the same ItemDye override");
        assert!(child.recolor);
        assert!(child.decrement_stack);

        let creative = sheep_dye_interaction(
            Some(E::Sheep),
            Some(&black),
            EntityInteractionState {
                creative: true,
                ..adult_white
            },
            0,
        )
        .expect("creative EntityPlayer still invokes ItemDye on the copied stack");
        assert!(creative.recolor);
        assert!(!creative.decrement_stack);

        // EnumDyeColor.byDyeDamage clamps malformed metadata to dye damage zero (black), and
        // a non-sheep/non-dye click does not claim this helper at all.
        black.damage = 16;
        assert_eq!(
            sheep_dye_interaction(Some(E::Sheep), Some(&black), adult_white, 0)
                .map(|prediction| prediction.new_fleece_metadata),
            Some(15)
        );
        assert_eq!(
            sheep_dye_interaction(Some(E::Cow), Some(&black), adult_white, 0),
            None
        );
        assert_eq!(
            sheep_dye_interaction(Some(E::Sheep), Some(&stack("wheat", 1)), adult_white, 0,),
            None
        );
    }

    #[test]
    fn sheep_dye_local_dispatch_carries_click_edge_color_for_reconciliation() {
        use mc_data::EntityType as E;

        let mut black = stack("dye", 1);
        black.damage = 0;
        let local = entity_local_interaction(
            Some(E::Sheep),
            Some(&black),
            EntityInteractionFacts {
                sheep_fleece_metadata: Some(0),
                ..EntityInteractionFacts::default()
            },
        )
        .expect("a live sheep dye click reaches the local item override");
        assert!(matches!(
            local,
            EntityLocalInteraction::SheepDye {
                expected_fleece_metadata: 0,
                prediction: SheepDyeInteraction {
                    consumed: true,
                    new_fleece_metadata: 15,
                    recolor: true,
                    decrement_stack: true,
                },
            }
        ));

        assert!(
            entity_local_interaction(
                Some(E::Sheep),
                Some(&black),
                EntityInteractionFacts::default(),
            )
            .is_none(),
            "a partial watcher fixture must not invent the click-edge colour"
        );

        let no_op = entity_local_interaction(
            Some(E::Sheep),
            Some(&black),
            EntityInteractionFacts {
                base: EntityInteractionState {
                    sheep_sheared: true,
                    ..EntityInteractionState::default()
                },
                sheep_fleece_metadata: Some(0),
                ..EntityInteractionFacts::default()
            },
        )
        .expect("ItemDye still consumes a sheared sheep click");
        assert!(matches!(
            no_op,
            EntityLocalInteraction::SheepDye {
                prediction: SheepDyeInteraction {
                    consumed: true,
                    recolor: false,
                    decrement_stack: false,
                    ..
                },
                ..
            }
        ));
    }

    /// The client-local interaction family keeps Java's visible mutation boundary separate from
    /// server-owned outcomes: babies get the happy timer (their network age byte is still -1),
    /// adults need the private love gate, cow/mooshroom containers mutate in-place, horse feed
    /// applies the literal heal/growth/temper table, and invalid mounts/supports remain no-ops.
    #[test]
    fn species_item_predictions_cover_age_food_owner_mount_and_creative_gates() {
        use mc_data::EntityType as E;

        let wheat = stack("wheat", 2);
        let fish = stack("fish", 2);
        let bucket = stack("bucket", 1);
        let bucket_stack = stack("bucket", 2);
        let bowl = stack("bowl", 1);
        let bowls = stack("bowl", 2);
        let shears = stack("shears", 1);
        let horse_food = stack("wheat", 2);

        let baby = EntityInteractionFacts {
            base: EntityInteractionState {
                child: true,
                ..EntityInteractionState::default()
            },
            ..EntityInteractionFacts::default()
        };
        assert_eq!(
            animal_feed_interaction(Some(E::Cow), Some(&wheat), baby),
            Some(AnimalFeedInteraction {
                starts_in_love: false,
                happy_particles: true,
                decrement_stack: true,
            })
        );
        assert_eq!(
            animal_feed_interaction(
                Some(E::Cow),
                Some(&wheat),
                EntityInteractionFacts {
                    base: EntityInteractionState {
                        child: true,
                        creative: true,
                        ..EntityInteractionState::default()
                    },
                    ..EntityInteractionFacts::default()
                }
            )
            .map(|prediction| prediction.decrement_stack),
            Some(false),
            "creative baby feed keeps the copied stack"
        );
        let adult_ready = EntityInteractionFacts {
            animal_breeding_ready: Some(true),
            ..EntityInteractionFacts::default()
        };
        assert_eq!(
            animal_feed_interaction(Some(E::Cow), Some(&wheat), adult_ready),
            Some(AnimalFeedInteraction {
                starts_in_love: true,
                happy_particles: false,
                decrement_stack: true,
            })
        );
        assert_eq!(
            animal_feed_interaction(
                Some(E::Cow),
                Some(&wheat),
                EntityInteractionFacts {
                    animal_breeding_ready: Some(false),
                    ..adult_ready
                }
            ),
            None,
            "an adult already in love does not consume a second feed"
        );
        assert_eq!(
            animal_feed_interaction(Some(E::Cow), Some(&stack("bread", 1)), adult_ready),
            None,
            "wrong food never reaches EntityAnimal.interact"
        );

        // Ocelot fish still reaches EntityAnimal.interact for both tame owners and wild cats on
        // the client. The taming/sitting AI gate is server-only and therefore has no local state
        // mutation to predict; its fish click is represented by the ordinary love branch.
        for ocelot_tamed in [false, true] {
            assert!(matches!(
                entity_local_interaction(
                    Some(E::Ozelot),
                    Some(&fish),
                    EntityInteractionFacts {
                        ocelot_tamed: Some(ocelot_tamed),
                        animal_breeding_ready: Some(true),
                        ..adult_ready
                    }
                ),
                Some(EntityLocalInteraction::AnimalFeed(AnimalFeedInteraction {
                    starts_in_love: true,
                    ..
                }))
            ));
        }
        assert_eq!(
            entity_interaction_consumed_with_facts(
                Some(E::Ozelot),
                Some(&stack("bread", 1)),
                EntityInteractionFacts {
                    ocelot_tamed: Some(true),
                    ..adult_ready
                }
            ),
            Some(false),
            "owner/sitting branch does not turn wrong food into a local consume"
        );

        assert_eq!(
            cow_milk_interaction(Some(E::Cow), Some(&bucket), adult_ready),
            Some(CowMilkInteraction {
                replace_selected: true,
                add_to_inventory: false,
                decrement_stack: true,
            })
        );
        assert_eq!(
            cow_milk_interaction(Some(E::Cow), Some(&bucket_stack), adult_ready)
                .map(|prediction| (prediction.replace_selected, prediction.add_to_inventory)),
            Some((false, true))
        );
        assert_eq!(
            cow_milk_interaction(
                Some(E::Cow),
                Some(&bucket),
                EntityInteractionFacts {
                    base: EntityInteractionState {
                        creative: true,
                        ..EntityInteractionState::default()
                    },
                    ..adult_ready
                }
            ),
            None,
            "creative cow bucket falls through to the ordinary item chain"
        );
        assert_eq!(
            cow_milk_interaction(Some(E::Cow), Some(&bucket), baby),
            None
        );
        assert!(cow_milk_interaction(Some(E::MushroomCow), Some(&bucket), adult_ready).is_some());

        assert_eq!(
            mooshroom_bowl_interaction(Some(E::MushroomCow), Some(&bowl), adult_ready),
            Some(MooshroomBowlInteraction {
                replace_selected: true,
                add_to_inventory: false,
                decrement_stack: false,
            })
        );
        assert_eq!(
            mooshroom_bowl_interaction(Some(E::MushroomCow), Some(&bowls), adult_ready).map(
                |prediction| (
                    prediction.replace_selected,
                    prediction.add_to_inventory,
                    prediction.decrement_stack
                )
            ),
            Some((false, true, true))
        );
        assert_eq!(
            mooshroom_bowl_interaction(
                Some(E::MushroomCow),
                Some(&bowls),
                EntityInteractionFacts {
                    base: EntityInteractionState {
                        creative: true,
                        ..EntityInteractionState::default()
                    },
                    ..adult_ready
                }
            )
            .map(|prediction| prediction.decrement_stack),
            Some(false)
        );
        assert_eq!(
            mooshroom_bowl_interaction(Some(E::MushroomCow), Some(&bowl), baby),
            None
        );
        assert!(mooshroom_shears_interaction(
            Some(E::MushroomCow),
            Some(&shears),
            adult_ready
        ));
        assert!(!mooshroom_shears_interaction(
            Some(E::MushroomCow),
            Some(&shears),
            baby
        ));
        assert!(!mooshroom_shears_interaction(
            Some(E::Cow),
            Some(&shears),
            adult_ready
        ));

        let horse_adult_hurt = EntityInteractionFacts {
            horse_tame: Some(true),
            horse_type: Some(0),
            horse_hurt: Some(true),
            horse_temper: Some(0),
            rideable_occupancy: RideableOccupancy::Empty,
            ..EntityInteractionFacts::default()
        };
        assert_eq!(
            horse_item_interaction(Some(E::EntityHorse), Some(&horse_food), horse_adult_hurt),
            Some(HorseItemInteraction {
                growth: 0,
                heal: 2.0,
                starts_in_love: false,
                temper_increase: 3,
                chest: false,
                decrement_stack: true,
            })
        );
        let horse_child = EntityInteractionFacts {
            base: EntityInteractionState {
                child: true,
                ..EntityInteractionState::default()
            },
            ..horse_adult_hurt
        };
        assert_eq!(
            horse_item_interaction(Some(E::EntityHorse), Some(&horse_food), horse_child)
                .map(|prediction| (prediction.growth, prediction.heal)),
            Some((20, 2.0))
        );
        assert_eq!(
            horse_item_interaction(
                Some(E::EntityHorse),
                Some(&stack("golden_carrot", 1)),
                horse_adult_hurt
            )
            .map(|prediction| prediction.starts_in_love),
            Some(true)
        );
        assert_eq!(
            horse_item_interaction(
                Some(E::EntityHorse),
                Some(&stack("wheat", 1)),
                EntityInteractionFacts {
                    horse_hurt: Some(false),
                    ..horse_adult_hurt
                }
            ),
            None,
            "full-health tame horse food falls through to mount"
        );
        assert!(horse_mount_intent(
            Some(&horse_food),
            EntityInteractionFacts {
                horse_hurt: Some(false),
                sneaking: Some(false),
                ..horse_adult_hurt
            }
        ));
        assert_eq!(
            horse_item_interaction(
                Some(E::EntityHorse),
                Some(&stack("chest", 1)),
                EntityInteractionFacts {
                    horse_tame: Some(true),
                    horse_type: Some(2),
                    horse_chested: Some(false),
                    rideable_occupancy: RideableOccupancy::Empty,
                    ..EntityInteractionFacts::default()
                }
            )
            .map(|prediction| prediction.chest),
            Some(true)
        );
        assert_eq!(
            horse_item_interaction(
                Some(E::EntityHorse),
                Some(&stack("chest", 1)),
                EntityInteractionFacts {
                    horse_tame: Some(false),
                    horse_type: Some(2),
                    horse_chested: Some(false),
                    rideable_occupancy: RideableOccupancy::Empty,
                    ..EntityInteractionFacts::default()
                }
            ),
            None,
            "an untamed pack horse rears instead of opening a chest"
        );
        assert_eq!(
            horse_item_interaction(
                Some(E::EntityHorse),
                Some(&horse_food),
                EntityInteractionFacts {
                    rideable_occupancy: RideableOccupancy::Player,
                    ..horse_adult_hurt
                }
            ),
            None,
            "occupied horses delegate before their feed branch"
        );
        assert_eq!(
            horse_item_interaction(
                Some(E::EntityHorse),
                Some(&horse_food),
                EntityInteractionFacts {
                    horse_type: Some(3),
                    ..horse_adult_hurt
                }
            ),
            None,
            "undead horses do not consume ordinary horse food"
        );

        let carrot = stack("carrot_on_a_stick", 1);
        assert_eq!(
            carrot_on_a_stick_interaction(Some(&carrot), true, true, true, false),
            Some(CarrotOnAStickInteraction {
                boost: true,
                damage: 7,
            })
        );
        let nearly_broken = ItemStack {
            damage: 18,
            ..carrot.clone()
        };
        assert_eq!(
            carrot_on_a_stick_interaction(Some(&nearly_broken), true, true, true, true)
                .map(|prediction| prediction.damage),
            Some(7),
            "remaining durability seven still accepts the boost"
        );
        let exhausted = ItemStack {
            damage: 19,
            ..carrot.clone()
        };
        assert!(
            carrot_on_a_stick_interaction(Some(&exhausted), true, true, true, false).is_none(),
            "remaining durability below seven rejects the callback"
        );
        for (saddled, mounted, controlled) in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            assert_eq!(
                carrot_on_a_stick_interaction(Some(&carrot), saddled, mounted, controlled, false),
                None,
                "wrong pig/mount/control gate: {saddled} {mounted} {controlled}"
            );
        }
    }

    /// java: `Entity.getCollisionBorderSize` (`:2036`) and the two overrides. The comment this
    /// replaced claimed the value was zero for everything but a boat; both halves were wrong.
    #[test]
    fn the_collision_border_sizes_match_the_two_overrides() {
        use mc_data::EntityType as E;
        assert!((collision_border_size(None) - 0.1).abs() < 1e-6);
        assert!((collision_border_size(Some(E::Boat)) - 0.1).abs() < 1e-6);
        assert!((collision_border_size(Some(E::ItemFrame)) - 0.0).abs() < 1e-6);
        for kind in [E::Fireball, E::SmallFireball, E::WitherSkull] {
            assert!(
                (collision_border_size(Some(kind)) - 1.0).abs() < 1e-6,
                "{kind:?}"
            );
        }
    }

    // ---------------------------------------------------------------- the placement gate

    fn stack(name: &str, count: i32) -> ItemStack {
        let id = mc_data::item_by_name(name).expect("item").0;
        ItemStack {
            item_id: i16::try_from(id).expect("id fits"),
            count,
            damage: 0,
            nbt: None,
        }
    }

    /// A loaded, all-air column plus whatever `blocks` names.
    fn world_of(blocks: &[(BlockPos, &str)]) -> World {
        let mut world = World::new();
        let mut done = std::collections::HashSet::new();
        for (pos, _) in blocks {
            if !done.insert(pos.chunk_pos()) {
                continue;
            }
            let bitmask = 0xFFFFu16;
            let data = vec![0u8; mc_world::expected_chunk_data_len(bitmask, true, true)];
            world
                .apply_chunk_data(pos.chunk_pos(), true, bitmask, &data)
                .expect("an empty column decodes");
        }
        for (pos, name) in blocks {
            let id = mc_data::block_by_name(name).expect(name);
            world.set_block(*pos, BlockState::new(id, 0));
        }
        world
    }

    /// W-MOUSE / F-P1-client-01-7: `middleClickMouse` asks the block for its item and virtual
    /// damage value, not for the raw block-state metadata. The lapis negative control is useful
    /// here because `BlockOre.getDamageValue` is hard-zero even though lapis' break drop is dye
    /// metadata `4`.
    #[test]
    fn block_pick_uses_java_item_and_pick_metadata_overrides() {
        let pos = BlockPos::new(2, 64, 2);
        let mut world = world_of(&[(pos, "stone")]);
        assert_eq!(
            pick_block_item(&world, pos),
            Some(PickedItem {
                item_id: 1,
                metadata: 0,
                meta_specific: true,
            })
        );

        let dirt = mc_data::block_by_name("dirt").expect("dirt");
        world.set_block(pos, BlockState::new(dirt, 2));
        assert_eq!(
            pick_block_item(&world, pos),
            Some(PickedItem {
                item_id: 3,
                metadata: 2,
                meta_specific: true,
            })
        );

        let lapis = mc_data::block_by_name("lapis_ore").expect("lapis ore");
        world.set_block(pos, BlockState::new(lapis, 0));
        assert_eq!(
            pick_block_item(&world, pos),
            Some(PickedItem {
                item_id: 21,
                metadata: 0,
                meta_specific: false,
            }),
            "BlockOre.getDamageValue is zero; only its break-drop override returns dye metadata"
        );

        for (double, single, block_meta, item_meta) in
            [(43, 44, 5, 5), (125, 126, 5, 5), (181, 182, 0, 0)]
        {
            world.set_block(pos, BlockState::new(mc_data::BlockId(double), block_meta));
            assert_eq!(
                pick_block_item(&world, pos),
                Some(PickedItem {
                    item_id: single,
                    metadata: item_meta,
                    meta_specific: true,
                }),
                "double slab {double} must pick single slab item {single} with variant bits"
            );
        }

        let crops = mc_data::block_by_name("wheat").expect("wheat");
        world.set_block(pos, BlockState::new(crops, 7));
        assert_eq!(
            pick_block_item(&world, pos),
            Some(PickedItem {
                item_id: 295,
                metadata: 0,
                meta_specific: false,
            })
        );

        // BlockCarrot and BlockPotato inherit BlockCrops.getItem, while the generated block
        // table deliberately has no ItemBlock row for either crop.
        let carrots = mc_data::block_by_name("carrots").expect("carrots");
        world.set_block(pos, BlockState::new(carrots, 7));
        assert_eq!(
            pick_block_item(&world, pos).map(|pick| pick.item_id),
            Some(391)
        );
        let potatoes = mc_data::block_by_name("potatoes").expect("potatoes");
        world.set_block(pos, BlockState::new(potatoes, 7));
        assert_eq!(
            pick_block_item(&world, pos).map(|pick| pick.item_id),
            Some(392)
        );

        let portal = mc_data::block_by_name("portal").expect("portal");
        world.set_block(pos, BlockState::new(portal, 0));
        assert_eq!(pick_block_item(&world, pos), None);
        let air = BlockState::AIR;
        world.set_block(pos, air);
        assert_eq!(pick_block_item(&world, pos), None);
    }

    /// W-MOUSE / F-P1-client-01-7: entity picks are creative-only and preserve the exact
    /// ItemFrame payload metadata, while entity types without a registered spawn egg are a no-op.
    #[test]
    fn entity_pick_forks_creative_only_and_preserves_frame_payload() {
        use mc_data::EntityType as E;

        assert_eq!(
            pick_entity_item(Some(E::Zombie), None, false),
            None,
            "survival middle-click never changes inventory from an entity"
        );
        assert_eq!(
            pick_entity_item(Some(E::Zombie), None, true),
            Some(PickedItem {
                item_id: 383,
                metadata: i16::from(E::Zombie.net_id()),
                meta_specific: true,
            })
        );
        assert_eq!(
            pick_entity_item(Some(E::Painting), None, true),
            Some(PickedItem {
                item_id: 321,
                metadata: 0,
                meta_specific: false,
            })
        );
        assert_eq!(
            pick_entity_item(Some(E::MinecartSpawner), None, true),
            Some(PickedItem {
                item_id: 328,
                metadata: 0,
                meta_specific: false,
            })
        );

        let frame = ItemStack {
            item_id: mc_data::item_by_name("diamond_sword")
                .expect("diamond sword")
                .0 as i16,
            count: 1,
            damage: 17,
            nbt: None,
        };
        assert_eq!(
            pick_entity_item(Some(E::ItemFrame), Some(&frame), true),
            Some(PickedItem {
                item_id: frame.item_id,
                metadata: frame.damage,
                meta_specific: true,
            })
        );
        assert_eq!(
            pick_entity_item(Some(E::ItemFrame), None, true),
            Some(PickedItem {
                item_id: 389,
                metadata: 0,
                meta_specific: false,
            })
        );
        assert_eq!(pick_entity_item(Some(E::EnderCrystal), None, true), None);
    }

    /// W-MOUSE / Ctrl branch: typed skull state is promoted to the same `SkullOwner` root shape
    /// Java's `pickBlockWithNBT` emits, while ordinary tile picks get BlockEntityTag + (+NBT).
    #[test]
    fn ctrl_block_pick_reconstructs_retained_tile_nbt() {
        let skull_pos = BlockPos::new(2, 64, 2);
        let mut world = world_of(&[(skull_pos, "skull")]);
        world.set_tile_entity(
            skull_pos,
            mc_world::TileEntity {
                kind: mc_world::TileKind::Skull,
                data: mc_world::TileData::Skull(mc_world::SkullState {
                    skull_type: 3,
                    rotation: 7,
                    owner_name: Some("Notch".to_owned()),
                    owner_profile: None,
                    skin_url: None,
                }),
            },
        );
        let skull = pick_block_with_nbt(&world, skull_pos)
            .expect("stored skull state should be available to Ctrl pick");
        let skull_tag = skull.nbt.as_ref().expect("SkullOwner root");
        assert_eq!(
            skull_tag
                .get_compound("SkullOwner")
                .and_then(|owner| owner.get_str("Name")),
            Some("Notch")
        );
        assert!(skull_tag.get("BlockEntityTag").is_none());

        let banner_pos = BlockPos::new(3, 64, 2);
        world.set_block(
            banner_pos,
            BlockState::new(
                mc_data::block_by_name("standing_banner").expect("banner"),
                0,
            ),
        );
        world.set_tile_entity(
            banner_pos,
            mc_world::TileEntity {
                kind: mc_world::TileKind::StandingBanner,
                data: mc_world::TileData::Banner(mc_world::BannerState {
                    base_color: 4,
                    patterns: vec![("b".to_owned(), 2)],
                    initialized: true,
                }),
            },
        );
        let banner = pick_block_with_nbt(&world, banner_pos).expect("banner pick");
        let tag = banner.nbt.as_ref().expect("BlockEntityTag");
        assert!(tag.get_compound("BlockEntityTag").is_some());
        assert_eq!(
            tag.get_compound("display")
                .and_then(|display| display.get_list("Lore"))
                .and_then(|lore| lore.first())
                .and_then(mc_nbt::Nbt::as_str),
            Some("(+NBT)")
        );

        // Negative control for the read-only helper: Java's World.getTileEntity creates a blank
        // chest on demand, while a plain snapshot cannot. The live helper must restore that
        // immediate-create edge and still attach the (+NBT) marker.
        let chest_pos = BlockPos::new(4, 64, 2);
        world.set_block(
            chest_pos,
            // Chest metadata zero is not a legal packed state in the generated 1.8.9
            // registry; Java's default south-facing chest is metadata two.
            BlockState::new(mc_data::block_by_name("chest").expect("chest"), 2),
        );
        world.remove_tile_entity(chest_pos);
        assert!(pick_block_with_nbt(&world, chest_pos).is_none());
        let chest = pick_block_with_nbt_or_create(&mut world, chest_pos).unwrap_or_else(|| {
            panic!(
                "lazy chest pick: block={:?} kind={:?} item={:?}",
                world.block_at(chest_pos),
                world.tile_kind_at(chest_pos),
                pick_block_item(&world, chest_pos)
            )
        });
        assert!(chest
            .nbt
            .as_ref()
            .and_then(|root| root.get_compound("BlockEntityTag"))
            .is_some());
    }

    /// java: `Item.registerItemBlock` registers each `ItemBlock` at `Block.getIdFromBlock(block)`,
    /// so [`placed_block`]'s id identity is exact. This is the oracle for that claim, and its
    /// negative control is `banner`, the one item where it does *not* hold.
    #[test]
    fn every_item_block_shares_its_blocks_id() {
        let mut checked = 0;
        let mut banner_seen = false;
        for id in 0u16..=mc_data::MAX_ITEM_ID {
            let props = mc_data::ItemId(id).props();
            if !props.registered || props.kind != mc_data::ItemKind::Block {
                continue;
            }
            checked += 1;
            let block = mc_data::block_by_name(props.name);
            if id == BANNER_ITEM_ID {
                banner_seen = true;
                assert_eq!(block, None, "`banner` is not a block registry name");
                continue;
            }
            assert_eq!(
                block,
                Some(mc_data::BlockId(id)),
                "{} is an ItemBlock at item id {id} but not block id {id}",
                props.name
            );
        }
        assert_eq!(checked, 151, "1.8.9 has exactly 151 ItemBlocks");
        assert!(banner_seen, "the negative control must be reachable");
        // And `placed_block` routes it to the block `ItemBanner`'s constructor passes to `super`.
        assert_eq!(placed_block(&stack("banner", 1)), Some(STANDING_BANNER));
    }

    /// `ItemSign.onItemUse` validates the support material, edit permission, and destination
    /// before its client-only early return. The prediction must not place a local block or tile.
    #[test]
    fn sign_item_use_matches_remote_validation_and_stays_read_only() {
        let clicked = BlockPos::new(0, 64, 0);
        let target = clicked.up();
        let sign = stack("sign", 1);
        let world = world_of(&[(clicked, "minecraft:stone")]);

        assert_eq!(
            sign_item_use(&world, clicked, Facing::Up, &sign, true),
            Some(SignUsePrediction {
                target,
                face: Facing::Up,
            })
        );
        assert!(
            world.block_at(target).props().is_air(),
            "the client returns from ItemSign before local placement"
        );
        assert!(
            sign_item_use(&world, clicked, Facing::North, &sign, true).is_some(),
            "horizontal support creates a wall-sign callback"
        );
        assert!(
            sign_item_use(&world, clicked, Facing::Down, &sign, true).is_none(),
            "ItemSign rejects the underside before offsetting"
        );

        let water = world_of(&[(clicked, "minecraft:water")]);
        assert!(
            sign_item_use(&water, clicked, Facing::Up, &sign, true).is_none(),
            "a non-solid support is rejected"
        );

        let occupied = world_of(&[(clicked, "minecraft:stone"), (target, "minecraft:stone")]);
        assert!(
            sign_item_use(&occupied, clicked, Facing::Up, &sign, true).is_none(),
            "a non-replaceable destination is rejected"
        );

        let cactus_neighbor = target.relative(Facing::North);
        let cactus = world_of(&[
            (clicked, "minecraft:stone"),
            (cactus_neighbor, "minecraft:cactus"),
        ]);
        assert!(
            sign_item_use(&cactus, clicked, Facing::Up, &sign, true).is_none(),
            "BlockSign.hasInvalidNeighbor rejects horizontal cactus"
        );

        assert!(
            sign_item_use(&world, clicked, Facing::Up, &sign, false).is_none(),
            "adventure without CanPlaceOn cannot edit the support"
        );
    }

    /// `ItemSkull.onItemUse` differs from `ItemSign`: replaceable clicks stay in place, while
    /// ordinary clicks offset only after a solid-material guard. Its successful remote branch
    /// remains read-only.
    #[test]
    fn skull_item_use_matches_remote_guards_and_stays_read_only() {
        let clicked = BlockPos::new(0, 64, 0);
        let above = clicked.up();
        let skull = stack("skull", 1);
        let stone = world_of(&[(clicked, "minecraft:stone")]);

        assert_eq!(
            skull_item_use(&stone, clicked, Facing::Up, &skull, true),
            Some(SkullUsePrediction {
                target: above,
                face: Facing::Up,
            })
        );
        assert!(
            stone.block_at(above).props().is_air(),
            "the remote callback must not place a skull locally"
        );

        for replaceable in ["minecraft:vine", "minecraft:snow_layer"] {
            let world = world_of(&[(clicked, replaceable)]);
            assert_eq!(
                skull_item_use(&world, clicked, Facing::North, &skull, true),
                Some(SkullUsePrediction {
                    target: clicked,
                    face: Facing::North,
                }),
                "{replaceable} is replaced in place"
            );
        }

        assert!(
            skull_item_use(&stone, clicked, Facing::Down, &skull, true).is_none(),
            "ItemSkull rejects DOWN before considering placement"
        );
        let water = world_of(&[(clicked, "minecraft:water")]);
        assert!(
            skull_item_use(&water, clicked, Facing::Up, &skull, true).is_none(),
            "a non-solid, non-Block-isReplaceable support is rejected"
        );
        let occupied = world_of(&[(clicked, "minecraft:stone"), (above, "minecraft:stone")]);
        assert!(
            skull_item_use(&occupied, clicked, Facing::Up, &skull, true).is_none(),
            "Blocks.skull.canPlaceBlockAt rejects a non-replaceable destination"
        );
        assert!(
            skull_item_use(&stone, clicked, Facing::Up, &skull, false).is_none(),
            "adventure without CanPlaceOn cannot edit the destination"
        );
        assert!(
            skull_item_use(&stone, clicked, Facing::Up, &stack("sign", 1), true).is_none(),
            "the helper must not consume another item subclass"
        );
    }

    /// `ItemHoe.onItemUse` checks edit permission first, then DOWN/air-above, then the grass and
    /// dirt variants. Its successful `useHoe` call plays the new state's step sound on the remote
    /// world but leaves both block state and durability to the server.
    #[test]
    fn hoe_item_use_matches_remote_order_and_is_read_only() {
        let pos = BlockPos::new(0, 64, 0);
        let hoe = stack("wooden_hoe", 1);
        let grass = world_of(&[(pos, "minecraft:grass")]);
        let farmland = BlockState::new(FARMLAND_BLOCK_ID, 0);
        let result = hoe_item_use(&grass, pos, Facing::Up, &hoe, true);
        assert_eq!(
            result,
            HoeUseResult::Consumed {
                target: pos,
                state: farmland,
            }
        );
        assert_eq!(
            grass.block_at(pos),
            BlockState::new(GRASS_BLOCK_ID, 0),
            "remote useHoe returns before the local setBlockState"
        );
        let sound = hoe_item_use_sound(result).expect("a successful hoe use plays its step sound");
        assert_eq!(sound.event.path, "step.gravel");
        assert_eq!(sound.volume, 1.0);
        assert_eq!(sound.pitch, 0.8);
        assert_eq!(sound.position, Some(pos.to_vec3() + DVec3::splat(0.5)));

        let mut coarse = world_of(&[(pos, "minecraft:dirt")]);
        coarse.set_block(pos, BlockState::new(DIRT_BLOCK_ID, 1));
        assert_eq!(
            hoe_item_use(&coarse, pos, Facing::Up, &hoe, true),
            HoeUseResult::Consumed {
                target: pos,
                state: BlockState::new(DIRT_BLOCK_ID, 0),
            },
            "coarse dirt becomes ordinary dirt"
        );

        let allowed = can_place_on_stack("wooden_hoe", &["minecraft:grass"]);
        assert!(matches!(
            hoe_item_use(&grass, pos, Facing::Up, &allowed, false),
            HoeUseResult::Consumed { .. }
        ));
    }

    #[test]
    fn hoe_item_use_rejects_permission_side_air_variant_and_item_controls() {
        let pos = BlockPos::new(0, 64, 0);
        let hoe = stack("diamond_hoe", 1);
        let grass = world_of(&[(pos, "minecraft:grass")]);

        assert_eq!(
            hoe_item_use(&grass, pos, Facing::Up, &hoe, false),
            HoeUseResult::Rejected,
            "adventure without CanPlaceOn fails before the block checks"
        );
        assert_eq!(
            hoe_item_use(&grass, pos, Facing::Down, &hoe, true),
            HoeUseResult::Rejected,
            "ItemHoe rejects DOWN"
        );

        let blocked_above = world_of(&[(pos, "minecraft:grass"), (pos.up(), "minecraft:stone")]);
        assert_eq!(
            hoe_item_use(&blocked_above, pos, Facing::Up, &hoe, true),
            HoeUseResult::Rejected,
            "the block above must have Material.air"
        );

        let dirt = world_of(&[(pos, "minecraft:dirt")]);
        assert_eq!(
            hoe_item_use(&dirt, pos, Facing::Up, &hoe, true),
            HoeUseResult::Consumed {
                target: pos,
                state: BlockState::new(FARMLAND_BLOCK_ID, 0)
            },
            "ordinary dirt becomes farmland"
        );
        let mut podzol = dirt;
        podzol.set_block(pos, BlockState::new(DIRT_BLOCK_ID, 2));
        assert_eq!(
            hoe_item_use(&podzol, pos, Facing::Up, &hoe, true),
            HoeUseResult::Rejected,
            "podzol is not one of ItemHoe's two dirt branches"
        );

        let stone = world_of(&[(pos, "minecraft:stone")]);
        assert_eq!(
            hoe_item_use(&stone, pos, Facing::Up, &hoe, true),
            HoeUseResult::Rejected
        );
        assert_eq!(
            hoe_item_use(&grass, pos, Facing::Up, &stack("stick", 1), true),
            HoeUseResult::NotApplicable
        );
        assert_eq!(hoe_item_use_sound(HoeUseResult::Rejected), None);
    }

    /// `PlayerControllerMP.onPlayerRightClick` must dispatch this bridge before its targeted
    /// packet: both button materials press locally, an already-pressed state remains consumed,
    /// and a later server block update can reset the prediction. The bridge intentionally does
    /// not invent a support check because Java's activation body never reads neighbours.
    #[test]
    fn button_activation_bridge_preserves_local_press_and_server_reset() {
        let pos = BlockPos::new(8, 70, 8);
        // Load the column before these direct callback probes; an unloaded client column reads
        // as EmptyChunk and would hide the button state entirely.
        let mut world = world_of(&[(pos, "stone")]);

        for button in [mc_data::BlockId(77), mc_data::BlockId(143)] {
            let unpressed = BlockState::new(button, 2);
            world.set_block(pos, unpressed);
            assert_eq!(
                activate_button(&mut world, pos).map(|activation| activation.consumed),
                Some(true)
            );
            assert_eq!(world.block_at(pos), BlockState::new(button, 10));
            assert_eq!(
                activate_button(&mut world, pos).map(|activation| activation.consumed),
                Some(true),
                "already-powered buttons still consume the click"
            );
            assert_eq!(world.block_at(pos), BlockState::new(button, 10));

            // S23/S23-equivalent server state wins over the client prediction.
            world.set_block(pos, unpressed);
            assert_eq!(world.block_at(pos), unpressed);
        }

        // Ordinary blocks do not enter the BlockButton callback, regardless of their support.
        let ordinary = pos.east();
        world.set_block(ordinary, BlockState::new(mc_data::BlockId(1), 0));
        assert_eq!(activate_button(&mut world, ordinary), None);
    }

    /// Every listed GUI/container-style override returns true on a remote world before its
    /// server-only body. The shared bridge must consume the targeted click without inventing a
    /// local tile, GUI, or placement state; the ordinary stone and fence-gate controls prove the
    /// dispatcher does not swallow unrelated/stateful callbacks.
    #[test]
    fn remote_block_activation_family_consumes_without_local_prediction() {
        let pos = BlockPos::new(8, 70, 8);
        let mut world = world_of(&[(pos, "stone")]);
        let remote_true = [
            23, 25, 26, 54, 58, 61, 62, 69, 85, 116, 117, 118, 130, 138, 145, 146, 154, 158, 188,
            189, 190, 191, 192,
        ];

        for id in remote_true {
            let state = mc_data::BlockId(id).default_state();
            world.set_block(pos, state);
            assert_eq!(
                activate_remote_block(&world, pos, true).map(|activation| activation.consumed),
                Some(true),
                "block id {id} must consume its remote activation"
            );
            assert_eq!(
                world.block_at(pos),
                state,
                "block id {id} has no client-side state write in its remote branch"
            );
        }

        for id in [151, 178] {
            let state = mc_data::BlockId(id).default_state();
            world.set_block(pos, state);
            assert_eq!(
                activate_remote_block(&world, pos, true).map(|activation| activation.consumed),
                Some(true),
                "editable daylight detector {id} consumes remotely"
            );
            assert_eq!(world.block_at(pos), state);
            assert_eq!(
                activate_remote_block(&world, pos, false).map(|activation| activation.consumed),
                Some(false),
                "non-editable daylight detector {id} delegates to Block's false default"
            );
        }

        world.set_block(pos, stone());
        assert_eq!(activate_remote_block(&world, pos, true), None);

        let gate_state = mc_data::BlockId(107).default_state();
        world.set_block(pos, gate_state);
        assert_eq!(
            activate_remote_block(&world, pos, true),
            None,
            "FenceGate has a separate stateful client callback"
        );
        let activation = world
            .activate_fence_gate(pos, 0.0)
            .expect("wooden gate activation");
        assert!(activation.consumed);
        assert_eq!(world.block_at(pos).meta() & 4, 4);
    }

    /// java: `ItemSlab.onItemUse` first consumes a matching half slab at the clicked cell, then
    /// tries the adjacent cell, and returns true even when its collision/set-state branch cannot
    /// mutate.  The negative controls force the non-merge `super` path and the permission gate.
    #[test]
    fn slab_item_use_predicts_both_merge_cells_and_preserves_consumed_failures() {
        let slab_pos = BlockPos::new(8, 65, 8);
        let stone = BlockPos::new(8, 64, 8);
        let world = world_of(&[
            (stone, "minecraft:stone"),
            (slab_pos, "minecraft:stone_slab"),
        ]);
        let slab = stack("stone_slab", 64);
        let double = BlockState::new(
            mc_data::block_by_name("double_stone_slab").expect("double slab"),
            0,
        );
        assert_eq!(
            slab_item_use(
                &world,
                slab_pos,
                Facing::Up,
                &ItemStack {
                    count: 0,
                    ..slab.clone()
                },
                true,
                &[]
            ),
            BlockUseResult::Rejected,
            "ItemSlab returns false before inspecting a zero-sized stack"
        );

        assert_eq!(
            slab_item_use(&world, slab_pos, Facing::Up, &slab, true, &[]),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: slab_pos,
                    state: double,
                }),
                decrement_stack: true,
            }
        );

        // `tryPlace` receives the adjacent cell, even when the original hit is an ordinary
        // block.  This catches an implementation that only checks the first merge arm.
        let adjacent = stone.relative(Facing::North);
        let adjacent_world = world_of(&[
            (stone, "minecraft:stone"),
            (adjacent, "minecraft:stone_slab"),
        ]);
        assert_eq!(
            slab_item_use(&adjacent_world, stone, Facing::North, &slab, true, &[]),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: adjacent,
                    state: double,
                }),
                decrement_stack: true,
            }
        );

        // Java returns true after finding the matching slab even when checkNoEntityCollision
        // rejects the full double slab.  A caller must therefore suppress the sentinel C08 but
        // must not decrement or locally write the stack/world.
        let blocker = Aabb::from_bounds(
            f64::from(slab_pos.x),
            f64::from(slab_pos.y),
            f64::from(slab_pos.z),
            f64::from(slab_pos.x + 1),
            f64::from(slab_pos.y + 1),
            f64::from(slab_pos.z + 1),
        );
        assert_eq!(
            slab_item_use(&world, slab_pos, Facing::Up, &slab, true, &[blocker]),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: false,
            }
        );

        let mismatched = ItemStack {
            damage: 1,
            ..slab.clone()
        };
        assert_eq!(
            slab_item_use(&world, slab_pos, Facing::Up, &mismatched, true, &[]),
            BlockUseResult::Delegate { pos: slab_pos },
            "a different slab variant reaches ItemBlock rather than merging"
        );
        assert_eq!(
            slab_item_use(&world, slab_pos, Facing::Up, &slab, false, &[]),
            BlockUseResult::Rejected,
            "ItemSlab checks canPlayerEdit before either merge arm"
        );
    }

    #[test]
    fn double_plant_item_use_predicts_ordered_halves_and_keeps_single_blocks_single() {
        let base = BlockPos::new(8, 64, 8);
        let lower = base.up();
        let upper = lower.up();
        let mut double_plant = stack("double_plant", 2);
        double_plant.damage = 5;
        let supported = world_of(&[(base, "minecraft:grass")]);
        assert_eq!(
            item_block_delegate_use(
                &supported,
                base,
                Facing::Up,
                0.5,
                0.0,
                &double_plant,
                true,
                &[],
            ),
            BlockUseResult::ConsumedPair {
                placements: [
                    BlockUsePlacement {
                        pos: lower,
                        state: BlockState::new(mc_data::BlockId(175), 5),
                    },
                    BlockUsePlacement {
                        pos: upper,
                        state: BlockState::new(mc_data::BlockId(175), 10),
                    },
                ],
                decrement_stack: true,
            },
            "ItemBlock writes the lower variant, then BlockDoublePlant writes default NORTH upper",
        );

        let mut invalid_variant = double_plant.clone();
        invalid_variant.damage = 7;
        let BlockUseResult::ConsumedPair { placements, .. } = item_block_delegate_use(
            &supported,
            base,
            Facing::Up,
            0.5,
            0.0,
            &invalid_variant,
            true,
            &[],
        ) else {
            panic!("an invalid item variant still follows ItemBlock's pair callback");
        };
        assert_eq!(
            placements[0].state,
            BlockState::new(mc_data::BlockId(175), 0),
            "EnumPlantType.byMetadata maps item metadata 7 to SUNFLOWER"
        );
        assert_eq!(placements[1].state.meta(), 10);

        let blocked_upper = world_of(&[(base, "minecraft:grass"), (upper, "minecraft:stone")]);
        assert_eq!(
            item_block_delegate_use(
                &blocked_upper,
                base,
                Facing::Up,
                0.5,
                0.0,
                &double_plant,
                true,
                &[],
            ),
            BlockUseResult::Rejected,
            "BlockDoublePlant.canPlaceBlockAt checks the upper cell before any write",
        );
        assert_eq!(
            item_block_delegate_use(
                &supported,
                base,
                Facing::Up,
                0.5,
                0.0,
                &double_plant,
                false,
                &[],
            ),
            BlockUseResult::Rejected,
            "canPlayerEdit remains before the pair callback",
        );

        // `BlockDoublePlant.isReplaceable` consults `getActualState`, which copies a variant
        // from the lower cell only when that cell is itself a double plant.  A malformed upper
        // state over dirt metadata 2 is the biting control: looking only at `lower.meta() & 7`
        // would incorrectly keep the click in place and manufacture a new pair.
        let orphan_upper = BlockPos::new(12, 64, 12);
        let mut malformed = world_of(&[(orphan_upper.down(), "minecraft:dirt")]);
        malformed.set_block(orphan_upper.down(), BlockState::new(mc_data::BlockId(3), 2));
        malformed.set_block(orphan_upper, BlockState::new(mc_data::BlockId(175), 10));
        assert_eq!(
            item_block_delegate_use(
                &malformed,
                orphan_upper,
                Facing::Up,
                0.5,
                0.0,
                &double_plant,
                true,
                &[],
            ),
            BlockUseResult::Rejected,
            "an orphan upper half is not replaceable just because its support uses variant metadata 2",
        );

        // Vanilla accepts y=255 as the lower cell because y=256 is air to the placement gate;
        // the second client write is then rejected by World.setBlockState's height limit.
        let height_base = BlockPos::new(8, 254, 8);
        let height_world = world_of(&[(height_base, "minecraft:grass")]);
        let BlockUseResult::ConsumedPair { placements, .. } = item_block_delegate_use(
            &height_world,
            height_base,
            Facing::Up,
            0.5,
            0.0,
            &double_plant,
            true,
            &[],
        ) else {
            panic!("height-limit lower cell remains a valid double-plant target");
        };
        assert_eq!(placements[0].pos.y, 255);
        assert_eq!(placements[1].pos.y, 256);
        let mut applied_height_world = height_world;
        for placement in placements {
            applied_height_world.set_block(placement.pos, placement.state);
        }
        assert_eq!(
            applied_height_world.block_at(BlockPos::new(8, 255, 8)),
            BlockState::new(mc_data::BlockId(175), 5)
        );
        assert_eq!(
            applied_height_world.block_at(BlockPos::new(8, 256, 8)),
            BlockState::AIR,
            "the out-of-range upper write is ignored, leaving server correction authoritative",
        );

        // The next target is already outside the world.  The placement gate still sees air
        // above the valid grass support, but Java's lower `setBlockState` fails before
        // `onBlockPlacedBy`, sound, or stack decrement can run.
        let top_target_base = BlockPos::new(8, 255, 8);
        let top_target_world = world_of(&[(top_target_base, "minecraft:grass")]);
        let BlockUseResult::ConsumedPair {
            placements,
            decrement_stack,
        } = item_block_delegate_use(
            &top_target_world,
            top_target_base,
            Facing::Up,
            0.5,
            0.0,
            &double_plant,
            true,
            &[],
        )
        else {
            panic!("the gate still accepts a target just above y=255");
        };
        assert_eq!(placements[0].pos.y, 256);
        assert_eq!(placements[1].pos.y, 257);
        assert!(
            !decrement_stack,
            "the failed lower write does not consume the stack"
        );

        let ordinary = stack("stone", 1);
        let BlockUseResult::Consumed {
            placement: Some(placement),
            decrement_stack,
        } = item_block_delegate_use(&supported, base, Facing::Up, 0.5, 0.0, &ordinary, true, &[])
        else {
            panic!("ordinary ItemBlock should keep its one-placement result");
        };
        assert_eq!(placement.pos, lower);
        assert_eq!(placement.state, BlockState::new(mc_data::BlockId(1), 0));
        assert!(decrement_stack);
    }

    #[test]
    fn door_item_use_selects_both_halves_yaw_hinge_and_all_seven_blocks() {
        let base = BlockPos::new(8, 64, 8);
        let target = base.up();
        let world = world_of(&[(base, "minecraft:stone")]);
        let cases = [
            ("wooden_door", 64),
            ("iron_door", 71),
            ("spruce_door", 193),
            ("birch_door", 194),
            ("jungle_door", 195),
            ("acacia_door", 196),
            ("dark_oak_door", 197),
        ];
        for (name, block) in cases {
            let result = door_item_use(&world, base, Facing::Up, 0.0, &stack(name, 2), true);
            assert_eq!(
                result,
                BlockUseResult::ConsumedPair {
                    placements: [
                        BlockUsePlacement {
                            pos: target,
                            state: BlockState::new(mc_data::BlockId(block), 1),
                        },
                        BlockUsePlacement {
                            pos: target.up(),
                            state: BlockState::new(mc_data::BlockId(block), 8),
                        },
                    ],
                    decrement_stack: true,
                },
                "{name} uses ItemDoor's explicit item-to-block relation"
            );
        }

        // BlockDoor stores a fresh lower half as facing.rotateY().horizontalIndex():
        // South/West/North/East therefore encode as 1/2/3/0.
        let door = stack("wooden_door", 1);
        for (yaw, lower_meta) in [(0.0, 1), (90.0, 2), (180.0, 3), (270.0, 0)] {
            let BlockUseResult::ConsumedPair { placements, .. } =
                door_item_use(&world, base, Facing::Up, yaw, &door, true)
            else {
                panic!("supported UP click must place a door");
            };
            assert_eq!(placements[0].state.meta(), lower_meta, "yaw {yaw}");
        }

        // At yaw zero the door faces south, so its right column is west. More normal cubes on
        // that side select a right hinge; mirroring them east selects a left hinge.
        let right = target.west();
        let right_heavy = world_of(&[
            (base, "minecraft:stone"),
            (right, "minecraft:stone"),
            (right.up(), "minecraft:stone"),
        ]);
        let BlockUseResult::ConsumedPair { placements, .. } =
            door_item_use(&right_heavy, base, Facing::Up, 0.0, &door, true)
        else {
            panic!("right-heavy neighbours must still permit placement");
        };
        assert_eq!(placements[1].state.meta(), 9, "right hinge bit");

        let left = target.east();
        let left_heavy = world_of(&[
            (base, "minecraft:stone"),
            (left, "minecraft:stone"),
            (left.up(), "minecraft:stone"),
        ]);
        let BlockUseResult::ConsumedPair { placements, .. } =
            door_item_use(&left_heavy, base, Facing::Up, 0.0, &door, true)
        else {
            panic!("left-heavy neighbours must still permit placement");
        };
        assert_eq!(placements[1].state.meta(), 8, "left hinge bit");
    }

    #[test]
    fn door_item_use_rejects_wrong_face_permission_support_and_occupied_upper() {
        let base = BlockPos::new(8, 64, 8);
        let target = base.up();
        let door = stack("wooden_door", 1);
        let supported = world_of(&[(base, "minecraft:stone")]);
        assert_eq!(
            door_item_use(&supported, base, Facing::North, 0.0, &door, true),
            BlockUseResult::Rejected
        );
        assert_eq!(
            door_item_use(&supported, base, Facing::Up, 0.0, &door, false),
            BlockUseResult::Rejected
        );

        let unsupported = world_of(&[(target, "minecraft:air")]);
        assert_eq!(
            door_item_use(&unsupported, target, Facing::Up, 0.0, &door, true),
            BlockUseResult::Rejected
        );
        let blocked_upper =
            world_of(&[(base, "minecraft:stone"), (target.up(), "minecraft:stone")]);
        assert_eq!(
            door_item_use(&blocked_upper, base, Facing::Up, 0.0, &door, true),
            BlockUseResult::Rejected,
            "BlockDoor.canPlaceBlockAt must inspect the upper cell"
        );
        assert_eq!(
            door_item_use(&supported, base, Facing::Up, 0.0, &stack("stone", 1), true),
            BlockUseResult::NotApplicable
        );
    }

    #[test]
    fn seed_items_require_their_exact_soil_up_face_and_air_target() {
        let pos = BlockPos::new(8, 64, 8);
        for (item, soil, crop) in [
            ("wheat_seeds", "farmland", 59),
            ("pumpkin_seeds", "farmland", 104),
            ("melon_seeds", "farmland", 105),
            ("nether_wart", "soul_sand", 115),
            ("carrot", "farmland", 141),
            ("potato", "farmland", 142),
        ] {
            let held = stack(item, 2);
            let world = world_of(&[(pos, soil)]);
            assert_eq!(
                seeds_item_use(&world, pos, Facing::Up, &held, true),
                BlockUseResult::Consumed {
                    placement: Some(BlockUsePlacement {
                        pos: pos.up(),
                        state: BlockState::new(mc_data::BlockId(crop), 0),
                    }),
                    decrement_stack: true,
                },
                "{item} must select its registered crop"
            );
            assert_eq!(
                seeds_item_use(&world, pos, Facing::North, &held, true),
                BlockUseResult::Rejected,
                "{item} rejects a non-UP face"
            );
            assert_eq!(
                seeds_item_use(&world, pos, Facing::Up, &held, false),
                BlockUseResult::Rejected,
                "{item} preserves canPlayerEdit"
            );

            let wrong_soil = if soil == "farmland" {
                "soul_sand"
            } else {
                "farmland"
            };
            assert_eq!(
                seeds_item_use(
                    &world_of(&[(pos, wrong_soil)]),
                    pos,
                    Facing::Up,
                    &held,
                    true
                ),
                BlockUseResult::Rejected,
                "{item} rejects the other seed soil"
            );
            assert_eq!(
                seeds_item_use(
                    &world_of(&[(pos, soil), (pos.up(), "stone")]),
                    pos,
                    Facing::Up,
                    &held,
                    true,
                ),
                BlockUseResult::Rejected,
                "{item} requires air above the soil"
            );
        }

        assert_eq!(
            seeds_item_use(
                &world_of(&[(pos, "farmland")]),
                pos,
                Facing::Up,
                &stack("bread", 1),
                true,
            ),
            BlockUseResult::NotApplicable
        );
    }

    #[test]
    fn redstone_reed_and_record_callbacks_keep_their_distinct_remote_contracts() {
        let floor = BlockPos::new(8, 64, 8);
        let target = floor.up();
        let stone_world = world_of(&[(floor, "minecraft:stone")]);

        assert_eq!(
            redstone_item_use(
                &stone_world,
                floor,
                Facing::Up,
                &stack("redstone", 2),
                true,
                &[],
            ),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: target,
                    state: BlockState::new(mc_data::BlockId(55), 0),
                }),
                decrement_stack: true,
            }
        );
        assert_eq!(
            redstone_item_use(
                &stone_world,
                floor,
                Facing::Up,
                &stack("redstone", 2),
                false,
                &[],
            ),
            BlockUseResult::Rejected,
            "the redstone callback keeps canPlayerEdit after the targeted C08"
        );

        for (item, block) in [
            ("string", 132),
            ("reeds", 83),
            ("cake", 92),
            ("repeater", 93),
            ("brewing_stand", 117),
            ("cauldron", 118),
            ("flower_pot", 140),
            ("comparator", 149),
        ] {
            assert_eq!(
                item_reed_block(&stack(item, 1)),
                Some(mc_data::BlockId(block)),
                "{item} must retain Item.registerItems' ItemReed backing block"
            );
        }
        let repeater = stack("repeater", 2);
        let BlockUseResult::Consumed {
            placement: Some(placement),
            decrement_stack: true,
        } = item_block_delegate_use(
            &stone_world,
            floor,
            Facing::Up,
            0.5,
            0.0,
            &repeater,
            true,
            &[],
        )
        else {
            panic!("a supported repeater ItemReed use must place locally");
        };
        assert_eq!(placement.pos, target);
        assert_eq!(placement.state.block(), mc_data::BlockId(93));
        assert_eq!(placement.state.meta(), 2, "yaw-zero opposite faces north");

        let empty = world_of(&[(floor, "minecraft:jukebox")]);
        let record = stack("record_13", 1);
        assert_eq!(
            record_item_use(&empty, floor, &record),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: false,
            },
            "the remote ItemRecord arm validates but does not insert or decrement"
        );
        let mut occupied = empty;
        occupied.set_block(floor, BlockState::new(mc_data::BlockId(84), 1));
        assert_eq!(
            record_item_use(&occupied, floor, &record),
            BlockUseResult::Rejected,
            "HAS_RECORD rejects the item callback and permits the untargeted fallback"
        );
    }

    #[test]
    fn flint_and_dye_callbacks_distinguish_permission_metadata_and_remote_authority() {
        let support = BlockPos::new(8, 64, 8);
        let target = support.up();
        let stone_world = world_of(&[(support, "minecraft:stone")]);
        let flint = stack("flint_and_steel", 1);
        assert_eq!(
            flint_and_steel_item_use(&stone_world, support, Facing::Up, &flint, true),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: target,
                    state: BlockState::new(mc_data::BlockId(51), 0),
                }),
                decrement_stack: false,
            }
        );
        assert_eq!(
            flint_and_steel_item_use(&stone_world, support, Facing::Up, &flint, false),
            BlockUseResult::Rejected
        );
        assert_eq!(
            flint_and_steel_item_use(
                &world_of(&[(support, "minecraft:stone"), (target, "minecraft:stone")]),
                support,
                Facing::Up,
                &flint,
                true,
            ),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: false,
            },
            "an editable occupied target still damages and consumes"
        );

        let mut jungle = world_of(&[(support, "minecraft:log")]);
        jungle.set_block(support, BlockState::new(mc_data::BlockId(17), 3));
        let mut brown = stack("dye", 2);
        brown.damage = 3;
        let cocoa_target = support.east();
        assert_eq!(
            dye_item_use(&jungle, support, Facing::East, &brown, true),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: cocoa_target,
                    state: BlockState::new(
                        mc_data::BlockId(127),
                        Facing::West.horizontal_index().expect("west is horizontal"),
                    ),
                }),
                decrement_stack: true,
            }
        );
        assert_eq!(
            dye_item_use(&jungle, support, Facing::Up, &brown, true),
            BlockUseResult::Rejected,
            "cocoa rejects vertical log faces"
        );
        let mut occupied = jungle;
        occupied.set_block(cocoa_target, BlockState::new(mc_data::BlockId(1), 0));
        assert_eq!(
            dye_item_use(&occupied, support, Facing::East, &brown, true),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: false,
            },
            "a valid jungle log consumes brown dye even when the offset cell is occupied"
        );

        let crop = BlockPos::new(9, 64, 9);
        let mut immature = world_of(&[(crop, "minecraft:wheat")]);
        immature.set_block(crop, BlockState::new(mc_data::BlockId(59), 6));
        let mut white = stack("dye", 1);
        white.damage = 15;
        assert_eq!(
            dye_item_use(&immature, crop, Facing::Up, &white, true),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: false,
            },
            "remote bonemeal validates but leaves growth/decrement to the server"
        );
        immature.set_block(crop, BlockState::new(mc_data::BlockId(59), 7));
        assert_eq!(
            dye_item_use(&immature, crop, Facing::Up, &white, true),
            BlockUseResult::Rejected,
            "mature crops cannot grow"
        );
    }

    #[test]
    fn flint_uses_java_portal_size_for_both_axes_and_rejects_bad_frames() {
        let framed = |x_axis: bool, width: i32, height: i32| {
            let mut blocks = Vec::new();
            let top = 64 + height + 1;
            for lateral in -1..=width {
                for y in 64..=top {
                    if lateral == -1 || lateral == width || y == 64 || y == top {
                        let pos = if x_axis {
                            BlockPos::new(lateral, y, 8)
                        } else {
                            BlockPos::new(8, y, lateral)
                        };
                        blocks.push((pos, "minecraft:obsidian"));
                    }
                }
            }
            let target = if x_axis {
                BlockPos::new(0, 65, 8)
            } else {
                BlockPos::new(8, 65, 0)
            };
            (world_of(&blocks), target)
        };
        let flint = stack("flint_and_steel", 1);

        for (x_axis, width, height, metadata) in [(true, 2, 3, 1), (false, 2, 3, 2)] {
            let (world, target) = framed(x_axis, width, height);
            let result = flint_and_steel_item_use(&world, target, Facing::Up, &flint, true);
            let BlockUseResult::ConsumedMany {
                placements,
                decrement_stack: false,
            } = result
            else {
                panic!("a valid {x_axis:?} portal frame must fill locally");
            };
            assert_eq!(placements.len(), (width * height) as usize);
            assert!(placements.iter().all(|placement| {
                placement.state == BlockState::new(PORTAL_BLOCK_ID, metadata)
            }));
            assert_eq!(placements[0].state.block(), PORTAL_BLOCK_ID);
            assert_eq!(placements.last().unwrap().state.meta(), metadata);
        }

        // Java's inclusive minimums reject a one-wide or two-high interior, while the inclusive
        // maximums still accept the full 21x21 interior and produce every state write.
        for (width, height) in [(1, 3), (2, 2)] {
            let (world, target) = framed(true, width, height);
            assert!(matches!(
                flint_and_steel_item_use(&world, target, Facing::Up, &flint, true),
                BlockUseResult::Consumed {
                    placement: Some(BlockUsePlacement { state, .. }),
                    decrement_stack: false,
                } if state.block() == FIRE_BLOCK_ID
            ));
        }
        let (world, target) = framed(true, 21, 21);
        let BlockUseResult::ConsumedMany { placements, .. } =
            flint_and_steel_item_use(&world, target, Facing::Up, &flint, true)
        else {
            panic!("the Java maximum 21x21 portal must remain reachable");
        };
        assert_eq!(placements.len(), 441);
        assert!(placements
            .iter()
            .all(|placement| placement.state == BlockState::new(PORTAL_BLOCK_ID, 1)));

        // A missing top frame and a solid obstruction in the interior both preserve ordinary
        // fire behavior; neither is allowed to create a partial portal.
        let (mut incomplete, target) = framed(true, 2, 3);
        incomplete.set_block(BlockPos::new(0, 68, 8), BlockState::AIR);
        assert!(matches!(
            flint_and_steel_item_use(&incomplete, target, Facing::Up, &flint, true),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement { state, .. }),
                decrement_stack: false,
            } if state.block() == FIRE_BLOCK_ID
        ));
        let (mut obstructed, target) = framed(true, 2, 3);
        obstructed.set_block(target.east(), BlockState::new(mc_data::BlockId(1), 0));
        assert!(matches!(
            flint_and_steel_item_use(&obstructed, target, Facing::Up, &flint, true),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement { state, .. }),
                decrement_stack: false,
            } if state.block() == FIRE_BLOCK_ID
        ));
    }

    #[test]
    fn hanging_callbacks_validate_faces_support_and_existing_hanging_boxes() {
        let support = BlockPos::new(8, 64, 8);
        let face = Facing::North;
        let anchor = support.relative(face);
        let frame = stack("item_frame", 2);
        let painting = stack("painting", 2);
        let stone = world_of(&[(support, "minecraft:stone")]);

        for held in [&frame, &painting] {
            assert_eq!(
                hanging_item_use(&stone, support, face, held, true, &[]),
                BlockUseResult::Consumed {
                    placement: None,
                    decrement_stack: true,
                },
                "a one-block solid backing accepts both hanging subclasses"
            );
            assert_eq!(
                hanging_item_use(&stone, support, Facing::Up, held, true, &[]),
                BlockUseResult::Rejected,
                "ItemHangingEntity rejects vertical faces before construction"
            );
            assert_eq!(
                hanging_item_use(&stone, support, face, held, false, &[]),
                BlockUseResult::Rejected,
                "the offset anchor is subject to canPlayerEdit"
            );
        }

        assert_eq!(
            hanging_item_use(&World::new(), support, face, &frame, true, &[]),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: false,
            },
            "invalid support still consumes the callback but keeps the stack"
        );

        let frame_position = hanging_position(anchor, face, 12, 12);
        let occupied = hanging_bounding_box(frame_position, face, 12, 12);
        assert_eq!(
            hanging_item_use(&stone, support, face, &frame, true, &[occupied]),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: false,
            },
            "an existing EntityHanging overlap invalidates the prospective surface"
        );

        let mut repeater = world_of(&[(support, "stone")]);
        repeater.set_block(support, BlockState::new(mc_data::BlockId(93), 0));
        assert_eq!(
            hanging_item_use(&repeater, support, face, &frame, true, &[]),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: true,
            },
            "redstone diodes are Java's explicit non-solid support exception"
        );
        assert_eq!(
            hanging_item_use(&stone, support, face, &stack("stick", 1), true, &[]),
            BlockUseResult::NotApplicable
        );
    }

    #[test]
    fn minecart_and_piston_callbacks_keep_their_distinct_client_contracts() {
        let pos = BlockPos::new(8, 64, 8);
        let mut rail_world = world_of(&[(pos, "stone")]);
        rail_world.set_block(pos, BlockState::new(mc_data::BlockId(66), 0));
        let stone_world = world_of(&[(pos, "minecraft:stone")]);
        for item_id in [328, 342, 343, 407, 408, 422] {
            let held = ItemStack {
                item_id,
                count: 1,
                damage: 0,
                nbt: None,
            };
            assert_eq!(
                minecart_item_use(&rail_world, pos, &held),
                BlockUseResult::Consumed {
                    placement: None,
                    decrement_stack: true,
                },
                "minecart item {item_id} consumes only a rail click"
            );
            assert_eq!(
                minecart_item_use(&stone_world, pos, &held),
                BlockUseResult::Rejected,
                "minecart item {item_id} rejects a non-rail"
            );
        }

        let piston = ItemStack {
            item_id: 33,
            count: 2,
            damage: 0,
            nbt: None,
        };
        let target = pos.up();
        assert_eq!(
            piston_item_use(
                &stone_world,
                pos,
                Facing::Up,
                &piston,
                true,
                &[],
                DVec3::new(8.5, 64.0, 8.5),
                65.62,
                0.0,
            ),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: target,
                    state: BlockState::new(mc_data::BlockId(33), Facing::North.index()),
                }),
                decrement_stack: true,
            },
            "ItemPiston metadata 7 is replaced by the opposite horizontal facing"
        );
        let BlockUseResult::Consumed {
            placement: Some(up),
            ..
        } = piston_item_use(
            &stone_world,
            pos,
            Facing::Up,
            &piston,
            true,
            &[],
            DVec3::new(8.5, 64.0, 8.5),
            68.0,
            0.0,
        )
        else {
            panic!("a supported piston must place");
        };
        assert_eq!(up.state.meta(), Facing::Up.index());

        let blocked = world_of(&[(pos, "minecraft:stone"), (target, "minecraft:stone")]);
        assert_eq!(
            piston_item_use(
                &blocked,
                pos,
                Facing::Up,
                &piston,
                true,
                &[],
                DVec3::new(8.5, 64.0, 8.5),
                65.62,
                0.0,
            ),
            BlockUseResult::Rejected
        );
    }

    #[test]
    fn anvil_and_armor_stand_callbacks_preserve_metadata_and_remote_entity_authority() {
        let hit = BlockPos::new(8, 64, 8);
        let target = hit.up();
        let base = world_of(&[(hit, "minecraft:stone")]);
        let anvil = ItemStack {
            item_id: 145,
            count: 2,
            damage: 2,
            nbt: None,
        };
        for (yaw, facing_meta) in [(0.0, 1), (90.0, 2), (180.0, 3), (270.0, 0)] {
            assert_eq!(
                anvil_item_use(&base, hit, Facing::Up, &anvil, true, &[], yaw),
                BlockUseResult::Consumed {
                    placement: Some(BlockUsePlacement {
                        pos: target,
                        state: BlockState::new(mc_data::BlockId(145), 8 | facing_meta),
                    }),
                    decrement_stack: true,
                },
                "ItemAnvilBlock damage and rotated placer facing must both survive: yaw={yaw}"
            );
        }
        let target_box = Aabb::from_bounds(8.0, 65.0, 8.0, 9.0, 66.0, 9.0);
        assert_eq!(
            anvil_item_use(&base, hit, Facing::Up, &anvil, true, &[target_box], 0.0),
            BlockUseResult::Rejected,
            "the ordinary ItemBlock entity-collision gate still applies to anvils"
        );

        let stand = ItemStack {
            item_id: 416,
            count: 2,
            damage: 0,
            nbt: None,
        };
        assert_eq!(
            armor_stand_item_use(&base, hit, Facing::Up, &stand, true, &[]),
            BlockUseResult::Consumed {
                placement: None,
                decrement_stack: true,
            },
            "the remote callback decrements but never predicts an ArmorStand entity"
        );
        assert_eq!(
            armor_stand_item_use(&base, hit, Facing::Down, &stand, true, &[]),
            BlockUseResult::Rejected
        );
        assert_eq!(
            armor_stand_item_use(
                &base,
                hit,
                Facing::Up,
                &stand,
                true,
                &[Aabb::from_bounds(8.25, 65.0, 8.25, 8.75, 66.8, 8.75)],
            ),
            BlockUseResult::Rejected,
            "every non-spectating entity vetoes the one-by-two stand volume"
        );
        let upper_blocked = world_of(&[(hit, "minecraft:stone"), (target.up(), "minecraft:stone")]);
        assert_eq!(
            armor_stand_item_use(&upper_blocked, hit, Facing::Up, &stand, true, &[]),
            BlockUseResult::Rejected,
            "both destination cells must be air or Block.isReplaceable"
        );
    }

    /// java: `ItemSnow.onItemUse` keeps an UP click on the snow layer in place, increments only
    /// through layer eight, and offsets every other non-replaceable click before delegating.
    #[test]
    fn snow_item_use_predicts_layer_increment_and_double_offset() {
        let snow_pos = BlockPos::new(8, 65, 8);
        let stone = snow_pos.down();
        let world = world_of(&[
            (stone, "minecraft:stone"),
            (snow_pos, "minecraft:snow_layer"),
        ]);
        // The registry has both `minecraft:snow` (the full snow block, ItemBlock id 80) and
        // `minecraft:snow_layer` (ItemSnow id 78).  Java's special callback is the latter.
        let snow = stack("snow_layer", 64);
        assert_eq!(
            snow_item_use(
                &world,
                snow_pos,
                Facing::Up,
                &ItemStack {
                    count: 0,
                    ..snow.clone()
                },
                true,
                &[]
            ),
            BlockUseResult::Rejected,
            "ItemSnow returns false before inspecting a zero-sized stack"
        );
        assert_eq!(
            snow_item_use(&world, snow_pos, Facing::Up, &stack("snow", 64), true, &[]),
            BlockUseResult::NotApplicable,
            "the full snow block (ItemBlock id 80) must not use ItemSnow's callback"
        );
        assert_eq!(
            snow_item_use(
                &world,
                snow_pos,
                Facing::Up,
                &stack("snowball", 1),
                true,
                &[]
            ),
            BlockUseResult::NotApplicable,
            "snowballs must not use the layered-block callback"
        );

        assert_eq!(
            snow_item_use(&world, snow_pos, Facing::Up, &snow, true, &[]),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: snow_pos,
                    state: BlockState::new(mc_data::BlockId(78), 1),
                }),
                decrement_stack: true,
            }
        );

        // A one-layer snow cell is itself replaceable, so even a horizontal click keeps the
        // original cell and increments it.  This is the first half of ItemSnow's deliberately
        // awkward offset condition.
        assert_eq!(
            snow_item_use(&world, snow_pos, Facing::North, &snow, true, &[]),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: snow_pos,
                    state: BlockState::new(mc_data::BlockId(78), 1),
                }),
                decrement_stack: true,
            }
        );

        // Once the layer is two blocks tall it is no longer replaceable.  A horizontal click
        // steps once before the callback inspects the adjacent cell, and the returned Delegate
        // preserves that first offset for ItemBlock's possible second offset.
        let tall = {
            let mut world = world_of(&[
                (stone, "minecraft:stone"),
                (snow_pos, "minecraft:snow_layer"),
            ]);
            world.set_block(snow_pos, BlockState::new(mc_data::BlockId(78), 1));
            world
        };
        let first_offset = snow_pos.relative(Facing::North);
        assert_eq!(
            snow_item_use(&tall, snow_pos, Facing::North, &snow, true, &[]),
            BlockUseResult::Delegate { pos: first_offset }
        );

        let seven_layers = world_of(&[
            (stone, "minecraft:stone"),
            (snow_pos, "minecraft:snow_layer"),
        ]);
        // Metadata six means seven layers; the next write is metadata seven (eight layers).
        let seven_layers = {
            let mut world = seven_layers;
            world.set_block(snow_pos, BlockState::new(mc_data::BlockId(78), 6));
            world
        };
        assert_eq!(
            snow_item_use(&seven_layers, snow_pos, Facing::Up, &snow, true, &[]),
            BlockUseResult::Consumed {
                placement: Some(BlockUsePlacement {
                    pos: snow_pos,
                    state: BlockState::new(mc_data::BlockId(78), 7),
                }),
                decrement_stack: true,
            }
        );

        let full = {
            let mut world = seven_layers;
            world.set_block(snow_pos, BlockState::new(mc_data::BlockId(78), 7));
            world
        };
        assert_eq!(
            snow_item_use(&full, snow_pos, Facing::Up, &snow, true, &[]),
            BlockUseResult::Delegate { pos: snow_pos },
            "an eighth layer reaches ItemBlock instead of writing metadata eight"
        );

        let blocker = Aabb::from_bounds(
            f64::from(snow_pos.x),
            f64::from(snow_pos.y),
            f64::from(snow_pos.z),
            f64::from(snow_pos.x + 1),
            f64::from(snow_pos.y) + 0.25,
            f64::from(snow_pos.z + 1),
        );
        assert_eq!(
            snow_item_use(&world, snow_pos, Facing::Up, &snow, true, &[blocker]),
            BlockUseResult::Delegate { pos: snow_pos },
            "a blocked increment falls through to ItemBlock.onItemUse"
        );
        assert_eq!(
            snow_item_use(&world, snow_pos, Facing::Up, &snow, false, &[]),
            BlockUseResult::Rejected,
            "ItemSnow calls canPlayerEdit on the original hit cell"
        );
    }

    #[test]
    fn log_item_block_placement_maps_face_axis_and_preserves_species() {
        let base = BlockPos::new(20, 64, 20);
        let world = world_of(&[(base, "minecraft:stone")]);
        for (name, id) in [("log", 17), ("log2", 162)] {
            for species in 0..=3_i16 {
                for (face, axis) in [
                    (Facing::Up, 0),
                    (Facing::Down, 0),
                    (Facing::East, 4),
                    (Facing::West, 4),
                    (Facing::North, 8),
                    (Facing::South, 8),
                ] {
                    let mut held = stack(name, 2);
                    held.damage = species;
                    assert_eq!(
                        item_block_delegate_use(&world, base, face, 0.5, 0.0, &held, true, &[]),
                        BlockUseResult::Consumed {
                            placement: Some(BlockUsePlacement {
                                pos: base.relative(face),
                                state: BlockState::new(mc_data::BlockId(id), species as u8 | axis,),
                            }),
                            decrement_stack: true,
                        },
                        "{name} species {species} face {face:?}"
                    );
                }
            }
        }
    }

    /// `BlockFlower.getStateFromMeta` normalizes unsupported item damage before the client
    /// writes its predicted state.  Keep valid red-flower variants intact, and include a
    /// non-flower control so this canonicalization cannot leak into the ordinary ItemBlock path.
    #[test]
    fn flower_item_block_placement_canonicalizes_malformed_metadata() {
        let clicked = BlockPos::new(20, 64, 20);
        let world = world_of(&[(clicked, "minecraft:stone")]);
        let metadata = |item_id: i16, damage: i16| {
            let held = ItemStack {
                item_id,
                count: 1,
                damage,
                nbt: None,
            };
            item_block_placed_metadata(
                &world,
                clicked.up(),
                Facing::Up,
                0.5,
                0.0,
                &held,
                mc_data::BlockId(item_id as u16),
            )
        };

        // Yellow flowers only have DANDELION (meta 0): every malformed damage, including a
        // signed negative value, resolves to the default variant.
        for damage in [-1, 1, 15] {
            assert_eq!(metadata(37, damage), 0, "yellow damage {damage}");
        }
        // Red flowers retain POPPY through OXEYE_DAISY (0..=8), while 9..15 and negative
        // damage resolve through EnumFlowerType.getType's default branch.
        for damage in 0..=8 {
            assert_eq!(metadata(38, damage), damage as u8, "red damage {damage}");
        }
        for damage in [-1, 9, 15] {
            assert_eq!(metadata(38, damage), 0, "red damage {damage}");
        }
        // Negative control: generic stone still preserves its raw item damage.
        assert_eq!(metadata(1, 7), 7);
    }

    /// `ItemBlock` inherits `Item.getMetadata`, which always returns zero.  Farmland then
    /// decodes that value through `BlockFarmland.getStateFromMeta`; malformed item damage must
    /// not leak into the locally predicted moisture state.
    #[test]
    fn farmland_item_block_placement_uses_plain_item_metadata_zero() {
        let clicked = BlockPos::new(20, 64, 20);
        let world = world_of(&[(clicked, "minecraft:stone")]);
        for damage in [-1, 0, 7, 8, 15, 32767] {
            let held = ItemStack {
                item_id: 60,
                count: 1,
                damage,
                nbt: None,
            };
            assert_eq!(
                item_block_placed_metadata(
                    &world,
                    clicked.up(),
                    Facing::Up,
                    0.5,
                    0.0,
                    &held,
                    mc_data::BlockId(60),
                ),
                0,
                "farmland damage {damage} must follow Item.getMetadata"
            );
        }
    }

    #[test]
    fn stair_item_block_placement_maps_yaw_and_hit_half_for_every_species() {
        let base = BlockPos::new(20, 64, 20);
        let world = world_of(&[(base, "minecraft:stone")]);
        let stairs = [
            "oak_stairs",
            "stone_stairs",
            "brick_stairs",
            "stone_brick_stairs",
            "nether_brick_stairs",
            "sandstone_stairs",
            "spruce_stairs",
            "birch_stairs",
            "jungle_stairs",
            "quartz_stairs",
            "acacia_stairs",
            "dark_oak_stairs",
            "red_sandstone_stairs",
        ];
        let yaws = [(0.0, 2), (90.0, 1), (180.0, 3), (-90.0, 0)];
        let halves = [
            (Facing::Up, 1.0, 0),
            (Facing::Down, 0.0, 4),
            (Facing::East, 0.5, 0),
            (Facing::East, 0.500_001, 4),
        ];

        for name in stairs {
            let held = stack(name, 2);
            let block = mc_data::block_by_name(name).expect("known stair block");
            assert!(is_stair_item(&held), "{name} must enter the stair callback");
            for (yaw, facing_meta) in yaws {
                for (face, hit_y, half_meta) in halves {
                    assert_eq!(
                        item_block_delegate_use(&world, base, face, hit_y, yaw, &held, true, &[],),
                        BlockUseResult::Consumed {
                            placement: Some(BlockUsePlacement {
                                pos: base.relative(face),
                                state: BlockState::new(block, facing_meta | half_meta),
                            }),
                            decrement_stack: true,
                        },
                        "{name} yaw {yaw} face {face:?} hitY {hit_y}"
                    );
                }
            }
        }

        assert!(!is_stair_item(&stack("stone", 1)));
    }

    #[test]
    fn generic_item_block_pose_metadata_covers_steep_pitch_and_oriented_families() {
        let clicked = BlockPos::new(20, 64, 20);
        let target = clicked.up();
        let world = world_of(&[(clicked, "minecraft:stone")]);

        // This is the exact four-quadrant horizontal negative control: when the player is not
        // within two blocks of the destination, yaw chooses the opposite horizontal face.
        for (yaw, expected) in [
            (0.0, Facing::North),
            (90.0, Facing::East),
            (180.0, Facing::South),
            (270.0, Facing::West),
        ] {
            assert_eq!(
                facing_from_entity(target, DVec3::new(100.0, 64.0, 100.0), 65.62, yaw,),
                expected,
                "yaw {yaw} must choose the opposite horizontal dispenser facing"
            );
        }
        assert_eq!(
            facing_from_entity(target, DVec3::new(20.5, 64.0, 20.5), 67.1, 0.0),
            Facing::Up,
            "a close, steep upward look chooses UP"
        );
        assert_eq!(
            facing_from_entity(target, DVec3::new(20.5, 64.0, 20.5), 63.9, 0.0),
            Facing::Down,
            "a close, steep downward look chooses DOWN"
        );

        let dispenser = ItemStack {
            item_id: 23,
            count: 2,
            damage: 0,
            nbt: None,
        };
        let BlockUseResult::Consumed {
            placement: Some(placement),
            decrement_stack: true,
        } = item_block_delegate_use_with_pose(
            &world,
            clicked,
            Facing::Up,
            0.5,
            0.0,
            &dispenser,
            true,
            &[],
            DVec3::new(20.5, 64.0, 20.5),
            67.1,
        )
        else {
            panic!("a supported dispenser must predict a placement");
        };
        assert_eq!(placement.pos, target);
        assert_eq!(
            placement.state,
            BlockState::new(mc_data::BlockId(23), Facing::Up.index())
        );

        // `ItemLeaves` is still an ordinary ItemBlock, but its virtual getMetadata adds the
        // persistent bit before BlockLeaves receives the callback.  A raw-damage bridge would
        // incorrectly predict a decayable leaf state (and the server would immediately correct
        // it).  ItemBanner is the explicit no-prediction exception: its client callback returns
        // true without a local write/decrement.
        let mut leaves = stack("leaves", 2);
        leaves.damage = 1;
        let leaves_result =
            item_block_delegate_use(&world, clicked, Facing::Up, 0.5, 0.0, &leaves, true, &[]);
        let BlockUseResult::Consumed {
            placement: Some(leaves_placement),
            decrement_stack: true,
        } = leaves_result
        else {
            panic!("ItemLeaves must use the generic ItemBlock placement callback");
        };
        assert_eq!(
            leaves_placement.state,
            BlockState::new(mc_data::BlockId(18), 5),
            "ItemLeaves metadata must preserve the species and force PERSISTENT"
        );
        assert_eq!(
            item_block_delegate_use(
                &world,
                clicked,
                Facing::Up,
                0.5,
                0.0,
                &stack("banner", 1),
                true,
                &[],
            ),
            BlockUseResult::NotApplicable,
            "ItemBanner owns a client-validating no-prediction callback"
        );

        // The ItemBlock family uses the legacy metadata codecs, not a universal Facing ordinal.
        for (id, expected) in [(77, 5), (69, 5), (50, 5), (154, 0)] {
            let held = ItemStack {
                item_id: id,
                count: 1,
                damage: 0,
                nbt: None,
            };
            let result =
                item_block_delegate_use(&world, clicked, Facing::Up, 0.5, 0.0, &held, true, &[]);
            let BlockUseResult::Consumed {
                placement: Some(placement),
                ..
            } = result
            else {
                panic!("block {id} should use the ordinary ItemBlock callback");
            };
            assert_eq!(placement.state.meta(), expected, "block {id}");
        }

        // A failed permission/collision branch remains a rejection and never manufactures a
        // local state write or decrement.  The server correction path can therefore replace the
        // successful prediction without a client-side fallback packet.
        assert_eq!(
            item_block_delegate_use(
                &world,
                clicked,
                Facing::Up,
                0.5,
                0.0,
                &dispenser,
                false,
                &[]
            ),
            BlockUseResult::Rejected
        );
        let blocked = Aabb::from_bounds(20.0, 65.0, 20.0, 21.0, 66.0, 21.0);
        assert_eq!(
            item_block_delegate_use(
                &world,
                clicked,
                Facing::Up,
                0.5,
                0.0,
                &dispenser,
                true,
                &[blocked]
            ),
            BlockUseResult::Rejected
        );
    }

    #[test]
    fn pre_c08_activation_bodies_cover_cake_pot_jukebox_diodes_and_ore() {
        let pos = BlockPos::new(12, 64, 12);

        let mut repeater = world_of(&[(pos, "stone")]);
        repeater.set_block(pos, BlockState::new(mc_data::BlockId(93), 0));
        assert!(
            activate_repeater(&mut repeater, pos, true)
                .unwrap()
                .consumed
        );
        assert_eq!(repeater.block_at(pos).meta(), 4);
        assert!(
            activate_repeater(&mut repeater, pos, true)
                .unwrap()
                .consumed
        );
        assert_eq!(repeater.block_at(pos).meta(), 8);
        assert!(
            !activate_repeater(&mut repeater, pos, false)
                .unwrap()
                .consumed
        );
        assert_eq!(repeater.block_at(pos).meta(), 8);

        let mut comparator = world_of(&[(pos, "stone")]);
        comparator.set_block(pos, BlockState::new(mc_data::BlockId(149), 8));
        assert!(
            activate_comparator(&mut comparator, pos, true)
                .unwrap()
                .consumed
        );
        assert_eq!(comparator.block_at(pos).meta(), 12);

        let mut jukebox = world_of(&[(pos, "stone")]);
        jukebox.set_block(pos, BlockState::new(mc_data::BlockId(84), 1));
        assert!(activate_jukebox(&mut jukebox, pos).unwrap().consumed);
        assert_eq!(jukebox.block_at(pos).meta(), 0);
        assert!(!activate_jukebox(&mut jukebox, pos).unwrap().consumed);

        let mut pot_world = world_of(&[(pos, "stone")]);
        pot_world.set_block(pos, BlockState::new(mc_data::BlockId(140), 0));
        let sapling = ItemStack {
            item_id: 6,
            count: 2,
            damage: 3,
            nbt: None,
        };
        // Discard setup writes so the following dirty set observes only the activation callback.
        pot_world.drain_dirty_sections();
        assert!(
            activate_flower_pot(&mut pot_world, pos, Some(&sapling))
                .unwrap()
                .consumed
        );
        assert!(
            pot_world
                .dirty_sections()
                .any(|section| section == pos.section_pos()),
            "successful flower-pot content insertion must immediately enqueue its section"
        );
        let pot = pot_world
            .tile_entity(pos)
            .and_then(|tile| tile.flower_pot())
            .expect("placing a flower pot creates its tile state");
        assert_eq!((pot.item_id, pot.data), (6, 3));
        // An occupied pot is Java's early `false` return: no tile mutation and no new
        // markBlockForUpdate/remesh signal.  Drain the success signal before this negative
        // control so a stale entry cannot make the assertion pass.
        pot_world.drain_dirty_sections();
        assert!(
            !activate_flower_pot(&mut pot_world, pos, Some(&sapling))
                .unwrap()
                .consumed
        );
        assert!(
            pot_world.dirty_sections().next().is_none(),
            "declined flower-pot activation must not enqueue a remesh"
        );

        let mut cake = FinishedFoodStats {
            food_level: 19,
            saturation: 19.5,
        };
        cake = cake_food_stats(cake.food_level, cake.saturation);
        assert_eq!(
            cake,
            FinishedFoodStats {
                food_level: 20,
                saturation: 19.9,
            }
        );
        assert_eq!(
            peaceful_food_regeneration(9, true, true, 10),
            Some(10),
            "Peaceful's tenth simulated tick refills one food point"
        );
        assert_eq!(
            peaceful_food_regeneration(9, true, true, 9),
            None,
            "the cadence is exactly every ten ticks"
        );
        assert_eq!(
            peaceful_food_regeneration(9, false, true, 10),
            None,
            "non-Peaceful worlds do not use EntityPlayer's special refill"
        );
        assert_eq!(
            peaceful_food_regeneration(20, true, true, 10),
            None,
            "full food never increments"
        );
        assert!(activate_dragon_egg(&pot_world, pos).is_none());
        pot_world.set_block(pos, BlockState::new(mc_data::BlockId(122), 0));
        assert!(activate_dragon_egg(&pot_world, pos).unwrap().consumed);
    }

    /// java: `Item.onItemUse` is `return false`, so anything without an override falls through to
    /// `sendUseItem`. The whole "cannot eat while looking at a block" family lives on this line.
    #[test]
    fn the_items_that_fall_through_to_send_use_item_are_the_ones_with_no_override() {
        for name in [
            "bread",
            "golden_apple",
            "bow",
            "potion",
            "milk_bucket",
            "diamond_sword",
            "ender_pearl",
            "snowball",
            "fishing_rod",
            "shears",
            "diamond_pickaxe",
            "compass",
        ] {
            assert!(
                !has_on_item_use_override(&stack(name, 1)),
                "{name} has no `onItemUse` override in 1.8.9"
            );
            assert!(placed_block(&stack(name, 1)).is_none(), "{name}");
        }
        // The negative control: these do override it, and are deliberately left alone.
        for name in [
            "flint_and_steel",
            "wooden_hoe",
            "wheat_seeds",
            "sign",
            "wooden_door",
            "minecart",
            "bed",
            "dye",
            "spawn_egg",
            "item_frame",
            "record_13",
            "armor_stand",
            "lead",
            "reeds",
            "repeater",
        ] {
            assert!(has_on_item_use_override(&stack(name, 1)), "{name}");
        }
    }

    /// java: `BlockLadder.onBlockPlaced` (`BlockLadder.java:91-108`) falls back to
    /// `for (EnumFacing f : EnumFacing.Plane.HORIZONTAL)` when the clicked face cannot hold the
    /// ladder (a top click), and that iterator is `{NORTH, EAST, SOUTH, WEST}`
    /// (`EnumFacing.java:454`). With supports both north and south of the cell the first match
    /// is NORTH (held by the south block). The scan used `Facing::HORIZONTALS` (S-W-N-E) and
    /// picked SOUTH: a predicted ladder on the other wall, colliding on the wrong side of the
    /// cell until the server's S23 arrived. `BlockTorch.onBlockPlaced` shares the loop.
    #[test]
    fn a_top_clicked_ladder_and_torch_take_java_plane_horizontal_first_support() {
        let floor = BlockPos::new(8, 64, 8);
        let target = floor.up();
        let world = world_of(&[
            (floor, "minecraft:stone"),
            (target.north(), "minecraft:stone"),
            (target.south(), "minecraft:stone"),
        ]);
        // Negative control: the old index-order scan really does pick the other wall here.
        let old_scan = Facing::HORIZONTALS
            .into_iter()
            .find(|f| world.is_normal_cube_at(target.relative(f.opposite())));
        assert_eq!(old_scan, Some(Facing::South));

        let ladder = stack("ladder", 16);
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &ladder,
            &[]
        ));
        let BlockUseResult::Consumed {
            placement: Some(placement),
            decrement_stack: true,
        } = item_block_delegate_use(&world, floor, Facing::Up, 1.0, 0.0, &ladder, true, &[])
        else {
            panic!("a supported ladder is placed");
        };
        assert_eq!(placement.pos, target);
        assert_eq!(
            placement.state,
            BlockState::new(mc_data::BlockId(65), Facing::North.index()),
            "java: NORTH is the first EnumFacing.Plane.HORIZONTAL facing with a support"
        );

        // `BlockTorch.onBlockPlaced` takes the same loop when the clicked face fails
        // `canPlaceAt`; a bottom click always does, and NORTH is torch metadata 4.
        let ceiling = BlockPos::new(8, 66, 8);
        let mut world = world;
        world.set_block(ceiling, stone());
        let torch = stack("torch", 1);
        let BlockUseResult::Consumed {
            placement: Some(placement),
            ..
        } = item_block_delegate_use(&world, ceiling, Facing::Down, 0.0, 0.0, &torch, true, &[])
        else {
            panic!("a torch with wall support below a ceiling is placed");
        };
        assert_eq!(placement.pos, target);
        assert_eq!(
            placement.state.meta(),
            4,
            "java: FACING=NORTH, not the S-W-N-E SOUTH (3)"
        );
    }

    /// java: `BlockTorch.canPlaceOn` accepts `block instanceof BlockFence`, which
    /// includes the five 1.8 wood fences (188..=192). With a wall beside the cell the old id list
    /// (85 and 113 only) failed the top click and fell back to a wall torch.
    #[test]
    fn a_torch_on_a_new_wood_fence_stands_up() {
        let fence = BlockPos::new(8, 64, 8);
        let target = fence.up();
        let world = world_of(&[
            (fence, "minecraft:spruce_fence"),
            (target.north(), "minecraft:stone"),
        ]);
        let torch = stack("torch", 1);
        let BlockUseResult::Consumed {
            placement: Some(placement),
            ..
        } = item_block_delegate_use(&world, fence, Facing::Up, 1.0, 0.0, &torch, true, &[])
        else {
            panic!("a torch on a fence is placed");
        };
        assert_eq!(placement.pos, target);
        assert_eq!(placement.state.meta(), 5, "java: FACING=UP on a BlockFence");
    }

    /// java: `World.canBlockBePlaced` tests `blockIn.getCollisionBoundingBox(this,
    /// pos, blockIn.getDefaultState())` against every live `preventEntitySpawning` entity, the
    /// local player included. The gate used `BlockState::new(block, 0)`, and metadata zero is not
    /// a legal ladder, furnace, chest or ender chest state: it read as air, which has no collision
    /// volume, so the entity test was skipped and a ladder (sold in BedWars) was predicted inside
    /// the player where Java refuses it. The Java box for a ladder is its stale mutable bounds;
    /// the ladder being placed stands in (see `can_block_be_placed`).
    #[test]
    fn a_ladder_or_furnace_is_refused_where_its_box_overlaps_the_player() {
        let wall = BlockPos::new(8, 65, 7);
        let target = wall.south();
        let world = world_of(&[
            (wall, "minecraft:stone"),
            (target.down(), "minecraft:stone"),
        ]);
        let ladder = stack("ladder", 16);
        // Negative control for the old gate: metadata zero is illegal for both blocks, reads as
        // air and has no collision volume, so `BlockState::new(block, 0)` never ran the test.
        for block in [65_u16, 61] {
            let meta_zero = BlockState::new(mc_data::BlockId(block), 0);
            assert!(!meta_zero.is_legal() && meta_zero.props().collision.is_empty());
        }
        // A SOUTH-facing ladder on the north wall: `setBlockBounds(0, 0, 0, 1, 1, 0.125)`.
        let touching_wall = Aabb::from_bounds(8.2, 65.0, 8.0, 8.8, 66.8, 8.6);
        let on_slab_face = Aabb::from_bounds(8.2, 65.0, 8.125, 8.8, 66.8, 8.725);

        assert!(!can_place_block_on_side(
            &world,
            wall,
            Facing::South,
            &ladder,
            &[touching_wall]
        ));
        assert!(matches!(
            item_block_delegate_use(
                &world,
                wall,
                Facing::South,
                0.5,
                0.0,
                &ladder,
                true,
                &[touching_wall]
            ),
            BlockUseResult::Rejected
        ));
        // Strict inequality: a box resting on the slab's face does not overlap it.
        assert!(can_place_block_on_side(
            &world,
            wall,
            Facing::South,
            &ladder,
            &[on_slab_face]
        ));
        assert!(matches!(
            item_block_delegate_use(
                &world,
                wall,
                Facing::South,
                0.5,
                0.0,
                &ladder,
                true,
                &[on_slab_face]
            ),
            BlockUseResult::Consumed {
                placement: Some(_),
                ..
            }
        ));

        // The furnace's default state (FACING=NORTH) is a full cube, so the same clear box is
        // inside it; an empty entity list is the control.
        let furnace = stack("furnace", 1);
        assert!(!can_place_block_on_side(
            &world,
            wall,
            Facing::South,
            &furnace,
            &[on_slab_face]
        ));
        assert!(can_place_block_on_side(
            &world,
            wall,
            Facing::South,
            &furnace,
            &[]
        ));
    }

    /// The entity test is a strict-inequality AABB overlap, so at a flush face the
    /// exact entity box decides it. `exact` is a box a move clamped onto x = 4 (its `minX` and
    /// `maxX` rounded independently on the way there); `rebuilt` is the same box rebuilt from its
    /// derived centre as `posX -+ width / 2`, which lands one ulp inside the cell west of it and
    /// refused a placement Java accepts. `Client::placement_blockers` now hands over the former.
    #[test]
    fn the_gate_decides_a_flush_face_on_the_exact_entity_box() {
        let half = f64::from(0.6_f32) / 2.0;
        let exact = Aabb::from_bounds(4.0, 65.0, 8.2, 4.600_000_023_841_856, 66.8, 8.8);
        let centre = (exact.min.x + exact.max.x) / 2.0;
        let rebuilt = Aabb::from_bounds(centre - half, 65.0, 8.2, centre + half, 66.8, 8.8);
        assert!(
            rebuilt.min.x < exact.min.x,
            "the rebuilt box pokes into x < 4"
        );

        let support = BlockPos::new(3, 64, 8);
        let world = world_of(&[(support, "minecraft:stone")]);
        let wool = stack("wool", 64);
        // The support's top face targets (3, 65, 8), x in [3, 4]: flush with the player.
        assert!(can_place_block_on_side(
            &world,
            support,
            Facing::Up,
            &wool,
            &[exact]
        ));
        assert!(!can_place_block_on_side(
            &world,
            support,
            Facing::Up,
            &wool,
            &[rebuilt]
        ));
    }

    /// java: `ItemDoor.placeDoor` mirrors the hinge only for `getBlock() == door`,
    /// the *same* door block (`ItemDoor.java:64-65`). A spruce door beside a new oak door is not
    /// one, and with no solid blocks either side the hinge stays LEFT (upper metadata 8). The old
    /// "any door" test flipped it RIGHT, which moves the predicted door's collision slab.
    #[test]
    fn a_door_mirrors_only_the_same_door_block() {
        let floor = BlockPos::new(8, 64, 8);
        let target = floor.up();
        // Yaw 0 is SOUTH (`EnumFacing.fromAngle`), whose `rotateYCCW` side is EAST.
        let east = target.east();
        let mut world = world_of(&[(floor, "minecraft:stone"), (east.down(), "minecraft:stone")]);
        world.set_block(east, BlockState::new(mc_data::BlockId(193), 0));
        world.set_block(east.up(), BlockState::new(mc_data::BlockId(193), 8));

        let BlockUseResult::ConsumedPair { placements, .. } = door_item_use(
            &world,
            floor,
            Facing::Up,
            0.0,
            &stack("wooden_door", 1),
            true,
        ) else {
            panic!("an oak door fits");
        };
        assert_eq!(placements[1].pos, target.up());
        assert_eq!(
            placements[1].state.meta(),
            8,
            "hinge LEFT: spruce is not this door"
        );

        // Positive control: an oak door in the same place does mirror the hinge.
        world.set_block(east, BlockState::new(mc_data::BlockId(64), 0));
        world.set_block(east.up(), BlockState::new(mc_data::BlockId(64), 8));
        let BlockUseResult::ConsumedPair { placements, .. } = door_item_use(
            &world,
            floor,
            Facing::Up,
            0.0,
            &stack("wooden_door", 1),
            true,
        ) else {
            panic!("an oak door fits");
        };
        assert_eq!(
            placements[1].state.meta(),
            9,
            "hinge RIGHT beside the same door"
        );
    }

    /// java: `ItemBucket.onItemRightClick` empties a liquid into the bucket (and
    /// sets it to air on the client) only when `BlockLiquid.LEVEL == 0` (`:62`, `:69`), and
    /// `LEVEL` is the whole metadata: a falling block (8) is not a source. The old `meta & 7`
    /// test predicted air inside a waterfall, which the server never confirms.
    #[test]
    fn an_empty_bucket_only_empties_a_level_zero_source_locally() {
        let pos = BlockPos::new(8, 64, 8);
        let mut world = world_of(&[(pos, "minecraft:stone")]);
        let water = mc_data::block_by_name("water").expect("water");
        let eye = DVec3::new(8.5, 66.0, 8.5);
        let bucket = stack("bucket", 1);
        // Positive control: a source block under a straight-down look is filled.
        world.set_block(pos, BlockState::new(water, 0));
        assert!(matches!(
            liquid_item_use(&world, eye, 0.0, 90.0, &bucket, true, false),
            Some(LiquidUsePrediction::BucketFill { source, .. }) if source == pos
        ));
        world.set_block(pos, BlockState::new(water, 8));
        assert!(
            liquid_item_use(&world, eye, 0.0, 90.0, &bucket, true, false).is_none(),
            "falling water (LEVEL 8) is never emptied locally"
        );
    }

    /// java: `BlockFence.onBlockActivated` is `return worldIn.isRemote ? true :
    /// ItemLead.attachToFence(...)` (`BlockFence.java:172-175`) for every `BlockFence`
    /// registration, and `nether_brick_fence` (113) is one (`Block.java:1375`). The table left it
    /// out, so right-clicking one with a block in hand predicted a placement beside it that the
    /// vanilla client never makes.
    #[test]
    fn every_block_fence_consumes_a_client_right_click() {
        let pos = BlockPos::new(8, 64, 8);
        for id in [85_u16, 113, 188, 189, 190, 191, 192] {
            let mut world = world_of(&[(pos, "minecraft:stone")]);
            world.set_block(pos, BlockState::new(mc_data::BlockId(id), 0));
            let activation =
                activate_remote_block(&world, pos, true).expect("every BlockFence activates");
            assert!(activation.consumed, "fence {id}");
        }
        // Negative control: a cobblestone wall is no BlockFence and has no override at all.
        let mut world = world_of(&[(pos, "minecraft:stone")]);
        world.set_block(pos, BlockState::new(mc_data::BlockId(139), 0));
        assert!(activate_remote_block(&world, pos, true).is_none());
    }

    /// java: `ItemBlock.canPlaceBlockOnSide` -> `World.canBlockBePlaced`. The audit's headline
    /// measurement — **151/151 into an occupied cell** — is this case.
    #[test]
    fn an_occupied_destination_cell_refuses_the_placement() {
        let floor = BlockPos::new(0, 64, 0);
        let above = BlockPos::new(0, 65, 0);
        let world = world_of(&[(floor, "minecraft:stone"), (above, "minecraft:stone")]);
        let dirt = stack("dirt", 64);

        // Clicking the top face of the floor targets the cell above it, which is stone.
        assert!(!can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &dirt,
            &[]
        ));
        // Clicking a side face targets an air cell, which is replaceable.
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::North,
            &dirt,
            &[]
        ));
        // A non-`ItemBlock` is not gated at all — vanilla's `instanceof ItemBlock` never fires.
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("bread", 1),
            &[]
        ));
    }

    /// **W-RCLICK's last piece.** `World.canBlockBePlaced`'s tail is
    /// `isReplaceable() && blockIn.canReplace(...)`, and `canReplace` -> `canPlaceBlockOnSide`
    /// -> `canPlaceBlockAt`. With the 28+7 overrides absent, that whole chain collapsed to the
    /// `isReplaceable` test the first conjunct already made, so the gate was a strict subset of
    /// vanilla's and a torch on thin air reached the wire as a `C08`.
    ///
    /// Every case has its paired control in the *same* world, so a gate that simply refused
    /// everything would fail as loudly as the one that allowed everything.
    ///
    /// **The block is at `(8, 64, 8)` on purpose.** `World.isBlockNormalCube(pos, true)`
    /// returns its `default` for a cell in an unloaded chunk, and on a client
    /// `ChunkProviderClient.provideChunk` hands out an `EmptyChunk` rather than null — so a
    /// column at `x = 0` has *four* neighbours in chunks this fixture never loaded, every one
    /// of which answers "yes, a normal cube" and silently holds the torch up. That is vanilla's
    /// behaviour, faithfully ported, and it is also exactly how this test first passed while
    /// asserting the opposite of what it meant.
    #[test]
    fn the_per_block_place_predicates_reach_the_gate() {
        let floor = BlockPos::new(8, 64, 8);
        let world = world_of(&[(floor, "minecraft:stone")]);

        // java: `BlockTorch.canPlaceBlockAt` (`:71`) tries every `FACING` value, which excludes
        // `DOWN`, so the cell under a block has no support at all.
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("torch", 1),
            &[]
        ));
        assert!(
            !can_place_block_on_side(&world, floor, Facing::Down, &stack("torch", 1), &[]),
            "java: a torch cannot hang from the underside of a block"
        );

        // java: `BlockRailBase.canPlaceBlockAt` (`:85`) is
        // `doesBlockHaveSolidTopSurface(worldIn, pos.down())`, so a rail placed against a side
        // face — whose target cell has air below it — is refused.
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("rail", 1),
            &[]
        ));
        assert!(
            !can_place_block_on_side(&world, floor, Facing::North, &stack("rail", 1), &[]),
            "java: a rail needs a solid top surface under it"
        );

        // java: `BlockVine.canPlaceBlockOnSide` (`:157`) is the cheapest proof that `side`
        // itself now reaches the overrides rather than only the target cell: its `switch` reads
        // `pos.up()` for `UP` and `pos.offset(side.getOpposite())` for the four horizontals, and
        // has **no `DOWN` arm at all**. Same block, same item, two faces, opposite answers.
        assert!(
            can_place_block_on_side(&world, floor, Facing::North, &stack("vine", 1), &[]),
            "the stone is south of the target cell, which is what a vine hangs on"
        );
        assert!(
            !can_place_block_on_side(&world, floor, Facing::Up, &stack("vine", 1), &[]),
            "java: the UP arm reads `pos.up()`, which is air"
        );

        // The control: a plain full cube is gated by nothing but the occupancy test, and still
        // places into the same air cell the rail and the vine were just judged on.
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::North,
            &stack("dirt", 1),
            &[]
        ));
    }

    /// `ItemReed` is the registration used by the repeater and comparator items (356/404). Their
    /// `onItemUse` placement predicate (evaluated here) runs *after* the live controller has
    /// queued the targeted C08; it is not the pre-C08 `ItemBlock` gate (L6).
    /// Both diode items share the `BlockRedstoneDiode` solid-top rule, while the outer material
    /// check keeps an occupied target from slipping through the direct predicate.
    #[test]
    fn diode_item_reed_placement_uses_support_and_occupancy_gates() {
        let floor = BlockPos::new(8, 64, 8);
        let target = floor.up();
        let mut world = world_of(&[(floor, "minecraft:stone")]);

        assert_eq!(
            item_reed_block(&stack("reeds", 1)),
            Some(mc_data::BlockId(83))
        );

        for name in ["repeater", "comparator"] {
            let item = stack(name, 1);
            let expected = if name == "repeater" {
                mc_data::BlockId(93)
            } else {
                mc_data::BlockId(149)
            };
            assert_eq!(item_reed_block(&item), Some(expected), "{name}");
            assert!(can_place_block_on_side(
                &world,
                floor,
                Facing::Up,
                &item,
                &[],
            ));

            // The target is replaceable above a solid top, but not once occupied.
            world.set_block(target, BlockState::new(mc_data::BlockId(1), 0));
            assert!(!can_place_block_on_side(
                &world,
                floor,
                Facing::Up,
                &item,
                &[],
            ));
            world.set_block(target, BlockState::AIR);

            // Glass is non-solid-top even though the destination above it is replaceable.
            world.set_block(floor, BlockState::new(mc_data::BlockId(20), 0));
            assert!(!can_place_block_on_side(
                &world,
                floor,
                Facing::Up,
                &item,
                &[],
            ));
            world.set_block(floor, stone());
        }

        // The helper must remain a narrow ItemReed bridge: ordinary generic items are not gated.
        assert!(item_reed_block(&stack("bread", 1)).is_none());
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("bread", 1),
            &[],
        ));
    }

    /// The live `PlayerControllerMP` ItemBlock gate must carry the provider-specific banner and
    /// pressure-plate predicates through `World.canBlockBePlaced`: horizontal cactus rejects a
    /// banner, while a plate accepts stone/fence support and rejects glass. The occupied-target
    /// control proves the pressure-plate override cannot bypass the outer replaceability test.
    #[test]
    fn banner_and_pressure_plate_predicates_reach_the_live_item_block_gate() {
        let floor = BlockPos::new(8, 64, 8);
        let target = floor.up();
        let mut world = world_of(&[(floor, "minecraft:stone")]);

        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("banner", 1),
            &[]
        ));
        world.set_block(
            target.relative(Facing::North),
            BlockState::new(mc_data::BlockId(81), 0),
        );
        assert!(!can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("banner", 1),
            &[]
        ));

        world.set_block(target.relative(Facing::North), BlockState::AIR);
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("stone_pressure_plate", 1),
            &[]
        ));
        world.set_block(floor, BlockState::new(mc_data::BlockId(20), 0));
        assert!(!can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("stone_pressure_plate", 1),
            &[]
        ));
        world.set_block(floor, BlockState::new(mc_data::BlockId(85), 0));
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("stone_pressure_plate", 1),
            &[]
        ));

        // The direct pressure-plate predicate accepts its support even when the target is
        // occupied; the live gate must still reject that placement at the outer material check.
        world.set_block(target, BlockState::new(mc_data::BlockId(1), 0));
        assert!(!can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("stone_pressure_plate", 1),
            &[]
        ));
    }

    /// **W-RCLICK.** `ItemBlock.canPlaceBlockOnSide` (`:126-140`) reassigns `side` to `UP` when
    /// the clicked block is a snow layer, and leaves `pos` alone. Until the per-block
    /// predicates were wired that reassignment reached nothing and the code said so; it is
    /// live now, and this is what it changes.
    ///
    /// A lever clicked on the *north* face of a snow layer sitting on stone: vanilla asks
    /// `func_181088_a(world, pos, UP.getOpposite())`, i.e. "is the block **below** a normal
    /// cube" — the stone, yes. Without the reassignment it would ask about the cell to the
    /// south, which is air, and the client would suppress a `C08` vanilla sends.
    #[test]
    fn a_snow_layer_forces_the_placement_side_to_up() {
        let stone = BlockPos::new(8, 64, 8);
        let snow = BlockPos::new(8, 65, 8);
        let world = world_of(&[(stone, "minecraft:stone"), (snow, "minecraft:snow_layer")]);

        assert!(
            can_place_block_on_side(&world, snow, Facing::North, &stack("lever", 1), &[]),
            "java: side becomes UP, so the support tested is the stone below the snow"
        );
        // The control: the *same* face of a snow layer with nothing underneath it. If the
        // reassignment were a no-op both cases would answer alike (the cell south of the snow
        // is air in both worlds); it is the block below that separates them, and only the
        // reassigned `UP` can reach it.
        let floating = BlockPos::new(5, 70, 5);
        let thin_air = world_of(&[(floating, "minecraft:snow_layer")]);
        assert!(!can_place_block_on_side(
            &thin_air,
            floating,
            Facing::North,
            &stack("lever", 1),
            &[]
        ));
    }

    /// java: `checkNoEntityCollision(bb, null)` with `entityIn == null`, so the *placing player*
    /// is one of the entities tested. This is why you cannot place a block inside your own feet.
    #[test]
    fn an_entity_standing_in_the_destination_cell_refuses_the_placement() {
        let floor = BlockPos::new(0, 64, 0);
        let world = world_of(&[(floor, "minecraft:stone")]);
        let dirt = stack("dirt", 64);
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &dirt,
            &[]
        ));
        let player = Aabb::from_bounds(-0.3, 65.0, -0.3, 0.3, 66.8, 0.3);
        assert!(!can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &dirt,
            &[player]
        ));
        // java: `getCollisionBoundingBox` returns null for a torch, so the entity test is skipped
        // entirely and a torch can be placed through a mob.
        assert!(can_place_block_on_side(
            &world,
            floor,
            Facing::Up,
            &stack("torch", 64),
            &[player]
        ));
    }

    /// java: `World.canBlockBePlaced` tests entities against the placed
    /// block's `getCollisionBoundingBox`, which five placeable blocks override with a box that is
    /// not their bounds. Each case puts an entity where only one of the two boxes reaches it.
    #[test]
    fn the_entity_test_uses_javas_overridden_collision_boxes() {
        let floor = BlockPos::new(0, 63, 0);
        let world = world_of(&[(floor, "minecraft:stone")]);
        let place = |item: &str, blocking: &[Aabb]| {
            can_place_block_on_side(&world, floor, Facing::Up, &stack(item, 8), blocking)
        };
        // A player standing on the floor in the destination cell (0, 64, 0).
        let standing = Aabb::from_bounds(0.2, 64.0, 0.2, 0.8, 65.8, 0.8);
        // BlockSnow: one layer collides as a zero-height box at y = 64, which Java's strict
        // `intersectsWith` does not find a box resting on y = 64 inside (the oracle's
        // `place_stone_cases` click). Its 0.125-high bounds refused it.
        assert!(
            place("snow_layer", &[standing]),
            "a snow layer under the feet"
        );
        // ...but a box crossing the y = 64 plane is inside the zero-height box.
        let crossing = Aabb::from_bounds(0.2, 63.5, 0.2, 0.8, 65.3, 0.8);
        assert!(!place("snow_layer", &[crossing]));
        // A full block refuses the standing player either way (the control).
        assert!(!place("stone", &[standing]));

        // BlockFenceGate: closed, 1.5 high. A player standing one block higher (feet at y = 65)
        // and overlapping the cell's middle quarter is inside it; the 1.0-high bounds are not.
        let above = Aabb::from_bounds(-0.05, 65.0, 0.2, 0.55, 66.8, 0.8);
        assert!(
            !place("fence_gate", &[above]),
            "the gate's box reaches y = 65.5"
        );
        assert!(place("fence_gate", &[]));
        // Off the gate's middle quarter (z 0.375..0.625) the same height is clear.
        let beside = Aabb::from_bounds(-0.05, 65.0, 0.65, 0.55, 66.8, 1.25);
        assert!(place("fence_gate", &[beside]));

        // BlockSoulSand: 0.875 high, so a box resting at y = 64.875 is clear of it.
        let on_soul_sand = Aabb::from_bounds(0.95, 64.875, 0.2, 1.55, 66.675, 0.8);
        assert!(place("soul_sand", &[on_soul_sand]));
        let lower = Aabb::from_bounds(0.95, 64.8, 0.2, 1.55, 66.6, 0.8);
        assert!(!place("soul_sand", &[lower]));

        // BlockCactus: 1/16 in from each side. Cactus needs sand under it.
        let sand_world = world_of(&[(floor, "minecraft:sand")]);
        let cactus = |blocking: &[Aabb]| {
            can_place_block_on_side(
                &sand_world,
                floor,
                Facing::Up,
                &stack("cactus", 8),
                blocking,
            )
        };
        let touching_the_rim = Aabb::from_bounds(0.97, 64.0, 0.2, 1.57, 65.8, 0.8);
        assert!(
            cactus(&[touching_the_rim]),
            "x 0.97 is past the inset 0.9375"
        );
        let inside_the_rim = Aabb::from_bounds(0.9, 64.0, 0.2, 1.5, 65.8, 0.8);
        assert!(!cactus(&[inside_the_rim]));

        // BlockFarmland: the full cube, though it is drawn 15/16 high.
        let at_fifteen_sixteenths = Aabb::from_bounds(0.2, 64.95, 0.2, 0.8, 66.75, 0.8);
        assert!(!place("farmland", &[at_fifteen_sixteenths]));
    }

    /// java: `Block.isReplaceable(World, BlockPos)` (`:387`) is `return false` on the base class
    /// and true for six blocks. It decides whether the placement lands *in* the clicked cell.
    #[test]
    fn a_replaceable_block_is_overwritten_in_place_rather_than_offset() {
        let grass_pos = BlockPos::new(0, 65, 0);
        let world = world_of(&[
            (BlockPos::new(0, 64, 0), "minecraft:stone"),
            (grass_pos, "minecraft:tallgrass"),
            (BlockPos::new(0, 66, 0), "minecraft:stone"),
        ]);
        let dirt = stack("dirt", 64);
        // Clicking the tall grass's top face: it is `isReplaceable`, so the target stays at the
        // grass cell (replaceable material) rather than stepping up into the stone above it.
        assert!(can_place_block_on_side(
            &world,
            grass_pos,
            Facing::Up,
            &dirt,
            &[]
        ));

        // The same geometry with a non-replaceable block in the middle steps up into stone.
        let world = world_of(&[
            (BlockPos::new(0, 64, 0), "minecraft:stone"),
            (grass_pos, "minecraft:cobblestone"),
            (BlockPos::new(0, 66, 0), "minecraft:stone"),
        ]);
        assert!(!can_place_block_on_side(
            &world,
            grass_pos,
            Facing::Up,
            &dirt,
            &[]
        ));
    }

    /// java: `ItemSlab.canPlaceBlockOnSide` (`ItemSlab.java:85`) — the one item override that
    /// changes which cell is tested. A matching bottom slab clicked on its top face accepts the
    /// placement *in place*, where the plain `ItemBlock` chain would step up and (here) succeed
    /// for the wrong reason, or fail when the cell above is occupied.
    #[test]
    fn a_matching_slab_merges_in_place_where_a_plain_item_block_would_not() {
        let slab_pos = BlockPos::new(0, 65, 0);
        let world = world_of(&[
            (BlockPos::new(0, 64, 0), "minecraft:stone"),
            (slab_pos, "minecraft:stone_slab"),
            (BlockPos::new(0, 66, 0), "minecraft:stone"),
        ]);
        let slab = stack("stone_slab", 64);
        // Bottom slab (meta 0), clicked on UP, same variant: `ItemSlab` returns true outright.
        assert!(can_place_block_on_side(
            &world,
            slab_pos,
            Facing::Up,
            &slab,
            &[]
        ));
        // A plain `ItemBlock` in the same spot steps up into the stone and is refused, which is
        // the negative control proving the slab branch is what answered above.
        assert!(!can_place_block_on_side(
            &world,
            slab_pos,
            Facing::Up,
            &stack("dirt", 64),
            &[]
        ));
        // A different slab variant does not merge, and falls through to `super`.
        let other = ItemStack {
            damage: 3,
            ..slab.clone()
        };
        assert!(!can_place_block_on_side(
            &world,
            slab_pos,
            Facing::Up,
            &other,
            &[]
        ));
    }

    /// java: `WorldBorder.contains(BlockPos)` (`:34`).
    #[test]
    fn the_world_border_excludes_the_block_it_should() {
        use crate::sync::env::BorderState;
        let border = BorderState {
            center_x: 0.0,
            center_z: 0.0,
            old_diameter: 16.0,
            new_diameter: 16.0,
            speed_millis: 0,
            ..BorderState::default()
        };
        // minX/minZ = -8, maxX/maxZ = 8.
        assert!(border_contains(&border, BlockPos::new(0, 64, 0), 0));
        assert!(border_contains(&border, BlockPos::new(7, 64, 7), 0));
        // java: `(pos.getX() + 1) > minX`, so -9 is out but -8 is in.
        assert!(border_contains(&border, BlockPos::new(-8, 64, 0), 0));
        assert!(!border_contains(&border, BlockPos::new(-9, 64, 0), 0));
        // java: `pos.getX() < maxX`, so 8 is out.
        assert!(!border_contains(&border, BlockPos::new(8, 64, 0), 0));
        // The default 60,000,000-block border contains everything a player can reach.
        let wide = BorderState::default();
        assert!(border_contains(&wide, BlockPos::new(1_000_000, 64, -3), 0));
    }

    /// The interaction gate must consume the same wall-clock lerp as the landed border state,
    /// rather than jumping straight to the packet's destination diameter. The midpoint block is
    /// inside a 15-wide transition but outside its 10-wide destination; the strict upper edge is
    /// excluded at both samples.
    #[test]
    fn world_border_interaction_gate_uses_the_lerped_diameter_and_strict_edges() {
        use crate::sync::env::BorderState;
        let border = BorderState {
            center_x: 0.0,
            center_z: 0.0,
            old_diameter: 20.0,
            new_diameter: 10.0,
            speed_millis: 1_000,
            start_time_millis: 0,
            end_time_millis: 1_000,
            ..BorderState::default()
        };

        // At t=500 Java's float-ratio lerp gives diameter 15: x=6 is inside, x=7 is inside,
        // and x=8 is outside. At t=1000 the transition is 10 wide, so x=6 is outside.
        assert!(border_contains(&border, BlockPos::new(6, 64, 0), 500));
        assert!(border_contains(&border, BlockPos::new(7, 64, 0), 500));
        assert!(!border_contains(&border, BlockPos::new(8, 64, 0), 500));
        assert!(!border_contains(&border, BlockPos::new(6, 64, 0), 1_000));

        // The lower edge uses Java's `(x + 1) > minX`: -8 is still in a 15-wide border
        // (min=-7.5), while -9 is out.
        assert!(border_contains(&border, BlockPos::new(-8, 64, 0), 500));
        assert!(!border_contains(&border, BlockPos::new(-9, 64, 0), 500));
    }

    /// java: `f < 0.0F ? 0.0F` — bedrock never progresses, which is what stops the client
    /// predicting a break the server will never confirm.
    #[test]
    fn unbreakable_blocks_never_accumulate_damage() {
        let bedrock = BlockState::new(mc_data::block_by_name("bedrock").expect("bedrock"), 0);
        assert_eq!(player_relative_hardness(bedrock, true), 0.0);
    }

    // ------------------------------------------------------------------ PlayerControllerMP

    fn hit_at(pos: BlockPos, face: Facing) -> BlockHit {
        BlockHit {
            pos,
            face,
            hit_vec: pos.to_vec3(),
        }
    }

    /// The whole point of `blockHitDelay`: a creative click sends one START and then goes quiet
    /// for five ticks, rather than one START per tick for as long as the button is held.
    #[test]
    fn creative_digging_sends_one_start_every_five_ticks() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = hit_at(target, Facing::Up);
        let mut controller = PlayerController::new();
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Creative, true, None, true),
            DigStep::StartAndDestroy(target, Facing::Up)
        );
        for tick in 0..5 {
            let tick_result =
                controller.damage_block(&world, hit, GameMode::Creative, true, None, true);
            assert_eq!(
                tick_result.step,
                DigStep::None,
                "tick {tick} broke the five-tick cooldown"
            );
            // java: `if (this.blockHitDelay > 0) { --this.blockHitDelay; return true; }` — the
            // cooldown ticks send nothing and swing anyway.
            assert!(tick_result.progressed, "tick {tick} must still swing");
        }
        assert_eq!(
            controller.damage_block(&world, hit, GameMode::Creative, true, None, true),
            DamageTick::sending(DigStep::StartAndDestroy(target, Facing::Up))
        );
    }

    /// java: `onPlayerDamageBlock` only calls `clickBlock` when the held target/item identity
    /// changes.  A live adventure/allow-edit change therefore does not cancel an already-held
    /// survival dig, while the first tick on a new target does re-enter the permission gate.
    #[test]
    fn a_mid_dig_permission_change_does_not_clear_same_target_progress() {
        let target = BlockPos::new(0, 64, 0);
        let other = BlockPos::new(1, 64, 0);
        let mut world = world_with(target);
        world.set_block(other, stone());
        let hit = hit_at(target, Facing::Up);
        let mut controller = PlayerController::new();
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Survival, true, None, true),
            DigStep::Start(target, Facing::Up)
        );
        let before = controller.damage();

        // `Adventure + !allowEdit + no CanDestroy` would be denied by clickBlock.  Java skips
        // that gate while `isHittingPosition` still matches, and continues the accumulation.
        let continued =
            controller.damage_block(&world, hit, GameMode::Adventure, false, None, true);
        assert!(continued.progressed);
        assert!(controller.damage() > before);
        assert_eq!(controller.hitting_block(), Some(target));

        // A different target is the gate re-entry point, so the same permission state refuses it
        // without clearing the old target's state.
        let refused = controller.damage_block(
            &world,
            hit_at(other, Facing::Up),
            GameMode::Adventure,
            false,
            None,
            true,
        );
        assert_eq!(refused, DamageTick::stalled());
        assert_eq!(controller.hitting_block(), Some(target));
    }

    /// java: the creative branch tests the world border only after the five-tick
    /// `blockHitDelay` branch.  Shrinking the border during that cooldown therefore still
    /// consumes all five delay ticks and returns a swing before the outside-border refusal.
    #[test]
    fn creative_border_shrink_during_cooldown_still_consumes_all_five_ticks() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = hit_at(target, Facing::Up);
        let mut controller = PlayerController::new();
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Creative, true, None, true),
            DigStep::StartAndDestroy(target, Facing::Up)
        );

        for tick in 0..5 {
            let result = controller.damage_block_with_border(
                &world,
                hit,
                GameMode::Creative,
                true,
                None,
                true,
                false,
            );
            assert_eq!(
                result.step,
                DigStep::None,
                "cooldown tick {tick} emitted a C07"
            );
            assert!(result.progressed, "cooldown tick {tick} must still swing");
        }
        assert_eq!(
            controller.damage_block_with_border(
                &world,
                hit,
                GameMode::Creative,
                true,
                None,
                true,
                false,
            ),
            DamageTick::stalled(),
            "only after the delay expires does the outside border refuse the click"
        );
    }

    /// java: `PlayerControllerMP.onPlayerDamageBlock:304` checks the counter *before* it
    /// increments it.  This is one mining sound on damage ticks 0, 4, 8, ...; the initial
    /// `clickBlock` only sends START and does not consume a sound beat.
    #[test]
    fn block_hit_sound_is_on_zero_then_every_fourth_accumulating_tick() {
        let target = BlockPos::new(10, 64, -3);
        let world = world_with(target);
        let hit = hit_at(target, Facing::North);
        let mut controller = PlayerController::new();
        controller.click_block(&world, hit, GameMode::Survival, true, None, true);

        let sounding_ticks: Vec<_> = (0..10)
            .filter_map(|tick| {
                controller
                    .damage_block(&world, hit, GameMode::Survival, true, None, true)
                    .hit_sound
                    .map(|sound| (tick, sound))
            })
            .collect();
        assert_eq!(
            sounding_ticks
                .iter()
                .map(|(tick, _)| *tick)
                .collect::<Vec<_>>(),
            vec![0, 4, 8],
            "a post-increment check would instead produce 3, 7, ..."
        );

        let request = sounding_ticks[0].1.play_request();
        assert_eq!(request.event.path, "step.stone");
        assert_eq!(request.volume, 0.25, "(SoundType.volume + 1) / 8");
        assert_eq!(request.pitch, 0.5, "SoundType.frequency * 0.5");
        assert_eq!(
            request.position,
            Some(DVec3::new(10.5, 64.5, -2.5)),
            "PositionedSoundRecord's BlockPos overload centres the source"
        );
        assert_eq!(request.attenuation, mc_audio::AttenuationType::Linear);
    }

    /// The sound's event comes from `SoundType.getStepSound`, including its otherwise easy to
    /// miss slime override.  The anvil's low sound-type volume is also a useful arithmetic
    /// control: incorrectly using the footstep `* 0.15` rule happens to pass on stone.
    #[test]
    fn block_hit_sound_uses_step_sound_overrides_and_mining_scales() {
        let pos = BlockPos::new(0, 64, 0);
        let state = |name| BlockState::new(mc_data::block_by_name(name).unwrap(), 0);

        let slime = BlockHitSound {
            pos,
            state: state("slime"),
        }
        .play_request();
        assert_eq!(slime.event.path, "mob.slime.small");
        assert_eq!(slime.volume, 0.25);
        assert_eq!(slime.pitch, 0.5);

        let anvil = BlockHitSound {
            pos,
            state: state("anvil"),
        }
        .play_request();
        assert_eq!(anvil.event.path, "step.anvil");
        assert_eq!(anvil.volume, 0.1625, "(0.3 + 1) / 8");
        assert_eq!(anvil.pitch, 0.5);
    }

    /// Cancelling or stalling a dig does not consume/reset a hidden audio beat.  In particular,
    /// an air update is the early-return branch before both the cadence increment and Java's
    /// sound construction; a replacement target starts its own counter from zero.
    #[test]
    fn block_hit_sound_resets_only_on_a_new_dig_and_not_on_non_progress() {
        let first = BlockPos::new(0, 64, 0);
        let second = BlockPos::new(1, 64, 0);
        let mut world = world_with(first);
        world.set_block(second, stone());
        let mut controller = PlayerController::new();
        let first_hit = hit_at(first, Facing::Up);
        let second_hit = hit_at(second, Facing::Up);
        controller.click_block(&world, first_hit, GameMode::Survival, true, None, true);
        assert!(
            controller
                .damage_block(&world, first_hit, GameMode::Survival, true, None, true)
                .hit_sound
                .is_some(),
            "the newly-started counter is zero"
        );

        // The target disappears: this must return false and emit no sound, without advancing a
        // counter that could make the next actual dig begin off cadence.
        world.set_block(first, BlockState::AIR);
        let stalled =
            controller.damage_block(&world, first_hit, GameMode::Survival, true, None, true);
        assert!(!stalled.progressed);
        assert!(stalled.hit_sound.is_none());

        // `damage_block` must call `clickBlock` for the new target (START only), then its first
        // accumulating tick begins at zero and sounds immediately.
        let start =
            controller.damage_block(&world, second_hit, GameMode::Survival, true, None, true);
        assert_eq!(start.step, DigStep::Start(second, Facing::Up));
        assert!(start.hit_sound.is_none());
        assert!(
            controller
                .damage_block(&world, second_hit, GameMode::Survival, true, None, true)
                .hit_sound
                .is_some(),
            "a stale counter after the air stall would suppress this beat"
        );
    }

    /// Survival: one START, then damage accumulates, then exactly one STOP. A client that sent
    /// STOP early gets the block back from the server; one that never sent it never breaks it.
    #[test]
    fn survival_digging_accumulates_to_one_stop() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = hit_at(target, Facing::Up);
        let mut controller = PlayerController::new();
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Survival, true, None, true),
            DigStep::Start(target, Facing::Up)
        );
        assert_eq!(controller.hitting_block(), Some(target));

        let mut stops = 0;
        let mut ticks = 0;
        let mut swings = 0;
        for _ in 0..2000 {
            ticks += 1;
            let tick = controller.damage_block(&world, hit, GameMode::Survival, true, None, true);
            if tick.progressed {
                swings += 1;
            }
            match tick.step {
                DigStep::None => {}
                DigStep::Finish(pos, face) => {
                    assert_eq!((pos, face), (target, Facing::Up));
                    stops += 1;
                    break;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(stops, 1, "after {ticks} ticks");
        // java: `sendClickBlockToController` swings on every tick `onPlayerDamageBlock` returns
        // true, and it returns true on every tick of the break — one `C0APacketAnimation` per
        // tick, not one per state change. mc-rs used to send two for this whole dig.
        assert_eq!(swings, ticks, "one C0A per tick of the break");
        assert_eq!(controller.hitting_block(), None);
        // Stone at 1.5 hardness with a bare hand on the ground is `1 / 1.5 / 100` per tick, so
        // exact arithmetic says 150 ticks. It takes **151**, and that is not a bug to round
        // away: `curBlockDamageMP` is a Java `float` and vanilla accumulates it the same way, so
        // the sum lands a few ulp short of `1.0` on the 150th tick. Asserting the ideal 150 here
        // would mean "correct" was a version that had quietly switched to `f64`.
        let ideal = (1.0 / (1.0 / stone().props().hardness / 100.0)).ceil() as i32;
        assert_eq!(ideal, 150);
        assert_eq!(ticks, 151, "the break took the wrong number of ticks");
    }

    /// java: `clickBlock` aborts the old target before starting the new one — and the abort
    /// itself is `OutboundState`'s, so what is asserted here is that a *new* START is produced.
    #[test]
    fn switching_target_restarts_the_dig_and_zeroes_the_damage() {
        let first = BlockPos::new(0, 64, 0);
        let second = BlockPos::new(1, 64, 0);
        let mut world = world_with(first);
        world.set_block(second, stone());
        let mut controller = PlayerController::new();
        controller.click_block(
            &world,
            hit_at(first, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        for _ in 0..10 {
            controller.damage_block(
                &world,
                hit_at(first, Facing::Up),
                GameMode::Survival,
                true,
                None,
                true,
            );
        }
        assert!(controller.damage() > 0.0);
        assert_eq!(
            controller.damage_block(
                &world,
                hit_at(second, Facing::Up),
                GameMode::Survival,
                true,
                None,
                true
            ),
            DamageTick::sending(DigStep::Start(second, Facing::Up))
        );
        assert_eq!(controller.hitting_block(), Some(second));
        assert_eq!(controller.damage(), 0.0);
    }

    /// `PlayerControllerMP.isHittingPosition` uses the held item as part of the dig identity.
    /// The negative controls are deliberately close: changing a tool's durability or any
    /// stack's count must keep the same dig, whereas changing a dye's metadata must restart it.
    #[test]
    fn held_item_identity_controls_whether_a_dig_restarts() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = hit_at(target, Facing::Up);

        let pick = item("iron_pickaxe");
        let worn_pick = ItemStack {
            damage: 17,
            ..pick.clone()
        };
        let mut controller = PlayerController::new();
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Survival, true, Some(&pick), true),
            DigStep::Start(target, Facing::Up)
        );
        assert_eq!(
            controller.click_block(
                &world,
                hit,
                GameMode::Survival,
                true,
                Some(&worn_pick),
                true
            ),
            DigStep::None,
            "damageable-item durability is intentionally not part of isHittingPosition"
        );

        let dye = item("dye");
        let a_different_dye = ItemStack {
            damage: 1,
            ..dye.clone()
        };
        let mut controller = PlayerController::new();
        controller.click_block(&world, hit, GameMode::Survival, true, Some(&dye), true);
        assert_eq!(
            controller.click_block(
                &world,
                hit,
                GameMode::Survival,
                true,
                Some(&a_different_dye),
                true
            ),
            DigStep::Start(target, Facing::Up),
            "a non-damageable item's metadata distinguishes its subtypes"
        );

        let larger_stack = ItemStack {
            count: 37,
            ..dye.clone()
        };
        let mut controller = PlayerController::new();
        controller.click_block(&world, hit, GameMode::Survival, true, Some(&dye), true);
        assert_eq!(
            controller.click_block(
                &world,
                hit,
                GameMode::Survival,
                true,
                Some(&larger_stack),
                true
            ),
            DigStep::None,
            "ItemStack.areItemStackTagsEqual does not compare stack size"
        );

        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert("display", mc_nbt::Nbt::String("changed".to_owned()));
        let tagged_dye = ItemStack {
            nbt: Some(tag),
            ..dye.clone()
        };
        let mut controller = PlayerController::new();
        controller.click_block(&world, hit, GameMode::Survival, true, Some(&dye), true);
        assert_eq!(
            controller.click_block(
                &world,
                hit,
                GameMode::Survival,
                true,
                Some(&tagged_dye),
                true
            ),
            DigStep::Start(target, Facing::Up),
            "an NBT change is part of ItemStack.areItemStackTagsEqual"
        );
    }

    /// java: `resetBlockRemoving` is guarded by `if (this.isHittingBlock)`, so releasing the
    /// button when nothing was being dug must not put an ABORT on the wire every tick.
    #[test]
    fn aborting_is_a_no_op_when_nothing_is_being_dug() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let mut controller = PlayerController::new();
        assert_eq!(controller.reset_block_removing(), DigStep::None);
        controller.click_block(
            &world,
            hit_at(target, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        assert_eq!(controller.reset_block_removing(), DigStep::Abort);
        assert_eq!(controller.reset_block_removing(), DigStep::None);
    }

    /// java: `onPlayerDamageBlock` continues on
    /// `isHittingPosition(posBlock)` alone (`:301`), and `resetBlockRemoving` (`:274-282`) keeps
    /// `currentBlock`/`currentItemHittingBlock`. So a held dig whose crosshair slips onto the sky
    /// (or an entity) for a tick sends `ABORT(currentBlock, DOWN)`, and on returning to the same
    /// block resumes from zero damage with **no new START**, then finishes with a STOP. The
    /// crack follows `sendBlockBreakProgress`: `-1` at the reset, then the resumed stages.
    #[test]
    fn a_held_dig_resumes_the_same_block_after_a_reset_without_a_new_start() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = hit_at(target, Facing::Up);
        let mut controller = PlayerController::new();
        assert_eq!(
            controller.damage_block(&world, hit, GameMode::Survival, true, None, true),
            DamageTick::sending(DigStep::Start(target, Facing::Up)),
            "the first held tick on a new block is clickBlock's START"
        );
        assert_eq!(
            controller.take_pending_abort(),
            None,
            "nothing was being dug"
        );
        assert_eq!(controller.take_break_progress(), Some((target, -1)));
        for _ in 0..20 {
            controller.damage_block(&world, hit, GameMode::Survival, true, None, true);
        }
        assert_eq!(
            controller.take_break_progress(),
            Some((target, (controller.damage() * 10.0) as i32 - 1))
        );

        // The crosshair leaves the block for one tick: `resetBlockRemoving`.
        assert_eq!(controller.reset_block_removing(), DigStep::Abort);
        assert_eq!(controller.take_pending_abort(), Some(target));
        assert_eq!(controller.take_break_progress(), Some((target, -1)));
        assert_eq!(controller.hitting_block(), None);
        assert_eq!(controller.damage(), 0.0);

        // Back on the same block with the same (empty) hand: vanilla's continuation, not
        // clickBlock. No packet, but it still returns true (swings) and draws the crack.
        let resumed = controller.damage_block(&world, hit, GameMode::Survival, true, None, true);
        assert_eq!(resumed.step, DigStep::None, "no new START: {resumed:?}");
        assert!(resumed.progressed);
        assert!(controller.damage() > 0.0);
        assert_eq!(
            controller.hitting_block(),
            None,
            "isHittingBlock stays false"
        );
        assert_eq!(controller.take_break_progress(), Some((target, -1)));

        // The resumed dig ends with exactly one STOP and no START.
        let mut steps = Vec::new();
        for _ in 0..200 {
            let tick = controller.damage_block(&world, hit, GameMode::Survival, true, None, true);
            assert!(tick.progressed);
            if tick.step != DigStep::None {
                steps.push(tick.step);
            }
            if matches!(tick.step, DigStep::Finish(..)) {
                break;
            }
        }
        assert_eq!(steps, [DigStep::Finish(target, Facing::Up)]);
        assert_eq!(controller.take_break_progress(), Some((target, -1)));

        // A reset while resuming is a no-op (`isHittingBlock` is false): no ABORT, no crack
        // clear, and the accumulated damage survives.
        let mut controller = PlayerController::new();
        controller.damage_block(&world, hit, GameMode::Survival, true, None, true);
        controller.reset_block_removing();
        controller.damage_block(&world, hit, GameMode::Survival, true, None, true);
        let carried = controller.damage();
        let _ = controller.take_pending_abort();
        let _ = controller.take_break_progress();
        assert_eq!(controller.reset_block_removing(), DigStep::None);
        assert_eq!(controller.take_pending_abort(), None);
        assert_eq!(controller.take_break_progress(), None);
        assert_eq!(controller.damage(), carried);
    }

    /// java: `clickBlock`'s instant-break branch (`:252-255`) calls
    /// `onPlayerDestroyBlock` and assigns nothing else, and `onPlayerDestroyBlock` (`:173`) moves
    /// `currentBlock` to `(x, -1, z)`. A held dig swept onto tall grass therefore keeps
    /// `isHittingBlock`, and the next target change sends `ABORT` at the old X/Z with `y = -1`,
    /// as does a reset. `getIsHittingBlock()` stays true meanwhile, which refuses
    /// `rightClickMouse`.
    #[test]
    fn an_instant_break_inside_a_held_dig_keeps_hitting_and_aborts_at_y_minus_one() {
        let first = BlockPos::new(0, 64, 0);
        let grass = BlockPos::new(1, 64, 0);
        let next = BlockPos::new(2, 64, 0);
        let mut world = world_with(first);
        let tallgrass = mc_data::block_by_name("tallgrass").expect("tallgrass");
        world.set_block(grass, BlockState::new(tallgrass, 1));
        world.set_block(next, stone());
        let mut controller = PlayerController::new();

        controller.damage_block(
            &world,
            hit_at(first, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        for _ in 0..10 {
            controller.damage_block(
                &world,
                hit_at(first, Facing::Up),
                GameMode::Survival,
                true,
                None,
                true,
            );
        }
        let carried = controller.damage();
        assert!(carried > 0.0);
        let _ = controller.take_pending_abort();
        let _ = controller.take_break_progress();

        // Swept onto the tall grass: ABORT(first), START(grass), local destroy.
        let instant = controller.damage_block(
            &world,
            hit_at(grass, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        assert_eq!(
            instant,
            DamageTick::sending(DigStep::StartAndDestroy(grass, Facing::Up))
        );
        assert_eq!(controller.take_pending_abort(), Some(first));
        assert_eq!(
            controller.take_break_progress(),
            None,
            "the instant branch never calls sendBlockBreakProgress"
        );
        // The driver's accepted local destroy is `onPlayerDestroyBlock`'s `currentBlock` edit.
        world.set_block(grass, BlockState::AIR);
        controller.note_block_destroyed();
        let stale = BlockPos::new(first.x, -1, first.z);
        assert_eq!(
            controller.hitting_block(),
            Some(stale),
            "isHittingBlock survives the instant branch"
        );
        assert_eq!(controller.damage(), carried, "so does curBlockDamageMP");

        // The next block: ABORT at y = -1, then its START.
        assert_eq!(
            controller.damage_block(
                &world,
                hit_at(next, Facing::Up),
                GameMode::Survival,
                true,
                None,
                true,
            ),
            DamageTick::sending(DigStep::Start(next, Facing::Up))
        );
        assert_eq!(controller.take_pending_abort(), Some(stale));
        assert_eq!(controller.hitting_block(), Some(next));

        // A release right after the instant break also names the y = -1 block.
        let mut controller = PlayerController::new();
        world.set_block(grass, BlockState::new(tallgrass, 1));
        controller.click_block(
            &world,
            hit_at(first, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        controller.damage_block(
            &world,
            hit_at(grass, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        controller.note_block_destroyed();
        let _ = controller.take_pending_abort();
        assert_eq!(controller.reset_block_removing(), DigStep::Abort);
        assert_eq!(controller.take_pending_abort(), Some(stale));
        assert_eq!(controller.hitting_block(), None);

        // An instant break with nothing being dug sends no ABORT and stays not-hitting.
        let mut controller = PlayerController::new();
        world.set_block(grass, BlockState::new(tallgrass, 1));
        assert_eq!(
            controller.click_block(
                &world,
                hit_at(grass, Facing::Up),
                GameMode::Survival,
                true,
                None,
                true,
            ),
            DigStep::StartAndDestroy(grass, Facing::Up)
        );
        assert_eq!(controller.take_pending_abort(), None);
        controller.note_block_destroyed();
        assert_eq!(controller.hitting_block(), None);
    }

    /// java: `currentBlock = new BlockPos(-1, -1, -1)` and `currentItemHittingBlock = null`. With
    /// the continuation keyed on `isHittingPosition` alone, a `(0, 0, 0)` default would let an
    /// empty-handed first hold on the origin block "resume" a dig that never started.
    #[test]
    fn a_fresh_controller_starts_at_the_java_sentinel_block() {
        let origin = BlockPos::new(0, 0, 0);
        let world = world_with(origin);
        let mut controller = PlayerController::new();
        assert_eq!(
            controller.damage_block(
                &world,
                hit_at(origin, Facing::Up),
                GameMode::Survival,
                true,
                None,
                true,
            ),
            DamageTick::sending(DigStep::Start(origin, Facing::Up))
        );
    }

    /// java: the creative branch of `clickBlock` (`:230-235`) sends START with no ABORT even if
    /// a survival dig left `isHittingBlock` set (a game-mode change mid-dig).
    #[test]
    fn a_creative_click_never_names_an_abort_target() {
        let target = BlockPos::new(0, 64, 0);
        let other = BlockPos::new(1, 64, 0);
        let mut world = world_with(target);
        world.set_block(other, stone());
        let mut controller = PlayerController::new();
        controller.click_block(
            &world,
            hit_at(target, Facing::Up),
            GameMode::Survival,
            true,
            None,
            true,
        );
        assert_eq!(controller.hitting_block(), Some(target));
        assert_eq!(
            controller.click_block(
                &world,
                hit_at(other, Facing::Up),
                GameMode::Creative,
                true,
                None,
                true,
            ),
            DigStep::StartAndDestroy(other, Facing::Up)
        );
        assert_eq!(controller.take_pending_abort(), None);
    }

    /// java: `clickMouse` sets `leftClickCounter = 10` on a miss, and `sendClickBlockToController(false)`
    /// zeroes it the moment the button comes up.
    #[test]
    fn a_missed_swing_locks_the_dig_loop_out_for_ten_ticks() {
        let mut controller = PlayerController::new();
        assert!(controller.may_damage_block());
        controller.note_missed_swing(false);
        assert!(!controller.may_damage_block());
        for _ in 0..10 {
            controller.tick();
        }
        assert!(controller.may_damage_block());

        controller.note_missed_swing(false);
        assert!(!controller.may_damage_block());
        controller.release_attack();
        assert!(
            controller.may_damage_block(),
            "releasing clears the lockout"
        );

        // Creative never sets it at all.
        controller.note_missed_swing(true);
        assert!(controller.may_damage_block());
    }

    /// java: `Minecraft.runTick` writes `leftClickCounter = 10000` while a GUI is open (and
    /// again when focus returns). This is a real lockout, unlike the zero written by
    /// `sendClickBlockToController(false)` for an ordinary attack-button release.
    #[test]
    fn gui_lockout_is_10000_ticks_and_ordinary_release_is_immediate() {
        let mut controller = PlayerController::new();
        assert!(controller.may_damage_block());

        controller.lock_out();
        assert_eq!(controller.left_click_counter, 10_000);
        assert!(
            !controller.may_damage_block(),
            "GUI entry denies digging immediately"
        );

        controller.tick();
        assert_eq!(controller.left_click_counter, 9_999);
        assert!(
            !controller.may_damage_block(),
            "the first tick only decrements the lockout"
        );

        // Cover the positive boundary: 9,999 elapsed ticks still leave one tick denied, while
        // the 10,000th makes the controller eligible again. A mutant `lock_out = 0` fails at
        // the first denial assertion, and a decrement/no-decrement mutant fails this boundary.
        for _ in 0..9_998 {
            controller.tick();
        }
        assert_eq!(controller.left_click_counter, 1);
        assert!(!controller.may_damage_block());
        controller.tick();
        assert_eq!(controller.left_click_counter, 0);
        assert!(controller.may_damage_block());

        // The physical attack release is a separate Java path and must clear a lockout without
        // waiting for its timer. This is the negative control against making release_attack the
        // GUI operation merely because both paths currently share a caller-side region.
        controller.lock_out();
        controller.release_attack();
        assert_eq!(controller.left_click_counter, 0);
        assert!(controller.may_damage_block());
    }

    /// java: `rightClickDelayTimer = 4`. Holding right-click fires five times a second, not
    /// twenty — the difference between using an item and spamming the server.
    #[test]
    fn a_held_right_click_repeats_every_four_ticks() {
        let mut controller = PlayerController::new();
        assert!(controller.right_click_ready());
        controller.note_right_click();
        for tick in 0..4 {
            assert!(!controller.right_click_ready(), "tick {tick}");
            controller.tick();
        }
        assert!(controller.right_click_ready());
    }

    /// The reaches differ between game modes and between blocks and entities, and all four
    /// numbers are anticheat-relevant.
    #[test]
    fn the_reach_distances_match_vanillas_four_numbers() {
        assert_eq!(PlayerController::block_reach(false), 4.5);
        assert_eq!(PlayerController::block_reach(true), 5.0);
        // java: the entity ray is `extendedReach() ? 6.0 : getBlockReachDistance()`. The
        // survival 3.0 is a post-loop rejection, not a ray length — see
        // `the_survival_three_block_limit_discards_the_block_hit_too`.
        assert_eq!(PlayerController::entity_reach(false), 4.5);
        assert_eq!(PlayerController::entity_reach(true), 6.0);
    }

    /// java: `getToolDigEfficiency` divides by five off the ground, and `canHarvestBlock` picks
    /// the 30 divisor over the 100 one. Stone needs a pickaxe, so bare hands take the ÷100 path.
    #[test]
    fn hardness_follows_the_thirty_hundred_split_and_the_airborne_penalty() {
        let stone = stone();
        let hardness = stone.props().hardness;
        assert!(hardness > 0.0);
        assert!(!stone.props().material.tool_not_required());
        let grounded = player_relative_hardness(stone, true);
        assert!(
            (grounded - 1.0 / hardness / 100.0).abs() < 1e-9,
            "{grounded}"
        );
        let airborne = player_relative_hardness(stone, false);
        assert!((airborne - grounded / 5.0).abs() < 1e-9, "{airborne}");

        let dirt = BlockState::new(mc_data::block_by_name("dirt").expect("dirt"), 0);
        assert!(dirt.props().material.tool_not_required());
        let dirt_speed = player_relative_hardness(dirt, true);
        assert!(
            (dirt_speed - 1.0 / dirt.props().hardness / 30.0).abs() < 1e-9,
            "{dirt_speed}"
        );
    }

    fn block(name: &str) -> BlockState {
        BlockState::new(mc_data::block_by_name(name).expect("known block"), 0)
    }

    fn enchanted_item(name: &str, enchantment_id: i16, level: i16) -> ItemStack {
        let mut entry = mc_nbt::NbtCompound::new();
        entry.insert("id", mc_nbt::Nbt::Short(enchantment_id));
        entry.insert("lvl", mc_nbt::Nbt::Short(level));
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert(
            "ench",
            mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(entry)]),
        );
        ItemStack {
            nbt: Some(nbt),
            ..item(name)
        }
    }

    fn dig_context(held: Option<&ItemStack>) -> DigContext {
        DigContext::from_player(held, None, None, false, false)
    }

    fn assert_f32(actual: f32, expected: f32, label: &str) {
        let tolerance = expected.abs().max(1.0) * 1.0E-6;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: expected {expected}, got {actual}"
        );
    }

    /// W-DIGSPEED: ItemPickaxe has separate strength and harvest-level decisions. Gold's speed
    /// is the sharp negative control: it is the fastest tier but cannot harvest diamond ore, so
    /// replacing `canHarvestBlock` with `getStrVsBlock > 1` changes the divisor and fails.
    #[test]
    fn pickaxe_strength_and_harvest_tier_are_independent() {
        let diamond_ore = block("diamond_ore");
        let hardness = diamond_ore.props().hardness;
        for (tool, strength, divisor) in [
            ("wooden_pickaxe", 2.0, 100.0),
            ("stone_pickaxe", 4.0, 100.0),
            ("iron_pickaxe", 6.0, 30.0),
            ("diamond_pickaxe", 8.0, 30.0),
            ("golden_pickaxe", 12.0, 100.0),
        ] {
            let held = item(tool);
            assert_f32(
                player_relative_hardness_with_context(diamond_ore, dig_context(Some(&held)), true),
                strength / hardness / divisor,
                tool,
            );
        }

        // Obsidian's equality check is stricter than the ordinary `>= tier` branches.
        let obsidian = block("obsidian");
        let diamond = item("diamond_pickaxe");
        let iron = item("iron_pickaxe");
        assert_f32(
            player_relative_hardness_with_context(obsidian, dig_context(Some(&diamond)), true),
            8.0 / 50.0 / 30.0,
            "diamond harvests obsidian",
        );
        assert_f32(
            player_relative_hardness_with_context(obsidian, dig_context(Some(&iron)), true),
            6.0 / 50.0 / 100.0,
            "iron does not harvest obsidian",
        );

        // Emerald follows the diamond/gold/redstone >= 2 branch, not the iron/lapis >= 1 one.
        // A stone pick is still effective (strength 4) but must take the non-harvestable divisor.
        let emerald_ore = block("emerald_ore");
        let stone_pick = item("stone_pickaxe");
        assert_f32(
            player_relative_hardness_with_context(
                emerald_ore,
                dig_context(Some(&stone_pick)),
                true,
            ),
            4.0 / emerald_ore.props().hardness / 100.0,
            "stone does not harvest emerald ore",
        );
        assert!(can_held_item_harvest(
            emerald_ore,
            dig_context(Some(&iron)).held_item,
        ));
    }

    /// The three ItemTool subclasses do not share one material predicate. Each sample below is
    /// in an EFFECTIVE_ON set while its material branch alone is insufficient; the paired stone
    /// or dirt samples make a mutant that treats every block as effective bite.
    #[test]
    fn tool_effective_sets_dispatch_by_item_subclass() {
        for (tool, proper, improper, speed) in [
            ("diamond_pickaxe", "rail", "dirt", 8.0),
            ("diamond_axe", "melon_block", "stone", 8.0),
            ("diamond_shovel", "soul_sand", "stone", 8.0),
        ] {
            let held = item(tool);
            assert_eq!(
                strength_vs_block(block(proper), dig_context(Some(&held)).held_item),
                speed,
                "{tool} on {proper}"
            );
            assert_eq!(
                strength_vs_block(block(improper), dig_context(Some(&held)).held_item),
                1.0,
                "{tool} must not be fast on {improper}"
            );
        }
    }

    /// ItemSword and ItemShears deliberately ignore ItemProps.efficiency. Their hard-coded web,
    /// plant/leaves and wool cases are different from one another and from every ItemTool.
    #[test]
    fn sword_and_shears_special_strength_and_harvest_cases_match_java() {
        let sword = item("diamond_sword");
        let shears = item("shears");
        let sword_id = dig_context(Some(&sword)).held_item;
        let shears_id = dig_context(Some(&shears)).held_item;

        assert_eq!(strength_vs_block(block("web"), sword_id), 15.0);
        assert_eq!(strength_vs_block(block("leaves"), sword_id), 1.5);
        assert_eq!(strength_vs_block(block("stone"), sword_id), 1.0);
        assert!(can_held_item_harvest(block("web"), sword_id));
        assert!(!can_held_item_harvest(block("stone"), sword_id));

        assert_eq!(strength_vs_block(block("web"), shears_id), 15.0);
        assert_eq!(strength_vs_block(block("leaves"), shears_id), 15.0);
        assert_eq!(strength_vs_block(block("wool"), shears_id), 5.0);
        assert_eq!(strength_vs_block(block("stone"), shears_id), 1.0);
        assert!(can_held_item_harvest(block("redstone_wire"), shears_id));
        assert!(can_held_item_harvest(block("tripwire"), shears_id));
    }

    /// Efficiency applies only after a tool has produced strength greater than one and uses the
    /// Java term `level² + 1`. The enchanted stick proves it is not a blanket enchantment bonus;
    /// Unbreaking proves the NBT scan is keyed to id 32.
    #[test]
    fn efficiency_reads_the_held_stacks_ench_list_and_keeps_the_speed_gate() {
        let stone = stone();
        let hardness = stone.props().hardness;
        let pick = enchanted_item("diamond_pickaxe", 32, 5);
        assert_f32(
            player_relative_hardness_with_context(stone, dig_context(Some(&pick)), true),
            (8.0 + 26.0) / hardness / 30.0,
            "Efficiency V diamond pickaxe",
        );

        let stick = enchanted_item("stick", 32, 5);
        assert_f32(
            player_relative_hardness_with_context(stone, dig_context(Some(&stick)), true),
            1.0 / hardness / 100.0,
            "Efficiency on base strength one",
        );
        let unbreaking = enchanted_item("diamond_pickaxe", 34, 5);
        assert_f32(
            player_relative_hardness_with_context(stone, dig_context(Some(&unbreaking)), true),
            8.0 / hardness / 30.0,
            "a different enchantment id",
        );
    }

    /// EntityPlayer applies Haste, then Mining Fatigue, then the underwater and airborne
    /// divisions. All four fatigue switch arms are pinned; collapsing the last arm to amplifier
    /// three only, or using a generic `0.3^(amp+1)`, fails on amplifier two or four.
    #[test]
    fn potion_water_and_ground_modifiers_match_the_exact_java_multipliers() {
        let stone = stone();
        let pick = item("diamond_pickaxe");
        let base = 8.0 / stone.props().hardness / 30.0;

        let haste = DigContext::from_player(Some(&pick), Some(1), None, false, false);
        assert_f32(
            player_relative_hardness_with_context(stone, haste, true),
            base * 1.4,
            "Haste II",
        );
        for (amplifier, multiplier) in [(0, 0.3), (1, 0.09), (2, 0.0027), (3, 8.1E-4), (4, 8.1E-4)]
        {
            let fatigue = DigContext::from_player(Some(&pick), None, Some(amplifier), false, false);
            assert_f32(
                player_relative_hardness_with_context(stone, fatigue, true),
                base * multiplier,
                &format!("Mining Fatigue amplifier {amplifier}"),
            );
        }

        let underwater = DigContext::from_player(Some(&pick), None, None, true, false);
        let aqua_affinity = DigContext::from_player(Some(&pick), None, None, true, true);
        assert_f32(
            player_relative_hardness_with_context(stone, underwater, true),
            base / 5.0,
            "underwater without Aqua Affinity",
        );
        assert_f32(
            player_relative_hardness_with_context(stone, aqua_affinity, true),
            base,
            "Aqua Affinity cancels only the water penalty",
        );
        assert_f32(
            player_relative_hardness_with_context(stone, aqua_affinity, false),
            base / 5.0,
            "Aqua Affinity does not cancel airborne",
        );
    }

    /// PlayerController must consume its stored context at the actual STOP-producing seam. A
    /// diamond pick takes six damage ticks on stone; the bare-hand mutant takes 151.
    #[test]
    fn controller_break_timing_consumes_the_resolved_dig_context() {
        let target = BlockPos::new(0, 64, 0);
        let world = world_with(target);
        let hit = hit_at(target, Facing::Up);
        let pick = item("diamond_pickaxe");
        let mut controller = PlayerController::new();
        controller.set_dig_context(dig_context(Some(&pick)));
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Survival, true, None, true),
            DigStep::Start(target, Facing::Up)
        );
        for tick in 1..=6 {
            let result = controller.damage_block(&world, hit, GameMode::Survival, true, None, true);
            if tick < 6 {
                assert_eq!(result.step, DigStep::None, "tick {tick}");
            } else {
                assert_eq!(result.step, DigStep::Finish(target, Facing::Up));
            }
        }
    }

    /// EntityLivingBase gives Haste precedence over Fatigue for swing animation length.
    #[test]
    fn swing_animation_length_uses_the_potion_specific_formula() {
        assert_eq!(arm_swing_animation_end(None, None), 6);
        assert_eq!(arm_swing_animation_end(Some(0), None), 5);
        assert_eq!(arm_swing_animation_end(Some(2), None), 3);
        assert_eq!(arm_swing_animation_end(None, Some(0)), 8);
        assert_eq!(arm_swing_animation_end(None, Some(2)), 12);
        assert_eq!(arm_swing_animation_end(Some(0), Some(3)), 5);
    }

    // ----------------------------------------------------------------- item use

    fn item(name: &str) -> ItemStack {
        ItemStack {
            item_id: mc_data::item_by_name(name).expect("known item").0 as i16,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    fn hungry() -> UseGuards {
        UseGuards {
            creative: false,
            disable_damage: false,
            food_level: 10,
            has_arrow: true,
        }
    }

    /// java: only five `Item` subclasses override `getMaxItemUseDuration`, and they are exactly
    /// the five `setItemInUse` call sites in `item/`. Everything else is `0`.
    #[test]
    fn exactly_five_item_classes_have_a_use_duration() {
        let g = hungry();
        assert_eq!(item_use_duration(&item("bow"), g), Some(72_000));
        assert_eq!(item_use_duration(&item("diamond_sword"), g), Some(72_000));
        assert_eq!(item_use_duration(&item("apple"), g), Some(32));
        assert_eq!(item_use_duration(&item("potion"), g), Some(32));
        assert_eq!(item_use_duration(&item("milk_bucket"), g), Some(32));
        // Every `ItemFood` subclass counts too, and `mc-data` does resolve those to
        // `ItemKind::Food` — unlike `ItemBucketMilk`. Pinned so a data regression is caught here
        // rather than as "food silently stopped working".
        for name in [
            "golden_apple",  // ItemAppleGold
            "mushroom_stew", // ItemSoup
            "cooked_fish",   // ItemFishFood
            "carrot",        // ItemSeedFood
            "rotten_flesh",  // plain ItemFood
        ] {
            assert_eq!(
                item_use_duration(&item(name), g),
                Some(32),
                "{name} is an ItemFood subclass and must be eatable"
            );
        }

        // Everything a player is likely to right-click that is *not* one of the five.
        for name in [
            "stone",
            "arrow",
            "ender_pearl",
            "diamond_pickaxe",
            "water_bucket",
            "bucket",
            "lava_bucket",
            "flint_and_steel",
            "fishing_rod",
        ] {
            assert_eq!(
                item_use_duration(&item(name), g),
                None,
                "{name} must not begin a timed use"
            );
        }
        // A server can put any id in an `ItemStack`; an unregistered one must not panic.
        assert_eq!(
            item_use_duration(
                &ItemStack {
                    item_id: 30_000,
                    count: 1,
                    damage: 0,
                    nbt: None
                },
                g
            ),
            None
        );
    }

    /// java: `Item.getItemUseAction` defaults to NONE and only the five concrete 1.8.9
    /// subclasses below override it. Potion metadata must not affect the action: the splash
    /// predicate belongs to `ItemPotion.onItemRightClick`, not this virtual getter.
    #[test]
    fn item_use_action_matches_the_five_vanilla_overrides() {
        assert_eq!(item_use_action(&item("apple")), ItemUseAction::Eat);
        assert_eq!(item_use_action(&item("mushroom_stew")), ItemUseAction::Eat);
        assert_eq!(item_use_action(&item("potion")), ItemUseAction::Drink);
        assert_eq!(item_use_action(&item("milk_bucket")), ItemUseAction::Drink);
        assert_eq!(
            item_use_action(&item("diamond_sword")),
            ItemUseAction::Block
        );
        assert_eq!(item_use_action(&item("bow")), ItemUseAction::Bow);

        let mut splash = item("potion");
        splash.damage = SPLASH_POTION_BIT;
        assert_eq!(item_use_action(&splash), ItemUseAction::Drink);

        for name in [
            "stone",
            "ender_pearl",
            "experience_bottle",
            "snowball",
            "fishing_rod",
            "flint_and_steel",
        ] {
            assert_eq!(
                item_use_action(&item(name)),
                ItemUseAction::None,
                "{name} inherits Item.getItemUseAction()"
            );
        }
        assert_eq!(
            item_use_action(&ItemStack {
                item_id: 30_000,
                count: 1,
                damage: 0,
                nbt: None,
            }),
            ItemUseAction::None,
            "an unknown server id must retain Item's safe NONE default"
        );
    }

    #[test]
    fn armor_items_map_to_container_player_armor_slots() {
        for (name, expected) in [
            ("diamond_helmet", 5),
            ("iron_chestplate", 6),
            ("golden_leggings", 7),
            ("leather_boots", 8),
        ] {
            assert_eq!(armor_inventory_slot(&item(name)), Some(expected), "{name}");
        }
        assert_eq!(armor_inventory_slot(&item("diamond_sword")), None);
        assert_eq!(
            armor_inventory_slot(&ItemStack {
                item_id: 30_000,
                count: 1,
                damage: 0,
                nbt: None,
            }),
            None
        );
    }

    /// java: `RenderItem.renderItemModel` (`:290-303`) has strict elapsed-use boundaries, while
    /// `AbstractClientPlayer.getFovModifier` (`:126-140`) eases the same duration over twenty
    /// ticks and caps its pull term at one.
    #[test]
    fn bow_draw_model_and_fov_thresholds_match_vanilla() {
        assert_eq!(bow_pull_stage(-1), BowPullStage::Base);
        assert_eq!(bow_pull_stage(0), BowPullStage::Base);
        assert_eq!(bow_pull_stage(1), BowPullStage::Pulling0);
        assert_eq!(bow_pull_stage(13), BowPullStage::Pulling0);
        assert_eq!(bow_pull_stage(14), BowPullStage::Pulling1);
        assert_eq!(bow_pull_stage(17), BowPullStage::Pulling1);
        assert_eq!(bow_pull_stage(18), BowPullStage::Pulling2);
        assert_eq!(bow_pull_stage(i32::MAX), BowPullStage::Pulling2);

        assert!((bow_fov_multiplier(0) - 1.0).abs() < f32::EPSILON);
        assert!((bow_fov_multiplier(10) - 0.9625).abs() < 1e-6);
        assert!((bow_fov_multiplier(20) - 0.85).abs() < 1e-6);
        assert!((bow_fov_multiplier(40) - 0.85).abs() < 1e-6);
    }

    #[test]
    fn item_in_use_action_follows_the_latched_stack() {
        let mut controller = PlayerController::new();
        assert_eq!(controller.item_in_use_action(), None);

        controller.set_item_in_use(item("diamond_sword"), 72_000);
        assert_eq!(controller.item_in_use_action(), Some(ItemUseAction::Block));
        controller.stop_using_item();
        assert_eq!(controller.item_in_use_action(), None);

        controller.set_item_in_use(item("bow"), BOW_MAX_USE_DURATION);
        assert_eq!(controller.item_in_use_action(), Some(ItemUseAction::Bow));
    }

    /// The first-person caller must read action, countdown, max duration, and elapsed duration
    /// from one latched stack. The deliberately wrong `32` argument is a biting control for the
    /// old duration-disconnected path that made an active bow look like food to a model/FOV
    /// consumer.
    #[test]
    fn bow_draw_render_state_tracks_ticks_thresholds_and_release() {
        let bow = item("bow");
        let mut controller = PlayerController::new();
        controller.set_item_in_use(bow.clone(), BOW_MAX_USE_DURATION);

        let state = controller
            .item_in_use_render_state()
            .expect("a started bow has a render state");
        assert_eq!(state.action, ItemUseAction::Bow);
        assert_eq!(state.remaining_count, BOW_MAX_USE_DURATION);
        assert_eq!(state.max_duration, BOW_MAX_USE_DURATION);
        assert_eq!(state.elapsed_duration, 0);
        assert_eq!(bow_pull_stage(state.elapsed_duration), BowPullStage::Base);

        controller.tick_item_in_use(Some(&bow));
        let state = controller.item_in_use_render_state().unwrap();
        assert_eq!(state.elapsed_duration, 1);
        assert_eq!(
            bow_pull_stage(state.elapsed_duration),
            BowPullStage::Pulling0
        );
        for _ in 1..13 {
            controller.tick_item_in_use(Some(&bow));
        }
        let state = controller.item_in_use_render_state().unwrap();
        assert_eq!(state.elapsed_duration, 13);
        assert_eq!(
            bow_pull_stage(state.elapsed_duration),
            BowPullStage::Pulling0
        );

        controller.tick_item_in_use(Some(&bow));
        let state = controller.item_in_use_render_state().unwrap();
        assert_eq!(state.elapsed_duration, 14);
        assert_eq!(
            bow_pull_stage(state.elapsed_duration),
            BowPullStage::Pulling1
        );
        for _ in 14..18 {
            controller.tick_item_in_use(Some(&bow));
        }
        let state = controller.item_in_use_render_state().unwrap();
        assert_eq!(state.elapsed_duration, 18);
        assert_eq!(
            bow_pull_stage(state.elapsed_duration),
            BowPullStage::Pulling2
        );

        assert_eq!(
            controller.item_in_use_duration(32),
            18,
            "the latched bow max duration wins over a disconnected caller default"
        );
        controller.stop_using_item();
        assert_eq!(controller.item_in_use_render_state(), None);
        assert_eq!(controller.item_in_use_duration(BOW_MAX_USE_DURATION), 0);
    }

    #[test]
    fn bow_release_charge_has_java_three_tick_minimum_and_twenty_tick_cap() {
        assert_eq!(bow_release_strength(BOW_MAX_USE_DURATION), None);
        assert_eq!(bow_release_strength(BOW_MAX_USE_DURATION - 2), None);
        // Java keeps this whole charge expression in `float`; the rounded square and divide
        // produce the representable f32 value 0.10750001 for exactly three ticks.
        assert_eq!(
            bow_release_strength(BOW_MAX_USE_DURATION - 3),
            Some(0.10750001)
        );
        assert_eq!(bow_release_strength(BOW_MAX_USE_DURATION - 20), Some(1.0));
        assert_eq!(bow_release_strength(0), Some(1.0));
    }

    /// A sword may block for the same 72,000-tick duration, but it is not a bow and must never
    /// route through the bow-pulling model thresholds.
    #[test]
    fn non_bow_use_state_keeps_block_action_separate_from_bow_models() {
        let sword = item("diamond_sword");
        let mut controller = PlayerController::new();
        controller.set_item_in_use(sword.clone(), BOW_MAX_USE_DURATION);
        controller.tick_item_in_use(Some(&sword));
        let state = controller.item_in_use_render_state().unwrap();
        assert_eq!(state.action, ItemUseAction::Block);
        assert_eq!(state.elapsed_duration, 1);
        assert_ne!(state.action, ItemUseAction::Bow);
    }

    /// `ItemEnderPearl`, `ItemExpBottle`, and `ItemSnowball` decrement on both client and
    /// server sides in survival; the creative branch skips the decrement. Fire charges and
    /// boats are deliberately absent because their client-side result depends on a valid block
    /// raytrace/placement path owned by the driver.
    #[test]
    fn projectile_right_click_stack_decrement_matches_the_client_item_bodies() {
        // `egg` is `ItemEgg`, whose `--itemStackIn.stackSize` sits outside the remote guard
        // exactly like the snowball's (H3).
        for name in ["ender_pearl", "experience_bottle", "snowball", "egg"] {
            let stack = item(name);
            assert!(
                client_right_click_decrements_stack(&stack, false),
                "{name} decrements in survival"
            );
            assert!(
                !client_right_click_decrements_stack(&stack, true),
                "{name} is preserved in creative"
            );
        }
        let mut splash = item("potion");
        splash.damage = 16_384 | 5;
        assert!(client_right_click_decrements_stack(&splash, false));
        assert!(!client_right_click_decrements_stack(&splash, true));
        let mut drinkable = splash;
        drinkable.damage = 5;
        assert!(!client_right_click_decrements_stack(&drinkable, false));

        // `ItemEnderEye` decrements only inside `if (!worldIn.isRemote)`.
        for name in ["fire_charge", "boat", "bow", "diamond_sword", "ender_eye"] {
            assert!(
                !client_right_click_decrements_stack(&item(name), false),
                "{name} has no unconditional client decrement"
            );
        }
    }

    fn can_place_on_stack(name: &str, blocks: &[&str]) -> ItemStack {
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert(
            "CanPlaceOn",
            mc_nbt::Nbt::list(
                blocks
                    .iter()
                    .map(|block| mc_nbt::Nbt::String((*block).to_owned()))
                    .collect(),
            ),
        );
        ItemStack {
            item_id: mc_data::item_by_name(name).expect("known item").0 as i16,
            count: 1,
            damage: 0,
            nbt: Some(nbt),
        }
    }

    fn source_water_world() -> (World, BlockPos) {
        let water = BlockPos::new(0, 64, 3);
        (world_of(&[(water, "minecraft:water")]), water)
    }

    #[test]
    fn boat_item_use_requires_a_clear_liquid_hit_eye_and_spawn_box() {
        let (world, water) = source_water_world();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let boat = item("boat");
        assert_eq!(
            boat_item_use(&world, eye, 0.0, 0.0, &boat, &[], &[], None),
            Some(water)
        );
        assert_eq!(
            boat_item_use(&world_of(&[]), eye, 0.0, 0.0, &boat, &[], &[], None),
            None,
            "a missed liquid ray preserves the boat"
        );
        assert_eq!(
            boat_item_use(
                &world,
                eye,
                0.0,
                0.0,
                &boat,
                &[Aabb::from_bounds(0.0, 64.0, 0.0, 1.0, 66.0, 1.0)],
                &[],
                None,
            ),
            None,
            "an eye inside a collidable entity vetoes use"
        );
        let blocked = world_of(&[(water, "minecraft:water"), (water.up(), "minecraft:stone")]);
        assert_eq!(
            boat_item_use(&blocked, eye, 0.0, 0.0, &boat, &[], &[], None),
            None,
            "the contracted spawn AABB still rejects block collision"
        );
        assert_eq!(
            boat_item_use(
                &world,
                eye,
                0.0,
                0.0,
                &boat,
                &[],
                &[Aabb::from_bounds(0.0, 65.0, 2.5, 1.0, 66.0, 3.5)],
                None,
            ),
            None,
            "the new boat's collision box rejects every overlapping entity"
        );
        let edge = mc_physics::collision::CollisionBorder::new(0.0, 0.0, 6.0, 29_999_984);
        assert_eq!(
            boat_item_use(&world, eye, 0.0, 0.0, &boat, &[], &[], Some(edge)),
            None,
            "an inside boat sees the world border's outside-block stone substitution"
        );
    }

    #[test]
    fn lily_pad_collision_is_empty_for_boats_but_shallow_for_other_entities() {
        let pad = BlockPos::new(0, 65, 3);
        let world = world_of(&[(pad, "minecraft:waterlily")]);
        let mask = Aabb::from_bounds(0.0, 65.0, 3.0, 1.0, 66.0, 4.0);
        assert!(
            collision_boxes_at_for_entity(&world, pad, &mask, Some(EntityType::Boat)).is_empty(),
            "BlockLilyPad.addCollisionBoxesToList skips EntityBoat"
        );
        let player_boxes = collision_boxes_at_for_entity(&world, pad, &mask, None);
        assert_eq!(
            player_boxes.len(),
            1,
            "the generic entity retains the pad box"
        );
        assert_eq!(
            player_boxes[0],
            Aabb::from_bounds(0.0, 65.0, 3.0, 1.0, 65.015625, 4.0),
            "non-boats retain BlockLilyPad's 1/64th-height collision"
        );
        assert_eq!(
            collision_boxes_at_for_entity(&world, pad, &mask, Some(EntityType::Zombie)).len(),
            1,
            "an unrelated mob follows the ordinary collision path"
        );
    }

    /// Both item bodies use the flat five-block liquid ray, and neither may predict through a
    /// non-water or a flowing-water state. Adventure permission is the client-side
    /// `canPlayerEdit` gate: a matching `CanPlaceOn` list admits the same source-water face.
    #[test]
    fn liquid_item_use_obeys_liquid_kind_source_and_permission_controls() {
        let (world, water) = source_water_world();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let bottle = item("glass_bottle");
        assert_eq!(
            liquid_item_use(&world, eye, 0.0, 0.0, &bottle, true, false),
            Some(LiquidUsePrediction::GlassBottle { water })
        );

        let denied = liquid_item_use(&world, eye, 0.0, 0.0, &bottle, false, false);
        assert_eq!(
            denied, None,
            "adventure without CanPlaceOn cannot edit water"
        );
        let allowed = can_place_on_stack("glass_bottle", &["minecraft:water"]);
        assert_eq!(
            liquid_item_use(&world, eye, 0.0, 0.0, &allowed, false, false),
            Some(LiquidUsePrediction::GlassBottle { water })
        );

        let stone = world_of(&[(water, "minecraft:stone")]);
        assert_eq!(
            liquid_item_use(&stone, eye, 0.0, 0.0, &bottle, true, false),
            None,
            "a glass bottle only fills from Material.water"
        );
        let lava = world_of(&[(water, "minecraft:lava")]);
        assert_eq!(
            liquid_item_use(&lava, eye, 0.0, 0.0, &bottle, true, false),
            None,
            "lava is not accepted by ItemGlassBottle"
        );

        let mut flowing = world;
        flowing.set_block(
            water,
            BlockState::new(mc_data::block_by_name("water").expect("water"), 1),
        );
        assert_eq!(
            liquid_item_use(&flowing, eye, 0.0, 0.0, &bottle, true, false),
            None,
            "stopOnLiquid only targets level-0 source states"
        );
    }

    #[test]
    fn bucket_use_distinguishes_source_fill_pour_vaporize_and_creative_results() {
        let (world, water) = source_water_world();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let empty = item("bucket");
        assert_eq!(
            liquid_item_use_in_dimension(&world, eye, 0.0, 0.0, &empty, true, false, 0),
            Some(LiquidUsePrediction::BucketFill {
                source: water,
                filled_item: WATER_BUCKET_ITEM_ID,
                preserve_held: false,
            })
        );
        assert_eq!(
            liquid_item_use_in_dimension(&world, eye, 0.0, 0.0, &empty, true, true, 0),
            Some(LiquidUsePrediction::BucketFill {
                source: water,
                filled_item: WATER_BUCKET_ITEM_ID,
                preserve_held: true,
            })
        );

        let lava_world = world_of(&[(water, "minecraft:lava")]);
        assert_eq!(
            liquid_item_use_in_dimension(&lava_world, eye, 0.0, 0.0, &empty, true, false, 0),
            Some(LiquidUsePrediction::BucketFill {
                source: water,
                filled_item: LAVA_BUCKET_ITEM_ID,
                preserve_held: false,
            })
        );

        let stone = world_of(&[(water, "minecraft:stone")]);
        let target = water.north();
        assert_eq!(
            liquid_item_use_in_dimension(
                &stone,
                eye,
                0.0,
                0.0,
                &item("water_bucket"),
                true,
                false,
                0,
            ),
            Some(LiquidUsePrediction::BucketPour {
                target,
                state: Some(BlockState::new(
                    mc_data::block_by_name("flowing_water").expect("flowing water"),
                    0,
                )),
                replace_with_empty: true,
            })
        );
        assert_eq!(
            liquid_item_use_in_dimension(
                &stone,
                eye,
                0.0,
                0.0,
                &item("water_bucket"),
                true,
                false,
                -1,
            ),
            Some(LiquidUsePrediction::BucketPour {
                target,
                state: None,
                replace_with_empty: true,
            })
        );
        assert_eq!(
            liquid_item_use_in_dimension(
                &stone,
                eye,
                0.0,
                0.0,
                &item("lava_bucket"),
                true,
                true,
                -1,
            ),
            Some(LiquidUsePrediction::BucketPour {
                target,
                state: Some(BlockState::new(
                    mc_data::block_by_name("flowing_lava").expect("flowing lava"),
                    0,
                )),
                replace_with_empty: false,
            })
        );
    }

    /// `ItemLilyPad` has a stricter local body than a bottle: it requires source water, an air
    /// cell above, and decrements only outside creative mode. Occupied placement and a flowing
    /// level are negative controls for the two branches that otherwise look interchangeable.
    #[test]
    fn lily_pad_prediction_obeys_source_air_placement_and_creative_controls() {
        let (world, water) = source_water_world();
        let eye = DVec3::new(0.5, 64.5, 0.5);
        let pad = item("waterlily");
        assert_eq!(
            liquid_item_use(&world, eye, 0.0, 0.0, &pad, true, false),
            Some(LiquidUsePrediction::LilyPad {
                water,
                place: water.up(),
                decrement_stack: true,
            })
        );
        assert_eq!(
            liquid_item_use(&world, eye, 0.0, 0.0, &pad, true, true),
            Some(LiquidUsePrediction::LilyPad {
                water,
                place: water.up(),
                decrement_stack: false,
            })
        );

        let (mut occupied, _) = source_water_world();
        occupied.set_block(water.up(), BlockState::new(mc_data::BlockId(1), 0));
        assert_eq!(
            liquid_item_use(&occupied, eye, 0.0, 0.0, &pad, true, false),
            None,
            "ItemLilyPad refuses a non-air placement cell"
        );

        let mut flowing = world;
        flowing.set_block(
            water,
            BlockState::new(mc_data::block_by_name("water").expect("water"), 1),
        );
        assert_eq!(
            liquid_item_use(&flowing, eye, 0.0, 0.0, &pad, true, false),
            None,
            "ItemLilyPad requires BlockLiquid.LEVEL == 0"
        );
    }

    /// java: `ItemBow.onItemRightClick:123` uses the creative/arrow guard before
    /// calling `setItemInUse`; without either condition the click is a complete no-op.
    #[test]
    fn a_bow_only_draws_with_an_arrow_or_in_creative() {
        let bow = item("bow");
        let mut g = hungry();
        g.has_arrow = false;
        assert_eq!(item_use_duration(&bow, g), None, "no arrows, not creative");
        g.creative = true;
        g.disable_damage = true;
        assert_eq!(item_use_duration(&bow, g), Some(72_000), "creative");
        g.creative = false;
        g.disable_damage = false;
        g.has_arrow = true;
        assert_eq!(item_use_duration(&bow, g), Some(72_000));
    }

    /// java: `ItemFood.onItemRightClick:97` — `canEat(alwaysEdible)` is
    /// `(alwaysEdible || foodLevel < 20) && !capabilities.disableDamage`. Both halves bite:
    /// a full player cannot eat, and neither can a creative one.
    #[test]
    fn food_obeys_both_halves_of_can_eat_and_the_golden_apple_exemption() {
        let apple = item("apple");
        let golden = item("golden_apple");

        let full = UseGuards {
            food_level: 20,
            ..hungry()
        };
        assert_eq!(item_use_duration(&apple, full), None, "a full player");
        assert_eq!(
            item_use_duration(&golden, full),
            Some(32),
            "java: golden_apple is the only `setAlwaysEdible()` registration (Item.java:831)"
        );

        let creative = UseGuards {
            creative: true,
            disable_damage: true,
            ..hungry()
        };
        assert_eq!(
            item_use_duration(&apple, creative),
            None,
            "`disableDamage` is set for CREATIVE, so a creative player cannot eat"
        );
        assert_eq!(
            item_use_duration(&golden, creative),
            None,
            "not even the always-edible one — `disableDamage` gates both"
        );
    }

    #[test]
    fn finished_food_stats_use_virtual_stack_values_and_java_caps() {
        assert_eq!(
            finished_food_stats(&item("apple"), 10, 1.0),
            Some(FinishedFoodStats {
                food_level: 14,
                saturation: 3.4,
            })
        );

        let mut cooked_salmon = item("cooked_fish");
        cooked_salmon.damage = 1;
        assert_eq!(
            finished_food_stats(&cooked_salmon, 5, 0.0),
            Some(FinishedFoodStats {
                food_level: 11,
                saturation: 9.6,
            }),
            "ItemFishFood metadata must reach its virtual heal/saturation values"
        );
        assert_eq!(
            finished_food_stats(&item("apple"), 19, 19.5),
            Some(FinishedFoodStats {
                food_level: 20,
                saturation: 20.0,
            }),
            "both hunger and saturation cap at the new food level"
        );
        assert_eq!(finished_food_stats(&item("bow"), 10, 1.0), None);
    }

    /// java: `ItemPotion.onItemRightClick:147` throws a splash potion instead of drinking it,
    /// and never calls `setItemInUse`. `isSplash(meta)` is `(meta & 16384) != 0`.
    #[test]
    fn a_splash_potion_is_thrown_rather_than_drunk() {
        let mut potion = item("potion");
        potion.damage = 8197; // an ordinary healing potion
        assert_eq!(item_use_duration(&potion, hungry()), Some(32));
        potion.damage = 8197 | 16384;
        assert_eq!(item_use_duration(&potion, hungry()), None);
    }

    /// java: `EntityPlayer.setItemInUse:2102`'s `if (stack != this.itemInUse)` guard. A held
    /// right-click calls `rightClickMouse` every four ticks; restarting the counter each time
    /// would freeze a bow's draw at zero forever.
    #[test]
    fn re_using_the_same_stack_does_not_restart_the_counter() {
        let mut c = PlayerController::new();
        assert!(!c.is_using_item());
        c.set_item_in_use(item("bow"), 72_000);
        assert!(c.is_using_item());
        c.tick_item_in_use(Some(&item("bow")));
        c.tick_item_in_use(Some(&item("bow")));
        assert_eq!(c.item_in_use_duration(72_000), 2, "two ticks of draw");
        c.set_item_in_use(item("bow"), 72_000);
        assert_eq!(
            c.item_in_use_duration(72_000),
            2,
            "the same stack must not reset the draw"
        );
    }

    /// Java compares the current held `ItemStack` object to `itemInUse` by reference, not by
    /// serialized value. An equal server replacement and another slot containing an equal stack
    /// must therefore both cancel the draw.
    #[test]
    fn equal_value_replacement_or_slot_switch_cancels_by_stack_identity() {
        let bow = item("bow");
        let mut c = PlayerController::new();
        c.set_item_in_use_at_slot(bow.clone(), 72_000, 0, 41);
        c.tick_item_in_use_at_slot(Some(&bow), 0, Some(41));
        assert_eq!(c.item_in_use_duration(72_000), 1);
        c.set_item_in_use_at_slot(bow.clone(), 72_000, 0, 41);
        assert_eq!(
            c.item_in_use_duration(72_000),
            1,
            "the same live stack identity must not restart the draw"
        );

        c.tick_item_in_use_at_slot(Some(&bow), 0, Some(42));
        assert!(
            !c.is_using_item(),
            "an equal replacement is a new Java object"
        );

        c.set_item_in_use_at_slot(bow.clone(), 72_000, 0, 43);
        c.tick_item_in_use_at_slot(Some(&bow), 1, Some(44));
        assert!(
            !c.is_using_item(),
            "another equal hotbar stack is not the latched object"
        );
    }

    /// java: `EntityPlayer.onUpdate`'s `if (itemstack != this.itemInUse) this.clearItemInUse();`
    /// — swapping the hotbar mid-draw cancels the use with no packet at all.
    #[test]
    fn changing_the_held_stack_cancels_the_use() {
        let mut c = PlayerController::new();
        c.set_item_in_use(item("bow"), 72_000);
        c.tick_item_in_use(Some(&item("diamond_sword")));
        assert!(!c.is_using_item());
        assert_eq!(c.item_in_use_duration(72_000), 0);

        // Emptying the hand does the same.
        c.set_item_in_use(item("bow"), 72_000);
        c.tick_item_in_use(None);
        assert!(!c.is_using_item());
    }

    /// The counter reaching zero must **not** end the use client-side: `onItemUseFinish` is
    /// guarded by `!worldObj.isRemote`, and the multiplayer-only rule means this world is always remote. Food
    /// is only swallowed when the release packet reaches the server.
    #[test]
    fn the_counter_running_out_does_not_end_a_use_on_a_remote_world() {
        let mut c = PlayerController::new();
        let apple = item("apple");
        c.set_item_in_use(apple.clone(), 32);
        for _ in 0..40 {
            c.tick_item_in_use(Some(&apple));
        }
        assert!(
            c.is_using_item(),
            "only `stop_using_item` (the release) may clear it on a client"
        );
        c.stop_using_item();
        assert!(!c.is_using_item());
    }

    /// java: `sendClickBlockToController`'s guard is `leftClickCounter <= 0 &&
    /// !thePlayer.isUsingItem()`. The second half was missing (A09-24's first half).
    #[test]
    fn a_drawn_bow_stops_the_dig_loop() {
        let mut c = PlayerController::new();
        assert!(c.may_damage_block());
        c.set_item_in_use(item("bow"), 72_000);
        assert!(!c.may_damage_block(), "you cannot mine while drawing a bow");
        c.stop_using_item();
        assert!(c.may_damage_block());
    }
}
