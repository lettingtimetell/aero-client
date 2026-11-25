//! The 1.7.10 rules of the crosshair and the dig loop: what `EntityRenderer.getMouseOver`,
//! `World.rayTraceBlocks` and `PlayerControllerMP` decide differently in vanilla 1.7.10.
//!
//! A child of [`super`] (`interact.rs`, which declares it as `interact::v5`) so the 1.7.10
//! controller methods share [`PlayerController`]'s fields with the 1.8.9 ones: one controller,
//! two rule sets, and a 1.8.9 session never calls anything here. The callers are the 1.7.10
//! click functions in `app/interact_v5.rs`, which belongs to the driver despite its name.
//!
//! `java17:` cites vanilla 1.7.10, `java18:` vanilla 1.8.9. Everything not named here is the
//! same in both versions and stays in `interact.rs`.

use mc_data::{BlockState, EntityType, Material};
use mc_math::{Aabb, BlockPos, DVec3, Facing};
use mc_protocol::{play::s2c::GameMode, ItemStack};
use mc_world::World;

use super::{
    can_collide_check, can_held_item_harvest, collision_ray_trace, collision_ray_trace_for_state,
    intercept, selection_bounds, BlockHit, BlockHitSound, DamageTick, DigContext, DigStep,
    EntityHit, HitResult, Pickable, PlayerController, RayTrace, RAYTRACE_STEPS,
};

/// java17: `EntityPlayer.getEyeHeight()` (`EntityPlayer.java:842-845`). The 1.7.10 local player's
/// `posY` is already its eye (`minY + 1.62F - ySize`), so this is only the extra height
/// `Entity.isInsideOfMaterial` adds on top of it.
pub const PLAYER_EYE_HEIGHT_ABOVE_POS_Y: f32 = 0.12;

/// The `C08PacketPlayerBlockPlacement` hit offsets as 1.7.10 computes them.
///
/// java17: `PlayerControllerMP.onPlayerRightClick` (`PlayerControllerMP.java:299-301`):
/// `(float)hitVector.xCoord - (float)x`, both operands narrowed to `float` before the
/// subtraction. 1.8.9 subtracts in `double` and narrows once (java18 `:393-395`,
/// [`BlockHit::cursor`]), so the two differ by a `float` ulp of the coordinate, which moves the
/// sixteenths byte near 1/16 boundaries.
#[must_use]
pub fn cursor(hit: &BlockHit) -> DVec3 {
    let axis = |hit: f64, block: i32| f64::from(hit as f32 - block as f32);
    DVec3::new(
        axis(hit.hit_vec.x, hit.pos.x),
        axis(hit.hit_vec.y, hit.pos.y),
        axis(hit.hit_vec.z, hit.pos.z),
    )
}

/// java17: `Entity.getCollisionBorderSize()` (`Entity.java:1614-1617`, `0.1F`) and its one
/// override, `EntityFireball` (`EntityFireball.java:256-259`, `1.0F`, inherited by the small
/// fireball and the wither skull). 1.7.10 has no `EntityItemFrame` override, so an item frame
/// keeps the default `0.1F` where 1.8.9 uses `0.0F` (java18 `EntityItemFrame.java:39-42`).
#[must_use]
pub fn collision_border_size(kind: Option<EntityType>) -> f32 {
    match kind {
        Some(EntityType::Fireball | EntityType::SmallFireball | EntityType::WitherSkull) => 1.0,
        _ => 0.1,
    }
}

/// java17: `Material.isAdventureModeExempt()` (`Material.java:23-28`, `:147-150`): glass,
/// redstone lamp, ice, packed ice and every `MaterialLogic` (`MaterialLogic.java:10`: plants,
/// vine, circuits, carpet, snow).
#[must_use]
pub fn is_adventure_mode_exempt_material(material: Material) -> bool {
    matches!(
        material,
        Material::Glass
            | Material::RedstoneLight
            | Material::Ice
            | Material::PackedIce
            | Material::Plants
            | Material::Vine
            | Material::Circuits
            | Material::Carpet
            | Material::Snow
    )
}

/// java17: `EntityPlayer.isCurrentToolAdventureModeExempt(x, y, z)`
/// (`EntityPlayer.java:1785-1815`).
///
/// ```text
/// if (capabilities.allowEdit) return true;
/// block = world.getBlock(x, y, z);
/// if (block.getMaterial() != air) {
///     if (block.getMaterial().isAdventureModeExempt()) return true;
///     held = getCurrentEquippedItem();
///     if (held != null && (held.canItemHarvestBlock(block) || held.getStrVsBlock(block) > 1.0F))
///         return true;
/// }
/// return false;
/// ```
///
/// It gates both the adventure dig (`clickBlock :140`, `onPlayerDestroyBlock :88`) and the
/// per-tick swing of `Minecraft.sendClickBlockToController` (`Minecraft.java:1326-1330`). 1.8.9
/// has none of it: its adventure rule is the `CanDestroy` NBT list (java18
/// `PlayerControllerMP.clickBlock`, `PlayerControllerMP.java:200-222`).
#[must_use]
pub fn is_current_tool_adventure_mode_exempt(
    world: &World,
    pos: BlockPos,
    held: Option<&ItemStack>,
    allow_edit: bool,
) -> bool {
    if allow_edit {
        return true;
    }
    let state = world.block_at(pos);
    let material = state.props().material;
    if material == Material::Air {
        return false;
    }
    if is_adventure_mode_exempt_material(material) {
        return true;
    }
    let Some(item) = held.and_then(registered_item) else {
        return false;
    };
    item_can_harvest_block(state, item) || strength_vs_block(state, Some(item)) > 1.0
}

/// java17: `EntityPlayer.canPlayerEdit(x, y, z, side, stack)` (`EntityPlayer.java:1817-1820`):
/// `allowEdit ? true : (stack != null ? stack.canEditBlocks() : false)`, and
/// `Item.canItemEditBlocks()` is `true` for every 1.7.10 item (`Item.java:679-682`, never
/// overridden). So any held stack may edit, adventure mode included; 1.8.9 asks the stack's
/// `CanPlaceOn` list instead ([`super::can_player_edit`]).
#[must_use]
pub fn can_player_edit(held: Option<&ItemStack>, allow_edit: bool) -> bool {
    allow_edit || held.is_some()
}

fn registered_item(stack: &ItemStack) -> Option<mc_data::ItemId> {
    let id = u16::try_from(stack.item_id).ok()?;
    mc_data::item_props(mc_data::ItemId(id))
        .registered
        .then_some(mc_data::ItemId(id))
}

/// java17: `ItemStack.canItemHarvestBlock` (`ItemStack.java:315-318`), the held item's own rule
/// (`ItemPickaxe`, `ItemSpade`, `ItemSword`, `ItemShears`, else `Item.java:490-493`: false),
/// which the exemption test calls directly. [`can_held_item_harvest`] is
/// `InventoryPlayer.func_146025_b` (`InventoryPlayer.java:580-591`): the same item rules behind
/// an `isToolNotRequired` shortcut, so it is only asked about materials that need a tool.
fn item_can_harvest_block(state: BlockState, item: mc_data::ItemId) -> bool {
    if state.props().material.tool_not_required() {
        // Of the item rules only the shears name blocks that need no tool: redstone wire and
        // tripwire (`ItemShears.java:34-37`). Web, snow and every pickaxe block need one.
        return mc_data::item_props(item).kind == mc_data::ItemKind::Shears
            && matches!(state.block().0, 55 | 132);
    }
    can_held_item_harvest(state, Some(item))
}

/// java17: `InventoryPlayer.getStrVsBlock` (`InventoryPlayer.java:479-489`) and the item
/// overrides. The same as 1.8.9 except two effective-block sets:
///
/// * `ItemPickaxe` (`ItemPickaxe.java:11`) has no packed ice (174); red sandstone (179) does
///   not exist in 1.7.10;
/// * `ItemAxe` (`ItemAxe.java:11`) has no melon (103) and no ladder (65).
///
/// Material shortcuts (pickaxe on rock/iron/anvil, axe on wood/plants/vine), the shovel set, the
/// sword (web 15, plants/vine/coral/leaves/gourd 1.5) and the shears (web/leaves 15, wool 5) are
/// identical (`ItemPickaxe.java:21-24`, `ItemAxe.java:20-23`, `ItemSpade.java:11`,
/// `ItemSword.java:34-45`, `ItemShears.java:39-42`).
#[must_use]
pub fn strength_vs_block(state: BlockState, held_item: Option<mc_data::ItemId>) -> f32 {
    use mc_data::ItemKind;

    let Some(item) = held_item else { return 1.0 };
    let props = mc_data::item_props(item);
    let block = state.block().0;
    let material = state.props().material;
    match props.kind {
        ItemKind::Pickaxe => {
            const EFFECTIVE_ON: &[u16] = &[
                4, 43, 44, 1, 24, 48, 15, 42, 16, 41, 14, 56, 57, 79, 87, 21, 22, 73, 74, 66, 28,
                27, 157,
            ];
            if matches!(material, Material::Iron | Material::Anvil | Material::Rock)
                || EFFECTIVE_ON.contains(&block)
            {
                props.efficiency
            } else {
                1.0
            }
        }
        ItemKind::Axe => {
            const EFFECTIVE_ON: &[u16] = &[5, 47, 17, 162, 54, 86, 91];
            if matches!(material, Material::Wood | Material::Plants | Material::Vine)
                || EFFECTIVE_ON.contains(&block)
            {
                props.efficiency
            } else {
                1.0
            }
        }
        ItemKind::Shovel => {
            const EFFECTIVE_ON: &[u16] = &[2, 3, 12, 13, 78, 80, 82, 60, 88, 110];
            if EFFECTIVE_ON.contains(&block) {
                props.efficiency
            } else {
                1.0
            }
        }
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

/// java17: `Block.getPlayerRelativeBlockHardness` (`Block.java:597-601`) with
/// `EntityPlayer.getBreakSpeed` (`EntityPlayer.java:715-760`).
///
/// The order of the `float` operations is 1.8.9's except for Mining Fatigue, which 1.7.10
/// scales linearly: `var3 *= 1.0F - (float)(amplifier + 1) * 0.2F` (`:744-747`), so amplifier 4
/// stops digging and higher amplifiers make the speed negative (java18 uses the table 0.3, 0.09,
/// 0.0027, 8.1E-4, `EntityPlayer.java:920-944`). The dead
/// `var6 * 0.08F` efficiency branch (`:728-731`, unreachable inside `var3 > 1.0F`) is left out.
/// `context.underwater` must be [`is_inside_of_water`] at the 1.7.10 eye (`:749`).
#[must_use]
pub fn player_relative_hardness(state: BlockState, context: DigContext, on_ground: bool) -> f32 {
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
        speed *= 1.0 + (i32::from(amplifier) + 1) as f32 * 0.2;
    }
    if let Some(amplifier) = context.fatigue_amplifier {
        speed *= 1.0 - (i32::from(amplifier) + 1) as f32 * 0.2;
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

/// java17: `Entity.isInsideOfMaterial(Material.water)` (`Entity.java:977-995`) for the local
/// player, whose `posY` is the eye and whose `getEyeHeight()` is `0.12F`: the test point is
/// 0.12 above the 1.8.9 one (java18 `Entity.isInsideOfMaterial`, `Entity.java:1196-1214`).
///
/// ```text
/// double y = posY + (double)getEyeHeight();
/// int by = floor_float((float)floor_double(y));
/// block = getBlock(floor(posX), by, floor(posZ));
/// if (material == water)
///     return y < (double)((float)(by + 1) - (getLiquidHeightPercent(meta) - 0.11111111F));
/// ```
#[must_use]
pub fn is_inside_of_water(world: &World, pos_x: f64, pos_y: f64, pos_z: f64) -> bool {
    let y = pos_y + f64::from(PLAYER_EYE_HEIGHT_ABOVE_POS_Y);
    // `floor_float((float)floor_double(y))`: the same integer for every reachable height.
    let block_y = mc_math::floor_f64(y);
    let pos = BlockPos::new(
        mc_math::floor_f64(pos_x),
        block_y,
        mc_math::floor_f64(pos_z),
    );
    let state = world.block_at(pos);
    if state.props().material != Material::Water {
        return false;
    }
    // java17: `BlockLiquid.getLiquidHeightPercent` (`BlockLiquid.java:66-74`).
    let meta = i32::from(state.meta());
    let level = if meta >= 8 { 0 } else { meta };
    let percent = (level + 1) as f32 / 9.0 - 0.111_111_11_f32;
    let surface = (block_y + 1) as f32 - percent;
    y < f64::from(surface)
}

/// java17: `World.rayTraceBlocks(start, end, false, false, true)`, the flags
/// `EntityLivingBase.rayTrace` (`EntityLivingBase.java:1925-1931`) casts the crosshair with.
#[must_use]
pub fn raytrace_blocks(world: &World, start: DVec3, end: DVec3) -> Option<RayTrace> {
    raytrace_blocks_with_options(world, start, end, false, false, true)
}

/// java17: `World.rayTraceBlocks(Vec3, Vec3, boolean, boolean, boolean)` (`World.java:937-1150`).
///
/// The walk is 1.8.9's ([`super::raytrace_blocks_with_options`]) without the zero-step tweak:
/// 1.8.9 turns a `0.0` step parameter into `-1.0E-4` on every axis (java18 `World.java:997-1010`),
/// 1.7.10 compares the raw parameters (java17 `World.java:1023-1043`). Two blocks also trace
/// differently: see [`collision_ray_trace_for_state_v5`].
#[must_use]
pub fn raytrace_blocks_with_options(
    world: &World,
    start: DVec3,
    end: DVec3,
    stop_on_liquid: bool,
    ignore_block_without_bounding_box: bool,
    return_last_uncollidable_block: bool,
) -> Option<RayTrace> {
    // java17: only `Double.isNaN` is tested, on both ends, then on the moving start in the loop.
    let nan = |v: DVec3| v.x.is_nan() || v.y.is_nan() || v.z.is_nan();
    if nan(start) || nan(end) {
        return None;
    }
    let (tx, ty, tz) = (
        mc_math::floor_f64(end.x),
        mc_math::floor_f64(end.y),
        mc_math::floor_f64(end.z),
    );
    let (mut lx, mut ly, mut lz) = (
        mc_math::floor_f64(start.x),
        mc_math::floor_f64(start.y),
        mc_math::floor_f64(start.z),
    );
    let mut cursor = start;
    let mut pos = BlockPos::new(lx, ly, lz);
    let state = world.block_at(pos);
    let has_collision_box = !state.props().collision.is_empty();
    if (!ignore_block_without_bounding_box || has_collision_box)
        && can_collide_check(state, stop_on_liquid)
    {
        if let Some(hit) = collision_ray_trace_for_state_v5(world, state, pos, cursor, end) {
            return Some(RayTrace::Block(hit));
        }
    }

    let mut last_uncollidable: Option<RayTrace> = None;
    let mut budget = RAYTRACE_STEPS;
    while budget >= 0 {
        budget -= 1;
        if nan(cursor) {
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
            plane_x = f64::from(lx) + 0.0;
        } else {
            step_x = false;
        }
        if ty > ly {
            plane_y = f64::from(ly) + 1.0;
        } else if ty < ly {
            plane_y = f64::from(ly) + 0.0;
        } else {
            step_y = false;
        }
        if tz > lz {
            plane_z = f64::from(lz) + 1.0;
        } else if tz < lz {
            plane_z = f64::from(lz) + 0.0;
        } else {
            step_z = false;
        }
        let mut t_x = 999.0;
        let mut t_y = 999.0;
        let mut t_z = 999.0;
        let delta = end - cursor;
        if step_x {
            t_x = (plane_x - cursor.x) / delta.x;
        }
        if step_y {
            t_y = (plane_y - cursor.y) / delta.y;
        }
        if step_z {
            t_z = (plane_z - cursor.z) / delta.z;
        }
        // java17 has no `if (d3 == -0.0D) d3 = -1.0E-4D;` here: the parameters are compared as
        // they are.
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
        lx = mc_math::floor_f64(cursor.x) - i32::from(face == Facing::East);
        ly = mc_math::floor_f64(cursor.y) - i32::from(face == Facing::Up);
        lz = mc_math::floor_f64(cursor.z) - i32::from(face == Facing::South);
        pos = BlockPos::new(lx, ly, lz);
        let state = world.block_at(pos);
        let has_collision_box = !state.props().collision.is_empty();
        if !ignore_block_without_bounding_box || has_collision_box {
            if can_collide_check(state, stop_on_liquid) {
                if let Some(hit) = collision_ray_trace_for_state_v5(world, state, pos, cursor, end)
                {
                    return Some(RayTrace::Block(hit));
                }
            } else {
                last_uncollidable = Some(RayTrace::Miss(BlockHit {
                    pos,
                    face,
                    hit_vec: cursor,
                }));
            }
        }
    }
    return_last_uncollidable_block
        .then_some(last_uncollidable)
        .flatten()
}

/// `Block.collisionRayTrace` with the two 1.7.10 shapes that differ from 1.8.9's:
///
/// * **Rails** (`BlockRailBase.setBlockBoundsBasedOnState`, java17 `BlockRailBase.java:60-72`)
///   test the *raw* metadata, `meta >= 2 && meta <= 5` for the 0.625-high slope box. A powered,
///   detector or activator rail keeps its powered bit (8) in that metadata, so an ascending
///   powered rail (metas 10-13) traces the flat 0.125 box; 1.8.9 asks the shape property and
///   traces 0.625 (java18 `BlockRailBase.java:65-78`).
/// * **The moving piston block** (36) has no `collisionRayTrace` override in 1.7.10 and is
///   targetable: its bounds are the carried block's, shifted back along the push by the progress
///   (`BlockPistonMoving.setBlockBoundsBasedOnState`, java17 `BlockPistonMoving.java:136-165`).
///   1.8.9 returns `null` for it (java18 `BlockPistonMoving.java:143-146`).
fn collision_ray_trace_for_state_v5(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    start: DVec3,
    end: DVec3,
) -> Option<BlockHit> {
    match state.block().0 {
        27 | 28 | 66 | 157 => {
            // `setBlockBounds(0, 0, 0, 1, 0.625F | 0.125F, 1)`: both heights are exact floats.
            let height = if (2..=5).contains(&state.meta()) {
                0.625
            } else {
                0.125
            };
            let bounds = Aabb {
                min: DVec3::new(0.0, 0.0, 0.0),
                max: DVec3::new(1.0, height, 1.0),
            };
            collision_ray_trace(bounds, pos, start, end)
        }
        36 => collision_ray_trace(moving_piston_ray_bounds(world, pos), pos, start, end),
        _ => collision_ray_trace_for_state(
            world,
            state,
            selection_bounds(world, pos, state),
            pos,
            start,
            end,
        ),
    }
}

/// The block-local bounds 1.7.10's `BlockPistonMoving` traces a ray against.
///
/// ```text
/// TileEntityPiston te = getTileEntity(x, y, z);
/// if (te != null) {
///     Block carried = te.getStoredBlockID();
///     if (carried == this || carried.getMaterial() == air) return;   // bounds unchanged
///     carried.setBlockBoundsBasedOnState(world, x, y, z);
///     float f = te.func_145860_a(0.0F); if (te.isExtending()) f = 1.0F - f;
///     int o = te.getPistonOrientation();
///     minX = carried.getBlockBoundsMinX() - (double)((float)Facing.offsetsXForSide[o] * f); ...
/// }
/// ```
///
/// Two approximations, both outside what the oracle records:
///
/// * without a tile entity (or carrying air or another moving block) Java keeps whatever bounds
///   the one shared `BlockPistonMoving` instance was last given, which depends on history; this
///   uses the constructor's full cube;
/// * the carried box is the carried state's static box. Java runs the carried block's
///   `setBlockBoundsBasedOnState(world, x, y, z)` at the moving block's position. Its metadata
///   there is the carried block's own (`BlockPistonBase` gives the moving block the pushed
///   block's metadata, `BlockPistonBase.java:175`, `:210`, `:470-475`), so a box that only
///   reads metadata comes out the same; a box that reads the neighbours (a fence, pane, wall,
///   or a door's other half) is read around the moving block there and is not modelled here.
fn moving_piston_ray_bounds(world: &World, pos: BlockPos) -> Aabb {
    let full = Aabb {
        min: DVec3::new(0.0, 0.0, 0.0),
        max: DVec3::new(1.0, 1.0, 1.0),
    };
    let Some(piston) = world.tile_entity(pos).and_then(|tile| tile.piston()) else {
        return full;
    };
    let carried = piston.piston_state;
    if carried.block().0 == 36 || carried.props().material == Material::Air {
        return full;
    }
    let mut progress = piston.progress_at(0.0);
    if piston.extending {
        progress = 1.0 - progress;
    }
    let (ox, oy, oz) = piston.facing.offset();
    let shift = |offset: i32| f64::from(offset as f32 * progress);
    let b = carried.props().bounds;
    Aabb {
        min: DVec3::new(
            b.min_x - shift(ox),
            b.min_y - shift(oy),
            b.min_z - shift(oz),
        ),
        max: DVec3::new(
            b.max_x - shift(ox),
            b.max_y - shift(oy),
            b.max_z - shift(oz),
        ),
    }
}

/// java17: `EntityRenderer.getMouseOver` (`EntityRenderer.java:293-388`).
///
/// ```text
/// double reach = playerController.getBlockReachDistance();        // 4.5, creative 5.0
/// objectMouseOver = renderViewEntity.rayTrace(reach, pt);
/// double var4 = reach; Vec3 eye = getPosition(pt);
/// if (playerController.extendedReach()) { reach = 6.0; var4 = 6.0; }
/// else { if (reach > 3.0) var4 = 3.0; reach = var4; }             // the entity ray is 3.0 long
/// if (objectMouseOver != null) var4 = objectMouseOver.hitVec.distanceTo(eye);
/// ... the 1.8.9 nearest-intercept loop over a `reach`-long ray ...
/// if (pointedEntity != null && (var12 < var4 || objectMouseOver == null))
///     objectMouseOver = new MovingObjectPosition(pointedEntity, hitVec);
/// ```
///
/// Against 1.8.9 ([`super::pick_with_riding`]): the survival entity ray is **3.0** long instead
/// of the block reach, and there is no rejection that turns an entity farther than 3.0 into a
/// MISS (java18 `EntityRenderer.java:496-500`), so the block behind an out-of-reach entity stays
/// targeted. The candidates need no spectator filter (1.7.10 has no spectators) and
/// carry 1.7.10's border sizes ([`collision_border_size`]).
///
/// `None` is Java's `objectMouseOver == null`: `rayTrace` keeps a MISS only for a block that
/// cannot stop the ray (air, liquids, fire), so a ray that crosses nothing but collidable blocks
/// whose bounds it misses, such as one looking up a ladder shaft, returns null
/// (`World.java:962-1140`), and no entity replaced it. 1.7.10's `clickMouse` then costs the
/// 10-tick lockout a MISS does not (`Minecraft.java:1348-1356`).
#[must_use]
pub fn pick(
    world: &World,
    eye: DVec3,
    look: DVec3,
    block_reach: f64,
    extended_reach: bool,
    riding_entity: Option<i32>,
    entities: impl IntoIterator<Item = Pickable>,
) -> Option<HitResult> {
    let trace = raytrace_blocks(world, eye, eye + look * block_reach);
    let entity_ray = entity_reach(block_reach, extended_reach);
    let mut d1 = entity_ray;
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
        let f1 = f64::from(candidate.border_size);
        let box_ = candidate.bounds.expand(f1, f1, f1);
        let point = intercept(box_, eye, entity_end);
        if box_.is_vec_inside(eye) {
            // java17: `if (0.0D < var12 || var12 == 0.0D)`, always true for a distance.
            pointed = Some((candidate.network_id, point.unwrap_or(eye)));
            d2 = 0.0;
        } else if let Some(point) = point {
            let d3 = mc_math::vec::distance_to(point, eye);
            if d3 < d2 || d2 == 0.0 {
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
    if let Some((network_id, hit_vec)) = pointed {
        if d2 < d1 || trace.is_none() {
            return Some(HitResult::Entity(EntityHit {
                network_id,
                hit_vec,
            }));
        }
    }
    trace.map(|trace| match trace.as_block() {
        Some(hit) => HitResult::Block(hit),
        None => HitResult::Miss,
    })
}

/// The length of 1.7.10's entity ray: `6.0` in creative (`extendedReach`), otherwise the block
/// reach capped at `3.0` (java17 `EntityRenderer.java:305-318`).
#[must_use]
pub fn entity_reach(block_reach: f64, extended_reach: bool) -> f64 {
    if extended_reach {
        6.0
    } else if block_reach > 3.0 {
        3.0
    } else {
        block_reach
    }
}

/// java17: `EntityHanging.setDirection` (`EntityHanging.java:39-107`): the box a 1.7.10 item
/// frame or painting is picked by, built around the position `setDirection` computes.
///
/// `direction` is 1.7.10's hanging direction (0 south, 1 west, 2 north, 3 east), `pos` the
/// entity position (`var5, var6, var7`, already offset off the wall). Both thin and wide
/// half-extents are `pixels / 32`, with the facing axis forced to `0.5F / 32`, and the box is
/// built with `var9 = -0.03125F` **added** to the max and subtracted from the min, all in
/// `float`: the wide axes shrink by 1/32 per side and the thin axis comes out inverted
/// (`min > max`). It is kept that way (no sorting), as Java keeps it: only the 0.1 collision
/// border ([`collision_border_size`]) gives the plate its depth when it is picked. 1.8.9 builds
/// a centred box with border 0 instead (java18 `EntityHanging.updateBoundingBox`, `:53-95`).
#[must_use]
pub fn hanging_bounding_box(pos: DVec3, direction: i32, width_px: i32, height_px: i32) -> Aabb {
    let mut half_x = width_px as f32;
    let half_y = height_px as f32;
    let mut half_z = width_px as f32;
    if direction != 2 && direction != 0 {
        half_x = 0.5;
    } else {
        half_z = 0.5;
    }
    let (half_x, half_y, half_z) = (half_x / 32.0, half_y / 32.0, half_z / 32.0);
    let (x, y, z) = (pos.x as f32, pos.y as f32, pos.z as f32);
    let inset = -0.031_25_f32;
    Aabb {
        min: DVec3::new(
            f64::from(x - half_x - inset),
            f64::from(y - half_y - inset),
            f64::from(z - half_z - inset),
        ),
        max: DVec3::new(
            f64::from(x + half_x + inset),
            f64::from(y + half_y + inset),
            f64::from(z + half_z + inset),
        ),
    }
}

/// The block a 1.7.10 `ItemBlock` places, or `None` for every other item.
///
/// java17: `Item.registerItems` registers an `ItemBlock` for every block but its 27 explicit
/// exceptions (`Item.java:251`, `:388-398`), so a few blocks that 1.8.9 gives no item (the
/// liquids 8-11, double slabs 43 and 125, fire, portal, end portal, cocoa, carrots, potatoes) are
/// `ItemBlock`s here. 1.7.10 registers no block above 175 and none of 165-169.
#[must_use]
pub fn placed_block(stack: &ItemStack) -> Option<mc_data::BlockId> {
    let id = u16::try_from(stack.item_id).ok()?;
    if id >= 176 || (165..=169).contains(&id) {
        return None;
    }
    if matches!(id, 8..=11 | 43 | 51 | 90 | 119 | 125 | 127 | 141 | 142) {
        return Some(mc_data::BlockId(id));
    }
    super::placed_block(stack)
}

/// The single slab a 1.7.10 `ItemSlab` places or merges with. java17: items 44 and 126, and the
/// double-slab items 43 and 125, which are `ItemSlab(.., true)` over the same single slabs
/// (`Item.java:316-331`).
fn slab_block(stack: &ItemStack) -> Option<mc_data::BlockId> {
    match stack.item_id {
        43 | 44 => Some(mc_data::BlockId(44)),
        125 | 126 => Some(mc_data::BlockId(126)),
        _ => None,
    }
}

/// java17: `ItemBlock.func_150936_a` (`ItemBlock.java:118-160`) and `ItemSlab`'s override
/// (`ItemSlab.java:79-130`): the test `PlayerControllerMP.onPlayerRightClick` makes before the
/// C08 (`PlayerControllerMP.java:309-317`). `false` means no C08 and no swing.
///
/// Against 1.8.9 ([`super::can_place_block_on_side`]): only snow layers (any depth, side set to
/// up), vine, tall grass and dead bush are placed into the clicked cell, so a double plant is
/// stepped past (java18 `BlockDoublePlant.isReplaceable`, `:69-82`); the slab merge compares
/// the unmasked stack metadata; and
/// [`can_place_entity_on_side`] is 1.7.10's.
#[must_use]
pub fn can_place_block_on_side(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    stack: &ItemStack,
    blocking: &[Aabb],
) -> bool {
    let Some(placing) = placed_block(stack) else {
        return true;
    };
    if let Some(single) = slab_block(stack) {
        let damage = i32::from(stack.damage);
        let existing = world.block_at(hit_pos);
        let top = existing.meta() & 8 != 0;
        if ((side == Facing::Up && !top) || (side == Facing::Down && top))
            && existing.block() == single
            && i32::from(existing.meta() & 7) == damage
        {
            return true;
        }
        let neighbour = world.block_at(hit_pos.relative(side));
        if neighbour.block() == single && i32::from(neighbour.meta() & 7) == damage {
            return true;
        }
    }
    let (target, side) = match world.block_at(hit_pos).block().0 {
        78 => (hit_pos, Facing::Up),
        106 | 31 | 32 => (hit_pos, side),
        _ => (hit_pos.relative(side), side),
    };
    can_place_entity_on_side(world, placing, target, side, blocking)
}

/// java17: `World.canPlaceEntityOnSide(block, x, y, z, false, side, entity, stack)`
/// (`World.java:3104-3109`) with `blocking` the boxes `checkNoEntityCollision` iterates:
///
/// ```text
/// AxisAlignedBB bb = block.getCollisionBoundingBoxFromPool(this, x, y, z);
/// return bb != null && !checkNoEntityCollision(bb, entity) ? false
///      : (existing.getMaterial() == circuits && block == anvil ? true
///         : existing.getMaterial().isReplaceable()
///           && block.canReplace(this, x, y, z, side, stack));
/// ```
///
/// The 1.7.10 differences from 1.8.9's `canBlockBePlaced`: the box is built in the cell, from
/// whatever metadata the cell holds, so a carpet's box is flat and never blocks
/// (`BlockCarpet.java:33-38`), a snow layer's height and a fence gate's shape follow the cell's
/// metadata (`BlockSnow.java:36-41`, `BlockFenceGate.java:36-40`); a double plant's material is
/// `plants`, not replaceable (`BlockDoublePlant.java:30`; java18 `vine`); and the blocks' own
/// tests are 1.7.10's ([`can_place_block_on_side_v5`]).
#[must_use]
pub fn can_place_entity_on_side(
    world: &World,
    placing: mc_data::BlockId,
    pos: BlockPos,
    side: Facing,
    blocking: &[Aabb],
) -> bool {
    let cell = world.block_at(pos);
    if let Some(bb) = placing_collision_box(world, placing, pos, side, cell) {
        if blocking.iter().any(|other| bb.intersects_with(other)) {
            return false;
        }
    }
    let material = mc_data::v5::material_v5(cell);
    if material == Material::Circuits && placing.0 == 145 {
        return true;
    }
    if !material.is_replaceable() {
        return false;
    }
    can_place_block_on_side_v5(world, placing, pos, side)
}

/// java17: `Block.canPlaceBlockOnSide` (`Block.java:809-812`, `canPlaceBlockAt` unless a block
/// overrides it) for the block placed at `pos` from a click on `side`.
///
/// 1.8.9's rules ([`World::can_place_block_on_side`]) except where 1.7.10's read differently:
///
/// * a snow layer below has a solid top at 8 layers, `(meta & 7) == 7`, where 1.8.9 needs 7
///   (java17 `World.doesBlockHaveSolidTopSurface`, `World.java:2438-2443`; java18
///   `World.java:2419-2423`), for rails (`BlockRailBase.java:89-92`), pumpkins
///   (`BlockPumpkin.java:109-112`), flower pots (`BlockFlowerPot.java:121-124`), doors
///   (`BlockDoor.java:360-363`), repeaters and comparators (`BlockRedstoneDiode.java:31-34`),
///   redstone wire (`BlockRedstoneWire.java:60-63`), pressure plates
///   (`BlockBasePressurePlate.java:71-74`), a lever on a floor (`BlockLever.java:43-46`) and
///   torches (`BlockTorch.java:43-59`); snow stacks on 8 layers only (`BlockSnow.java:70-74`;
///   java18 7 or 8, `BlockSnow.java:82-86`);
/// * a replaceable cell is one of 1.7.10's replaceable materials (a double plant is not);
/// * a torch stands on a fence, nether brick fence, glass or cobblestone wall but not on stained
///   glass, which 1.8.9 adds (`BlockTorch.canPlaceTorchOn`, `:43-54`);
/// * a button attaches to walls only (`BlockButton.canPlaceBlockOnSide`, `BlockButton.java:48-51`).
#[must_use]
pub fn can_place_block_on_side_v5(
    world: &World,
    block: mc_data::BlockId,
    pos: BlockPos,
    side: Facing,
) -> bool {
    let below = pos.down();
    let below_state = world.block_at(below);
    let solid_top = || mc_data::v5::has_solid_top_surface_v5(below_state);
    // `Block.canPlaceBlockAt`: `getBlock(x, y, z).blockMaterial.isReplaceable()`.
    let replaceable = |at: BlockPos| mc_data::v5::material_v5(world.block_at(at)).is_replaceable();
    match block.0 {
        27 | 28 | 66 | 157 => solid_top(),
        86 | 91 | 140 => replaceable(pos) && solid_top(),
        64 | 71 => pos.y < 255 && solid_top() && replaceable(pos) && replaceable(pos.up()),
        93 | 94 | 149 | 150 => solid_top() && replaceable(pos),
        // ... or glowstone below.
        55 => solid_top() || below_state.block().0 == 89,
        // ... or a fence below: `BlockFence.isFence`, the fence and the nether brick fence.
        70 | 72 | 147 | 148 => solid_top() || matches!(below_state.block().0, 85 | 113),
        69 if side == Facing::Up => solid_top(),
        // A normal cube beside (`isBlockNormalCubeDefault(.., true)`), or `canPlaceTorchOn`.
        50 | 75 | 76 => {
            Facing::HORIZONTALS
                .iter()
                .any(|&facing| world.is_block_normal_cube(pos.relative(facing), true))
                || solid_top()
                || matches!(below_state.block().0, 85 | 113 | 20 | 139)
        }
        78 => {
            let props = below_state.props();
            match props.block.0 {
                79 | 174 => false,
                _ if props.material == Material::Leaves => true,
                78 if below_state.meta() & 7 == 7 => true,
                _ => props.opaque_cube && props.material.info().blocks_movement,
            }
        }
        77 | 143 => {
            side.axis().is_horizontal() && world.is_normal_cube_at(pos.relative(side.opposite()))
        }
        _ => world.can_place_block_on_side(block, pos, side),
    }
}

/// java17: `getCollisionBoundingBoxFromPool(world, x, y, z)` of the block being placed, built in
/// the target cell (see [`can_place_entity_on_side`]). `None` is Java's null.
fn placing_collision_box(
    world: &World,
    placing: mc_data::BlockId,
    pos: BlockPos,
    side: Facing,
    cell: BlockState,
) -> Option<Aabb> {
    let (x, y, z) = (f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));
    match placing.0 {
        // `(double)((float)y + 0 * 0.0625F)`: flat.
        171 => Some(Aabb {
            min: DVec3::new(x, y, z),
            max: DVec3::new(x + 1.0, y, z + 1.0),
        }),
        // `(float)y + (float)(meta & 7) * 0.125F` with the cell's metadata.
        78 => {
            let layers = f32::from(cell.meta() & 7);
            Some(Aabb::from_bounds(
                x,
                y,
                z,
                x + 1.0,
                f64::from(pos.y as f32 + layers * 0.125),
                z + 1.0,
            ))
        }
        // The cell's metadata decides open (null) and the axis.
        107 => {
            super::overridden_collision_box(BlockState::new(placing, cell.meta()), pos).flatten()
        }
        _ => {
            let state = placing.default_state();
            if let Some(overridden) = super::overridden_collision_box(state, pos) {
                return overridden;
            }
            let props = state.props();
            if props.collision.is_empty() {
                return None;
            }
            let bounds = if placing.0 == super::LADDER_BLOCK_ID {
                BlockState::new(
                    placing,
                    super::ladder_placed_facing(world, pos, side).index(),
                )
                .props()
                .bounds
            } else {
                props.bounds
            };
            let b = bounds.offset(x, y, z);
            Some(Aabb::from_bounds(
                b.min_x, b.min_y, b.min_z, b.max_x, b.max_y, b.max_z,
            ))
        }
    }
}

/// java17: `ItemBlock.onItemUse`'s choice of cell and side (`ItemBlock.java:42-79`): a one-layer
/// snow layer is replaced with the side forced to up, vine, tall grass and dead bush are
/// replaced, everything else steps one cell along the side. 1.8.9 also replaces double-plant grass
/// and fern and keeps the clicked side on snow (java18 `ItemBlock.java:38-56`).
#[must_use]
pub fn item_block_target(world: &World, hit_pos: BlockPos, side: Facing) -> (BlockPos, Facing) {
    let clicked = world.block_at(hit_pos);
    match clicked.block().0 {
        78 if clicked.meta() & 7 < 1 => (hit_pos, Facing::Up),
        106 | 31 | 32 => (hit_pos, side),
        _ => (hit_pos.relative(side), side),
    }
}

/// java17: `ItemBlock.onItemUse` (`ItemBlock.java:42-92`) decides by `canPlaceEntityOnSide` in
/// the cell [`item_block_target`] chooses, for a non-empty stack. `delegate` is the 1.8.9
/// `ItemBlock` model's result for the same cell ([`super::item_block_delegate_use`]); where the
/// two rule sets disagree ([`can_place_entity_on_side`]) 1.7.10's answer stands, a placement
/// 1.8.9 refused taking 1.8.9's `onBlockPlaced` state.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn item_block_use(
    world: &World,
    hit_pos: BlockPos,
    clicked_side: Facing,
    hit_y: f64,
    yaw: f64,
    stack: &ItemStack,
    blocking: &[Aabb],
    delegate: super::BlockUseResult,
) -> super::BlockUseResult {
    use super::BlockUseResult as Use;
    let Some(placing) = placed_block(stack) else {
        return delegate;
    };
    let (target, side) = item_block_target(world, hit_pos, clicked_side);
    let placed =
        stack.count != 0 && can_place_entity_on_side(world, placing, target, side, blocking);
    match delegate {
        // `NotApplicable`: an `ItemBlock` only 1.7.10 has (liquids, fire, portals, cocoa, crops,
        // double slabs), which the 1.8.9 model does not know.
        Use::Rejected | Use::NotApplicable if placed => Use::Consumed {
            placement: Some(super::BlockUsePlacement {
                pos: target,
                state: BlockState::new(
                    placing,
                    super::item_block_placed_metadata(
                        world, target, side, hit_y, yaw, stack, placing,
                    ),
                ),
            }),
            decrement_stack: true,
        },
        Use::NotApplicable
        | Use::Consumed { .. }
        | Use::ConsumedPair { .. }
        | Use::ConsumedMany { .. }
            if !placed =>
        {
            Use::Rejected
        }
        other => other,
    }
}

/// java17: `ItemDoor.onItemUse` (`ItemDoor.java:23-62`): a top face only, the door always one
/// cell up, and the door's `canPlaceBlockAt` there ([`can_place_block_on_side_v5`]). 1.8.9 puts
/// the door into a replaceable clicked block instead (java18 `ItemDoor.onItemUse`). The halves
/// are 1.8.9's ([`super::door_item_use`]) when it builds them in the same cell; otherwise the
/// door is placed and the server's block changes draw it.
#[must_use]
pub fn door_item_use(
    world: &World,
    hit_pos: BlockPos,
    side: Facing,
    yaw: f64,
    held: &ItemStack,
) -> super::BlockUseResult {
    use super::BlockUseResult as Use;
    let Some(door) = super::door_item_block(held) else {
        return Use::NotApplicable;
    };
    let target = hit_pos.up();
    if side != Facing::Up || !can_place_block_on_side_v5(world, door, target, Facing::Up) {
        return Use::Rejected;
    }
    let halves = super::door_item_use(world, hit_pos, side, yaw, held, true);
    if matches!(&halves, Use::ConsumedPair { placements, .. } if placements[0].pos == target) {
        halves
    } else {
        Use::Consumed {
            placement: None,
            decrement_stack: true,
        }
    }
}

/// The held stack and facts the shared entity classifiers
/// ([`super::entity_interaction_consumed_with_facts`], [`super::entity_local_interaction`],
/// [`super::horse_mount_intent`]) need in order to answer
/// for 1.7.10, where 1.7.10's client-side `interact` bodies differ from 1.8.9's only in which
/// item or fact counts:
///
/// * a chicken breeds with any `ItemSeeds`: wheat, pumpkin and melon seeds and nether wart
///   (java17 `EntityChicken.isBreedingItem`, `EntityChicken.java:152-155`; 1.8.9: wheat seeds);
/// * a horse eats bread, with wheat's yes/no answers (java17 `EntityHorse.interact`,
///   `EntityHorse.java:764-781`: bread `(7.0, 180, 3)` against wheat `(2.0, 60, 3)`; every branch
///   decides on the same three signs, only the local amounts differ);
/// * a cow or mooshroom child is milked too (java17 `EntityCow.interact`, `EntityCow.java:110`:
///   `bucket && !creative`; 1.8.9 adds `!isChild()`);
/// * an angry wolf can be leashed (java17 `EntityWolf` has no `allowLeashing` override; 1.8.9
///   adds `!isAngry()`, java18 `EntityWolf.java:635-638`).
#[must_use]
pub fn entity_classifier_inputs(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    mut facts: super::EntityInteractionFacts,
) -> (Option<ItemStack>, super::EntityInteractionFacts) {
    let mut held = held.cloned();
    if let Some(stack) = held.as_mut() {
        match kind {
            Some(EntityType::Chicken) if matches!(stack.item_id, 361 | 362 | 372) => {
                stack.item_id = super::WHEAT_SEEDS_ITEM_ID;
            }
            Some(EntityType::EntityHorse) if stack.item_id == BREAD_ITEM_ID => {
                stack.item_id = super::WHEAT_ITEM_ID;
            }
            Some(EntityType::Cow | EntityType::MushroomCow)
                if stack.item_id == super::BUCKET_ITEM_ID =>
            {
                facts.base.child = false;
            }
            Some(EntityType::Wolf) if stack.item_id == super::LEAD_ITEM_ID => {
                facts.wolf_angry = Some(false);
            }
            _ => {}
        }
    }
    (held, facts)
}

/// `minecraft:bread`, which a 1.7.10 horse eats.
const BREAD_ITEM_ID: i16 = 297;

/// java17: `EntityAnimal.interact` (`EntityAnimal.java:310-333`) has no child branch: a child
/// offered its breeding item falls through to `EntityAgeable.interact`, which returns false with
/// no local effect, where 1.8.9 feeds it (java18 `EntityAnimal.java:164-169`). A tamed, hurt
/// wolf still eats its meat in `EntityWolf.interact`, before `super` (java17
/// `EntityWolf.java:328-343`). `held` and `facts` are [`entity_classifier_inputs`]'s.
#[must_use]
pub fn child_breeding_declined(
    kind: Option<EntityType>,
    held: Option<&ItemStack>,
    facts: super::EntityInteractionFacts,
) -> bool {
    if !facts.base.child || !super::animal_breeding_item(kind, held.map(|stack| stack.item_id)) {
        return false;
    }
    !(kind == Some(EntityType::Wolf)
        && facts.wolf_tamed == Some(true)
        && facts.wolf_hurt == Some(true))
}

fn is_adventure(game_mode: GameMode) -> bool {
    // java17: `GameType.isAdventure()` is `this == ADVENTURE` (`WorldSettings.java:143-146`);
    // 1.7.10 has no spectator mode, and the protocol-5 join maps unknown ids to survival.
    matches!(game_mode, GameMode::Adventure)
}

fn is_creative(game_mode: GameMode) -> bool {
    matches!(game_mode, GameMode::Creative)
}

impl PlayerController {
    /// java17: `PlayerControllerMP.clickBlock` (`PlayerControllerMP.java:138-181`).
    ///
    /// 1.8.9's [`Self::click_block`] with the 1.7.10 gate: `if (!currentGameType.isAdventure() ||
    /// thePlayer.isCurrentToolAdventureModeExempt(x, y, z))` instead of the spectator and
    /// `CanDestroy` checks, and no world border (java18 `:200-227`). Creative, the ABORT of a
    /// previous dig (with the new click's face), the first-click callback and the instant break
    /// are the same statements.
    pub fn click_block_v5(
        &mut self,
        world: &World,
        hit: BlockHit,
        game_mode: GameMode,
        allow_edit: bool,
        held: Option<&ItemStack>,
        on_ground: bool,
    ) -> DigStep {
        self.pending_block_click = None;
        self.pending_abort = None;
        if is_adventure(game_mode)
            && !is_current_tool_adventure_mode_exempt(world, hit.pos, held, allow_edit)
        {
            return DigStep::None;
        }
        if is_creative(game_mode) {
            self.hit_delay = 5;
            return DigStep::StartAndDestroy(hit.pos, hit.face);
        }
        if self.hitting && self.is_hitting_position(hit.pos, held) {
            return DigStep::None;
        }
        self.pending_abort = self.hitting.then_some(self.current_block);
        let state = world.block_at(hit.pos);
        let solid = !state.props().is_air();
        if solid && self.damage == 0.0 {
            self.pending_block_click = world.on_block_clicked(hit.pos);
        }
        if solid && player_relative_hardness(state, self.dig_context, on_ground) >= 1.0 {
            return DigStep::StartAndDestroy(hit.pos, hit.face);
        }
        self.hitting = true;
        self.current_block = hit.pos;
        self.current_item_hitting_block = held.cloned();
        self.damage = 0.0;
        self.step_sound_counter = 0.0;
        self.pending_break_progress = Some((hit.pos, -1));
        DigStep::Start(hit.pos, hit.face)
    }

    /// java17: `PlayerControllerMP.onPlayerDamageBlock` (`PlayerControllerMP.java:195-247`),
    /// after its `syncCurrentPlayItem()` (the caller's).
    ///
    /// The method returns `void` in 1.7.10: the caller swings on the adventure exemption, not on
    /// a result (java17 `Minecraft.java:1324-1330`), so [`DamageTick::progressed`] carries
    /// nothing here. The cooldown, the
    /// creative branch (no world border), the continuation of the same block and tool, the
    /// accumulation with 1.7.10's [`player_relative_hardness`] and the target change through
    /// [`Self::click_block_v5`] follow 1.8.9's statement order.
    pub fn damage_block_v5(
        &mut self,
        world: &World,
        hit: BlockHit,
        game_mode: GameMode,
        allow_edit: bool,
        held: Option<&ItemStack>,
        on_ground: bool,
    ) -> DamageTick {
        if self.hit_delay > 0 {
            self.hit_delay -= 1;
            return DamageTick::stalled();
        }
        if is_creative(game_mode) {
            self.hit_delay = 5;
            self.pending_abort = None;
            return DamageTick::sending(DigStep::StartAndDestroy(hit.pos, hit.face));
        }
        if self.is_hitting_position(hit.pos, held) {
            let state = world.block_at(hit.pos);
            if state.props().is_air() {
                self.hitting = false;
                return DamageTick::stalled();
            }
            self.damage += player_relative_hardness(state, self.dig_context, on_ground);
            let hit_sound = (self.step_sound_counter % 4.0 == 0.0).then_some(BlockHitSound {
                pos: hit.pos,
                state,
            });
            self.step_sound_counter += 1.0;
            let step = if self.damage >= 1.0 {
                self.hitting = false;
                self.damage = 0.0;
                self.step_sound_counter = 0.0;
                self.hit_delay = 5;
                DigStep::Finish(hit.pos, hit.face)
            } else {
                DigStep::None
            };
            // java17: `destroyBlockInWorldPartially(.., (int)(curBlockDamageMP * 10.0F) - 1)`
            // after both branches; the finished dig's zeroed damage gives `-1`.
            self.pending_break_progress =
                Some((self.current_block, (self.damage * 10.0) as i32 - 1));
            return DamageTick {
                step,
                progressed: true,
                hit_sound,
            };
        }
        DamageTick::sending(self.click_block_v5(world, hit, game_mode, allow_edit, held, on_ground))
    }

    /// java17: `PlayerControllerMP.resetBlockRemoving` (`PlayerControllerMP.java:183-193`).
    ///
    /// ```text
    /// if (isHittingBlock) addToSendQueue(new C07PacketPlayerDigging(1, currentBlock.., -1));
    /// isHittingBlock = false;
    /// curBlockDamageMP = 0.0F;
    /// destroyBlockInWorldPartially(entityId, currentBlock.., -1);
    /// ```
    ///
    /// The ABORT's face is `-1` (byte `0xFF`), and the reset happens whether or not a dig was in
    /// progress (java18 does all of it inside `if (isHittingBlock)` with face `DOWN`,
    /// `PlayerControllerMP.java:274-283`). [`DigStep::Abort`] from this method is that face-`-1`
    /// ABORT,
    /// its block taken by [`Self::take_pending_abort`].
    pub fn reset_block_removing_v5(&mut self) -> DigStep {
        self.pending_block_click = None;
        let step = if self.hitting {
            self.pending_abort = Some(self.current_block);
            DigStep::Abort
        } else {
            DigStep::None
        };
        self.hitting = false;
        self.damage = 0.0;
        self.pending_break_progress = Some((self.current_block, -1));
        step
    }

    /// java17: `PlayerControllerMP.blockHitDelay`, which the 1.7.10 oracle compares.
    #[cfg(test)]
    pub(crate) fn block_hit_delay(&self) -> i32 {
        self.hit_delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(name: &str, meta: u8) -> BlockState {
        BlockState::new(mc_data::block_by_name(name).expect(name), meta)
    }

    /// A loaded, empty column around `pos`.
    fn world_at(pos: BlockPos) -> World {
        let mut world = World::new();
        let bitmask = 1u16 << ((pos.y >> 4) as u16);
        let data = vec![0u8; mc_world::expected_chunk_data_len(bitmask, true, true)];
        world
            .apply_chunk_data(pos.chunk_pos(), true, bitmask, &data)
            .expect("an empty column decodes");
        world
    }

    fn stack(id: i16) -> ItemStack {
        ItemStack {
            item_id: id,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    fn context(held: Option<&ItemStack>, fatigue: Option<i8>, underwater: bool) -> DigContext {
        DigContext::from_player(held, None, fatigue, underwater, false)
    }

    #[test]
    fn the_cursor_narrows_each_operand_before_subtracting() {
        // At x = 30000.4999 java17 sends byte 8, java18 byte 7.
        let hit = BlockHit {
            pos: BlockPos::new(30_000, 64, 0),
            face: Facing::Up,
            hit_vec: DVec3::new(30_000.499_9, 65.0, 0.5),
        };
        let v5 = cursor(&hit);
        let v47 = hit.cursor();
        assert_eq!(((v5.x as f32 * 16.0) as i32) as u8, 8);
        assert_eq!(
            ((v47.x as f32 * 16.0) as i32) as u8,
            7,
            "negative control: 1.8.9"
        );
        // Small coordinates agree.
        let near = BlockHit {
            pos: BlockPos::new(3, 64, 7),
            face: Facing::Up,
            hit_vec: DVec3::new(3.25, 65.0, 7.75),
        };
        assert_eq!(cursor(&near), near.cursor());
    }

    #[test]
    fn mining_fatigue_scales_linearly() {
        // Stone (hardness 1.5) with a diamond pickaxe (8.0), harvestable (/30).
        let pick = stack(278);
        let stone = state("stone", 0);
        let none = player_relative_hardness(stone, context(Some(&pick), None, false), true);
        assert_eq!(none, 8.0 / 1.5 / 30.0);
        let one = player_relative_hardness(stone, context(Some(&pick), Some(0), false), true);
        assert_eq!(one, 8.0 * 0.8 / 1.5 / 30.0);
        let two = player_relative_hardness(stone, context(Some(&pick), Some(1), false), true);
        assert_eq!(two, 8.0 * (1.0 - 2.0 * 0.2) / 1.5 / 30.0);
        // Amplifier 4 stops digging; 5 goes negative.
        let five = player_relative_hardness(stone, context(Some(&pick), Some(4), false), true);
        assert_eq!(five, 8.0 * (1.0 - 5.0_f32 * 0.2) / 1.5 / 30.0);
        let six = player_relative_hardness(stone, context(Some(&pick), Some(5), false), true);
        assert!(six < 0.0, "{six}");
        // Negative control: 1.8.9's table gives 0.3 for amplifier 0.
        let v47 = super::super::player_relative_hardness_with_context(
            stone,
            context(Some(&pick), Some(0), false),
            true,
        );
        assert_eq!(v47, 8.0 * 0.3 / 1.5 / 30.0);
        assert_ne!(v47, one);
    }

    #[test]
    fn packed_ice_melon_and_ladder_are_not_effective_blocks() {
        // The real 1.7.10 client's tick counts for these three digs: 15, 30 and 13.
        let pick = stack(278);
        let axe = stack(279);
        // A ladder has no metadata-0 state: 2 is one facing north.
        for (name, meta, tool, v5_ticks) in [
            ("packed_ice", 0, &pick, 15),
            ("melon_block", 0, &axe, 30),
            ("ladder", 2, &axe, 13),
        ] {
            let block = state(name, meta);
            let per_tick = player_relative_hardness(block, context(Some(tool), None, false), true);
            let mut damage = 0.0_f32;
            let mut ticks = 0;
            while damage < 1.0 {
                damage += per_tick;
                ticks += 1;
            }
            assert_eq!(ticks, v5_ticks, "{name}");
            let v47 = super::super::player_relative_hardness_with_context(
                block,
                context(Some(tool), None, false),
                true,
            );
            assert!(v47 > per_tick, "negative control: 1.8.9 digs {name} faster");
        }
        // Stone stays effective for the pickaxe; logs for the axe.
        let stone = state("stone", 0);
        assert_eq!(strength_vs_block(stone, registered_item(&pick)), 8.0);
        assert_eq!(
            strength_vs_block(state("log", 0), registered_item(&axe)),
            8.0
        );
    }

    #[test]
    fn the_water_test_is_twelve_hundredths_above_the_eye() {
        // A still water source at y = 65, tested 0.12 above the 1.7.10 eye: an eye at 64.9
        // tests 65.02 (inside), 64.7 tests 64.82 (below the block), 65.85 tests 65.97 (under the
        // surface) and 65.9 tests 66.02 (above it).
        let pos = BlockPos::new(0, 65, 0);
        let mut world = world_at(pos);
        world.set_block(pos, state("water", 0));
        assert!(is_inside_of_water(&world, 0.5, 64.9, 0.5));
        assert!(
            !is_inside_of_water(&world, 0.5, 64.7, 0.5),
            "64.82 is below the block"
        );
        // The source's surface is at 66 - (1/9 - 0.11111111F) = just below 66.
        assert!(is_inside_of_water(&world, 0.5, 65.85, 0.5));
        assert!(
            !is_inside_of_water(&world, 0.5, 65.9, 0.5),
            "66.02 is above the surface"
        );
    }

    #[test]
    fn the_adventure_exemption_reads_material_then_tool() {
        let pos = BlockPos::new(1, 64, 1);
        let mut world = world_at(pos);
        world.set_block(pos, state("glass", 0));
        assert!(is_current_tool_adventure_mode_exempt(
            &world, pos, None, false
        ));
        world.set_block(pos, state("stone", 0));
        assert!(!is_current_tool_adventure_mode_exempt(
            &world, pos, None, false
        ));
        let pick = stack(278);
        assert!(is_current_tool_adventure_mode_exempt(
            &world,
            pos,
            Some(&pick),
            false
        ));
        let sword = stack(276);
        assert!(!is_current_tool_adventure_mode_exempt(
            &world,
            pos,
            Some(&sword),
            false
        ));
        // allowEdit exempts everything, even air.
        assert!(is_current_tool_adventure_mode_exempt(
            &world,
            BlockPos::new(1, 65, 1),
            None,
            true
        ));
        assert!(!is_current_tool_adventure_mode_exempt(
            &world,
            BlockPos::new(1, 65, 1),
            None,
            false
        ));
        // A shovel on snow: harvestable by the item's own rule.
        world.set_block(pos, state("snow", 0));
        assert!(is_current_tool_adventure_mode_exempt(
            &world,
            pos,
            Some(&stack(277)),
            false
        ));
        // Wool with shears: strength 5 > 1.
        world.set_block(pos, state("wool", 0));
        assert!(is_current_tool_adventure_mode_exempt(
            &world,
            pos,
            Some(&stack(359)),
            false
        ));
        assert!(!is_current_tool_adventure_mode_exempt(
            &world,
            pos,
            Some(&stack(1)),
            false
        ));
    }

    #[test]
    fn any_held_stack_may_edit_in_1_7_10() {
        assert!(can_player_edit(Some(&stack(1)), false));
        assert!(!can_player_edit(None, false));
        assert!(can_player_edit(None, true));
    }

    #[test]
    fn the_survival_entity_ray_is_three_blocks_long_and_keeps_the_block_behind() {
        // A 0.6-wide entity whose box (with its 0.1 border) starts 3.1 from the eye, and the
        // wall's face 3.5 from it.
        let eye = DVec3::new(0.5, 65.62, 0.5);
        let wall = BlockPos::new(0, 65, 4);
        let mut world = world_at(wall);
        world.set_block(wall, state("stone", 0));
        let look = super::super::look_vector(0.0, 0.0);
        let entity = Pickable {
            network_id: 9,
            bounds: super::super::size_bounding_box(DVec3::new(0.5, 64.0, 4.0), 0.6, 1.8),
            border_size: 0.1,
            collidable: true,
        };
        let v5 = pick(&world, eye, look, 4.5, false, None, [entity]);
        assert!(
            matches!(v5, Some(HitResult::Block(hit)) if hit.pos == wall),
            "1.7.10 targets the wall behind: {v5:?}"
        );
        let v47 = super::super::pick(&world, eye, look, 4.5, false, [entity]);
        assert_eq!(
            v47,
            HitResult::Miss,
            "negative control: 1.8.9's MISS conversion"
        );
        // Within 3.0 both hit the entity.
        let near = Pickable {
            bounds: super::super::size_bounding_box(DVec3::new(0.5, 64.0, 2.5), 0.6, 1.8),
            ..entity
        };
        assert!(matches!(
            pick(&world, eye, look, 4.5, false, None, [near]),
            Some(HitResult::Entity(hit)) if hit.network_id == 9
        ));
        // Creative: 6.0.
        assert_eq!(entity_reach(5.0, true), 6.0);
        assert_eq!(entity_reach(4.5, false), 3.0);
    }

    #[test]
    fn a_ray_up_a_ladder_shaft_finds_nothing_at_all() {
        // java17 `World.rayTraceBlocks` keeps a MISS only for a block that cannot stop the ray
        // (`World.java:962-1140`). Looking straight up a shaft of ladders crosses five
        // collidable cells whose thin boxes the ray misses, so `objectMouseOver` is null.
        let eye = DVec3::new(0.5, 64.62, 0.5);
        let look = super::super::look_vector(0.0, -90.0);
        let mut world = world_at(BlockPos::new(0, 64, 0));
        assert_eq!(
            pick(&world, eye, look, 4.5, false, None, std::iter::empty()),
            Some(HitResult::Miss),
            "negative control: an open shaft keeps its last air block as a MISS"
        );
        for y in 65..=69 {
            world.set_block(BlockPos::new(0, y, 0), state("ladder", 2));
        }
        assert_eq!(
            pick(&world, eye, look, 4.5, false, None, std::iter::empty()),
            None
        );
    }

    #[test]
    fn a_snow_layer_holds_a_rail_at_eight_layers() {
        // java17 `World.doesBlockHaveSolidTopSurface` (`World.java:2438-2443`): a snow layer's
        // top is solid at `(meta & 7) == 7`; java18 at 7 layers, meta 6.
        let snow = BlockPos::new(8, 64, 8);
        let above = snow.up();
        let rail = mc_data::BlockId(66);
        let mut world = world_at(snow);
        for (meta, v5, v47) in [(7, true, false), (6, false, true), (0, false, false)] {
            world.set_block(snow, state("snow_layer", meta));
            assert_eq!(
                can_place_block_on_side_v5(&world, rail, above, Facing::Up),
                v5,
                "meta {meta}"
            );
            assert_eq!(
                world.can_place_block_on_side(rail, above, Facing::Up),
                v47,
                "negative control, 1.8.9, meta {meta}"
            );
        }
        // java17 `BlockSnow.canPlaceBlockAt` (`BlockSnow.java:70-74`): snow stacks on 8 layers.
        let snow_layer = mc_data::BlockId(78);
        world.set_block(snow, state("snow_layer", 6));
        assert!(!can_place_block_on_side_v5(
            &world,
            snow_layer,
            above,
            Facing::Up
        ));
        assert!(
            world.can_place_block_on_side(snow_layer, above, Facing::Up),
            "negative control: 1.8.9 stacks on 7 layers"
        );
        world.set_block(snow, state("snow_layer", 7));
        assert!(can_place_block_on_side_v5(
            &world,
            snow_layer,
            above,
            Facing::Up
        ));
    }

    #[test]
    fn a_torch_stands_on_glass_but_not_on_stained_glass() {
        // java17 `BlockTorch.canPlaceTorchOn` (`BlockTorch.java:43-54`): a solid top, a fence,
        // a nether brick fence, glass or a cobblestone wall; java18 adds stained glass.
        let support = BlockPos::new(8, 64, 8);
        let cell = support.up();
        let torch = mc_data::BlockId(50);
        let mut world = world_at(support);
        world.set_block(support, state("stained_glass", 0));
        assert!(!can_place_block_on_side_v5(&world, torch, cell, Facing::Up));
        assert!(
            world.can_place_block_on_side(torch, cell, Facing::Up),
            "negative control: 1.8.9"
        );
        world.set_block(support, state("glass", 0));
        assert!(can_place_block_on_side_v5(&world, torch, cell, Facing::Up));
        // `canPlaceBlockAt` (`:56-59`) also takes a normal cube beside, whatever is below.
        world.set_block(support, state("stained_glass", 0));
        world.set_block(cell.relative(Facing::North), state("stone", 0));
        assert!(can_place_block_on_side_v5(&world, torch, cell, Facing::Up));
    }

    #[test]
    fn an_item_block_is_placed_by_the_1_7_10_rule() {
        // A torch clicked onto an 8-layer snow layer: 1.7.10's `ItemBlock.onItemUse` places it on
        // top (`canPlaceEntityOnSide` with 1.7.10's solid top); 1.8.9's model refuses.
        let snow = BlockPos::new(8, 64, 8);
        let mut world = world_at(snow);
        world.set_block(snow, state("snow_layer", 7));
        let torch = stack(50);
        let delegate = super::super::item_block_delegate_use(
            &world,
            snow,
            Facing::Up,
            1.0,
            0.0,
            &torch,
            true,
            &[],
        );
        assert_eq!(
            delegate,
            super::super::BlockUseResult::Rejected,
            "negative control: 1.8.9"
        );
        let v5 = item_block_use(&world, snow, Facing::Up, 1.0, 0.0, &torch, &[], delegate);
        assert!(
            matches!(
                v5,
                super::super::BlockUseResult::Consumed {
                    placement: Some(placement),
                    decrement_stack: true,
                } if placement.pos == snow.up() && placement.state.block().0 == 50
            ),
            "{v5:?}"
        );
    }

    #[test]
    fn a_door_goes_one_cell_up_from_a_snow_layer() {
        // java17 `ItemDoor.onItemUse` (`ItemDoor.java:23-62`) puts the door one cell up and asks
        // `BlockDoor.canPlaceBlockAt` there: above a one-layer snow layer, whose top is not
        // solid, it is refused. java18 puts the door into the replaceable snow layer.
        let ground = BlockPos::new(8, 64, 8);
        let snow = ground.up();
        let mut world = world_at(ground);
        world.set_block(ground, state("stone", 0));
        world.set_block(snow, state("snow_layer", 0));
        let door = stack(324);
        assert_eq!(
            door_item_use(&world, snow, Facing::Up, 0.0, &door),
            super::super::BlockUseResult::Rejected
        );
        assert!(
            matches!(
                super::super::door_item_use(&world, snow, Facing::Up, 0.0, &door, true),
                super::super::BlockUseResult::ConsumedPair { placements, .. }
                    if placements[0].pos == snow
            ),
            "negative control: 1.8.9"
        );
        // From the stone's top face both put it into the snow layer's cell, one up.
        assert!(matches!(
            door_item_use(&world, ground, Facing::Up, 0.0, &door),
            super::super::BlockUseResult::ConsumedPair { placements, .. }
                if placements[0].pos == snow
        ));
        assert_eq!(
            door_item_use(&world, ground, Facing::North, 0.0, &door),
            super::super::BlockUseResult::Rejected
        );
    }

    #[test]
    fn ascending_powered_rails_trace_the_flat_box() {
        // A powered rail ascending east and powered (meta 2 | 8 = 10).
        let rail = BlockPos::new(2, 64, 0);
        let mut world = world_at(rail);
        world.set_block(rail, state("golden_rail", 10));
        // A ray passing 0.3 above the rail's base: inside the 0.625 slope box, above 0.125.
        let start = DVec3::new(0.5, 64.3, 0.5);
        let end = DVec3::new(5.5, 64.3, 0.5);
        let v5 = raytrace_blocks(&world, start, end).and_then(RayTrace::as_block);
        assert!(v5.is_none(), "1.7.10 passes over the flat box: {v5:?}");
        let v47 = super::super::raytrace_blocks(&world, start, end).and_then(RayTrace::as_block);
        assert_eq!(
            v47.map(|hit| hit.pos),
            Some(rail),
            "negative control: 1.8.9 slope box"
        );
        // An unpowered ascending rail (meta 2) is the slope box in both.
        world.set_block(rail, state("golden_rail", 2));
        let v5 = raytrace_blocks(&world, start, end).and_then(RayTrace::as_block);
        assert_eq!(v5.map(|hit| hit.pos), Some(rail));
    }

    #[test]
    fn a_ray_starting_on_a_block_boundary_takes_the_untweaked_step() {
        // The ray starts exactly on the plane x = 1.0 and travels -x, so the first x
        // step parameter is 0. 1.8.9 replaces it with -1.0E-4 and moves y and z backwards along
        // the ray while x stays on the plane; 1.7.10 steps with 0 and the cursor stays put.
        let target = BlockPos::new(0, 64, 0);
        let mut world = world_at(target);
        world.set_block(target, state("stone", 0));
        let start = DVec3::new(1.0, 64.5, 0.5);
        let end = DVec3::new(-3.0, 63.0, 1.5);
        let v5 = raytrace_blocks(&world, start, end)
            .and_then(RayTrace::as_block)
            .expect("1.7.10 hits the block");
        assert_eq!((v5.pos, v5.face), (target, Facing::East));
        assert_eq!(v5.hit_vec, DVec3::new(1.0, 64.5, 0.5));
        let v47 = super::super::raytrace_blocks(&world, start, end)
            .and_then(RayTrace::as_block)
            .expect("1.8.9 hits the block");
        assert_eq!(v47.pos, target);
        assert!(
            v47.hit_vec.z < 0.5,
            "negative control: the nudged cursor {v47:?}"
        );
        // The C08 bytes differ: z 0.5 -> 8 against 0.4999 -> 7.
        let byte = |v: f64| ((v as f32 * 16.0) as i32) as u8;
        assert_eq!(byte(cursor(&v5).z), 8);
        assert_eq!(byte(v47.cursor().z), 7);
    }

    #[test]
    fn a_moving_piston_is_targetable_with_its_carried_box() {
        // 1.8.9 looks through block 36; 1.7.10 traces the carried block.
        let pos = BlockPos::new(2, 64, 0);
        let mut world = world_at(pos);
        world.set_block(pos, state("piston_extension", 5));
        let start = DVec3::new(0.5, 64.5, 0.5);
        let end = DVec3::new(5.5, 64.5, 0.5);
        let v47 = super::super::raytrace_blocks(&world, start, end).and_then(RayTrace::as_block);
        assert!(v47.is_none(), "negative control: 1.8.9 returns null for 36");
        let v5 = raytrace_blocks(&world, start, end).and_then(RayTrace::as_block);
        assert_eq!(
            v5.map(|hit| hit.pos),
            Some(pos),
            "no tile entity: the full cube"
        );
    }

    #[test]
    fn a_moving_piston_shifts_its_carried_box_by_the_progress() {
        // java17 `BlockPistonMoving.setBlockBoundsBasedOnState` (`BlockPistonMoving.java:
        // 136-165`): the carried block's bounds minus the push offset times the progress. A
        // bottom slab pulled down (retracting, facing down) half way: `0.5..1.0` in its cell.
        let pos = BlockPos::new(2, 64, 0);
        let mut world = world_at(pos);
        world.set_block(pos, state("piston_extension", 0));
        world.set_tile_entity(
            pos,
            mc_world::TileEntity {
                kind: mc_world::TileKind::Piston,
                data: mc_world::TileData::Piston(mc_world::tileent::PistonState {
                    piston_state: state("stone_slab", 0),
                    facing: Facing::Down,
                    extending: false,
                    should_head_be_rendered: false,
                    progress: 0.5,
                    last_progress: 0.5,
                }),
            },
        );
        let ray = |y: f64| {
            let (start, end) = (DVec3::new(0.5, y, 0.5), DVec3::new(5.5, y, 0.5));
            (
                raytrace_blocks(&world, start, end).and_then(RayTrace::as_block),
                super::super::raytrace_blocks(&world, start, end).and_then(RayTrace::as_block),
            )
        };
        let (v5, v47) = ray(64.75);
        assert_eq!(v5.map(|hit| hit.pos), Some(pos), "the shifted upper half");
        assert!(v47.is_none(), "negative control: 1.8.9 returns null for 36");
        assert_eq!(ray(64.25).0, None, "the slab's own lower half is empty now");
    }

    #[test]
    fn hanging_boxes_are_inverted_on_the_facing_axis() {
        // A 1.7.10 item frame (9 px) facing south (direction 0).
        let pos = DVec3::new(0.5, 64.5, 1.0625);
        let b = hanging_bounding_box(pos, 0, 9, 9);
        assert!(b.min.z > b.max.z, "thin axis inverted: {b:?}");
        assert_eq!(b.max.x - b.min.x, f64::from(9.0_f32 / 32.0 * 2.0 - 0.0625));
        // Grown by the 0.1 border it has depth again.
        let grown = b.expand(0.1, 0.1, 0.1);
        assert!(grown.min.z < grown.max.z);
        assert_eq!(collision_border_size(Some(EntityType::ItemFrame)), 0.1);
        assert_eq!(
            super::super::collision_border_size(Some(EntityType::ItemFrame)),
            0.0,
            "negative control: 1.8.9"
        );
    }

    fn controller_on(world: &World, pos: BlockPos) -> (PlayerController, BlockHit) {
        let mut controller = PlayerController::new();
        controller.set_dig_context(context(None, None, false));
        let _ = world;
        (
            controller,
            BlockHit {
                pos,
                face: Facing::Up,
                hit_vec: DVec3::new(f64::from(pos.x) + 0.5, f64::from(pos.y) + 1.0, 0.5),
            },
        )
    }

    #[test]
    fn reset_aborts_with_the_hit_block_and_always_clears_progress() {
        let pos = BlockPos::new(0, 64, 0);
        let mut world = world_at(pos);
        world.set_block(pos, state("stone", 0));
        let (mut controller, hit) = controller_on(&world, pos);
        assert_eq!(
            controller.click_block_v5(&world, hit, GameMode::Survival, true, None, true),
            DigStep::Start(pos, Facing::Up)
        );
        let _ = controller.take_break_progress();
        assert_eq!(controller.reset_block_removing_v5(), DigStep::Abort);
        assert_eq!(controller.take_pending_abort(), Some(pos));
        assert_eq!(controller.take_break_progress(), Some((pos, -1)));
        // Not digging: no ABORT, but the crack is cleared again.
        assert_eq!(controller.reset_block_removing_v5(), DigStep::None);
        assert_eq!(controller.take_pending_abort(), None);
        assert_eq!(controller.take_break_progress(), Some((pos, -1)));
        // Negative control: 1.8.9's reset is a no-op when not digging.
        assert_eq!(controller.reset_block_removing(), DigStep::None);
        assert_eq!(controller.take_break_progress(), None);
    }

    #[test]
    fn adventure_digs_exempt_materials_and_harvestable_blocks_only() {
        let pos = BlockPos::new(0, 64, 0);
        let mut world = world_at(pos);
        world.set_block(pos, state("glass", 0));
        let (mut controller, hit) = controller_on(&world, pos);
        assert_eq!(
            controller.click_block_v5(&world, hit, GameMode::Adventure, false, None, true),
            DigStep::Start(pos, Facing::Up),
            "glass is adventure exempt in 1.7.10"
        );
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Adventure, false, None, true),
            DigStep::None,
            "negative control: 1.8.9 needs CanDestroy"
        );
        world.set_block(pos, state("stone", 0));
        let (mut controller, hit) = controller_on(&world, pos);
        assert_eq!(
            controller.click_block_v5(&world, hit, GameMode::Adventure, false, None, true),
            DigStep::None
        );
        let pick = stack(278);
        assert_eq!(
            controller.click_block_v5(&world, hit, GameMode::Adventure, false, Some(&pick), true),
            DigStep::Start(pos, Facing::Up)
        );
    }

    #[test]
    fn the_dig_finishes_with_1_7_10_fatigue_timing() {
        // The real 1.7.10 client's fatigue-0 dig (oracle `focus_dig_fatigue0`): stone, diamond
        // pickaxe, Mining Fatigue I: 8 ticks of damage after START (factor 0.8). 1.8.9 takes 19
        // (factor 0.3).
        let pos = BlockPos::new(0, 64, 0);
        let mut world = world_at(pos);
        world.set_block(pos, state("stone", 0));
        let pick = stack(278);
        let mut controller = PlayerController::new();
        controller.set_dig_context(context(Some(&pick), Some(0), false));
        let hit = BlockHit {
            pos,
            face: Facing::Up,
            hit_vec: DVec3::new(0.5, 65.0, 0.5),
        };
        assert_eq!(
            controller.click_block_v5(&world, hit, GameMode::Survival, true, Some(&pick), true),
            DigStep::Start(pos, Facing::Up)
        );
        let mut ticks = 0;
        loop {
            ticks += 1;
            let tick = controller.damage_block_v5(
                &world,
                hit,
                GameMode::Survival,
                true,
                Some(&pick),
                true,
            );
            if tick.step == DigStep::Finish(pos, Facing::Up) {
                break;
            }
            assert_eq!(tick.step, DigStep::None);
            assert!(ticks < 100);
        }
        assert_eq!(ticks, 8);
        // The five-tick hit delay follows.
        for _ in 0..5 {
            let tick = controller.damage_block_v5(
                &world,
                hit,
                GameMode::Survival,
                true,
                Some(&pick),
                true,
            );
            assert_eq!(tick.step, DigStep::None);
        }

        // Negative control: the same dig through the 1.8.9 controller.
        let mut controller = PlayerController::new();
        controller.set_dig_context(context(Some(&pick), Some(0), false));
        assert_eq!(
            controller.click_block(&world, hit, GameMode::Survival, true, Some(&pick), true),
            DigStep::Start(pos, Facing::Up)
        );
        let mut ticks = 0;
        loop {
            ticks += 1;
            let tick =
                controller.damage_block(&world, hit, GameMode::Survival, true, Some(&pick), true);
            if tick.step == DigStep::Finish(pos, Facing::Up) {
                break;
            }
            assert!(ticks < 100);
        }
        assert_eq!(ticks, 19, "negative control: 1.8.9's fatigue factor");
    }
}
