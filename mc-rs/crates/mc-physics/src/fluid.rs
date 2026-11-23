//! Liquid detection and the current a liquid pushes an entity with.
//!
//! java: `Entity.handleWaterMovement`/`isInLava`/`isOffsetPositionInLiquid`,
//! `World.handleMaterialAcceleration`/`isMaterialInBB`/`isAnyLiquid`, and
//! `BlockLiquid.getFlowVector`.

use mc_data::{block_props, Material};
use mc_math::{floor_f64, sqrt_d, vec3, Aabb, BlockPos, DVec3};
use mc_world::World;

use crate::collision::{CollisionBorder, EntityCollisionCandidate};
use crate::state::{PhysicsState, WaterEntry};

/// java: `BlockLiquid.getLiquidHeightPercent` — `(meta + 1) / 9.0F`, with falling levels
/// (`meta >= 8`) treated as a full source.
#[must_use]
pub fn liquid_height_percent(meta: u8) -> f32 {
    let meta = if meta >= 8 { 0 } else { meta };
    f32::from(meta + 1) / 9.0
}

/// java: `Entity.handleWaterMovement` (`Entity.java:1111`).
///
/// Sets [`PhysicsState::in_water`], zeroes `fall_distance` on entry, and applies the current
/// (see [`material_acceleration`]). The probe box is the entity box expanded `-0.4` in Y and
/// then contracted by `0.001` on every axis — the expansion is *negative*, so it shrinks the
/// box vertically, which is why your feet have to be properly in the water.
pub fn handle_water_movement(w: &World, s: &mut PhysicsState) -> bool {
    let bb = s
        .aabb()
        .expand(0.0, -0.400_000_005_960_464_5, 0.0)
        .contract(0.001, 0.001, 0.001);
    if material_acceleration(w, &bb, Material::Water, s) {
        if !s.in_water && !s.first_update {
            s.water_entry = Some(water_entry(s));
        }
        s.fall_distance = 0.0;
        s.in_water = true;
    } else {
        s.in_water = false;
    }
    s.in_water
}

/// The non-random output of java: `Entity.resetHeight`.
///
/// The Java method uses the entity-local `Random` only while fanning this recipe out to a sound
/// and particles. Physics reports the values and exact random controls to the client owner
/// instead of introducing a second, divergent RNG stream.
#[must_use]
pub fn water_entry(s: &PhysicsState) -> WaterEntry {
    // java: the additions/multiplications are doubles through `sqrt_double`, then `* 0.2F`
    // narrows back to f32 before the clamp.
    let speed = sqrt_d(
        s.vel.x * s.vel.x * 0.200_000_002_980_232_24
            + s.vel.y * s.vel.y
            + s.vel.z * s.vel.z * 0.200_000_002_980_232_24,
    ) * 0.2;
    let sound_volume = speed.min(1.0);

    // Java compares an int widened to f32 against an f32 threshold. Keeping the loop is exact
    // for non-standard entity widths too; a cast/ceil shortcut is not necessarily equivalent.
    let threshold = 1.0 + s.width * 20.0;
    let mut particle_count = 0_u32;
    while (particle_count as f32) < threshold {
        particle_count += 1;
    }

    WaterEntry {
        particle_origin: DVec3::new(s.pos.x, f64::from(floor_f64(s.aabb().min.y) + 1), s.pos.z),
        sound_volume,
        particle_count,
        particle_width: s.width,
        bubble_velocity: s.vel,
        bubble_y_random_subtract: 0.2,
        splash_velocity: s.vel,
        sound_pitch_base: 1.0,
        sound_pitch_jitter: 0.4,
    }
}

/// java: `Entity.isInLava` — `isMaterialInBB(bb.expand(-0.1, -0.4, -0.1), Material.lava)`.
#[must_use]
pub fn is_in_lava(w: &World, s: &PhysicsState) -> bool {
    let bb = s.aabb().expand(
        -0.100_000_001_490_116_12,
        -0.400_000_005_960_464_5,
        -0.100_000_001_490_116_12,
    );
    is_material_in_bb(w, &bb, Material::Lava)
}

/// java: `World.isMaterialInBB` — any block of that material inside the scan range.
#[must_use]
pub fn is_material_in_bb(w: &World, bb: &Aabb, material: Material) -> bool {
    for_each_block_in(bb, |pos| block_props(w.block_at(pos)).material == material)
}

/// java: `World.isAnyLiquid` (`World.java:2012`).
///
/// Unlike every other `World` material query this one scans
/// `floor(min)..=floor(max)` **inclusive**, not `floor(min)..floor(max + 1.0)`. The two agree
/// except where `max + 1.0` rounds up across an integer, so the loop is written out rather than
/// sharing [`for_each_block_in`].
#[must_use]
pub fn is_any_liquid(w: &World, bb: &Aabb) -> bool {
    let (x0, x1) = (floor_f64(bb.min.x), floor_f64(bb.max.x));
    let (y0, y1) = (floor_f64(bb.min.y), floor_f64(bb.max.y));
    let (z0, z1) = (floor_f64(bb.min.z), floor_f64(bb.max.z));
    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                if block_props(w.block_at(BlockPos::new(x, y, z)))
                    .material
                    .is_liquid()
                {
                    return true;
                }
            }
        }
    }
    false
}

/// java: `Entity.isOffsetPositionInLiquid` -> `isLiquidPresentInAABB`: the offset box must be
/// free of collision boxes **and** contain no liquid. This is what lets a swimming entity climb
/// out of water onto a ledge.
///
/// This form has no entity boxes and no world border; the swim/lava ledge exit in
/// [`crate::tick_player`] uses [`is_offset_position_in_liquid_with`].
#[must_use]
pub fn is_offset_position_in_liquid(w: &World, s: &PhysicsState, offset: DVec3) -> bool {
    let mut outside_border = false;
    is_offset_position_in_liquid_with(w, s, offset, &[], None, &mut outside_border)
}

/// [`is_offset_position_in_liquid`] through the same collision query as the movement sweep.
///
/// java: `isLiquidPresentInAABB` asks `World.getCollidingBoundingBoxes(this, bb)`, the query
/// `moveEntity` itself uses, so boat boxes and the world border's stone count as "not free",
/// and the call updates the entity's `outsideBorder` latch from its (post-move) position.
pub fn is_offset_position_in_liquid_with(
    w: &World,
    s: &PhysicsState,
    offset: DVec3,
    entity_collisions: &[EntityCollisionCandidate],
    border: Option<CollisionBorder>,
    outside_border: &mut bool,
) -> bool {
    let bb = s.aabb().offset(offset.x, offset.y, offset.z);
    let mut boxes = Vec::new();
    match border {
        Some(border) => crate::collision::collision_boxes_in_with_border_and_entities(
            w,
            &bb,
            border,
            s.pos.x,
            s.pos.z,
            outside_border,
            entity_collisions,
            &mut boxes,
        ),
        None => crate::collision::collision_boxes_in_with_entities(
            w,
            &bb,
            entity_collisions,
            &mut boxes,
        ),
    }
    boxes.is_empty() && !is_any_liquid(w, &bb)
}

/// java: `World.handleMaterialAcceleration` (`World.java:2077`).
///
/// Returns whether the box touched the material, and pushes the entity along the summed flow
/// vector at `0.014` blocks per tick per unit. The height test is
/// `(double) l >= (double)((float)(y + 1) - getLiquidHeightPercent(level))` where `l` is the
/// *integer* upper scan bound — an oddity of vanilla's that makes the test far laxer than it
/// looks, and one that has to be reproduced rather than "fixed".
pub fn material_acceleration(
    w: &World,
    bb: &Aabb,
    material: Material,
    s: &mut PhysicsState,
) -> bool {
    material_acceleration_vel(w, bb, material, s.flying, &mut s.vel)
}

/// [`material_acceleration`] on a bare motion vector: the only [`PhysicsState`] fields it reads
/// are `flying` and `vel`. Non-player entities (items twice per tick, XP orbs, squid probes) used
/// to build a whole throw-away `PhysicsState::new_player` (whose `flying` is `false`) per call
/// just to carry `vel` (PHYS-2).
pub fn material_acceleration_vel(
    w: &World,
    bb: &Aabb,
    material: Material,
    flying: bool,
    vel: &mut DVec3,
) -> bool {
    let x0 = floor_f64(bb.min.x);
    let x1 = floor_f64(bb.max.x + 1.0);
    let y0 = floor_f64(bb.min.y);
    let y1 = floor_f64(bb.max.y + 1.0);
    let z0 = floor_f64(bb.min.z);
    let z1 = floor_f64(bb.max.z + 1.0);

    // java: `World.handleMaterialAcceleration:2086` returns **false outright** — not `flag` —
    // when the region is not loaded. See [`crate::is_area_loaded`] for what that actually tests
    // on a client.
    if !crate::is_area_loaded(w, BlockPos::new(x0, y0, z0), BlockPos::new(x1, y1, z1)) {
        return false;
    }

    let mut flag = false;
    let mut vec = DVec3::ZERO;

    for x in x0..x1 {
        for y in y0..y1 {
            for z in z0..z1 {
                let pos = BlockPos::new(x, y, z);
                let state = w.block_at(pos);
                let props = block_props(state);
                if props.material != material {
                    continue;
                }
                // java: the f32 subtraction happens before the widening.
                let d0 = f64::from((y + 1) as f32 - liquid_height_percent(state.meta()));
                if f64::from(y1) >= d0 {
                    flag = true;
                    vec += flow_vector(w, pos, material);
                }
            }
        }
    }

    // java: `EntityPlayer.isPushedByWater()` (`:2299`) is `!this.capabilities.isFlying`, so a
    // creative player hovering in a river is not dragged by the current.
    if mc_math::vec::length(vec) > 0.0 && !flying {
        let vec = mc_math::vec::normalize(vec);
        let d1 = 0.014;
        *vel += vec * d1;
    }

    flag
}

/// java: `BlockLiquid.getFlowVector` (`BlockLiquid.java:150`).
///
/// The four horizontal neighbours contribute `level_difference * direction`; a non-liquid,
/// non-blocking neighbour instead contributes the level of the block *below* it, which is what
/// makes water accelerate toward a drop. A falling column (`level >= 8`) next to a solid face
/// gets a `-6` downward term. The result is normalised with vanilla's `1e-4` cutoff.
fn flow_vector(w: &World, pos: BlockPos, material: Material) -> DVec3 {
    let mut vec = DVec3::ZERO;
    let i = effective_flow_decay(w, pos, material);

    // java: `EnumFacing.Plane.HORIZONTAL.facings()` is NORTH, EAST, SOUTH, WEST. This
    // calculation visits the same four neighbours in its own order; its accumulated result does
    // not depend on the plane iterator's ordering.
    for neighbour in [pos.north(), pos.south(), pos.west(), pos.east()] {
        let mut j = effective_flow_decay(w, neighbour, material);

        if j < 0 {
            if !block_props(w.block_at(neighbour))
                .material
                .blocks_movement()
            {
                j = effective_flow_decay(w, neighbour.down(), material);
                if j >= 0 {
                    let k = j - (i - 8);
                    vec += vec3(
                        f64::from((neighbour.x - pos.x) * k),
                        f64::from((neighbour.y - pos.y) * k),
                        f64::from((neighbour.z - pos.z) * k),
                    );
                }
            }
        } else {
            let l = j - i;
            vec += vec3(
                f64::from((neighbour.x - pos.x) * l),
                f64::from((neighbour.y - pos.y) * l),
                f64::from((neighbour.z - pos.z) * l),
            );
        }
    }

    if i32::from(w.block_at(pos).meta()) >= 8 {
        for (neighbour, facing) in [
            (pos.north(), mc_math::Facing::North),
            (pos.south(), mc_math::Facing::South),
            (pos.west(), mc_math::Facing::West),
            (pos.east(), mc_math::Facing::East),
        ] {
            if is_block_solid(w, neighbour, facing, material)
                || is_block_solid(w, neighbour.up(), facing, material)
            {
                vec = mc_math::vec::normalize(vec) + vec3(0.0, -6.0, 0.0);
                break;
            }
        }
    }

    mc_math::vec::normalize(vec)
}

/// java: `BlockLiquid.getEffectiveFlowDecay` — the level, or `-1` when the block is not the
/// same liquid, with falling levels folded to `0`.
fn effective_flow_decay(w: &World, pos: BlockPos, material: Material) -> i32 {
    let state = w.block_at(pos);
    if block_props(state).material != material {
        return -1;
    }
    let level = i32::from(state.meta());
    if level >= 8 {
        0
    } else {
        level
    }
}

/// java: `BlockLiquid.isBlockSolid` — same liquid is never solid; from above everything else
/// is; ice is not; otherwise `Block.isBlockSolid`, which is `material.isSolid()`.
fn is_block_solid(w: &World, pos: BlockPos, side: mc_math::Facing, material: Material) -> bool {
    let m = block_props(w.block_at(pos)).material;
    if m == material {
        false
    } else if side == mc_math::Facing::Up {
        true
    } else if m == Material::Ice {
        false
    } else {
        m.is_solid()
    }
}

/// java: the shared `floor(min) .. floor(max + 1.0)` scan every `World` material query uses.
fn for_each_block_in(bb: &Aabb, mut pred: impl FnMut(BlockPos) -> bool) -> bool {
    let x0 = floor_f64(bb.min.x);
    let x1 = floor_f64(bb.max.x + 1.0);
    let y0 = floor_f64(bb.min.y);
    let y1 = floor_f64(bb.max.y + 1.0);
    let z0 = floor_f64(bb.min.z);
    let z1 = floor_f64(bb.max.z + 1.0);
    for x in x0..x1 {
        for y in y0..y1 {
            for z in z0..z1 {
                if pred(BlockPos::new(x, y, z)) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{block_by_name, BlockState};

    fn pool() -> World {
        let mut w = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let water = BlockState::new(block_by_name("water").unwrap(), 0);
        for x in -4..=4 {
            for z in -4..=4 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 59, z), stone);
                for y in 60..=63 {
                    crate::scene::set_block(&mut w, BlockPos::new(x, y, z), water);
                }
            }
        }
        w
    }

    /// PHYS-2: the bare-velocity probe is the `PhysicsState` one. Flowing water at every level
    /// (so the flow vector and its normalisation run), flying on and off, random boxes
    /// straddling the pool edge and an unloaded region.
    #[test]
    fn velocity_probe_matches_the_physics_state_probe() {
        let mut w = pool();
        let flowing = block_by_name("flowing_water").unwrap();
        for (i, x) in (5..=12).enumerate() {
            for z in -4_i32..=4 {
                let state = BlockState::new(flowing, (i as u8 + z.unsigned_abs() as u8) % 16);
                crate::scene::set_block(&mut w, BlockPos::new(x, 62, z), state);
            }
        }
        let mut rng = mc_math::JavaRandom::new(0xf1_0a7);
        for case in 0..3_000 {
            let min = DVec3::new(
                rng.next_f64() * 24.0 - 8.0,
                58.0 + rng.next_f64() * 7.0,
                rng.next_f64() * 14.0 - 7.0,
            );
            let bb = Aabb::from_bounds(
                min.x,
                min.y,
                min.z,
                min.x + 0.1 + rng.next_f64() * 1.5,
                min.y + 0.1 + rng.next_f64() * 2.0,
                min.z + 0.1 + rng.next_f64() * 1.5,
            );
            let motion = DVec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64());
            let flying = rng.next_bool();
            let mut s = PhysicsState::new_player(DVec3::ZERO);
            s.vel = motion;
            s.flying = flying;
            let want = material_acceleration(&w, &bb, Material::Water, &mut s);
            let mut vel = motion;
            let got = material_acceleration_vel(&w, &bb, Material::Water, flying, &mut vel);
            assert_eq!((got, vel), (want, s.vel), "case {case}: {bb:?}");
        }
        assert!(!PhysicsState::new_player(DVec3::ZERO).flying);
    }

    #[test]
    fn liquid_heights_match_the_java_table() {
        assert_eq!(liquid_height_percent(0), 1.0 / 9.0);
        assert_eq!(liquid_height_percent(7), 8.0 / 9.0);
        assert_eq!(
            liquid_height_percent(8),
            1.0 / 9.0,
            "falling folds to source"
        );
        assert_eq!(liquid_height_percent(15), 1.0 / 9.0);
    }

    #[test]
    fn a_submerged_player_is_in_water_and_stops_falling() {
        let w = pool();
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 61.0, 0.5));
        s.fall_distance = 12.0;
        assert!(handle_water_movement(&w, &mut s));
        assert!(s.in_water);
        assert_eq!(s.fall_distance, 0.0);
    }

    #[test]
    fn a_player_above_the_pool_is_not_in_water() {
        let w = pool();
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 66.0, 0.5));
        assert!(!handle_water_movement(&w, &mut s));
    }

    #[test]
    fn a_uniform_pool_has_no_current() {
        let w = pool();
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 61.0, 0.5));
        handle_water_movement(&w, &mut s);
        assert_eq!(s.vel, DVec3::ZERO, "symmetric source blocks cancel out");
    }

    #[test]
    fn flowing_water_pushes_downhill() {
        let mut w = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let flowing = block_by_name("flowing_water").unwrap();
        for x in -2..=6 {
            crate::scene::set_block(&mut w, BlockPos::new(x, 59, 0), stone);
        }
        // A descending ramp of levels 0..=3 running east: the current flows toward the
        // higher-level (shallower) end.
        for (x, level) in (0..=3).enumerate() {
            crate::scene::set_block(
                &mut w,
                BlockPos::new(x as i32, 60, 0),
                BlockState::new(flowing, level as u8),
            );
        }
        let mut s = PhysicsState::new_player(DVec3::new(1.5, 60.0, 0.5));
        handle_water_movement(&w, &mut s);
        assert!(
            s.vel.x > 0.0,
            "pushed toward the shallow end, got {:?}",
            s.vel
        );
        assert!(s.vel.x <= 0.014 + 1e-12);
    }

    #[test]
    fn lava_is_detected_by_its_own_shrunken_box() {
        let mut w = World::new();
        let lava = BlockState::new(block_by_name("lava").unwrap(), 0);
        for x in -2..=2 {
            for z in -2..=2 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 60, z), lava);
            }
        }
        let s = PhysicsState::new_player(DVec3::new(0.5, 60.0, 0.5));
        assert!(is_in_lava(&w, &s));
        let s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        assert!(!is_in_lava(&w, &s));
    }

    #[test]
    fn offset_position_in_liquid_needs_free_space_and_no_liquid() {
        let w = pool();
        let s = PhysicsState::new_player(DVec3::new(0.5, 61.0, 0.5));
        assert!(!is_offset_position_in_liquid(
            &w,
            &s,
            DVec3::new(0.0, 0.0, 0.0)
        ));
        // Well above the pool: no blocks, no liquid.
        assert!(is_offset_position_in_liquid(
            &w,
            &s,
            DVec3::new(0.0, 20.0, 0.0)
        ));
    }

    /// java: `World.isAnyLiquid` scans `floor(min)..=floor(max)`. Where `max + 1.0` rounds up
    /// across an integer the `floor(max + 1.0)` form reads one block too many.
    #[test]
    fn any_liquid_scans_the_inclusive_floor_range() {
        let mut w = World::new();
        let water = BlockState::new(block_by_name("water").unwrap(), 0);
        crate::scene::set_block(&mut w, BlockPos::new(1, 64, 0), water);
        // The largest double below 1.0: floor(max) = 0, but max + 1.0 rounds to exactly 2.0.
        let max_x = 1.0 - f64::EPSILON / 2.0;
        assert_eq!(max_x.floor(), 0.0);
        assert_eq!(max_x + 1.0, 2.0);
        let bb = Aabb::from_bounds(0.2, 64.2, 0.2, max_x, 64.8, 0.8);
        assert!(
            !is_any_liquid(&w, &bb),
            "column x = 1 is outside Java's scan"
        );
        // Negative control: the exclusive `floor(max + 1.0)` form does reach x = 1.
        assert!(for_each_block_in(&bb, |p| {
            block_props(w.block_at(p)).material.is_liquid()
        }));
        // Inside the box proper, the liquid is found by both.
        let reaching = Aabb::from_bounds(0.2, 64.2, 0.2, 1.2, 64.8, 0.8);
        assert!(is_any_liquid(&w, &reaching));
    }

    /// java: `isLiquidPresentInAABB` asks `getCollidingBoundingBoxes(this, bb)`, so the world
    /// border's stone keeps a swimmer pressed against the border from getting the `0.3` kick.
    #[test]
    fn swim_exit_test_sees_the_world_border_like_the_sweep() {
        let w = pool();
        // A border whose -X wall is at x = 0.0, with the swimmer just inside it.
        let border = CollisionBorder::new(10.0, 0.0, 20.0, 29_999_984);
        let s = PhysicsState::new_player(DVec3::new(0.35, 63.0, 0.5));
        let up_and_out = DVec3::new(-0.1, 1.0, 0.0);
        let mut latch = false;
        assert!(
            !is_offset_position_in_liquid_with(&w, &s, up_and_out, &[], Some(border), &mut latch),
            "the border's stone column at x = -1 occupies the offset box"
        );
        assert!(!latch);
        // Negative control: without the border (the pre-wave call) the same probe is free.
        assert!(is_offset_position_in_liquid(&w, &s, up_and_out));
        // A boat beside the swimmer blocks it too.
        let boat = Aabb::from_bounds(-0.5, 63.5, 0.0, 0.5, 64.5, 1.0);
        let candidate = EntityCollisionCandidate::new(boat, Some(boat));
        let mut latch = false;
        assert!(!is_offset_position_in_liquid_with(
            &w,
            &s,
            up_and_out,
            &[candidate],
            None,
            &mut latch
        ));
    }

    #[test]
    fn water_entry_recipe_is_an_air_to_water_edge_with_java_volume_and_count() {
        let w = pool();
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 61.25, 0.5));
        s.first_update = false;
        s.vel = DVec3::new(3.0, -4.0, 0.0);

        assert!(handle_water_movement(&w, &mut s));
        let effect = s.water_entry.expect("air-to-water edge must reset height");
        // sqrt(3^2 * 0.2 + (-4)^2) * 0.2, where sqrt_double returns f32 first.
        assert_eq!(effect.sound_volume.to_bits(), 0.843_800_96_f32.to_bits());
        assert_eq!(effect.particle_count, 13, "1 + 0.6F * 20.0F");
        assert_eq!(effect.particle_origin, DVec3::new(0.5, 62.0, 0.5));
        assert_eq!(effect.bubble_velocity, s.vel);
        assert_eq!(effect.splash_velocity, s.vel);
        assert_eq!(effect.particle_width.to_bits(), 0.6_f32.to_bits());
        assert_eq!(effect.bubble_y_random_subtract.to_bits(), 0.2_f32.to_bits());
        assert_eq!(effect.sound_pitch_base.to_bits(), 1.0_f32.to_bits());
        assert_eq!(effect.sound_pitch_jitter.to_bits(), 0.4_f32.to_bits());
        assert_eq!(WaterEntry::SOUND_NAME, "game.neutral.swim.splash");
    }

    #[test]
    fn water_entry_fires_once_per_entry_not_stay_or_initial_spawn() {
        let w = pool();
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 61.0, 0.5));

        assert!(handle_water_movement(&w, &mut s));
        assert!(
            s.water_entry.is_none(),
            "firstUpdate suppresses resetHeight"
        );

        s.first_update = false;
        s.water_entry = None;
        assert!(handle_water_movement(&w, &mut s));
        assert!(
            s.water_entry.is_none(),
            "remaining in water is not a new edge"
        );

        s.set_position(DVec3::new(0.5, 66.0, 0.5));
        assert!(!handle_water_movement(&w, &mut s));
        assert!(!s.in_water);

        s.set_position(DVec3::new(0.5, 61.0, 0.5));
        assert!(handle_water_movement(&w, &mut s));
        assert!(
            s.water_entry.is_some(),
            "exit then re-entry must reset height once"
        );
    }
}
