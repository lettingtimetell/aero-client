//! `Entity.isEntityInsideOpaqueBlock`'s eight eye-corner probes.

use mc_data::block_props;
use mc_math::{floor_f64, BlockPos, DVec3};
use mc_world::World;

/// Tests whether an entity's eye-width probe enters any visually opaque block.
///
/// java: `Entity.isEntityInsideOpaqueBlock`.  `pos` is the entity location and `eye_height` is
/// `getEyeHeight()`, deliberately separate because Java adds the small Y probe before it adds
/// eye height.  The Y probe is a fixed `0.1F`, while X/Z use the f32 expression
/// `(+/- 0.5F) * width * 0.8F`; conflating those two widths moves several sampled cells.
#[must_use]
pub fn is_entity_inside_opaque_block(
    w: &World,
    pos: DVec3,
    eye_height: f32,
    width: f32,
    no_clip: bool,
) -> bool {
    if no_clip {
        return false;
    }

    for i in 0..8 {
        // Keep Java's f32 multiply order and axis-bit order: bit 0 is Y, bit 1 is X, bit 2 is Z.
        let y_probe = ((i & 1) as f32 - 0.5) * 0.1;
        let x_probe = (((i >> 1) & 1) as f32 - 0.5) * width * 0.8;
        let z_probe = (((i >> 2) & 1) as f32 - 0.5) * width * 0.8;
        let cell = BlockPos::new(
            floor_f64(pos.x + f64::from(x_probe)),
            floor_f64(pos.y + f64::from(y_probe) + f64::from(eye_height)),
            floor_f64(pos.z + f64::from(z_probe)),
        );
        if block_props(w.block_at(cell)).visually_opaque {
            return true;
        }
    }
    false
}

/// Select the opaque block texture used by `ItemRenderer.renderOverlays` after the predicate
/// above succeeds.
///
/// Vanilla repeats the eight probes with X on bit 0 and Y on bit 1 (the predicate uses those two
/// bits in the opposite order), and overwrites the candidate for every opaque corner. The sets
/// of positions are identical; the different enumeration order makes the final texture
/// observable when two block types meet at the player's eye.
#[must_use]
pub fn opaque_overlay_block_at_eye(
    w: &World,
    pos: DVec3,
    eye_height: f32,
    width: f32,
    no_clip: bool,
) -> Option<mc_data::BlockState> {
    if no_clip || !is_entity_inside_opaque_block(w, pos, eye_height, width, false) {
        return None;
    }

    let mut selected = None;
    for i in 0..8 {
        let x_probe = ((i & 1) as f32 - 0.5) * width * 0.8;
        let y_probe = (((i >> 1) & 1) as f32 - 0.5) * 0.1;
        let z_probe = (((i >> 2) & 1) as f32 - 0.5) * width * 0.8;
        let cell = BlockPos::new(
            floor_f64(pos.x + f64::from(x_probe)),
            floor_f64(pos.y + f64::from(y_probe) + f64::from(eye_height)),
            floor_f64(pos.z + f64::from(z_probe)),
        );
        let state = w.block_at(cell);
        if block_props(state).visually_opaque {
            selected = Some(state);
        }
    }
    selected
}

/// [`is_entity_inside_opaque_block`] for the local-player physics state.
///
/// This is intentionally a query only. Camera/vignette consumption remains the client layer's
/// responsibility, so adding this does not make an unintegrated visual prediction look live.
#[must_use]
pub fn physics_state_inside_opaque_block(w: &World, s: &crate::PhysicsState) -> bool {
    let mut eye_height = crate::state::PLAYER_EYE_HEIGHT;
    if s.sneaking {
        eye_height -= 0.08;
    }
    is_entity_inside_opaque_block(w, s.pos, eye_height, s.width, s.no_clip)
}

/// [`opaque_overlay_block_at_eye`] for the local player's current physics state.
#[must_use]
pub fn physics_state_opaque_overlay_block(
    w: &World,
    s: &crate::PhysicsState,
) -> Option<mc_data::BlockState> {
    let mut eye_height = crate::state::PLAYER_EYE_HEIGHT;
    if s.sneaking {
        eye_height -= 0.08;
    }
    opaque_overlay_block_at_eye(w, s.pos, eye_height, s.width, s.no_clip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{block_by_name, BlockState};

    fn stone() -> BlockState {
        BlockState::new(block_by_name("stone").unwrap(), 0)
    }

    #[test]
    fn one_opaque_eye_corner_is_enough_but_an_open_corner_is_not() {
        let pos = DVec3::new(0.5, 64.0, 0.5);
        let mut w = World::new();
        // With width 0.6, the first (-Y,-X,-Z) probe samples (0, 65, 0).
        crate::scene::set_block(&mut w, BlockPos::new(0, 65, 0), stone());
        assert!(is_entity_inside_opaque_block(&w, pos, 1.62, 0.6, false));

        let open = World::new();
        assert!(!is_entity_inside_opaque_block(&open, pos, 1.62, 0.6, false));
    }

    #[test]
    fn negative_coordinates_floor_the_base_and_the_corner_offset() {
        let mut w = World::new();
        // The negative X/Z corners are applied before flooring, so -0.99 - 0.24 reaches -2.
        crate::scene::set_block(&mut w, BlockPos::new(-2, 65, -2), stone());
        assert!(is_entity_inside_opaque_block(
            &w,
            DVec3::new(-0.99, 64.0, -0.99),
            1.62,
            0.6,
            false
        ));
    }

    #[test]
    fn no_clip_skips_all_eight_corners() {
        let mut w = World::new();
        crate::scene::set_block(&mut w, BlockPos::new(0, 65, 0), stone());
        assert!(!is_entity_inside_opaque_block(
            &w,
            DVec3::new(0.5, 64.0, 0.5),
            1.62,
            0.6,
            true
        ));
    }

    #[test]
    fn y_probe_is_fixed_and_x_uses_bit_one_not_bit_zero() {
        let mut w = World::new();
        // This is sampled only by i=2: X positive, Y negative, Z negative.
        crate::scene::set_block(&mut w, BlockPos::new(1, 65, 0), stone());
        assert!(is_entity_inside_opaque_block(
            &w,
            DVec3::new(0.99, 64.0, 0.5),
            1.62,
            0.6,
            false
        ));
    }

    #[test]
    fn overlay_texture_uses_item_renderers_last_opaque_corner() {
        let mut w = World::new();
        let dirt = BlockState::new(block_by_name("dirt").unwrap(), 0);
        crate::scene::set_block(&mut w, BlockPos::new(0, 65, 0), stone());
        crate::scene::set_block(&mut w, BlockPos::new(1, 65, 1), dirt);

        assert_eq!(
            opaque_overlay_block_at_eye(&w, DVec3::new(0.5, 64.0, 0.5), 1.62, 1.25, false,),
            Some(dirt),
            "i=7's positive-X/Z corner is ItemRenderer's final texture candidate"
        );
    }
}
