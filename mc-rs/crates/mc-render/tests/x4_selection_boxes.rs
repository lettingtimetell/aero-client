//! Item 17 ("torch hitbox too large"): the block outline is
//! vanilla's selection box for every small block.
//!
//! `RenderGlobal.drawSelectionBox` outlines `Block.getSelectedBoundingBox`, which reads the
//! block's bound fields as `getMouseOver`'s raytrace left them: `Block.collisionRayTrace` runs
//! `setBlockBoundsBasedOnState` first, and `BlockTorch.collisionRayTrace` writes the torch box
//! itself. The extracted registry row for torches is the constructor's full cube, so mc-rs drew a
//! whole-block outline around every torch while the pick (special-cased) was already right.
//!
//! The expected boxes below are copied from the 1.8.9 sources (the `setBlockBounds` arguments
//! in each block's `setBlockBoundsBasedOnState` / `collisionRayTrace` / constructor), not from
//! mc-rs: `BlockTorch:184-211`, `BlockLever:161-194`, `BlockButton.updateBlockBounds`,
//! `BlockLadder`, `BlockFlowerPot.setBlockBoundsForItemRender`, `BlockSkull`, `BlockSign`,
//! `BlockWallSign`, `BlockBanner`, `BlockTripWireHook`, `BlockTripWire`, `BlockCarpet`,
//! `BlockSnow`, `BlockCake`, `BlockRedstoneWire`, `BlockRedstoneDiode`,
//! `BlockBasePressurePlate`, `BlockDaylightDetector`, `BlockCocoa`, `BlockVine`.

use mc_data::BlockState;
use mc_math::{BlockPos, DVec3};
use mc_render::terrain::{fixture, selected_bounding_box, SelectionBox};

fn state(name: &str, meta: u8) -> BlockState {
    let id = mc_data::block_by_name(name).unwrap_or_else(|| panic!("{name} is registered"));
    BlockState::new(id, meta)
}

fn close(a: DVec3, b: DVec3) -> bool {
    (a - b).abs().max_element() < 1e-6
}

/// `(block, meta, min, max)` in block-local coordinates, from the Java sources.
#[allow(clippy::type_complexity)]
const VANILLA: &[(&str, u8, [f64; 3], [f64; 3])] = &[
    // BlockTorch.collisionRayTrace: f = 0.15 on walls, 0.1 on the floor.
    ("torch", 5, [0.4, 0.0, 0.4], [0.6, 0.6, 0.6]),
    ("torch", 1, [0.0, 0.2, 0.35], [0.3, 0.8, 0.65]),
    ("torch", 2, [0.7, 0.2, 0.35], [1.0, 0.8, 0.65]),
    ("torch", 3, [0.35, 0.2, 0.0], [0.65, 0.8, 0.3]),
    ("torch", 4, [0.35, 0.2, 0.7], [0.65, 0.8, 1.0]),
    ("redstone_torch", 5, [0.4, 0.0, 0.4], [0.6, 0.6, 0.6]),
    ("redstone_torch", 3, [0.35, 0.2, 0.0], [0.65, 0.8, 0.3]),
    (
        "unlit_redstone_torch",
        1,
        [0.0, 0.2, 0.35],
        [0.3, 0.8, 0.65],
    ),
    // BlockLever: f = 0.1875 on walls, 0.25 up/down.
    ("lever", 1, [0.0, 0.2, 0.3125], [0.375, 0.8, 0.6875]),
    ("lever", 4, [0.3125, 0.2, 0.625], [0.6875, 0.8, 1.0]),
    ("lever", 5, [0.25, 0.0, 0.25], [0.75, 0.6, 0.75]),
    ("lever", 0, [0.25, 0.4, 0.25], [0.75, 1.0, 0.75]),
    // BlockButton.updateBlockBounds: depth 2/16, 1/16 pressed.
    (
        "stone_button",
        5,
        [0.3125, 0.0, 0.375],
        [0.6875, 0.125, 0.625],
    ),
    (
        "stone_button",
        1,
        [0.0, 0.375, 0.3125],
        [0.125, 0.625, 0.6875],
    ),
    (
        "wooden_button",
        12,
        [0.3125, 0.375, 0.9375],
        [0.6875, 0.625, 1.0],
    ),
    // BlockLadder: 0.125 thick against its wall.
    ("ladder", 2, [0.0, 0.0, 0.875], [1.0, 1.0, 1.0]),
    ("ladder", 5, [0.0, 0.0, 0.0], [0.125, 1.0, 1.0]),
    // BlockFlowerPot.setBlockBoundsForItemRender: f = 0.375, f1 = f / 2.
    (
        "flower_pot",
        0,
        [0.3125, 0.0, 0.3125],
        [0.6875, 0.375, 0.6875],
    ),
    // BlockSkull: floor, and a north-facing wall skull.
    ("skull", 1, [0.25, 0.0, 0.25], [0.75, 0.5, 0.75]),
    ("skull", 2, [0.25, 0.25, 0.5], [0.75, 0.75, 1.0]),
    // BlockSign / BlockWallSign.
    ("standing_sign", 0, [0.25, 0.0, 0.25], [0.75, 1.0, 0.75]),
    ("wall_sign", 2, [0.0, 0.28125, 0.875], [1.0, 0.78125, 1.0]),
    ("wall_sign", 5, [0.0, 0.28125, 0.0], [0.125, 0.78125, 1.0]),
    // BlockBanner standing / hanging.
    ("standing_banner", 0, [0.25, 0.0, 0.25], [0.75, 1.0, 0.75]),
    ("wall_banner", 2, [0.0, 0.0, 0.875], [1.0, 0.78125, 1.0]),
    // BlockTripWireHook (meta & 3: 0 south) and BlockTripWire (unsuspended).
    ("tripwire_hook", 0, [0.3125, 0.2, 0.0], [0.6875, 0.8, 0.375]),
    ("tripwire", 0, [0.0, 0.0, 0.0], [1.0, 0.09375, 1.0]),
    // Thin floor blocks.
    ("carpet", 4, [0.0, 0.0, 0.0], [1.0, 0.0625, 1.0]),
    ("snow_layer", 0, [0.0, 0.0, 0.0], [1.0, 0.125, 1.0]),
    ("snow_layer", 3, [0.0, 0.0, 0.0], [1.0, 0.5, 1.0]),
    ("redstone_wire", 0, [0.0, 0.0, 0.0], [1.0, 0.0625, 1.0]),
    ("unpowered_repeater", 0, [0.0, 0.0, 0.0], [1.0, 0.125, 1.0]),
    (
        "stone_pressure_plate",
        0,
        [0.0625, 0.0, 0.0625],
        [0.9375, 0.0625, 0.9375],
    ),
    ("daylight_detector", 0, [0.0, 0.0, 0.0], [1.0, 0.375, 1.0]),
    // BlockCake: x starts at (1 + bites * 2) / 16.
    ("cake", 0, [0.0625, 0.0, 0.0625], [0.9375, 0.5, 0.9375]),
    ("cake", 3, [0.4375, 0.0, 0.0625], [0.9375, 0.5, 0.9375]),
    // BlockCocoa age 0 (j = 4, k = 5, f = 2) facing south (meta 0).
    ("cocoa", 0, [0.375, 0.4375, 0.6875], [0.625, 0.75, 0.9375]),
    // BlockVine on the south face only.
    ("vine", 1, [0.0, 0.0, 0.9375], [1.0, 1.0, 1.0]),
];

/// Every listed small block outlines exactly vanilla's selection box, torches included.
#[test]
fn small_block_outlines_are_vanillas_selection_boxes() {
    let mut world = fixture::blank_world(1);
    let pos = BlockPos::new(3, 70, -5);
    for &(name, meta, min, max) in VANILLA {
        let block = state(name, meta);
        world.set_block(pos, block);
        let got = selected_bounding_box(&world, pos, world.block_at(pos));
        let origin = pos.to_vec3();
        let (got_min, got_max) = (got.min - origin, got.max - origin);
        assert!(
            close(got_min, DVec3::from_array(min)) && close(got_max, DVec3::from_array(max)),
            "{name}:{meta}: outline {got_min:?}..{got_max:?}, vanilla {min:?}..{max:?}"
        );
    }
}

/// The drawn outline is the torch box grown by vanilla's 0.002, not a whole block.
#[test]
fn the_torch_outline_is_the_torch_box_not_the_cell() {
    let mut world = fixture::blank_world(1);
    let pos = BlockPos::new(0, 64, 0);
    world.set_block(pos, state("torch", 5));
    let outline = SelectionBox::for_block_in_world(pos, world.block_at(pos), &world);
    let size = outline.max - outline.min;
    assert!(
        (size.x - 0.204).abs() < 1e-5 && (size.y - 0.604).abs() < 1e-5,
        "floor torch outline {size:?}"
    );
    assert!(outline.min.x > 0.39 && outline.max.x < 0.61);

    // Control: a full block is still the whole cell.
    world.set_block(pos, state("stone", 0));
    let cube = SelectionBox::for_block_in_world(pos, world.block_at(pos), &world);
    assert!(((cube.max - cube.min).x - 1.004).abs() < 1e-5);
}
