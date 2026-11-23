//! Hazard **H8**: vanilla's exact ambient-occlusion formula.
//!
//! These tests need no GPU and no jar. They build a synthetic world, hand a hand-built
//! `BakedQuad` straight to [`update_vertex_brightness`], and compare the result against
//! numbers derived by hand from `BlockModelRenderer.AmbientOcclusionFace`. Every expected
//! value below is written out as the arithmetic vanilla performs, so a future reader can
//! re-derive it rather than trust it.
//!
//! The critical case is `both_side_blocks_opaque_skips_the_corner_sample`: that is the
//! "distinct rule when both side blocks are opaque" the plan's hazard note calls out, and it
//! is the one an approximation gets wrong in a way screenshots do not reveal.

use mc_assets::model::{BakedQuad, Vertex};
use mc_math::{BlockPos, Facing};
use mc_render::terrain::ao::{fill_quad_bounds, update_vertex_brightness};
use mc_render::terrain::fixture;
use mc_render::terrain::MeshSettings;
use mc_world::World;

/// `Block.getAmbientOcclusionLightValue()` for a normal cube. java: `Block.java:1099`.
const OPAQUE_AO: f32 = 0.2;
/// ... and for anything light passes through.
const CLEAR_AO: f32 = 1.0;

/// A full-cube top face, the way `FaceBakery` bakes one: `y == 1` on all four corners.
fn up_face_quad() -> BakedQuad {
    let v = |x: f32, z: f32| Vertex {
        pos: [x, 1.0, z],
        uv: [0.0, 0.0],
    };
    BakedQuad {
        verts: [v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0), v(1.0, 0.0)],
        face: Some(Facing::Up),
        shade_face: Some(Facing::Up),
        tint_index: None,
        shade: true,
    }
}

/// A quarter-size top face — the shape a partial block (slab top, cake, cocoa) bakes.
fn partial_up_face_quad() -> BakedQuad {
    let v = |x: f32, z: f32| Vertex {
        pos: [x, 1.0, z],
        uv: [0.0, 0.0],
    };
    BakedQuad {
        verts: [v(0.25, 0.25), v(0.25, 0.75), v(0.75, 0.75), v(0.75, 0.25)],
        face: Some(Facing::Up),
        shade_face: Some(Facing::Up),
        tint_index: None,
        shade: true,
    }
}

fn floor_world() -> World {
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    w
}

fn ao_at(w: &World, pos: BlockPos, quad: &BakedQuad) -> [f32; 4] {
    let state = w.block_at(pos);
    let full_cube = mc_data::block_props(state).full_cube;
    let qb = fill_quad_bounds(quad, Facing::Up, full_cube);
    update_vertex_brightness(
        w,
        state,
        pos,
        Facing::Up,
        &qb,
        Some(&quad.verts),
        MeshSettings::default(),
    )
    .color_multiplier
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn an_unobstructed_top_face_is_fully_lit_at_every_corner() {
    let w = floor_world();
    let ao = ao_at(&w, BlockPos::new(8, 16, 8), &up_face_quad());
    assert!(
        ao.iter().all(|v| close(*v, 1.0)),
        "flat ground must be unoccluded, got {ao:?}"
    );
}

#[test]
fn a_wall_on_one_side_darkens_exactly_two_corners() {
    // Floor + a two-block-tall wall along x = 9. The top face at (8, 16, 8) has its two
    // east-side corners occluded.
    //
    // java: for facing UP, EnumNeighborInfo.UP's neighbours are [EAST, WEST, NORTH, SOUTH] and
    // the quad is flush, so every sample is taken from y = 17.
    //   ao(east)  = 0.2   ao(west) = ao(north) = ao(south) = 1.0   ao(centre) = 1.0
    //   corner(east,north) and corner(east,south) are stone too (the wall runs along z),
    //   because `!flag2 && !flag` is false when the east neighbour blocks light.
    //   c0 = (south + east + corner(east,south) + centre) * 0.25 = (1.0 + 0.2 + 0.2 + 1.0)/4
    //   c1 = (north + east + corner(east,north) + centre) * 0.25 = same
    //   c2 = (north + west + corner(west,north) + centre) * 0.25 = (1.0 + 1.0 + 1.0 + 1.0)/4
    //   c3 = (south + west + corner(west,south) + centre) * 0.25 = same
    // VertexTranslations.UP is (2, 3, 0, 1), so c0 -> vertex 2 and c1 -> vertex 3.
    let mut w = floor_world();
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(9, 17, 0),
        BlockPos::new(9, 18, 15),
        stone,
    );

    let expected_dark = (CLEAR_AO + OPAQUE_AO + OPAQUE_AO + CLEAR_AO) * 0.25; // 0.6
    let ao = ao_at(&w, BlockPos::new(8, 16, 8), &up_face_quad());
    assert!(close(ao[2], expected_dark), "vertex 2 = {} ", ao[2]);
    assert!(close(ao[3], expected_dark), "vertex 3 = {}", ao[3]);
    assert!(close(ao[0], 1.0), "vertex 0 = {}", ao[0]);
    assert!(close(ao[1], 1.0), "vertex 1 = {}", ao[1]);

    // ... and the dark corners are the two that physically touch the wall (x == 1.0 locally).
    let quad = up_face_quad();
    for (i, v) in quad.verts.iter().enumerate() {
        let touches_wall = v.pos[0] > 0.5;
        assert_eq!(
            touches_wall,
            ao[i] < 0.99,
            "vertex {i} at {:?} has AO {}",
            v.pos,
            ao[i]
        );
    }
}

#[test]
fn both_side_blocks_opaque_skips_the_corner_sample() {
    // HAZARD H8. Two walls meeting at a corner, with the diagonal cell itself left EMPTY.
    //
    //   plan view at y = 17, looking down; O = stone, . = air, X = the block being lit
    //        z-1:  .  O
    //        z  :  X  O
    //             x  x+1
    //
    // Vanilla's `if (!flag2 && !flag)` asks whether light gets *past* each side block, by
    // testing the cell one step along the face normal (here, y = 18). With the walls two tall
    // both tests fail, so the corner is never sampled and the side value (0.2) is reused:
    //
    //   c1 = (ao(north) + ao(east) + REUSED ao(east) + ao(centre)) * 0.25
    //      = (0.2 + 0.2 + 0.2 + 1.0) * 0.25 = 0.4
    //
    // A naive "always sample the diagonal" implementation reads the empty corner instead and
    // gets (0.2 + 0.2 + 1.0 + 1.0) * 0.25 = 0.6 — visibly too bright, and wrong in a way that
    // only shows up as the camera moves.
    let mut w = floor_world();
    let stone = fixture::state_of("minecraft:stone");
    let (x, z) = (8, 8);
    fixture::fill(
        &mut w,
        BlockPos::new(x + 1, 17, z),
        BlockPos::new(x + 1, 18, z),
        stone,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(x, 17, z - 1),
        BlockPos::new(x, 18, z - 1),
        stone,
    );
    // The diagonal stays air.
    assert!(w.block_at(BlockPos::new(x + 1, 17, z - 1)).props().is_air());

    let vanilla = (OPAQUE_AO + OPAQUE_AO + OPAQUE_AO + CLEAR_AO) * 0.25; // 0.4
    let naive = (OPAQUE_AO + OPAQUE_AO + CLEAR_AO + CLEAR_AO) * 0.25; // 0.6
    let ao = ao_at(&w, BlockPos::new(x, 16, z), &up_face_quad());
    // VertexTranslations.UP maps the (east, north) corner to vertex 3.
    assert!(
        close(ao[3], vanilla),
        "expected the vanilla skip-the-corner value {vanilla}, got {} (naive would be {naive})",
        ao[3]
    );
    assert!(!close(ao[3], naive));
}

#[test]
fn one_tall_walls_do_sample_the_corner() {
    // The mirror image of the test above: with the walls only ONE block tall, light does get
    // past them (the cells at y = 18 are air), so vanilla samples the diagonal after all and
    // the same corner comes out brighter.
    let mut w = floor_world();
    let stone = fixture::state_of("minecraft:stone");
    let (x, z) = (8, 8);
    w.set_block(BlockPos::new(x + 1, 17, z), stone);
    w.set_block(BlockPos::new(x, 17, z - 1), stone);

    let sampled = (OPAQUE_AO + OPAQUE_AO + CLEAR_AO + CLEAR_AO) * 0.25; // 0.6
    let ao = ao_at(&w, BlockPos::new(x, 16, z), &up_face_quad());
    assert!(
        close(ao[3], sampled),
        "one-tall walls must sample the empty diagonal: got {}",
        ao[3]
    );
}

#[test]
fn an_occupied_diagonal_darkens_the_corner_even_with_clear_sides() {
    // Only the diagonal is filled: both side tests pass (`!flag2 && !flag` is false because
    // both sides are air and therefore translucent), so the diagonal is sampled.
    //   c = (1.0 + 1.0 + 0.2 + 1.0) * 0.25 = 0.8
    let mut w = floor_world();
    let stone = fixture::state_of("minecraft:stone");
    let (x, z) = (8, 8);
    w.set_block(BlockPos::new(x + 1, 17, z - 1), stone);

    let expected = (CLEAR_AO + CLEAR_AO + OPAQUE_AO + CLEAR_AO) * 0.25; // 0.8
    let ao = ao_at(&w, BlockPos::new(x, 16, z), &up_face_quad());
    assert!(close(ao[3], expected), "got {}", ao[3]);
    assert!(close(ao[0], 1.0) && close(ao[1], 1.0) && close(ao[2], 1.0));
}

#[test]
fn a_full_enclosure_darkens_every_corner_to_the_minimum() {
    // Surrounded on all four sides and diagonals: every term is 0.2 except the centre, which
    // is the air the face points into.
    //   c = (0.2 + 0.2 + 0.2 + 1.0) * 0.25 = 0.4
    let mut w = floor_world();
    let stone = fixture::state_of("minecraft:stone");
    let (x, z) = (8, 8);
    for dx in -1..=1 {
        for dz in -1..=1 {
            if dx == 0 && dz == 0 {
                continue;
            }
            fixture::fill(
                &mut w,
                BlockPos::new(x + dx, 17, z + dz),
                BlockPos::new(x + dx, 18, z + dz),
                stone,
            );
        }
    }
    let expected = (OPAQUE_AO + OPAQUE_AO + OPAQUE_AO + CLEAR_AO) * 0.25; // 0.4
    let ao = ao_at(&w, BlockPos::new(x, 16, z), &up_face_quad());
    assert!(ao.iter().all(|v| close(*v, expected)), "{ao:?}");
}

#[test]
fn a_partial_quad_takes_the_interpolated_path_and_lands_between_the_corners() {
    // A quad that does not span its face uses the `Orientation` weight tables instead of the
    // four raw corners, so the shrunken quad's corners sit part-way between them. This only
    // applies to side faces in vanilla (`EnumNeighborInfo.field_178289_i`), so an UP face
    // with `partial` set still takes the flat path — which is exactly what this asserts.
    let quad = partial_up_face_quad();
    let qb = fill_quad_bounds(&quad, Facing::Up, true);
    assert!(
        qb.partial,
        "a quarter-size top face must be flagged partial"
    );

    let mut w = floor_world();
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(9, 17, 0),
        BlockPos::new(9, 18, 15),
        stone,
    );
    let full = ao_at(&w, BlockPos::new(8, 16, 8), &up_face_quad());
    let part = ao_at(&w, BlockPos::new(8, 16, 8), &quad);
    assert_eq!(
        full, part,
        "UP has no orientation tables, so partial and full quads must agree"
    );
}

#[test]
fn a_partial_side_quad_interpolates_between_the_corner_values() {
    // A west-facing quad shrunk to the middle of its face. WEST *does* carry orientation
    // tables, so the interpolated path runs and every vertex should end up strictly between
    // the darkest and brightest full-face corner rather than equal to one of them.
    let v = |y: f32, z: f32| Vertex {
        pos: [0.0, y, z],
        uv: [0.0, 0.0],
    };
    let full = BakedQuad {
        verts: [v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0), v(1.0, 0.0)],
        face: Some(Facing::West),
        shade_face: Some(Facing::West),
        tint_index: None,
        shade: true,
    };
    let shrunk = BakedQuad {
        verts: [v(0.25, 0.25), v(0.25, 0.75), v(0.75, 0.75), v(0.75, 0.25)],
        face: Some(Facing::West),
        shade_face: Some(Facing::West),
        tint_index: None,
        shade: true,
    };

    // A wall running along the -z side of the block, so the west face is unevenly lit.
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    w.set_block(BlockPos::new(8, 16, 8), stone);
    fixture::fill(
        &mut w,
        BlockPos::new(7, 15, 7),
        BlockPos::new(7, 17, 7),
        stone,
    );

    let pos = BlockPos::new(8, 16, 8);
    let state = w.block_at(pos);
    let full_qb = fill_quad_bounds(&full, Facing::West, true);
    let shrunk_qb = fill_quad_bounds(&shrunk, Facing::West, true);
    assert!(!full_qb.partial);
    assert!(shrunk_qb.partial);

    let a = update_vertex_brightness(
        &w,
        state,
        pos,
        Facing::West,
        &full_qb,
        Some(&full.verts),
        MeshSettings::default(),
    )
    .color_multiplier;
    let b = update_vertex_brightness(
        &w,
        state,
        pos,
        Facing::West,
        &shrunk_qb,
        Some(&shrunk.verts),
        MeshSettings::default(),
    )
    .color_multiplier;
    assert_ne!(a, b, "the interpolated path must differ from the flat one");

    let lo = a.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(hi > lo, "the test setup must produce unequal corners");
    for v in b {
        assert!(
            v >= lo - 1e-5 && v <= hi + 1e-5,
            "interpolated value {v} escaped the corner range {lo}..{hi}"
        );
    }
}

#[test]
fn the_brightness_channel_is_the_packed_light_vanilla_produces() {
    // With uniform sky light 15 and no block light anywhere, every corner's packed brightness
    // is `15 << 20`, whose vertex bytes are (sky, block) = (240, 0).
    let w = floor_world();
    let pos = BlockPos::new(8, 16, 8);
    let state = w.block_at(pos);
    let quad = up_face_quad();
    let qb = fill_quad_bounds(&quad, Facing::Up, true);
    let ao = update_vertex_brightness(
        &w,
        state,
        pos,
        Facing::Up,
        &qb,
        Some(&quad.verts),
        MeshSettings::default(),
    );
    for b in ao.brightness {
        assert_eq!(mc_render::terrain::light::sky_byte(b), 240);
        assert_eq!(mc_render::terrain::light::block_byte(b), 0);
    }
}

#[test]
fn quad_bounds_flags_match_vanillas_switch() {
    // java: fillQuadBounds — `flush` is "the quad is coplanar with the face", `partial` is
    // "the quad does not cover the whole face".
    let full = fill_quad_bounds(&up_face_quad(), Facing::Up, true);
    assert!(full.flush);
    assert!(!full.partial);

    let part = fill_quad_bounds(&partial_up_face_quad(), Facing::Up, true);
    assert!(part.flush, "still coplanar with the top of the block");
    assert!(part.partial);

    // A quad floating in the middle of the block is neither.
    let v = |x: f32, z: f32| Vertex {
        pos: [x, 0.5, z],
        uv: [0.0, 0.0],
    };
    let floating = BakedQuad {
        verts: [v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0), v(1.0, 0.0)],
        face: Some(Facing::Up),
        shade_face: Some(Facing::Up),
        tint_index: None,
        shade: true,
    };
    // `full_cube = false` is the case that matters: a non-full-cube model's mid-height quad
    // must not be treated as flush.
    let mid = fill_quad_bounds(&floating, Facing::Up, false);
    assert!(!mid.flush);
    assert!(!mid.partial);
}

// ---- S11 -> S01 §13: `is_opaque_cube`'s fancy-graphics fix-up reaching the AO centre sample --

#[test]
fn fancy_graphics_leaves_lend_their_own_light_for_the_centre_sample() {
    // java: BlockModelRenderer.java:446 — `boundsFlags.get(0) || !...isOpaqueCube()` on the
    // block the face points into decides whether the AO centre sample (`i3`) is retaken at that
    // neighbour's position instead of the rendered block's own. `BlockLeaves.isOpaqueCube()` is
    // `!fancyGraphics`, not a constant (BlockLeaves.java:278-281), and `mc_data` only recorded
    // the fast-graphics value (`opaque_cube: true` for every leaf state — see
    // `cull_face::is_opaque_cube`'s doc comment). Reading that frozen field directly here,
    // instead of going through `cull_face::is_opaque_cube` the way `should_side_be_rendered`
    // already does, would silently keep leaves "opaque" for AO purposes even under fancy
    // graphics, so a quad next to fancy-graphics leaves would never borrow the leaves' own light
    // the way vanilla does.
    //
    // Renders a stone block with a synthetic quad at y = 0.125, told `full_cube: false` so it is
    // NOT flush against the top face (a flush quad, e.g. a real full cube's own top face, always
    // retakes the centre sample regardless of `is_opaque_cube` via `qb.flush ||`
    // short-circuiting -- so the branch this test targets only fires for a quad short of its
    // block's own boundary, e.g. a snow layer, carpet, or anything embedded in a full block's
    // model). Leaves sit directly above.
    let mut w = fixture::blank_world(1);

    // Poke the block-light nibble at the leaves' position to 10, the same way a chunk packet
    // would, leaving every other cell -- including the rendered block's own -- at the blank
    // world's default 0. Pattern matches `a_torch_puts_block_light_into_the_vertices_around_it`
    // in s11_mesher.rs.
    //
    // Both blocks go into the **payload** rather than through `World::set_block`. Since
    // GAP A12-01 the client recomputes light on every block change (`World.setBlockState:369`
    // -> `checkLight`), so placing the leaves afterwards would immediately propagate the 10
    // away to its true value of 0 — correctly, but that is not the state this AO test is about.
    let pos = BlockPos::new(8, 16, 8);
    let stone = fixture::state_of("minecraft:stone");
    let leaves = fixture::state_of("minecraft:leaves");
    let mut payload = fixture::blank_column_payload(true, 1);
    // Blocks: little-endian u16 at `y << 8 | z << 4 | x` within the section (H6).
    // section 1 starts at cell 4096; within it the index is `y << 8 | z << 4 | x`.
    let block_at = |y: usize| 2 * (4096 + ((y & 15) << 8) + (8 << 4) + 8);
    payload[block_at(16)] = (stone.id & 0xFF) as u8;
    payload[block_at(16) + 1] = (stone.id >> 8) as u8;
    payload[block_at(17)] = (leaves.id & 0xFF) as u8;
    payload[block_at(17) + 1] = (leaves.id >> 8) as u8;
    let block_light_start = 16 * 4096 * 2;
    // (x, y, z) = (8, 17, 8): section index 1, local y = 1. java: NibbleArray.getCoordinateIndex
    // packs `y << 8 | z << 4 | x`, two cells per byte, low nibble first -- x = 8 is even, so
    // this lands in the LOW nibble of its byte (see chunk.rs's `nibble_packing_matches_java_
    // low_high_order`).
    let section = 1usize;
    let local_index = (1usize << 8) | (8usize << 4) | 8usize;
    let byte_index = block_light_start + section * 2048 + (local_index >> 1);
    payload[byte_index] = 10;
    w.apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 0xFFFF, &payload)
        .expect("payload");

    assert_eq!(w.block_at(pos), stone);
    assert_eq!(w.block_at(pos.up()), leaves);
    assert_eq!(
        w.block_light_at(pos.up()),
        10,
        "the nibble must have landed on the leaves' own cell"
    );
    assert_eq!(
        w.block_light_at(pos),
        0,
        "the rendered block's own cell must stay untouched"
    );

    let v = |x: f32, z: f32| Vertex {
        pos: [x, 0.125, z],
        uv: [0.0, 0.0],
    };
    let quad = BakedQuad {
        verts: [v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0), v(1.0, 0.0)],
        face: Some(Facing::Up),
        shade_face: Some(Facing::Up),
        tint_index: None,
        shade: true,
    };
    let qb = fill_quad_bounds(&quad, Facing::Up, false);
    assert!(
        !qb.flush,
        "a quad short of the block's own boundary must not be flush"
    );

    let state = w.block_at(pos);
    let fancy = MeshSettings {
        fancy_graphics: true,
        ..MeshSettings::default()
    };
    let fast = MeshSettings {
        fancy_graphics: false,
        ..MeshSettings::default()
    };

    let ao_fancy =
        update_vertex_brightness(&w, state, pos, Facing::Up, &qb, Some(&quad.verts), fancy);
    let ao_fast =
        update_vertex_brightness(&w, state, pos, Facing::Up, &qb, Some(&quad.verts), fast);

    // Fast graphics: leaves ARE opaque (`isOpaqueCube() == !false == true`), so the centre
    // sample stays the rendered block's own (unlit) light. Every corner averages three
    // untouched in-plane/diagonal neighbours (block channel 0) with the untouched centre (0):
    // zero throughout.
    for (i, &b) in ao_fast.brightness.iter().enumerate() {
        assert_eq!(
            mc_render::terrain::light::block_byte(b),
            0,
            "fast graphics: vertex {i} must carry the rendered block's own (unlit) light, got {b:#010x}"
        );
    }

    // Fancy graphics: leaves are NOT opaque (`isOpaqueCube() == !true == false`), so the centre
    // sample switches to the leaves' own light. Three untouched neighbours (block channel 0)
    // plus the leaves' block light (10 << 4 = 0xA0), averaged with vanilla's plain `>> 2`:
    // (0 + 0 + 0 + 0xA0) >> 2 = 0x28 = 40.
    for (i, &b) in ao_fancy.brightness.iter().enumerate() {
        assert_eq!(
            mc_render::terrain::light::block_byte(b),
            40,
            "fancy graphics: vertex {i} must borrow the leaves' light, got {b:#010x}"
        );
    }

    assert_ne!(
        ao_fancy.brightness, ao_fast.brightness,
        "the fancy/fast difference here is exactly the bug this test exists to catch"
    );

    // The AO shading itself is untouched by this fix: `centre_ao` is gated on `qb.flush` alone
    // and never consults `is_opaque_cube`, so both settings must still agree there -- this fix
    // only ever changes the light channel, never the occlusion shading.
    assert_eq!(
        ao_fancy.color_multiplier, ao_fast.color_multiplier,
        "is_opaque_cube must not affect the AO shading, only the light channel"
    );
}
