//! S11 acceptance tests: the mesher against real vanilla assets.
//!
//! `mesh_section` is a pure function of a `World`, so every claim about it can be checked
//! against an exact number rather than by eye. These tests bake the real 1.8.9 jar (skipping
//! cleanly when it is absent) and then assert precise quad counts, precise layer assignment
//! and precise face culling on hand-built worlds.

use std::path::{Path, PathBuf};
use std::sync::Once;

use mc_data::RenderLayer;
use mc_math::{BlockPos, ChunkSectionPos, Facing};
use mc_render::terrain::{fixture, mesh_section, mesh_section_with, ChunkMesh, MeshSettings};
use mc_render::terrain::{overlay::crack_vertices, BreakProgress};
use mc_render::Camera;
use mc_world::World;

// ---- vanilla asset bootstrap ---------------------------------------------------------------

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_jar() -> Option<PathBuf> {
    let p = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    p.is_file().then_some(p)
}

/// Bake the real vanilla assets exactly once for the whole test binary, and install the
/// colormaps too. Returns `false` when the jar is not on disk.
pub fn install_vanilla_assets() -> bool {
    static ONCE: Once = Once::new();
    static mut OK: bool = false;
    ONCE.call_once(|| {
        let Some(jar) = vanilla_jar() else {
            eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found; set MC_JARS_DIR");
            return;
        };
        let mut rm = mc_assets::ResourceManager::new();
        rm.push_pack(Box::new(
            mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
        ));
        let (assets, report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
        assert!(report.is_clean(), "vanilla bake was not clean");
        mc_assets::model::install(assets);
        mc_render::terrain::tint::load_colormaps(&rm).expect("colormaps");
        // SAFETY: only written inside `call_once`, only read after it has returned.
        unsafe { OK = true };
    });
    unsafe { OK }
}

macro_rules! require_assets {
    () => {
        if !install_vanilla_assets() {
            return;
        }
    };
}

// ---- helpers ---------------------------------------------------------------------------------

/// A world with one 16x16 layer of `name` filling y = 16..=16 (section 1), and the rest air.
fn slab_world(name: &str, y: i32) -> World {
    let mut w = fixture::blank_world(1);
    let s = fixture::state_of(name);
    fixture::fill(&mut w, BlockPos::new(0, y, 0), BlockPos::new(15, y, 15), s);
    w
}

fn layer_quads(mesh: &ChunkMesh, layer: RenderLayer) -> usize {
    mesh.layers()
        .find(|(l, _)| *l == layer)
        .map(|(_, b)| b.quad_count())
        .unwrap_or(0)
}

// ---- task 1: per-section mesher, neighbour face culling ---------------------------------------

/// Wool and stained clay mesh like any other cube: all four light bytes of every vertex equal a
/// stone block's in the same spot (with and without smooth lighting, at the world's edges too),
/// so the two spare bytes are zero at every height, as vanilla's.
#[test]
fn wool_and_clay_vertices_carry_only_vanillas_light_at_every_height() {
    require_assets!();
    for name in [
        "minecraft:wool",
        "minecraft:stained_hardened_clay",
        "minecraft:stone",
    ] {
        for y in [0, 15, 16, 99, 110, 255] {
            for smooth_lighting in [false, true] {
                let settings = MeshSettings {
                    ambient_occlusion: smooth_lighting,
                    ..Default::default()
                };
                let section = ChunkSectionPos::new(0, y >> 4, 0);
                let mut world = fixture::blank_world(1);
                fixture::place(&mut world, BlockPos::new(8, y, 8), name);
                let mesh = mesh_section_with(&world, section, settings);
                assert_eq!(mesh.quad_count(), 6);
                fixture::place(&mut world, BlockPos::new(8, y, 8), "minecraft:stone");
                let control = mesh_section_with(&world, section, settings);
                let lights = |mesh: &ChunkMesh| {
                    mesh.layers()
                        .flat_map(|(_, buffer)| buffer.vertices().iter().map(|v| v.light))
                        .collect::<Vec<_>>()
                };
                assert_eq!(lights(&mesh), lights(&control), "{name} at y={y}");
                for (_, buffer) in mesh.layers() {
                    for vertex in buffer.vertices() {
                        assert_eq!(vertex.light[2..], [0, 0], "{name} at y={y}");
                    }
                }
            }
        }
    }
}

#[test]
fn a_single_stone_block_emits_exactly_six_faces() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    fixture::place(&mut w, BlockPos::new(8, 20, 8), "minecraft:stone");
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    assert_eq!(mesh.quad_count(), 6, "one exposed cube is six quads");
    assert_eq!(layer_quads(&mesh, RenderLayer::Solid), 6);
}

#[test]
fn two_adjacent_stone_blocks_cull_the_shared_face() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    fixture::place(&mut w, BlockPos::new(8, 20, 8), "minecraft:stone");
    fixture::place(&mut w, BlockPos::new(9, 20, 8), "minecraft:stone");
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    // 12 faces minus the two that touch.
    assert_eq!(mesh.quad_count(), 10);
}

#[test]
fn a_solid_section_emits_only_its_outer_shell() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 31, 15),
        stone,
    );
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    // The section is surrounded by air above and below and by air in the neighbouring
    // columns, so exactly the six 16x16 outer faces survive.
    assert_eq!(mesh.quad_count(), 6 * 16 * 16);
    // ... and its VisGraph says nothing can see through it.
    for a in Facing::VALUES {
        for b in Facing::VALUES {
            assert!(!mesh.visibility.is_visible(a, b));
        }
    }
}

#[test]
fn a_section_buried_inside_solid_stone_emits_nothing() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(-16, 0, -16),
        BlockPos::new(31, 47, 31),
        stone,
    );
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    assert!(
        mesh.is_empty(),
        "a fully buried section must produce no quads"
    );
}

#[test]
fn culling_is_symmetric_across_a_section_boundary() {
    require_assets!();
    // Two stone blocks straddling the y=32 section boundary must not draw the shared face
    // from either side.
    let mut w = fixture::blank_world(1);
    fixture::place(&mut w, BlockPos::new(4, 31, 4), "minecraft:stone");
    fixture::place(&mut w, BlockPos::new(4, 32, 4), "minecraft:stone");
    let below = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    let above = mesh_section(&w, ChunkSectionPos::new(0, 2, 0));
    assert_eq!(below.quad_count(), 5);
    assert_eq!(above.quad_count(), 5);
}

// ---- task 3: layer assignment -----------------------------------------------------------------

#[test]
fn every_block_lands_on_vanillas_own_render_layer() {
    require_assets!();
    let cases: [(&str, RenderLayer); 7] = [
        ("minecraft:stone", RenderLayer::Solid),
        // java: BlockLeaves.getBlockLayer / BlockGrass.getBlockLayer
        ("minecraft:leaves", RenderLayer::CutoutMipped),
        ("minecraft:grass", RenderLayer::CutoutMipped),
        ("minecraft:glass", RenderLayer::Cutout),
        ("minecraft:tallgrass", RenderLayer::Cutout),
        ("minecraft:stained_glass", RenderLayer::Translucent),
        ("minecraft:ice", RenderLayer::Translucent),
    ];
    for (name, expected) in cases {
        let mut w = fixture::blank_world(1);
        fixture::place(&mut w, BlockPos::new(8, 20, 8), name);
        let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
        assert!(mesh.quad_count() > 0, "{name} produced no geometry");
        for (layer, buf) in mesh.layers() {
            if layer == expected {
                assert!(!buf.is_empty(), "{name} missing from {layer:?}");
            } else {
                assert!(buf.is_empty(), "{name} leaked into {layer:?}");
            }
        }
    }
}

#[test]
fn water_lands_on_the_translucent_layer_and_lava_on_solid() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    fixture::place(&mut w, BlockPos::new(8, 20, 8), "minecraft:water");
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    assert!(!mesh.translucent.is_empty(), "water must be translucent");
    assert!(mesh.solid.is_empty());

    let mut w = fixture::blank_world(1);
    fixture::place(&mut w, BlockPos::new(8, 20, 8), "minecraft:lava");
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    // java: BlockLiquid.getBlockLayer — lava is SOLID.
    assert!(!mesh.solid.is_empty(), "lava must be solid");
    assert!(mesh.translucent.is_empty());
}

// ---- task 2: smooth lighting --------------------------------------------------------------------

#[test]
fn ambient_occlusion_darkens_the_corners_of_an_inside_edge() {
    require_assets!();
    // An L-shaped notch: the top face of the floor block next to a wall must have two dark
    // corners (occluded by the wall) and two bright ones.
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(8, 17, 0),
        BlockPos::new(8, 20, 15),
        stone,
    );

    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    // Find the top face of the block at (7, 16, 4): section-local (7, 0, 4), y = 1.0.
    let tops: Vec<_> = mesh
        .solid
        .vertices()
        .chunks(4)
        .filter(|q| {
            q.iter().all(|v| {
                (v.position()[1] - 1.0).abs() < 1e-4
                    && v.position()[0] >= 7.0 - 1e-4
                    && v.position()[0] <= 8.0 + 1e-4
                    && v.position()[2] >= 4.0 - 1e-4
                    && v.position()[2] <= 5.0 + 1e-4
            })
        })
        .collect();
    assert_eq!(tops.len(), 1, "expected exactly one top quad there");
    let q = tops[0];
    let bright: Vec<u8> = q.iter().map(|v| v.color[0]).collect();
    let dark = bright.iter().filter(|c| **c < 255).count();
    assert_eq!(
        dark, 2,
        "the two corners touching the wall must be occluded, got {bright:?}"
    );
    // The two occluded corners are the ones nearest x = 8 (the wall).
    for v in q {
        let near_wall = (v.position()[0] - 8.0).abs() < 1e-4;
        assert_eq!(
            near_wall,
            v.color[0] < 255,
            "corner {:?} occlusion does not match its position",
            v.position()
        );
    }
}

#[test]
fn disabling_smooth_lighting_makes_every_corner_of_a_face_equal() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(8, 17, 0),
        BlockPos::new(8, 20, 15),
        stone,
    );

    let flat = MeshSettings {
        ambient_occlusion: false,
        ..MeshSettings::default()
    };
    let mesh = mesh_section_with(&w, ChunkSectionPos::new(0, 1, 0), flat);
    for q in mesh.solid.vertices().chunks(4) {
        let c = q[0].color;
        assert!(
            q.iter().all(|v| v.color == c),
            "flat lighting must be uniform per quad"
        );
    }
}

#[test]
fn a_light_emitting_block_never_takes_the_smooth_lighting_path() {
    require_assets!();
    // java: `block.getLightValue() == 0` gates the AO path — glowstone is always flat.
    let mut w = fixture::blank_world(1);
    let glow = fixture::state_of("minecraft:glowstone");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        glow,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(8, 17, 0),
        BlockPos::new(8, 20, 15),
        glow,
    );
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    for q in mesh.solid.vertices().chunks(4) {
        let c = q[0].color;
        assert!(q.iter().all(|v| v.color == c));
    }
}

// ---- face brightness ---------------------------------------------------------------------------

#[test]
fn face_brightness_follows_vanillas_directional_constants() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    fixture::place(&mut w, BlockPos::new(8, 20, 8), "minecraft:stone");
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));

    // Group the six faces by their plane and read back the red channel: with no occlusion
    // anywhere the vertex colour is exactly the face shade.
    // java: FaceBakery.getFaceBrightness — UP 1.0, DOWN 0.5, N/S 0.8, W/E 0.6.
    let mut seen: Vec<(char, u8)> = Vec::new();
    for q in mesh.solid.vertices().chunks(4) {
        let ys: Vec<f32> = q.iter().map(|v| v.position()[1]).collect();
        let xs: Vec<f32> = q.iter().map(|v| v.position()[0]).collect();
        let zs: Vec<f32> = q.iter().map(|v| v.position()[2]).collect();
        let flat = |v: &Vec<f32>| v.iter().all(|a| (a - v[0]).abs() < 1e-4);
        let axis = if flat(&ys) {
            if ys[0] > 4.5 {
                'u'
            } else {
                'd'
            }
        } else if flat(&zs) {
            'z'
        } else if flat(&xs) {
            'x'
        } else {
            '?'
        };
        seen.push((axis, q[0].color[0]));
    }
    let of = |a: char| seen.iter().find(|(c, _)| *c == a).map(|(_, v)| *v);
    assert_eq!(of('u'), Some(255));
    assert_eq!(of('d'), Some((0.5f32 * 255.0) as u8));
    assert_eq!(of('z'), Some((0.8f32 * 255.0) as u8));
    assert_eq!(of('x'), Some((0.6f32 * 255.0) as u8));
}

// ---- task 5: biome tint --------------------------------------------------------------------------

#[test]
fn grass_is_tinted_by_its_biome_and_stone_is_not() {
    require_assets!();
    let mut w = slab_world("minecraft:grass", 20);
    // A tinted quad has a non-grey colour; an untinted one is pure grey (r == g == b).
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    let tinted = mesh
        .layers()
        .flat_map(|(_, b)| b.vertices())
        .any(|v| v.color[0] != v.color[1] || v.color[1] != v.color[2]);
    assert!(tinted, "grass must pick up a biome tint");

    w = slab_world("minecraft:stone", 20);
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    assert!(
        mesh.layers()
            .flat_map(|(_, b)| b.vertices())
            .all(|v| v.color[0] == v.color[1] && v.color[1] == v.color[2]),
        "stone must not be tinted"
    );
}

#[test]
fn different_biomes_give_grass_different_colours() {
    require_assets!();
    // Desert (id 2) vs Swampland (id 6): swamp is a hard-coded constant, desert comes from
    // the colormap, so they must differ.
    let colours: Vec<[u8; 4]> = [2u8, 6]
        .into_iter()
        .map(|biome| {
            let mut w = World::new();
            w.set_has_sky_light(true);
            for cx in -1..=1 {
                for cz in -1..=1 {
                    let payload = fixture::blank_column_payload(true, biome);
                    w.apply_chunk_data(mc_math::ChunkPos::new(cx, cz), true, 0xFFFF, &payload)
                        .expect("payload");
                }
            }
            let grass = fixture::state_of("minecraft:grass");
            fixture::fill(
                &mut w,
                BlockPos::new(0, 20, 0),
                BlockPos::new(15, 20, 15),
                grass,
            );
            let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
            // The top face of a grass block is the tinted one.
            let colour = mesh
                .layers()
                .flat_map(|(_, b)| b.vertices())
                .find(|v| v.color[0] != v.color[1])
                .map(|v| v.color)
                .expect("a tinted grass vertex");
            colour
        })
        .collect();
    assert_ne!(colours[0], colours[1], "biome tint did not change");
}

// ---- lighting values ------------------------------------------------------------------------------

#[test]
fn sky_light_reaches_the_vertices_as_vanillas_level_times_sixteen() {
    require_assets!();
    // The block arrives in the **chunk packet**, not through `World::set_block`. This test is
    // about the mesher turning a stored nibble into a vertex light byte, and since GAP A12-01
    // a client-side placement recomputes the column's sky light (correctly: a block under open
    // sky shadows everything below it — see `mc-world`'s `roof_over_open_ground` parity
    // fixture), which would change the very nibbles this is reading.
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    let mut payload = fixture::blank_column_payload(true, 1);
    // Blocks are little-endian `u16` at `y << 8 | z << 4 | x` within the section (H6).
    let idx = 2 * (4096 + ((20usize & 15) << 8) + (8 << 4) + 8);
    payload[idx] = (stone.id & 0xFF) as u8;
    payload[idx + 1] = (stone.id >> 8) as u8;
    w.apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 0xFFFF, &payload)
        .expect("payload");
    assert_eq!(w.block_at(BlockPos::new(8, 20, 8)), stone);

    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    for v in mesh.solid.vertices() {
        assert_eq!(v.light[0], 15 * 16, "full sky light is 240");
        assert_eq!(v.light[1], 0, "no block light in an empty world");
    }
}

#[test]
fn a_torch_puts_block_light_into_the_vertices_around_it() {
    require_assets!();
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    // The server is authoritative for light, so set the nibble directly the way a chunk
    // packet would have.
    let mesh_before = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    assert!(mesh_before.solid.vertices().iter().all(|v| v.light[1] == 0));

    // Rebuild the column with block light 14 everywhere.
    let mut payload = fixture::blank_column_payload(true, 1);
    let block_light_start = 16 * 4096 * 2;
    for b in &mut payload[block_light_start..block_light_start + 16 * 2048] {
        *b = 0xEE; // both nibbles = 14
    }
    w.apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 0xFFFF, &payload)
        .expect("payload");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    assert!(
        mesh.solid.vertices().iter().any(|v| v.light[1] == 14 * 16),
        "block light did not reach the vertices"
    );
}

// ---- vanilla hazard canaries -----------------------------------------------------------------------

#[test]
fn stairs_and_fences_produce_geometry() {
    require_assets!();
    // Hazard H12's canaries: if uvlock or the parent chain were wrong these would be empty.
    // `default_state()` is `meta = 0`, which is not a *legal* state for every block (a torch's
    // metas run 1..=5 and a ladder's 2..=5), so each canary names the meta it needs.
    for (name, meta) in [
        ("minecraft:oak_stairs", 0),
        ("minecraft:fence", 0),
        ("minecraft:cobblestone_wall", 0),
        ("minecraft:torch", 5),
        ("minecraft:ladder", 2),
    ] {
        let mut w = fixture::blank_world(1);
        let block = mc_data::block_by_name(name).unwrap_or_else(|| panic!("{name}"));
        let state = mc_data::BlockState::new(block, meta);
        assert!(state.is_legal(), "{name} meta {meta} is not a legal state");
        w.set_block(BlockPos::new(8, 20, 8), state);
        let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
        assert!(mesh.quad_count() > 0, "{name} produced no geometry");
    }
}

#[test]
fn crack_overlay_uses_the_fences_actual_connected_model() {
    require_assets!();
    let mut world = fixture::blank_world(1);
    let pos = BlockPos::new(8, 20, 8);
    fixture::place(&mut world, pos, "minecraft:fence");
    let stored = world.block_at(pos);
    let progress = BreakProgress { pos, stage: 3 };
    let cam = Camera::default();

    let isolated = mc_render::terrain::mesher::actual_state(&world, stored, pos);
    let isolated_vertices = crack_vertices(progress, &isolated, &cam);

    fixture::place(&mut world, pos.east(), "minecraft:stone");
    let connected = mc_render::terrain::mesher::actual_state(&world, stored, pos);
    let connected_vertices = crack_vertices(progress, &connected, &cam);
    assert_ne!(
        connected_vertices, isolated_vertices,
        "negative control: a raw-state lookup gives the isolated fence crack geometry"
    );

    world.set_block(pos.east(), mc_data::BlockState::AIR);
    let disconnected = mc_render::terrain::mesher::actual_state(&world, stored, pos);
    assert_eq!(
        crack_vertices(progress, &disconnected, &cam),
        isolated_vertices,
        "removing the neighbour must remove the connected arm again"
    );
}

#[test]
fn meshing_every_legal_state_never_panics() {
    require_assets!();
    // The mesher sees whatever the server sends, including illegal states.
    let mut w = fixture::blank_world(1);
    let mut placed = 0usize;
    for id in 0..4096u16 {
        let state = mc_data::BlockState::from_raw(id);
        let x = (placed % 16) as i32;
        let z = ((placed / 16) % 16) as i32;
        let y = 16 + ((placed / 256) % 16) as i32;
        w.set_block(BlockPos::new(x, y, z), state);
        placed += 1;
        if placed == 4096 {
            break;
        }
    }
    let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
    assert!(mesh.quad_count() > 0);
}
