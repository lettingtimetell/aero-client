//! End to end: the production mesher (`mesh_section`, fast paths on) draws the model of each
//! block's *actual* state, with the real 1.8.9 assets baked.
//!
//! The user-visible symptoms these pin: connected blocks drawn
//! unconnected, the top of double tall grass drawn as a flower, and doors "cut in half". All
//! three came from the release client never applying `Block.getActualState`, so each test also
//! checks that the raw-metadata model would have failed it (the built-in negative control).

use std::path::{Path, PathBuf};
use std::sync::Once;

use mc_assets::sprite::UvRect;
use mc_assets::ResourceLocation;
use mc_data::block::ActualBlockState;
use mc_data::BlockState;
use mc_math::{BlockPos, ChunkSectionPos};
use mc_render::terrain::{fixture, mesh_section, ChunkMesh, TerrainVertex};

// ---- vanilla asset bootstrap (as tests/s11_mesher.rs) ----------------------------------------

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn install_vanilla_assets() -> bool {
    static ONCE: Once = Once::new();
    static mut OK: bool = false;
    ONCE.call_once(|| {
        let Some(jar) = jars_dir()
            .map(|d| d.join("versions").join("1.8.9").join("1.8.9.jar"))
            .filter(|p| p.is_file())
        else {
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

/// The first legal packed state of `name` whose stored tuple has every `wanted` value.
fn stored_with(name: &str, wanted: &[(&str, &str)]) -> BlockState {
    let block = mc_data::block_by_name(name).unwrap();
    (0..16)
        .map(|meta| BlockState::new(block, meta))
        .filter(|state| state.is_legal())
        .find(|state| {
            let stored = ActualBlockState::from_stored(*state);
            wanted
                .iter()
                .all(|(property, value)| stored.get(property) == Some(*value))
        })
        .unwrap_or_else(|| panic!("{name} stores no state with {wanted:?}"))
}

fn sprite_rect(texture: &str) -> UvRect {
    mc_assets::model::sprite(&ResourceLocation::minecraft(format!(
        "textures/blocks/{texture}"
    )))
    .uv
}

fn uv_in(uv: [f32; 2], rect: UvRect) -> bool {
    const E: f32 = 1e-6;
    uv[0] >= rect.min_u - E
        && uv[0] <= rect.max_u + E
        && uv[1] >= rect.min_v - E
        && uv[1] <= rect.max_v + E
}

/// Every quad of every layer of `mesh`, as its four vertices.
fn quads(mesh: &ChunkMesh) -> Vec<[TerrainVertex; 4]> {
    mesh.layers()
        .flat_map(|(_, buffer)| {
            buffer
                .vertices()
                .chunks_exact(4)
                .map(|q| [q[0], q[1], q[2], q[3]])
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Quads textured from `rect` (every vertex UV inside it).
fn quads_using(mesh: &ChunkMesh, rect: UvRect) -> Vec<[TerrainVertex; 4]> {
    quads(mesh)
        .into_iter()
        .filter(|q| q.iter().all(|v| uv_in(v.uv, rect)))
        .collect()
}

/// `[min_x, max_x, min_z, max_z]` over some positions.
fn xz_extent<'a>(positions: impl IntoIterator<Item = &'a [f32; 3]>) -> [f32; 4] {
    let mut e = [f32::MAX, f32::MIN, f32::MAX, f32::MIN];
    for p in positions {
        e[0] = e[0].min(p[0]);
        e[1] = e[1].max(p[0]);
        e[2] = e[2].min(p[2]);
        e[3] = e[3].max(p[2]);
    }
    e
}

fn quads_xz_extent(quads: &[[TerrainVertex; 4]]) -> [f32; 4] {
    xz_extent(quads.iter().flatten().map(|v| &v.pos))
}

fn approx_eq(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

fn model_quads(
    model: &mc_assets::model::BakedModel,
) -> impl Iterator<Item = &mc_assets::model::BakedQuad> {
    model
        .general_quads
        .iter()
        .chain(model.face_quads.iter().flatten())
}

const SECTION: ChunkSectionPos = ChunkSectionPos::new(0, 1, 0);

// ---- tests -----------------------------------------------------------------------------------

#[test]
fn the_top_of_double_tall_grass_is_drawn_with_the_grass_top_texture() {
    require_assets!();
    let mut world = fixture::blank_world(0);
    let lower = stored_with(
        "double_plant",
        &[("half", "lower"), ("variant", "double_grass")],
    );
    let upper = stored_with("double_plant", &[("half", "upper")]);
    world.set_block(BlockPos::new(8, 16, 8), lower);
    world.set_block(BlockPos::new(8, 17, 8), upper);
    let mesh = mesh_section(&world, SECTION);

    let grass_top = sprite_rect("double_plant_grass_top");
    let grass_bottom = sprite_rect("double_plant_grass_bottom");
    let top_quads = quads_using(&mesh, grass_top);
    assert!(!top_quads.is_empty(), "no quad uses the grass top texture");
    assert!(
        top_quads
            .iter()
            .flatten()
            .all(|v| v.pos[1] >= 1.0 - 1e-4 && v.pos[1] <= 2.0 + 1e-4),
        "the grass top texture is only on the upper block"
    );
    assert!(!quads_using(&mesh, grass_bottom).is_empty());
    // Every quad is one of the two grass halves: no flower texture anywhere.
    assert_eq!(
        top_quads.len() + quads_using(&mesh, grass_bottom).len(),
        quads(&mesh).len()
    );

    // Negative control: the upper half's raw metadata resolves to the peony, whose model a
    // raw-metadata lookup draws and which has no quad textured with the grass top.
    assert_eq!(
        ActualBlockState::from_stored(upper).get("variant"),
        Some("paeonia")
    );
    let raw = mc_assets::model::model_for(upper);
    assert!(model_quads(raw).next().is_some());
    assert!(model_quads(raw).all(|q| !q.verts.iter().all(|v| uv_in(v.uv, grass_top))));
}

#[test]
fn both_halves_of_a_door_are_drawn_as_one_slab() {
    require_assets!();
    let mut world = fixture::blank_world(0);
    // A closed north-facing door: its lower half stores the facing, its upper half does not.
    let lower = stored_with(
        "wooden_door",
        &[("half", "lower"), ("facing", "north"), ("open", "false")],
    );
    let upper = stored_with(
        "wooden_door",
        &[("half", "upper"), ("hinge", "left"), ("powered", "false")],
    );
    world.set_block(BlockPos::new(8, 16, 8), lower);
    world.set_block(BlockPos::new(8, 17, 8), upper);
    let mesh = mesh_section(&world, SECTION);

    let all = quads(&mesh);
    let lower_quads: Vec<_> = all
        .iter()
        .copied()
        .filter(|q| q.iter().all(|v| v.pos[1] <= 1.0 + 1e-4))
        .collect();
    let upper_quads: Vec<_> = all
        .iter()
        .copied()
        .filter(|q| q.iter().all(|v| v.pos[1] >= 1.0 - 1e-4))
        .collect();
    assert!(!lower_quads.is_empty() && !upper_quads.is_empty());
    assert_eq!(lower_quads.len() + upper_quads.len(), all.len());
    let lower_extent = quads_xz_extent(&lower_quads);
    let upper_extent = quads_xz_extent(&upper_quads);
    assert!(
        approx_eq(lower_extent, upper_extent),
        "door halves differ: lower {lower_extent:?}, upper {upper_extent:?}"
    );
    // A door is 3/16 thick on one horizontal axis and a full block on the other.
    let mut size = [
        lower_extent[1] - lower_extent[0],
        lower_extent[3] - lower_extent[2],
    ];
    size.sort_by(f32::total_cmp);
    assert!(
        (size[0] - 3.0 / 16.0).abs() < 1e-4 && (size[1] - 1.0).abs() < 1e-4,
        "{size:?}"
    );

    // Negative control: the upper half's raw metadata says facing east, and its model's slab
    // lies on the other axis, which is what a raw-metadata lookup drew.
    let raw_upper = ActualBlockState::from_stored(upper);
    assert_eq!(raw_upper.get("facing"), Some("east"));
    let raw_positions: Vec<[f32; 3]> = model_quads(mc_assets::model::model_for(upper))
        .flat_map(|q| {
            q.verts
                .iter()
                .map(|v| [v.pos[0] + 8.0, v.pos[1], v.pos[2] + 8.0])
        })
        .collect();
    assert!(!approx_eq(xz_extent(&raw_positions), lower_extent));
}

#[test]
fn a_fence_grows_an_arm_towards_a_solid_neighbour() {
    require_assets!();
    let fence = BlockPos::new(8, 16, 8);
    let planks = sprite_rect("planks_oak");

    let mut world = fixture::blank_world(0);
    fixture::place(&mut world, fence, "minecraft:fence");
    let alone = quads_using(&mesh_section(&world, SECTION), planks);
    // The post alone, 6/16..10/16 on both axes: all a raw-metadata lookup can draw.
    assert!(approx_eq(
        quads_xz_extent(&alone),
        [8.375, 8.625, 8.375, 8.625]
    ));

    fixture::place(&mut world, fence.east(), "minecraft:stone");
    let connected = quads_using(&mesh_section(&world, SECTION), planks);
    assert!(connected.len() > alone.len());
    // The arm reaches the stone's face at x = 9.
    assert!(approx_eq(
        quads_xz_extent(&connected),
        [8.375, 9.0, 8.375, 8.625]
    ));
}

#[test]
fn a_fence_at_a_section_edge_reconnects_when_the_next_section_changes() {
    require_assets!();
    // java: `RenderGlobal.markBlockForUpdate` marks the (x-1..x+1, y-1..y+1, z-1..z+1) box, so
    // a block placed in one section re-meshes the fence across the boundary in the next one.
    let mut world = fixture::blank_world(1);
    let fence = BlockPos::new(15, 16, 8); // the last column of section (0, 1, 0)
    fixture::place(&mut world, fence, "minecraft:fence");
    world.drain_dirty_sections();

    fixture::place(&mut world, fence.east(), "minecraft:stone"); // x = 16: section (1, 1, 0)
    let dirty = world.drain_dirty_sections();
    assert!(
        dirty.contains(&SECTION),
        "the fence's section must be re-meshed too: {dirty:?}"
    );
    let planks = sprite_rect("planks_oak");
    let fence_quads = quads_using(&mesh_section(&world, SECTION), planks);
    assert!(approx_eq(
        quads_xz_extent(&fence_quads),
        [15.375, 16.0, 8.375, 8.625]
    ));
}
