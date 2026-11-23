//! Connected textures on a real bake: the vanilla 1.8.9 jar plus
//! a generated CTM pack (`mcpatcher/ctm/glass/glass.properties`, `method=ctm`, 47 numbered
//! tiles), meshed by the production mesher.
//!
//! The registry is process-wide, so this binary installs exactly one bake (the jar + the
//! generated pack) and every test reads it. It skips, loudly, without the jar
//! (`MC_JARS_DIR`, default `../MCP-919-main/jars`).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use mc_assets::model::BakedQuad;
use mc_assets::{FolderResourcePack, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_data::BlockState;
use mc_math::{BlockPos, ChunkSectionPos, Facing};
use mc_render::terrain::ctm::{fixed_orientation, orientation};
use mc_render::terrain::{fixture, mesh_section_with, ChunkMesh, CtmMode, MeshSettings};
use mc_world::World;

fn jar() -> Option<PathBuf> {
    let dir = std::env::var_os("MC_JARS_DIR").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../MCP-919-main/jars"),
        PathBuf::from,
    );
    let jar = dir.join("versions/1.8.9/1.8.9.jar");
    jar.is_file().then_some(jar)
}

/// A folder pack with one clear-glass rule and 47 distinct 16×16 tiles.
fn write_ctm_pack(root: &Path) {
    let dir = root.join("assets/minecraft/mcpatcher/ctm/glass");
    std::fs::create_dir_all(&dir).expect("ctm dir");
    std::fs::write(dir.join("glass.properties"), "method=ctm\r\ntiles=0-46\r\n").expect("rule");
    for n in 0..47u8 {
        let pixel = image::Rgba([n * 5, 255 - n * 5, 128, 160]);
        image::RgbaImage::from_pixel(16, 16, pixel)
            .save(dir.join(format!("{n}.png")))
            .expect("tile png");
    }
}

/// Bake and install the jar + generated CTM pack once for this binary.
fn install() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let Some(jar) = jar() else {
            eprintln!("SKIP: the vanilla 1.8.9 jar is not available");
            return false;
        };
        let pack = tempfile::tempdir().expect("tempdir");
        write_ctm_pack(pack.path());
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(ZipResourcePack::open(&jar).expect("open jar")));
        rm.push_pack(Box::new(
            FolderResourcePack::open(pack.path()).expect("open ctm pack"),
        ));
        let (assets, report) =
            mc_assets::model::bake_all_with_ctm(&rm, 4096, 4, true).expect("bake with CTM");
        assert!(report.is_clean(), "{:?}", report.failures);
        let table = assets.ctm().expect("a CTM bake carries a table");
        assert_eq!(table.rule_count(), 1, "the generated glass rule");
        mc_assets::model::install(assets);
        true
    })
}

fn settings(mode: CtmMode) -> MeshSettings {
    MeshSettings {
        connected_textures: mode,
        ..MeshSettings::default()
    }
}

fn state(name: &str, meta: u8) -> BlockState {
    BlockState::new(mc_data::block_by_name(name).expect("block"), meta)
}

/// One emitted quad, in section-local coordinates.
#[derive(Debug, Clone, Copy)]
struct Quad {
    min: [f32; 3],
    max: [f32; 3],
    uv: [f32; 2],
}

fn quads(mesh: &ChunkMesh) -> Vec<Quad> {
    let mut out = Vec::new();
    for (_, buffer) in mesh.layers() {
        for i in 0..buffer.quad_count() {
            let verts = buffer.quad(i);
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            let mut uv = [0.0f32; 2];
            for v in verts {
                for (axis, value) in v.position().into_iter().enumerate() {
                    min[axis] = min[axis].min(value);
                    max[axis] = max[axis].max(value);
                }
                uv[0] += v.texcoord()[0] / 4.0;
                uv[1] += v.texcoord()[1] / 4.0;
            }
            out.push(Quad { min, max, uv });
        }
    }
    out
}

/// The generated tile a UV falls in, if any.
fn tile_of(uv: [f32; 2]) -> Option<usize> {
    (0..47).find(|n| {
        let tile = ResourceLocation::minecraft(format!("mcpatcher/ctm/glass/{n}"));
        mc_assets::ctm::rect_contains(&mc_assets::model::sprite(&tile).uv, uv)
    })
}

fn sprite_contains(name: &str, uv: [f32; 2]) -> bool {
    let location = ResourceLocation::minecraft(format!("textures/blocks/{name}"));
    mc_assets::ctm::rect_contains(&mc_assets::model::sprite(&location).uv, uv)
}

const EPS: f32 = 1e-4;

/// The tile on the full-block face of the block at `pos` (world) facing `face`, meshed in
/// section (0, 4, 0).
fn face_tile(mesh: &ChunkMesh, pos: BlockPos, face: Facing) -> Option<usize> {
    let local = [pos.x as f32, (pos.y - 64) as f32, pos.z as f32];
    let (axis, plane) = match face {
        Facing::Down => (1, local[1]),
        Facing::Up => (1, local[1] + 1.0),
        Facing::North => (2, local[2]),
        Facing::South => (2, local[2] + 1.0),
        Facing::West => (0, local[0]),
        Facing::East => (0, local[0] + 1.0),
    };
    let found: Vec<Quad> = quads(mesh)
        .into_iter()
        .filter(|q| (q.min[axis] - plane).abs() < EPS && (q.max[axis] - plane).abs() < EPS)
        .filter(|q| {
            (0..3).filter(|&a| a != axis).all(|a| {
                (q.min[a] - local[a]).abs() < EPS && (q.max[a] - local[a] - 1.0).abs() < EPS
            })
        })
        .collect();
    assert_eq!(found.len(), 1, "one quad for {pos:?} {face:?}: {found:?}");
    tile_of(found[0].uv)
}

/// The spec's §3.6 worked example: a 3×3 glass wall seen from the south.
#[test]
fn a_3x3_glass_wall_picks_tiles_26_14_13_39() {
    if !install() {
        return;
    }
    let mut w = fixture::blank_world(1);
    let glass = state("glass", 0);
    fixture::fill(
        &mut w,
        BlockPos::new(2, 66, 5),
        BlockPos::new(4, 68, 5),
        glass,
    );
    let section = ChunkSectionPos::new(0, 4, 0);
    for mode in [CtmMode::Fancy, CtmMode::Fast] {
        let mesh = mesh_section_with(&w, section, settings(mode));
        let at = |x, y, face| face_tile(&mesh, BlockPos::new(x, y, 5), face);
        // Seen from the south: left is west (-x).
        assert_eq!(at(3, 67, Facing::South), Some(26), "{mode:?} centre");
        assert_eq!(at(3, 68, Facing::South), Some(14), "{mode:?} top middle");
        assert_eq!(at(2, 68, Facing::South), Some(13), "{mode:?} top left");
        assert_eq!(at(4, 66, Facing::South), Some(39), "{mode:?} bottom right");
        assert_eq!(at(2, 67, Facing::South), Some(25), "{mode:?} left middle");
        // Seen from the north the wall is mirrored: its top-left is the east end.
        assert_eq!(
            at(3, 67, Facing::North),
            Some(26),
            "{mode:?} centre (north)"
        );
        assert_eq!(
            at(4, 68, Facing::North),
            Some(13),
            "{mode:?} top left (north)"
        );
        assert_eq!(
            at(2, 66, Facing::North),
            Some(39),
            "{mode:?} bottom right (north)"
        );
        // The top face of the top row: a row of three seen from above.
        assert_eq!(
            at(3, 68, Facing::Up),
            Some(2),
            "{mode:?} top face, middle of a row"
        );
        assert_eq!(
            at(2, 68, Facing::Up),
            Some(1),
            "{mode:?} top face, west end"
        );
    }
}

/// Fancy reads corners; Fast assumes them (the format's two modes).
#[test]
fn fancy_reads_corners_and_fast_assumes_them() {
    if !install() {
        return;
    }
    // An L of three: the corner block connects right and up, but its up-right is missing.
    let mut w = fixture::blank_world(1);
    let glass = state("glass", 0);
    for (x, y) in [(2, 66), (3, 66), (2, 67)] {
        w.set_block(BlockPos::new(x, y, 5), glass);
    }
    let section = ChunkSectionPos::new(0, 4, 0);
    let corner = BlockPos::new(2, 66, 5);
    let fancy = mesh_section_with(&w, section, settings(CtmMode::Fancy));
    let fast = mesh_section_with(&w, section, settings(CtmMode::Fast));
    assert_eq!(
        face_tile(&fancy, corner, Facing::South),
        Some(16),
        "R U, no UR"
    );
    assert_eq!(
        face_tile(&fast, corner, Facing::South),
        Some(37),
        "R U, UR assumed"
    );
}

#[test]
fn stacked_panes_lose_the_edges_between_them() {
    if !install() {
        return;
    }
    let mut w = fixture::blank_world(1);
    let pane = state("glass_pane", 0);
    // Two free-standing pane columns: one of two, one of three.
    for y in 66..=67 {
        w.set_block(BlockPos::new(8, y, 8), pane);
    }
    for y in 66..=68 {
        w.set_block(BlockPos::new(12, y, 3), pane);
    }
    let section = ChunkSectionPos::new(0, 4, 0);
    // Up/Down `glass_pane_top` quads on a horizontal plane between two panes of a column.
    let seams = |mesh: &ChunkMesh| {
        quads(mesh)
            .into_iter()
            .filter(|q| (q.max[1] - q.min[1]).abs() < EPS)
            .filter(|q| sprite_contains("glass_pane_top", q.uv))
            .filter(|q| {
                let y = q.min[1];
                let between = |lo: f32, hi: f32| y > lo + EPS && y < hi - EPS;
                let in_column = |x: f32, z: f32| {
                    q.min[0] >= x - EPS
                        && q.max[0] <= x + 1.0 + EPS
                        && q.min[2] >= z - EPS
                        && q.max[2] <= z + 1.0 + EPS
                };
                (in_column(8.0, 8.0) && between(2.0, 4.0))
                    || (in_column(12.0, 3.0) && between(2.0, 5.0))
            })
            .count()
    };
    let off = mesh_section_with(&w, section, settings(CtmMode::Off));
    let fancy = mesh_section_with(&w, section, settings(CtmMode::Fancy));
    assert!(
        seams(&off) > 0,
        "control: vanilla draws the edges between stacked panes"
    );
    assert_eq!(seams(&fancy), 0, "the stacked-pane edge merge removes them");
    // The outer ends keep their edges: top of each column, bottom of each column.
    let ends = |mesh: &ChunkMesh| {
        quads(mesh)
            .into_iter()
            .filter(|q| (q.max[1] - q.min[1]).abs() < EPS)
            .filter(|q| sprite_contains("glass_pane_top", q.uv))
            .count()
    };
    assert!(ends(&fancy) > 0, "the outermost edges stay");
    assert_eq!(
        off.quad_count() - fancy.quad_count(),
        seams(&off),
        "nothing but the seams disappears"
    );
    // The pane faces themselves are connected glass: some land on CTM tiles.
    assert!(quads(&fancy).iter().any(|q| tile_of(q.uv).is_some()));
    assert!(quads(&off).iter().all(|q| tile_of(q.uv).is_none()));
}

/// Mode Off on a CTM bake: the table is installed but never read.
#[test]
fn off_leaves_every_quad_on_its_vanilla_sprite_even_with_a_table() {
    if !install() {
        return;
    }
    let mut w = fixture::blank_world(1);
    fixture::fill(
        &mut w,
        BlockPos::new(1, 65, 1),
        BlockPos::new(6, 69, 1),
        state("glass", 0),
    );
    fixture::fill(
        &mut w,
        BlockPos::new(1, 65, 4),
        BlockPos::new(6, 67, 4),
        state("glass_pane", 0),
    );
    let section = ChunkSectionPos::new(0, 4, 0);
    let off = mesh_section_with(&w, section, settings(CtmMode::Off));
    let fancy = mesh_section_with(&w, section, settings(CtmMode::Fancy));
    assert!(quads(&off).iter().all(|q| tile_of(q.uv).is_none()));
    assert!(
        quads(&fancy)
            .iter()
            .filter(|q| tile_of(q.uv).is_some())
            .count()
            > 20
    );
    // Off is exactly the default settings, i.e. what the client meshed before the mod existed.
    assert_eq!(off, mesh_section_with(&w, section, MeshSettings::default()));
}

/// UV-derived orientation equals the format's fixed table on a cube, and fixes the
/// mirrored back face of a pane.
#[test]
fn derived_orientation_matches_the_fixed_table_on_a_cube_and_mirrors_on_a_pane() {
    if !install() {
        return;
    }
    let glass = mc_assets::model::model_for(state("glass", 0));
    for face in Facing::VALUES {
        let quad: &BakedQuad = &glass.face_quads[face.index() as usize][0];
        assert_eq!(quad.face, Some(face));
        assert_eq!(orientation(quad, face), fixed_orientation(face), "{face:?}");
    }
    // A north-south pane: its west face uses uv [16,0,0,16], so texture-left is south.
    let mut actual = mc_data::block::ActualBlockState::from_stored(state("glass_pane", 0));
    assert!(actual.set("north", "true"));
    assert!(actual.set("south", "true"));
    let pane = mc_assets::model::model_for_actual(&actual);
    let west = pane
        .general_quads
        .iter()
        .find(|q| q.face == Some(Facing::West))
        .expect("pane west face");
    let [left, right, down, up] = orientation(west, Facing::West);
    assert_eq!(
        (left, right, down, up),
        (Facing::South, Facing::North, Facing::Down, Facing::Up)
    );
    let east = pane
        .general_quads
        .iter()
        .find(|q| q.face == Some(Facing::East))
        .expect("pane east face");
    assert_eq!(
        orientation(east, Facing::East),
        fixed_orientation(Facing::East)
    );
}

/// The reference pack (read-only, when present) through the real bake: the tiles fit the GPU's
/// atlas, and glass and stained glass resolve to the sets the spec predicts.
#[test]
fn the_users_pack_bakes_with_its_ctm_tiles() {
    let Some(jar) = jar() else {
        eprintln!("SKIP: no jar");
        return;
    };
    // `MC_REFERENCE_PACK`, or else the only folder in the workspace's `resourcepacks/`.
    let packs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resourcepacks");
    let only_folder = || {
        let mut dirs = std::fs::read_dir(&packs)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir());
        let only = dirs.next()?;
        dirs.next().is_none().then_some(only)
    };
    let Some(root) = std::env::var_os("MC_REFERENCE_PACK")
        .map(std::path::PathBuf::from)
        .or_else(only_folder)
    else {
        eprintln!("SKIP: no reference pack (MC_REFERENCE_PACK, or one folder in resourcepacks)");
        return;
    };
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).expect("open jar")));
    rm.push_pack(Box::new(
        FolderResourcePack::open(&root).expect("open pack"),
    ));
    // (The vanilla bake of the same stack, measured once for the report: 512x512, about half the
    // time. Off baking no table is covered by mc-assets' bake tests.)
    let started = Instant::now();
    let (assets, report) = mc_assets::model::bake_all_with_ctm(&rm, 16384, 4, true).expect("bake");
    let ctm_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert!(report.is_clean());
    let table = assets.ctm().expect("table");
    println!(
        "user pack: CTM atlas {}x{} in {ctm_ms:.0} ms; {} rules, {} CTM states",
        assets.atlas.w,
        assets.atlas.h,
        table.rule_count(),
        table.slot_count(),
    );
    assert_eq!((assets.atlas.w, assets.atlas.h), (4096, 2048));
    assert_eq!(
        table.rule_count(),
        34,
        "17 active + glass silver + 16 block rules"
    );
    let glass = table.slot(state("glass", 0)).expect("glass has CTM");
    let rule = table
        .rule(glass.entries()[0].rule_by_face[Facing::South.index() as usize])
        .expect("glass south rule");
    assert_eq!(
        rule.source,
        "minecraft:mcpatcher/ctm/glass/Normal/glass.properties"
    );
    let silver = table
        .slot(state("stained_glass", 8))
        .expect("light gray glass");
    let rule = table
        .rule(silver.entries()[0].rule_by_face[Facing::Up.index() as usize])
        .expect("silver rule");
    assert_eq!(
        rule.source,
        "minecraft:mcpatcher/ctm/glass light gray/glass_silver.properties"
    );
    // Every pane state has its edge entry; stained pane edges reach the block rules.
    let black_pane = table
        .slot(state("stained_glass_pane", 15))
        .expect("black pane");
    let edge = black_pane
        .entries()
        .iter()
        .find(|e| e.pane_edge)
        .expect("edge entry");
    let rule = table
        .rule(edge.rule_by_face[Facing::Up.index() as usize])
        .expect("edge block rule");
    assert_eq!(
        rule.source,
        "minecraft:mcpatcher/ctm/glass_stained/glass_black/glass_black.properties"
    );
}

/// Meshing cost, Off against Fancy/Fast. Run explicitly (`-- --ignored --nocapture`), ideally
/// with `--profile dev` so the mesher is optimised as in the client.
#[test]
#[ignore = "timing measurement, run explicitly"]
fn meshing_cost_measurement() {
    if !install() {
        return;
    }
    let section = ChunkSectionPos::new(0, 4, 0);
    let glass = state("glass", 0);
    let pane = state("glass_pane", 0);
    let stone = state("stone", 0);
    // Glass-heavy: a SkyWars cage field, every other column a glass pillar or pane wall.
    let mut glassy = fixture::blank_world(1);
    for x in -1..17i32 {
        for z in -1..17i32 {
            for y in 64..80 {
                let s = match (x.rem_euclid(4), z.rem_euclid(4)) {
                    (0, _) | (_, 0) => glass,
                    (2, 2) => pane,
                    _ if y == 64 => stone,
                    _ => continue,
                };
                glassy.set_block(BlockPos::new(x, y, z), s);
            }
        }
    }
    // No glass at all: what most sections look like.
    let mut plain = fixture::blank_world(1);
    let dirt = state("dirt", 0);
    let grass = state("grass", 0);
    for x in -1..17 {
        for z in -1..17 {
            for y in 64..72 {
                let s = if y < 68 {
                    stone
                } else if y < 71 {
                    dirt
                } else {
                    grass
                };
                plain.set_block(BlockPos::new(x, y, z), s);
            }
        }
    }
    let time = |w: &World, mode: CtmMode| {
        let mut samples = Vec::new();
        for _ in 0..41 {
            let started = Instant::now();
            let mesh = mesh_section_with(w, section, settings(mode));
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(mesh);
        }
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    for (name, w) in [("glass-heavy", &glassy), ("no glass", &plain)] {
        let off = time(w, CtmMode::Off);
        let fast = time(w, CtmMode::Fast);
        let fancy = time(w, CtmMode::Fancy);
        println!(
            "{name}: median ms/section  off {off:.3}  fast {fast:.3} ({:+.1}%)  fancy {fancy:.3} ({:+.1}%)",
            (fast / off - 1.0) * 100.0,
            (fancy / off - 1.0) * 100.0
        );
    }
}
