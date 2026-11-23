//! The CPU half of the menu scene: bake vanilla 1.8.9 privately, build the greenhouse as a
//! `mc_world::World`, mesh it with the real terrain mesher, and turn the quads into the scene's
//! own vertex format with a material per quad.
//!
//! Done **once per process** and cached ([`prepared`]), on a background thread when
//! [`super::prewarm`] was called. Nothing here touches the game's registries: the bake is
//! [`mc_assets::model::bake_all_unpublished`], and the meshing runs inside
//! [`mc_assets::model::with_thread_registry`] and [`crate::terrain::tint::with_thread_colormaps`]
//! on this thread only.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_data::BlockState;
use mc_math::{ChunkPos, ChunkSectionPos};

use super::greenhouse::{self, Fall, Voxels, FLOOR, VOX_MAX, VOX_MIN};
use super::MenuSceneError;
use crate::terrain::cull_face::MeshSettings;
use crate::terrain::mesher::mesh_section_with;
use crate::terrain::tint::{self, Colormaps};
use crate::terrain::vertex::TerrainVertex;

/// The biome every column gets. Jungle (21): the lushest grass and foliage tints in 1.8.9.
const BIOME: u8 = 21;

/// What each quad is made of; drives the shader's lighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Material {
    /// Anything opaque without a special case.
    Solid = 0,
    /// The white quartz structure.
    Quartz = 1,
    /// Leaves: translucent, sway.
    Leaves = 2,
    /// Cross plants and flowers: translucent, sway from the root.
    Plant = 3,
    /// Vines.
    Vine = 4,
    /// Glass blocks and panes: the frame is drawn opaque, the inside by the glass pass.
    Glass = 5,
    /// Sea lanterns: emissive.
    Light = 6,
    /// The grass block and dirt.
    Ground = 7,
    /// A glass pane's edge strip (`glass_pane_top`): drawn as frame, no glass pass.
    GlassEdge = 8,
}

/// The scene's vertex: 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SceneVertex {
    /// Scene-space position (the greenhouse spans about 50 x 40 x 90 blocks near the origin, so
    /// `f32` is exact enough without camera-relative tricks).
    pub(super) pos: [f32; 3],
    /// Atlas UV.
    pub(super) uv: [f32; 2],
    /// Biome tint x ambient occlusion, with vanilla's baked face shade divided back out where
    /// the model had one (the shader lights faces itself). Alpha unused.
    pub(super) color: [u8; 4],
    /// Quad normal (snorm), `w` unused.
    pub(super) normal: [i8; 4],
    /// `x` = [`Material`], `y` = wind weight (0..255), `z` = sky visibility (0..255).
    pub(super) extra: [u8; 4],
}

impl SceneVertex {
    pub(super) const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Unorm8x4,
        3 => Snorm8x4,
        4 => Uint8x4,
    ];

    pub(super) const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// One water vertex: falls, pools and mist all share it. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct WaterVertex {
    /// `xyz` = position (mist: the billboard centre), `w` = kind (0 fall, 1 pool, 2 mist).
    pub(super) pos: [f32; 4],
    /// `xyz` = face normal (fall faces), `w` = per-fall seed.
    pub(super) normal: [f32; 4],
    /// Fall: `x` across the face 0..1, `y` = the fall's top, `z` = the pool surface, `w` = the
    /// face width. Pool: `xy` = the pool centre (x, z), `z` = radius, `w` = surface y.
    /// Mist: `xy` = corner (-1..1), `z` = size, `w` = the pool surface y.
    pub(super) params: [f32; 4],
}

impl WaterVertex {
    pub(super) const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Float32x4,
    ];

    pub(super) const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WaterVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Quad ranges (in quads) into [`PreparedScene::vertices`].
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Ranges {
    /// Opaque quads.
    pub(super) opaque: (u32, u32),
    /// Alpha-tested quads, glass included (its frame is drawn here).
    pub(super) cutout: (u32, u32),
    /// The glass quads alone, a tail of `cutout`, drawn again by the glass pass.
    pub(super) glass: (u32, u32),
}

/// Measurements for the report.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrepareStats {
    /// The private vanilla bake, in milliseconds.
    pub bake_ms: f64,
    /// Building the greenhouse world and meshing it, in milliseconds.
    pub mesh_ms: f64,
    /// Total block quads.
    pub quads: u32,
    /// The vanilla atlas, width x height.
    pub atlas_size: (u32, u32),
    /// The atlas's GPU size including its mip chain, in bytes.
    pub atlas_bytes: u64,
}

/// Everything the GPU side needs, built once.
pub(super) struct PreparedScene {
    pub(super) atlas: &'static mc_assets::Atlas,
    pub(super) vertices: Vec<SceneVertex>,
    pub(super) ranges: Ranges,
    pub(super) water: Vec<WaterVertex>,
    /// Water quads (4 vertices each, same quad index pattern): falls then pools, then mist.
    pub(super) water_quads: u32,
    pub(super) mist_first_quad: u32,
    /// The interior volume (min, max) for the haze and light shafts.
    pub(super) interior: ([f32; 3], [f32; 3]),
    pub(super) stats: PrepareStats,
}

/// The private vanilla bake and colormaps. `'static` because the per-thread overrides hand out
/// `&'static` references, exactly as the installed registry does.
struct Vanilla {
    assets: mc_assets::BakedAssets,
    colormaps: Colormaps,
}

static VANILLA: OnceLock<Vanilla> = OnceLock::new();
static PREPARED: OnceLock<Result<PreparedScene, MenuSceneError>> = OnceLock::new();
static PREWARM: OnceLock<()> = OnceLock::new();

/// Start [`prepared`] on a background thread. Later calls are no-ops.
pub(super) fn prewarm(jar: &Path) {
    let jar = jar.to_path_buf();
    PREWARM.get_or_init(move || {
        let spawned = std::thread::Builder::new()
            .name("menu-scene-prewarm".to_owned())
            .spawn(move || {
                let _ = prepared(&jar);
            });
        if let Err(error) = spawned {
            tracing::warn!(%error, "menu scene: could not start the prewarm thread");
        }
    });
}

/// Whether [`prepared`] has finished (successfully or not), so it would not block. Never blocks.
pub(super) fn is_prepared() -> bool {
    PREPARED.get().is_some()
}

/// The scene, prepared once per process. Blocks while another thread is preparing it.
///
/// A panic anywhere in the private bake or the meshing becomes an error, so the title screen
/// falls back to its plain background instead of taking the client down.
pub(super) fn prepared(jar: &Path) -> Result<&'static PreparedScene, MenuSceneError> {
    PREPARED
        .get_or_init(|| {
            let jar = jar.to_path_buf();
            // Recovered here, so not the game's crash report either.
            mc_core::panic_scope::contain(|| prepare(jar))
                .unwrap_or_else(|_| Err(MenuSceneError::Bake("preparation panicked".to_owned())))
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn prepare(jar: PathBuf) -> Result<PreparedScene, MenuSceneError> {
    let started = Instant::now();
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).map_err(|e| {
        MenuSceneError::Jar(format!("{}: {e}", jar.display()))
    })?));
    // The atlas cap only bounds the stitch; vanilla's sprites fit 1024 x 512 at 16x.
    let (assets, report) = mc_assets::model::bake_all_unpublished(&rm, 4096, 4)
        .map_err(|e| MenuSceneError::Bake(e.to_string()))?;
    if !report.is_clean() {
        tracing::warn!(
            failures = report.failures.len(),
            "menu scene: the vanilla bake was not clean"
        );
    }
    let colormaps = tint::read_colormaps(&rm).map_err(|e| MenuSceneError::Bake(e.to_string()))?;
    let _ = VANILLA.set(Vanilla { assets, colormaps });
    let vanilla = VANILLA
        .get()
        .expect("set just above, or by an earlier prepare");
    let bake_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let scene = greenhouse::build();
    let world = to_world(&scene.voxels);
    let sky = sky_visibility(&scene.voxels);
    let mut built = mc_assets::model::with_thread_registry(&vanilla.assets, || {
        tint::with_thread_colormaps(&vanilla.colormaps, || mesh_all(&world, &scene.voxels, &sky))
    });
    let mesh_ms = started.elapsed().as_secs_f64() * 1000.0;

    let (water, water_quads, mist_first_quad) = water_geometry(&scene.falls);
    let atlas = &vanilla.assets.atlas;
    let atlas_bytes = std::iter::once(atlas.texture.len() as u64)
        .chain(atlas.mipmaps.iter().map(|m| m.len() as u64))
        .sum();
    built.stats = PrepareStats {
        bake_ms,
        mesh_ms,
        quads: (built.vertices.len() / 4) as u32,
        atlas_size: (atlas.w, atlas.h),
        atlas_bytes,
    };
    tracing::info!(
        bake_ms,
        mesh_ms,
        quads = built.stats.quads,
        atlas = ?(atlas.w, atlas.h),
        "menu scene prepared"
    );
    Ok(PreparedScene {
        atlas,
        vertices: built.vertices,
        ranges: built.ranges,
        water,
        water_quads,
        mist_first_quad,
        interior: (
            [
                greenhouse::X_MIN as f32 - 1.0,
                FLOOR as f32 - 1.0,
                greenhouse::Z_MIN as f32 - 8.0,
            ],
            [
                greenhouse::X_MAX as f32 + 2.0,
                VOX_MAX[1] as f32,
                greenhouse::Z_MAX as f32 + 2.0,
            ],
        ),
        stats: built.stats,
    })
}

/// The voxels as loaded chunk columns: full sky light, no block light, one biome.
fn to_world(v: &Voxels) -> mc_world::World {
    let mut world = mc_world::World::new();
    world.set_has_sky_light(true);
    let (c0x, c1x) = (VOX_MIN[0].div_euclid(16), VOX_MAX[0].div_euclid(16));
    let (c0z, c1z) = (VOX_MIN[2].div_euclid(16), VOX_MAX[2].div_euclid(16));
    for cz in c0z - 1..=c1z + 1 {
        for cx in c0x - 1..=c1x + 1 {
            let mut data = vec![0u8; 16 * 4096 * 2 + 16 * 2048];
            data.resize(data.len() + 16 * 2048, 0xFF);
            data.resize(data.len() + 256, BIOME);
            for sy in 0..16 {
                for y in 0..16 {
                    for z in 0..16 {
                        for x in 0..16 {
                            let s = v.get(cx * 16 + x, sy * 16 + y, cz * 16 + z);
                            if s == BlockState::AIR {
                                continue;
                            }
                            let i = (sy * 4096 + ((y << 8) | (z << 4) | x)) as usize * 2;
                            data[i..i + 2].copy_from_slice(&s.id.to_le_bytes());
                        }
                    }
                }
            }
            world
                .apply_chunk_data(ChunkPos::new(cx, cz), true, 0xFFFF, &data)
                .expect("a full-column payload of the documented length");
        }
    }
    world
}

/// How much open sky each cell sees straight up, softened sideways: a cheap skylight
/// occlusion for the ambient term, so the ground under a canopy is darker than an open lawn.
struct SkyField {
    values: Vec<f32>,
}

const FX: i32 = VOX_MAX[0] - VOX_MIN[0] + 1;
const FY: i32 = VOX_MAX[1] - VOX_MIN[1] + 1;
const FZ: i32 = VOX_MAX[2] - VOX_MIN[2] + 1;

impl SkyField {
    fn at(&self, x: i32, y: i32, z: i32) -> f32 {
        let (lx, ly, lz) = (x - VOX_MIN[0], y - VOX_MIN[1], z - VOX_MIN[2]);
        if lx < 0 || lz < 0 || lx >= FX || lz >= FZ || ly >= FY {
            return 1.0;
        }
        if ly < 0 {
            return 0.0;
        }
        self.values[((ly * FZ + lz) * FX + lx) as usize]
    }

    /// Trilinear, with cell values at cell centres.
    fn sample(&self, p: [f32; 3]) -> f32 {
        let (fx, fy, fz) = (p[0] - 0.5, p[1] - 0.5, p[2] - 0.5);
        let (x0, y0, z0) = (fx.floor() as i32, fy.floor() as i32, fz.floor() as i32);
        let (tx, ty, tz) = (fx - fx.floor(), fy - fy.floor(), fz - fz.floor());
        let mut sum = 0.0;
        for (dz, wz) in [(0, 1.0 - tz), (1, tz)] {
            for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                    sum += wx * wy * wz * self.at(x0 + dx, y0 + dy, z0 + dz);
                }
            }
        }
        sum
    }
}

fn opacity(s: BlockState) -> f32 {
    if s == BlockState::AIR {
        return 0.0;
    }
    match mc_data::block_info(s.block()).name {
        "minecraft:glass" | "minecraft:glass_pane" => 0.02,
        "minecraft:leaves" | "minecraft:leaves2" => 0.2,
        "minecraft:vine" => 0.08,
        "minecraft:tallgrass"
        | "minecraft:double_plant"
        | "minecraft:red_flower"
        | "minecraft:yellow_flower" => 0.03,
        _ => 0.85,
    }
}

fn sky_visibility(v: &Voxels) -> SkyField {
    let mut values = vec![0.0f32; (FX * FY * FZ) as usize];
    let idx = |lx: i32, ly: i32, lz: i32| ((ly * FZ + lz) * FX + lx) as usize;
    for lz in 0..FZ {
        for lx in 0..FX {
            let mut t = 1.0f32;
            for ly in (0..FY).rev() {
                values[idx(lx, ly, lz)] = t;
                t *= 1.0 - opacity(v.get(lx + VOX_MIN[0], ly + VOX_MIN[1], lz + VOX_MIN[2]));
            }
        }
    }
    // Soften sideways: two box blurs of radius 3 per layer.
    let mut tmp = values.clone();
    for _ in 0..2 {
        for ly in 0..FY {
            for lz in 0..FZ {
                for lx in 0..FX {
                    let mut s = 0.0;
                    let mut n = 0.0;
                    for k in -3..=3 {
                        let x = (lx + k).clamp(0, FX - 1);
                        s += values[idx(x, ly, lz)];
                        n += 1.0;
                    }
                    tmp[idx(lx, ly, lz)] = s / n;
                }
            }
            for lz in 0..FZ {
                for lx in 0..FX {
                    let mut s = 0.0;
                    let mut n = 0.0;
                    for k in -3..=3 {
                        let z = (lz + k).clamp(0, FZ - 1);
                        s += tmp[idx(lx, ly, z)];
                        n += 1.0;
                    }
                    values[idx(lx, ly, lz)] = s / n;
                }
            }
        }
    }
    SkyField { values }
}

/// Sprite rectangles that pick a quad's material.
struct SpriteTable {
    rects: Vec<(mc_assets::sprite::UvRect, Material)>,
}

impl SpriteTable {
    /// Must run inside `with_thread_registry`, so [`mc_assets::model::sprite`] reads the private
    /// vanilla bake.
    fn new() -> Self {
        let mut rects = Vec::new();
        let mut add = |name: &str, material: Material| {
            let loc = ResourceLocation::minecraft(format!("textures/blocks/{name}"));
            rects.push((mc_assets::model::sprite(&loc).uv, material));
        };
        add("glass", Material::Glass);
        add("glass_pane_top", Material::GlassEdge);
        add("vine", Material::Vine);
        for name in [
            "leaves_oak",
            "leaves_spruce",
            "leaves_birch",
            "leaves_jungle",
            "leaves_acacia",
            "leaves_big_oak",
        ] {
            add(name, Material::Leaves);
        }
        for name in [
            "quartz_block_top",
            "quartz_block_side",
            "quartz_block_bottom",
            "quartz_block_chiseled",
            "quartz_block_chiseled_top",
            "quartz_block_lines",
            "quartz_block_lines_top",
        ] {
            add(name, Material::Quartz);
        }
        for name in ["sea_lantern", "glowstone"] {
            add(name, Material::Light);
        }
        for name in ["grass_top", "grass_side", "grass_side_overlay", "dirt"] {
            add(name, Material::Ground);
        }
        Self { rects }
    }

    fn material(&self, uv: [f32; 2], fallback: Material) -> Material {
        self.rects
            .iter()
            .find(|(r, _)| {
                uv[0] >= r.min_u && uv[0] <= r.max_u && uv[1] >= r.min_v && uv[1] <= r.max_v
            })
            .map_or(fallback, |(_, m)| *m)
    }
}

struct Built {
    vertices: Vec<SceneVertex>,
    ranges: Ranges,
    stats: PrepareStats,
}

/// Vanilla's baked face brightness (`FaceBakery.getFaceShadeColor`) for an axis normal, or
/// `None` for a quad that is not axis aligned.
fn face_shade(n: [f32; 3]) -> Option<f32> {
    if n[1] > 0.99 {
        Some(1.0)
    } else if n[1] < -0.99 {
        Some(0.5)
    } else if n[2].abs() > 0.99 {
        Some(0.8)
    } else if n[0].abs() > 0.99 {
        Some(0.6)
    } else {
        None
    }
}

fn mesh_all(world: &mc_world::World, v: &Voxels, sky: &SkyField) -> Built {
    let table = SpriteTable::new();
    let mut opaque = Vec::new();
    let mut cutout = Vec::new();
    let mut glass = Vec::new();
    let settings = MeshSettings::default();
    let (c0x, c1x) = (VOX_MIN[0].div_euclid(16), VOX_MAX[0].div_euclid(16));
    let (c0z, c1z) = (VOX_MIN[2].div_euclid(16), VOX_MAX[2].div_euclid(16));
    for sy in 0..=VOX_MAX[1].div_euclid(16) {
        for cz in c0z..=c1z {
            for cx in c0x..=c1x {
                if !v.section_occupied(cx, sy, cz) {
                    continue;
                }
                let mesh = mesh_section_with(world, ChunkSectionPos::new(cx, sy, cz), settings);
                let origin = [(cx * 16) as f32, (sy * 16) as f32, (cz * 16) as f32];
                for quad in mesh.solid.vertices().chunks_exact(4) {
                    push_quad(&table, sky, quad, origin, Material::Solid, &mut opaque);
                }
                for buffer in [&mesh.cutout_mipped, &mesh.cutout, &mesh.translucent] {
                    for quad in buffer.vertices().chunks_exact(4) {
                        let before = cutout.len();
                        push_quad(&table, sky, quad, origin, Material::Plant, &mut cutout);
                        if cutout.len() > before && cutout[before].extra[0] == Material::Glass as u8
                        {
                            let q: Vec<SceneVertex> = cutout.drain(before..).collect();
                            glass.extend(q);
                        }
                    }
                }
            }
        }
    }
    let q = |n: usize| (n / 4) as u32;
    let opaque_quads = q(opaque.len());
    let plain_cutout = q(cutout.len());
    let glass_quads = q(glass.len());
    let mut vertices = opaque;
    // These vertices stay in memory for the life of the process (the GPU side
    // is rebuilt from them after `release_gpu_memory`): grow once to the exact total instead of
    // leaving `extend`'s doubling slack (about 10 MB) behind. Same bytes either way.
    if crate::w21_mr::trim_enabled() {
        vertices.reserve_exact(cutout.len() + glass.len());
    }
    vertices.extend(cutout);
    vertices.extend(glass);
    Built {
        vertices,
        ranges: Ranges {
            opaque: (0, opaque_quads),
            cutout: (opaque_quads, opaque_quads + plain_cutout + glass_quads),
            glass: (
                opaque_quads + plain_cutout,
                opaque_quads + plain_cutout + glass_quads,
            ),
        },
        stats: PrepareStats::default(),
    }
}

fn push_quad(
    table: &SpriteTable,
    sky: &SkyField,
    quad: &[TerrainVertex],
    origin: [f32; 3],
    fallback: Material,
    out: &mut Vec<SceneVertex>,
) {
    let p: Vec<[f32; 3]> = quad
        .iter()
        .map(|v| {
            [
                v.pos[0] + origin[0],
                v.pos[1] + origin[1],
                v.pos[2] + origin[2],
            ]
        })
        .collect();
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
    let mut n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-8 {
        return;
    }
    n = [n[0] / len, n[1] / len, n[2] / len];
    let uv_mid = [
        quad.iter().map(|v| v.uv[0]).sum::<f32>() * 0.25,
        quad.iter().map(|v| v.uv[1]).sum::<f32>() * 0.25,
    ];
    let material = table.material(uv_mid, fallback);
    // Vanilla baked the face shade into the colour for shaded models; this shader lights faces
    // itself, so divide it back out. Plants and vines are `shade: false` in 1.8.9.
    let unshade = match material {
        Material::Plant | Material::Vine => None,
        _ => face_shade(n),
    };
    for (v, pos) in quad.iter().zip(p.iter()) {
        let mut color = v.color;
        if let Some(shade) = unshade {
            for c in &mut color[..3] {
                *c = (f32::from(*c) / shade).round().min(255.0) as u8;
            }
        }
        let wind = match material {
            Material::Leaves => 1.0,
            Material::Vine => 0.6,
            Material::Plant => ((pos[1] - FLOOR as f32) / 2.0).clamp(0.0, 1.0).powf(1.3),
            _ => 0.0,
        };
        let probe = [
            pos[0] + n[0] * 0.6,
            pos[1] + n[1] * 0.6,
            pos[2] + n[2] * 0.6,
        ];
        let vis = sky.sample(probe).clamp(0.0, 1.0);
        out.push(SceneVertex {
            pos: *pos,
            uv: v.uv,
            color,
            normal: [
                (n[0] * 127.0).round() as i8,
                (n[1] * 127.0).round() as i8,
                (n[2] * 127.0).round() as i8,
                0,
            ],
            extra: [material as u8, (wind * 255.0) as u8, (vis * 255.0) as u8, 0],
        });
    }
}

/// The waterfalls' columns, the pools' surfaces and the mist billboards.
fn water_geometry(falls: &[Fall]) -> (Vec<WaterVertex>, u32, u32) {
    let mut out = Vec::new();
    let mut quads = 0u32;
    let v = |pos: [f32; 3], kind: f32, normal: [f32; 3], seed: f32, params: [f32; 4]| WaterVertex {
        pos: [pos[0], pos[1], pos[2], kind],
        normal: [normal[0], normal[1], normal[2], seed],
        params,
    };
    for f in falls {
        let (x0, x1, z0, z1) = (f.x - f.half, f.x + f.half, f.z - f.half, f.z + f.half);
        let (top, bottom) = (f.top, f.pool_y - 0.05);
        let w = f.half * 2.0;
        // Four faces, counter-clockwise seen from outside.
        let faces: [([f32; 3], [[f32; 2]; 2]); 4] = [
            ([0.0, 0.0, -1.0], [[x1, z0], [x0, z0]]),
            ([0.0, 0.0, 1.0], [[x0, z1], [x1, z1]]),
            ([-1.0, 0.0, 0.0], [[x0, z0], [x0, z1]]),
            ([1.0, 0.0, 0.0], [[x1, z1], [x1, z0]]),
        ];
        for (n, [a, b]) in faces {
            let p = |c: [f32; 2], y: f32, u: f32| {
                v([c[0], y, c[1]], 0.0, n, f.seed, [u, f.top, f.pool_y, w])
            };
            out.push(p(a, bottom, 0.0));
            out.push(p(b, bottom, 1.0));
            out.push(p(b, top, 1.0));
            out.push(p(a, top, 0.0));
            quads += 1;
        }
    }
    for f in falls {
        let r = f.pool_r + 0.6;
        let y = f.pool_y;
        let params = [f.x, f.z, f.pool_r, y];
        let n = [0.0, 1.0, 0.0];
        out.push(v([f.x - r, y, f.z + r], 1.0, n, f.seed, params));
        out.push(v([f.x + r, y, f.z + r], 1.0, n, f.seed, params));
        out.push(v([f.x + r, y, f.z - r], 1.0, n, f.seed, params));
        out.push(v([f.x - r, y, f.z - r], 1.0, n, f.seed, params));
        quads += 1;
    }
    let mist_first = quads;
    for f in falls {
        for i in 0..5 {
            let a = i as f32 * 1.256 + f.seed;
            let c = [
                f.x + a.cos() * 0.9,
                f.pool_y + 0.6 + i as f32 * 0.35,
                f.z + a.sin() * 0.9,
            ];
            let size = 2.2 + i as f32 * 0.5;
            let n = [0.0, 1.0, 0.0];
            for corner in [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] {
                out.push(v(
                    c,
                    2.0,
                    n,
                    f.seed + i as f32 * 3.1,
                    [corner[0], corner[1], size, f.pool_y],
                ));
            }
            quads += 1;
        }
    }
    (out, quads, mist_first)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_greenhouse_world_reads_back_the_voxels() {
        let scene = greenhouse::build();
        let world = to_world(&scene.voxels);
        let mut checked = 0;
        for (x, y, z) in [
            (0, FLOOR - 1, 0),
            (21, FLOOR, 4),
            (-1, FLOOR + 3, 30),
            (23, 37, 40),
        ] {
            assert_eq!(
                world.block_at(mc_math::BlockPos::new(x, y, z)),
                scene.voxels.get(x, y, z),
                "({x}, {y}, {z})"
            );
            checked += 1;
        }
        assert_eq!(checked, 4);
        assert_ne!(
            scene.voxels.get(0, FLOOR - 1, 0),
            BlockState::AIR,
            "grass floor"
        );
    }

    #[test]
    fn the_greenhouse_is_deterministic() {
        let a = greenhouse::build();
        let b = greenhouse::build();
        let mut same = true;
        for y in VOX_MIN[1]..=VOX_MAX[1] {
            for z in VOX_MIN[2]..=VOX_MAX[2] {
                for x in VOX_MIN[0]..=VOX_MAX[0] {
                    same &= a.voxels.get(x, y, z) == b.voxels.get(x, y, z);
                }
            }
        }
        assert!(same);
        assert_eq!(a.falls.len(), 2);
    }
}
