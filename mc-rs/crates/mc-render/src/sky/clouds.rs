//! Clouds — fast (flat quad grid) and fancy (3D box grid).
//!
//! java: `RenderGlobal.renderClouds` (dispatcher + fast path, line 1420),
//! `renderCloudsFancy` (line 1494).
//!
//! # Relationship to vanilla
//!
//! Both paths are eye-relative, which is what vanilla is too: `renderClouds` emits a grid running
//! `-256..256` with no camera term in X/Z (`RenderGlobal.java:1467`), and `renderCloudsFancy`
//! emits `j1 * 8 - frac(d1)` (`:1577`) — in both cases the mesh is centred on the player and the
//! *world* anchoring is carried by the texture offset (`f8`/`f9`, `f17`/`f18`) plus, for fancy
//! clouds, the sub-cell fraction `f19`/`f20`. This port reproduces that, so no [`crate::Camera`]
//! relative-position juggling is needed here.
//!
//! The fast path was already a faithful transcription. The fancy one was
//! not: it mapped the whole 256x256 `clouds.png` across every 96-block cell instead of the 8x8
//! texel patch vanilla reads, drew one wall per cell boundary instead of vanilla's eight
//! per-cloud-unit walls, and dropped the world anchoring, the `f3` above/below gates and the
//! interior-wall culling. It has been rewritten term for term.
//!
//! # Where the clouds draw, and the depth they leave
//!
//! java: `EntityRenderer.renderWorldPass` (`:1364`, `:1474`) draws the
//! clouds at one of two points, picked by the render-view entity's eye height this tick
//! (`posY + getEyeHeight()`, not interpolated) against a literal `128.0`, not the cloud height:
//!
//! * **below 128**, right after the sky and before any terrain;
//! * **at or above 128**, after translucent terrain and the weather, last before the hand.
//!
//! Both times with depth writes on and the ambient `GL_LEQUAL` test, and the depth stays in the
//! frame's depth buffer. Below, terrain behind a cloud is depth-rejected (the cloud hides it and
//! the sky shows through the cloud's 20 % transparency) while terrain in front of a cloud draws
//! over it. Above, the clouds blend over the ground beneath them, and anything rising in front
//! of them hides them. [`CloudsPhase::install`] registers the clouds at both points
//! ([`crate::PhaseOrder::SKY`] and [`ABOVE_CLOUDS_ORDER`]), and [`CloudsPhase::view_eye_y`]
//! enables one of the two each frame.
//!
//! The depth written is the clouds' own projection's, as in vanilla: `renderCloudsCheck` widens
//! the far plane to `farPlaneDistance * 4` around the draw, where terrain uses
//! `farPlaneDistance * sqrt(2)` (near 0.05 for both). So a terrain fragment at eye-space depth
//! `z_t` passes against a cloud at `z_c` only when, to first order in the near plane,
//! `1/z_t >= 1/z_c + 1/far_terrain - 1/far_clouds`: terrain a little in front of a cloud can still
//! lose to it, by vanilla's own margin.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{DVec3, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::pipeline::{FrameClearHandoff, PassTarget, PhaseList, PhaseOrder};
use crate::sky::environment::Environment;
use crate::sky::fog::{FogDistance, FogPhase, FogState};
use crate::sky::textures::{self, SkyTexture};
use crate::{Camera, Gpu, RenderPhase};

/// java: `WorldProvider.getCloudHeight` (Overworld default).
pub const DEFAULT_CLOUD_HEIGHT: f32 = 128.0;

/// java: `EntityRenderer.renderWorldPass`'s `128.0D`: the eye height (`posY + getEyeHeight()`)
/// at and above which the clouds draw after the world instead of before it. A literal, not the
/// cloud height.
pub const ABOVE_CLOUDS_EYE_Y: f64 = 128.0;

/// Where [`CloudsPhase::install`] puts the clouds' second registration, the one that draws while
/// the eye is at or above [`ABOVE_CLOUDS_EYE_Y`]. java's `aboveClouds` step comes after
/// translucent terrain and the weather, last before the hand. Here it comes after the
/// screen-space fog wherever that is registered ([`FogPhase::ORDER`] at the latest, so these
/// clouds, which fog themselves, are never fogged twice) and before the anti-aliasing
/// ([`crate::PhaseOrder::TEMPORAL_AA`]), which resolves them with the rest of the world. The late
/// volumetric clouds use the same order; the two never draw in the same frame.
pub const ABOVE_CLOUDS_ORDER: PhaseOrder = PhaseOrder(FogPhase::ORDER.0 + 2);

/// [`RenderPhase::name`] of the registration at [`ABOVE_CLOUDS_ORDER`].
pub const ABOVE_CLOUDS_PHASE_NAME: &str = "sky/clouds-above";

/// [`RenderPhase::name`] of a [`CloudsPhase`], and of its registration at
/// [`crate::PhaseOrder::SKY`].
const PHASE_NAME: &str = "sky/clouds";

/// Whether an eye at `view_eye_y` puts the clouds after the world. java: `renderWorldPass`'s
/// `entity.posY + (double)entity.getEyeHeight() >= 128.0D`. `None` (no render-view entity fed)
/// counts as below.
#[inline]
fn eye_above_clouds(view_eye_y: Option<f64>) -> bool {
    view_eye_y.is_some_and(|y| y >= ABOVE_CLOUDS_EYE_Y)
}

/// Texel scale for the flat/fast cloud texture. java: `4.8828125E-4F` (`= 1.0 / 2048.0`).
const FAST_TEXEL: f32 = 1.0 / 2048.0;
/// Cell size (world blocks) fast clouds are tiled at.
const FAST_CELL: f32 = 32.0;
/// Half-span of the fast cloud grid.
const FAST_SPAN: f32 = 256.0;
/// java: `RenderGlobal.renderClouds`/`renderCloudsFancy`'s cloud-scroll multiplier.
const CLOUD_SCROLL_PER_TICK: f64 = 0.029_999_999_329_447_746;
/// java: both cloud paths fold their scrolling coordinate back into a 2048-unit period before
/// converting it to texture coordinates.
const CLOUD_COORDINATE_PERIOD: f64 = 2048.0;

/// java: `RenderGlobal.renderCloudsFancy`'s `f1` — one fancy "cloud unit" is this many blocks,
/// applied as `GlStateManager.scale(12.0F, 1.0F, 12.0F)` (`RenderGlobal.java:1545`) so it scales
/// X and Z only.
const FANCY_CELL_SCALE: f32 = 12.0;
/// Cloud-space cell size (before the ×12 world scale). java: the `j1 * 8` cell stride.
const FANCY_CELL: f32 = 8.0;
/// Box thickness, in **blocks** — the `scale` above leaves Y alone. java: `renderCloudsFancy`'s
/// `f2` (`RenderGlobal.java:1501`).
const FANCY_THICKNESS: f32 = 4.0;
/// Cells per axis: `j1`/`k1` run `-3..=4`. java: `RenderGlobal.java:1570`-`:1572`.
const FANCY_CELLS: i32 = 8;
/// Texel scale for the fancy cloud texture: one cloud unit is exactly one texel of the 256x256
/// `clouds.png`. java: `renderCloudsFancy`'s `f16` (`0.00390625F` = `1/256`).
const FANCY_TEXEL: f32 = 1.0 / 256.0;
/// java: `renderCloudsFancy`'s `f21` (`9.765625E-4F` = `1/1024`) — the sliver vanilla pulls the
/// top face and the `+X`/`+Z` walls back by so coincident faces of adjacent cells do not z-fight.
const FANCY_EPSILON: f32 = 1.0 / 1024.0;

/// `GameSettings.renderClouds`: `0` = off, `1` = fast, `2` = fancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CloudMode {
    /// Not drawn.
    Off,
    /// Flat scrolling quad grid.
    #[default]
    Fast,
    /// 3D box grid with directional face shading.
    Fancy,
}

impl CloudMode {
    /// From vanilla's `options.txt`/`GameSettings.clouds` encoding.
    #[must_use]
    pub fn from_setting(v: i32) -> Self {
        match v {
            0 => CloudMode::Off,
            1 => CloudMode::Fast,
            _ => CloudMode::Fancy,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

impl CloudVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<CloudVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4],
    };
}

/// One vertex of the resident fancy-cloud lattice. java: `renderCloudsFancy`'s geometry depends
/// on the frame only through `f3`, `f17`..`f20` and the cloud colour, so the lattice is built once
/// and the vertex shader looks every frame-dependent value up in [`FancyCloudFrame::table`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct FancyCloudVertex {
    /// The frame-independent part of the UV, `cloud_uv * FANCY_TEXEL`. Exact: the cloud-space
    /// coordinates are small multiples of `0.5`, and the scale is a power of two.
    uv: [f32; 2],
    /// `x | z << 8 | y << 16 | colour << 24`: indices into the table's X, Z, Y and colour rows.
    index: u32,
}

impl FancyCloudVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<FancyCloudVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32],
    };
}

/// Distinct X (or Z) coordinates one fancy cell row uses: `sx(f24)`, `sx(f24 + l)` for
/// `l = 1..=7`, `sx(f24 + 8)`, then the eight `+X` walls' `sx(f24 + i + 1 - 1/1024)`.
const FANCY_AXIS_TERMS: usize = 17;
/// [`FancyCloudFrame::table`] layout, in `vec4` rows (`clouds.wgsl`'s `FancyCloudTable`).
const FANCY_TABLE_AXIS_ROWS: usize = FANCY_CELLS as usize * FANCY_AXIS_TERMS / 4;
const FANCY_TABLE_Y_ROW: usize = 2 * FANCY_TABLE_AXIS_ROWS;
const FANCY_TABLE_UV_ROW: usize = FANCY_TABLE_Y_ROW + 1;
const FANCY_TABLE_COLOR_ROW: usize = FANCY_TABLE_UV_ROW + 1;
const FANCY_TABLE_ROWS: usize = FANCY_TABLE_COLOR_ROW + 4;
/// Rows of [`FancyCloudFrame::table`]'s colour block: top, bottom, X side, Z side.
const FANCY_TOP: usize = 0;
const FANCY_BOTTOM: usize = 1;
const FANCY_X_SIDE: usize = 2;
const FANCY_Z_SIDE: usize = 3;
/// Y rows: the slab's bottom, its top, and the top face pulled down by `FANCY_EPSILON`.
const FANCY_Y_BOTTOM: usize = 0;
const FANCY_Y_TOP: usize = 1;
const FANCY_Y_TOP_FACE: usize = 2;

/// `renderCloudsFancy`'s per-frame scalars, `RenderGlobal.java:1502`-`:1509`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FancyCloudFrame {
    f3: f32,
    f17: f32,
    f18: f32,
    f19: f32,
    f20: f32,
}

impl FancyCloudFrame {
    fn new(camera_pos: DVec3, cloud_height: f32, tick: f32) -> Self {
        // java: `RenderGlobal.java:1502`-`:1509`.
        let d0 = f64::from(tick);
        let mut d1 = (camera_pos.x + d0 * CLOUD_SCROLL_PER_TICK) / f64::from(FANCY_CELL_SCALE);
        let mut d2 = camera_pos.z / f64::from(FANCY_CELL_SCALE) + 0.330_000_013_113_021_85;
        let f3 = cloud_height - camera_pos.y as f32 + 0.33;
        d1 = wrap_cloud_coordinate(d1);
        d2 = wrap_cloud_coordinate(d2);
        Self {
            f3,
            f17: d1.floor() as f32 * FANCY_TEXEL,
            f18: d2.floor() as f32 * FANCY_TEXEL,
            f19: (d1 - d1.floor()) as f32,
            f20: (d2 - d2.floor()) as f32,
        }
    }

    /// The `f3 > -5` (undersides) and `f3 <= 5` (tops) gates, `RenderGlobal.java:1580`/`:1588`.
    /// The only frame inputs the lattice's *shape* depends on.
    fn face_gates(&self) -> (bool, bool) {
        (self.f3 > -5.0, self.f3 <= 5.0)
    }

    /// Every frame-dependent value the fancy vertex shader needs, each computed with exactly the
    /// `f32` expression [`fancy_cloud_vertices`] writes into its vertices, so the shader only
    /// *selects* values and the vertex stage sees bit-identical positions, UVs and colours.
    fn table(&self, color: Vec3) -> [[f32; 4]; FANCY_TABLE_ROWS] {
        let sx = |u: f32| u * FANCY_CELL_SCALE;
        let axis = |cell_origin: f32, fraction: f32, term: usize| {
            let f24 = cell_origin - fraction;
            match term {
                0 => sx(f24),
                8 => sx(f24 + 8.0),
                1..=7 => sx(f24 + term as f32),
                _ => sx(f24 + (term - 9) as f32 + 1.0 - FANCY_EPSILON),
            }
        };
        let mut table = [[0.0_f32; 4]; FANCY_TABLE_ROWS];
        for cell in 0..FANCY_CELLS as usize {
            // `j1`/`k1` run `-3..=4`; `f22`/`f23 = j1 * 8`.
            let origin = (cell as i32 - (FANCY_CELLS / 2 - 1)) as f32 * FANCY_CELL;
            for term in 0..FANCY_AXIS_TERMS {
                let slot = cell * FANCY_AXIS_TERMS + term;
                table[slot / 4][slot % 4] = axis(origin, self.f19, term);
                table[FANCY_TABLE_AXIS_ROWS + slot / 4][slot % 4] = axis(origin, self.f20, term);
            }
        }
        let f3 = self.f3;
        table[FANCY_TABLE_Y_ROW][FANCY_Y_BOTTOM] = f3;
        table[FANCY_TABLE_Y_ROW][FANCY_Y_TOP] = f3 + FANCY_THICKNESS;
        table[FANCY_TABLE_Y_ROW][FANCY_Y_TOP_FACE] = f3 + FANCY_THICKNESS - FANCY_EPSILON;
        table[FANCY_TABLE_UV_ROW] = [self.f17, self.f18, 0.0, 0.0];
        // java: `RenderGlobal.java:1528`-`:1536`.
        table[FANCY_TABLE_COLOR_ROW + FANCY_TOP] = [color.x, color.y, color.z, 0.8];
        table[FANCY_TABLE_COLOR_ROW + FANCY_BOTTOM] =
            [color.x * 0.7, color.y * 0.7, color.z * 0.7, 0.8];
        table[FANCY_TABLE_COLOR_ROW + FANCY_X_SIDE] =
            [color.x * 0.9, color.y * 0.9, color.z * 0.9, 0.8];
        table[FANCY_TABLE_COLOR_ROW + FANCY_Z_SIDE] =
            [color.x * 0.8, color.y * 0.8, color.z * 0.8, 0.8];
        table
    }
}

/// The fancy lattice in [`fancy_cloud_vertices`]' exact emission order (cell by cell: bottom,
/// top, `-X`, `+X`, `-Z`, `+Z`), so blending order between overlapping faces is unchanged. Each
/// vertex carries table indices instead of values; only the two face gates shape it, so it is
/// rebuilt only when one of them flips (crossing ±5 blocks of the cloud slab).
fn fancy_cloud_lattice(bottom: bool, top: bool) -> Vec<FancyCloudVertex> {
    const HALF: i32 = FANCY_CELLS / 2 - 1;
    let mut out = Vec::new();
    let mut quad = |x: [usize; 4], y: [usize; 4], z: [usize; 4], uv: [[f32; 2]; 4], c: usize| {
        let v = |i: usize| FancyCloudVertex {
            uv: uv[i],
            index: x[i] as u32 | (z[i] as u32) << 8 | (y[i] as u32) << 16 | (c as u32) << 24,
        };
        out.extend([v(0), v(1), v(2), v(0), v(2), v(3)]);
    };
    for j1 in -HALF..=FANCY_CELLS / 2 {
        for k1 in -HALF..=FANCY_CELLS / 2 {
            let f22 = j1 as f32 * FANCY_CELL;
            let f23 = k1 as f32 * FANCY_CELL;
            let xr = (j1 + HALF) as usize * FANCY_AXIS_TERMS;
            let zr = (k1 + HALF) as usize * FANCY_AXIS_TERMS;
            let t = |cloud: f32| cloud * FANCY_TEXEL;
            // Table terms: 0 = `f24`, 8 = `f24 + 8`, l = `f24 + l`, 9 + i = `+X` wall `i`.
            let (x0, x8, z0, z8) = (xr, xr + 8, zr, zr + 8);
            let face_uv = [
                [t(f22), t(f23 + 8.0)],
                [t(f22 + 8.0), t(f23 + 8.0)],
                [t(f22 + 8.0), t(f23)],
                [t(f22), t(f23)],
            ];
            // Bottom face. java: `:1580`-`:1586`.
            if bottom {
                let y = [FANCY_Y_BOTTOM; 4];
                quad([x0, x8, x8, x0], y, [z8, z8, z0, z0], face_uv, FANCY_BOTTOM);
            }
            // Top face. java: `:1588`-`:1594`.
            if top {
                let y = [FANCY_Y_TOP_FACE; 4];
                quad([x0, x8, x8, x0], y, [z8, z8, z0, z0], face_uv, FANCY_TOP);
            }
            let wall_y = [FANCY_Y_BOTTOM, FANCY_Y_TOP, FANCY_Y_TOP, FANCY_Y_BOTTOM];
            let x_wall_uv = |uu: f32| {
                [
                    [uu, t(f23 + 8.0)],
                    [uu, t(f23 + 8.0)],
                    [uu, t(f23)],
                    [uu, t(f23)],
                ]
            };
            let z_wall_uv = |vv: f32| {
                [
                    [t(f22), vv],
                    [t(f22 + 8.0), vv],
                    [t(f22 + 8.0), vv],
                    [t(f22), vv],
                ]
            };
            // -X walls. java: `:1596`-`:1605`.
            if j1 > -1 {
                for l1 in 0..FANCY_CELL as usize {
                    let x = xr + l1;
                    let uu = t(f22 + l1 as f32 + 0.5);
                    quad(
                        [x; 4],
                        wall_y,
                        [z8, z8, z0, z0],
                        x_wall_uv(uu),
                        FANCY_X_SIDE,
                    );
                }
            }
            // +X walls. java: `:1607`-`:1616`.
            if j1 <= 1 {
                for i2 in 0..FANCY_CELL as usize {
                    let x = xr + 9 + i2;
                    let uu = t(f22 + i2 as f32 + 0.5);
                    quad(
                        [x; 4],
                        wall_y,
                        [z8, z8, z0, z0],
                        x_wall_uv(uu),
                        FANCY_X_SIDE,
                    );
                }
            }
            // -Z walls. java: `:1618`-`:1627`.
            let z_wall_y = [FANCY_Y_TOP, FANCY_Y_TOP, FANCY_Y_BOTTOM, FANCY_Y_BOTTOM];
            if k1 > -1 {
                for j2 in 0..FANCY_CELL as usize {
                    let z = zr + j2;
                    let vv = t(f23 + j2 as f32 + 0.5);
                    quad(
                        [x0, x8, x8, x0],
                        z_wall_y,
                        [z; 4],
                        z_wall_uv(vv),
                        FANCY_Z_SIDE,
                    );
                }
            }
            // +Z walls. java: `:1629`-`:1638`.
            if k1 <= 1 {
                for k2 in 0..FANCY_CELL as usize {
                    let z = zr + 9 + k2;
                    let vv = t(f23 + k2 as f32 + 0.5);
                    quad(
                        [x0, x8, x8, x0],
                        z_wall_y,
                        [z; 4],
                        z_wall_uv(vv),
                        FANCY_Z_SIDE,
                    );
                }
            }
        }
    }
    out
}

// Fancy-cloud faces that cannot draw anything.
//
// Every fancy face samples a fixed patch of `clouds.png`: a wall one texel column (or row) and 8
// texels along the cell, a top/bottom face the cell's 8x8 texels. Both fancy pipelines discard a
// fragment whose texel fails the 0.1 alpha test, so a face whose whole patch fails it writes no
// depth (pre-pass) and no colour: drawing it only costs rasterisation and texture fetches. Which
// patch a face samples moves only with the whole-texel offsets `f17`/`f18`, so the faces worth
// drawing are listed once per offset (and gate) change, in the lattice's own emission order, and
// drawn through an index buffer over the unchanged resident lattice. The faces that remain are
// the same faces, in the same order, as before, so the image is unchanged bit for bit.
// `MC_NW_F2_CLOUD_CULL=0` draws every face, as before.

/// Whether fancy-cloud face culling is on. Read once; `MC_NW_F2_CLOUD_CULL=0` turns it off.
fn cloud_cull_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_NW_F2_CLOUD_CULL").is_none_or(|value| value != "0"))
}

/// The only `clouds.png` size culling handles: `FANCY_TEXEL` makes one cloud unit one texel of a
/// 256-wide texture. Any other size (an HD pack) keeps drawing every face.
const COVERAGE_SIZE: i32 = 256;
/// A texel whose alpha byte is at most this can never pass the clouds' alpha test: `25 / 255`
/// is already under the `0.1` cutoff, and the vertex alpha that multiplies it (0.8 for fancy
/// faces) is at most 1. The margin dwarfs any unorm-conversion or `f32` rounding.
const COVERAGE_ALPHA_FLOOR: u8 = 25;

/// Which texels of a 256x256 `clouds.png` may survive the alpha test, as bit rows (bit `c` of
/// `rows[r]`) and their transpose (bit `r` of `cols[c]`).
struct CloudCoverage {
    rows: Box<[[u64; 4]; 256]>,
    cols: Box<[[u64; 4]; 256]>,
}

impl CloudCoverage {
    /// `None` unless the texture is exactly 256x256 RGBA8.
    fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Option<Self> {
        let size = COVERAGE_SIZE as usize;
        if width as usize != size || height as usize != size || rgba.len() != size * size * 4 {
            return None;
        }
        let mut rows = Box::new([[0_u64; 4]; 256]);
        let mut cols = Box::new([[0_u64; 4]; 256]);
        for (i, pixel) in rgba.chunks_exact(4).enumerate() {
            if pixel[3] > COVERAGE_ALPHA_FLOOR {
                let (c, r) = (i % size, i / size);
                rows[r][c / 64] |= 1 << (c % 64);
                cols[c][r / 64] |= 1 << (r % 64);
            }
        }
        Some(Self { rows, cols })
    }

    fn bit(mask: &[u64; 4], i: i32) -> bool {
        let i = i.rem_euclid(COVERAGE_SIZE) as usize;
        (mask[i / 64] >> (i % 64)) & 1 != 0
    }

    /// Whether any texel of the `cols` x `rows` rectangle at `(col0, row0)` may survive the alpha
    /// test. Coordinates wrap, as the clouds' `Repeat` sampler does.
    fn any(&self, col0: i32, cols: i32, row0: i32, rows: i32) -> bool {
        if cols == 1 {
            let mask = &self.cols[col0.rem_euclid(COVERAGE_SIZE) as usize];
            return (row0..row0 + rows).any(|r| Self::bit(mask, r));
        }
        (row0..row0 + rows).any(|r| {
            let mask = &self.rows[r.rem_euclid(COVERAGE_SIZE) as usize];
            (col0..col0 + cols).any(|c| Self::bit(mask, c))
        })
    }
}

/// The `clouds.png` texels one fancy face can sample, in cloud units before the frame's
/// whole-texel offset (`f17`/`f18` x 256) is added.
///
/// A face's UVs span `[f22, f22 + 8]` (or `[f23, f23 + 8]`) along the cell; a fragment can land
/// on either end texel and perspective interpolation can stray a hair past an edge, so the span
/// is widened by one texel on each side (10 texels). A wall's other coordinate is a texel centre
/// (`+ 0.5`) at every vertex, so it is exactly one texel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaceFootprint {
    col0: i16,
    cols: u8,
    row0: i16,
    rows: u8,
}

/// One [`FaceFootprint`] per face of [`fancy_cloud_lattice`]`(bottom, top)`, in its emission
/// order (face `i` is lattice vertices `6 * i .. 6 * i + 6`).
fn fancy_cloud_footprints(bottom: bool, top: bool) -> Vec<FaceFootprint> {
    const HALF: i32 = FANCY_CELLS / 2 - 1;
    const CELL: i32 = FANCY_CELL as i32;
    let mut out = Vec::new();
    let span = |origin: i32| (origin - 1, CELL + 2);
    let fp = |(col0, cols): (i32, i32), (row0, rows): (i32, i32)| FaceFootprint {
        col0: col0 as i16,
        cols: cols as u8,
        row0: row0 as i16,
        rows: rows as u8,
    };
    for j1 in -HALF..=FANCY_CELLS / 2 {
        for k1 in -HALF..=FANCY_CELLS / 2 {
            let (f22, f23) = (j1 * CELL, k1 * CELL);
            if bottom {
                out.push(fp(span(f22), span(f23)));
            }
            if top {
                out.push(fp(span(f22), span(f23)));
            }
            // -X then +X walls: texel column `f22 + l`.
            for wall_side in [j1 > -1, j1 <= 1] {
                if wall_side {
                    out.extend((0..CELL).map(|l| fp((f22 + l, 1), span(f23))));
                }
            }
            // -Z then +Z walls: texel row `f23 + l`.
            for wall_side in [k1 > -1, k1 <= 1] {
                if wall_side {
                    out.extend((0..CELL).map(|l| fp(span(f22), (f23 + l, 1))));
                }
            }
        }
    }
    out
}

/// Append the lattice vertex indices of every face in `footprints` that may draw something at
/// whole-texel offset `(ox, oz)`, in emission order.
fn fancy_visible_indices(
    footprints: &[FaceFootprint],
    coverage: &CloudCoverage,
    ox: i32,
    oz: i32,
    out: &mut Vec<u16>,
) {
    for (face, fp) in footprints.iter().enumerate() {
        let col0 = i32::from(fp.col0) + ox;
        let row0 = i32::from(fp.row0) + oz;
        if coverage.any(col0, i32::from(fp.cols), row0, i32::from(fp.rows)) {
            let first = (face * 6) as u16;
            out.extend(first..first + 6);
        }
    }
}

/// The per-texture and per-offset state of fancy-cloud culling.
struct FancyCull {
    coverage: CloudCoverage,
    /// Footprints of the lattice built for `gates`.
    footprints: Vec<FaceFootprint>,
    gates: Option<(bool, bool)>,
    /// `(ox, oz)` the index list was built for.
    offset: Option<(i32, i32)>,
    indices: Vec<u16>,
    /// Built on the first fancy frame; sized for the fullest lattice.
    buffer: Option<wgpu::Buffer>,
    count: u32,
}

impl FancyCull {
    fn new(coverage: CloudCoverage) -> Self {
        Self {
            coverage,
            footprints: Vec::new(),
            gates: None,
            offset: None,
            indices: Vec::new(),
            buffer: None,
            count: 0,
        }
    }

    /// Bring the index list up to date for this frame; returns the index count to draw. Only a
    /// gate flip or a whole-texel offset change does any work.
    fn update(&mut self, gpu: &Gpu, gates: (bool, bool), frame: &FancyCloudFrame) -> u32 {
        if self.gates != Some(gates) {
            self.footprints = fancy_cloud_footprints(gates.0, gates.1);
            self.gates = Some(gates);
            self.offset = None;
        }
        // `f17`/`f18` are whole texels times 1/256, so these are exact.
        let offset = (
            (frame.f17 * COVERAGE_SIZE as f32).round() as i32,
            (frame.f18 * COVERAGE_SIZE as f32).round() as i32,
        );
        if self.offset == Some(offset) {
            return self.count;
        }
        self.offset = Some(offset);
        self.indices.clear();
        fancy_visible_indices(
            &self.footprints,
            &self.coverage,
            offset.0,
            offset.1,
            &mut self.indices,
        );
        let buffer = self.buffer.get_or_insert_with(|| {
            let faces = fancy_cloud_footprints(true, true).len() as u64;
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cloud fancy visible faces"),
                size: faces * 6 * std::mem::size_of::<u16>() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        // Six `u16`s per face: always a multiple of the 4-byte copy alignment.
        if !self.indices.is_empty() {
            gpu.write_buffer_staged(buffer, 0, bytemuck::cast_slice(&self.indices));
        }
        self.count = self.indices.len() as u32;
        self.count
    }
}

fn push_quad(out: &mut Vec<CloudVertex>, p: [Vec3; 4], uv: [Vec2; 4], color: [f32; 4]) {
    let v = |i: usize| CloudVertex {
        pos: p[i].to_array(),
        uv: uv[i].to_array(),
        color,
    };
    out.push(v(0));
    out.push(v(1));
    out.push(v(2));
    out.push(v(0));
    out.push(v(2));
    out.push(v(3));
}

/// java: `cloudTickCounter + partialTicks`, applied once by `RenderGlobal.renderClouds` before
/// either cloud path turns the result into a scroll offset.
#[inline]
fn cloud_tick_at(cloud_tick_counter: f32, partial_ticks: f32) -> f32 {
    cloud_tick_counter + partial_ticks
}

#[inline]
fn wrap_cloud_coordinate(value: f64) -> f64 {
    value - (value / CLOUD_COORDINATE_PERIOD).floor() * CLOUD_COORDINATE_PERIOD
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudFogUniform {
    /// rgb = fog colour, a = 0 for linear or 1 for exponential fog.
    color: [f32; 4],
    /// x = linear start, y = linear end, z = exponential density, w = distance metric.
    params: [f32; 4],
    /// x = Java's temporary cloud far plane. The remaining lanes are reserved for future
    /// cloud-only projection terms and keep this block std140-aligned.
    projection: [f32; 4],
}

impl Default for CloudFogUniform {
    fn default() -> Self {
        // A disabled fog state has an invalid/empty linear interval, which the shader treats as
        // fully clear. This keeps startup and tests allocation-free and leaves cloud alpha intact.
        Self {
            color: [0.0; 4],
            params: [0.0; 4],
            projection: [0.0; 4],
        }
    }
}

impl CloudFogUniform {
    fn from_state(state: Option<(FogState, FogDistance)>) -> Self {
        let Some((state, distance)) = state else {
            return Self::default();
        };
        let (params, mode) = state.mode.cloud_uniform_params(distance);
        Self {
            color: [state.color.x, state.color.y, state.color.z, mode],
            params,
            projection: [0.0; 4],
        }
    }

    fn with_cloud_far(mut self, terrain_far: f32, near: f32) -> Self {
        let java_cloud_far = terrain_far * (2.0 * std::f32::consts::SQRT_2);
        self.projection[0] = java_cloud_far.max(near + f32::EPSILON);
        self
    }
}

/// Apply the cloud-colour transform that Java performs in `RenderGlobal`, separately from the
/// integer upload transform that `TextureUtil.anaglyphColor` applies to `clouds.png` bytes.
///
/// java: `RenderGlobal.renderClouds:3122-3129` and `renderCloudsFancy:3197-3204`. The source
/// colour is a `float` `Vec3`, so this intentionally keeps the weighted arithmetic in `f32`
/// instead of routing the tint through the RGBA8 upload helper (which would quantize it and would
/// apply the texture transform twice).
#[inline]
fn cloud_color_for_render(color: Vec3, anaglyph: bool) -> Vec3 {
    if !anaglyph {
        return color;
    }
    Vec3::new(
        (color.x * 30.0 + color.y * 59.0 + color.z * 11.0) / 100.0,
        (color.x * 30.0 + color.y * 70.0) / 100.0,
        (color.x * 30.0 + color.z * 70.0) / 100.0,
    )
}

/// java: `RenderGlobal.renderClouds`'s fast path (the `shouldRenderClouds() != 2` branch).
fn fast_cloud_vertices(
    camera_pos: DVec3,
    cloud_height: f32,
    tick: f32,
    color: Vec3,
) -> Vec<CloudVertex> {
    // java: `RenderGlobal.java:1456` — `d2 * 0.029999999329447746D`, the same literal the
    // fancy path already carries.
    let scroll = f64::from(tick) * CLOUD_SCROLL_PER_TICK;
    let x_offset = wrap_cloud_coordinate(camera_pos.x + scroll);
    let z_offset = wrap_cloud_coordinate(camera_pos.z);
    let y = cloud_height - camera_pos.y as f32 + 0.33;
    let rgba = [color.x, color.y, color.z, 0.8];

    // 16 × 16 cells, six vertices each: sized once instead of regrowing ~11 times per frame.
    let cells_per_axis = (2.0 * FAST_SPAN / FAST_CELL) as usize;
    let mut out = Vec::with_capacity(cells_per_axis * cells_per_axis * 6);
    let mut i = -(FAST_SPAN as i32);
    while (i as f32) < FAST_SPAN {
        let mut j = -(FAST_SPAN as i32);
        while (j as f32) < FAST_SPAN {
            let x0 = i as f32;
            let x1 = x0 + FAST_CELL;
            let z0 = j as f32;
            let z1 = z0 + FAST_CELL;
            let world_x0 = x_offset + f64::from(x0);
            let world_x1 = x_offset + f64::from(x1);
            let world_z0 = z_offset + f64::from(z0);
            let world_z1 = z_offset + f64::from(z1);
            let uv = [
                Vec2::new(world_x0 as f32 * FAST_TEXEL, world_z0 as f32 * FAST_TEXEL),
                Vec2::new(world_x1 as f32 * FAST_TEXEL, world_z0 as f32 * FAST_TEXEL),
                Vec2::new(world_x1 as f32 * FAST_TEXEL, world_z1 as f32 * FAST_TEXEL),
                Vec2::new(world_x0 as f32 * FAST_TEXEL, world_z1 as f32 * FAST_TEXEL),
            ];
            push_quad(
                &mut out,
                [
                    Vec3::new(x0, y, z0),
                    Vec3::new(x1, y, z0),
                    Vec3::new(x1, y, z1),
                    Vec3::new(x0, y, z1),
                ],
                uv,
                rgba,
            );
            j += FAST_CELL as i32;
        }
        i += FAST_CELL as i32;
    }
    out
}

/// java: `RenderGlobal.renderCloudsFancy` (`RenderGlobal.java:1494`) — a grid of shaded boxes.
///
/// Transcribed term for term, including the parts an earlier pass of this port invented instead:
///
/// * **Texture coordinates.** Vanilla scales cloud-space coordinates by `0.00390625` (`1/256`),
///   so one cloud unit is exactly one texel of the 256x256 `clouds.png` and one 8-unit cell reads
///   an 8x8 texel patch (`RenderGlobal.java:1582`). The pre-fix port mapped the *whole* texture
///   across every cell, which turned clouds into a full-sky noise field — the visible symptom
///   that prompted this rewrite.
/// * **`GlStateManager.scale(12, 1, 12)`** (`:1545`): X and Z are in cloud units and get scaled;
///   **Y is not**, so the slab really is 4 blocks thick, not 48.
/// * **World anchoring** via `f19`/`f20`, the fractional part of the scrolled camera position in
///   cloud units (`:1540`), with the whole part folded into the texture offset `f17`/`f18`. That
///   is what makes the grid appear pinned to the world while staying centred on the camera.
/// * **The `f3 > -5` / `f3 <= 5` gates** (`:1580`, `:1588`): below the clouds you get only the
///   undersides, above them only the tops. Standing on the ground, the top face is not drawn
///   at all.
/// * **Per-cloud-unit side walls** — eight quads per cell per axis, each sampling the texel
///   centre `+0.5` next to it (`:1600`), not one quad per cell boundary.
/// * **Interior-wall culling** (`j1 > -1`, `j1 <= 1`, `k1 > -1`, `k1 <= 1`): only the walls facing
///   away from the camera's cell are emitted.
///
/// Vanilla draws the whole grid **twice** (`:1547`): pass 0 with `colorMask(false…)` to lay down
/// depth, pass 1 with `GL_LEQUAL` so only the nearest surface shades. [`CloudsPhase`] does the
/// same with two pipelines over these vertices (see its `RenderPhase::record`).
///
/// Production no longer evaluates this per frame: [`fancy_cloud_lattice`] plus
/// [`FancyCloudFrame::table`] reproduce it bit for bit on the GPU side, and this stays as the
/// test oracle they are pinned against.
#[cfg(test)]
fn fancy_cloud_vertices(
    camera_pos: DVec3,
    cloud_height: f32,
    tick: f32,
    color: Vec3,
) -> Vec<CloudVertex> {
    // java: `RenderGlobal.java:1502`-`:1509`.
    let d0 = f64::from(tick);
    let mut d1 = (camera_pos.x + d0 * CLOUD_SCROLL_PER_TICK) / f64::from(FANCY_CELL_SCALE);
    let mut d2 = camera_pos.z / f64::from(FANCY_CELL_SCALE) + 0.330_000_013_113_021_85;
    let f3 = cloud_height - camera_pos.y as f32 + 0.33;
    d1 = wrap_cloud_coordinate(d1);
    d2 = wrap_cloud_coordinate(d2);
    let f17 = d1.floor() as f32 * FANCY_TEXEL;
    let f18 = d2.floor() as f32 * FANCY_TEXEL;
    let f19 = (d1 - d1.floor()) as f32;
    let f20 = (d2 - d2.floor()) as f32;

    // java: `RenderGlobal.java:1528`-`:1536`.
    let top_color = [color.x, color.y, color.z, 0.8];
    let bottom_color = [color.x * 0.7, color.y * 0.7, color.z * 0.7, 0.8];
    let x_side = [color.x * 0.9, color.y * 0.9, color.z * 0.9, 0.8];
    let z_side = [color.x * 0.8, color.y * 0.8, color.z * 0.8, 0.8];

    // `GlStateManager.scale(12, 1, 12)` — X/Z only.
    let sx = |u: f32| u * FANCY_CELL_SCALE;

    let mut out = Vec::new();
    // java: `RenderGlobal.java:1570`-`:1572` — `for (j1 = -3; j1 <= 4; ++j1)`, i.e. exactly
    // `FANCY_CELLS` cells per axis, offset one cell toward +X/+Z of the camera's own cell.
    for j1 in -(FANCY_CELLS / 2 - 1)..=FANCY_CELLS / 2 {
        for k1 in -(FANCY_CELLS / 2 - 1)..=FANCY_CELLS / 2 {
            let f22 = j1 as f32 * FANCY_CELL;
            let f23 = k1 as f32 * FANCY_CELL;
            let f24 = f22 - f19;
            let f25 = f23 - f20;
            let u = |cloud_u: f32| cloud_u * FANCY_TEXEL + f17;
            let v = |cloud_v: f32| cloud_v * FANCY_TEXEL + f18;

            // Bottom face. java: `:1580`-`:1586`.
            if f3 > -5.0 {
                push_quad(
                    &mut out,
                    [
                        Vec3::new(sx(f24), f3, sx(f25 + 8.0)),
                        Vec3::new(sx(f24 + 8.0), f3, sx(f25 + 8.0)),
                        Vec3::new(sx(f24 + 8.0), f3, sx(f25)),
                        Vec3::new(sx(f24), f3, sx(f25)),
                    ],
                    [
                        Vec2::new(u(f22), v(f23 + 8.0)),
                        Vec2::new(u(f22 + 8.0), v(f23 + 8.0)),
                        Vec2::new(u(f22 + 8.0), v(f23)),
                        Vec2::new(u(f22), v(f23)),
                    ],
                    bottom_color,
                );
            }

            // Top face. java: `:1588`-`:1594`.
            if f3 <= 5.0 {
                let y = f3 + FANCY_THICKNESS - FANCY_EPSILON;
                push_quad(
                    &mut out,
                    [
                        Vec3::new(sx(f24), y, sx(f25 + 8.0)),
                        Vec3::new(sx(f24 + 8.0), y, sx(f25 + 8.0)),
                        Vec3::new(sx(f24 + 8.0), y, sx(f25)),
                        Vec3::new(sx(f24), y, sx(f25)),
                    ],
                    [
                        Vec2::new(u(f22), v(f23 + 8.0)),
                        Vec2::new(u(f22 + 8.0), v(f23 + 8.0)),
                        Vec2::new(u(f22 + 8.0), v(f23)),
                        Vec2::new(u(f22), v(f23)),
                    ],
                    top_color,
                );
            }

            // -X walls, one per cloud unit. java: `:1596`-`:1605`.
            if j1 > -1 {
                for l1 in 0..FANCY_CELL as i32 {
                    let x = sx(f24 + l1 as f32);
                    let uu = u(f22 + l1 as f32 + 0.5);
                    push_quad(
                        &mut out,
                        [
                            Vec3::new(x, f3, sx(f25 + 8.0)),
                            Vec3::new(x, f3 + FANCY_THICKNESS, sx(f25 + 8.0)),
                            Vec3::new(x, f3 + FANCY_THICKNESS, sx(f25)),
                            Vec3::new(x, f3, sx(f25)),
                        ],
                        [
                            Vec2::new(uu, v(f23 + 8.0)),
                            Vec2::new(uu, v(f23 + 8.0)),
                            Vec2::new(uu, v(f23)),
                            Vec2::new(uu, v(f23)),
                        ],
                        x_side,
                    );
                }
            }

            // +X walls. java: `:1607`-`:1616`.
            if j1 <= 1 {
                for i2 in 0..FANCY_CELL as i32 {
                    let x = sx(f24 + i2 as f32 + 1.0 - FANCY_EPSILON);
                    let uu = u(f22 + i2 as f32 + 0.5);
                    push_quad(
                        &mut out,
                        [
                            Vec3::new(x, f3, sx(f25 + 8.0)),
                            Vec3::new(x, f3 + FANCY_THICKNESS, sx(f25 + 8.0)),
                            Vec3::new(x, f3 + FANCY_THICKNESS, sx(f25)),
                            Vec3::new(x, f3, sx(f25)),
                        ],
                        [
                            Vec2::new(uu, v(f23 + 8.0)),
                            Vec2::new(uu, v(f23 + 8.0)),
                            Vec2::new(uu, v(f23)),
                            Vec2::new(uu, v(f23)),
                        ],
                        x_side,
                    );
                }
            }

            // -Z walls. java: `:1618`-`:1627`.
            if k1 > -1 {
                for j2 in 0..FANCY_CELL as i32 {
                    let z = sx(f25 + j2 as f32);
                    let vv = v(f23 + j2 as f32 + 0.5);
                    push_quad(
                        &mut out,
                        [
                            Vec3::new(sx(f24), f3 + FANCY_THICKNESS, z),
                            Vec3::new(sx(f24 + 8.0), f3 + FANCY_THICKNESS, z),
                            Vec3::new(sx(f24 + 8.0), f3, z),
                            Vec3::new(sx(f24), f3, z),
                        ],
                        [
                            Vec2::new(u(f22), vv),
                            Vec2::new(u(f22 + 8.0), vv),
                            Vec2::new(u(f22 + 8.0), vv),
                            Vec2::new(u(f22), vv),
                        ],
                        z_side,
                    );
                }
            }

            // +Z walls. java: `:1629`-`:1638`.
            if k1 <= 1 {
                for k2 in 0..FANCY_CELL as i32 {
                    let z = sx(f25 + k2 as f32 + 1.0 - FANCY_EPSILON);
                    let vv = v(f23 + k2 as f32 + 0.5);
                    push_quad(
                        &mut out,
                        [
                            Vec3::new(sx(f24), f3 + FANCY_THICKNESS, z),
                            Vec3::new(sx(f24 + 8.0), f3 + FANCY_THICKNESS, z),
                            Vec3::new(sx(f24 + 8.0), f3, z),
                            Vec3::new(sx(f24), f3, z),
                        ],
                        [
                            Vec2::new(u(f22), vv),
                            Vec2::new(u(f22 + 8.0), vv),
                            Vec2::new(u(f22 + 8.0), vv),
                            Vec2::new(u(f22), vv),
                        ],
                        z_side,
                    );
                }
            }
        }
    }
    out
}

fn bind_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    tex: &SkyTexture,
    sampler: &wgpu::Sampler,
    fog_uniform: &wgpu::Buffer,
    fancy_table: &wgpu::Buffer,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clouds group1"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: fog_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: fancy_table.as_entire_binding(),
            },
        ],
    })
}

/// Clouds. [`CloudsPhase::install`] registers them where vanilla draws them (see the
/// [module docs](self)): at [`crate::PhaseOrder::SKY`], after [`crate::sky::dome::SkyPhase`] so
/// insertion order draws them on top of (in front of) the sky plane, and at
/// [`ABOVE_CLOUDS_ORDER`]. A `CloudsPhase` inserted on its own at [`crate::PhaseOrder::SKY`]
/// always draws there.
pub struct CloudsPhase {
    /// java: `GameSettings.renderClouds`.
    pub mode: CloudMode,
    /// This frame's time/weather/dimension state.
    pub environment: Environment,
    /// This frame's render interpolation fraction.
    pub partial_ticks: f32,
    /// java: `WorldProvider.getCloudHeight`.
    pub cloud_height: f32,
    /// java: `this.cloudTickCounter` — whole ticks elapsed. `partial_ticks` is added once by
    /// [`CloudsPhase::record`] when it builds Java's interpolated cloud scroll time.
    pub cloud_tick_counter: f32,
    /// java: `EntityRenderer.renderWorldPass`'s
    /// `entity.posY + entity.getEyeHeight()` for the render-view entity: this tick's position,
    /// not interpolated, and the entity's eye, not the camera (F5 orbits it). At or above
    /// [`ABOVE_CLOUDS_EYE_Y`] the registration at [`ABOVE_CLOUDS_ORDER`] draws the clouds, below
    /// it the one at [`crate::PhaseOrder::SKY`] (see [`CloudsPhase::install`]). `None`, the
    /// default, counts as below.
    pub view_eye_y: Option<f64>,

    /// Java's global `GameSettings.anaglyph` upload switch. The resident cloud texture keeps its
    /// decoded source bytes inside [`SkyTexture`], so a live toggle can replay the exact upload
    /// transform without mutating or trying to invert the original PNG pixels.
    anaglyph: bool,

    pipeline: wgpu::RenderPipeline,
    /// java: `RenderGlobal.java:1547`-`:1551` — the `i1 == 0` iteration, `colorMask(false, …)`.
    /// Depth only, so the colour pass that follows shades the nearest cloud surface and nothing
    /// behind it. Only fancy clouds need it; the fast path is a single flat sheet.
    depth_prepass_pipeline: wgpu::RenderPipeline,
    /// Fancy clouds' colour pass over the resident lattice (`vs_fancy`).
    fancy_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    fog_uniform: wgpu::Buffer,
    fog_write: crate::context::ChangedWrite,
    vertex_buffer: Option<wgpu::Buffer>,
    vertex_capacity: usize,
    /// The fancy lattice ([`fancy_cloud_lattice`]), resident; sized for the fullest gate set.
    fancy_vertices: wgpu::Buffer,
    fancy_vertex_count: u32,
    /// The `(undersides, tops)` gates `fancy_vertices` was built for; `None` until first use.
    fancy_gates: Option<(bool, bool)>,
    /// This frame's [`FancyCloudFrame::table`], bound at `@group(1) @binding(3)`.
    fancy_table: wgpu::Buffer,
    fancy_table_write: crate::context::ChangedWrite,
    fog_state: Option<(FogState, FogDistance)>,
    texture: Option<(SkyTexture, wgpu::BindGroup)>,
    /// `Some` while culling is on and `clouds.png` is 256x256.
    fancy_cull: Option<FancyCull>,
}

impl CloudsPhase {
    /// Build the phase. The cloud texture starts unset — see [`CloudsPhase::load_texture`].
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("clouds"),
                source: wgpu::ShaderSource::Wgsl(include_str!("clouds.wgsl").into()),
            });

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("clouds group1"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: Some(
                                    std::num::NonZeroU64::new(
                                        std::mem::size_of::<CloudFogUniform>() as u64,
                                    )
                                    .unwrap(),
                                ),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: Some(
                                    std::num::NonZeroU64::new(
                                        (FANCY_TABLE_ROWS * std::mem::size_of::<[f32; 4]>()) as u64,
                                    )
                                    .unwrap(),
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("clouds layout"),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let depth = |write: bool, compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState {
                format: crate::DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        };

        let make = |label: &str,
                    vs: &str,
                    fs: &str,
                    vbuf: wgpu::VertexBufferLayout<'static>,
                    write_color: bool,
                    depth_state: Option<wgpu::DepthStencilState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(vbuf)],
                    },
                    primitive: wgpu::PrimitiveState {
                        // java: `RenderGlobal.java:1496` — `GlStateManager.disableCull()`.
                        cull_mode: None,
                        ..wgpu::PrimitiveState::default()
                    },
                    depth_stencil: depth_state,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: if write_color {
                                wgpu::ColorWrites::ALL
                            } else {
                                wgpu::ColorWrites::empty()
                            },
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };

        // java: `RenderGlobal.java:1438` blend + the ambient `GL_LEQUAL` depth func, with depth
        // writes on: `renderWorldPass` leaves `depthMask(true)` at both of its cloud points.
        let pipeline = make(
            "clouds",
            "vs",
            "fs",
            CloudVertex::LAYOUT,
            true,
            depth(true, wgpu::CompareFunction::LessEqual),
        );
        // Vanilla's colour pass writes depth too, but only where the pre-pass already wrote the
        // same value (`GL_LEQUAL` lets nothing nearer through), so leaving it off stores the
        // same buffer.
        let fancy_pipeline = make(
            "clouds fancy",
            "vs_fancy",
            "fs",
            FancyCloudVertex::LAYOUT,
            true,
            depth(false, wgpu::CompareFunction::LessEqual),
        );
        // `fs_prepass` makes `fs`'s exact discard decision and nothing else: the colour target
        // has an empty write mask, so the fog maths `fs` would run here is never observable.
        let depth_prepass_pipeline = make(
            "clouds depth prepass",
            "vs_fancy",
            "fs_prepass",
            FancyCloudVertex::LAYOUT,
            false,
            depth(true, wgpu::CompareFunction::LessEqual),
        );

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("clouds sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let fog_uniform = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cloud fog uniform"),
                contents: bytemuck::bytes_of(&CloudFogUniform::default()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let fancy_table = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cloud fancy table"),
                contents: bytemuck::cast_slice(&[[0.0_f32; 4]; FANCY_TABLE_ROWS][..]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let fancy_vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud fancy lattice"),
            size: (fancy_cloud_lattice(true, true).len() * std::mem::size_of::<FancyCloudVertex>())
                as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            mode: CloudMode::default(),
            environment: Environment::default(),
            partial_ticks: 0.0,
            cloud_height: DEFAULT_CLOUD_HEIGHT,
            cloud_tick_counter: 0.0,
            view_eye_y: None,
            anaglyph: false,
            pipeline,
            depth_prepass_pipeline,
            fancy_pipeline,
            bind_group_layout,
            sampler,
            fog_uniform,
            fog_write: crate::context::ChangedWrite::default(),
            vertex_buffer: None,
            vertex_capacity: 0,
            fancy_vertices,
            fancy_vertex_count: 0,
            fancy_gates: None,
            fancy_table,
            fancy_table_write: crate::context::ChangedWrite::default(),
            fog_state: None,
            texture: None,
            fancy_cull: None,
        }
    }

    /// Register `clouds` at both of vanilla's points (see the
    /// [module docs](self)): at [`crate::PhaseOrder::SKY`] as `sky/clouds` (register the sky dome
    /// there first, so the clouds land on it) and at [`ABOVE_CLOUDS_ORDER`] as
    /// [`ABOVE_CLOUDS_PHASE_NAME`]. Each frame [`Self::draws_above`] enables exactly one of them;
    /// both draw the same way, only the point in the frame differs. Allocates nothing.
    pub fn install(clouds: &Rc<RefCell<CloudsPhase>>, phases: &mut PhaseList) {
        for above in [false, true] {
            phases.insert(
                if above {
                    ABOVE_CLOUDS_ORDER
                } else {
                    PhaseOrder::SKY
                },
                Box::new(CloudsSlot {
                    clouds: Rc::clone(clouds),
                    above,
                }),
            );
        }
    }

    /// Whether this frame's clouds belong after the world: [`Self::view_eye_y`] is at or above
    /// [`ABOVE_CLOUDS_EYE_Y`].
    #[must_use]
    pub fn draws_above(&self) -> bool {
        eye_above_clouds(self.view_eye_y)
    }

    /// Build the phase with Java's global upload-time anaglyph switch selected.
    ///
    /// The flag is retained before `clouds.png` is loaded, so its first upload already uses the
    /// requested integer, alpha-preserving `TextureUtil.anaglyphColor` transform. Use
    /// [`Self::set_anaglyph_with_gpu`] when changing an installed phase live.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let mut phase = Self::new(gpu);
        phase.anaglyph = anaglyph;
        phase
    }

    /// Load `clouds.png`. Left unset (nothing drawn) if the resource is missing.
    pub fn load_texture(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        if let Ok(tex) = textures::load_with_anaglyph(
            gpu,
            rm,
            "textures/environment/clouds.png",
            "clouds",
            self.anaglyph,
        ) {
            let group = bind_group(
                gpu,
                &self.bind_group_layout,
                &tex,
                &self.sampler,
                &self.fog_uniform,
                &self.fancy_table,
            );
            // A (re)loaded texture means new coverage and a fresh list.
            self.fancy_cull = if cloud_cull_enabled() {
                CloudCoverage::from_rgba(tex.width, tex.height, tex.source_rgba())
                    .map(FancyCull::new)
            } else {
                None
            };
            self.texture = Some((tex, group));
        }
    }

    /// Whether Java's upload-time integer anaglyph conversion is selected for `clouds.png`.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global transform and replay the resident cloud texture from its retained
    /// decoded source bytes. The early return makes repeated driver fan-out idempotent while the
    /// raw-source replay makes `false -> true -> false` byte-for-byte stable.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        if let Some((texture, _)) = &self.texture {
            texture.set_anaglyph(gpu, enabled);
        }
    }

    /// Set the transform for subsequent resource uploads without touching a resident texture.
    /// Use [`Self::set_anaglyph_with_gpu`] when changing an installed phase live.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Set the same per-frame fog state used by [`FogPhase`]. `None` leaves cloud RGB unchanged;
    /// cloud alpha remains owned by the texture and vertex colour in either case.
    pub fn set_fog(&mut self, state: Option<(FogState, FogDistance)>) {
        self.fog_state = state;
    }

    /// Turn fancy-face culling on or off for this phase, for same-process
    /// comparisons; the client keeps the `MC_NW_F2_CLOUD_CULL` default that
    /// [`Self::load_texture`] applies. Returns whether culling is on afterwards (it stays off
    /// without a loaded 256x256 `clouds.png`).
    pub fn set_fancy_face_culling(&mut self, on: bool) -> bool {
        self.fancy_cull = match (&self.texture, on) {
            (Some((tex, _)), true) => {
                CloudCoverage::from_rgba(tex.width, tex.height, tex.source_rgba())
                    .map(FancyCull::new)
            }
            _ => None,
        };
        self.fancy_cull.is_some()
    }

    /// `(indices drawn, lattice vertices)` of the last culled fancy frame;
    /// `None` while culling is off or before the first fancy frame.
    #[must_use]
    pub fn fancy_face_cull_stats(&self) -> Option<(u32, u32)> {
        let cull = self.fancy_cull.as_ref()?;
        cull.offset?;
        Some((cull.count, self.fancy_vertex_count))
    }
}

impl RenderPhase for CloudsPhase {
    fn name(&self) -> &'static str {
        PHASE_NAME
    }

    fn enabled(&self) -> bool {
        self.mode != CloudMode::Off && self.environment.dimension.has_clouds()
    }

    // `record` makes its queue writes, then opens its one pass with
    // `Gpu::begin_pass` on the frame's colour target and the shared depth buffer before drawing
    // anything, and nothing else in it touches either: both clears can wait for that pass.
    fn frame_clear_handoff(&self) -> FrameClearHandoff {
        FrameClearHandoff::ColorAndDepth
    }

    // One pass, configured as the world pass, and
    // nothing else on the encoder (the depth stays, so no restore pass follows it): the clouds
    // join the shared world pass. At `PhaseOrder::SKY` they open it, and the opaque terrain
    // recorded next draws into the same pass. Without a texture they draw nothing.
    fn shares_world_pass(&self) -> bool {
        self.texture.is_some()
    }

    /// # Depth
    ///
    /// Vanilla draws fancy clouds twice (`RenderGlobal.java:1547`): once with
    /// `colorMask(false, false, false, false)` to lay down depth, then again for colour under the
    /// ambient `GL_LEQUAL` depth func, so only the nearest cloud surface shades. Without that,
    /// every one of the eight per-cloud-unit interior walls blends over the face in front of it
    /// and the undersides come out as a waffle of grid lines instead of flat white. Fast clouds
    /// are one draw.
    ///
    /// Both write their depth into the frame's shared depth buffer and
    /// leave it there, as vanilla does at both of its cloud points (see the
    /// [module docs](self)). Recorded at [`crate::PhaseOrder::SKY`], before anything else writes
    /// depth, that depth is what rejects terrain behind a cloud; recorded at
    /// [`ABOVE_CLOUDS_ORDER`], the clouds are depth-tested against the finished world. Either way
    /// there is nothing to hand back: one pass (or a share of the world pass), no depth target of
    /// the clouds' own, no restore pass. `clouds_keep_their_depth_for_terrain_drawn_after_them`
    /// in `tests/s20_sky.rs` and `tests/clouds_vanilla_depth.rs` pin it.
    ///
    /// # Resident fancy geometry
    ///
    /// The fancy lattice is built once per `f3` gate combination ([`fancy_cloud_lattice`]) and
    /// every frame-dependent value — `f3`, `f17`..`f20` and the four face colours — goes up as a
    /// ~1.2 KB table ([`FancyCloudFrame::table`]) that `vs_fancy` indexes. Each table entry is the
    /// same `f32` expression [`fancy_cloud_vertices`] used to evaluate per vertex, and the shader
    /// only selects entries (plus one exact UV addition), so the vertex stage sees the same bits.
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.record_to(gpu, PassTarget::Own { enc, view }, cam);
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass), cam);
    }
}

impl CloudsPhase {
    /// The frame's cloud draws (see `RenderPhase::record`), into a pass of their own or the
    /// shared world pass.
    fn record_to(&mut self, gpu: &Gpu, target: PassTarget<'_>, cam: &Camera) {
        mc_core::profile_scope!("sky.clouds.record");
        if self.texture.is_none() {
            return;
        }
        let fog_uniform =
            CloudFogUniform::from_state(self.fog_state).with_cloud_far(cam.far, cam.near);
        // Read only by this phase's passes.
        self.fog_write
            .write_for_frame(gpu, &self.fog_uniform, bytemuck::bytes_of(&fog_uniform));
        let tick = cloud_tick_at(self.cloud_tick_counter, self.partial_ticks);
        // Java has two distinct anaglyph boundaries here: `TextureUtil.anaglyphColor` rewrites
        // the decoded PNG upload bytes, while `RenderGlobal` transforms the sampled world cloud
        // tint as floats before applying its fast/fancy face-shading multipliers. Keep both paths
        // explicit so neither conversion is accidentally reused for the other.
        let color = cloud_color_for_render(
            self.environment.cloud_color(self.partial_ticks),
            self.anaglyph,
        );
        match self.mode {
            CloudMode::Off => {}
            // The fast path is one flat sheet: nothing can overlap, so there is nothing for a
            // depth pre-pass to resolve. java: `renderClouds` has no second iteration.
            CloudMode::Fast => {
                let verts = fast_cloud_vertices(cam.pos, self.cloud_height, tick, color);
                if verts.is_empty() {
                    return;
                }
                if verts.len() > self.vertex_capacity {
                    self.vertex_capacity = verts.len().next_power_of_two();
                    self.vertex_buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("cloud verts"),
                        size: (self.vertex_capacity * std::mem::size_of::<CloudVertex>()) as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                }
                let buf = self
                    .vertex_buffer
                    .as_ref()
                    .expect("sized immediately above");
                gpu.write_buffer_staged(buf, 0, bytemuck::cast_slice(&verts));

                let Some((_, group)) = &self.texture else {
                    return;
                };
                let count = verts.len() as u32;
                let mut own = None;
                let pass = target.begin(gpu, PHASE_NAME, &mut own);
                pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
                pass.set_bind_group(1, group, &[]);
                pass.set_vertex_buffer(
                    0,
                    buf.slice(..count as u64 * std::mem::size_of::<CloudVertex>() as u64),
                );
                pass.set_pipeline(&self.pipeline);
                pass.draw(0..count, 0..1);
            }
            CloudMode::Fancy => {
                let frame = FancyCloudFrame::new(cam.pos, self.cloud_height, tick);
                let gates = frame.face_gates();
                if self.fancy_gates != Some(gates) {
                    // Only when the eye crosses within ±5 blocks of the slab (or on first use).
                    let lattice = fancy_cloud_lattice(gates.0, gates.1);
                    if !lattice.is_empty() {
                        gpu.write_buffer_staged(
                            &self.fancy_vertices,
                            0,
                            bytemuck::cast_slice(&lattice),
                        );
                    }
                    self.fancy_vertex_count = lattice.len() as u32;
                    self.fancy_gates = Some(gates);
                }
                let table = frame.table(color);
                // Read only by this phase's passes.
                self.fancy_table_write.write_for_frame(
                    gpu,
                    &self.fancy_table,
                    bytemuck::cast_slice(&table[..]),
                );
                let count = self.fancy_vertex_count;
                if count == 0 {
                    return;
                }
                // `Some(n)`: draw only the `n` listed lattice indices.
                let culled = match self.fancy_cull.as_mut() {
                    Some(cull) if self.texture.is_some() => Some(cull.update(gpu, gates, &frame)),
                    _ => None,
                };
                if culled == Some(0) {
                    // No face can draw: the pre-pass would write no depth and the colour pass
                    // nothing, so skipping both leaves the same frame.
                    return;
                }
                let Some((_, group)) = &self.texture else {
                    return;
                };
                // The pre-pass and the colour pass share the frame's depth buffer and keep what
                // they write there (see `RenderPhase::record`).
                let mut own = None;
                let pass = target.begin(gpu, PHASE_NAME, &mut own);
                pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
                pass.set_bind_group(1, group, &[]);
                pass.set_vertex_buffer(
                    0,
                    self.fancy_vertices
                        .slice(..u64::from(count) * std::mem::size_of::<FancyCloudVertex>() as u64),
                );
                let indices = self.fancy_cull.as_ref().and_then(|c| c.buffer.as_ref());
                match (culled, indices) {
                    (Some(n), Some(indices)) => {
                        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                        pass.set_pipeline(&self.depth_prepass_pipeline);
                        pass.draw_indexed(0..n, 0, 0..1);
                        pass.set_pipeline(&self.fancy_pipeline);
                        pass.draw_indexed(0..n, 0, 0..1);
                    }
                    _ => {
                        pass.set_pipeline(&self.depth_prepass_pipeline);
                        pass.draw(0..count, 0..1);
                        pass.set_pipeline(&self.fancy_pipeline);
                        pass.draw(0..count, 0..1);
                    }
                }
            }
        }
    }
}

/// One of [`CloudsPhase::install`]'s two registrations of a shared
/// [`CloudsPhase`]: enabled only on the frames whose eye height ([`CloudsPhase::draws_above`])
/// puts the clouds at its point, and otherwise exactly the phase it wraps.
struct CloudsSlot {
    clouds: Rc<RefCell<CloudsPhase>>,
    /// The registration at [`ABOVE_CLOUDS_ORDER`]; else the one at [`crate::PhaseOrder::SKY`].
    above: bool,
}

impl RenderPhase for CloudsSlot {
    fn name(&self) -> &'static str {
        if self.above {
            ABOVE_CLOUDS_PHASE_NAME
        } else {
            PHASE_NAME
        }
    }

    fn enabled(&self) -> bool {
        let clouds = self.clouds.borrow();
        clouds.enabled() && clouds.draws_above() == self.above
    }

    fn frame_clear_handoff(&self) -> FrameClearHandoff {
        self.clouds.borrow().frame_clear_handoff()
    }

    fn shares_world_pass(&self) -> bool {
        self.clouds.borrow().shares_world_pass()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.clouds.borrow_mut().record(gpu, enc, view, cam);
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, cam: &Camera) {
        self.clouds.borrow_mut().record_into_pass(gpu, pass, cam);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `cargo check` never parses WGSL; this does, with the same naga front end and validator
    /// wgpu runs at pipeline creation, so a shader mistake fails a unit test instead of a
    /// player's first frame.
    #[test]
    fn clouds_wgsl_parses_and_validates() {
        let source = include_str!("clouds.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("clouds.wgsl validates");
    }


    /// A deterministic 256x256 pattern shaped like `clouds.png` (alpha 1 or 255, ~20% opaque
    /// blobs), plus isolated texels with alphas around [`COVERAGE_ALPHA_FLOOR`].
    fn cull_test_rgba(seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 8
        };
        let mut rgba = vec![255_u8; 256 * 256 * 4];
        for pixel in rgba.chunks_exact_mut(4) {
            pixel[3] = 1;
        }
        let mut set = |x: u32, y: u32, a: u8| {
            rgba[((y % 256 * 256 + x % 256) * 4 + 3) as usize] = a;
        };
        for _ in 0..60 {
            let (cx, cy, r) = (next() % 256, next() % 256, 1 + next() % 7);
            for dy in 0..2 * r {
                for dx in 0..2 * r {
                    set(cx + dx, cy + dy, 255);
                }
            }
        }
        for a in [24_u8, 25, 26, 32, 128] {
            for _ in 0..40 {
                set(next() % 256, next() % 256, a);
            }
        }
        rgba
    }

    fn cull_test_alpha(rgba: &[u8], c: i32, r: i32) -> u8 {
        rgba[((r.rem_euclid(256) * 256 + c.rem_euclid(256)) * 4 + 3) as usize]
    }

    #[test]
    fn cull_footprints_follow_the_lattice_face_for_face() {
        for (bottom, top) in [(true, true), (true, false), (false, true), (false, false)] {
            let lattice = fancy_cloud_lattice(bottom, top);
            let footprints = fancy_cloud_footprints(bottom, top);
            assert_eq!(
                lattice.len(),
                footprints.len() * 6,
                "gates ({bottom}, {top})"
            );
            for (face, fp) in footprints.iter().enumerate() {
                for v in &lattice[face * 6..face * 6 + 6] {
                    // Frame-independent UVs in texels: exact, the scale is a power of two.
                    let (u, w) = (v.uv[0] * 256.0, v.uv[1] * 256.0);
                    for (value, start, len, axis) in
                        [(u, fp.col0, fp.cols, "u"), (w, fp.row0, fp.rows, "v")]
                    {
                        let start = f32::from(start);
                        if len == 1 {
                            assert_eq!(value, start + 0.5, "face {face} {axis}: a texel centre");
                        } else {
                            // The face spans [start + 1, start + len - 1]; one texel of margin
                            // each side.
                            assert!(
                                value >= start + 1.0 && value <= start + f32::from(len) - 1.0,
                                "face {face} {axis} = {value} outside {start}+{len}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The list keeps every face with a texel that can survive the alpha test anywhere in the
    /// patch its UVs span, drops every face with none even one texel around it, and drops some.
    #[test]
    fn cull_keeps_every_face_that_can_draw_and_drops_the_rest() {
        let rgba = cull_test_rgba(0x5eed);
        let coverage = CloudCoverage::from_rgba(256, 256, &rgba).expect("256x256");
        let survives = |c: i32, r: i32| cull_test_alpha(&rgba, c, r) > COVERAGE_ALPHA_FLOOR;
        let mut dropped_total = 0;
        for (bottom, top) in [(true, true), (true, false), (false, true)] {
            let lattice = fancy_cloud_lattice(bottom, top);
            let footprints = fancy_cloud_footprints(bottom, top);
            for (ox, oz) in [
                (0, 0),
                (7, 250),
                (255, 1),
                (1000, 2047),
                (2047, 64),
                (300, 300),
            ] {
                let mut indices = Vec::new();
                fancy_visible_indices(&footprints, &coverage, ox, oz, &mut indices);
                assert_eq!(indices.len() % 6, 0);
                let kept: Vec<usize> = indices
                    .chunks(6)
                    .map(|face| {
                        let first = face[0] as usize;
                        assert_eq!(first % 6, 0);
                        assert!(face
                            .iter()
                            .enumerate()
                            .all(|(i, &v)| v as usize == first + i));
                        first / 6
                    })
                    .collect();
                assert!(kept.windows(2).all(|w| w[0] < w[1]), "emission order kept");
                for face in 0..footprints.len() {
                    // The texels the face's UVs span, from the lattice itself: `lo..=hi`.
                    let verts = &lattice[face * 6..face * 6 + 6];
                    let span = |axis: usize, offset: i32| {
                        let texel = |v: &FancyCloudVertex| v.uv[axis] * 256.0;
                        let lo = verts.iter().map(texel).fold(f32::MAX, f32::min).floor();
                        let hi = verts.iter().map(texel).fold(f32::MIN, f32::max).ceil() - 1.0;
                        (lo as i32 + offset, hi.max(lo) as i32 + offset)
                    };
                    let (cols, rows) = (span(0, ox), span(1, oz));
                    let any = |(c0, c1): (i32, i32), (r0, r1): (i32, i32)| {
                        (r0..=r1).any(|r| (c0..=c1).any(|c| survives(c, r)))
                    };
                    // One texel of margin along a span, none across a wall.
                    let widen = |(lo, hi): (i32, i32)| {
                        if lo == hi {
                            (lo, hi)
                        } else {
                            (lo - 1, hi + 1)
                        }
                    };
                    let inside = any(cols, rows);
                    let near = any(widen(cols), widen(rows));
                    let is_kept = kept.binary_search(&face).is_ok();
                    assert!(
                        !inside || is_kept,
                        "face {face} at ({ox}, {oz}) can draw: dropped"
                    );
                    assert_eq!(is_kept, near, "face {face} at ({ox}, {oz})");
                    dropped_total += usize::from(!is_kept);
                }
            }
        }
        assert!(
            dropped_total > 1000,
            "culling dropped only {dropped_total} faces"
        );
    }

    #[test]
    fn cull_handles_only_full_and_empty_extremes_and_256_textures() {
        assert!(CloudCoverage::from_rgba(512, 512, &vec![255; 512 * 512 * 4]).is_none());
        assert!(CloudCoverage::from_rgba(256, 256, &[255; 16]).is_none());
        let footprints = fancy_cloud_footprints(true, true);
        let opaque = CloudCoverage::from_rgba(256, 256, &vec![255; 256 * 256 * 4]).unwrap();
        let mut indices = Vec::new();
        fancy_visible_indices(&footprints, &opaque, 17, 230, &mut indices);
        assert_eq!(indices.len(), fancy_cloud_lattice(true, true).len());
        let mut clear = vec![255_u8; 256 * 256 * 4];
        for pixel in clear.chunks_exact_mut(4) {
            pixel[3] = COVERAGE_ALPHA_FLOOR;
        }
        let empty = CloudCoverage::from_rgba(256, 256, &clear).unwrap();
        indices.clear();
        fancy_visible_indices(&footprints, &empty, 17, 230, &mut indices);
        assert!(indices.is_empty());
        // The u16 index list can address the fullest lattice.
        assert!(fancy_cloud_lattice(true, true).len() <= usize::from(u16::MAX) + 1);
    }

    #[test]
    fn cloud_resource_owner_routes_png_through_anaglyph_upload_and_replay() {
        let source = include_str!("clouds.rs");
        let loader = concat!("textures::load_with_anaglyph", "(");
        assert_eq!(
            source.matches(loader).count(),
            1,
            "clouds.png must cross one anaglyph-aware upload boundary"
        );
        let owner_call = source
            .split_once("textures/environment/clouds.png")
            .map(|(_, rest)| rest)
            .expect("cloud resource path must have an owner call");
        assert!(
            owner_call.contains("self.anaglyph"),
            "cloud startup upload must use the retained phase setting"
        );
        assert!(
            source.contains("texture.set_anaglyph(gpu, enabled)"),
            "live cloud toggles must replay resident source bytes"
        );
    }

    #[test]
    fn cloud_upload_uses_java_colored_bytes_and_preserves_alpha() {
        // `textures::load_with_anaglyph` delegates to this shared TextureUtil byte oracle. Keep a
        // coloured pixel here: grayscale-only controls cannot distinguish a missing transform
        // from a correct transform, while the alpha byte proves the upload is RGBA-preserving.
        let source = [255, 0, 0, 7, 0, 255, 0, 19, 0, 0, 255, 31];
        assert_eq!(
            mc_assets::sprite::anaglyph_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31]
        );
        assert_eq!(
            mc_assets::sprite::anaglyph_rgba(&source, false),
            source,
            "disabled cloud upload must retain every source byte"
        );
    }

    #[test]
    fn cloud_colour_anaglyph_is_float_path_once_and_disabled_is_identity() {
        let source = Vec3::new(0.17, 0.53, 0.91);
        let disabled = cloud_color_for_render(source, false);
        assert_eq!(
            disabled.to_array().map(f32::to_bits),
            source.to_array().map(f32::to_bits),
            "the non-anaglyph cloud colour path must be an exact identity"
        );

        let enabled = cloud_color_for_render(source, true);
        let expected = Vec3::new(
            (source.x * 30.0 + source.y * 59.0 + source.z * 11.0) / 100.0,
            (source.x * 30.0 + source.y * 70.0) / 100.0,
            (source.x * 30.0 + source.z * 70.0) / 100.0,
        );
        assert_eq!(
            enabled.to_array().map(f32::to_bits),
            expected.to_array().map(f32::to_bits),
            "RenderGlobal's cloud tint uses weighted float channels"
        );
        assert_ne!(
            cloud_color_for_render(enabled, true)
                .to_array()
                .map(f32::to_bits),
            enabled.to_array().map(f32::to_bits),
            "the upload byte transform and cloud tint must each run once, not recursively"
        );
    }

    #[test]
    fn cloud_colour_transform_reaches_fast_vertices_without_touching_alpha() {
        let base = Vec3::new(0.13, 0.61, 0.89);
        let transformed = cloud_color_for_render(base, true);
        let verts = fast_cloud_vertices(DVec3::ZERO, 128.0, 0.0, transformed);
        assert_eq!(verts[0].color[3].to_bits(), 0.8_f32.to_bits());
        assert_eq!(verts[0].color[0].to_bits(), transformed.x.to_bits());
        assert_eq!(verts[0].color[1].to_bits(), transformed.y.to_bits());
        assert_eq!(verts[0].color[2].to_bits(), transformed.z.to_bits());
    }

    #[test]
    fn fast_grid_covers_the_expected_span() {
        let verts = fast_cloud_vertices(DVec3::ZERO, 128.0, 0.0, Vec3::ONE);
        // (512/32)^2 = 256 cells * 6 verts.
        assert_eq!(verts.len(), 256 * 6);
    }

    #[test]
    fn fast_clouds_sit_at_cloud_height_relative_to_the_camera() {
        let verts = fast_cloud_vertices(DVec3::new(0.0, 80.0, 0.0), 128.0, 0.0, Vec3::ONE);
        let expected_y = 128.0 - 80.0 + 0.33;
        assert!((verts[0].pos[1] - expected_y).abs() < 1e-3);
    }

    #[test]
    fn fast_cloud_uv_scrolls_with_tick() {
        let a = fast_cloud_vertices(DVec3::ZERO, 128.0, 0.0, Vec3::ONE);
        let b = fast_cloud_vertices(DVec3::ZERO, 128.0, 100.0, Vec3::ONE);
        assert_ne!(a[0].uv, b[0].uv);
    }

    #[test]
    fn cloud_partial_ticks_scroll_fast_and_fancy_once() {
        let whole_tick = 120.0;
        let base_fast = fast_cloud_vertices(
            DVec3::ZERO,
            128.0,
            cloud_tick_at(whole_tick, 0.0),
            Vec3::ONE,
        );
        let base_fancy = fancy_cloud_vertices(
            DVec3::new(0.0, 64.0, 0.0),
            128.0,
            cloud_tick_at(whole_tick, 0.0),
            Vec3::ONE,
        );
        for partial_ticks in [0.0, 0.25, 0.5, 0.75] {
            let fast = fast_cloud_vertices(
                DVec3::ZERO,
                128.0,
                cloud_tick_at(whole_tick, partial_ticks),
                Vec3::ONE,
            );
            let fast_delta = fast[0].uv[0] - base_fast[0].uv[0];
            let expected_fast_delta = partial_ticks * CLOUD_SCROLL_PER_TICK as f32 * FAST_TEXEL;
            assert!(
                (fast_delta - expected_fast_delta).abs() < 1e-7,
                "fast partial={partial_ticks}: got {fast_delta}, expected {expected_fast_delta}"
            );

            let fancy = fancy_cloud_vertices(
                DVec3::new(0.0, 64.0, 0.0),
                128.0,
                cloud_tick_at(whole_tick, partial_ticks),
                Vec3::ONE,
            );
            let fancy_delta = fancy[0].pos[0] - base_fancy[0].pos[0];
            let expected_fancy_delta = -(partial_ticks * CLOUD_SCROLL_PER_TICK as f32);
            assert!(
                (fancy_delta - expected_fancy_delta).abs() < 3e-5,
                "fancy partial={partial_ticks}: got {fancy_delta}, expected {expected_fancy_delta}"
            );
        }
    }

    #[test]
    fn cloud_scroll_coordinates_roll_over_like_java() {
        let fast_rollover = (CLOUD_COORDINATE_PERIOD / CLOUD_SCROLL_PER_TICK) as f32;
        let fast = fast_cloud_vertices(DVec3::ZERO, 128.0, fast_rollover, Vec3::ONE);
        assert!(
            fast[0].uv[0].abs() < 1.0,
            "fast cloud UV must be folded at the 2048-unit rollover: {}",
            fast[0].uv[0]
        );

        let fancy_rollover =
            (CLOUD_COORDINATE_PERIOD * f64::from(FANCY_CELL_SCALE) / CLOUD_SCROLL_PER_TICK) as f32;
        let fancy =
            fancy_cloud_vertices(DVec3::new(0.0, 64.0, 0.0), 128.0, fancy_rollover, Vec3::ONE);
        assert!(
            fancy[0].pos[0].abs() < 400.0,
            "fancy cloud mesh must remain in the rolled cell window: {}",
            fancy[0].pos[0]
        );
        assert!(
            fancy[0].uv[0] < 8.0,
            "fancy cloud UV must be folded at the 2048-unit rollover: {}",
            fancy[0].uv[0]
        );
    }

    #[test]
    fn fast_cloud_alpha_matches_vanillas_flat_08() {
        let verts = fast_cloud_vertices(DVec3::ZERO, 128.0, 0.0, Vec3::ONE);
        assert!((verts[0].color[3] - 0.8).abs() < 1e-6);
    }

    /// java: `RenderGlobal.java:1582` — one cloud unit is one texel of the 256x256 `clouds.png`,
    /// so a cell spans exactly `8/256` of the texture. The pre-fix port spanned the whole texture
    /// per cell (32x too much), which is what made clouds render as sky-wide noise.
    #[test]
    fn fancy_cloud_uv_spans_one_texel_per_cloud_unit() {
        let verts = fancy_cloud_vertices(DVec3::ZERO, 128.0, 0.0, Vec3::ONE);
        let us: Vec<f32> = verts.iter().map(|v| v.uv[0]).collect();
        let vs: Vec<f32> = verts.iter().map(|v| v.uv[1]).collect();
        let span = |xs: &[f32]| {
            xs.iter().copied().fold(f32::MIN, f32::max)
                - xs.iter().copied().fold(f32::MAX, f32::min)
        };
        // 8 cells per axis * 8 units per cell = 64 units = 64 texels = 0.25 of the texture.
        let expected = 64.0 * FANCY_TEXEL;
        assert!(
            (span(&us) - expected).abs() < 1e-4,
            "u span {} expected {expected}",
            span(&us)
        );
        assert!(
            (span(&vs) - expected).abs() < 1e-4,
            "v span {} expected {expected}",
            span(&vs)
        );
        // And a single cell's own patch is exactly 8 texels wide, never the whole sheet.
        assert!(span(&us) < 1.0, "a cell must not stretch the whole texture");
    }

    /// java: `RenderGlobal.java:1580`/`:1588` — standing below the clouds you see only their
    /// undersides; from above, only their tops.
    #[test]
    fn fancy_clouds_gate_the_top_and_bottom_faces_on_eye_height() {
        // A horizontal face is a quad whose six vertices all share one Y; the walls never do.
        let flat_quads_at = |verts: &[CloudVertex], y: f32| {
            verts
                .chunks(6)
                .filter(|q| q.iter().all(|v| (v.pos[1] - y).abs() < 1e-3))
                .count()
        };
        // Eye far below the clouds: `f3 = 128 - 64 + 0.33 = 64.33` -> bottom only.
        let below = fancy_cloud_vertices(DVec3::new(0.0, 64.0, 0.0), 128.0, 0.0, Vec3::ONE);
        let f3_below = 128.0 - 64.0 + 0.33_f32;
        assert_eq!(
            flat_quads_at(&below, f3_below + FANCY_THICKNESS - FANCY_EPSILON),
            0,
            "top face must not be emitted from below the clouds"
        );
        assert_eq!(
            flat_quads_at(&below, f3_below),
            (FANCY_CELLS * FANCY_CELLS) as usize,
            "one bottom face per cell"
        );
        // Eye far above: `f3 = 128 - 200 + 0.33 = -71.67` -> top only.
        let above = fancy_cloud_vertices(DVec3::new(0.0, 200.0, 0.0), 128.0, 0.0, Vec3::ONE);
        let f3_above = 128.0 - 200.0 + 0.33_f32;
        assert_eq!(
            flat_quads_at(&above, f3_above),
            0,
            "bottom face must not be emitted from above the clouds"
        );
        assert_eq!(
            flat_quads_at(&above, f3_above + FANCY_THICKNESS - FANCY_EPSILON),
            (FANCY_CELLS * FANCY_CELLS) as usize,
            "one top face per cell"
        );
    }

    /// java: `RenderGlobal.java:1596`-`:1638` — `j1 > -1`, `j1 <= 1`, `k1 > -1`, `k1 <= 1`. Only
    /// the outward-facing walls of each cell are emitted, which is 4 of the 8 cells per axis
    /// getting both walls and the rest getting one.
    #[test]
    fn fancy_clouds_cull_interior_walls() {
        let verts = fancy_cloud_vertices(DVec3::new(0.0, 64.0, 0.0), 128.0, 0.0, Vec3::ONE);
        // Per axis: j1 in -3..=4. `j1 > -1` fires for 0..=4 (5 cells), `j1 <= 1` for -3..=1
        // (5 cells) -> 10 wall runs of 8 quads, per k1 (8 of them). Same again for the k1 axis.
        let cells = FANCY_CELLS as usize;
        let x_walls = 10 * cells * FANCY_CELL as usize;
        let z_walls = 10 * cells * FANCY_CELL as usize;
        // Bottom faces only (eye is below), one per cell.
        let bottoms = cells * cells;
        assert_eq!(verts.len(), (bottoms + x_walls + z_walls) * 6);
    }

    #[test]
    fn fancy_face_shading_darkens_the_underside() {
        // Eye below the clouds, so the first quad emitted is a bottom face and the first wall
        // follows it. java: `RenderGlobal.java:1528`-`:1536` — bottom 0.7, X walls 0.9.
        let verts = fancy_cloud_vertices(DVec3::new(0.0, 64.0, 0.0), 128.0, 0.0, Vec3::ONE);
        let bottom_brightness = verts[0].color[0];
        assert!(
            (bottom_brightness - 0.7).abs() < 1e-6,
            "{bottom_brightness}"
        );
        let wall = verts
            .iter()
            .find(|v| (v.color[0] - 0.9).abs() < 1e-6)
            .expect("an X-side wall at 0.9");
        assert!(wall.color[0] > bottom_brightness);
    }

    /// java: `RenderGlobal.java:1545` — `scale(12.0F, 1.0F, 12.0F)` leaves Y alone, so the slab
    /// is 4 blocks thick, not 48.
    #[test]
    fn fancy_cloud_slab_is_four_blocks_thick_and_ninety_six_wide() {
        let verts = fancy_cloud_vertices(DVec3::new(0.0, 130.0, 0.0), 128.0, 0.0, Vec3::ONE);
        let ys: Vec<f32> = verts.iter().map(|v| v.pos[1]).collect();
        let hi = ys.iter().copied().fold(f32::MIN, f32::max);
        let lo = ys.iter().copied().fold(f32::MAX, f32::min);
        assert!((hi - lo - FANCY_THICKNESS).abs() < 1e-2, "{lo}..{hi}");
        // One cell is 8 cloud units * 12 blocks = 96 blocks across.
        let xs: Vec<f32> = verts.iter().map(|v| v.pos[0]).collect();
        let x_hi = xs.iter().copied().fold(f32::MIN, f32::max);
        let x_lo = xs.iter().copied().fold(f32::MAX, f32::min);
        let expected = FANCY_CELLS as f32 * FANCY_CELL * FANCY_CELL_SCALE;
        assert!((x_hi - x_lo - expected).abs() < 1.0, "{x_lo}..{x_hi}");
    }

    /// java: `RenderGlobal.java:1540` — moving the camera by a whole cloud unit (12 blocks) puts
    /// the mesh back where it started and advances the texture offset by exactly one texel, which
    /// is how vanilla keeps clouds pinned to the world while the grid stays centred on the player.
    #[test]
    fn fancy_clouds_are_world_anchored_by_the_fractional_cell_offset() {
        let a = fancy_cloud_vertices(DVec3::new(0.0, 64.0, 0.0), 128.0, 0.0, Vec3::ONE);
        let b = fancy_cloud_vertices(
            DVec3::new(f64::from(FANCY_CELL_SCALE), 64.0, 0.0),
            128.0,
            0.0,
            Vec3::ONE,
        );
        assert!(
            (a[0].pos[0] - b[0].pos[0]).abs() < 1e-3,
            "{} vs {}",
            a[0].pos[0],
            b[0].pos[0]
        );
        assert!(
            (b[0].uv[0] - a[0].uv[0] - FANCY_TEXEL).abs() < 1e-6,
            "{} vs {}",
            a[0].uv[0],
            b[0].uv[0]
        );
        // Half a cloud unit shifts the mesh, not the texture.
        let half = fancy_cloud_vertices(
            DVec3::new(f64::from(FANCY_CELL_SCALE) / 2.0, 64.0, 0.0),
            128.0,
            0.0,
            Vec3::ONE,
        );
        assert!((half[0].uv[0] - a[0].uv[0]).abs() < 1e-6);
        assert!((half[0].pos[0] - a[0].pos[0]).abs() > 1.0);
    }

    #[test]
    fn cloud_mode_maps_from_settings() {
        assert_eq!(CloudMode::from_setting(0), CloudMode::Off);
        assert_eq!(CloudMode::from_setting(1), CloudMode::Fast);
        assert_eq!(CloudMode::from_setting(2), CloudMode::Fancy);
    }

    /// java: `renderWorldPass` — `posY + getEyeHeight() < 128.0D` draws
    /// the clouds before terrain, `>= 128.0D` after the world; nothing fed counts as below.
    #[test]
    fn the_eye_splits_the_cloud_points_at_vanillas_literal_128() {
        assert!(!eye_above_clouds(None));
        assert!(!eye_above_clouds(Some(127.999_999)));
        assert!(eye_above_clouds(Some(128.0)));
        assert!(eye_above_clouds(Some(1000.0)));
        assert!(!eye_above_clouds(Some(-64.0)));
        // java: `posY + (double)1.62F` with the feet at 126.38 lands on the upper side.
        assert!(eye_above_clouds(Some(126.38 + f64::from(1.62_f32))));
    }

    /// The after-the-world point comes after everything java draws before `aboveClouds`
    /// (translucent terrain, the weather) and after the screen-space fog at either of its
    /// orders, and before the anti-aliasing and the hand.
    #[test]
    fn the_above_clouds_order_sits_after_the_world_and_before_the_resolve() {
        for earlier in [
            PhaseOrder::SKY,
            PhaseOrder::TERRAIN_SOLID,
            PhaseOrder::PARTICLES,
            PhaseOrder::WEATHER,
            FogPhase::BEFORE_TRANSLUCENT_ORDER,
            PhaseOrder::TERRAIN_TRANSLUCENT,
            FogPhase::ORDER,
        ] {
            assert!(ABOVE_CLOUDS_ORDER > earlier, "{earlier:?}");
        }
        assert!(ABOVE_CLOUDS_ORDER < PhaseOrder::TEMPORAL_AA);
        assert!(ABOVE_CLOUDS_ORDER < PhaseOrder::FIRST_PERSON);
        assert_eq!(ABOVE_CLOUDS_ORDER, crate::sky::volumetric::LATE_ORDER);
    }

    /// CPU mirror of `clouds.wgsl`'s `vs_fancy`: pure table selects plus the one UV addition.
    fn shade_fancy(
        lattice: &[FancyCloudVertex],
        table: &[[f32; 4]; FANCY_TABLE_ROWS],
    ) -> Vec<CloudVertex> {
        let axis = |base: usize, i: usize| table[base + i / 4][i % 4];
        lattice
            .iter()
            .map(|v| {
                let xi = (v.index & 0xff) as usize;
                let zi = ((v.index >> 8) & 0xff) as usize;
                let yi = ((v.index >> 16) & 0xff) as usize;
                let ci = ((v.index >> 24) & 0xff) as usize;
                CloudVertex {
                    pos: [
                        axis(0, xi),
                        table[FANCY_TABLE_Y_ROW][yi],
                        axis(FANCY_TABLE_AXIS_ROWS, zi),
                    ],
                    uv: [
                        v.uv[0] + table[FANCY_TABLE_UV_ROW][0],
                        v.uv[1] + table[FANCY_TABLE_UV_ROW][1],
                    ],
                    color: table[FANCY_TABLE_COLOR_ROW + ci],
                }
            })
            .collect()
    }

    fn cloud_bits(vertices: &[CloudVertex]) -> Vec<[u32; 9]> {
        vertices
            .iter()
            .map(|v| {
                let [x, y, z] = v.pos.map(f32::to_bits);
                let [u, w] = v.uv.map(f32::to_bits);
                let [r, g, b, a] = v.color.map(f32::to_bits);
                [x, y, z, u, w, r, g, b, a]
            })
            .collect()
    }

    /// #17 / SKY-1: the resident lattice plus the per-frame table must hand the vertex stage the
    /// exact vertices — order, positions, UVs and colours, bit for bit — that the per-frame
    /// `renderCloudsFancy` rebuild produced. Covers all three face-gate combinations, fractional
    /// and far-from-origin cameras, scroll rollover and non-white (night, anaglyph) tints.
    #[test]
    fn resident_fancy_lattice_reproduces_the_per_frame_mesh_bit_for_bit() {
        let cameras = [
            DVec3::new(0.0, 64.0, 0.0),
            DVec3::new(5.3, 70.62, -11.9),
            DVec3::new(-1234.567, 100.0, 9876.5432),
            DVec3::new(29_999_983.25, 255.0, -29_999_983.75),
            DVec3::new(0.0, 126.0, 0.0),
            DVec3::new(0.0, 128.0, 0.0),
            DVec3::new(0.0, 132.66, 0.0),
            DVec3::new(0.0, 133.33, 0.0),
            DVec3::new(-17.0, 200.0, 3.5),
            DVec3::new(6.0, -40.0, 6.0),
        ];
        let ticks = [0.0, 0.37, 1234.5, 68_266.67, 1.0e7];
        let colors = [
            Vec3::ONE,
            Vec3::new(0.11, 0.12, 0.15),
            cloud_color_for_render(Vec3::new(0.9, 0.85, 0.7), true),
        ];
        let mut gate_sets = std::collections::BTreeSet::new();
        for camera in cameras {
            for tick in ticks {
                for color in colors {
                    let old = fancy_cloud_vertices(camera, DEFAULT_CLOUD_HEIGHT, tick, color);
                    let frame = FancyCloudFrame::new(camera, DEFAULT_CLOUD_HEIGHT, tick);
                    let (bottom, top) = frame.face_gates();
                    gate_sets.insert((bottom, top));
                    let lattice = fancy_cloud_lattice(bottom, top);
                    let new = shade_fancy(&lattice, &frame.table(color));
                    assert_eq!(new.len(), old.len(), "camera={camera:?} tick={tick}");
                    assert!(
                        cloud_bits(&new) == cloud_bits(&old),
                        "camera={camera:?} tick={tick} color={color:?}"
                    );
                }
            }
        }
        // Below, inside and above the slab: all three gate combinations were exercised.
        assert_eq!(gate_sets.len(), 3, "{gate_sets:?}");
    }

    #[test]
    fn fancy_table_fits_its_wgsl_struct_and_indices_fit_their_bytes() {
        // `clouds.wgsl`: xs[34] + zs[34] + ys + uv_offset + colors[4].
        assert_eq!(FANCY_TABLE_AXIS_ROWS, 34);
        assert_eq!(FANCY_TABLE_ROWS, 74);
        assert_eq!(FANCY_TABLE_ROWS * 16, 1184);
        let source = include_str!("clouds.wgsl");
        assert!(source.contains("xs: array<vec4<f32>, 34>,"));
        assert!(source.contains("zs: array<vec4<f32>, 34>,"));
        let full = fancy_cloud_lattice(true, true);
        // 64 undersides + 64 tops + 4 x 320 walls, as six vertices each.
        assert_eq!(full.len(), (64 + 64 + 4 * 320) * 6);
        assert!(full.iter().all(|v| (v.index & 0xff) < 136
            && ((v.index >> 8) & 0xff) < 136
            && ((v.index >> 16) & 0xff) < 3
            && (v.index >> 24) < 4));
    }
}
