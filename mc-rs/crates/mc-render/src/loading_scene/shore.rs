//! The neon night shore: one of the two animated loading screens.
//!
//! Waves break on a beach of vanilla sand blocks at night. Most of each wave is dark water; its
//! crest carries a ribbon of neon light (cyan in front, then blue, yellow, orange and pink, the
//! colours of the user's light-trail photo). A sea fog swallows the horizon, the white water at
//! the break and along the swash's edge is lace-like foam, and film grain lies over everything.
//!
//! **The colour push** (the user's rule, read closely): when a wave's swash comes to rest at its
//! farthest point on the beach, just before it drains back, two of the trailing colours surge
//! forward through the bands ahead of them, each with its own lobed wave shape. They cover the cyan
//! front band and, once the front has started to drain back, the sand it uncovered, but never the
//! border the cyan front drew (the farthest line it reached so far): they are clamped to it. Where
//! they arrive they replace the colour that was there (they take the cyan's colour away, they do
//! not add light to it). The cyan line left at that border fades out as the wave drains or as a
//! pushing colour takes it over. `tests/tw_lb_shore_render.rs` checks the rule on rendered frames.
//!
//! # How it is drawn
//!
//! Everything is analytic, so there is no geometry and no simulation state: one full-screen pass
//! ray-casts the scene into an HDR target (a 2D DDA through a height map of sand-block columns,
//! and a few fixed-point steps onto the wave surface), a five-level bloom chain spreads the neon,
//! and a composite pass tone maps, grades and adds the grain. The block textures are vanilla's
//! `sand.png`, `gravel.png` and `clay.png` from the 1.8.9 jar ([`JAR_TEXTURES`]; the loading
//! screen shows while the game's own asset load is still running). The ray cast runs at 0.75x the
//! frame's size (at most 1440x810) and the composite upscales it with a Catmull-Rom filter; the
//! grain is added at the frame's own size.
//!
//! It plugs into the loading screen as a [`Scene`] ([`super::SceneKind::Shore`]).
//!
//! # Cost
//!
//! Only while it is drawn: the curtain drops the scene when the loading screen is gone, and every
//! GPU resource goes with it. Per frame: the ray cast (at 0.75x, see `SCENE_SCALE`), 9 small
//! bloom passes, the composite and one 1.2 KB uniform write (the camera plus the table of the
//! waves in view); no allocation after the first frame at a size. Measured about 0.46 ms of GPU
//! time at 1080p on an RTX 5070 Ti, the curtain included.

use glam::Vec3;

use super::{JarTextures, Scene, SceneFrame, SceneInit, SCENE_FORMAT};
use crate::Gpu;

const SHADER: &str = include_str!("shore.wgsl");
const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BLOOM_LEVELS: usize = 5;
/// The ray cast's resolution, as a fraction of the frame's width and height (see
/// [`internal_size`]): 0.75 halves its cost, measured, with no visible loss under the grain.
const SCENE_SCALE: f64 = 0.75;
/// The most pixels the scene ray-casts (the composite upscales beyond).
const MAX_SCENE_PIXELS: f64 = 1440.0 * 810.0;
/// The noise texture's size (tileable).
const NOISE: u32 = 256;
/// The height map: world x from `HMAP_X0`, z from `HMAP_Z0` (wrapping in z). Must match
/// `shore.wgsl`.
const HMAP_X0: i32 = -64;
const HMAP_Z0: i32 = -64;
const HMAP_W: u32 = 160;
const HMAP_H: u32 = 256;
const BLOCK_MIPS: u32 = 5;

/// The vanilla textures the shore reads from the jar (relative to `assets/minecraft/`), in the
/// order of the block ids in its height map: sand, gravel, clay.
pub const JAR_TEXTURES: &[&str] = &[
    "textures/blocks/sand.png",
    "textures/blocks/gravel.png",
    "textures/blocks/clay.png",
];

// ---- the waves' timing, mirrored from `shore.wgsl` (a test checks the constants match)
const PERIOD: f32 = 5.4;
const C: f32 = 3.8;
const U0: f32 = 62.0;
const OBL: f32 = 0.055;
/// Where along the shore (`v`, world z) the middle of the picture is.
const V_FOCUS: f32 = -6.0;
/// How long after the first frame the first wave comes to rest there (the colour push). We
/// measured a warm start at about 0.6 s and set a 2 s minimum, so the loading screen usually
/// shows for about 2 s: at 1.2 s the swash is running up on the first frame (the curtain fades in
/// over its first 0.35 s), the colours surge from about 0.5 s and have pushed past the draining
/// front to the border by 2 s.
pub const FIRST_REST_S: f64 = 1.2;

/// The vanilla block textures the shore uses, 16x16 RGBA each.
#[derive(Debug, Clone)]
pub struct ShoreTextures {
    layers: Vec<Vec<u8>>,
    vanilla: usize,
}

impl ShoreTextures {
    /// The textures from what the loading screen read out of the jar; any that is missing (or
    /// not 16x16) gets a plain stand-in.
    #[must_use]
    pub fn from_jar_textures(jar: &JarTextures) -> ShoreTextures {
        let fallback = ShoreTextures::fallback();
        let mut vanilla = 0;
        let layers = JAR_TEXTURES
            .iter()
            .zip(fallback.layers)
            .map(|(path, stand_in)| match jar.get(path) {
                Some(image) if image.width == 16 && image.height == 16 => {
                    vanilla += 1;
                    image.rgba.clone()
                }
                _ => stand_in,
            })
            .collect();
        ShoreTextures { layers, vanilla }
    }

    /// Read the textures straight from the vanilla jar at `jar` (render harnesses).
    #[must_use]
    pub fn from_jar(jar: &std::path::Path) -> ShoreTextures {
        ShoreTextures::from_jar_textures(&JarTextures::load(jar, JAR_TEXTURES))
    }

    /// Plain sand-coloured stand-ins, for when the jar cannot be read.
    #[must_use]
    pub fn fallback() -> ShoreTextures {
        let tint: [[f32; 3]; 3] = [
            [219.0, 211.0, 160.0],
            [136.0, 126.0, 126.0],
            [160.0, 166.0, 179.0],
        ];
        let layers = tint
            .iter()
            .enumerate()
            .map(|(layer, rgb)| {
                let mut px = Vec::with_capacity(16 * 16 * 4);
                for i in 0..256_i32 {
                    let n = (hash(i, layer as i32, 3) >> 24) as f32 / 255.0;
                    let k = 0.86 + 0.2 * n;
                    for c in rgb {
                        px.push((c * k).clamp(0.0, 255.0) as u8);
                    }
                    px.push(255);
                }
                px
            })
            .collect();
        ShoreTextures { layers, vanilla: 0 }
    }

    /// How many of the three are vanilla's own (the rest are stand-ins).
    #[must_use]
    pub fn vanilla_count(&self) -> usize {
        self.vanilla
    }
}

/// The grading knobs. The defaults are the tuned look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShoreLook {
    /// Scene exposure before the tone map.
    pub exposure: f32,
    /// Film grain strength (display-space amplitude).
    pub grain: f32,
    /// How much of the bloom chain is added.
    pub bloom: f32,
    /// How many times a second the grain changes.
    pub grain_hz: f32,
    /// How much the area behind the loading screen's logo is darkened, `0..=1`.
    pub logo_shade: f32,
}

impl Default for ShoreLook {
    fn default() -> Self {
        ShoreLook {
            exposure: 1.5,
            grain: 0.10,
            bloom: 0.24,
            grain_hz: 24.0,
            logo_shade: 0.25,
        }
    }
}

/// A top-down map of one wave's neon ribbon instead of the scene: the tests read it back to check
/// the colour push, and the report shows it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagram {
    /// The wave (its index in the scene's endless sequence; see [`ShoreScene::focus_wave`]).
    pub wave: i32,
    /// Write test channels (R = the cyan front band as laid out, G = a pushing colour) instead of
    /// the colours.
    pub test_channels: bool,
    /// Clamp the pushing colours to the border (always on in the scene; off only to prove the
    /// test can see a violation).
    pub clamp: bool,
    /// `u` (blocks seaward of the terrace edge) at the top row.
    pub u_top: f32,
    /// `u` from the top row to the bottom row.
    pub u_span: f32,
    /// `v` (blocks along the shore) at the left column.
    pub v_left: f32,
    /// `v` from the left column to the right.
    pub v_span: f32,
}

/// Must match `struct Frame` in `shore.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniform {
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    screen: [f32; 4],
    out_size: [f32; 4],
    look: [f32; 4],
    extra: [f32; 4],
    map: [f32; 4],
    logo: [f32; 4],
    /// x = the first wave in `waves`, y = [`ShoreScene::x_land`].
    table: [f32; 4],
    /// [`WAVE_SLOTS`] waves' [`wave_params`].
    waves: [[f32; 4]; WAVE_SLOTS * 4],
}

struct Targets {
    size: (u32, u32),
    scene: wgpu::TextureView,
    bloom: Vec<wgpu::TextureView>,
    /// Bind groups: scene → bloom 0, bloom i → i + 1 (down), bloom i + 1 → i (up), composite.
    g_first: wgpu::BindGroup,
    g_down: Vec<wgpu::BindGroup>,
    g_up: Vec<wgpu::BindGroup>,
    g_composite: wgpu::BindGroup,
    g_none: wgpu::BindGroup,
}

struct Output {
    key: (wgpu::TextureFormat, u32),
    composite: wgpu::RenderPipeline,
}

/// The shore scene's GPU side. Main-thread only. Drop it to free everything.
pub struct ShoreScene {
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    input_layout: wgpu::BindGroupLayout,
    frame_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    dummy: wgpu::TextureView,
    scene_pipe: wgpu::RenderPipeline,
    bloom_first: wgpu::RenderPipeline,
    bloom_down: wgpu::RenderPipeline,
    bloom_up: wgpu::RenderPipeline,
    output: Option<Output>,
    targets: Option<Targets>,
    _textures: Vec<wgpu::Texture>,
    /// Seconds since the first frame.
    time: f64,
    /// The waves' clock at the first frame (from the run's seed).
    start: f64,
    focus_wave: i32,
    look: ShoreLook,
    diagram: Option<Diagram>,
    logo: [f32; 4],
    /// Seaward of this x no block rises above the water (the ray cast skips that part).
    x_land: f32,
    /// The uniform's wave table and the first wave in it (rebuilt when the first changes).
    table: ([[f32; 4]; WAVE_SLOTS * 4], i32),
}

impl std::fmt::Debug for ShoreScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShoreScene")
            .field("time", &self.time)
            .field("start", &self.start)
            .field("look", &self.look)
            .field("targets", &self.targets.as_ref().map(|t| t.size))
            .finish_non_exhaustive()
    }
}

/// A 32-bit integer hash (deterministic, platform independent).
fn hash(x: i32, y: i32, salt: u32) -> u32 {
    let h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ salt.wrapping_mul(0xcb1a_b31f);
    hash_u(h)
}

/// A 32-bit integer finaliser (lowbias32).
fn hash_u(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

/// Wave `i`'s random number `salt`, `0..1`.
fn rnd(i: i32, salt: u32) -> f32 {
    let h = hash_u(
        (i as u32).wrapping_mul(0x9e37_79b9)
            ^ salt.wrapping_mul(0x85eb_ca6b).wrapping_add(0x0123_4567),
    );
    (h >> 8) as f32 / 16_777_216.0
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

/// How many waves the shader's table holds (`WAVE_SLOTS` in `shore.wgsl`).
const WAVE_SLOTS: usize = 16;
/// Waves the shader evaluates per pixel (`NWAVES` in `shore.wgsl`).
const NWAVES: i32 = 7;

/// Wave `i`'s parameters, packed as the shader's `wave` unpacks them: `[t0, obl, smax, lam]`,
/// `[ph1, ph2, ph3, gain]`, `[p1, p2, lam1, lam2]`, `[ph4, ph5, 0, 0]`. The one place their
/// formulas live: when it spawns, how oblique it runs, how far up the beach it reaches, its
/// run-up lobes, its brightness, and which two trailing colours push at the run-up (yellow and
/// pink, orange and blue, yellow and orange, or pink and blue) with their own lobes.
fn wave_params(i: i32) -> [[f32; 4]; 4] {
    use std::f32::consts::TAU;
    let (p1, p2) = match (rnd(i, 9) * 4.0) as i32 {
        0 => (2.0, 4.0),
        1 => (3.0, 1.0),
        2 => (2.0, 3.0),
        _ => (4.0, 1.0),
    };
    [
        [
            i as f32 * PERIOD + (rnd(i, 1) - 0.5) * 1.4,
            OBL + (rnd(i, 2) - 0.5) * 0.05,
            7.0 + rnd(i, 3) * 3.2,
            11.0 + rnd(i, 4) * 7.0,
        ],
        [
            rnd(i, 5) * TAU,
            rnd(i, 6) * TAU,
            rnd(i, 7) * TAU,
            0.85 + 0.3 * rnd(i, 8),
        ],
        [p1, p2, 5.0 + rnd(i, 10) * 5.0, 4.0 + rnd(i, 11) * 5.0],
        [rnd(i, 12) * TAU, rnd(i, 13) * TAU, 0.0, 0.0],
    ]
}

/// How far along the shore, ahead of the camera, the wave table reaches (blocks): well into the
/// fog. At the first frame (the camera at z = -32) this is the earlier fixed window's end,
/// the map's length plus 64.
const VIEW_AHEAD: f32 = 288.0;
/// How far behind the camera the table reaches (blocks; nothing behind it is in view). The
/// table's slack covers it ([`table_base`]; checked by the coverage test).
#[cfg(test)]
const VIEW_BEHIND: f32 = 72.0;

/// The first wave of the table the shader gets at waves' time `clock` with the camera at
/// `eye_z` along the shore: every wave a pixel in view can ask for (`newest_wave` in
/// `shore.wgsl` from [`VIEW_BEHIND`] behind the camera to [`VIEW_AHEAD`] ahead of it, and the
/// `NWAVES - 1` before each) is in it, with one wave to spare for the GPU rounding a floor the
/// other way.
///
/// The window follows the camera. It was fixed to the map's
/// first 256 blocks, which held for a loading screen of a few seconds; as a title-screen scene the
/// shore can stay up for an hour while the camera drifts along it at 0.3 blocks a second, and
/// after about ten minutes the far end of the view asked for older waves that were no longer in
/// the table (their swash, foam and wet sand would vanish there).
fn table_base(clock: f32, eye_z: f32) -> i32 {
    let newest = |v: f32| ((clock - v * OBL) / PERIOD).floor() as i32 + 1;
    newest(eye_z + VIEW_AHEAD) - NWAVES
}

/// When wave `i`'s swash comes to rest at `v` along the shore (its farthest point up the beach,
/// the moment of the colour push), on the waves' clock. Mirrors `local` in `shore.wgsl`.
#[must_use]
pub fn rest_time(i: i32, v: f32) -> f64 {
    use std::f32::consts::TAU;
    let [[t0, obl, smax_mean, lam], [ph1, ph2, ph3, _], _, _] = wave_params(i);
    let wobble = 1.2 * (v * 0.083 + ph1).sin() + 0.6 * (v * 0.21 + ph2).sin();
    let tau0 = (U0 + wobble) / C;
    let s1 = (TAU * v / lam + ph3).sin();
    let s2 = (TAU * v / (0.41 * lam) + 2.0 * ph3).sin();
    let lobe = 0.62 * s1 + 0.26 * s2 + 0.12 * (TAU * v / (0.17 * lam) + 3.0 * ph3).sin();
    let smax = smax_mean * (1.0 + 0.10 * lobe);
    let taum = tau0 + 2.0 * smax / C;
    f64::from(t0) + f64::from(v * obl) + f64::from(taum)
}

/// Smooth 1D value noise in `-1..1` with a feature every `cell` blocks, periodic over
/// [`HMAP_H`] so the map wraps without a seam.
fn edge_noise(z: i32, cell: i32, salt: u32) -> f32 {
    let cells = (HMAP_H as i32 / cell).max(1);
    let x = z.rem_euclid(HMAP_H as i32) as f32 / cell as f32;
    let i = x.floor() as i32;
    let f = x - x.floor();
    let f = f * f * (3.0 - 2.0 * f);
    let a = unit(hash(i.rem_euclid(cells), 0, salt)) * 2.0 - 1.0;
    let b = unit(hash((i + 1).rem_euclid(cells), 0, salt)) * 2.0 - 1.0;
    a + (b - a) * f
}

/// Smooth 2D value noise in `0..1` with a feature every `cell` blocks.
fn patch_noise(x: i32, z: i32, cell: i32, salt: u32) -> f32 {
    let fx = x as f32 / cell as f32;
    let fz = z as f32 / cell as f32;
    let (ix, iz) = (fx.floor() as i32, fz.floor() as i32);
    let (tx, tz) = (fx - fx.floor(), fz - fz.floor());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sz) = (s(tx), s(tz));
    let g = |a: i32, b: i32| unit(hash(a, b, salt));
    let top = g(ix, iz) + (g(ix + 1, iz) - g(ix, iz)) * sx;
    let bottom = g(ix, iz + 1) + (g(ix + 1, iz + 1) - g(ix, iz + 1)) * sx;
    top + (bottom - top) * sz
}

/// The beach as block columns: `(top height, block)` for the column at world `(x, z)`.
///
/// Dry sand terraces step up inland (heights 1 to 3); the swash terrace's tops are at y = 0, just
/// above still water; out to sea the bed steps down one block at a time. Every edge is jagged the
/// way a Minecraft beach is, a few cells of the swash terrace are dips that hold water, and gravel
/// and clay patch the deeper bed.
#[must_use]
pub fn column(x: i32, z: i32) -> (i32, u8) {
    let xc = x as f32 + 0.5;
    let edge = |base: f32, amp: f32, cell: i32, salt: u32| {
        base + amp * edge_noise(z, cell, salt)
            + amp * 0.45 * edge_noise(z, (cell / 3).max(2), salt + 50)
    };
    let step = edge(-12.5, 1.5, 8, 3);
    let steps = [
        (edge(-31.0, 3.0, 12, 1), 3),
        (edge(-21.0, 2.4, 10, 2), 2),
        (step, 1),
        (edge(0.0, 0.7, 7, 4), 0),
        (edge(6.5, 1.8, 9, 5), -1),
        (edge(15.0, 2.6, 11, 6), -2),
        (edge(27.0, 3.5, 13, 7), -3),
        (edge(42.0, 4.0, 16, 8), -4),
    ];
    let mut h = -5;
    for (limit, height) in steps {
        if xc < limit {
            h = height;
            break;
        }
    }
    // Dips in the swash terrace that keep a little water (not right at the step).
    if h == 0 && patch_noise(x, z, 3, 21) > 0.87 && xc > step + 1.5 {
        h = -1;
    }
    let mut block = 0_u8;
    if h <= -2 {
        if patch_noise(x, z, 5, 31) > 0.68 {
            block = 1;
        } else if patch_noise(x, z, 6, 37) > 0.74 {
            block = 2;
        }
    }
    (h, block)
}

/// The height map texture's texels (`Rg8Uint`: height + 16, block), and the x seaward of which no
/// column reaches the water (every top is below y = 0).
fn height_map() -> (Vec<u8>, f32) {
    let mut out = Vec::with_capacity((HMAP_W * HMAP_H * 2) as usize);
    let mut x_land = HMAP_X0;
    for zi in 0..HMAP_H as i32 {
        for xi in 0..HMAP_W as i32 {
            let (h, block) = column(HMAP_X0 + xi, HMAP_Z0 + zi);
            out.push((h + 16).clamp(0, 255) as u8);
            out.push(block);
            if h >= 0 {
                x_land = x_land.max(HMAP_X0 + xi + 1);
            }
        }
    }
    (out, x_land as f32)
}

/// The tileable noise texture (`NOISE` square, RGBA8): R = foam density (strands between
/// bubbles), G = smooth fractal noise (the water's ripples), B = unused (0), A = white noise
/// (the film grain).
fn noise_pixels() -> Vec<u8> {
    let n = NOISE as usize;
    // Distances to the two nearest feature points of a periodic cellular pattern.
    let worley = |x: f32, y: f32, cells: i32, salt: u32| -> (f32, f32) {
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (mut f1, mut f2) = (9.0_f32, 9.0_f32);
        for oy in -1..=1 {
            for ox in -1..=1 {
                let (cx, cy) = (ix + ox, iy + oy);
                let h = hash(cx.rem_euclid(cells), cy.rem_euclid(cells), salt);
                let px = cx as f32 + 0.1 + 0.8 * (h & 0xFFFF) as f32 / 65_536.0;
                let py = cy as f32 + 0.1 + 0.8 * (h >> 16) as f32 / 65_536.0;
                let d = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
        (f1, f2)
    };
    let perlin = |x: f32, y: f32, period: i32, salt: u32| -> f32 {
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let dot = |gx: i32, gy: i32, dx: f32, dy: f32| {
            let h = hash(gx.rem_euclid(period), gy.rem_euclid(period), salt);
            let a = unit(h) * std::f32::consts::TAU;
            a.cos() * dx + a.sin() * dy
        };
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (u, v) = (fade(fx), fade(fy));
        let a = dot(ix, iy, fx, fy);
        let b = dot(ix + 1, iy, fx - 1.0, fy);
        let c = dot(ix, iy + 1, fx, fy - 1.0);
        let d = dot(ix + 1, iy + 1, fx - 1.0, fy - 1.0);
        let top = a + (b - a) * u;
        let bottom = c + (d - c) * u;
        (top + (bottom - top) * v) * 0.7 + 0.5
    };
    // The foam density: a warped network of strands between bubbles at three sizes, torn by a
    // softer fractal. Rank-normalised afterwards, so "the top 30% of the values" covers exactly
    // 30% of the texture: the shader's foam amount then maps linearly to coverage.
    let mut foam = vec![0.0_f32; n * n];
    for y in 0..n {
        for x in 0..n {
            let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
            let wx = perlin(u * 4.0, v * 4.0, 4, 201) - 0.5;
            let wy = perlin(u * 4.0 + 0.5, v * 4.0 + 0.3, 4, 202) - 0.5;
            let (wu, wv) = (u + wx * 0.07, v + wy * 0.07);
            let strand = |cells: f32, salt: u32, width: f32| {
                let (f1, f2) = worley(wu * cells, wv * cells, cells as i32, salt);
                1.0 - ((f2 - f1) / width).clamp(0.0, 1.0)
            };
            let big = strand(9.0, 301, 0.42);
            let mid = strand(19.0, 302, 0.38);
            let small = strand(41.0, 303, 0.34);
            let torn =
                perlin(u * 8.0, v * 8.0, 8, 304) * 0.6 + perlin(u * 16.0, v * 16.0, 16, 305) * 0.4;
            foam[y * n + x] = big * 0.42 + mid * 0.33 + small * 0.25 + (torn - 0.5) * 0.55;
        }
    }
    let mut order: Vec<u32> = (0..(n * n) as u32).collect();
    order.sort_by(|&a, &b| foam[a as usize].total_cmp(&foam[b as usize]));
    let mut rank = vec![0.0_f32; n * n];
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as f32 / (n * n - 1) as f32;
    }
    let mut out = vec![0_u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
            let fbm = perlin(u * 8.0, v * 8.0, 8, 102) * 0.5
                + perlin(u * 16.0, v * 16.0, 16, 103) * 0.3
                + perlin(u * 32.0, v * 32.0, 32, 104) * 0.2;
            let white = unit(hash(x as i32, y as i32, 106));
            let i = (y * n + x) * 4;
            for (k, c) in [rank[y * n + x], fbm, 0.0, white].into_iter().enumerate() {
                out[i + k] = (c.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let c = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round() as u8
}

/// 2x2 box-filtered mips of a square RGBA8 image, level 0 first (colour averaged in linear light
/// when `srgb`).
fn mip_chain(level0: Vec<u8>, size: u32, levels: u32, srgb: bool) -> Vec<Vec<u8>> {
    let mut chain = vec![level0];
    let mut s = size as usize;
    for _ in 1..levels {
        let prev = chain.last().expect("level 0");
        let half = (s / 2).max(1);
        let mut next = vec![0_u8; half * half * 4];
        for y in 0..half {
            for x in 0..half {
                for k in 0..4 {
                    let at =
                        |px: usize, py: usize| prev[(py.min(s - 1) * s + px.min(s - 1)) * 4 + k];
                    let four = [
                        at(2 * x, 2 * y),
                        at(2 * x + 1, 2 * y),
                        at(2 * x, 2 * y + 1),
                        at(2 * x + 1, 2 * y + 1),
                    ];
                    next[(y * half + x) * 4 + k] = if srgb && k < 3 {
                        linear_to_srgb(four.iter().map(|&c| srgb_to_linear(c)).sum::<f32>() * 0.25)
                    } else {
                        ((four.iter().map(|&c| u32::from(c)).sum::<u32>() + 2) / 4) as u8
                    };
                }
            }
        }
        chain.push(next);
        s = half;
    }
    chain
}

/// The camera at `t` seconds since the first frame: high above the swash terrace, looking down
/// the beach into the fog with the sea on the left, drifting slowly along the shore.
fn shot(t: f32) -> (Vec3, Vec3, f32) {
    let eye = Vec3::new(
        7.0 + 0.3 * (t * 0.09).sin(),
        11.5 + 0.2 * (t * 0.07).sin(),
        -32.0 + 0.30 * t,
    );
    let yaw: f32 = -0.33 + 0.012 * (t * 0.05).sin();
    let pitch: f32 = 0.40;
    let fwd = Vec3::new(
        yaw.sin() * pitch.cos(),
        -pitch.sin(),
        yaw.cos() * pitch.cos(),
    );
    (eye, fwd, 50_f32.to_radians())
}

/// The ray cast's size for a `width` x `height` frame: [`SCENE_SCALE`] of it, and at most
/// [`MAX_SCENE_PIXELS`]. The composite upscales it with a Catmull-Rom filter and adds the grain at
/// the frame's own size, which hides the difference under the fog and the glow.
fn internal_size(width: u32, height: u32) -> (u32, u32) {
    let pixels = f64::from(width) * f64::from(height);
    let s = SCENE_SCALE.min((MAX_SCENE_PIXELS / pixels.max(1.0)).sqrt());
    (
        ((f64::from(width) * s).round() as u32).max(1),
        ((f64::from(height) * s).round() as u32).max(1),
    )
}

fn texture_2d(
    gpu: &Gpu,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    (mips, layers): (u32, u32),
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: layers,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// Upload one mip of one layer (`bpp` bytes per texel).
fn upload(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    (mip, layer): (u32, u32),
    size: (u32, u32),
    bpp: u32,
    data: &[u8],
) {
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.0 * bpp),
            rows_per_image: Some(size.1),
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
}

fn fullscreen(
    enc: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    pipeline: &wgpu::RenderPipeline,
    groups: [&wgpu::BindGroup; 2],
) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, groups[0], &[]);
    pass.set_bind_group(1, groups[1], &[]);
    pass.draw(0..3, 0..1);
}

#[allow(clippy::too_many_arguments)]
fn pipeline(
    gpu: &Gpu,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    label: &str,
    fs: &str,
    format: wgpu::TextureFormat,
    samples: u32,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: samples,
                ..wgpu::MultisampleState::default()
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(fs),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

impl ShoreScene {
    /// The scene for the loading screen ([`super::SceneKind::Shore`]).
    #[must_use]
    pub fn new(gpu: &Gpu, init: &SceneInit<'_>) -> ShoreScene {
        ShoreScene::with_textures(
            gpu,
            &ShoreTextures::from_jar_textures(init.textures),
            init.seed,
        )
    }

    /// The scene from textures already read; `seed` varies the waves from run to run (the first
    /// colour push still comes [`FIRST_REST_S`] in).
    #[allow(clippy::too_many_lines)] // One constructor: textures, layouts, pipelines.
    #[must_use]
    pub fn with_textures(gpu: &Gpu, textures: &ShoreTextures, seed: u32) -> ShoreScene {
        let device = &gpu.device;
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;

        let blocks = texture_2d(
            gpu,
            "shore blocks",
            (16, 16),
            wgpu::TextureFormat::Rgba8UnormSrgb,
            (BLOCK_MIPS, JAR_TEXTURES.len() as u32),
            sampled,
        );
        for (layer, pixels) in textures.layers.iter().enumerate() {
            for (mip, data) in mip_chain(pixels.clone(), 16, BLOCK_MIPS, true)
                .iter()
                .enumerate()
            {
                let s = (16_u32 >> mip).max(1);
                upload(gpu, &blocks, (mip as u32, layer as u32), (s, s), 4, data);
            }
        }
        let heights = texture_2d(
            gpu,
            "shore height map",
            (HMAP_W, HMAP_H),
            wgpu::TextureFormat::Rg8Uint,
            (1, 1),
            sampled,
        );
        let (height_texels, x_land) = height_map();
        upload(gpu, &heights, (0, 0), (HMAP_W, HMAP_H), 2, &height_texels);
        let noise_mips = NOISE.ilog2() + 1;
        let noise = texture_2d(
            gpu,
            "shore noise",
            (NOISE, NOISE),
            wgpu::TextureFormat::Rgba8Unorm,
            (noise_mips, 1),
            sampled,
        );
        for (mip, data) in mip_chain(noise_pixels(), NOISE, noise_mips, false)
            .iter()
            .enumerate()
        {
            let s = (NOISE >> mip).max(1);
            upload(gpu, &noise, (mip as u32, 0), (s, s), 4, data);
        }
        let dummy_tex = texture_2d(
            gpu,
            "shore dummy",
            (1, 1),
            HDR,
            (1, 1),
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = |label: &str, address: wgpu::AddressMode, filter: wgpu::FilterMode, mip| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: address,
                address_mode_v: address,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter: mip,
                ..wgpu::SamplerDescriptor::default()
            })
        };
        // Minecraft's look: nearest texels up close, mip-mapped far away.
        let block_sampler = sampler(
            "shore block sampler",
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Nearest,
            wgpu::MipmapFilterMode::Linear,
        );
        let noise_sampler = sampler(
            "shore noise sampler",
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        );
        let lin_sampler = sampler(
            "shore linear sampler",
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Nearest,
        );

        let frag = wgpu::ShaderStages::FRAGMENT;
        let tex = |binding: u32, sample_type, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let samp = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let d2 = wgpu::TextureViewDimension::D2;
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shore group0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex(1, float, wgpu::TextureViewDimension::D2Array),
                samp(2),
                tex(3, wgpu::TextureSampleType::Uint, d2),
                tex(4, float, d2),
                samp(5),
                samp(6),
            ],
        });
        let input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shore group1"),
            entries: &[tex(0, float, d2), tex(1, float, d2)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shore layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(&input_layout)],
            immediate_size: 0,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shore frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blocks_view = blocks.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..wgpu::TextureViewDescriptor::default()
        });
        let heights_view = heights.create_view(&wgpu::TextureViewDescriptor::default());
        let noise_view = noise.create_view(&wgpu::TextureViewDescriptor::default());
        let view_entry = |binding: u32, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        let sampler_entry = |binding: u32, sampler| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Sampler(sampler),
        };
        let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shore group0"),
            layout: &frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                view_entry(1, &blocks_view),
                sampler_entry(2, &block_sampler),
                view_entry(3, &heights_view),
                view_entry(4, &noise_view),
                sampler_entry(5, &noise_sampler),
                sampler_entry(6, &lin_sampler),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shore"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let build = |label: &str, fs: &str, blend| {
            pipeline(gpu, &module, &layout, label, fs, HDR, 1, blend)
        };
        let scene_pipe = build("shore scene", "fs_scene", None);
        let bloom_first = build("shore bloom first", "fs_bloom_first", None);
        let bloom_down = build("shore bloom down", "fs_bloom_down", None);
        let bloom_up = build("shore bloom up", "fs_bloom_up", Some(additive));
        let focus_wave = 5 + (seed % 128) as i32 * 3;
        ShoreScene {
            module,
            layout,
            input_layout,
            frame_group,
            uniform,
            dummy,
            scene_pipe,
            bloom_first,
            bloom_down,
            bloom_up,
            output: None,
            targets: None,
            _textures: vec![blocks, heights, noise, dummy_tex],
            time: 0.0,
            start: rest_time(focus_wave, V_FOCUS) - FIRST_REST_S,
            focus_wave,
            look: ShoreLook::default(),
            diagram: None,
            logo: [0.0; 4],
            x_land,
            table: ([[0.0; 4]; WAVE_SLOTS * 4], i32::MIN),
        }
    }

    /// Advance to `time_s` seconds since the first frame.
    pub fn update(&mut self, time_s: f64) {
        self.time = time_s.max(0.0);
    }

    /// The wave that comes to rest [`FIRST_REST_S`] after the first frame in the middle of the
    /// picture (its index in the waves' sequence, for [`Diagram::wave`]).
    #[must_use]
    pub fn focus_wave(&self) -> i32 {
        self.focus_wave
    }

    /// The waves' clock at the first frame.
    #[must_use]
    pub fn start_clock(&self) -> f64 {
        self.start
    }

    /// The grading knobs in use.
    #[must_use]
    pub fn look(&self) -> ShoreLook {
        self.look
    }

    /// Replace the grading knobs (the render harness).
    pub fn set_look(&mut self, look: ShoreLook) {
        self.look = look;
    }

    /// Draw a band diagram instead of the scene (tests and the report), or the scene again.
    pub fn set_diagram(&mut self, diagram: Option<Diagram>) {
        self.diagram = diagram;
    }

    fn ensure_targets(&mut self, gpu: &Gpu, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        self.targets = None;
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let view = |t: wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let scene = view(texture_2d(gpu, "shore hdr", size, HDR, (1, 1), usage));
        let mut bloom = Vec::with_capacity(BLOOM_LEVELS);
        let mut s = size;
        for _ in 0..BLOOM_LEVELS {
            s = ((s.0 / 2).max(1), (s.1 / 2).max(1));
            bloom.push(view(texture_2d(gpu, "shore bloom", s, HDR, (1, 1), usage)));
        }
        let group = |a: &wgpu::TextureView, b: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shore group1"),
                layout: &self.input_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(b),
                    },
                ],
            })
        };
        let g_first = group(&scene, &self.dummy);
        let g_down = (1..BLOOM_LEVELS)
            .map(|i| group(&bloom[i - 1], &self.dummy))
            .collect();
        let g_up = (0..BLOOM_LEVELS - 1)
            .map(|i| group(&bloom[i + 1], &self.dummy))
            .collect();
        let g_composite = group(&scene, &bloom[0]);
        let g_none = group(&self.dummy, &self.dummy);
        self.targets = Some(Targets {
            size,
            scene,
            bloom,
            g_first,
            g_down,
            g_up,
            g_composite,
            g_none,
        });
    }

    fn ensure_output(&mut self, gpu: &Gpu, key: (wgpu::TextureFormat, u32)) {
        if self.output.as_ref().is_some_and(|o| o.key == key) {
            return;
        }
        let composite = pipeline(
            gpu,
            &self.module,
            &self.layout,
            "shore composite",
            "fs_composite",
            key.0,
            key.1,
            None,
        );
        self.output = Some(Output { key, composite });
    }

    fn write_uniform(
        &mut self,
        gpu: &Gpu,
        size: (u32, u32),
        out: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        let (eye, fwd, fov_y) = shot(self.time as f32);
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let tan_y = (fov_y * 0.5).tan();
        let fwd = fwd.normalize();
        let right = fwd.cross(Vec3::Y).normalize() * (tan_y * aspect);
        let up = right.normalize().cross(fwd) * tan_y;
        let look = self.look;
        let (mode, wave, clamp, map) = match self.diagram {
            Some(d) => (
                if d.test_channels { 2.0 } else { 1.0 },
                d.wave as f32,
                if d.clamp { 1.0 } else { 0.0 },
                [d.u_top, d.u_span, d.v_left, d.v_span],
            ),
            None => (0.0, 0.0, 1.0, [0.0; 4]),
        };
        let grain_frame = (self.time * f64::from(look.grain_hz)).floor() % 4096.0;
        let clock = (self.start + self.time) as f32;
        let base = table_base(clock, eye.z);
        if self.table.1 != base {
            for (k, slot) in self.table.0.chunks_exact_mut(4).enumerate() {
                slot.copy_from_slice(&wave_params(base + k as i32));
            }
            self.table.1 = base;
        }
        let u = FrameUniform {
            cam_pos: [eye.x, eye.y, eye.z, clock],
            cam_fwd: [fwd.x, fwd.y, fwd.z, mode],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            screen: [
                size.0 as f32,
                size.1 as f32,
                1.0 / size.0.max(1) as f32,
                1.0 / size.1.max(1) as f32,
            ],
            out_size: [
                out.0 as f32,
                out.1 as f32,
                1.0 / out.0.max(1) as f32,
                1.0 / out.1.max(1) as f32,
            ],
            look: [
                look.exposure,
                look.grain,
                look.bloom,
                if format.is_srgb() { 0.0 } else { 1.0 },
            ],
            extra: [look.logo_shade, grain_frame as f32, clamp, wave],
            map,
            logo: self.logo,
            table: [base as f32, self.x_land, 0.0, 0.0],
            waves: self.table.0,
        };
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    /// Record one frame into `view`: `out` pixels of `format` at `samples` samples. Writes every
    /// pixel.
    pub fn record_into(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        out: (u32, u32),
        (format, samples): (wgpu::TextureFormat, u32),
    ) {
        let out = (out.0.max(1), out.1.max(1));
        // Diagrams are read back pixel by pixel: those draw at the frame's own size.
        let size = if self.diagram.is_some() {
            out
        } else {
            internal_size(out.0, out.1)
        };
        self.ensure_targets(gpu, size);
        self.ensure_output(gpu, (format, samples));
        self.write_uniform(gpu, size, out, format);
        let (Some(t), Some(output)) = (self.targets.as_ref(), self.output.as_ref()) else {
            return;
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        let fg = &self.frame_group;
        fullscreen(
            enc,
            "shore scene",
            &t.scene,
            clear,
            &self.scene_pipe,
            [fg, &t.g_none],
        );
        if self.diagram.is_none() {
            let first = &self.bloom_first;
            fullscreen(
                enc,
                "shore bloom",
                &t.bloom[0],
                clear,
                first,
                [fg, &t.g_first],
            );
            for i in 1..BLOOM_LEVELS {
                let g = &t.g_down[i - 1];
                fullscreen(
                    enc,
                    "shore bloom",
                    &t.bloom[i],
                    clear,
                    &self.bloom_down,
                    [fg, g],
                );
            }
            for i in (0..BLOOM_LEVELS - 1).rev() {
                let g = &t.g_up[i];
                let load = wgpu::LoadOp::Load;
                fullscreen(
                    enc,
                    "shore bloom",
                    &t.bloom[i],
                    load,
                    &self.bloom_up,
                    [fg, g],
                );
            }
        }
        let composite = &output.composite;
        fullscreen(
            enc,
            "shore composite",
            view,
            clear,
            composite,
            [fg, &t.g_composite],
        );
    }
}

impl Scene for ShoreScene {
    fn name(&self) -> &'static str {
        "shore"
    }

    fn render(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        frame: &SceneFrame,
    ) {
        self.update(f64::from(frame.time_s));
        self.logo = frame.logo;
        self.record_into(
            gpu,
            enc,
            target,
            (frame.width, frame.height),
            (SCENE_FORMAT, 1),
        );
    }

    /// The ray cast's HDR target and the bloom chain: rebuilt on the next frame.
    fn release_gpu_memory(&mut self) {
        self.targets = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_beach_steps_down_from_dry_terraces_to_the_swash_terrace_and_the_sea() {
        for z in (HMAP_Z0..HMAP_Z0 + HMAP_H as i32).step_by(5) {
            assert_eq!(column(-50, z).0, 3, "inland at z {z}");
            let terrace = column(-6, z).0;
            assert!(terrace == 0 || terrace == -1, "terrace at z {z}: {terrace}");
            assert!(column(60, z).0 <= -4, "deep sea at z {z}");
            // Going out to sea the columns never rise, apart from the terrace's dips.
            let mut last = 3;
            for x in -60..70 {
                let (h, _) = column(x, z);
                let dip = h == -1 && last == 0;
                if !dip {
                    assert!(h <= last, "x {x} z {z}: {h} after {last}");
                    last = h;
                }
            }
        }
    }

    #[test]
    fn the_map_wraps_along_the_shore_without_a_seam() {
        for x in -60..90 {
            let a = column(x, HMAP_Z0 + HMAP_H as i32 - 1);
            let b = column(x, HMAP_Z0 + HMAP_H as i32);
            let c = column(x, HMAP_Z0);
            assert_eq!(b.0, c.0, "x {x}: the map repeats every {HMAP_H} blocks");
            assert!((a.0 - b.0).abs() <= 2, "x {x}: {a:?} next to {b:?}");
        }
        let (texels, x_land) = height_map();
        assert_eq!(texels.len(), (HMAP_W * HMAP_H * 2) as usize);
        // Seaward of x_land every column is under still water; just landward of it, one is not.
        assert!(x_land > -1.0 && x_land < 4.0, "{x_land}");
        for z in HMAP_Z0..HMAP_Z0 + HMAP_H as i32 {
            for x in x_land as i32..HMAP_X0 + HMAP_W as i32 {
                assert!(column(x, z).0 < 0, "({x}, {z}) rises above the water");
            }
        }
        assert!((HMAP_Z0..HMAP_Z0 + HMAP_H as i32).any(|z| column(x_land as i32 - 1, z).0 >= 0));
    }

    /// Every wave a pixel can ask for is in the shader's table, whatever the time: over two hours
    /// of the camera's drift (the shore is a title-screen scene now), for
    /// the earliest and the latest start a run's seed can give, over the whole stretch of shore in
    /// view (and the earlier fixed window, at the start).
    #[test]
    fn the_wave_table_covers_every_wave_on_screen() {
        let starts = [
            rest_time(5, V_FOCUS) - FIRST_REST_S,
            rest_time(5 + 127 * 3, V_FOCUS) - FIRST_REST_S,
        ];
        let mut checked = 0_u64;
        for start in starts {
            for step in 0..4000 {
                let time = step as f32 * 1.87;
                let clock = (start + f64::from(time)) as f32;
                let eye_z = shot(time).0.z;
                let base = table_base(clock, eye_z);
                let near = (eye_z - VIEW_BEHIND).floor() as i32;
                let far = (eye_z + VIEW_AHEAD).ceil() as i32;
                let lo = if step == 0 {
                    near.min(HMAP_Z0 - 40)
                } else {
                    near
                };
                for v in (lo..=far).step_by(3) {
                    let newest = ((clock - v as f32 * OBL) / PERIOD).floor() as i32 + 1;
                    for k in 0..NWAVES {
                        let i = newest - k;
                        assert!(
                            i >= base && i < base + WAVE_SLOTS as i32,
                            "time {time}, clock {clock}, v {v}: wave {i} outside \
                             {base}..+{WAVE_SLOTS}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 1_000_000, "{checked}");
        // Negative control: the earlier fixed window (the map's first 256 blocks plus 64)
        // misses waves once the camera has drifted for half an hour.
        let time = 1800.0_f32;
        let clock = (starts[0] + f64::from(time)) as f32;
        let fixed = table_base(clock, -32.0);
        let v = shot(time).0.z + VIEW_AHEAD * 0.5;
        let newest = ((clock - v * OBL) / PERIOD).floor() as i32 + 1;
        assert!(
            newest - (NWAVES - 1) < fixed,
            "the fixed window would have covered wave {} at v {v}",
            newest - (NWAVES - 1)
        );
        // Consecutive waves arrive about a period apart, and push two different trailing colours.
        for i in 0..200 {
            let gap = wave_params(i + 1)[0][0] - wave_params(i)[0][0];
            assert!((gap - PERIOD).abs() <= 1.4, "wave {i}: {gap}");
            let [p1, p2, ..] = wave_params(i)[2];
            assert!(p1 != p2 && (1.0..=4.0).contains(&p1) && (1.0..=4.0).contains(&p2));
        }
    }

    #[test]
    fn the_noise_tiles_and_its_mips_shrink_to_one_texel() {
        let px = noise_pixels();
        assert_eq!(px.len(), (NOISE * NOISE * 4) as usize);
        // The channels the shader reads (foam, ripples, grain) use their range.
        for k in [0, 1, 3] {
            let (lo, hi) = px
                .chunks(4)
                .fold((255_u8, 0_u8), |(lo, hi), c| (lo.min(c[k]), hi.max(c[k])));
            assert!(
                lo < 60 && hi > 190,
                "channel {k} uses its range: {lo}..{hi}"
            );
        }
        // The foam density is rank-normalised: a threshold covers that share of the texture.
        let above = px.chunks(4).filter(|c| c[0] >= 128).count() as f32;
        assert!(
            (above / (NOISE * NOISE) as f32 - 0.5).abs() < 0.01,
            "{above}"
        );
        let chain = mip_chain(px, NOISE, NOISE.ilog2() + 1, false);
        assert_eq!(chain.len(), 9);
        assert_eq!(chain.last().map(Vec::len), Some(4));
    }

    /// The foam texture at a few coverages, for a human to look at (`SHORE_OUT` or the lane's
    /// scratch folder).
    #[test]
    #[ignore = "writes a review PNG"]
    fn foam_texture_png() {
        let px = noise_pixels();
        let n = NOISE as usize;
        let out = std::env::var_os("SHORE_OUT").map_or_else(
            || {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .ancestors()
                    .nth(3)
                    .expect("root")
                    .join("taskwave-2026-11-12/scratch/LB/png")
            },
            std::path::PathBuf::from,
        );
        std::fs::create_dir_all(&out).expect("dir");
        // Four tiles side by side: the raw density, then 85%, 50% and 20% coverage.
        let mut img = vec![0_u8; n * 4 * n * 4];
        for (tile, cover) in [None, Some(0.85), Some(0.5), Some(0.2)]
            .into_iter()
            .enumerate()
        {
            for y in 0..n {
                for x in 0..n {
                    let d = f32::from(px[(y * n + x) * 4]) / 255.0;
                    let g = match cover {
                        None => d,
                        Some(c) => {
                            let t = 1.0 - c;
                            ((d - t) / 0.06 + 0.5).clamp(0.0, 1.0)
                        }
                    };
                    let i = (y * n * 4 + tile * n + x) * 4;
                    let v = (g * 255.0) as u8;
                    img[i..i + 4].copy_from_slice(&[v, v, v, 255]);
                }
            }
        }
        image::save_buffer(
            out.join("foam-texture.png"),
            &img,
            (n * 4) as u32,
            n as u32,
            image::ExtendedColorType::Rgba8,
        )
        .expect("write");
    }

    #[test]
    fn the_ray_cast_runs_at_three_quarters_and_never_above_its_cap() {
        assert_eq!(internal_size(1920, 1080), (1440, 810));
        assert_eq!(internal_size(1280, 720), (960, 540));
        for (w, h) in [(3840, 2160), (2560, 1440), (3440, 1440)] {
            let (iw, ih) = internal_size(w, h);
            assert!(
                f64::from(iw) * f64::from(ih) <= MAX_SCENE_PIXELS * 1.01,
                "{w}x{h}"
            );
            let aspect = |a: u32, b: u32| a as f32 / b as f32;
            assert!((aspect(iw, ih) - aspect(w, h)).abs() < 0.01, "{w}x{h}");
        }
        assert_eq!(internal_size(1, 1), (1, 1));
    }

    #[test]
    fn missing_jar_textures_fall_back_to_stand_ins() {
        let t = ShoreTextures::from_jar_textures(&JarTextures::empty());
        assert_eq!(t.vanilla_count(), 0);
        assert_eq!(t.layers.len(), JAR_TEXTURES.len());
        assert!(t.layers.iter().all(|l| l.len() == 16 * 16 * 4));
    }

    /// The Rust mirror of the waves' timing must use the shader's numbers.
    #[test]
    fn the_wave_timing_mirror_matches_the_shader() {
        for (name, value) in [("PERIOD", PERIOD), ("C", C), ("U0", U0), ("OBL", OBL)] {
            let line = format!("const {name}: f32 = {value:?};");
            assert!(SHADER.contains(&line), "shore.wgsl lacks `{line}`");
        }
        assert!(SHADER.contains(&format!("const WAVE_SLOTS: i32 = {WAVE_SLOTS};")));
        assert!(SHADER.contains(&format!("const NWAVES: i32 = {NWAVES};")));
        assert!(SHADER.contains("l.taum = l.tau0 + C / l.a;"));
        assert!(SHADER.contains(
            "let lobe = 0.62 * s1 + 0.26 * s2 + 0.12 * sin(TAU * v / (0.17 * w.lam) + 3.0 * w.ph3);"
        ));
        assert!(SHADER.contains("return w.smax * (1.0 + 0.10 * lobe);"));
        // Every run's first push comes the same time in, whatever the seed.
        for seed in [0_u32, 1, 77, 127, 128, 4_000_000_000] {
            let focus = 5 + (seed % 128) as i32 * 3;
            let start = rest_time(focus, V_FOCUS) - FIRST_REST_S;
            assert!((rest_time(focus, V_FOCUS) - start - FIRST_REST_S).abs() < 1e-9);
            assert!(start > 0.0 && start < 3000.0, "seed {seed}: {start}");
        }
    }
}
