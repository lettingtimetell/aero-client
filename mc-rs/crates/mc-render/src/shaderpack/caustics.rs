//! The Underwater pack's caustic texture.
//!
//! Real caustics are sunlight focused by the moving water surface: every ray refracts through the
//! local surface slope and lands somewhere on the floor, and where many rays land together the
//! floor is bright. This module bakes exactly that, once, on the CPU:
//!
//! * the surface is a sum of [`WAVES`] travelling sine waves with **integer** wave numbers per
//!   tile and **integer** temporal frequencies per loop, so the result tiles seamlessly in space
//!   and loops seamlessly in time;
//! * a regular grid of photons (stratified, so the estimate is smooth) is pushed down through the
//!   slope field (`q = p + S ∇h`, the small-slope refraction map) and splatted bilinearly into the
//!   tile; the photon density, normalised to a mean of 1, is the caustic intensity;
//! * the red channel is the sharp caustic (`intensity / 4`), the green channel a blurred copy
//!   (`intensity / 2`) for the light shafts, whose cross-section is the same light out of focus.
//!
//! The result is a [`SIZE`]² × [`FRAMES`] `Rg8Unorm` 3D texture (2 MiB) sampled with `Repeat` in
//! all three axes, so linear filtering interpolates smoothly between time slices and wraps the
//! loop. The bake takes tens of milliseconds in an optimised build, so [`Caustics`] runs it on a
//! worker thread the first time a pack needs it and keeps the bytes for the rest of the process;
//! until it is ready, [`placeholder_texel`] (a neutral caustic) is bound instead.
//!
//! [`WAVES`] also feeds the WGSL ([`wave_wgsl`]): the sky pass shades the water surface's
//! underside with the same waves that made the caustics, so the two move together.

use std::f32::consts::TAU;
use std::sync::{Arc, OnceLock};

/// Texels per side of one tile.
pub const SIZE: usize = 128;
/// Time slices in one loop (the 3D texture's depth).
pub const FRAMES: usize = 64;
/// Photons per side of the stratified grid, per frame: 4 per texel.
const PHOTONS: usize = 2 * SIZE;
/// Bytes per texel (`Rg8Unorm`).
pub const TEXEL_BYTES: usize = 2;
/// The texture's format.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
/// The red channel stores `intensity / RED_RANGE`.
pub const RED_RANGE: f32 = 4.0;
/// The green channel stores `blurred intensity / GREEN_RANGE`.
pub const GREEN_RANGE: f32 = 2.0;
/// How strongly the surface focuses light: the RMS curvature times the refraction distance. About
/// 0.8 puts the floor just past the first focus in places, which is where real sunlit caustics
/// draw their thin bright network.
const FOCUS: f32 = 0.8;
/// The shafts' blur radius (Gaussian sigma), in texels.
const SHAFT_BLUR_SIGMA: f32 = 3.0;

/// One travelling wave of the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wave {
    /// Cycles per tile along x.
    pub kx: i32,
    /// Cycles per tile along y (world z).
    pub ky: i32,
    /// Cycles per time loop.
    pub m: i32,
    /// Relative amplitude.
    pub amp: f32,
    /// Phase offset, radians.
    pub phase: f32,
}

const fn wave(kx: i32, ky: i32, m: i32, amp: f32, phase: f32) -> Wave {
    Wave {
        kx,
        ky,
        m,
        amp,
        phase,
    }
}

/// The surface: a swell plus cross-chop from many directions, none of them perpendicular to the
/// main swell (two equal perpendicular waves would print a grid). Wave numbers 5 to 8 per tile
/// make cells of about a block and a half on the 12-block tile, so a tile holds enough cells that
/// its repeat does not show. Amplitudes fall off with wave number: the long waves shape the
/// network, the short ones break it up. Temporal frequencies stay at 1 to 3 cycles per loop, so a
/// caustic line moves at most 3/64 of its wavelength between two slices of the texture.
pub const WAVES: [Wave; 12] = [
    wave(5, 2, 2, 1.00, 0.0),
    wave(2, 5, 2, 0.85, 1.7),
    wave(-3, 4, 2, 0.75, 4.1),
    wave(4, -3, 1, 0.60, 2.3),
    wave(6, 1, 3, 0.42, 5.2),
    wave(-1, 6, 2, 0.40, 0.9),
    wave(3, -6, 3, 0.30, 3.6),
    wave(-5, 4, 3, 0.28, 1.2),
    wave(7, 3, 3, 0.20, 5.9),
    wave(1, -7, 3, 0.18, 2.8),
    wave(-6, -4, 3, 0.15, 4.6),
    wave(4, 6, 2, 0.18, 0.4),
];

/// The RMS slope of the surface, `|∇h|` in height units per tile: the sky shading scales its
/// ripples by this so a wave-set change keeps the same tilt.
#[must_use]
pub fn slope_rms() -> f32 {
    let sum: f32 = WAVES
        .iter()
        .map(|w| {
            let k2 = (w.kx * w.kx + w.ky * w.ky) as f32 * TAU * TAU;
            w.amp * w.amp * k2 * 0.5
        })
        .sum();
    sum.sqrt()
}

/// The refraction distance `S` in tile units: [`FOCUS`] over the surface's RMS curvature.
#[must_use]
pub fn focus_scale() -> f32 {
    let curvature_sq: f32 = WAVES
        .iter()
        .map(|w| {
            let k2 = (w.kx * w.kx + w.ky * w.ky) as f32 * TAU * TAU;
            w.amp * w.amp * k2 * k2 * 0.5
        })
        .sum();
    FOCUS / curvature_sq.sqrt()
}

/// The texel a pack binds until the bake is ready: a caustic intensity of exactly 1 in both
/// channels, i.e. no change to the picture.
#[must_use]
pub fn placeholder_texel() -> [u8; TEXEL_BYTES] {
    [
        (255.0 / RED_RANGE).round() as u8,
        (255.0 / GREEN_RANGE).round() as u8,
    ]
}

/// A complex number as `(re, im)`.
#[derive(Clone, Copy)]
struct C(f32, f32);

impl C {
    fn cis(angle: f32) -> C {
        C(angle.cos(), angle.sin())
    }
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
}

/// Splat `weight` at continuous tile position `(x, y)` (texel units) with bilinear weights,
/// wrapping at the tile edges.
fn splat(grid: &mut [f32], x: f32, y: f32) {
    let n = SIZE as f32;
    let x = x.rem_euclid(n) - 0.5;
    let y = y.rem_euclid(n) - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let wrap = |v: f32| (v as i32).rem_euclid(SIZE as i32) as usize;
    let (xa, xb) = (wrap(x0), wrap(x0 + 1.0));
    let (ya, yb) = (wrap(y0), wrap(y0 + 1.0));
    grid[ya * SIZE + xa] += (1.0 - fx) * (1.0 - fy);
    grid[ya * SIZE + xb] += fx * (1.0 - fy);
    grid[yb * SIZE + xa] += (1.0 - fx) * fy;
    grid[yb * SIZE + xb] += fx * fy;
}

/// A separable wrapped blur with the given 1D kernel (odd length, centred).
fn blur_wrapped(src: &[f32], kernel: &[f32]) -> Vec<f32> {
    let half = (kernel.len() / 2) as i32;
    let n = SIZE as i32;
    let mut tmp = vec![0.0f32; SIZE * SIZE];
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let sx = (x + i as i32 - half).rem_euclid(n);
                sum += k * src[(y * n + sx) as usize];
            }
            tmp[(y * n + x) as usize] = sum;
        }
    }
    let mut out = vec![0.0f32; SIZE * SIZE];
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let sy = (y + i as i32 - half).rem_euclid(n);
                sum += k * tmp[(sy * n + x) as usize];
            }
            out[(y * n + x) as usize] = sum;
        }
    }
    out
}

/// A normalised Gaussian kernel of `sigma` texels, 3 sigma each side.
fn gaussian(sigma: f32) -> Vec<f32> {
    let half = (sigma * 3.0).ceil() as i32;
    let mut k: Vec<f32> = (-half..=half)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// The photon density of time slice `frame` (mean 1), before any blur.
fn density(frame: usize, s: f32) -> Vec<f32> {
    density_at(frame as f32 / FRAMES as f32, s)
}

/// The photon density at loop time `t` (`0..1`), mean 1, before any blur.
fn density_at(t: f32, s: f32) -> Vec<f32> {
    let mut grid = vec![0.0f32; SIZE * SIZE];
    let n = PHOTONS as f32;
    // Per photon column, every wave's phasor along x; per photon row, along y times the frame's
    // temporal phasor. cos(theta) is then the real part of their product.
    let columns: Vec<[C; WAVES.len()]> = (0..PHOTONS)
        .map(|j| {
            let x = (j as f32 + 0.5) / n;
            WAVES.map(|w| C::cis(TAU * w.kx as f32 * x))
        })
        .collect();
    let to_texel = SIZE as f32;
    let mut rows = [C(0.0, 0.0); WAVES.len()];
    for l in 0..PHOTONS {
        let y = (l as f32 + 0.5) / n;
        for (row, w) in rows.iter_mut().zip(WAVES.iter()) {
            *row = C::cis(TAU * w.ky as f32 * y).mul(C::cis(w.phase - TAU * w.m as f32 * t));
        }
        for (j, column) in columns.iter().enumerate() {
            let x = (j as f32 + 0.5) / n;
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            for ((c, r), w) in column.iter().zip(rows.iter()).zip(WAVES.iter()) {
                let cos_theta = c.0 * r.0 - c.1 * r.1;
                let a = w.amp * TAU * cos_theta;
                gx += a * w.kx as f32;
                gy += a * w.ky as f32;
            }
            splat(&mut grid, (x + s * gx) * to_texel, (y + s * gy) * to_texel);
        }
    }
    let mean = (PHOTONS * PHOTONS) as f32 / (SIZE * SIZE) as f32;
    for v in &mut grid {
        *v /= mean;
    }
    grid
}

/// Bake the whole texture: `SIZE * SIZE * FRAMES` texels of [`TEXEL_BYTES`], slice after slice,
/// rows of `SIZE` texels. Deterministic: the same bytes on every machine and run.
#[must_use]
pub fn bake() -> Vec<u8> {
    let s = focus_scale();
    let tent = [0.25, 0.5, 0.25];
    let shaft_kernel = gaussian(SHAFT_BLUR_SIGMA);
    let mut out = Vec::with_capacity(SIZE * SIZE * FRAMES * TEXEL_BYTES);
    for frame in 0..FRAMES {
        let raw = density(frame, s);
        let sharp = blur_wrapped(&raw, &tent);
        let soft = blur_wrapped(&raw, &shaft_kernel);
        for (a, b) in sharp.iter().zip(soft.iter()) {
            out.push(encode(*a / RED_RANGE));
            out.push(encode(*b / GREEN_RANGE));
        }
    }
    out
}

fn encode(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The process-wide bake, started on a worker thread by the first [`Caustics::poll`] and kept for
/// the rest of the run (a resource reload or a pack toggle never bakes twice).
#[derive(Debug, Default)]
pub struct Caustics {
    started: bool,
}

static BAKED: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Caustics {
    /// The baked bytes if they are ready; otherwise make sure the bake is running and return
    /// `None`. Cheap after the first call: one atomic load.
    pub fn poll(&mut self) -> Option<Arc<Vec<u8>>> {
        if let Some(bytes) = BAKED.get() {
            return Some(Arc::clone(bytes));
        }
        if !self.started {
            self.started = true;
            if !STARTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
                let spawned = std::thread::Builder::new()
                    .name("shaderpack-caustics".to_owned())
                    .spawn(|| {
                        let _ = BAKED.set(Arc::new(bake()));
                    });
                if spawned.is_err() {
                    // No thread (a sandbox): bake here, once.
                    let _ = BAKED.set(Arc::new(bake()));
                }
            }
        }
        BAKED.get().map(Arc::clone)
    }

    /// Block until the bake is done (tests and screenshot tools).
    pub fn wait(&mut self) -> Arc<Vec<u8>> {
        loop {
            if let Some(bytes) = self.poll() {
                return bytes;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

/// The waves as WGSL constants for the surface shading: `UW_WAVE_COUNT`, and `UW_WAVES`, one
/// `vec4(kx, ky, m, amp)` each, plus `UW_WAVE_PHASE`.
#[must_use]
pub fn wave_wgsl() -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "const UW_WAVE_COUNT: u32 = {}u;", WAVES.len());
    let _ = write!(s, "const UW_WAVES = array<vec4<f32>, {}>(", WAVES.len());
    for (i, w) in WAVES.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(
            s,
            "vec4<f32>({:.1}, {:.1}, {:.1}, {:.6})",
            w.kx as f32, w.ky as f32, w.m as f32, w.amp
        );
    }
    s.push_str(");\n");
    let _ = write!(s, "const UW_WAVE_PHASE = array<f32, {}>(", WAVES.len());
    for (i, w) in WAVES.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{:.6}", w.phase);
    }
    s.push_str(");\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bake_has_the_right_size_and_a_mean_of_one() {
        let bytes = bake();
        assert_eq!(bytes.len(), SIZE * SIZE * FRAMES * TEXEL_BYTES);
        // Mean of the sharp channel, decoded: clipping only removes a little of the top.
        let slice = &bytes[..SIZE * SIZE * TEXEL_BYTES];
        let mean_red: f32 = slice
            .chunks_exact(2)
            .map(|t| f32::from(t[0]) / 255.0 * RED_RANGE)
            .sum::<f32>()
            / (SIZE * SIZE) as f32;
        let mean_green: f32 = slice
            .chunks_exact(2)
            .map(|t| f32::from(t[1]) / 255.0 * GREEN_RANGE)
            .sum::<f32>()
            / (SIZE * SIZE) as f32;
        assert!((0.85..=1.02).contains(&mean_red), "red mean {mean_red}");
        assert!(
            (0.9..=1.02).contains(&mean_green),
            "green mean {mean_green}"
        );
        // A real caustic: bright lines (well above the mean) and dim cells (below it).
        let bright = slice.chunks_exact(2).filter(|t| t[0] >= 128).count();
        let dim = slice.chunks_exact(2).filter(|t| t[0] <= 40).count();
        assert!(bright > SIZE * SIZE / 100, "bright texels {bright}");
        assert!(dim > SIZE * SIZE / 10, "dim texels {dim}");
    }

    #[test]
    fn the_loop_closes_and_time_moves_the_pattern() {
        // Integer temporal frequencies: the end of the loop is its start again, so the 3D
        // texture's Repeat wrap in time shows no jump.
        let s = focus_scale();
        let start = density_at(0.0, s);
        let end = density_at(1.0, s);
        let worst = start
            .iter()
            .zip(end.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.05, "loop seam {worst}");
        // ...while a slice later in the loop really is a different pattern.
        let later = density(FRAMES / 4, s);
        let moved = start
            .iter()
            .zip(later.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (SIZE * SIZE) as f32;
        assert!(moved > 0.2, "mean change {moved}");
        for w in WAVES {
            assert!(w.kx != 0 || w.ky != 0);
            assert!(w.m > 0);
        }
    }

    /// Writes four time slices of both channels side by side, tiled 2x2 to show the wrap:
    /// `MC_TW_SC_PNG_DIR=<dir> cargo test -p mc-render --lib caustics -- --ignored`.
    #[test]
    #[ignore = "writes a PNG; run by hand"]
    fn write_the_bake_as_a_png() {
        let Some(dir) = std::env::var_os("MC_TW_SC_PNG_DIR") else {
            return;
        };
        let bytes = bake();
        let tile = 2 * SIZE;
        let slices = [0, FRAMES / 4, FRAMES / 2, 3 * FRAMES / 4];
        let width = (tile * slices.len()) as u32;
        let height = (tile * 2) as u32;
        let mut img = image::GrayImage::new(width, height);
        for (s, frame) in slices.iter().enumerate() {
            for channel in 0..2 {
                for y in 0..tile {
                    for x in 0..tile {
                        let i = ((frame * SIZE + y % SIZE) * SIZE + x % SIZE) * TEXEL_BYTES;
                        let v = bytes[i + channel];
                        let (px, py) = ((s * tile + x) as u32, (channel * tile + y) as u32);
                        img.put_pixel(px, py, image::Luma([v]));
                    }
                }
            }
        }
        let path = std::path::Path::new(&dir).join("caustics_bake.png");
        img.save(&path).expect("write png");
        eprintln!("wrote {}", path.display());
    }

    #[test]
    fn the_placeholder_is_neutral() {
        let [r, g] = placeholder_texel();
        assert!((f32::from(r) / 255.0 * RED_RANGE - 1.0).abs() < 0.01);
        assert!((f32::from(g) / 255.0 * GREEN_RANGE - 1.0).abs() < 0.01);
    }

    #[test]
    fn the_wgsl_constants_parse() {
        let source = format!(
            "{}\nfn f() -> f32 {{ return UW_WAVES[0].x + UW_WAVE_PHASE[1]; }}",
            wave_wgsl()
        );
        wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }
}
