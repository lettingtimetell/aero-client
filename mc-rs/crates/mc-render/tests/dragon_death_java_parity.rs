//! The ender dragon's death rays render like vanilla 1.8.9's
//! `LayerEnderDragonDeath`.
//!
//! The reference is vanilla: `javaref/DragonDeathRef.java` (`generate-dragon-death.ps1`) runs the
//! unmodified recompiled 1.8.9 `RenderManager.renderEntityStatic` over a real `RenderDragon`
//! drawing a dying dragon in an LWJGL Pbuffer, with the vanilla jar's resources and the world
//! pass's state at `renderEntities`. Its `LayerEnderDragonDeath` is swapped for a subclass that
//! records the layer's entry matrix (the dragon's living-model matrix, mc-rs's `base`) and then
//! runs the vanilla body unchanged. It records every shard, the live GL state, and the real pixels
//! (the dragon's body suppressed). What it established (`the_java_reference_*`):
//!
//! * every shard is its own 5-vertex `GL_TRIANGLE_FAN` (3 triangles), white at the centre with
//!   alpha `(int)(255 * (1 - f1))`, magenta and transparent at the rim, smooth-shaded;
//! * they are drawn additively (`SRC_ALPHA, ONE`) with no alpha test and no depth writes, with
//!   **back faces culled** (`enableCull()`; `GL_BACK`, counter-clockwise front), texturing off but
//!   the lightmap stage on at the dragon's own light (`(0, 0)` in the reference world).
//!
//! The comparisons rebuild each case from mc-rs's own code:
//!
//! * `dragon_death_vertices_match_java` (CPU): `emit_dragon_death_shards` from Java's layer matrix
//!   must be each fan's 3 triangles as a triangle list, every vertex within 1e-2 blocks of Java's
//!   (the shards are up to 35 blocks long, 45 blocks away), colours exact;
//! * `dragon_death_pixels_match_java` (GPU): those vertices under `LayerBlend::DragonDeathShards`
//!   through the real [`EntityPhase`] with the dragon's combined light, against Java's coverage,
//!   mean colour and colour grid. Three negative controls must fail every visible case: the
//!   5-vertex fans submitted as a triangle list, the triangles' winding reversed (the culled
//!   side), and the full-bright lightmap texel instead of the dragon's own.
//!
//! Skips (loudly) without a wgpu adapter. Set `MC_DRAGON_DEATH_DUMP=<dir>` to write each mc-rs
//! frame as a PNG, and `MC_DRAGON_DEATH_JAVA_DIR=<dir>` (the generator's output folder) to also
//! print an exact per-pixel comparison with Java's frames.

// The tables are generated: their values are data, and one may land near a constant.
#![allow(clippy::approx_constant)]

use std::path::PathBuf;

use glam::Mat4;
use mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT;
use mc_render::entity::{EntityPhase, EntityVertex, LayerBlend};
use mc_render::sky::lightmap::Lightmap;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 960;
const H: u32 = 540;
/// Vanilla's default field of view, which `DragonDeathRef` renders with.
const FOV: f32 = 70.0;
/// `DragonDeathRef.CLEAR`.
const CLEAR: [u8; 3] = [0x20, 0x18, 0x30];
/// `DragonDeathRef.LIGHTMAP` and `LIGHTMAP_DARK`: every texel but (block 0, sky 0), and that one.
const LIGHTMAP: u8 = 252;
const LIGHTMAP_DARK: u8 = 150;
/// The dragon's packed combined light in the reference world (its light is 0 everywhere).
const DRAGON_COMBINED_LIGHT: u32 = 0;

// GL enums as `glGet` returned them.
const GL_SRC_ALPHA: i32 = 770;
const GL_ONE: i32 = 1;
const GL_BACK: i32 = 1029;
const GL_CCW: i32 = 2305;
const GL_LEQUAL: i32 = 515;
const GL_SMOOTH: i32 = 7425;

/// The live GL state at the first shard's draw call.
#[derive(Debug, Clone, Copy)]
struct State {
    blend: bool,
    /// `GL_BLEND_SRC`, `GL_BLEND_DST`.
    blend_func: [i32; 2],
    alpha_test: bool,
    cull: bool,
    cull_face: i32,
    front_face: i32,
    depth_test: bool,
    depth_func: i32,
    depth_mask: bool,
    lighting: bool,
    fog: bool,
    /// `GL_TEXTURE_2D` on unit 0 and on unit 1 (the lightmap).
    texture0: bool,
    texture1: bool,
    shade_model: i32,
    /// Unit 1's current texture coordinate: the lightmap coordinate of every shard vertex.
    lightmap: [f32; 2],
}

/// Java's coverage mask: every pixel that is not the clear colour.
#[derive(Debug, Clone, Copy)]
struct Mask {
    count: u64,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    cx: f64,
    cy: f64,
}

/// One shard, `GL_TRIANGLE_FAN` order: camera-relative position, RGBA bytes.
type Shard = [([f32; 3], [u8; 4]); 5];

/// One vanilla render.
struct Ref {
    name: &'static str,
    /// Camera yaw and pitch, degrees.
    camera: [f32; 2],
    /// `EntityDragon.deathTicks`.
    death_ticks: u32,
    partial: f32,
    /// The layer's entry modelview, column-major: the dragon's living-model matrix.
    base: [f32; 16],
    state: State,
    shards: &'static [Shard],
    mask: Mask,
    mean: [f64; 3],
    /// Row-major 4x4 cells over the mask bounds: `[pixels, mean r, g, b]`.
    grid: [[f64; 4]; 16],
}

fn emit(r: &Ref) -> Vec<EntityVertex> {
    let mut out = Vec::new();
    mc_render::entity::emit_dragon_death_shards(
        Mat4::from_cols_array(&r.base),
        r.death_ticks,
        r.partial,
        &mut out,
    );
    out
}

/// GL's fan triangles: `(0, i + 1, i + 2)`.
const FAN_TRIANGLES: [[usize; 3]; 3] = [[0, 1, 2], [0, 2, 3], [0, 3, 4]];

#[test]
fn the_java_reference_draws_culled_additive_lightmapped_fans() {
    for r in REFS {
        let s = r.state;
        assert!(
            s.blend && s.blend_func == [GL_SRC_ALPHA, GL_ONE],
            "{}",
            r.name
        );
        assert!(!s.alpha_test, "{}", r.name);
        assert!(
            s.cull && s.cull_face == GL_BACK && s.front_face == GL_CCW,
            "{}",
            r.name
        );
        assert!(
            s.depth_test && s.depth_func == GL_LEQUAL && !s.depth_mask,
            "{}",
            r.name
        );
        assert!(!s.lighting && s.fog, "{}", r.name);
        assert!(
            !s.texture0 && s.texture1,
            "{}: untextured, lightmapped",
            r.name
        );
        assert_eq!(s.shade_model, GL_SMOOTH, "{}", r.name);
        assert_eq!(s.lightmap, [0.0, 0.0], "{}: the dragon's own light", r.name);
        // java: `(f + f * f) / 2 * 60` shards with `f = (deathTicks + partial) / 200`.
        let f = (r.death_ticks as f32 + r.partial) / 200.0;
        let count = ((f + f * f) / 2.0 * 60.0).ceil() as usize;
        assert_eq!(r.shards.len(), count, "{}", r.name);
        let f1 = if f > 0.8 { (f - 0.8) / 0.2 } else { 0.0 };
        let centre = (255.0 * (1.0 - f1)) as i32 as u8;
        for shard in r.shards {
            assert_eq!(shard[0].1, [255, 255, 255, centre], "{}", r.name);
            assert!(
                shard[1..].iter().all(|v| v.1 == [255, 0, 255, 0]),
                "{}",
                r.name
            );
            assert_eq!(
                shard[4].0, shard[1].0,
                "{}: the fan closes on its first rim vertex",
                r.name
            );
        }
    }
}

#[test]
fn dragon_death_vertices_match_java() {
    let mut failures = Vec::new();
    for r in REFS {
        let out = emit(r);
        if out.len() != r.shards.len() * 9 {
            failures.push(format!(
                "{}: {} vertices; Java's {} 5-vertex fans are {} triangle-list vertices",
                r.name,
                out.len(),
                r.shards.len(),
                r.shards.len() * 9
            ));
            continue;
        }
        let mut worst = 0.0_f64;
        for (index, (shard, ours)) in r.shards.iter().zip(out.chunks_exact(9)).enumerate() {
            for (t, triangle) in FAN_TRIANGLES.iter().enumerate() {
                for (k, &corner) in triangle.iter().enumerate() {
                    let (java_pos, java_rgba) = shard[corner];
                    let vertex = ours[t * 3 + k];
                    let distance = vertex
                        .pos
                        .iter()
                        .zip(java_pos)
                        .map(|(a, b)| f64::from(*a - b).abs())
                        .fold(0.0, f64::max);
                    worst = worst.max(distance);
                    if distance > 1e-2 {
                        failures.push(format!(
                            "{} shard {index} triangle {t} vertex {k}: {:?} vs Java {java_pos:?}",
                            r.name, vertex.pos
                        ));
                    }
                    let bytes = vertex.color.map(|c| (c * 255.0).round());
                    if bytes != java_rgba.map(f32::from) {
                        failures.push(format!(
                            "{} shard {index} triangle {t} vertex {k}: colour {:?} vs Java bytes \
                             {java_rgba:?}",
                            r.name, vertex.color
                        ));
                    }
                }
            }
        }
        eprintln!("{:8} worst position error {worst:.2e} blocks", r.name);
    }
    assert!(
        failures.is_empty(),
        "mc-rs's death rays are not vanilla's LayerEnderDragonDeath fans:\n{}",
        failures.join("\n")
    );
}

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

/// `EntityRenderer.orientCamera`'s rotation at the eye, `gluPerspective(70, 16/9, 0.05, far)`.
fn camera(r: &Ref) -> Camera {
    let mut camera = Camera::default();
    camera.yaw = r.camera[0];
    camera.pitch = r.camera[1];
    camera.fov = FOV;
    camera.set_viewport(W, H);
    camera.set_view_distance_chunks(8);
    camera.update();
    camera
}

/// How one case's rays are drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Draws {
    /// The client's state: the fans' triangles under `LayerBlend::DragonDeathShards`.
    Client,
    /// Negative control: the 5-vertex fans submitted as a triangle list.
    FansAsList,
    /// Negative control: every triangle's winding reversed, so culling keeps the other side.
    Reversed,
    /// Negative control: the full-bright lightmap texel instead of the dragon's own light.
    FullBright,
}

/// Render one case through the real `EntityPhase` and return its RGBA pixels.
fn render_case(gpu: &mut Gpu, r: &Ref, mode: Draws) -> Vec<u8> {
    let camera = camera(r);
    let mut phase = EntityPhase::new(gpu);
    let mut lightmap = [LIGHTMAP, LIGHTMAP, LIGHTMAP, 255].repeat(256);
    lightmap[..3].copy_from_slice(&[LIGHTMAP_DARK; 3]);
    phase.set_lightmap(gpu, &Lightmap { rgba: lightmap });
    phase.begin();
    let verts = emit(r);
    let (verts, light) = match mode {
        Draws::Client => (verts, DRAGON_COMBINED_LIGHT),
        Draws::FansAsList => (
            verts
                .chunks_exact(9)
                .flat_map(|fan| [fan[0], fan[1], fan[2], fan[5], fan[8]])
                .collect(),
            DRAGON_COMBINED_LIGHT,
        ),
        Draws::Reversed => (
            verts
                .chunks_exact(3)
                .flat_map(|t| [t[0], t[2], t[1]])
                .collect(),
            DRAGON_COMBINED_LIGHT,
        ),
        Draws::FullBright => (verts, FULL_BRIGHT_COMBINED_LIGHT),
    };
    phase.push_blended_lit(
        EntityPhase::FALLBACK_TEXTURE,
        LayerBlend::DragonDeathShards,
        light,
        &verts,
    );
    phase.finish(gpu);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: f64::from(CLEAR[0]) / 255.0,
        g: f64::from(CLEAR[1]) / 255.0,
        b: f64::from(CLEAR[2]) / 255.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &camera, r.partial);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("readback")
}

fn is_clear(pixel: &[u8]) -> bool {
    pixel[..3] == CLEAR
}

/// Coverage statistics and the mean colour, as `DragonDeathRef` computes them.
fn stats_of(pixels: &[u8]) -> (Mask, [f64; 3]) {
    let mut mask = Mask {
        count: 0,
        min_x: W,
        max_x: 0,
        min_y: H,
        max_y: 0,
        cx: 0.0,
        cy: 0.0,
    };
    let mut sum = [0.0_f64; 3];
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        if is_clear(pixel) {
            continue;
        }
        let x = (index as u32) % W;
        let y = (index as u32) / W;
        mask.count += 1;
        mask.cx += f64::from(x) + 0.5;
        mask.cy += f64::from(y) + 0.5;
        mask.min_x = mask.min_x.min(x);
        mask.max_x = mask.max_x.max(x);
        mask.min_y = mask.min_y.min(y);
        mask.max_y = mask.max_y.max(y);
        for (s, c) in sum.iter_mut().zip(pixel) {
            *s += f64::from(*c);
        }
    }
    let n = mask.count.max(1) as f64;
    mask.cx /= n;
    mask.cy /= n;
    (mask, sum.map(|s| s / n))
}

/// The 4x4 colour grid over `bounds` (Java's mask bounds, so both sides bin the same pixels).
fn grid_of(pixels: &[u8], bounds: &Mask) -> [[f64; 4]; 16] {
    let mut grid = [[0.0_f64; 4]; 16];
    let bw = bounds.max_x - bounds.min_x + 1;
    let bh = bounds.max_y - bounds.min_y + 1;
    for y in bounds.min_y..=bounds.max_y {
        for x in bounds.min_x..=bounds.max_x {
            let i = ((y * W + x) * 4) as usize;
            let pixel = &pixels[i..i + 4];
            if is_clear(pixel) {
                continue;
            }
            let cell = (((y - bounds.min_y) * 4 / bh) * 4 + (x - bounds.min_x) * 4 / bw) as usize;
            grid[cell][0] += 1.0;
            for (sum, &channel) in grid[cell][1..].iter_mut().zip(&pixel[..3]) {
                *sum += f64::from(channel);
            }
        }
    }
    for cell in &mut grid {
        let n = cell[0].max(1.0);
        for value in &mut cell[1..] {
            *value /= n;
        }
    }
    grid
}

/// Everything that differs from Java beyond rasteriser noise, or nothing.
fn differences(r: &Ref, pixels: &[u8]) -> Vec<String> {
    let (ours, mean) = stats_of(pixels);
    let java = r.mask;
    if java.count == 0 {
        // Vanilla's rays have faded out completely: nothing may be drawn.
        return if ours.count == 0 {
            Vec::new()
        } else {
            vec![format!("{} pixels where Java draws none", ours.count)]
        };
    }
    let mut out = Vec::new();
    let count_slack = (java.count as f64 * 0.01).max(20.0);
    if (ours.count as f64 - java.count as f64).abs() > count_slack {
        out.push(format!("pixels {} vs Java {}", ours.count, java.count));
    }
    let bounds = [
        (ours.min_x, java.min_x),
        (ours.max_x, java.max_x),
        (ours.min_y, java.min_y),
        (ours.max_y, java.max_y),
    ];
    if bounds.iter().any(|(a, b)| a.abs_diff(*b) > 1) {
        out.push(format!(
            "bounds x {}..{} y {}..{} vs Java x {}..{} y {}..{}",
            ours.min_x,
            ours.max_x,
            ours.min_y,
            ours.max_y,
            java.min_x,
            java.max_x,
            java.min_y,
            java.max_y
        ));
    }
    if (ours.cx - java.cx).abs() > 0.5 || (ours.cy - java.cy).abs() > 0.5 {
        out.push(format!(
            "centroid ({:.3}, {:.3}) vs Java ({:.3}, {:.3})",
            ours.cx, ours.cy, java.cx, java.cy
        ));
    }
    if mean.iter().zip(&r.mean).any(|(a, b)| (a - b).abs() > 1.0) {
        out.push(format!("mean colour {mean:.2?} vs Java {:.2?}", r.mean));
    }
    let grid = grid_of(pixels, &java);
    for (cell, (ours, java)) in grid.iter().zip(&r.grid).enumerate() {
        if java[0] < 50.0 {
            continue;
        }
        let colour = (1..4).any(|c| (ours[c] - java[c]).abs() > 2.0);
        let count = (ours[0] - java[0]).abs() > java[0] * 0.03 + 5.0;
        if colour || count {
            out.push(format!(
                "cell {cell}: {:.0} px mean ({:.1}, {:.1}, {:.1}) vs Java {:.0} px ({:.1}, {:.1}, \
                 {:.1})",
                ours[0], ours[1], ours[2], ours[3], java[0], java[1], java[2], java[3]
            ));
        }
    }
    out
}

/// An exact pixel-for-pixel comparison against the generator's PNG, when it is available.
fn exact_comparison(r: &Ref, pixels: &[u8]) -> Option<String> {
    let dir = PathBuf::from(std::env::var_os("MC_DRAGON_DEATH_JAVA_DIR")?);
    let java = image::open(dir.join(format!("gl-{}.png", r.name)))
        .ok()?
        .into_rgba8();
    let java = java.as_raw();
    let (mut both, mut only_java, mut only_ours, mut same, mut worst) = (0, 0, 0, 0, 0u8);
    for (a, b) in pixels.chunks_exact(4).zip(java.chunks_exact(4)) {
        match (is_clear(a), is_clear(b)) {
            (true, true) => {}
            (false, true) => only_ours += 1,
            (true, false) => only_java += 1,
            (false, false) => {
                both += 1;
                let diff = (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0);
                worst = worst.max(diff);
                if diff == 0 {
                    same += 1;
                }
            }
        }
    }
    Some(format!(
        "exact: {both} px covered by both ({same} identical, worst channel diff {worst}), \
         {only_java} Java-only, {only_ours} mc-rs-only"
    ))
}

/// Render every case, print and check the comparison, and return the failing cases.
fn check_cases(gpu: &mut Gpu, mode: Draws, dump: Option<&std::path::Path>) -> Vec<String> {
    let mut failures = Vec::new();
    for r in REFS {
        let pixels = render_case(gpu, r, mode);
        if let Some(dir) = dump {
            std::fs::create_dir_all(dir).expect("dump dir");
            let suffix = match mode {
                Draws::Client => "",
                Draws::FansAsList => "-fansaslist",
                Draws::Reversed => "-reversed",
                Draws::FullBright => "-fullbright",
            };
            image::RgbaImage::from_raw(W, H, pixels.clone())
                .expect("rgba frame")
                .save(dir.join(format!("rs-{}{suffix}.png", r.name)))
                .expect("write png");
        }
        let (ours, mean) = stats_of(&pixels);
        eprintln!(
            "{mode:?} {:8} n {:6} vs {:6}  c ({:7.2}, {:7.2}) vs ({:7.2}, {:7.2})  mean \
             ({:6.2}, {:6.2}, {:6.2}) vs ({:6.2}, {:6.2}, {:6.2})",
            r.name,
            ours.count,
            r.mask.count,
            ours.cx,
            ours.cy,
            r.mask.cx,
            r.mask.cy,
            mean[0],
            mean[1],
            mean[2],
            r.mean[0],
            r.mean[1],
            r.mean[2],
        );
        if let Some(exact) = exact_comparison(r, &pixels) {
            eprintln!("    {exact}");
        }
        let diffs = differences(r, &pixels);
        if !diffs.is_empty() {
            failures.push(format!("{}: {}", r.name, diffs.join("; ")));
        }
    }
    failures
}

#[test]
fn dragon_death_pixels_match_java() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let dump = std::env::var_os("MC_DRAGON_DEATH_DUMP").map(PathBuf::from);
    let failures = check_cases(&mut gpu, Draws::Client, dump.as_deref());
    assert!(
        failures.is_empty(),
        "mc-rs's death rays do not render like vanilla's:\n{}",
        failures.join("\n")
    );
    // Negative controls: every case vanilla draws something in must notice each of them.
    let visible = REFS.iter().filter(|r| r.mask.count > 0).count();
    for control in [Draws::FansAsList, Draws::Reversed, Draws::FullBright] {
        let failing = check_cases(&mut gpu, control, dump.as_deref());
        assert_eq!(
            failing.len(),
            visible,
            "{control:?} must fail every visible case, failed only: {failing:#?}"
        );
    }
}

// -------------------------------------------------------------------------------------------
// Generated by DragonDeathRef on this machine:
// GL_VENDOR=NVIDIA Corporation GL_RENDERER=NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2 GL_VERSION=4.6.0 NVIDIA 610.88
// -------------------------------------------------------------------------------------------

#[rustfmt::skip]
const REFS: &[Ref] = &[
    Ref {
        name: "early",
        camera: [0.0, 0.0],
        death_ticks: 20, partial: 0.5,
        // the layer's entry modelview (column-major): the dragon's living-model matrix
        base: [-1.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -10.4921875, 46.0, 1.0],
        state: State { blend: true, blend_func: [770, 1], alpha_test: false, cull: true, cull_face: 1029, front_face: 2305, depth_test: true, depth_func: 515, depth_mask: false, lighting: false, fog: true, texture0: false, texture1: true, shade_model: 7425, lightmap: [0.0, 0.0] },
        // 4 shards: camera-relative position, rgba; GL_TRIANGLE_FAN order
        shards: &[
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-2.0113235, -27.048016, 38.23766], [255, 0, 255, 0]), ([2.041215, -27.356861, 39.294262], [255, 0, 255, 0]), ([0.9660518, -25.980175, 35.475307], [255, 0, 255, 0]), ([-2.0113235, -27.048016, 38.23766], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([2.0054646, -10.37903, 28.229218], [255, 0, 255, 0]), ([0.63879526, -14.477538, 28.891586], [255, 0, 255, 0]), ([4.866802, -13.478933, 29.373096], [255, 0, 255, 0]), ([2.0054646, -10.37903, 28.229218], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([16.91306, -1.8181281, 43.85555], [255, 0, 255, 0]), ([16.024845, -0.2110157, 42.57597], [255, 0, 255, 0]), ([16.85654, -2.0618014, 41.631428], [255, 0, 255, 0]), ([16.91306, -1.8181281, 43.85555], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-8.758514, -12.5816345, 51.312943], [255, 0, 255, 0]), ([-8.192944, -11.083132, 52.37138], [255, 0, 255, 0]), ([-7.3747387, -12.80121, 52.62549], [255, 0, 255, 0]), ([-8.758514, -12.5816345, 51.312943], [255, 0, 255, 0])],
        ],
        mask: Mask { count: 7530, min_x: 325, max_x: 545, min_y: 272, max_y: 539, cx: 455.7594, cy: 416.2491 },
        mean: [80.1821, 46.8627, 96.1482],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [953.0, 55.292, 28.954, 71.292], [588.0, 105.629, 61.753, 121.629], [79.0, 143.759, 107.392, 159.759], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1177.0, 117.438, 79.850, 133.221], [419.0, 83.826, 49.346, 99.826],
            [0.0, 0.000, 0.000, 0.000], [100.0, 43.120, 25.150, 59.120], [1990.0, 86.082, 48.898, 102.082], [15.0, 86.333, 43.667, 102.333],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1846.0, 58.589, 29.904, 74.589], [363.0, 52.887, 27.956, 68.887],
        ],
    },
    Ref {
        name: "mid",
        camera: [0.0, 0.0],
        death_ticks: 100, partial: 0.0,
        // the layer's entry modelview (column-major): the dragon's living-model matrix
        base: [-1.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -10.4921875, 46.0, 1.0],
        state: State { blend: true, blend_func: [770, 1], alpha_test: false, cull: true, cull_face: 1029, front_face: 2305, depth_test: true, depth_func: 515, depth_mask: false, lighting: false, fog: true, texture0: false, texture1: true, shade_model: 7425, lightmap: [0.0, 0.0] },
        // 23 shards: camera-relative position, rgba; GL_TRIANGLE_FAN order
        shards: &[
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-12.109911, -21.867588, 37.24303], [255, 0, 255, 0]), ([-8.777781, -24.423283, 37.25685], [255, 0, 255, 0]), ([-9.49274, -21.923172, 33.95929], [255, 0, 255, 0]), ([-12.109911, -21.867588, 37.24303], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([5.8741846, -1.9911475, 31.242481], [255, 0, 255, 0]), ([2.8975754, -4.439545, 29.181065], [255, 0, 255, 0]), ([6.8577065, -6.077194, 30.041637], [255, 0, 255, 0]), ([5.8741846, -1.9911475, 31.242481], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([12.824777, -22.577843, 47.04212], [255, 0, 255, 0]), ([14.269296, -20.897556, 47.35686], [255, 0, 255, 0]), ([13.928595, -21.709791, 45.29932], [255, 0, 255, 0]), ([12.824777, -22.577843, 47.04212], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-10.144711, -4.2051325, 41.02219], [255, 0, 255, 0]), ([-10.579382, -4.3371468, 42.88748], [255, 0, 255, 0]), ([-10.988508, -5.79025, 41.701336], [255, 0, 255, 0]), ([-10.144711, -4.2051325, 41.02219], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-0.3089616, -9.053355, 37.883476], [255, 0, 255, 0]), ([4.0506973, -9.122774, 39.400513], [255, 0, 255, 0]), ([1.5272334, -5.277881, 39.803883], [255, 0, 255, 0]), ([-0.3089616, -9.053355, 37.883476], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-5.0242615, -6.407278, 44.51451], [255, 0, 255, 0]), ([-2.3836043, -4.563487, 41.752575], [255, 0, 255, 0]), ([-1.7104557, -4.162709, 45.922462], [255, 0, 255, 0]), ([-5.0242615, -6.407278, 44.51451], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([10.487932, -4.8269825, 52.617485], [255, 0, 255, 0]), ([7.4719925, -2.8750687, 54.31528], [255, 0, 255, 0]), ([9.345003, -1.3016901, 51.18393], [255, 0, 255, 0]), ([10.487932, -4.8269825, 52.617485], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-9.82096, 1.1415691, 52.130356], [255, 0, 255, 0]), ([-9.145112, 0.095911026, 54.0033], [255, 0, 255, 0]), ([-10.924025, -0.6759424, 52.863987], [255, 0, 255, 0]), ([-9.82096, 1.1415691, 52.130356], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-8.885967, 8.3308735, 50.256454], [255, 0, 255, 0]), ([-8.521858, 7.074072, 53.418182], [255, 0, 255, 0]), ([-11.315135, 6.3232546, 51.589783], [255, 0, 255, 0]), ([-8.885967, 8.3308735, 50.256454], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-0.48779207, -4.2652464, 41.510414], [255, 0, 255, 0]), ([-3.0237386, -6.003628, 40.47242], [255, 0, 255, 0]), ([-0.11466205, -6.391733, 39.08766], [255, 0, 255, 0]), ([-0.48779207, -4.2652464, 41.510414], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-12.994532, 8.663347, 35.551903], [255, 0, 255, 0]), ([-14.341532, 8.205515, 36.861134], [255, 0, 255, 0]), ([-14.368009, 7.3743534, 35.115643], [255, 0, 255, 0]), ([-12.994532, 8.663347, 35.551903], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([3.2570202, -13.829966, 33.333546], [255, 0, 255, 0]), ([3.2738707, -11.641441, 32.69247], [255, 0, 255, 0]), ([1.3390709, -12.844505, 32.59094], [255, 0, 255, 0]), ([3.2570202, -13.829966, 33.333546], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-5.2026687, -18.446274, 34.385353], [255, 0, 255, 0]), ([-7.868459, -18.858112, 36.925285], [255, 0, 255, 0]), ([-4.775321, -20.886433, 37.140514], [255, 0, 255, 0]), ([-5.2026687, -18.446274, 34.385353], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([0.5567791, -9.208996, 57.67865], [255, 0, 255, 0]), ([-2.9645836, -6.997564, 57.13331], [255, 0, 255, 0]), ([0.7631432, -5.0854883, 56.94197], [255, 0, 255, 0]), ([0.5567791, -9.208996, 57.67865], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([24.104929, -8.309017, 38.740177], [255, 0, 255, 0]), ([23.579367, -7.2764473, 36.9845], [255, 0, 255, 0]), ([23.681175, -9.377544, 36.978447], [255, 0, 255, 0]), ([24.104929, -8.309017, 38.740177], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([17.87524, -10.93893, 47.217674], [255, 0, 255, 0]), ([18.039337, -7.311947, 45.341496], [255, 0, 255, 0]), ([18.156036, -10.75614, 43.144535], [255, 0, 255, 0]), ([17.87524, -10.93893, 47.217674], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-10.001999, -1.4914608, 59.906956], [255, 0, 255, 0]), ([-8.443864, 1.9383354, 58.667213], [255, 0, 255, 0]), ([-6.3923106, -0.39590645, 61.131233], [255, 0, 255, 0]), ([-10.001999, -1.4914608, 59.906956], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([7.517785, -9.935567, 47.09561], [255, 0, 255, 0]), ([7.5332794, -6.4090242, 44.2008], [255, 0, 255, 0]), ([7.9365635, -10.666843, 42.61148], [255, 0, 255, 0]), ([7.517785, -9.935567, 47.09561], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-8.742408, 1.8352518, 61.41489], [255, 0, 255, 0]), ([-5.9831357, 5.617896, 59.617504], [255, 0, 255, 0]), ([-3.9432976, 2.3131514, 62.791054], [255, 0, 255, 0]), ([-8.742408, 1.8352518, 61.41489], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([5.4088435, -8.180355, 25.938], [255, 0, 255, 0]), ([6.261269, -11.489738, 26.27943], [255, 0, 255, 0]), ([8.5124655, -9.039882, 27.131512], [255, 0, 255, 0]), ([5.4088435, -8.180355, 25.938], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([3.133202, -29.61871, 39.587337], [255, 0, 255, 0]), ([6.804914, -27.975769, 37.187088], [255, 0, 255, 0]), ([2.587056, -28.188026, 35.160404], [255, 0, 255, 0]), ([3.133202, -29.61871, 39.587337], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-2.319479, -10.544929, 51.03281], [255, 0, 255, 0]), ([1.2202033, -8.184025, 51.26281], [255, 0, 255, 0]), ([1.5034984, -12.401526, 50.724888], [255, 0, 255, 0]), ([-2.319479, -10.544929, 51.03281], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 255]), ([-16.696733, -8.6977, 31.756742], [255, 0, 255, 0]), ([-18.749102, -9.006013, 35.194073], [255, 0, 255, 0]), ([-17.511168, -12.317938, 33.290947], [255, 0, 255, 0]), ([-16.696733, -8.6977, 31.756742], [255, 0, 255, 0])],
        ],
        mask: Mask { count: 31928, min_x: 234, max_x: 684, min_y: 177, max_y: 539, cx: 491.9768, cy: 372.7839 },
        mean: [90.7698, 53.0982, 106.2233],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1993.0, 65.783, 33.608, 81.783], [1458.0, 54.215, 28.580, 70.215],
            [1636.0, 59.210, 30.554, 75.210], [3845.0, 114.187, 65.850, 129.255], [4808.0, 136.915, 90.427, 150.619], [84.0, 36.405, 24.095, 52.405],
            [697.0, 56.099, 29.333, 72.099], [3571.0, 81.826, 44.846, 97.770], [4290.0, 130.637, 82.823, 146.025], [3516.0, 61.235, 32.211, 77.235],
            [0.0, 0.000, 0.000, 0.000], [2583.0, 71.260, 36.144, 87.260], [2313.0, 64.983, 32.656, 80.983], [1134.0, 43.454, 25.256, 59.454],
        ],
    },
    Ref {
        name: "fading",
        camera: [0.0, 0.0],
        death_ticks: 170, partial: 0.25,
        // the layer's entry modelview (column-major): the dragon's living-model matrix
        base: [-1.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -10.4921875, 46.0, 1.0],
        state: State { blend: true, blend_func: [770, 1], alpha_test: false, cull: true, cull_face: 1029, front_face: 2305, depth_test: true, depth_func: 515, depth_mask: false, lighting: false, fog: true, texture0: false, texture1: true, shade_model: 7425, lightmap: [0.0, 0.0] },
        // 48 shards: camera-relative position, rgba; GL_TRIANGLE_FAN order
        shards: &[
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-19.923712, -13.862099, 38.241318], [255, 0, 255, 0]), ([-17.95156, -18.415283, 37.12012], [255, 0, 255, 0]), ([-17.785488, -14.658071, 33.694504], [255, 0, 255, 0]), ([-19.923712, -13.862099, 38.241318], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([15.298869, -2.7485433, 36.001198], [255, 0, 255, 0]), ([12.049949, -2.3040662, 31.890392], [255, 0, 255, 0]), ([14.530417, -6.850465, 32.801716], [255, 0, 255, 0]), ([15.298869, -2.7485433, 36.001198], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-6.565548, -28.423195, 37.174683], [255, 0, 255, 0]), ([-4.0442386, -29.563618, 38.628197], [255, 0, 255, 0]), ([-3.8289235, -28.572311, 35.67157], [255, 0, 255, 0]), ([-6.565548, -28.423195, 37.174683], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([5.5600286, 3.0193806, 39.465374], [255, 0, 255, 0]), ([4.0977216, 4.124895, 41.591713], [255, 0, 255, 0]), ([2.8781621, 3.7599173, 39.089386], [255, 0, 255, 0]), ([5.5600286, 3.0193806, 39.465374], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([2.802703, -6.876314, 36.19545], [255, 0, 255, 0]), ([4.819012, -11.903285, 37.17558], [255, 0, 255, 0]), ([7.1646395, -7.496845, 39.4949], [255, 0, 255, 0]), ([2.802703, -6.876314, 36.19545], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-3.4199643, -3.0283809, 48.300926], [255, 0, 255, 0]), ([1.279216, -1.5948434, 46.823196], [255, 0, 255, 0]), ([0.6921202, -4.6299887, 50.91775], [255, 0, 255, 0]), ([-3.4199643, -3.0283809, 48.300926], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([11.981478, -21.427797, 45.34298], [255, 0, 255, 0]), ([11.3317375, -20.556992, 50.081116], [255, 0, 255, 0]), ([14.379247, -17.788551, 47.496582], [255, 0, 255, 0]), ([11.981478, -21.427797, 45.34298], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([3.465394, 2.2379045, 58.799046], [255, 0, 255, 0]), ([3.0540147, -0.30961514, 60.58219], [255, 0, 255, 0]), ([0.58778405, 1.448492, 59.766167], [255, 0, 255, 0]), ([3.465394, 2.2379045, 58.799046], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([10.5068, 11.421452, 45.830143], [255, 0, 255, 0]), ([10.73526, 10.494923, 50.032597], [255, 0, 255, 0]), ([7.216641, 12.402447, 48.434868], [255, 0, 255, 0]), ([10.5068, 11.421452, 45.830143], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([7.151245, -6.8185625, 40.478127], [255, 0, 255, 0]), ([4.1903515, -4.7844224, 38.434925], [255, 0, 255, 0]), ([5.1274104, -8.670875, 37.387318], [255, 0, 255, 0]), ([7.151245, -6.8185625, 40.478127], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([20.325268, -3.7273245, 28.072598], [255, 0, 255, 0]), ([19.865616, -1.0193424, 28.716007], [255, 0, 255, 0]), ([18.50349, -2.2050242, 26.548626], [255, 0, 255, 0]), ([20.325268, -3.7273245, 28.072598], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-12.290563, -15.657217, 39.176453], [255, 0, 255, 0]), ([-10.323678, -17.322718, 37.333916], [255, 0, 255, 0]), ([-11.370324, -14.488916, 36.37893], [255, 0, 255, 0]), ([-12.290563, -15.657217, 39.176453], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-14.446895, -13.752586, 36.690468], [255, 0, 255, 0]), ([-14.880132, -9.194392, 36.332798], [255, 0, 255, 0]), ([-16.196072, -11.33136, 40.17927], [255, 0, 255, 0]), ([-14.446895, -13.752586, 36.690468], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([1.3301971, -3.7027617, 59.185184], [255, 0, 255, 0]), ([3.5267904, 0.13380527, 56.67985], [255, 0, 255, 0]), ([6.2642922, -3.940813, 57.993496], [255, 0, 255, 0]), ([1.3301971, -3.7027617, 59.185184], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-2.77844, -30.36995, 61.35], [255, 0, 255, 0]), ([-1.5265884, -32.07788, 59.237427], [255, 0, 255, 0]), ([-4.488658, -31.66198, 59.26354], [255, 0, 255, 0]), ([-2.77844, -30.36995, 61.35], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-11.850509, -10.863276, 61.073227], [255, 0, 255, 0]), ([-8.734996, -14.569872, 62.213364], [255, 0, 255, 0]), ([-12.722485, -15.456742, 59.374405], [255, 0, 255, 0]), ([-11.850509, -10.863276, 61.073227], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([8.028812, 11.445669, 38.76471], [255, 0, 255, 0]), ([11.746498, 10.169477, 41.611977], [255, 0, 255, 0]), ([7.7783685, 12.17725, 43.55636], [255, 0, 255, 0]), ([8.028812, 11.445669, 38.76471], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-7.491408, -5.0826845, 50.30441], [255, 0, 255, 0]), ([-4.208902, -7.7682457, 53.72747], [255, 0, 255, 0]), ([-8.086875, -10.455653, 50.99893], [255, 0, 255, 0]), ([-7.491408, -5.0826845, 50.30441], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([9.165353, 8.882347, 29.482964], [255, 0, 255, 0]), ([11.839124, 10.545246, 34.47593], [255, 0, 255, 0]), ([6.3461466, 12.371767, 33.31983], [255, 0, 255, 0]), ([9.165353, 8.882347, 29.482964], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([14.435546, -16.806522, 58.160397], [255, 0, 255, 0]), ([11.22232, -19.399992, 59.436848], [255, 0, 255, 0]), ([11.591257, -15.432056, 61.1103], [255, 0, 255, 0]), ([14.435546, -16.806522, 58.160397], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-14.1334095, -27.650373, 39.483166], [255, 0, 255, 0]), ([-13.495863, -28.64617, 44.928127], [255, 0, 255, 0]), ([-9.781117, -30.629925, 41.27951], [255, 0, 255, 0]), ([-14.1334095, -27.650373, 39.483166], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-2.6071606, -6.682296, 34.682915], [255, 0, 255, 0]), ([-4.706897, -3.1316195, 37.763885], [255, 0, 255, 0]), ([-7.069295, -7.652203, 37.061646], [255, 0, 255, 0]), ([-2.6071606, -6.682296, 34.682915], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([16.865881, -20.309029, 55.940475], [255, 0, 255, 0]), ([19.078798, -20.568518, 51.57305], [255, 0, 255, 0]), ([15.895527, -24.046371, 52.918808], [255, 0, 255, 0]), ([16.865881, -20.309029, 55.940475], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-0.1551953, 3.921423, 47.428642], [255, 0, 255, 0]), ([-2.0773654, 2.8338223, 49.95472], [255, 0, 255, 0]), ([-3.4638944, 3.5754528, 46.990482], [255, 0, 255, 0]), ([-0.1551953, 3.921423, 47.428642], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-10.993062, -9.392764, 21.096588], [255, 0, 255, 0]), ([-6.4841433, -9.385348, 19.436447], [255, 0, 255, 0]), ([-8.659913, -5.235054, 20.498795], [255, 0, 255, 0]), ([-10.993062, -9.392764, 21.096588], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-5.9125967, 3.4444437, 52.440014], [255, 0, 255, 0]), ([-1.4280732, 2.3796854, 55.42665], [255, 0, 255, 0]), ([-6.0831213, -0.37314606, 56.385002], [255, 0, 255, 0]), ([-5.9125967, 3.4444437, 52.440014], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-10.099758, -22.205593, 34.13611], [255, 0, 255, 0]), ([-13.340057, -21.875427, 38.55663], [255, 0, 255, 0]), ([-9.039548, -25.289587, 38.553783], [255, 0, 255, 0]), ([-10.099758, -22.205593, 34.13611], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([18.266954, -17.71457, 61.08793], [255, 0, 255, 0]), ([18.827166, -12.904055, 62.08821], [255, 0, 255, 0]), ([21.365088, -14.970572, 58.38079], [255, 0, 255, 0]), ([18.266954, -17.71457, 61.08793], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([13.003084, -3.9548464, 49.745056], [255, 0, 255, 0]), ([13.066557, -1.6728315, 44.933125], [255, 0, 255, 0]), ([14.905138, -6.633021, 45.552547], [255, 0, 255, 0]), ([13.003084, -3.9548464, 49.745056], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-8.69285, -6.7903576, 51.3051], [255, 0, 255, 0]), ([-9.840225, -4.36424, 47.619965], [255, 0, 255, 0]), ([-6.8188066, -2.6950488, 50.59796], [255, 0, 255, 0]), ([-8.69285, -6.7903576, 51.3051], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-11.195892, -9.888134, 49.936268], [255, 0, 255, 0]), ([-11.88302, -13.31625, 46.21685], [255, 0, 255, 0]), ([-12.563108, -8.347156, 45.265568], [255, 0, 255, 0]), ([-11.195892, -9.888134, 49.936268], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([2.0561442, -24.459503, 42.6685], [255, 0, 255, 0]), ([-1.6102607, -24.546291, 44.895638], [255, 0, 255, 0]), ([2.1413043, -24.213783, 46.95142], [255, 0, 255, 0]), ([2.0561442, -24.459503, 42.6685], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-3.2218046, -3.5574975, 37.489567], [255, 0, 255, 0]), ([-6.3844166, -4.2046905, 39.610477], [255, 0, 255, 0]), ([-5.3746595, -6.697706, 36.838264], [255, 0, 255, 0]), ([-3.2218046, -3.5574975, 37.489567], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([11.921986, -6.092037, 35.78843], [255, 0, 255, 0]), ([12.065386, -10.591676, 35.377804], [255, 0, 255, 0]), ([13.960585, -8.58728, 38.95935], [255, 0, 255, 0]), ([11.921986, -6.092037, 35.78843], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([19.598755, -25.032621, 41.385246], [255, 0, 255, 0]), ([20.504177, -23.973545, 45.52389], [255, 0, 255, 0]), ([22.024015, -21.507006, 42.256336], [255, 0, 255, 0]), ([19.598755, -25.032621, 41.385246], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-4.6292434, -13.16695, 50.46766], [255, 0, 255, 0]), ([-6.0152354, -9.679426, 50.367596], [255, 0, 255, 0]), ([-2.8388674, -10.383952, 52.24077], [255, 0, 255, 0]), ([-4.6292434, -13.16695, 50.46766], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-1.1664038, 0.6575556, 59.753662], [255, 0, 255, 0]), ([2.1420238, 4.187499, 56.682014], [255, 0, 255, 0]), ([4.5104303, -0.117824554, 59.631172], [255, 0, 255, 0]), ([-1.1664038, 0.6575556, 59.753662], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([16.087997, -13.289915, 43.376495], [255, 0, 255, 0]), ([16.018398, -9.756779, 39.879803], [255, 0, 255, 0]), ([14.865594, -14.446254, 38.698414], [255, 0, 255, 0]), ([16.087997, -13.289915, 43.376495], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([3.8581932, -11.790596, 53.62677], [255, 0, 255, 0]), ([1.095475, -10.891403, 54.47307], [255, 0, 255, 0]), ([3.2957382, -8.851616, 54.07824], [255, 0, 255, 0]), ([3.8581932, -11.790596, 53.62677], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-6.361085, -7.5762854, 51.947754], [255, 0, 255, 0]), ([-7.926668, -5.0819407, 49.001854], [255, 0, 255, 0]), ([-4.6844788, -3.8197565, 51.292484], [255, 0, 255, 0]), ([-6.361085, -7.5762854, 51.947754], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([6.6607294, -8.625909, 53.654022], [255, 0, 255, 0]), ([8.8514385, -11.538456, 51.46865], [255, 0, 255, 0]), ([5.5419865, -12.7204895, 53.85803], [255, 0, 255, 0]), ([6.6607294, -8.625909, 53.654022], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([7.601278, -17.169003, 51.366436], [255, 0, 255, 0]), ([6.8338656, -13.558881, 54.379673], [255, 0, 255, 0]), ([10.3000765, -13.249725, 51.12509], [255, 0, 255, 0]), ([7.601278, -17.169003, 51.366436], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-7.9778495, -19.691116, 42.7244], [255, 0, 255, 0]), ([-8.574226, -19.125301, 45.725433], [255, 0, 255, 0]), ([-6.15722, -20.916487, 44.930347], [255, 0, 255, 0]), ([-7.9778495, -19.691116, 42.7244], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-5.847176, -3.0488172, 40.44284], [255, 0, 255, 0]), ([-8.116544, -4.882258, 45.109093], [255, 0, 255, 0]), ([-8.56038, -7.8366446, 40.487255], [255, 0, 255, 0]), ([-5.847176, -3.0488172, 40.44284], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-6.7821045, -12.248669, 49.234753], [255, 0, 255, 0]), ([-8.280985, -11.4858055, 46.90663], [255, 0, 255, 0]), ([-7.5700297, -9.511268, 48.86746], [255, 0, 255, 0]), ([-6.7821045, -12.248669, 49.234753], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([14.706428, -10.498637, 49.327812], [255, 0, 255, 0]), ([13.474137, -8.126995, 51.890594], [255, 0, 255, 0]), ([14.735506, -6.858696, 48.648365], [255, 0, 255, 0]), ([14.706428, -10.498637, 49.327812], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([19.973085, -8.747514, 38.72026], [255, 0, 255, 0]), ([20.464674, -7.2762194, 42.091873], [255, 0, 255, 0]), ([19.622849, -5.0860815, 39.216225], [255, 0, 255, 0]), ([19.973085, -8.747514, 38.72026], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 189]), ([-3.2086353, -6.0581903, 51.98213], [255, 0, 255, 0]), ([-5.383094, -3.469593, 48.5335], [255, 0, 255, 0]), ([-0.8731885, -2.389759, 49.881783], [255, 0, 255, 0]), ([-3.2086353, -6.0581903, 51.98213], [255, 0, 255, 0])],
        ],
        mask: Mask { count: 61898, min_x: 203, max_x: 682, min_y: 128, max_y: 539, cx: 462.4380, cy: 365.7588 },
        mean: [94.7243, 55.5173, 109.6895],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [4914.0, 63.286, 30.588, 79.286], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [3543.0, 47.796, 26.879, 63.796], [4880.0, 94.851, 48.699, 110.843], [7834.0, 117.610, 70.216, 131.535], [0.0, 0.000, 0.000, 0.000],
            [895.0, 42.859, 25.264, 58.859], [9769.0, 120.319, 66.224, 134.900], [9467.0, 160.869, 113.280, 173.289], [7213.0, 65.536, 32.578, 81.536],
            [1538.0, 41.557, 25.081, 57.557], [1374.0, 58.678, 31.065, 74.678], [5866.0, 72.121, 37.795, 88.121], [4605.0, 48.137, 26.758, 64.137],
        ],
    },
    Ref {
        name: "final",
        camera: [0.0, 0.0],
        death_ticks: 199, partial: 0.9,
        // the layer's entry modelview (column-major): the dragon's living-model matrix
        base: [-1.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -10.4921875, 46.0, 1.0],
        state: State { blend: true, blend_func: [770, 1], alpha_test: false, cull: true, cull_face: 1029, front_face: 2305, depth_test: true, depth_func: 515, depth_mask: false, lighting: false, fog: true, texture0: false, texture1: true, shade_model: 7425, lightmap: [0.0, 0.0] },
        // 60 shards: camera-relative position, rgba; GL_TRIANGLE_FAN order
        shards: &[
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-28.209984, -8.877125, 38.51715], [255, 0, 255, 0]), ([-26.883923, -16.061424, 36.231926], [255, 0, 255, 0]), ([-25.81326, -10.241308, 31.376272], [255, 0, 255, 0]), ([-28.209984, -8.877125, 38.51715], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([24.808573, -5.6646433, 36.88673], [255, 0, 255, 0]), ([21.32044, -3.338398, 30.27835], [255, 0, 255, 0]), ([22.826242, -10.929306, 31.445631], [255, 0, 255, 0]), ([24.808573, -5.6646433, 36.88673], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-17.876356, -25.577124, 28.349174], [255, 0, 255, 0]), ([-15.096874, -29.951582, 30.705818], [255, 0, 255, 0]), ([-13.1014185, -27.54111, 25.94931], [255, 0, 255, 0]), ([-17.876356, -25.577124, 28.349174], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([18.35037, 1.1442003, 38.293976], [255, 0, 255, 0]), ([17.00548, 4.318614, 42.41788], [255, 0, 255, 0]), ([14.953205, 5.2403374, 37.536068], [255, 0, 255, 0]), ([18.35037, 1.1442003, 38.293976], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([8.320556, -5.4203396, 30.720188], [255, 0, 255, 0]), ([7.6917963, -13.457537, 30.315067], [255, 0, 255, 0]), ([13.395356, -10.082623, 34.923557], [255, 0, 255, 0]), ([8.320556, -5.4203396, 30.720188], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-3.520482, 1.3465996, 55.219833], [255, 0, 255, 0]), ([4.1447873, 1.7247467, 54.618793], [255, 0, 255, 0]), ([0.8869636, -3.774621, 58.9091], [255, 0, 255, 0]), ([-3.520482, 1.3465996, 55.219833], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([12.176769, -29.562155, 36.91133], [255, 0, 255, 0]), ([11.037392, -31.389719, 44.0211], [255, 0, 255, 0]), ([16.873928, -27.189468, 42.15504], [255, 0, 255, 0]), ([12.176769, -29.562155, 36.91133], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([9.82987, 0.0077342987, 66.96987], [255, 0, 255, 0]), ([8.038017, -4.975857, 69.0895], [255, 0, 255, 0]), ([4.453108, -0.5485077, 68.79256], [255, 0, 255, 0]), ([9.82987, 0.0077342987, 66.96987], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([19.412956, 14.690989, 44.158215], [255, 0, 255, 0]), ([20.559372, 12.750357, 50.65549], [255, 0, 255, 0]), ([15.322725, 16.96356, 49.198303], [255, 0, 255, 0]), ([19.412956, 14.690989, 44.158215], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([14.242999, -9.063182, 36.831703], [255, 0, 255, 0]), ([10.801515, -4.4497366, 33.4013], [255, 0, 255, 0]), ([10.203087, -10.934501, 31.824242], [255, 0, 255, 0]), ([14.242999, -9.063182, 36.831703], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([24.912727, -19.82558, 23.35829], [255, 0, 255, 0]), ([26.718761, -14.760384, 23.705744], [255, 0, 255, 0]), ([23.072277, -16.06977, 19.960262], [255, 0, 255, 0]), ([24.912727, -19.82558, 23.35829], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-21.5125, -5.073905, 41.372257], [255, 0, 255, 0]), ([-21.34336, -9.855287, 38.208366], [255, 0, 255, 0]), ([-20.016312, -4.8016987, 35.84151], [255, 0, 255, 0]), ([-21.5125, -5.073905, 41.372257], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-23.645687, -7.0056496, 38.9937], [255, 0, 255, 0]), ([-21.440735, -0.515378, 36.923546], [255, 0, 255, 0]), ([-23.024645, -1.733911, 43.79957], [255, 0, 255, 0]), ([-23.645687, -7.0056496, 38.9937], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([7.064919, -2.783935, 65.78514], [255, 0, 255, 0]), ([12.304031, 0.9048176, 61.607353], [255, 0, 255, 0]), ([13.396505, -6.423204, 63.509525], [255, 0, 255, 0]), ([7.064919, -2.783935, 65.78514], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-12.799232, -24.601055, 72.60979], [255, 0, 255, 0]), ([-12.968623, -29.236341, 69.54605], [255, 0, 255, 0]), ([-17.263742, -25.707756, 69.487976], [255, 0, 255, 0]), ([-12.799232, -24.601055, 72.60979], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-11.767362, 7.750139, 63.22311], [255, 0, 255, 0]), ([-11.142171, 1.6428547, 67.604225], [255, 0, 255, 0]), ([-17.172014, 2.4897842, 63.153366], [255, 0, 255, 0]), ([-11.767362, 7.750139, 63.22311], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([19.120424, 3.857211, 24.232243], [255, 0, 255, 0]), ([24.268309, 1.5878534, 29.07207], [255, 0, 255, 0]), ([20.807718, 8.071514, 30.103344], [255, 0, 255, 0]), ([19.120424, 3.857211, 24.232243], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-7.3165765, 7.09264, 46.32405], [255, 0, 255, 0]), ([-4.336991, 5.4515953, 53.58458], [255, 0, 255, 0]), ([-11.515168, 2.583232, 51.45495], [255, 0, 255, 0]), ([-7.3165765, 7.09264, 46.32405], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([13.878602, -13.54128, 14.67466], [255, 0, 255, 0]), ([20.081411, -8.978063, 18.203518], [255, 0, 255, 0]), ([13.506244, -5.079574, 14.553486], [255, 0, 255, 0]), ([13.878602, -13.54128, 14.67466], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([23.733427, -12.566154, 60.45764], [255, 0, 255, 0]), ([19.306845, -11.42437, 65.612305], [255, 0, 255, 0]), ([22.250713, -6.111964, 62.359184], [255, 0, 255, 0]), ([23.733427, -12.566154, 60.45764], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-26.279018, -22.06629, 54.593025], [255, 0, 255, 0]), ([-23.655247, -18.970993, 61.649048], [255, 0, 255, 0]), ([-21.114737, -26.316666, 59.231964], [255, 0, 255, 0]), ([-26.279018, -22.06629, 54.593025], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-6.895358, -16.581764, 29.427464], [255, 0, 255, 0]), ([-7.7997904, -9.014812, 28.217033], [255, 0, 255, 0]), ([-12.916633, -12.873161, 32.515358], [255, 0, 255, 0]), ([-6.895358, -16.581764, 29.427464], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([18.310673, -6.442094, 68.66214], [255, 0, 255, 0]), ([21.260265, -12.821482, 66.12965], [255, 0, 255, 0]), ([15.273359, -12.98811, 70.595276], [255, 0, 255, 0]), ([18.310673, -6.442094, 68.66214], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([8.452432, 1.7047844, 27.84389], [255, 0, 255, 0]), ([5.078347, 5.9976196, 30.13956], [255, 0, 255, 0]), ([2.7609549, 2.1604357, 26.267836], [255, 0, 255, 0]), ([8.452432, 1.7047844, 27.84389], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-22.272482, -27.917263, 28.200472], [255, 0, 255, 0]), ([-16.998861, -33.061382, 28.485357], [255, 0, 255, 0]), ([-17.102402, -28.190866, 22.951466], [255, 0, 255, 0]), ([-22.272482, -27.917263, 28.200472], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-1.576141, 9.016956, 28.648527], [255, 0, 255, 0]), ([5.2349577, 11.01078, 32.469006], [255, 0, 255, 0]), ([-1.8987739, 13.027241, 35.632725], [255, 0, 255, 0]), ([-1.576141, 9.016956, 28.648527], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-16.548252, -28.284725, 52.841366], [255, 0, 255, 0]), ([-21.055628, -21.811579, 54.491287], [255, 0, 255, 0]), ([-15.458871, -24.269146, 59.74293], [255, 0, 255, 0]), ([-16.548252, -28.284725, 52.841366], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([18.811638, -0.5238905, 70.69519], [255, 0, 255, 0]), ([21.852741, 4.352973, 65.85658], [255, 0, 255, 0]), ([24.536127, -2.6493974, 66.31807], [255, 0, 255, 0]), ([18.811638, -0.5238905, 70.69519], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([18.940329, 1.1308222, 50.99668], [255, 0, 255, 0]), ([21.771715, -2.7076235, 44.706997], [255, 0, 255, 0]), ([21.260107, -6.3840523, 51.673725], [255, 0, 255, 0]), ([18.940329, 1.1308222, 50.99668], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-13.411937, 4.2112455, 45.445473], [255, 0, 255, 0]), ([-13.447715, 3.0650682, 38.411835], [255, 0, 255, 0]), ([-8.63641, 7.4714775, 41.27961], [255, 0, 255, 0]), ([-13.411937, 4.2112455, 45.445473], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-15.695295, 3.0163012, 46.57495], [255, 0, 255, 0]), ([-19.31156, -3.6925917, 45.69215], [255, 0, 255, 0]), ([-17.120789, 0.32413864, 39.533016], [255, 0, 255, 0]), ([-15.695295, 3.0163012, 46.57495], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-2.7446551, -28.594936, 55.995136], [255, 0, 255, 0]), ([-8.229624, -25.15495, 58.25773], [255, 0, 255, 0]), ([-2.1146104, -24.181515, 61.207104], [255, 0, 255, 0]), ([-2.7446551, -28.594936, 55.995136], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-0.3800246, -8.751913, 27.08486], [255, 0, 255, 0]), ([-6.040276, -6.076302, 28.551344], [255, 0, 255, 0]), ([-5.4888616, -12.476311, 28.258894], [255, 0, 255, 0]), ([-0.3800246, -8.751913, 27.08486], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([20.2855, -13.24025, 35.197372], [255, 0, 255, 0]), ([19.46901, -19.31559, 38.756523], [255, 0, 255, 0]), ([21.969961, -13.572692, 42.07475], [255, 0, 255, 0]), ([20.2855, -13.24025, 35.197372], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([19.85371, -33.97333, 52.63277], [255, 0, 255, 0]), ([21.610065, -29.733543, 57.83162], [255, 0, 255, 0]), ([24.854961, -29.256592, 51.721474], [255, 0, 255, 0]), ([19.85371, -33.97333, 52.63277], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-8.722719, -11.142628, 57.688553], [255, 0, 255, 0]), ([-10.475405, -5.3669515, 55.80825], [255, 0, 255, 0]), ([-5.1416163, -6.1317263, 59.114834], [255, 0, 255, 0]), ([-8.722719, -11.142628, 57.688553], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-3.3584986, 7.212734, 64.07938], [255, 0, 255, 0]), ([1.9472419, 11.686642, 59.529984], [255, 0, 255, 0]), ([4.8113885, 5.9413395, 64.78877], [255, 0, 255, 0]), ([-3.3584986, 7.212734, 64.07938], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([21.416517, -18.510036, 50.3885], [255, 0, 255, 0]), ([23.60452, -14.326187, 44.510933], [255, 0, 255, 0]), ([20.993992, -21.31155, 43.40184], [255, 0, 255, 0]), ([21.416517, -18.510036, 50.3885], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([5.307505, -13.291815, 60.96753], [255, 0, 255, 0]), ([0.5316159, -10.624491, 62.13668], [255, 0, 255, 0]), ([5.235232, -7.709353, 61.318153], [255, 0, 255, 0]), ([5.307505, -13.291815, 60.96753], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-9.0457735, -5.8908796, 59.038277], [255, 0, 255, 0]), ([-11.478842, -1.1431246, 54.93043], [255, 0, 255, 0]), ([-5.6702905, -0.12814903, 58.181274], [255, 0, 255, 0]), ([-9.0457735, -5.8908796, 59.038277], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([10.14948, -7.724885, 60.343674], [255, 0, 255, 0]), ([13.618203, -12.749353, 57.31112], [255, 0, 255, 0]), ([8.025057, -14.175352, 60.938652], [255, 0, 255, 0]), ([10.14948, -7.724885, 60.343674], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([10.69736, -24.440277, 53.375908], [255, 0, 255, 0]), ([10.565381, -19.398296, 58.69791], [255, 0, 255, 0]), ([15.664428, -19.046852, 53.44042], [255, 0, 255, 0]), ([10.69736, -24.440277, 53.375908], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-13.600083, -22.679104, 36.003284], [255, 0, 255, 0]), ([-15.156451, -23.150768, 41.444748], [255, 0, 255, 0]), ([-10.89723, -26.318398, 39.42465], [255, 0, 255, 0]), ([-13.600083, -22.679104, 36.003284], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-6.7980394, 4.967823, 38.452908], [255, 0, 255, 0]), ([-12.353234, 2.0512943, 43.529644], [255, 0, 255, 0]), ([-11.984704, -0.6849327, 35.945457], [255, 0, 255, 0]), ([-6.7980394, 4.967823, 38.452908], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-13.552383, -14.80217, 51.89511], [255, 0, 255, 0]), ([-15.904247, -12.036832, 47.843876], [255, 0, 255, 0]), ([-14.302984, -9.4330435, 52.343708], [255, 0, 255, 0]), ([-13.552383, -14.80217, 51.89511], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([21.150898, -18.612064, 46.970253], [255, 0, 255, 0]), ([20.700043, -15.699388, 52.50494], [255, 0, 255, 0]), ([22.755386, -12.571104, 47.47396], [255, 0, 255, 0]), ([21.150898, -18.612064, 46.970253], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([21.81305, -13.43326, 26.55496], [255, 0, 255, 0]), ([25.19236, -13.10879, 31.837135], [255, 0, 255, 0]), ([23.675182, -7.8538876, 28.752953], [255, 0, 255, 0]), ([21.81305, -13.43326, 26.55496], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([3.053681, -9.581808, 60.517662], [255, 0, 255, 0]), ([0.09184408, -2.889062, 59.445282], [255, 0, 255, 0]), ([7.2807436, -4.009251, 58.110107], [255, 0, 255, 0]), ([3.053681, -9.581808, 60.517662], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-21.241873, 5.9836283, 40.848698], [255, 0, 255, 0]), ([-25.010435, -1.1859636, 41.523365], [255, 0, 255, 0]), ([-22.167091, 1.24751, 34.30832], [255, 0, 255, 0]), ([-21.241873, 5.9836283, 40.848698], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([8.087932, 3.9627123, 17.686384], [255, 0, 255, 0]), ([11.923236, -3.3815145, 16.443651], [255, 0, 255, 0]), ([15.787045, 2.6887627, 20.735218], [255, 0, 255, 0]), ([8.087932, 3.9627123, 17.686384], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([29.284971, 3.6310377, 50.916172], [255, 0, 255, 0]), ([26.499308, 8.372337, 51.503838], [255, 0, 255, 0]), ([28.35602, 6.8555317, 46.52001], [255, 0, 255, 0]), ([29.284971, 3.6310377, 50.916172], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-30.454199, -4.697001, 58.04357], [255, 0, 255, 0]), ([-31.07361, -0.24067307, 53.824146], [255, 0, 255, 0]), ([-28.383808, 0.9895544, 59.237038], [255, 0, 255, 0]), ([-30.454199, -4.697001, 58.04357], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([21.30688, -27.787495, 33.205006], [255, 0, 255, 0]), ([25.772484, -22.160217, 35.025974], [255, 0, 255, 0]), ([22.38349, -22.218248, 28.435253], [255, 0, 255, 0]), ([21.30688, -27.787495, 33.205006], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([22.647102, -25.978674, 61.125065], [255, 0, 255, 0]), ([26.960194, -19.761929, 59.67304], [255, 0, 255, 0]), ([26.644232, -25.512281, 54.554893], [255, 0, 255, 0]), ([22.647102, -25.978674, 61.125065], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([2.3483024, 6.7722263, 25.086443], [255, 0, 255, 0]), ([0.7735524, 0.061484337, 20.85053], [255, 0, 255, 0]), ([8.347038, 1.6859608, 23.187845], [255, 0, 255, 0]), ([2.3483024, 6.7722263, 25.086443], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([21.277773, -34.541046, 44.325447], [255, 0, 255, 0]), ([25.616655, -29.675524, 39.91157], [255, 0, 255, 0]), ([20.220644, -34.30311, 36.527355], [255, 0, 255, 0]), ([21.277773, -34.541046, 44.325447], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([2.8499534, -36.937725, 34.53498], [255, 0, 255, 0]), ([3.8248277, -34.078213, 28.772886], [255, 0, 255, 0]), ([-2.2260892, -35.394844, 30.768806], [255, 0, 255, 0]), ([2.8499534, -36.937725, 34.53498], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([8.891672, -10.049717, 62.961212], [255, 0, 255, 0]), ([7.132907, -16.680082, 62.339996], [255, 0, 255, 0]), ([2.53094, -12.096637, 64.63279], [255, 0, 255, 0]), ([8.891672, -10.049717, 62.961212], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-6.4518557, 23.1594, 48.51089], [255, 0, 255, 0]), ([-13.830787, 20.1011, 51.81328], [255, 0, 255, 0]), ([-13.124069, 21.416862, 43.299744], [255, 0, 255, 0]), ([-6.4518557, 23.1594, 48.51089], [255, 0, 255, 0])],
            [([0.0, -9.4921875, 44.0], [255, 255, 255, 0]), ([-4.1047697, 0.11841869, 17.403889], [255, 0, 255, 0]), ([-8.780803, 5.0776415, 21.039536], [255, 0, 255, 0]), ([-11.410234, -1.8743005, 18.933264], [255, 0, 255, 0]), ([-4.1047697, 0.11841869, 17.403889], [255, 0, 255, 0])],
        ],
        mask: Mask { count: 0, min_x: 0, max_x: 0, min_y: 0, max_y: 0, cx: 0.0000, cy: 0.0000 },
        mean: [0.0000, 0.0000, 0.0000],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "turned",
        camera: [20.0, -10.0],
        death_ticks: 120, partial: 0.6,
        // the layer's entry modelview (column-major): the dragon's living-model matrix
        base: [-1.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 10.0, -2.4921875, 41.0, 1.0],
        state: State { blend: true, blend_func: [770, 1], alpha_test: false, cull: true, cull_face: 1029, front_face: 2305, depth_test: true, depth_func: 515, depth_mask: false, lighting: false, fog: true, texture0: false, texture1: true, shade_model: 7425, lightmap: [0.0, 0.0] },
        // 29 shards: camera-relative position, rgba; GL_TRIANGLE_FAN order
        shards: &[
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([-4.1254435, -11.661589, 32.479862], [255, 0, 255, 0]), ([-1.208581, -14.670155, 32.205444], [255, 0, 255, 0]), ([-1.7159061, -11.943608, 29.052], [255, 0, 255, 0]), ([-4.1254435, -11.661589, 32.479862], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([18.019072, 6.5635357, 27.849857], [255, 0, 255, 0]), ([14.892106, 4.9241757, 25.273258], [255, 0, 255, 0]), ([18.46825, 2.565567, 26.14117], [255, 0, 255, 0]), ([18.019072, 6.5635357, 27.849857], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([17.924, -18.257015, 40.05575], [255, 0, 255, 0]), ([19.775055, -17.190674, 40.723267], [255, 0, 255, 0]), ([19.392288, -17.509, 38.541195], [255, 0, 255, 0]), ([17.924, -18.257015, 40.05575], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([3.078826, 7.1983395, 34.961742], [255, 0, 255, 0]), ([2.244525, 7.1245594, 36.689217], [255, 0, 255, 0]), ([1.6299562, 5.9787426, 35.276688], [255, 0, 255, 0]), ([3.078826, 7.1983395, 34.961742], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([10.203503, -0.4139396, 32.95882], [255, 0, 255, 0]), ([14.25132, -1.919167, 34.590477], [255, 0, 255, 0]), ([12.946066, 2.447658, 35.325966], [255, 0, 255, 0]), ([10.203503, -0.4139396, 32.95882], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([5.5625057, 2.2527332, 40.14403], [255, 0, 255, 0]), ([8.544321, 4.1095266, 37.76453], [255, 0, 255, 0]), ([9.174642, 3.5846586, 41.927334], [255, 0, 255, 0]), ([5.5625057, 2.2527332, 40.14403], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([22.087357, -1.7183326, 46.737347], [255, 0, 255, 0]), ([19.88742, 0.47159636, 49.217842], [255, 0, 255, 0]), ([21.835766, 2.2140052, 46.225014], [255, 0, 255, 0]), ([22.087357, -1.7183326, 46.737347], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([3.769858, 10.698954, 48.391068], [255, 0, 255, 0]), ([4.1553416, 9.340244, 50.141327], [255, 0, 255, 0]), ([2.2090116, 9.207709, 49.02215], [255, 0, 255, 0]), ([3.769858, 10.698954, 48.391068], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([6.7403636, 18.542355, 43.87412], [255, 0, 255, 0]), ([6.6776347, 17.451084, 47.116623], [255, 0, 255, 0]), ([3.8089294, 17.438147, 45.251278], [255, 0, 255, 0]), ([6.7403636, 18.542355, 43.87412], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([11.892821, 3.4902568, 36.687134], [255, 0, 255, 0]), ([9.010901, 2.8937058, 35.31999], [255, 0, 255, 0]), ([11.6672, 1.3797035, 34.23243], [255, 0, 255, 0]), ([11.892821, 3.4902568, 36.687134], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([9.419111, 19.631416, 27.89561], [255, 0, 255, 0]), ([7.820189, 20.03048, 28.906698], [255, 0, 255, 0]), ([7.6857862, 19.1301, 27.200966], [255, 0, 255, 0]), ([9.419111, 19.631416, 27.89561], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([11.373159, -9.372634, 30.09996], [255, 0, 255, 0]), ([12.6469555, -7.7405624, 29.14357], [255, 0, 255, 0]), ([10.390083, -7.7642264, 28.81633], [255, 0, 255, 0]), ([11.373159, -9.372634, 30.09996], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([3.6578312, -9.328225, 29.09757], [255, 0, 255, 0]), ([0.5938282, -7.994446, 30.69764], [255, 0, 255, 0]), ([2.2222662, -11.031722, 32.05817], [255, 0, 255, 0]), ([3.6578312, -9.328225, 29.09757], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([8.387064, -0.29109383, 52.544422], [255, 0, 255, 0]), ([6.4577293, 3.236651, 51.352573], [255, 0, 255, 0]), ([10.621433, 3.1899204, 51.85253], [255, 0, 255, 0]), ([8.387064, -0.29109383, 52.544422], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([31.29667, -13.694404, 41.769592], [255, 0, 255, 0]), ([31.912657, -12.852616, 39.94291], [255, 0, 255, 0]), ([30.865221, -14.676784, 39.96023], [255, 0, 255, 0]), ([31.29667, -13.694404, 41.769592], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([20.331085, -12.883951, 48.770725], [255, 0, 255, 0]), ([23.39975, -10.595638, 47.339333], [255, 0, 255, 0]), ([22.003166, -13.746301, 45.142612], [255, 0, 255, 0]), ([20.331085, -12.883951, 48.770725], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([0.04252243, 12.380294, 50.202045], [255, 0, 255, 0]), ([3.5307708, 14.266124, 50.265327], [255, 0, 255, 0]), ([2.726142, 11.493113, 52.98414], [255, 0, 255, 0]), ([0.04252243, 12.380294, 50.202045], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([12.741484, -6.0252523, 45.18321], [255, 0, 255, 0]), ([16.362871, -3.798548, 43.526703], [255, 0, 255, 0]), ([14.84799, -7.5621552, 41.439125], [255, 0, 255, 0]), ([12.741484, -6.0252523, 45.18321], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([-0.72346115, 14.267867, 51.02694], [255, 0, 255, 0]), ([3.9256978, 15.735744, 52.20304], [255, 0, 255, 0]), ([1.6723795, 12.151078, 54.891235], [255, 0, 255, 0]), ([-0.72346115, 14.267867, 51.02694], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([25.788826, -2.039371, 28.625721], [255, 0, 255, 0]), ([24.981503, -5.34198, 28.13971], [255, 0, 255, 0]), ([26.819191, -4.4098105, 30.887377], [255, 0, 255, 0]), ([25.788826, -2.039371, 28.625721], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([4.625949, -19.398314, 29.78831], [255, 0, 255, 0]), ([9.1974745, -20.349268, 30.160788], [255, 0, 255, 0]), ([7.5950894, -17.964893, 26.460793], [255, 0, 255, 0]), ([4.625949, -19.398314, 29.78831], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([3.272325, -0.7427571, 42.182045], [255, 0, 255, 0]), ([6.0869875, -1.0909156, 45.36207], [255, 0, 255, 0]), ([4.7340035, -4.5792503, 43.322987], [255, 0, 255, 0]), ([3.272325, -0.7427571, 42.182045], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([13.089369, 8.083848, 20.887438], [255, 0, 255, 0]), ([9.37145, 9.494414, 21.444117], [255, 0, 255, 0]), ([9.923981, 5.9402213, 19.659285], [255, 0, 255, 0]), ([13.089369, 8.083848, 20.887438], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([19.205677, 0.3066194, 45.189217], [255, 0, 255, 0]), ([20.195812, -1.4154894, 43.72506], [255, 0, 255, 0]), ([18.96143, -2.0829916, 45.75503], [255, 0, 255, 0]), ([19.205677, 0.3066194, 45.189217], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([-7.122078, -0.38469124, 54.01981], [255, 0, 255, 0]), ([-7.3320656, 3.3892422, 52.991356], [255, 0, 255, 0]), ([-4.897547, 2.270537, 55.849094], [255, 0, 255, 0]), ([-7.122078, -0.38469124, 54.01981], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([23.45736, -4.1790833, 41.36996], [255, 0, 255, 0]), ([23.836971, -1.7859143, 37.4545], [255, 0, 255, 0]), ([22.959362, -6.304058, 37.31538], [255, 0, 255, 0]), ([23.45736, -4.1790833, 41.36996], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([1.3024712, -15.1939535, 41.264023], [255, 0, 255, 0]), ([5.148716, -17.130901, 39.63791], [255, 0, 255, 0]), ([1.9432688, -15.577998, 36.721733], [255, 0, 255, 0]), ([1.3024712, -15.1939535, 41.264023], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([15.494666, 14.802549, 22.649708], [255, 0, 255, 0]), ([18.01752, 16.27334, 25.466812], [255, 0, 255, 0]), ([14.146973, 17.48808, 25.376703], [255, 0, 255, 0]), ([15.494666, 14.802549, 22.649708], [255, 0, 255, 0])],
            [([10.0, -1.4921875, 39.0], [255, 255, 255, 255]), ([11.06719, 10.463456, 42.970684], [255, 0, 255, 0]), ([8.60712, 8.768987, 46.253334], [255, 0, 255, 0]), ([6.660366, 10.181608, 42.523125], [255, 0, 255, 0]), ([11.06719, 10.463456, 42.970684], [255, 0, 255, 0])],
        ],
        mask: Mask { count: 88813, min_x: 0, max_x: 395, min_y: 23, max_y: 539, cx: 171.0813, cy: 316.8440 },
        mean: [96.9961, 57.0185, 112.3896],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [8145.0, 47.430, 26.317, 63.430], [2143.0, 61.150, 30.394, 77.150], [2802.0, 52.405, 27.820, 68.405], [0.0, 0.000, 0.000, 0.000],
            [4218.0, 48.541, 26.795, 64.541], [7572.0, 102.182, 51.679, 118.182], [6587.0, 96.511, 51.448, 112.270], [2194.0, 76.631, 34.523, 92.631],
            [7997.0, 95.735, 48.617, 111.735], [10223.0, 153.319, 101.119, 167.202], [9132.0, 157.407, 116.271, 170.051], [2017.0, 78.926, 43.687, 94.926],
            [4829.0, 73.294, 40.439, 89.294], [5915.0, 101.571, 55.118, 117.571], [9830.0, 100.348, 54.761, 116.348], [5209.0, 57.062, 29.860, 73.062],
        ],
    },
];
