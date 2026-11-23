//! The guardian beam renders like vanilla 1.8.9's `RenderGuardian.doRender`.
//!
//! The reference is vanilla, not a re-derivation of it. `javaref/GuardianBeamRef.java`
//! (`generate-guardian-beam.ps1`) runs the unmodified recompiled 1.8.9
//! `RenderManager.renderEntityStatic` over a real `RenderGuardian` in an LWJGL Pbuffer, with the
//! vanilla jar's resources and the world pass's state at `renderEntities`, and records for every
//! case the beam's own Tessellator buffer (camera-relative positions, UVs, RGBA bytes), the live GL
//! state at its draw, and the real pixels with the guardian body suppressed, reduced to coverage
//! statistics against the clear colour. What the reference established (`the_java_reference_*`):
//!
//! * the beam is drawn with blending **off** (`disableBlend()` comes before the unused
//!   `tryBlendFuncSeparate(770, 1, 1, 0)`), the world pass's alpha test (`GL_GREATER 0.1`) still
//!   on, depth `GL_LEQUAL` with writes on, culling and lighting off, fog on, and the lightmap
//!   stage at `(240, 240)` whatever the guardian's own light; `guardian_beam.png` repeats in both
//!   directions with nearest filtering and no mip levels. Its texels are fully opaque or fully
//!   transparent, so the beam is crisp opaque lines, and the transparent texels write no depth;
//! * its colour bytes are `64 + (int)(f * f * 240)`, `32 + (int)(f * f * 192)` and
//!   `128 - (int)(f * f * 64)` stored through `(byte)` casts, so from progress 0.894 the red
//!   channel wraps and the beam turns green for the last tenth of the wind-up.
//!
//! The comparisons rebuild each case from mc-rs's own code:
//!
//! * `guardian_beam_vertices_match_java` (CPU): `objects::emit_guardian_beam` against the Java
//!   buffer: three triangulated quads whose corners lie within 2e-4 blocks of Java's, UVs within
//!   1e-6, colour bytes exact;
//! * `guardian_beam_pixels_match_java` (GPU): those vertices under `LayerBlend::GuardianBeam`
//!   through the real [`EntityPhase`], with `guardian_beam.png` loaded from the vanilla jar the way
//!   the client loads it, against Java's coverage, mean colour and colour grid. Two negative
//!   controls must fail: the beam without the alpha test (`LayerBlend::Additive`), and the beam's
//!   twelve quad corners submitted as a triangle list.
//!
//! Skips (loudly) without the vanilla jar (set `MC_JARS_DIR`) or a wgpu adapter. Set
//! `MC_GB_DUMP=<dir>` to write each mc-rs frame as a PNG, and `MC_GB_JAVA_DIR=<dir>` (the
//! generator's output folder) to also print an exact per-pixel comparison with Java's frames.

// The tables are generated: their values are data, and one may land near a constant.
#![allow(clippy::approx_constant)]

use std::path::{Path, PathBuf};

use glam::DVec3;
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::entity::objects;
use mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT;
use mc_render::entity::{EntityPhase, EntityVertex, LayerBlend};
use mc_render::sky::lightmap::Lightmap;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 960;
const H: u32 = 540;
/// Vanilla's default field of view, which `GuardianBeamRef` renders with.
const FOV: f32 = 70.0;
/// `GuardianBeamRef.EYE_*`: the camera, in world coordinates.
const EYE: DVec3 = DVec3::new(0.0, 64.0, 0.0);
/// `GuardianBeamRef.CLEAR`.
const CLEAR: [u8; 3] = [0x18, 0x30, 0x60];
/// `GuardianBeamRef.LIGHTMAP`: every texel of the uniform lightmap, the full-bright corner of
/// `EntityRenderer.updateLightmap` in daylight.
const LIGHTMAP: u8 = 252;
/// `GuardianBeamRef.BACKDROP`: camera-relative corners, drawn after the beam.
const BACKDROP: [[f32; 3]; 4] = [
    [-10.0, 4.0, 10.0],
    [10.0, 4.0, 10.0],
    [10.0, -10.0, 10.0],
    [-10.0, -10.0, 10.0],
];
/// `GuardianBeamRef.BACKDROP_RGB`.
const BACKDROP_RGB: [u8; 3] = [0x70, 0x90, 0x50];
/// Where a `GL_QUADS` corner lands in a triangle-list quad (`0-1-2 / 0-2-3`).
const CORNERS: [usize; 4] = [0, 1, 2, 5];

// GL enums as `glGet` returned them.
const GL_SRC_ALPHA: i32 = 770;
const GL_ONE: i32 = 1;
const GL_ZERO: i32 = 0;
const GL_GREATER: i32 = 516;
const GL_LEQUAL: i32 = 515;
const GL_REPEAT: i32 = 10497;
const GL_NEAREST: i32 = 9728;

/// What the guardian targets: its height picks the beam's end point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    RemotePlayer,
    LocalPlayer,
    Guardian,
}

/// The live GL state at the beam's draw call.
#[derive(Debug, Clone, Copy)]
struct State {
    blend: bool,
    /// `GL_BLEND_SRC_RGB`, `GL_BLEND_DST_RGB`, `GL_BLEND_SRC_ALPHA`, `GL_BLEND_DST_ALPHA`.
    blend_func: [i32; 4],
    alpha_test: bool,
    alpha_func: i32,
    alpha_ref: f32,
    depth_test: bool,
    depth_func: i32,
    depth_mask: bool,
    cull: bool,
    lighting: bool,
    fog: bool,
    /// `GL_TEXTURE_2D` on unit 0 (the beam texture) and unit 1 (the lightmap).
    texture0: bool,
    texture1: bool,
    /// The bound texture's `GL_TEXTURE_WRAP_S`/`_T`.
    wrap: [i32; 2],
    /// The bound texture's `GL_TEXTURE_MIN_FILTER`/`_MAG_FILTER`.
    filter: [i32; 2],
    /// The bound texture's `GL_TEXTURE_MAX_LEVEL`.
    max_level: i32,
    /// Unit 1's current texture coordinate: the lightmap coordinate of every beam vertex.
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

/// One vanilla render.
struct Ref {
    name: &'static str,
    /// Camera yaw and pitch, degrees.
    camera: [f32; 2],
    elder: bool,
    /// The guardian's feet: previous tick, current tick.
    guardian: [[f64; 3]; 2],
    /// `EntityGuardian.getEyeHeight()`.
    eye_height: f32,
    target: Target,
    /// The target's feet: previous tick, current tick.
    target_feet: [[f64; 3]; 2],
    /// The target's `height`.
    target_height: f32,
    /// `EntityGuardian.field_175479_bo` and `func_175464_ck()`.
    attack_ticks: u32,
    max_attack_ticks: u32,
    ticks_existed: u32,
    /// `World.getTotalWorldTime()`.
    world_time: i64,
    partial: f32,
    /// Whether the opaque backdrop is drawn after the beam.
    backdrop: bool,
    state: State,
    /// The beam buffer, `GL_QUADS` order: camera-relative position, UV, RGBA bytes.
    verts: [([f32; 3], [f32; 2], [u8; 4]); 12],
    mask: Mask,
    mean: [f64; 3],
    /// Row-major 4x4 cells over the mask bounds: `[pixels, mean r, g, b]`.
    grid: [[f64; 4]; 16],
}

fn lerp(prev: [f64; 3], pos: [f64; 3], partial: f32) -> DVec3 {
    let prev = DVec3::from_array(prev);
    prev + (DVec3::from_array(pos) - prev) * f64::from(partial)
}

/// `RenderGuardian.doRender`'s inputs, as the client's beam hook builds them: the guardian's
/// interpolated feet plus its eye height, the target's interpolated feet plus half its height,
/// and `func_175477_p(partialTicks)`.
fn emit(r: &Ref) -> Vec<EntityVertex> {
    let guardian_eye = lerp(r.guardian[0], r.guardian[1], r.partial)
        + DVec3::new(0.0, f64::from(r.eye_height), 0.0);
    let target_centre = lerp(r.target_feet[0], r.target_feet[1], r.partial)
        + DVec3::new(0.0, f64::from(r.target_height * 0.5), 0.0);
    let progress = (r.attack_ticks as f32 + r.partial) / r.max_attack_ticks as f32;
    let mut out = Vec::new();
    objects::emit_guardian_beam(
        guardian_eye,
        target_centre,
        EYE,
        r.world_time,
        progress,
        r.ticks_existed,
        r.partial,
        &mut out,
    );
    out
}

#[test]
fn the_java_reference_draws_an_opaque_alpha_tested_full_bright_two_sided_beam() {
    for r in REFS {
        let s = r.state;
        assert!(
            !s.blend,
            "{}: vanilla draws the beam with blending off",
            r.name
        );
        // The function is set but unused: `disableBlend()` came first.
        assert_eq!(
            s.blend_func,
            [GL_SRC_ALPHA, GL_ONE, GL_ONE, GL_ZERO],
            "{}",
            r.name
        );
        assert!(s.alpha_test && s.alpha_func == GL_GREATER, "{}", r.name);
        assert!((s.alpha_ref - 0.1).abs() < 1e-6, "{}", r.name);
        assert!(
            s.depth_test && s.depth_func == GL_LEQUAL && s.depth_mask,
            "{}",
            r.name
        );
        assert!(!s.cull && !s.lighting, "{}", r.name);
        assert!(
            s.fog,
            "{}: the beam is fogged like the rest of the pass",
            r.name
        );
        assert!(
            s.texture0 && s.texture1,
            "{}: texture and lightmap stages",
            r.name
        );
        assert_eq!(s.wrap, [GL_REPEAT, GL_REPEAT], "{}", r.name);
        assert_eq!(s.filter, [GL_NEAREST, GL_NEAREST], "{}", r.name);
        assert_eq!(s.max_level, 0, "{}", r.name);
        // Full bright although the reference world's combined light is 0 everywhere.
        assert_eq!(s.lightmap, [240.0, 240.0], "{}", r.name);
        // Opaque vertices, one colour per draw.
        assert!(
            r.verts.iter().all(|v| v.2 == r.verts[0].2 && v.2[3] == 255),
            "{}",
            r.name
        );
        let (eye, ticks) = if r.elder { (0.998_75, 60) } else { (0.425, 80) };
        assert_eq!(
            (r.eye_height, r.max_attack_ticks),
            (eye, ticks),
            "{}",
            r.name
        );
        let height = if r.target == Target::Guardian {
            0.85
        } else {
            1.8
        };
        assert_eq!(r.target_height, height, "{}", r.name);
    }
    // The colour ramp's byte overflow: green near the end of the wind-up.
    let colour = |name: &str| {
        REFS.iter()
            .find(|r| r.name == name)
            .map(|r| r.verts[0].2)
            .expect("case")
    };
    assert_eq!(colour("start_side"), [64, 32, 128, 255]);
    assert_eq!(colour("green_wrap"), [24, 205, 71, 255]);
    assert_eq!(colour("full_charge"), [52, 227, 63, 255]);
}

#[test]
fn guardian_beam_vertices_match_java() {
    let mut failures = Vec::new();
    for r in REFS {
        let out = emit(r);
        if out.len() != 18 {
            failures.push(format!(
                "{}: {} vertices; Java's 3 GL_QUADS are 18 triangle-list vertices",
                r.name,
                out.len()
            ));
            continue;
        }
        let (mut worst_position, mut worst_uv) = (0.0_f64, 0.0_f32);
        for quad in 0..3 {
            for (corner, &index) in CORNERS.iter().enumerate() {
                let (java_pos, java_uv, _) = r.verts[quad * 4 + corner];
                let ours = out[quad * 6 + index];
                for (a, b) in ours.pos.iter().zip(java_pos) {
                    worst_position = worst_position.max(f64::from(*a - b).abs());
                }
                for (a, b) in ours.uv.iter().zip(java_uv) {
                    worst_uv = worst_uv.max((a - b).abs());
                }
            }
        }
        eprintln!(
            "{:16} worst position error {worst_position:.2e} blocks, worst uv error {worst_uv:.2e}",
            r.name
        );
        for quad in 0..3 {
            let q = &out[quad * 6..quad * 6 + 6];
            let same = |a: &EntityVertex, b: &EntityVertex| a.pos == b.pos && a.uv == b.uv;
            if !same(&q[3], &q[0]) || !same(&q[4], &q[2]) {
                failures.push(format!("{} quad {quad}: not a 0-1-2 / 0-2-3 quad", r.name));
            }
            for (corner, &index) in CORNERS.iter().enumerate() {
                let (java_pos, java_uv, java_rgba) = r.verts[quad * 4 + corner];
                let ours = q[index];
                let distance = ours
                    .pos
                    .iter()
                    .zip(java_pos)
                    .map(|(a, b)| f64::from(*a - b).abs())
                    .fold(0.0, f64::max);
                if distance > 2e-4 {
                    failures.push(format!(
                        "{} quad {quad} corner {corner}: position {:?} vs Java {java_pos:?}",
                        r.name, ours.pos
                    ));
                }
                if ours
                    .uv
                    .iter()
                    .zip(java_uv)
                    .any(|(a, b)| (a - b).abs() > 1e-6)
                {
                    failures.push(format!(
                        "{} quad {quad} corner {corner}: uv {:?} vs Java {java_uv:?}",
                        r.name, ours.uv
                    ));
                }
                let bytes = ours.color.map(|c| (c * 255.0).round());
                let exact = ours
                    .color
                    .iter()
                    .zip(bytes)
                    .all(|(c, b)| (c * 255.0 - b).abs() < 1e-3);
                if !exact || bytes != java_rgba.map(f32::from) {
                    failures.push(format!(
                        "{} quad {quad} corner {corner}: colour {:?} vs Java bytes {java_rgba:?}",
                        r.name, ours.color
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "mc-rs's guardian beam is not vanilla's RenderGuardian buffer:\n{}",
        failures.join("\n")
    );
}

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let dir = PathBuf::from(dir);
        return dir.is_dir().then_some(dir);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let dir = root.join("MCP-919-main").join("jars");
    dir.is_dir().then_some(dir)
}

/// A resource manager over the vanilla jar, the only pack the reference used.
fn vanilla_resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(resources)
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

/// How one case's beam is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Draws {
    /// The client's state: the triangulated beam under `LayerBlend::GuardianBeam`.
    Client,
    /// Negative control: no alpha test (the additive layer state, which has none).
    NoCutout,
    /// Negative control: the twelve `GL_QUADS` corners submitted as a triangle list.
    QuadCorners,
}

/// The backdrop quad as a 4-vertex strip under the untextured `LayerBlend::Leash` state, which
/// draws the vertex colour itself.
fn backdrop() -> Vec<EntityVertex> {
    let color = [
        f32::from(BACKDROP_RGB[0]) / 255.0,
        f32::from(BACKDROP_RGB[1]) / 255.0,
        f32::from(BACKDROP_RGB[2]) / 255.0,
        1.0,
    ];
    [0usize, 1, 3, 2]
        .iter()
        .map(|&i| EntityVertex {
            pos: BACKDROP[i],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color,
        })
        .collect()
}

/// Render one case through the real `EntityPhase` and return its RGBA pixels.
fn render_case(gpu: &mut Gpu, resources: &ResourceManager, r: &Ref, mode: Draws) -> Vec<u8> {
    let camera = camera(r);
    let mut phase = EntityPhase::new(gpu);
    assert!(
        phase.load_texture(gpu, resources, objects::GUARDIAN_BEAM_TEXTURE),
        "guardian_beam.png from the vanilla jar"
    );
    phase.set_lightmap(
        gpu,
        &Lightmap {
            rgba: [LIGHTMAP, LIGHTMAP, LIGHTMAP, 255].repeat(256),
        },
    );
    phase.begin();
    let verts = emit(r);
    let (verts, blend) = match mode {
        Draws::Client => (verts, LayerBlend::GuardianBeam),
        Draws::NoCutout => (verts, LayerBlend::Additive),
        Draws::QuadCorners => (
            verts
                .chunks_exact(6)
                .flat_map(|quad| CORNERS.map(|i| quad[i]))
                .collect(),
            LayerBlend::GuardianBeam,
        ),
    };
    phase.push_blended_lit(
        objects::GUARDIAN_BEAM_TEXTURE,
        blend,
        FULL_BRIGHT_COMBINED_LIGHT,
        &verts,
    );
    if r.backdrop {
        // Untextured and unlit, as Java draws it.
        phase.push_blended_lit(
            EntityPhase::SOLID_TEXTURE,
            LayerBlend::Leash,
            FULL_BRIGHT_COMBINED_LIGHT,
            &backdrop(),
        );
    }
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

/// Coverage statistics and the mean colour, as `GuardianBeamRef` computes them.
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
    let mut out = Vec::new();
    // GL and wgpu agree on pixel centres and the top-left rule; what is left is pixels whose
    // centre or texel coordinate sits within float noise of an edge.
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
    let dir = PathBuf::from(std::env::var_os("MC_GB_JAVA_DIR")?);
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
fn check_cases(
    gpu: &mut Gpu,
    resources: &ResourceManager,
    mode: Draws,
    dump: Option<&Path>,
) -> Vec<String> {
    let mut failures = Vec::new();
    for r in REFS {
        let pixels = render_case(gpu, resources, r, mode);
        if let Some(dir) = dump {
            std::fs::create_dir_all(dir).expect("dump dir");
            let suffix = match mode {
                Draws::Client => "",
                Draws::NoCutout => "-nocutout",
                Draws::QuadCorners => "-quadcorners",
            };
            image::RgbaImage::from_raw(W, H, pixels.clone())
                .expect("rgba frame")
                .save(dir.join(format!("rs-{}{suffix}.png", r.name)))
                .expect("write png");
        }
        let (ours, mean) = stats_of(&pixels);
        eprintln!(
            "{mode:?} {:16} n {:6} vs {:6}  c ({:7.2}, {:7.2}) vs ({:7.2}, {:7.2})  mean \
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
fn guardian_beam_pixels_match_java() {
    let Some(resources) = vanilla_resources() else {
        eprintln!("SKIP: no vanilla jar; set MC_JARS_DIR");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    let dump = std::env::var_os("MC_GB_DUMP").map(PathBuf::from);
    let failures = check_cases(&mut gpu, &resources, Draws::Client, dump.as_deref());
    assert!(
        failures.is_empty(),
        "mc-rs's guardian beam does not render like vanilla's:\n{}",
        failures.join("\n")
    );
    // Negative controls: the comparison must see the alpha test and the triangulation.
    for control in [Draws::NoCutout, Draws::QuadCorners] {
        let failing = check_cases(&mut gpu, &resources, control, dump.as_deref());
        assert_eq!(
            failing.len(),
            REFS.len(),
            "{control:?} must fail every case, failed only: {failing:#?}"
        );
    }
}

// -------------------------------------------------------------------------------------------
// Generated by GuardianBeamRef on this machine:
// GL_VENDOR=NVIDIA Corporation GL_RENDERER=NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2 GL_VERSION=4.6.0 NVIDIA 610.88
// -------------------------------------------------------------------------------------------

#[rustfmt::skip]
const REFS: &[Ref] = &[
    Ref {
        name: "start_side",
        camera: [0.0, 5.0],
        elder: false,
        guardian: [[-2.5, 62.8, 5.0], [-2.5, 62.8, 5.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[2.5, 62.0, 5.5], [2.5, 62.0, 5.5]],
        target_height: 1.8,
        attack_ticks: 0, max_attack_ticks: 80, ticks_existed: 10,
        world_time: 1000, partial: 0.0, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.4796, -1.087148, 5.78323], [0.4999, 14.088593], [64, 32, 128, 255]),
            ([-2.5133626, -0.69760543, 5.1839333], [0.4999, -1.0], [64, 32, 128, 255]),
            ([-2.4866374, -0.85239464, 4.8160667], [0.0, -1.0], [64, 32, 128, 255]),
            ([3.5063248, -1.2419372, 5.415363], [0.0, 14.088593], [64, 32, 128, 255]),
            ([3.4734058, -1.3485084, 5.675284], [0.4999, 14.088593], [64, 32, 128, 255]),
            ([-2.5195565, -0.9589659, 5.0759873], [0.4999, -1.0], [64, 32, 128, 255]),
            ([-2.4804435, -0.5910342, 4.9240127], [0.0, -1.0], [64, 32, 128, 255]),
            ([3.512519, -0.98057663, 5.5233088], [0.0, 14.088593], [64, 32, 128, 255]),
            ([3.4601412, -1.2707963, 5.858443], [0.5, 1.0], [64, 32, 128, 255]),
            ([3.4867873, -1.4251244, 5.491672], [1.0, 1.0], [64, 32, 128, 255]),
            ([3.5257835, -1.0582888, 5.34015], [1.0, 0.5], [64, 32, 128, 255]),
            ([3.4991374, -0.9039607, 5.7069206], [0.5, 0.5], [64, 32, 128, 255]),
        ],
        mask: Mask { count: 2272, min_x: 236, max_x: 673, min_y: 285, max_y: 326, cx: 458.5317, cy: 304.8776 },
        mean: [55.6651, 27.5647, 111.2562],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [4.0, 54.000, 27.000, 109.000], [78.0, 54.615, 27.205, 109.821], [249.0, 55.446, 27.482, 110.928],
            [62.0, 55.081, 27.403, 110.613], [229.0, 55.533, 27.511, 111.044], [323.0, 55.848, 27.616, 111.464], [308.0, 55.558, 27.519, 111.078],
            [350.0, 56.051, 27.731, 111.920], [302.0, 55.589, 27.530, 111.119], [185.0, 55.768, 27.589, 111.357], [31.0, 56.323, 27.774, 112.097],
            [141.0, 55.794, 27.617, 111.440], [10.0, 57.000, 28.000, 113.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "quarter_side",
        camera: [0.0, 5.0],
        elder: false,
        guardian: [[-2.5, 62.8, 5.0], [-2.5, 62.8, 5.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[2.5, 62.0, 5.5], [2.5, 62.0, 5.5]],
        target_height: 1.8,
        attack_ticks: 20, max_attack_ticks: 80, ticks_existed: 11,
        world_time: 1037, partial: 0.0, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.4984274, -1.3027346, 5.454821], [0.4999, 14.588593], [79, 44, 124, 255]),
            ([-2.494535, -0.9131922, 4.855525], [0.4999, -0.5], [79, 44, 124, 255]),
            ([-2.505465, -0.63680786, 5.144475], [0.0, -0.5], [79, 44, 124, 255]),
            ([3.4874973, -1.0263504, 5.7437716], [0.0, 14.588593], [79, 44, 124, 255]),
            ([3.5160089, -1.0205415, 5.462429], [0.4999, 14.588593], [79, 44, 124, 255]),
            ([-2.4769533, -0.630999, 4.863133], [0.4999, -0.5], [79, 44, 124, 255]),
            ([-2.5230467, -0.91900104, 5.136867], [0.0, -0.5], [79, 44, 124, 255]),
            ([3.4699159, -1.3085436, 5.736163], [0.0, 14.588593], [79, 44, 124, 255]),
            ([3.5213895, -1.158751, 5.3187923], [0.5, 0.5], [79, 44, 124, 255]),
            ([3.5104918, -0.88319016, 5.6068816], [1.0, 0.5], [79, 44, 124, 255]),
            ([3.4645352, -1.1703341, 5.8798], [1.0, 0.0], [79, 44, 124, 255]),
            ([3.4754329, -1.445895, 5.5917106], [0.5, 0.0], [79, 44, 124, 255]),
        ],
        mask: Mask { count: 2304, min_x: 234, max_x: 671, min_y: 285, max_y: 327, cx: 454.7691, cy: 305.3811 },
        mean: [68.8424, 38.2218, 108.0642],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [63.0, 68.048, 37.698, 106.746], [251.0, 68.649, 38.100, 107.749],
            [74.0, 68.338, 37.892, 107.230], [241.0, 68.693, 38.129, 107.822], [374.0, 68.741, 38.160, 107.901], [323.0, 68.765, 38.176, 107.941],
            [376.0, 69.263, 38.487, 108.750], [311.0, 68.756, 38.170, 107.926], [147.0, 68.735, 38.156, 107.891], [16.0, 69.625, 38.750, 109.375],
            [127.0, 69.677, 38.732, 109.409], [1.0, 70.000, 39.000, 110.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "half_moving",
        camera: [0.0, 5.0],
        elder: false,
        guardian: [[-2.6, 62.8, 5.0], [-2.5, 62.85, 5.1]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[2.4, 62.0, 5.5], [2.6, 62.0, 5.4]],
        target_height: 1.8,
        attack_ticks: 40, max_attack_ticks: 80, ticks_existed: 12,
        world_time: 2000, partial: 0.5, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.4925053, -1.0316225, 5.6741805], [0.4999, 14.444733], [125, 81, 112, 255]),
            ([-2.5520012, -0.6126957, 5.1954083], [0.4999, -0.75], [125, 81, 112, 255]),
            ([-2.5479987, -0.8873042, 4.904592], [0.0, -0.75], [125, 81, 112, 255]),
            ([3.496508, -1.306231, 5.383364], [0.0, 14.444733], [125, 81, 112, 255]),
            ([3.4736683, -1.3136938, 5.6651845], [0.4999, 14.444733], [125, 81, 112, 255]),
            ([-2.5708382, -0.894767, 5.1864123], [0.4999, -0.75], [125, 81, 112, 255]),
            ([-2.5291617, -0.6052329, 4.913588], [0.0, -0.75], [125, 81, 112, 255]),
            ([3.5153449, -1.0241598, 5.39236], [0.0, 14.444733], [125, 81, 112, 255]),
            ([3.4717348, -1.1763673, 5.809753], [0.5, 1.0], [125, 81, 112, 255]),
            ([3.475726, -1.4501576, 5.5198035], [1.0, 1.0], [125, 81, 112, 255]),
            ([3.5172784, -1.1614863, 5.247792], [1.0, 0.5], [125, 81, 112, 255]),
            ([3.5132873, -0.8876959, 5.5377417], [0.5, 0.5], [125, 81, 112, 255]),
        ],
        mask: Mask { count: 2329, min_x: 232, max_x: 674, min_y: 285, max_y: 327, cx: 457.1217, cy: 305.6185 },
        mean: [108.2671, 70.6840, 97.2508],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [67.0, 108.388, 70.791, 97.388], [291.0, 108.282, 70.711, 97.282],
            [32.0, 108.000, 70.500, 97.000], [213.0, 108.178, 70.634, 97.178], [378.0, 108.339, 70.754, 97.339], [307.0, 108.111, 70.583, 97.111],
            [347.0, 108.674, 70.937, 97.605], [322.0, 108.286, 70.714, 97.286], [153.0, 107.778, 70.333, 96.778], [12.0, 106.000, 69.000, 95.000],
            [192.0, 108.375, 70.708, 97.302], [15.0, 106.267, 69.200, 95.267], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "late_oblique",
        camera: [-10.0, 20.0],
        elder: false,
        guardian: [[-1.5, 63.0, 6.0], [-1.5, 63.0, 6.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[3.0, 61.5, 3.0], [3.0, 61.5, 3.0]],
        target_height: 1.8,
        attack_ticks: 68, max_attack_ticks: 80, ticks_existed: 13,
        world_time: 5005, partial: 0.0, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.8434267, -1.5899069, 2.4268255], [0.4999, 15.761501], [237, 170, 82, 255]),
            ([-1.4740713, -0.37869972, 5.9718237], [0.4999, -0.5], [237, 170, 82, 255]),
            ([-1.5259287, -0.77130026, 6.0281763], [0.0, -0.5], [237, 170, 82, 255]),
            ([3.7915697, -1.9825075, 2.4831781], [0.0, 15.761501], [237, 170, 82, 255]),
            ([3.929728, -1.7773044, 2.620305], [0.4999, 15.761501], [237, 170, 82, 255]),
            ([-1.3877699, -0.56609714, 6.165303], [0.4999, -0.5], [237, 170, 82, 255]),
            ([-1.6122301, -0.58390284, 5.834697], [0.0, -0.5], [237, 170, 82, 255]),
            ([3.7052684, -1.79511, 2.2896986], [0.0, 15.761501], [237, 170, 82, 255]),
            ([3.955245, -1.5816154, 2.5917203], [0.5, 0.5], [237, 170, 82, 255]),
            ([3.903542, -1.9730463, 2.6479049], [1.0, 0.5], [237, 170, 82, 255]),
            ([3.6797514, -1.990799, 2.3182833], [1.0, 0.0], [237, 170, 82, 255]),
            ([3.7314544, -1.5993681, 2.2620988], [0.5, 0.0], [237, 170, 82, 255]),
        ],
        mask: Mask { count: 3674, min_x: 99, max_x: 665, min_y: 163, max_y: 359, cx: 354.1595, cy: 266.9110 },
        mean: [206.0321, 147.7643, 71.3038],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [36.0, 206.111, 147.833, 71.278], [667.0, 205.618, 147.463, 71.154],
            [0.0, 0.000, 0.000, 0.000], [120.0, 206.867, 148.400, 71.467], [785.0, 205.525, 147.394, 71.131], [55.0, 204.345, 146.509, 70.836],
            [231.0, 206.403, 148.052, 71.351], [859.0, 205.536, 147.402, 71.134], [28.0, 202.714, 145.286, 70.429], [0.0, 0.000, 0.000, 0.000],
            [887.0, 207.277, 148.667, 71.759], [6.0, 205.000, 147.000, 71.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "green_wrap",
        camera: [0.0, 5.0],
        elder: false,
        guardian: [[-2.5, 62.8, 5.0], [-2.5, 62.8, 5.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[2.5, 62.0, 5.5], [2.5, 62.0, 5.5]],
        target_height: 1.8,
        attack_ticks: 76, max_attack_ticks: 80, ticks_existed: 14,
        world_time: 6000, partial: 0.0, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.5162692, -1.0281707, 5.454868], [0.4999, 14.088593], [24, 205, 71, 255]),
            ([-2.476693, -0.63862824, 4.8555717], [0.4999, -1.0], [24, 205, 71, 255]),
            ([-2.523307, -0.9113718, 5.1444283], [0.0, -1.0], [24, 205, 71, 255]),
            ([3.4696555, -1.3009143, 5.7437243], [0.0, 14.088593], [24, 205, 71, 255]),
            ([3.4887428, -1.0188165, 5.7362127], [0.4999, 14.088593], [24, 205, 71, 255]),
            ([-2.5042195, -0.62927395, 5.136916], [0.4999, -1.0], [24, 205, 71, 255]),
            ([-2.4957805, -0.9207261, 4.863084], [0.0, -1.0], [24, 205, 71, 255]),
            ([3.497182, -1.3102686, 5.46238], [0.0, 14.088593], [24, 205, 71, 255]),
            ([3.511993, -0.8832851, 5.591807], [0.5, 1.0], [24, 205, 71, 255]),
            ([3.465518, -1.1552161, 5.8798027], [1.0, 1.0], [24, 205, 71, 255]),
            ([3.4739318, -1.4458, 5.606786], [1.0, 0.5], [24, 205, 71, 255]),
            ([3.5204067, -1.1738689, 5.31879], [0.5, 0.5], [24, 205, 71, 255]),
        ],
        mask: Mask { count: 2187, min_x: 236, max_x: 672, min_y: 287, max_y: 326, cx: 453.7579, cy: 307.3532 },
        mean: [20.5857, 178.1527, 61.7590],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [49.0, 21.000, 181.000, 63.000], [178.0, 20.624, 178.365, 61.871],
            [36.0, 21.056, 181.528, 63.167], [176.0, 20.744, 179.210, 62.233], [268.0, 20.623, 178.362, 61.869], [321.0, 20.520, 177.642, 61.561],
            [305.0, 20.708, 179.121, 62.134], [338.0, 20.541, 177.790, 61.624], [221.0, 20.412, 176.882, 61.235], [49.0, 20.122, 174.857, 60.367],
            [213.0, 20.563, 178.225, 61.695], [33.0, 20.061, 174.424, 60.182], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "full_charge",
        camera: [0.0, 5.0],
        elder: false,
        guardian: [[-2.5, 62.8, 5.0], [-2.5, 62.8, 5.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[2.5, 62.0, 5.5], [2.5, 62.0, 5.5]],
        target_height: 1.8,
        attack_ticks: 80, max_attack_ticks: 80, ticks_existed: 15,
        world_time: 6100, partial: 0.75, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.4957657, -0.9852227, 5.6878204], [0.4999, 14.463593], [52, 227, 63, 255]),
            ([-2.4971967, -0.59568024, 5.0885243], [0.4999, -0.625], [52, 227, 63, 255]),
            ([-2.5028033, -0.95431983, 4.9114757], [0.0, -0.625], [52, 227, 63, 255]),
            ([3.490159, -1.3438623, 5.5107718], [0.0, 14.463593], [52, 227, 63, 255]),
            ([3.4694428, -1.2521654, 5.777535], [0.4999, 14.463593], [52, 227, 63, 255]),
            ([-2.5235193, -0.862623, 5.178239], [0.4999, -0.625], [52, 227, 63, 255]),
            ([-2.4764807, -0.6873771, 4.821761], [0.0, -0.625], [52, 227, 63, 255]),
            ([3.5164819, -1.0769196, 5.4210577], [0.0, 14.463593], [52, 227, 63, 255]),
            ([3.4723082, -1.0731189, 5.8652644], [0.5, 0.5], [52, 227, 63, 255]),
            ([3.4667177, -1.4306898, 5.6887436], [1.0, 0.5], [52, 227, 63, 255]),
            ([3.5136166, -1.2559662, 5.333328], [1.0, 0.0], [52, 227, 63, 255]),
            ([3.519207, -0.8983952, 5.509849], [0.5, 0.0], [52, 227, 63, 255]),
        ],
        mask: Mask { count: 2185, min_x: 233, max_x: 674, min_y: 285, max_y: 327, cx: 454.8551, cy: 307.2126 },
        mean: [45.2005, 197.8142, 54.7844],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [1.0, 46.000, 201.000, 56.000], [40.0, 46.000, 201.000, 56.000], [180.0, 45.367, 198.467, 55.050],
            [55.0, 46.164, 201.709, 56.164], [188.0, 45.223, 197.894, 54.835], [297.0, 45.205, 197.822, 54.808], [291.0, 45.189, 197.756, 54.784],
            [290.0, 45.421, 198.745, 55.048], [302.0, 45.159, 197.636, 54.738], [221.0, 44.986, 196.946, 54.480], [82.0, 44.268, 194.073, 53.402],
            [192.0, 45.188, 197.781, 54.745], [46.0, 44.261, 194.043, 53.391], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "elder_at_viewer",
        camera: [0.0, 30.0],
        elder: true,
        guardian: [[-1.0, 60.5, 6.0], [-1.0, 60.5, 6.0]],
        eye_height: 0.99875,
        target: Target::LocalPlayer,
        target_feet: [[0.0, 62.38, 0.0], [0.0, 62.38, 0.0]],
        target_height: 1.8,
        attack_ticks: 30, max_attack_ticks: 60, ticks_existed: 16,
        world_time: 7000, partial: 0.25, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([-0.02932012, -0.36906028, -0.9570694], [0.4999, 17.470514], [125, 80, 112, 255]),
            ([-1.1870949, -2.431345, 5.9895706], [0.4999, -0.875], [125, 80, 112, 255]),
            ([-0.812905, -2.571155, 6.0104294], [0.0, -0.875], [125, 80, 112, 255]),
            ([0.34486985, -0.50887024, -0.93621063], [0.0, 17.470514], [125, 80, 112, 255]),
            ([0.22101855, -0.26020813, -0.8830309], [0.4999, 17.470514], [125, 80, 112, 255]),
            ([-0.9367562, -2.3224928, 6.063609], [0.4999, -0.875], [125, 80, 112, 255]),
            ([-1.0632437, -2.6800072, 5.936391], [0.0, -0.875], [125, 80, 112, 255]),
            ([0.09453106, -0.6177224, -1.0102491], [0.0, 17.470514], [125, 80, 112, 255]),
            ([0.03429258, -0.19104409, -0.8936186], [0.5, 1.0], [125, 80, 112, 255]),
            ([0.4073677, -0.33043742, -0.8728223], [1.0, 1.0], [125, 80, 112, 255]),
            ([0.28125703, -0.68688655, -0.99966145], [1.0, 0.5], [125, 80, 112, 255]),
            ([-0.091818035, -0.5474931, -1.0204577], [0.5, 0.5], [125, 80, 112, 255]),
        ],
        mask: Mask { count: 6257, min_x: 449, max_x: 584, min_y: 218, max_y: 539, cx: 519.9248, cy: 422.8749 },
        mean: [108.4229, 69.8172, 97.4229],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [31.0, 106.258, 68.194, 95.258], [563.0, 108.274, 69.705, 97.274], [31.0, 110.000, 71.000, 99.000],
            [0.0, 0.000, 0.000, 0.000], [416.0, 108.212, 69.659, 97.212], [666.0, 108.090, 69.568, 97.090], [87.0, 110.000, 71.000, 99.000],
            [127.0, 107.764, 69.323, 96.764], [828.0, 108.507, 69.880, 97.507], [711.0, 108.065, 69.549, 97.065], [154.0, 110.000, 71.000, 99.000],
            [697.0, 106.482, 68.362, 95.482], [641.0, 108.746, 70.059, 97.746], [798.0, 109.233, 70.425, 98.233], [507.0, 110.000, 71.000, 99.000],
        ],
    },
    Ref {
        name: "end_on",
        camera: [0.0, 0.0],
        elder: false,
        guardian: [[0.0, 63.0, 2.5], [0.0, 63.0, 2.5]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[0.4, 62.6, 10.0], [0.4, 62.6, 10.0]],
        target_height: 1.8,
        attack_ticks: 50, max_attack_ticks: 80, ticks_existed: 16,
        world_time: 7007, partial: 0.0, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([0.58057225, -0.33599764, 10.9902], [0.4999, 20.777584], [157, 107, 103, 255]),
            ([0.1273172, -0.42098397, 2.4916697], [0.4999, -0.5], [157, 107, 103, 255]),
            ([-0.1273172, -0.729016, 2.5083303], [0.0, -0.5], [157, 107, 103, 255]),
            ([0.32593778, -0.6440296, 11.006861], [0.0, 20.777584], [157, 107, 103, 255]),
            ([0.29938203, -0.36243984, 11.005461], [0.4999, 20.777584], [157, 107, 103, 255]),
            ([-0.15387295, -0.4474262, 2.5069308], [0.4999, -0.5], [157, 107, 103, 255]),
            ([0.15387295, -0.7025738, 2.4930692], [0.0, -0.5], [157, 107, 103, 255]),
            ([0.6071279, -0.61758745, 10.9916], [0.0, 20.777584], [157, 107, 103, 255]),
            ([0.42677838, -0.20926282, 10.997134], [0.5, 1.0], [157, 107, 103, 255]),
            ([0.17290263, -0.51637703, 11.013746], [1.0, 1.0], [157, 107, 103, 255]),
            ([0.47973162, -0.77076447, 10.999927], [1.0, 0.5], [157, 107, 103, 255]),
            ([0.73360735, -0.46365023, 10.9833145], [0.5, 0.5], [157, 107, 103, 255]),
        ],
        mask: Mask { count: 501, min_x: 459, max_x: 496, min_y: 283, max_y: 375, cx: 472.6457, cy: 329.9591 },
        mean: [136.7325, 93.4810, 89.4990],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [67.0, 138.343, 94.507, 90.642], [26.0, 136.923, 93.615, 89.615], [1.0, 139.000, 95.000, 91.000], [0.0, 0.000, 0.000, 0.000],
            [60.0, 136.300, 93.200, 89.200], [50.0, 136.240, 93.160, 89.160], [22.0, 138.182, 94.455, 90.455], [0.0, 0.000, 0.000, 0.000],
            [59.0, 136.559, 93.373, 89.373], [50.0, 136.600, 93.400, 89.400], [61.0, 136.246, 93.164, 89.164], [26.0, 139.000, 95.000, 91.000],
            [35.0, 134.886, 92.257, 88.257], [17.0, 136.529, 93.353, 89.353], [17.0, 136.176, 93.118, 89.118], [10.0, 133.600, 91.400, 87.400],
        ],
    },
    Ref {
        name: "straight_down",
        camera: [0.0, 10.0],
        elder: false,
        guardian: [[0.0, 65.0, 5.0], [0.0, 65.0, 5.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[0.0, 59.1, 5.0], [0.0, 59.1, 5.0]],
        target_height: 1.8,
        attack_ticks: 30, max_attack_ticks: 80, ticks_existed: 17,
        world_time: 3000, partial: 0.0, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([0.18755636, -4.999955, 5.073464], [0.4999, 15.0625], [97, 59, 119, 255]),
            ([0.18601894, 1.4250444, 5.073464], [0.4999, -1.0], [97, 59, 119, 255]),
            ([-0.18601894, 1.4249555, 4.926536], [0.0, -1.0], [97, 59, 119, 255]),
            ([-0.18448155, -5.000044, 4.926536], [0.0, 15.0625], [97, 59, 119, 255]),
            ([-0.07192644, -5.000017, 5.186019], [0.4999, 15.0625], [97, 59, 119, 255]),
            ([-0.07346384, 1.4249824, 5.186019], [0.4999, -1.0], [97, 59, 119, 255]),
            ([0.07346384, 1.4250175, 4.813981], [0.0, -1.0], [97, 59, 119, 255]),
            ([0.07500125, -4.999982, 4.813981], [0.0, 15.0625], [97, 59, 119, 255]),
            ([0.113757156, -4.9999723, 5.25871], [0.5, 0.5], [97, 59, 119, 255]),
            ([-0.25717226, -5.000061, 5.11222], [1.0, 0.5], [97, 59, 119, 255]),
            ([-0.110682346, -5.0000257, 4.74129], [1.0, 0.0], [97, 59, 119, 255]),
            ([0.26024705, -4.999937, 4.88778], [0.5, 0.0], [97, 59, 119, 255]),
        ],
        mask: Mask { count: 2355, min_x: 469, max_x: 493, min_y: 81, max_y: 539, cx: 481.2435, cy: 305.8333 },
        mean: [84.2357, 51.1261, 103.2463],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [122.0, 84.459, 51.230, 103.459], [218.0, 84.202, 51.101, 103.202], [171.0, 84.713, 51.357, 103.713], [100.0, 82.960, 50.480, 101.960],
            [116.0, 84.793, 51.397, 103.793], [217.0, 84.378, 51.189, 103.378], [195.0, 84.728, 51.364, 103.728], [92.0, 82.739, 50.370, 101.739],
            [97.0, 84.598, 51.299, 103.598], [193.0, 83.907, 50.953, 102.907], [199.0, 84.613, 51.307, 103.613], [73.0, 82.548, 50.274, 101.548],
            [112.0, 85.304, 51.732, 104.411], [201.0, 84.050, 51.055, 103.085], [185.0, 84.697, 51.373, 103.730], [64.0, 82.375, 50.188, 101.375],
        ],
    },
    Ref {
        name: "big_world_time",
        camera: [-20.0, 5.0],
        elder: false,
        guardian: [[-2.5, 62.8, 5.0], [-2.5, 62.8, 5.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[2.5, 62.0, 5.5], [2.5, 62.0, 5.5]],
        target_height: 1.8,
        attack_ticks: 48, max_attack_ticks: 80, ticks_existed: 18,
        world_time: 123456789, partial: 0.6, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.4840136, -1.0501716, 5.763123], [0.4999, 14.088593], [152, 102, 105, 255]),
            ([-2.5089486, -0.66062903, 5.163827], [0.4999, -1.0], [152, 102, 105, 255]),
            ([-2.4910514, -0.88937104, 4.836173], [0.0, -1.0], [152, 102, 105, 255]),
            ([3.5019112, -1.2789135, 5.4354696], [0.0, 14.088593], [152, 102, 105, 255]),
            ([3.4710321, -1.328105, 5.7122846], [0.4999, 14.088593], [152, 102, 105, 255]),
            ([-2.5219305, -0.9385624, 5.1129885], [0.4999, -1.0], [152, 102, 105, 255]),
            ([-2.4780695, -0.6114377, 4.8870115], [0.0, -1.0], [152, 102, 105, 255]),
            ([3.5148926, -1.0009801, 5.4863076], [0.0, 14.088593], [152, 102, 105, 255]),
            ([3.4621754, -1.2135874, 5.8752866], [0.5, 1.0], [152, 102, 105, 255]),
            ([3.480019, -1.4416478, 5.5486093], [1.0, 1.0], [152, 102, 105, 255]),
            ([3.5237494, -1.1154977, 5.3233056], [1.0, 0.5], [152, 102, 105, 255]),
            ([3.5059056, -0.8874373, 5.649983], [0.5, 0.5], [152, 102, 105, 255]),
        ],
        mask: Mask { count: 2864, min_x: 394, max_x: 884, min_y: 296, max_y: 324, cx: 660.8135, cy: 309.5409 },
        mean: [131.8027, 88.6875, 91.2367],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [228.0, 132.303, 89.026, 91.605], [177.0, 131.797, 88.678, 91.237], [136.0, 130.875, 88.125, 90.500], [101.0, 130.683, 88.010, 90.347],
            [244.0, 132.320, 89.016, 91.631], [234.0, 131.799, 88.679, 91.239], [262.0, 131.901, 88.740, 91.321], [310.0, 132.065, 88.839, 91.452],
            [96.0, 131.490, 88.500, 90.979], [256.0, 131.715, 88.629, 91.172], [305.0, 132.000, 88.800, 91.400], [243.0, 131.798, 88.679, 91.239],
            [0.0, 0.000, 0.000, 0.000], [3.0, 129.000, 87.000, 89.000], [63.0, 130.587, 87.952, 90.270], [206.0, 131.670, 88.602, 91.136],
        ],
    },
    Ref {
        name: "guardian_target",
        camera: [0.0, 5.0],
        elder: false,
        guardian: [[-2.5, 62.8, 5.0], [-2.5, 62.8, 5.0]],
        eye_height: 0.425,
        target: Target::Guardian,
        target_feet: [[2.5, 62.5, 6.0], [2.5, 62.5, 6.0]],
        target_height: 0.85,
        attack_ticks: 10, max_attack_ticks: 80, ticks_existed: 20,
        world_time: 999, partial: 0.1, backdrop: false,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.4490147, -1.0440745, 6.3720407], [0.4999, 14.819581], [67, 35, 127, 255]),
            ([-2.5298731, -0.6853409, 5.1762633], [0.4999, -0.4500122], [67, 35, 127, 255]),
            ([-2.4701269, -0.8646592, 4.8237367], [0.0, -0.4500122], [67, 35, 127, 255]),
            ([3.5087605, -1.223393, 6.0195136], [0.0, 14.819581], [67, 35, 127, 255]),
            ([3.450982, -1.3121245, 6.281789], [0.4999, 14.819581], [67, 35, 127, 255]),
            ([-2.5279057, -0.9533907, 5.086012], [0.4999, -0.4500122], [67, 35, 127, 255]),
            ([-2.4720943, -0.59660935, 4.913988], [0.0, -0.4500122], [67, 35, 127, 255]),
            ([3.506793, -0.95534307, 6.109765], [0.0, 14.819581], [67, 35, 127, 255]),
            ([3.4212804, -1.2222009, 6.4572706], [0.5, 1.0], [67, 35, 127, 255]),
            ([3.4808488, -1.400985, 6.105794], [1.0, 1.0], [67, 35, 127, 255]),
            ([3.5364947, -1.0452666, 5.9342833], [1.0, 0.5], [67, 35, 127, 255]),
            ([3.4769263, -0.8664825, 6.28576], [0.5, 0.5], [67, 35, 127, 255]),
        ],
        mask: Mask { count: 2058, min_x: 260, max_x: 674, min_y: 285, max_y: 316, cx: 470.6147, cy: 300.3426 },
        mean: [58.1409, 30.5690, 110.2847],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [14.0, 57.000, 30.000, 108.000], [64.0, 57.469, 30.234, 108.938], [168.0, 58.024, 30.512, 110.048],
            [137.0, 57.978, 30.489, 109.956], [238.0, 58.134, 30.567, 110.269], [241.0, 58.170, 30.585, 110.340], [234.0, 58.162, 30.581, 110.325],
            [245.0, 58.527, 30.759, 111.061], [230.0, 58.078, 30.539, 110.157], [220.0, 58.182, 30.591, 110.364], [131.0, 58.145, 30.573, 110.290],
            [117.0, 58.111, 30.538, 110.256], [19.0, 58.368, 30.684, 110.737], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "occlusion",
        camera: [-10.0, 20.0],
        elder: false,
        guardian: [[-1.5, 63.0, 6.0], [-1.5, 63.0, 6.0]],
        eye_height: 0.425,
        target: Target::RemotePlayer,
        target_feet: [[3.0, 61.5, 3.0], [3.0, 61.5, 3.0]],
        target_height: 1.8,
        attack_ticks: 40, max_attack_ticks: 80, ticks_existed: 19,
        world_time: 4321, partial: 0.0, backdrop: true,
        state: State { blend: false, blend_func: [770, 1, 1, 0], alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [240.0, 240.0] },
        // camera-relative position, uv, rgba; GL_QUADS order
        verts: [
            ([3.7342715, -1.6935625, 2.2985082], [0.4999, 15.761501], [124, 80, 112, 255]),
            ([-1.5832267, -0.4823552, 5.8435063], [0.4999, -0.5], [124, 80, 112, 255]),
            ([-1.4167733, -0.6676448, 6.1564937], [0.0, -0.5], [124, 80, 112, 255]),
            ([3.900725, -1.8788521, 2.6114955], [0.0, 15.761501], [124, 80, 112, 255]),
            ([3.8971295, -1.6129155, 2.5152414], [0.4999, 15.761501], [124, 80, 112, 255]),
            ([-1.4203684, -0.4017083, 6.0602393], [0.4999, -0.5], [124, 80, 112, 255]),
            ([-1.5796316, -0.7482917, 5.9397607], [0.0, -0.5], [124, 80, 112, 255]),
            ([3.7378669, -1.9594989, 2.3947623], [0.0, 15.761501], [124, 80, 112, 255]),
            ([3.8139138, -1.5210631, 2.3590345], [0.5, 0.5], [124, 80, 112, 255]),
            ([3.9798713, -1.7058005, 2.6710894], [1.0, 0.5], [124, 80, 112, 255]),
            ([3.8210826, -2.0513513, 2.5509691], [1.0, 0.0], [124, 80, 112, 255]),
            ([3.6551251, -1.8666139, 2.2389143], [0.5, 0.0], [124, 80, 112, 255]),
        ],
        mask: Mask { count: 339731, min_x: 99, max_x: 959, min_y: 0, max_y: 500, cx: 591.4143, cy: 223.3098 },
        mean: [111.9609, 143.2621, 80.1740],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [17425.0, 112.000, 144.000, 80.000], [27090.0, 112.000, 144.000, 80.000], [27090.0, 112.000, 144.000, 80.000], [27090.0, 112.000, 144.000, 80.000],
            [13530.0, 112.000, 144.000, 80.000], [26875.0, 111.894, 142.047, 80.457], [26875.0, 111.902, 142.301, 80.392], [26875.0, 112.000, 144.000, 80.000],
            [10793.0, 111.503, 133.612, 82.504], [26875.0, 111.910, 142.496, 80.345], [26875.0, 112.000, 144.000, 80.000], [26665.0, 112.000, 144.000, 80.000],
            [2564.0, 112.000, 144.000, 80.000], [13418.0, 112.000, 144.000, 80.000], [19705.0, 112.000, 144.000, 80.000], [19986.0, 112.000, 144.000, 80.000],
        ],
    },
];
