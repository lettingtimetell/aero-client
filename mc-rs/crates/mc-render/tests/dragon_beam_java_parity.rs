//! The ender dragon's crystal beam renders like vanilla 1.8.9's
//! `RenderDragon.drawRechargeRay`.
//!
//! The reference is vanilla: `javaref/DragonBeamRef.java` (`generate-dragon-beam.ps1`) runs the
//! unmodified recompiled 1.8.9 `RenderManager.renderEntityStatic` over a real `RenderDragon` whose
//! dragon has a `healingEnderCrystal`, in an LWJGL Pbuffer with the vanilla jar's resources and the
//! world pass's state at `renderEntities`, and records the beam's Tessellator buffer, the live GL
//! state at its draw and the real pixels (the dragon's body suppressed). What it established
//! (`the_java_reference_*`):
//!
//! * the beam is one 18-vertex `GL_TRIANGLE_STRIP` (16 triangles), smooth-shaded from black at the
//!   dragon to white at the crystal;
//! * it is drawn with blending off, the world pass's alpha test (`GL_GREATER 0.1`) on, depth
//!   `GL_LEQUAL` with writes, culling and lighting off, fog on, and the lightmap stage at the
//!   dragon's own brightness (`LayerEnderDragonEyes` restores it with `setLightmap`): `(0, 0)` in
//!   the reference world, whose light is 0. `endercrystal_beam.png` is all fully opaque or fully
//!   transparent texels (its transparent ones white), so the beam is opaque lines.
//!
//! The comparisons rebuild each case from mc-rs's own code:
//!
//! * `dragon_beam_vertices_match_java` (CPU): `emit_dragon_beam` must be the strip's 16 triangles
//!   as a triangle list, every vertex within 2e-3 blocks of Java's, UVs within 1e-5, colours exact;
//! * `dragon_beam_pixels_match_java` (GPU): those vertices under `LayerBlend::DragonBeam` through the
//!   real [`EntityPhase`], with `endercrystal_beam.png` loaded from the vanilla jar and the
//!   dragon's combined light, against Java's coverage, mean colour and colour grid. Two negative
//!   controls must fail: the 18 strip vertices submitted as a triangle list, and the beam without
//!   the alpha test.
//!
//! Skips (loudly) without the vanilla jar (set `MC_JARS_DIR`) or a wgpu adapter. Set
//! `MC_DRAGON_BEAM_DUMP=<dir>` to write each mc-rs frame as a PNG, and
//! `MC_DRAGON_BEAM_JAVA_DIR=<dir>` (the generator's output folder) to also print an exact
//! per-pixel comparison with Java's frames.

// The tables are generated: their values are data, and one may land near a constant.
#![allow(clippy::approx_constant)]

use std::path::{Path, PathBuf};

use glam::DVec3;
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::entity::{EntityPhase, EntityVertex, LayerBlend, DRAGON_BEAM_TEXTURE};
use mc_render::sky::lightmap::Lightmap;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 960;
const H: u32 = 540;
/// Vanilla's default field of view, which `DragonBeamRef` renders with.
const FOV: f32 = 70.0;
/// `DragonBeamRef.EYE_*`: the camera, in world coordinates.
const EYE: DVec3 = DVec3::new(0.0, 64.0, 0.0);
/// `DragonBeamRef.CLEAR`.
const CLEAR: [u8; 3] = [0x20, 0x18, 0x30];
/// `DragonBeamRef.LIGHTMAP` and `LIGHTMAP_DARK`: every texel but (block 0, sky 0), and that one.
const LIGHTMAP: u8 = 252;
const LIGHTMAP_DARK: u8 = 150;
/// The dragon's packed combined light in the reference world (its light is 0 everywhere).
const DRAGON_COMBINED_LIGHT: u32 = 0;
/// `DragonBeamRef.BACKDROP`: camera-relative corners, drawn after the beam.
const BACKDROP: [[f32; 3]; 4] = [
    [-30.0, 12.0, 30.0],
    [30.0, 12.0, 30.0],
    [30.0, -20.0, 30.0],
    [-30.0, -20.0, 30.0],
];
/// `DragonBeamRef.BACKDROP_RGB`.
const BACKDROP_RGB: [u8; 3] = [0x50, 0x70, 0x90];
/// Cases the pixel comparison skips. A dragon 123,456,789 ticks old puts its beam's V coordinate
/// near -1.2 million, where the float the rasteriser interpolates no longer resolves a texel: GL
/// and wgpu pick different texels from the same vertices. The vertex comparison (exact UVs)
/// covers that case.
const PIXEL_EXEMPT: &[&str] = &["huge_ticks"];

// GL enums as `glGet` returned them.
const GL_GREATER: i32 = 516;
const GL_LEQUAL: i32 = 515;
const GL_REPEAT: i32 = 10497;
const GL_NEAREST: i32 = 9728;
const GL_SMOOTH: i32 = 7425;

/// The live GL state at the beam's draw call.
#[derive(Debug, Clone, Copy)]
struct State {
    blend: bool,
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
    shade_model: i32,
    wrap: [i32; 2],
    filter: [i32; 2],
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
    /// The dragon's feet: previous tick, current tick.
    dragon: [[f64; 3]; 2],
    /// The crystal's current position (`drawRechargeRay` does not interpolate it).
    crystal: [f64; 3],
    /// `EntityEnderCrystal.innerRotation`.
    inner_rotation: i32,
    /// The dragon's `ticksExisted`.
    ticks_existed: u32,
    partial: f32,
    /// Whether the opaque backdrop is drawn after the beam.
    backdrop: bool,
    state: State,
    /// The beam strip, `GL_TRIANGLE_STRIP` order: camera-relative position, UV, RGBA bytes.
    verts: [([f32; 3], [f32; 2], [u8; 4]); 18],
    mask: Mask,
    mean: [f64; 3],
    /// Row-major 4x4 cells over the mask bounds: `[pixels, mean r, g, b]`.
    grid: [[f64; 4]; 16],
}

/// `drawRechargeRay`'s inputs, as the client's beam hook builds them: the dragon's interpolated
/// feet and the crystal's current position, both camera-relative.
fn emit(r: &Ref) -> Vec<EntityVertex> {
    let prev = DVec3::from_array(r.dragon[0]);
    let dragon = prev + (DVec3::from_array(r.dragon[1]) - prev) * f64::from(r.partial);
    let mut out = Vec::new();
    mc_render::entity::emit_dragon_beam(
        (dragon - EYE).as_vec3(),
        (DVec3::from_array(r.crystal) - EYE).as_vec3(),
        r.inner_rotation as f32,
        r.ticks_existed,
        r.partial,
        &mut out,
    );
    out
}

/// The strip vertices of GL triangle `i` (`i, i+1, i+2`; odd triangles swap the first two to keep
/// the strip's winding).
fn strip_triangle(i: usize) -> [usize; 3] {
    if i.is_multiple_of(2) {
        [i, i + 1, i + 2]
    } else {
        [i + 1, i, i + 2]
    }
}

#[test]
fn the_java_reference_draws_an_opaque_alpha_tested_strip_at_the_dragons_light() {
    for r in REFS {
        let s = r.state;
        assert!(!s.blend, "{}: blending off", r.name);
        assert!(s.alpha_test && s.alpha_func == GL_GREATER, "{}", r.name);
        assert!((s.alpha_ref - 0.1).abs() < 1e-6, "{}", r.name);
        assert!(
            s.depth_test && s.depth_func == GL_LEQUAL && s.depth_mask,
            "{}",
            r.name
        );
        assert!(!s.cull && !s.lighting, "{}", r.name);
        assert!(s.fog, "{}", r.name);
        assert!(
            s.texture0 && s.texture1,
            "{}: texture and lightmap stages",
            r.name
        );
        assert_eq!(s.shade_model, GL_SMOOTH, "{}", r.name);
        assert_eq!(s.wrap, [GL_REPEAT, GL_REPEAT], "{}", r.name);
        assert_eq!(s.filter, [GL_NEAREST, GL_NEAREST], "{}", r.name);
        assert_eq!(s.max_level, 0, "{}", r.name);
        // The dragon's own light, which `LayerEnderDragonEyes.setLightmap` restored.
        assert_eq!(s.lightmap, [0.0, 0.0], "{}", r.name);
        // Black at the dragon, white at the crystal, alternating along the strip.
        for (i, v) in r.verts.iter().enumerate() {
            let expected = if i.is_multiple_of(2) {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            };
            assert_eq!(v.2, expected, "{} vertex {i}", r.name);
        }
    }
}

#[test]
fn dragon_beam_vertices_match_java() {
    let mut failures = Vec::new();
    for r in REFS {
        let out = emit(r);
        if out.len() != 48 {
            failures.push(format!(
                "{}: {} vertices; Java's 18-vertex strip is 16 triangles, 48 list vertices",
                r.name,
                out.len()
            ));
            continue;
        }
        let (mut worst_position, mut worst_uv) = (0.0_f64, 0.0_f32);
        for triangle in 0..16 {
            for (k, &strip) in strip_triangle(triangle).iter().enumerate() {
                let (java_pos, java_uv, java_rgba) = r.verts[strip];
                let ours = out[triangle * 3 + k];
                let distance = ours
                    .pos
                    .iter()
                    .zip(java_pos)
                    .map(|(a, b)| f64::from(*a - b).abs())
                    .fold(0.0, f64::max);
                let uv = ours
                    .uv
                    .iter()
                    .zip(java_uv)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                worst_position = worst_position.max(distance);
                worst_uv = worst_uv.max(uv);
                if distance > 2e-3 || uv > 1e-5 {
                    failures.push(format!(
                        "{} triangle {triangle} vertex {k}: {:?} {:?} vs Java {java_pos:?} \
                         {java_uv:?}",
                        r.name, ours.pos, ours.uv
                    ));
                }
                let bytes = ours.color.map(|c| (c * 255.0).round());
                if bytes != java_rgba.map(f32::from) {
                    failures.push(format!(
                        "{} triangle {triangle} vertex {k}: colour {:?} vs Java bytes \
                         {java_rgba:?}",
                        r.name, ours.color
                    ));
                }
            }
        }
        eprintln!(
            "{:14} worst position error {worst_position:.2e} blocks, worst uv error {worst_uv:.2e}",
            r.name
        );
    }
    assert!(
        failures.is_empty(),
        "mc-rs's crystal beam is not vanilla's drawRechargeRay strip:\n{}",
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
    /// The client's state: the beam's triangles under `LayerBlend::DragonBeam`.
    Client,
    /// Negative control: the 18 strip vertices submitted as a triangle list.
    StripAsList,
    /// Negative control: no alpha test (the additive layer state, which has none).
    NoCutout,
}

/// The backdrop quad as a 4-vertex strip under the untextured `LayerBlend::Leash` state.
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

/// The strip the triangles were built from: the first triangle's three vertices, then each later
/// triangle's last one.
fn strip_of(triangles: &[EntityVertex]) -> Vec<EntityVertex> {
    let mut strip = triangles.iter().take(3).copied().collect::<Vec<_>>();
    strip.extend(triangles.chunks_exact(3).skip(1).map(|t| t[2]));
    strip
}

/// Render one case through the real `EntityPhase` and return its RGBA pixels.
fn render_case(gpu: &mut Gpu, resources: &ResourceManager, r: &Ref, mode: Draws) -> Vec<u8> {
    let camera = camera(r);
    let mut phase = EntityPhase::new(gpu);
    assert!(
        phase.load_texture(gpu, resources, DRAGON_BEAM_TEXTURE),
        "endercrystal_beam.png from the vanilla jar"
    );
    let mut lightmap = [LIGHTMAP, LIGHTMAP, LIGHTMAP, 255].repeat(256);
    lightmap[..3].copy_from_slice(&[LIGHTMAP_DARK; 3]);
    phase.set_lightmap(gpu, &Lightmap { rgba: lightmap });
    phase.begin();
    let verts = emit(r);
    let (verts, blend) = match mode {
        Draws::Client => (verts, LayerBlend::DragonBeam),
        Draws::StripAsList => (strip_of(&verts), LayerBlend::DragonBeam),
        Draws::NoCutout => (verts, LayerBlend::Additive),
    };
    phase.push_blended_lit(DRAGON_BEAM_TEXTURE, blend, DRAGON_COMBINED_LIGHT, &verts);
    if r.backdrop {
        phase.push_blended_lit(
            EntityPhase::SOLID_TEXTURE,
            LayerBlend::Leash,
            DRAGON_COMBINED_LIGHT,
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

/// Coverage statistics and the mean colour, as `DragonBeamRef` computes them.
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
    let dir = PathBuf::from(std::env::var_os("MC_DRAGON_BEAM_JAVA_DIR")?);
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
    for r in REFS.iter().filter(|r| !PIXEL_EXEMPT.contains(&r.name)) {
        let pixels = render_case(gpu, resources, r, mode);
        if let Some(dir) = dump {
            std::fs::create_dir_all(dir).expect("dump dir");
            let suffix = match mode {
                Draws::Client => "",
                Draws::StripAsList => "-stripaslist",
                Draws::NoCutout => "-nocutout",
            };
            image::RgbaImage::from_raw(W, H, pixels.clone())
                .expect("rgba frame")
                .save(dir.join(format!("rs-{}{suffix}.png", r.name)))
                .expect("write png");
        }
        let (ours, mean) = stats_of(&pixels);
        eprintln!(
            "{mode:?} {:14} n {:6} vs {:6}  c ({:7.2}, {:7.2}) vs ({:7.2}, {:7.2})  mean \
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
fn dragon_beam_pixels_match_java() {
    let Some(resources) = vanilla_resources() else {
        eprintln!("SKIP: no vanilla jar; set MC_JARS_DIR");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    let dump = std::env::var_os("MC_DRAGON_BEAM_DUMP").map(PathBuf::from);
    let failures = check_cases(&mut gpu, &resources, Draws::Client, dump.as_deref());
    assert!(
        failures.is_empty(),
        "mc-rs's crystal beam does not render like vanilla's:\n{}",
        failures.join("\n")
    );
    // Negative controls: the comparison must see the triangulation and the alpha test. The
    // backdrop case judges what the beam's transparent texels hide, and its backdrop outweighs
    // the beam's own pixels, so only the alpha-test control has to fail it.
    for control in [Draws::StripAsList, Draws::NoCutout] {
        let failing = check_cases(&mut gpu, &resources, control, dump.as_deref());
        for r in REFS.iter().filter(|r| !PIXEL_EXEMPT.contains(&r.name)) {
            if r.backdrop && control != Draws::NoCutout {
                continue;
            }
            assert!(
                failing
                    .iter()
                    .any(|failure| failure.starts_with(&format!("{}:", r.name))),
                "{control:?} must fail {}, failed only: {failing:#?}",
                r.name
            );
        }
    }
}

// -------------------------------------------------------------------------------------------
// Generated by DragonBeamRef on this machine:
// GL_VENDOR=NVIDIA Corporation GL_RENDERER=NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2 GL_VERSION=4.6.0 NVIDIA 610.88
// -------------------------------------------------------------------------------------------

#[rustfmt::skip]
const REFS: &[Ref] = &[
    Ref {
        name: "side",
        camera: [0.0, 0.0],
        dragon: [[-8.0, 58.0, 20.0], [-8.0, 58.0, 20.0]],
        crystal: [6.0, 62.0, 18.0],
        inner_rotation: 1234, ticks_existed: 100, partial: 0.25, backdrop: false,
        state: State { blend: false, alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, shade_model: 7425, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [0.0, 0.0] },
        // camera-relative position, uv, rgba; GL_TRIANGLE_STRIP order
        verts: [
            ([-7.9653683, -4.1458635, 19.995052], [0.0, -0.54802537], [0, 0, 0, 255]),
            ([6.1731586, -1.3375182, 17.975266], [0.0, -1.0024999], [255, 255, 255, 255]),
            ([-7.9605117, -4.1031413, 20.101501], [0.125, -0.54802537], [0, 0, 0, 255]),
            ([6.197442, -1.1239059, 18.507511], [0.125, -1.0024999], [255, 255, 255, 255]),
            ([-7.978787, -4.0, 20.148493], [0.25, -0.54802537], [0, 0, 0, 255]),
            ([6.1060658, -0.60820055, 18.742464], [0.25, -1.0024999], [255, 255, 255, 255]),
            ([-8.009488, -3.896859, 20.108498], [0.375, -0.54802537], [0, 0, 0, 255]),
            ([5.9525585, -0.0924952, 18.542494], [0.375, -1.0024999], [255, 255, 255, 255]),
            ([-8.034632, -3.8541365, 20.004948], [0.5, -0.54802537], [0, 0, 0, 255]),
            ([5.8268414, 0.121117115, 18.02474], [0.5, -1.0024999], [255, 255, 255, 255]),
            ([-8.039489, -3.896859, 19.898499], [0.625, -0.54802537], [0, 0, 0, 255]),
            ([5.802558, -0.0924952, 17.492495], [0.625, -1.0024999], [255, 255, 255, 255]),
            ([-8.021214, -4.0, 19.851507], [0.75, -0.54802537], [0, 0, 0, 255]),
            ([5.8939342, -0.60820055, 17.25754], [0.75, -1.0024999], [255, 255, 255, 255]),
            ([-7.9905114, -4.1031413, 19.891502], [0.875, -0.54802537], [0, 0, 0, 255]),
            ([6.0474415, -1.1239059, 17.457512], [0.875, -1.0024999], [255, 255, 255, 255]),
            ([-7.9653683, -4.1458635, 19.995052], [0.0, -0.54802537], [0, 0, 0, 255]),
            ([6.1731586, -1.3375182, 17.975266], [0.0, -1.0024999], [255, 255, 255, 255]),
        ],
        mask: Mask { count: 1937, min_x: 347, max_x: 634, min_y: 267, max_y: 349, cx: 466.0767, cy: 308.8934 },
        mean: [90.4992, 90.4992, 90.4992],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [231.0, 134.117, 134.117, 134.117], [1.0, 113.000, 113.000, 113.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [381.0, 130.919, 130.919, 130.919], [408.0, 99.824, 99.824, 99.824], [19.0, 78.368, 78.368, 78.368], [0.0, 0.000, 0.000, 0.000],
            [2.0, 117.000, 117.000, 117.000], [240.0, 92.542, 92.542, 92.542], [360.0, 61.342, 61.342, 61.342], [25.0, 37.200, 37.200, 37.200],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [48.0, 45.167, 45.167, 45.167], [222.0, 20.185, 20.185, 20.185],
        ],
    },
    Ref {
        name: "crystal_below",
        camera: [0.0, 10.0],
        dragon: [[0.0, 70.0, 16.0], [0.0, 70.0, 16.0]],
        crystal: [2.0, 50.0, 16.0],
        inner_rotation: 5, ticks_existed: 37, partial: 0.5, backdrop: false,
        state: State { blend: false, alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, shade_model: 7425, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [0.0, 0.0] },
        // camera-relative position, uv, rgba; GL_TRIANGLE_STRIP order
        verts: [
            ([-0.14930017, 7.9855275, 16.0], [0.0, 0.2727738], [0, 0, 0, 255]),
            ([1.2534904, -12.704414, 16.0], [0.0, -0.375], [255, 255, 255, 255]),
            ([-0.10557116, 7.989766, 16.106066], [0.125, 0.2727738], [0, 0, 0, 255]),
            ([1.4721355, -12.68322, 16.53033], [0.125, -0.375], [255, 255, 255, 255]),
            ([-1.3113417E-8, 8.0, 16.15], [0.25, 0.2727738], [0, 0, 0, 255]),
            ([1.9999912, -12.632051, 16.75], [0.25, -0.375], [255, 255, 255, 255]),
            ([0.10557114, 8.010234, 16.106066], [0.375, 0.2727738], [0, 0, 0, 255]),
            ([2.527847, -12.580883, 16.53033], [0.375, -0.375], [255, 255, 255, 255]),
            ([0.14930017, 8.014473, 16.0], [0.5, 0.2727738], [0, 0, 0, 255]),
            ([2.7464921, -12.559689, 16.0], [0.5, -0.375], [255, 255, 255, 255]),
            ([0.10557116, 8.010234, 15.893934], [0.625, 0.2727738], [0, 0, 0, 255]),
            ([2.527847, -12.580883, 15.46967], [0.625, -0.375], [255, 255, 255, 255]),
            ([1.3113417E-8, 8.0, 15.85], [0.75, 0.2727738], [0, 0, 0, 255]),
            ([1.9999914, -12.632051, 15.25], [0.75, -0.375], [255, 255, 255, 255]),
            ([-0.10557114, 7.989766, 15.893934], [0.875, 0.2727738], [0, 0, 0, 255]),
            ([1.4721355, -12.68322, 15.46967], [0.875, -0.375], [255, 255, 255, 255]),
            ([-0.14930017, 7.9855275, 16.0], [0.0, 0.2727738], [0, 0, 0, 255]),
            ([1.2534904, -12.704414, 16.0], [0.0, -0.375], [255, 255, 255, 255]),
        ],
        mask: Mask { count: 3614, min_x: 422, max_x: 481, min_y: 1, max_y: 488, cx: 452.9925, cy: 290.2773 },
        mean: [85.8708, 85.8708, 85.8708],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [69.0, 30.261, 30.261, 30.261], [494.0, 19.846, 19.846, 19.846],
            [0.0, 0.000, 0.000, 0.000], [52.0, 66.673, 66.673, 66.673], [478.0, 54.854, 54.854, 54.854], [140.0, 46.336, 46.336, 46.336],
            [36.0, 105.778, 105.778, 105.778], [587.0, 92.249, 92.249, 92.249], [532.0, 87.182, 87.182, 87.182], [0.0, 0.000, 0.000, 0.000],
            [414.0, 133.208, 133.208, 133.208], [668.0, 127.787, 127.787, 127.787], [144.0, 120.986, 120.986, 120.986], [0.0, 0.000, 0.000, 0.000],
        ],
    },
    Ref {
        name: "along",
        camera: [0.0, 5.0],
        dragon: [[0.0, 60.0, 6.0], [0.0, 60.0, 6.0]],
        crystal: [0.5, 63.0, 30.0],
        inner_rotation: 99999, ticks_existed: 2000, partial: 0.9, backdrop: false,
        state: State { blend: false, alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, shade_model: 7425, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [0.0, 0.0] },
        // camera-relative position, uv, rgba; GL_TRIANGLE_STRIP order
        verts: [
            ([2.9531136E-4, -2.1493285, 6.014175], [0.0, -19.255463], [0, 0, 0, 255]),
            ([0.5014819, -0.46748757, 30.070871], [0.0, -20.008999], [255, 255, 255, 255]),
            ([-0.10583418, -2.105591, 6.0122323], [0.125, -19.255463], [0, 0, 0, 255]),
            ([-0.029165626, -0.248801, 30.06116], [0.125, -20.008999], [255, 255, 255, 255]),
            ([-0.14996746, -2.0, 6.003124], [0.25, -19.255463], [0, 0, 0, 255]),
            ([-0.24983197, 0.27915478, 30.01562], [0.25, -20.008999], [255, 255, 255, 255]),
            ([-0.10625181, -1.8944088, 5.992186], [0.375, -19.255463], [0, 0, 0, 255]),
            ([-0.031253815, 0.80711055, 29.960928], [0.375, -20.008999], [255, 255, 255, 255]),
            ([-2.9531136E-4, -1.8506715, 5.985825], [0.5, -19.255463], [0, 0, 0, 255]),
            ([0.49852875, 1.0257971, 29.929125], [0.5, -20.008999], [255, 255, 255, 255]),
            ([0.10583418, -1.8944088, 5.9877677], [0.625, -19.255463], [0, 0, 0, 255]),
            ([1.0291762, 0.80711055, 29.938837], [0.625, -20.008999], [255, 255, 255, 255]),
            ([0.14996746, -2.0, 5.996876], [0.75, -19.255463], [0, 0, 0, 255]),
            ([1.2498426, 0.27915478, 29.984377], [0.75, -20.008999], [255, 255, 255, 255]),
            ([0.10625181, -2.105591, 6.007814], [0.875, -19.255463], [0, 0, 0, 255]),
            ([1.0312644, -0.248801, 30.039068], [0.875, -20.008999], [255, 255, 255, 255]),
            ([2.9531136E-4, -2.1493285, 6.014175], [0.0, -19.255463], [0, 0, 0, 255]),
            ([0.5014819, -0.46748757, 30.070871], [0.0, -20.008999], [255, 255, 255, 255]),
        ],
        mask: Mask { count: 930, min_x: 464, max_x: 488, min_y: 223, max_y: 369, cx: 476.9118, cy: 298.4452 },
        mean: [36.4624, 36.4624, 36.4624],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [68.0, 96.632, 96.632, 96.632], [59.0, 94.475, 94.475, 94.475], [53.0, 98.736, 98.736, 98.736], [9.0, 79.000, 79.000, 79.000],
            [53.0, 43.547, 43.547, 43.547], [74.0, 46.257, 46.257, 46.257], [81.0, 45.111, 45.111, 45.111], [32.0, 38.781, 38.781, 38.781],
            [47.0, 17.426, 17.426, 17.426], [92.0, 13.946, 13.946, 13.946], [112.0, 14.821, 14.821, 14.821], [52.0, 14.135, 14.135, 14.135],
            [11.0, 4.727, 4.727, 4.727], [66.0, 4.455, 4.455, 4.455], [47.0, 2.085, 2.085, 2.085], [74.0, 3.432, 3.432, 3.432],
        ],
    },
    Ref {
        name: "moving",
        camera: [5.0, 0.0],
        dragon: [[-9.0, 58.0, 22.0], [-7.0, 58.5, 21.0]],
        crystal: [6.0, 62.0, 18.0],
        inner_rotation: 0, ticks_existed: 3, partial: 0.6, backdrop: false,
        state: State { blend: false, alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, shade_model: 7425, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [0.0, 0.0] },
        // camera-relative position, uv, rgba; GL_TRIANGLE_STRIP order
        verts: [
            ([-7.7711267, -3.8470228, 21.392885], [0.0, 0.41713965], [0, 0, 0, 255]),
            ([6.1443644, -1.5604663, 17.96443], [0.0, -0.036], [255, 255, 255, 255]),
            ([-7.75421, -3.8039606, 21.497955], [0.125, 0.41713965], [0, 0, 0, 255]),
            ([6.2289476, -1.3451562, 18.489779], [0.125, -0.036], [255, 255, 255, 255]),
            ([-7.764117, -3.6999998, 21.545645], [0.25, 0.41713965], [0, 0, 0, 255]),
            ([6.1794147, -0.82535195, 18.728222], [0.25, -0.036], [255, 255, 255, 255]),
            ([-7.7950435, -3.596039, 21.508017], [0.375, 0.41713965], [0, 0, 0, 255]),
            ([6.024782, -0.3055477, 18.540081], [0.375, -0.036], [255, 255, 255, 255]),
            ([-7.8288736, -3.5529768, 21.407114], [0.5, 0.41713965], [0, 0, 0, 255]),
            ([5.85563, -0.09023762, 18.035566], [0.5, -0.036], [255, 255, 255, 255]),
            ([-7.8457904, -3.596039, 21.302044], [0.625, 0.41713965], [0, 0, 0, 255]),
            ([5.7710466, -0.3055477, 17.510218], [0.625, -0.036], [255, 255, 255, 255]),
            ([-7.8358836, -3.6999998, 21.254354], [0.75, 0.41713965], [0, 0, 0, 255]),
            ([5.8205795, -0.82535195, 17.271774], [0.75, -0.036], [255, 255, 255, 255]),
            ([-7.804957, -3.8039606, 21.291983], [0.875, 0.41713965], [0, 0, 0, 255]),
            ([5.975212, -1.3451562, 17.459915], [0.875, -0.036], [255, 255, 255, 255]),
            ([-7.7711267, -3.8470228, 21.392885], [0.0, 0.41713965], [0, 0, 0, 255]),
            ([6.1443644, -1.5604663, 17.96443], [0.0, -0.036], [255, 255, 255, 255]),
        ],
        mask: Mask { count: 1914, min_x: 311, max_x: 583, min_y: 275, max_y: 336, cx: 415.2769, cy: 305.6379 },
        mean: [97.6092, 97.6092, 97.6092],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [293.0, 137.024, 137.024, 137.024], [10.0, 106.100, 106.100, 106.100], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [381.0, 133.806, 133.806, 133.806], [255.0, 104.216, 104.216, 104.216], [19.0, 79.632, 79.632, 79.632], [0.0, 0.000, 0.000, 0.000],
            [64.0, 126.375, 126.375, 126.375], [268.0, 97.052, 97.052, 97.052], [361.0, 66.986, 66.986, 66.986], [26.0, 41.115, 41.115, 41.115],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [64.0, 54.906, 54.906, 54.906], [173.0, 21.295, 21.295, 21.295],
        ],
    },
    Ref {
        name: "huge_ticks",
        camera: [-5.0, 0.0],
        dragon: [[-8.0, 58.0, 20.0], [-8.0, 58.0, 20.0]],
        crystal: [6.0, 62.0, 18.0],
        inner_rotation: 77, ticks_existed: 123456789, partial: 0.4, backdrop: false,
        state: State { blend: false, alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, shade_model: 7425, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [0.0, 0.0] },
        // camera-relative position, uv, rgba; GL_TRIANGLE_STRIP order
        verts: [
            ([-7.9672503, -4.1463065, 19.995321], [0.0, -1234567.4], [0, 0, 0, 255]),
            ([6.1637487, -1.5337753, 17.97661], [0.0, -1234567.9], [255, 255, 255, 255]),
            ([-7.9618425, -4.103454, 20.101692], [0.125, -1234567.4], [0, 0, 0, 255]),
            ([6.1907873, -1.3195143, 18.508461], [0.125, -1234567.9], [255, 255, 255, 255]),
            ([-7.978787, -4.0, 20.148493], [0.25, -1234567.4], [0, 0, 0, 255]),
            ([6.1060658, -0.802243, 18.742464], [0.25, -1234567.9], [255, 255, 255, 255]),
            ([-8.008158, -3.8965456, 20.108309], [0.375, -1234567.4], [0, 0, 0, 255]),
            ([5.9592123, -0.2849717, 18.541544], [0.375, -1234567.9], [255, 255, 255, 255]),
            ([-8.03275, -3.8536935, 20.004679], [0.5, -1234567.4], [0, 0, 0, 255]),
            ([5.8362513, -0.07071066, 18.023396], [0.5, -1234567.9], [255, 255, 255, 255]),
            ([-8.038157, -3.8965456, 19.898308], [0.625, -1234567.4], [0, 0, 0, 255]),
            ([5.8092127, -0.2849717, 17.491543], [0.625, -1234567.9], [255, 255, 255, 255]),
            ([-8.021214, -4.0, 19.851507], [0.75, -1234567.4], [0, 0, 0, 255]),
            ([5.8939342, -0.802243, 17.25754], [0.75, -1234567.9], [255, 255, 255, 255]),
            ([-7.9918423, -4.103454, 19.891691], [0.875, -1234567.4], [0, 0, 0, 255]),
            ([6.0407877, -1.3195143, 17.458462], [0.875, -1234567.9], [255, 255, 255, 255]),
            ([-7.9672503, -4.1463065, 19.995321], [0.0, -1234567.4], [0, 0, 0, 255]),
            ([6.1637487, -1.5337753, 17.97661], [0.0, -1234567.9], [255, 255, 255, 255]),
        ],
        mask: Mask { count: 1862, min_x: 386, max_x: 675, min_y: 274, max_y: 352, cx: 522.9984, cy: 316.5021 },
        mean: [79.6332, 79.6332, 79.6332],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [246.0, 131.614, 131.614, 131.614], [11.0, 105.000, 105.000, 105.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [163.0, 121.742, 121.742, 121.742], [431.0, 98.367, 98.367, 98.367], [12.0, 73.667, 73.667, 73.667], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [194.0, 85.412, 85.412, 85.412], [391.0, 62.256, 62.256, 62.256], [7.0, 37.286, 37.286, 37.286],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [88.0, 45.227, 45.227, 45.227], [319.0, 20.276, 20.276, 20.276],
        ],
    },
    Ref {
        name: "occlusion",
        camera: [0.0, 0.0],
        dragon: [[-8.0, 58.0, 20.0], [-8.0, 58.0, 20.0]],
        crystal: [6.0, 62.0, 18.0],
        inner_rotation: 1234, ticks_existed: 100, partial: 0.25, backdrop: true,
        state: State { blend: false, alpha_test: true, alpha_func: 516, alpha_ref: 0.1, depth_test: true, depth_func: 515, depth_mask: true, cull: false, lighting: false, fog: true, texture0: true, texture1: true, shade_model: 7425, wrap: [10497, 10497], filter: [9728, 9728], max_level: 0, lightmap: [0.0, 0.0] },
        // camera-relative position, uv, rgba; GL_TRIANGLE_STRIP order
        verts: [
            ([-7.9653683, -4.1458635, 19.995052], [0.0, -0.54802537], [0, 0, 0, 255]),
            ([6.1731586, -1.3375182, 17.975266], [0.0, -1.0024999], [255, 255, 255, 255]),
            ([-7.9605117, -4.1031413, 20.101501], [0.125, -0.54802537], [0, 0, 0, 255]),
            ([6.197442, -1.1239059, 18.507511], [0.125, -1.0024999], [255, 255, 255, 255]),
            ([-7.978787, -4.0, 20.148493], [0.25, -0.54802537], [0, 0, 0, 255]),
            ([6.1060658, -0.60820055, 18.742464], [0.25, -1.0024999], [255, 255, 255, 255]),
            ([-8.009488, -3.896859, 20.108498], [0.375, -0.54802537], [0, 0, 0, 255]),
            ([5.9525585, -0.0924952, 18.542494], [0.375, -1.0024999], [255, 255, 255, 255]),
            ([-8.034632, -3.8541365, 20.004948], [0.5, -0.54802537], [0, 0, 0, 255]),
            ([5.8268414, 0.121117115, 18.02474], [0.5, -1.0024999], [255, 255, 255, 255]),
            ([-8.039489, -3.896859, 19.898499], [0.625, -0.54802537], [0, 0, 0, 255]),
            ([5.802558, -0.0924952, 17.492495], [0.625, -1.0024999], [255, 255, 255, 255]),
            ([-8.021214, -4.0, 19.851507], [0.75, -0.54802537], [0, 0, 0, 255]),
            ([5.8939342, -0.60820055, 17.25754], [0.75, -1.0024999], [255, 255, 255, 255]),
            ([-7.9905114, -4.1031413, 19.891502], [0.875, -0.54802537], [0, 0, 0, 255]),
            ([6.0474415, -1.1239059, 17.457512], [0.875, -1.0024999], [255, 255, 255, 255]),
            ([-7.9653683, -4.1458635, 19.995052], [0.0, -0.54802537], [0, 0, 0, 255]),
            ([6.1731586, -1.3375182, 17.975266], [0.0, -1.0024999], [255, 255, 255, 255]),
        ],
        mask: Mask { count: 317292, min_x: 94, max_x: 865, min_y: 116, max_y: 526, cx: 480.0000, cy: 321.5000 },
        mean: [80.0641, 111.8687, 143.6734],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [19879.0, 80.000, 112.000, 144.000], [19879.0, 80.000, 112.000, 144.000], [19879.0, 80.000, 112.000, 144.000], [19879.0, 80.000, 112.000, 144.000],
            [19879.0, 80.000, 112.000, 144.000], [19879.0, 82.162, 112.239, 142.317], [19879.0, 79.834, 111.351, 142.869], [19879.0, 80.000, 112.000, 144.000],
            [19879.0, 80.000, 112.000, 144.000], [19879.0, 80.000, 112.000, 144.000], [19879.0, 79.027, 110.314, 141.601], [19879.0, 80.000, 112.000, 144.000],
            [19686.0, 80.000, 112.000, 144.000], [19686.0, 80.000, 112.000, 144.000], [19686.0, 80.000, 112.000, 144.000], [19686.0, 80.000, 112.000, 144.000],
        ],
    },
];
