//! "Arrows don't visually appear 1-1 with vanilla, look at all
//! cases" and "Z-fighting on arrows (at the least the ones stuck in the ground)".
//!
//! The reference below is vanilla, not a re-derivation of it. `crates/mc-client/javaref/
//! ArrowRenderRef.java` (`generate-arrows.ps1`) runs the unmodified recompiled vanilla 1.8.9
//! `RenderManager.renderEntityStatic` over a real `RenderArrow`, and over a real `RenderPlayer`
//! with its `LayerArrow`, in an LWJGL Pbuffer at 1920x1080. The state is
//! `EntityRenderer.renderWorldPass`'s at `renderEntities` (back faces culled, depth LEQUAL, alpha
//! 0.1, blend off, `RenderHelper.enableStandardItemLighting()` under the camera rotation, an
//! all-white lightmap), and the camera sits at the eye with `gluPerspective(90, 16/9, 0.05,
//! 128 sqrt 2)`. For every case it recorded:
//!
//! * each Tessellator draw with the live GL state (`GL_CULL_FACE`, `GL_LIGHTING`, `GL_LIGHT0`,
//!   `GL_NORMALIZE`, the current normal), and its vertices projected to window pixels (x right,
//!   y down) with their eye-space depth and UV;
//! * the real pixels, reduced to a coverage mask against a magenta clear, the mean colour, and a
//!   4x4 grid of per-cell pixel counts and mean colours over the mask bounds.
//!
//! What the reference established (`the_java_reference_*`):
//!
//! * a free arrow draws with back faces **culled** and lighting **on**, but with its normal
//!   `(0.05625, 0, 0)` under `GL_RESCALE_NORMAL` and `GL_NORMALIZE` off, so it is lit 0.40-0.455;
//! * an arrow stuck in a player (`LayerArrow`) draws **two-sided** and **unlit**;
//! * with F3+B every stuck arrow draws its own debug box and look line, and the arrows after the
//!   first draw culled under ambient-only lighting, because `renderDebugBoundingBox` ends with
//!   `enableCull()` and `enableLighting()`.
//!
//! The comparisons rebuild each case from mc-rs's own code:
//!
//! * `arrow_vertices_project_like_java` (CPU): `objects::emit_arrow` through [`Camera`]; every
//!   quad corner within 0.01 px and 1e-5 of eye depth of Java's, UVs exact;
//! * `stuck_arrow_vertices_project_like_java` (CPU): a posed [`RenderManager`] player, its seeded
//!   `stuck_arrow_attachments` and `emit_arrow_from`, against `LayerArrow`'s own draws;
//! * `arrow_pixels_match_java` (GPU): `objects::emit_arrow_lit` under `objects::ARROW_BLEND`
//!   through the real [`EntityPhase`] at 1920x1080, against Java's coverage, mean colour and
//!   colour grid. A negative control renders the state mc-rs used before (`LayerBlend::Opaque`:
//!   two-sided and shader-lit) and requires it to fail.
//!
//! Skips (loudly) without the vanilla jar (set `MC_JARS_DIR`) or a wgpu adapter. Set
//! `MC_ARROW_DUMP=<dir>` to write each mc-rs render as a PNG, and `MC_ARROW_JAVA_DIR=<dir>` (the
//! generator's PNG folder) to also print an exact per-pixel comparison with Java's PNGs. The
//! `PACK_ARROWS` cases render a resource pack's `textures/entity/arrow.png` when
//! `MC_ARROW_PACK=<pack folder>` is set (Java used a copy of a resource pack's arrow texture),
//! and are skipped otherwise.

// The tables are generated: their pixel coordinates are data, and one may land near a constant.
#![allow(clippy::approx_constant)]

use std::path::{Path, PathBuf};

use glam::{Vec3, Vec4};
use mc_assets::{FolderResourcePack, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::entity::objects;
use mc_render::entity::render::{EntityTransform, FULL_BRIGHT_COMBINED_LIGHT};
use mc_render::entity::{
    emit_arrow_from, entity_matrix, renderer_for_player, AnimParams, EntityAnimState, EntityPhase,
    EntityVertex, LayerBlend, ModelKind, RenderManager, SkinModel,
};
use mc_render::sky::lightmap::Lightmap;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 1920;
const H: u32 = 1080;
/// The user's FOV setting (`fov:0.5`), which `ArrowRenderRef` renders with.
const FOV: f32 = 90.0;

/// Java's coverage mask: every pixel that is not the magenta clear colour.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    PositionTex,
    // The player tables keep RenderArrow's draws only, so the debug-box format never appears.
    #[allow(dead_code)]
    PositionColor,
    #[allow(dead_code)]
    Other,
}

/// One Tessellator draw as it reached the GL.
struct Draw {
    format: Format,
    cull: bool,
    lighting: bool,
    light0: bool,
    normalize: bool,
    /// `GL_CURRENT_NORMAL`, in object space.
    normal: [f32; 3],
    /// Window x, window y (down), eye depth, u, v.
    verts: &'static [[f64; 5]],
}

/// The `LayerArrow` inputs of a player case.
#[derive(Debug, Clone, Copy)]
struct PlayerArgs {
    entity_id: i32,
    ticks: u32,
    arrows: i32,
    /// `EnumPlayerModelParts` mask, watched index 10.
    parts: u8,
    hitboxes: bool,
    /// The F3+B `POSITION_COLOR` draws the case made (the player's and every stuck arrow's).
    debug_draws: usize,
}

/// One vanilla render.
struct Ref {
    name: &'static str,
    /// Camera yaw and pitch, degrees.
    camera: [f32; 2],
    /// The arrow (or the player's feet) relative to the eye.
    rel: [f32; 3],
    /// `prevRotationYaw`, `rotationYaw`, `prevRotationPitch`, `rotationPitch`.
    rotation: [f32; 4],
    /// `EntityArrow.arrowShake`.
    shake: i32,
    partial: f32,
    burning: bool,
    player: Option<PlayerArgs>,
    mask: Mask,
    mean: [f64; 3],
    /// Row-major 4x4 cells over the mask bounds: `[pixels, mean r, g, b]`.
    grid: [[f64; 4]; 16],
    /// Every draw for an arrow case; `RenderArrow`'s draws only for a player case.
    draws: &'static [Draw],
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

/// `textures/entity/arrow.png` from the vanilla jar, with `pack` on top when given.
fn arrow_texture(pack: Option<&Path>) -> Option<(Vec<u8>, u32, u32)> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    if let Some(pack) = pack {
        rm.push_pack(Box::new(FolderResourcePack::open(pack).ok()?));
    }
    let (rgba, width, height, _) = rm
        .load_png_rgba(&ResourceLocation::minecraft(objects::ARROW_TEXTURE))
        .ok()?;
    Some((rgba, width, height))
}

/// `EntityRenderer.orientCamera`'s rotation at the eye, `gluPerspective(90, 16/9, 0.05, far)`.
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

/// One camera-relative point to window pixels (x right, y down) and its eye-space depth.
fn to_window(camera: &Camera, p: [f32; 3]) -> [f64; 3] {
    let point = Vec4::new(p[0], p[1], p[2], 1.0);
    let clip = camera.view_proj() * point;
    let x = (f64::from(clip.x) / f64::from(clip.w) * 0.5 + 0.5) * f64::from(W);
    let y = (1.0 - (f64::from(clip.y) / f64::from(clip.w) * 0.5 + 0.5)) * f64::from(H);
    [x, y, f64::from(-(camera.view * point).z)]
}

/// `RenderArrow.doRender`'s inputs: its plain lerp of the (already unwrapped) previous rotation,
/// and `f9 = arrowShake - partialTicks`.
fn arrow_inputs(r: &Ref) -> (Vec3, f32, f32, f32) {
    let [prev_yaw, yaw, prev_pitch, pitch] = r.rotation;
    let yaw = prev_yaw + (yaw - prev_yaw) * r.partial;
    let pitch = prev_pitch + (pitch - prev_pitch) * r.partial;
    (Vec3::from(r.rel), yaw, pitch, r.shake as f32 - r.partial)
}

/// The window positions of quad `q`'s four corners in an `emit_quad` stream (`0-1-2 / 0-2-3`).
fn quad_corners(camera: &Camera, out: &[EntityVertex], q: usize) -> [([f64; 3], [f32; 2]); 4] {
    [0usize, 1, 2, 5].map(|i| {
        let vertex = out[q * 6 + i];
        (to_window(camera, vertex.pos), vertex.uv)
    })
}

/// Compare emitted quads, in order, against Java's quad draws. Returns the worst window
/// distance and depth difference, and pushes a failure for any UV mismatch.
fn compare_quads(
    name: &str,
    camera: &Camera,
    out: &[EntityVertex],
    draws: &[&Draw],
    failures: &mut Vec<String>,
) -> (f64, f64) {
    let mut worst = (0.0_f64, 0.0_f64);
    if out.len() != draws.len() * 6 {
        failures.push(format!(
            "{name}: mc-rs emitted {} quads, Java drew {}",
            out.len() / 6,
            draws.len()
        ));
        return worst;
    }
    for (q, draw) in draws.iter().enumerate() {
        for (corner, (ours, uv)) in quad_corners(camera, out, q).iter().enumerate() {
            let java = draw.verts[corner];
            worst.0 = worst.0.max((ours[0] - java[0]).hypot(ours[1] - java[1]));
            worst.1 = worst.1.max((ours[2] - java[2]).abs());
            if [uv[0], uv[1]] != [java[3] as f32, java[4] as f32] {
                failures.push(format!(
                    "{name}: quad {q} corner {corner} uv {uv:?}, Java [{}, {}]",
                    java[3], java[4]
                ));
            }
        }
    }
    worst
}

/// The facts the Java reference itself established, kept as assertions so a regenerated table
/// that changed them cannot pass silently.
#[test]
fn the_java_reference_culls_free_arrows_and_draws_stuck_ones_two_sided_and_unlit() {
    for r in ARROWS {
        let arrow = &r.draws[..6];
        for (q, d) in arrow.iter().enumerate() {
            assert_eq!(d.format, Format::PositionTex, "{} quad {q}", r.name);
            assert!(
                d.cull,
                "{} quad {q}: RenderArrow keeps GL_CULL_FACE",
                r.name
            );
            assert!(
                d.lighting && d.light0,
                "{} quad {q}: item lights on",
                r.name
            );
            assert!(!d.normalize, "{} quad {q}: GL_NORMALIZE off", r.name);
            let length = Vec3::from(d.normal).length();
            assert!(
                (length - objects::ARROW_SCALE).abs() < 1e-6,
                "{} quad {q}",
                r.name
            );
        }
        if r.burning {
            // `Render.renderEntityOnFire`: one draw of two layers, unlit, culling unchanged.
            assert_eq!(r.draws.len(), 7, "{}", r.name);
            let fire = &r.draws[6];
            assert!(fire.cull && !fire.lighting, "{}", r.name);
            assert_eq!(fire.verts.len(), 8, "{}", r.name);
        } else {
            assert_eq!(r.draws.len(), 6, "{}", r.name);
        }
    }
    for r in PLAYERS {
        let p = r.player.expect("a player case");
        assert_eq!(r.draws.len(), 6 * p.arrows as usize, "{}", r.name);
        for (i, d) in r.draws.iter().enumerate() {
            if p.hitboxes && i >= 6 {
                // `renderDebugBoundingBox` ended with `enableCull()` and `enableLighting()`; the
                // item lights LayerArrow switched off stay off, so this is ambient light only.
                assert!(d.cull && d.lighting && !d.light0, "{} draw {i}", r.name);
            } else {
                // `RendererLivingEntity.doRender:92` disableCull, LayerArrow's
                // `RenderHelper.disableStandardItemLighting()`.
                assert!(!d.cull && !d.lighting, "{} draw {i}", r.name);
            }
        }
        let expected_debug = if p.hitboxes {
            // The player's box, eye box and look line (3 + 3 + 1), then a box and a look line
            // (3 + 1) for every stuck arrow: `renderEntityWithPosYaw` never hides them.
            7 + 4 * p.arrows as usize
        } else {
            0
        };
        assert_eq!(p.debug_draws, expected_debug, "{}", r.name);
    }
}

#[test]
fn arrow_vertices_project_like_java() {
    let mut failures = Vec::new();
    for r in ARROWS {
        let camera = camera(r);
        let (rel, yaw, pitch, shake) = arrow_inputs(r);
        let mut out = Vec::new();
        objects::emit_arrow(rel, yaw, pitch, shake, &mut out);
        let draws: Vec<&Draw> = r.draws.iter().take(6).collect();
        let (distance, depth) = compare_quads(r.name, &camera, &out, &draws, &mut failures);
        eprintln!(
            "{:18} worst window distance {distance:.5} px, depth {depth:.2e}",
            r.name
        );
        if distance > 0.01 || depth > 1e-5 {
            failures.push(format!(
                "{}: worst vertex distance {distance:.4} px, depth {depth:.2e}",
                r.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "mc-rs's arrow quads are not where vanilla 1.8.9 draws them: {failures:#?}"
    );
}

#[test]
fn stuck_arrow_vertices_project_like_java() {
    let mut failures = Vec::new();
    for r in PLAYERS {
        let p = r.player.expect("a player case");
        let camera = camera(r);
        // `UUID(0, 0)` hashes to 0, so `DefaultPlayerSkin` gave the Java player Steve's classic
        // arms.
        let renderer = renderer_for_player(SkinModel::Classic);
        let mut models = RenderManager::new();
        models.set_player_model_parts(renderer.model, p.parts as i8);
        let params = AnimParams {
            age_in_ticks: p.ticks as f32 + r.partial,
            partial_ticks: r.partial,
            entity: EntityAnimState {
                entity_id: p.entity_id,
                arrow_count: p.arrows,
                ticks_existed: p.ticks,
                ..EntityAnimState::default()
            },
            ..AnimParams::still()
        };
        models.pose_with(renderer.model, &params);
        let base = entity_matrix(&EntityTransform {
            rel: Vec3::from(r.rel),
            body_yaw: r.rotation[1],
            scale: renderer.scale,
            ..EntityTransform::default()
        });
        let mut out = Vec::new();
        for attachment in models.stuck_arrow_attachments(
            renderer.model,
            p.entity_id,
            p.arrows,
            params.scale,
            base,
        ) {
            // LayerArrow's temporary EntityArrow has `arrowShake = 0`.
            emit_arrow_from(
                attachment.matrix,
                attachment.yaw,
                attachment.pitch,
                -r.partial,
                &mut out,
            );
        }
        assert_eq!(renderer.model, ModelKind::Player);
        let draws: Vec<&Draw> = r.draws.iter().collect();
        let (distance, depth) = compare_quads(r.name, &camera, &out, &draws, &mut failures);
        eprintln!(
            "{:18} {} stuck arrows: worst window distance {distance:.5} px, depth {depth:.2e}",
            r.name, p.arrows
        );
        if distance > 0.01 || depth > 1e-5 {
            failures.push(format!(
                "{}: worst vertex distance {distance:.4} px, depth {depth:.2e}",
                r.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "mc-rs's LayerArrow arrows are not where vanilla 1.8.9 draws them: {failures:#?}"
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

/// How one case is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Draws {
    /// The fixed state: `ARROW_BLEND` (culled, unlit pipeline) with `arrow_light` in the colour.
    Fixed,
    /// mc-rs before W6: `LayerBlend::Opaque` (two-sided, the shader's unit-normal item light).
    Old,
}

/// Render one arrow case through the real `EntityPhase` and return its RGBA pixels.
fn render_case(gpu: &mut Gpu, texture: &(Vec<u8>, u32, u32), r: &Ref, mode: Draws) -> Vec<u8> {
    let camera = camera(r);
    let mut phase = EntityPhase::new(gpu);
    phase.upload_texture(
        gpu,
        objects::ARROW_TEXTURE,
        &texture.0,
        texture.1,
        texture.2,
    );
    // An all-white lightmap, as the Java reference: the lightmap stage is the identity.
    phase.set_lightmap(gpu, &Lightmap::full_bright());
    phase.begin();
    let (rel, yaw, pitch, shake) = arrow_inputs(r);
    let mut verts = Vec::new();
    let blend = match mode {
        Draws::Fixed => {
            objects::emit_arrow_lit(rel, yaw, pitch, shake, &mut verts);
            objects::ARROW_BLEND
        }
        Draws::Old => {
            objects::emit_arrow(rel, yaw, pitch, shake, &mut verts);
            LayerBlend::Opaque
        }
    };
    phase.push_blended_lit(
        objects::ARROW_TEXTURE,
        blend,
        FULL_BRIGHT_COMBINED_LIGHT,
        &verts,
    );
    phase.finish(gpu);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &camera, r.partial);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("readback")
}

fn is_clear(pixel: &[u8]) -> bool {
    pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 255
}

/// Coverage statistics and the mean colour, as `ArrowRenderRef` computes them.
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
    // GL and D3D12/Vulkan agree on pixel centres and the top-left rule; what is left is edge
    // pixels whose centre sits within float noise of an edge.
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
    if (ours.cx - java.cx).abs() > 0.25 || (ours.cy - java.cy).abs() > 0.25 {
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
        if java[0] < 100.0 {
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
    let dir = PathBuf::from(std::env::var_os("MC_ARROW_JAVA_DIR")?);
    let java = image::open(dir.join(format!("{}.png", r.name)))
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
    texture: &(Vec<u8>, u32, u32),
    refs: &[Ref],
    mode: Draws,
    dump: Option<&Path>,
) -> Vec<String> {
    let mut failures = Vec::new();
    for r in refs {
        // The fire stack is mc-client's (`appearance::emit_fire_overlay`); its Java vertices are
        // checked there. Here only the arrow is drawn, so the burning case has nothing to match.
        if r.burning {
            continue;
        }
        let pixels = render_case(gpu, texture, r, mode);
        if let Some(dir) = dump {
            std::fs::create_dir_all(dir).expect("dump dir");
            let suffix = if mode == Draws::Old { "_old" } else { "" };
            image::RgbaImage::from_raw(W, H, pixels.clone())
                .expect("rgba frame")
                .save(dir.join(format!("{}{suffix}.png", r.name)))
                .expect("write png");
        }
        let (ours, mean) = stats_of(&pixels);
        eprintln!(
            "{mode:?} {:24} n {:6} vs {:6}  c ({:8.3}, {:7.3}) vs ({:8.3}, {:7.3})  mean \
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
fn arrow_pixels_match_java() {
    let Some(texture) = arrow_texture(None) else {
        eprintln!("SKIP: no vanilla jar; set MC_JARS_DIR");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    let dump = std::env::var_os("MC_ARROW_DUMP").map(PathBuf::from);

    let failures = check_cases(&mut gpu, &texture, ARROWS, Draws::Fixed, dump.as_deref());
    assert!(
        failures.is_empty(),
        "mc-rs's arrows do not render like vanilla 1.8.9 at 1920x1080:\n{}",
        failures.join("\n")
    );

    // Negative control: the state mc-rs drew arrows with before. Two-sided, the coplanar pairs
    // z-fight; shader-lit with unit normals, the upper faces come out up to twice as bright.
    let old = check_cases(&mut gpu, &texture, ARROWS, Draws::Old, dump.as_deref());
    let drawn = ARROWS.iter().filter(|r| !r.burning).count();
    assert_eq!(
        old.len(),
        drawn,
        "every case must tell the old two-sided, shader-lit arrow apart from vanilla:\n{}",
        old.join("\n")
    );

    // The reference pack, when one is supplied (read only).
    let Some(pack) = std::env::var_os("MC_ARROW_PACK").map(PathBuf::from) else {
        eprintln!("SKIP: PACK_ARROWS; set MC_ARROW_PACK to a pack folder with an arrow texture");
        return;
    };
    let texture = arrow_texture(Some(&pack)).expect("the pack's textures/entity/arrow.png");
    let failures = check_cases(
        &mut gpu,
        &texture,
        PACK_ARROWS,
        Draws::Fixed,
        dump.as_deref(),
    );
    assert!(
        failures.is_empty(),
        "mc-rs's arrows do not render like vanilla 1.8.9 with the pack texture:\n{}",
        failures.join("\n")
    );
}

// -------------------------------------------------------------------------------------------
// Generated by ArrowRenderRef on this machine:
// GL_VENDOR=NVIDIA Corporation GL_RENDERER=NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2 GL_VERSION=4.6.0 NVIDIA 610.88
// -------------------------------------------------------------------------------------------

#[rustfmt::skip]
const ARROWS: &[Ref] = &[
    Ref {
        name: "side_level",
        camera: [0.0, 0.0],
        rel: [0.0, 0.0, 1.5],
        rotation: [90.0, 90.0, 0.0, 0.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 8512, min_x: 878, max_x: 1215, min_y: 504, max_y: 575, cx: 1079.6438, cy: 540.0000 },
        mean: [84.0188, 79.0600, 70.7902],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [916.0, 94.816, 94.816, 94.816],
            [1011.0, 78.473, 76.918, 73.915], [504.0, 71.476, 57.673, 35.673], [663.0, 76.668, 66.302, 49.379], [1162.0, 82.559, 78.688, 71.497],
            [1011.0, 74.786, 73.077, 69.801], [504.0, 78.411, 62.673, 38.673], [663.0, 81.932, 70.347, 51.885], [1162.0, 89.885, 85.608, 77.825],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [916.0, 100.762, 100.762, 100.762],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1209.1794, 540.0000, 1.340901, 0.0, 0.15625],
                    [1182.7499, 597.2756, 1.500000, 0.15625, 0.15625],
                    [1161.3894, 540.0000, 1.659099, 0.15625, 0.3125],
                    [1182.7499, 482.7244, 1.500000, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1182.7499, 482.7244, 1.500000, 0.0, 0.15625],
                    [1161.3894, 540.0000, 1.659099, 0.15625, 0.15625],
                    [1182.7499, 597.2756, 1.500000, 0.15625, 0.3125],
                    [1209.1794, 540.0000, 1.340901, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1216.6087, 509.7584, 1.420450, 0.0, 0.0],
                    [874.4637, 509.7584, 1.420450, 0.5, 0.0],
                    [883.0793, 567.1956, 1.579549, 0.5, 0.15625],
                    [1190.7620, 567.1956, 1.579549, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1190.7620, 512.8044, 1.579549, 0.0, 0.0],
                    [883.0793, 512.8044, 1.579549, 0.5, 0.0],
                    [874.4637, 570.2416, 1.420450, 0.5, 0.15625],
                    [1216.6087, 570.2416, 1.420450, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1190.7620, 567.1956, 1.579549, 0.0, 0.0],
                    [883.0793, 567.1956, 1.579549, 0.5, 0.0],
                    [874.4637, 509.7584, 1.420450, 0.5, 0.15625],
                    [1216.6087, 509.7584, 1.420450, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1216.6087, 570.2416, 1.420450, 0.0, 0.0],
                    [874.4637, 570.2416, 1.420450, 0.5, 0.0],
                    [883.0793, 512.8045, 1.579549, 0.5, 0.15625],
                    [1190.7620, 512.8045, 1.579549, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "side_from_above",
        camera: [0.0, 30.0],
        rel: [0.0, -0.75, 1.25],
        rotation: [90.0, 90.0, 0.0, 0.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 11104, min_x: 876, max_x: 1229, min_y: 507, max_y: 591, cx: 1078.4271, cy: 549.3049 },
        mean: [87.9678, 83.0112, 74.9236],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [164.0, 92.171, 92.171, 92.171], [0.0, 0.000, 0.000, 0.000], [68.0, 99.000, 99.000, 99.000], [1181.0, 103.210, 103.210, 103.210],
            [1286.0, 84.733, 82.457, 78.171], [792.0, 76.467, 61.333, 37.778], [998.0, 81.833, 70.533, 52.613], [1403.0, 93.382, 91.790, 88.892],
            [1208.0, 74.935, 73.011, 69.395], [616.0, 78.050, 62.472, 38.472], [821.0, 83.487, 71.907, 53.695], [1218.0, 89.625, 85.364, 77.497],
            [126.0, 69.000, 69.000, 69.000], [0.0, 0.000, 0.000, 0.000], [39.0, 99.000, 99.000, 99.000], [1184.0, 103.026, 103.026, 103.026],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1213.1733, 582.5816, 1.319748, 0.0, 0.15625],
                    [1177.3762, 597.0194, 1.537081, 0.15625, 0.15625],
                    [1169.4413, 521.3727, 1.595315, 0.15625, 0.3125],
                    [1202.4741, 495.6141, 1.377982, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1202.4741, 495.6141, 1.377982, 0.0, 0.15625],
                    [1169.4413, 521.3727, 1.595315, 0.15625, 0.15625],
                    [1177.3762, 597.0194, 1.537081, 0.15625, 0.3125],
                    [1213.1733, 582.5816, 1.319748, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1230.2271, 538.1592, 1.348865, 0.0, 0.0],
                    [869.9242, 538.1592, 1.348865, 0.5, 0.0],
                    [882.4236, 558.4929, 1.566198, 0.5, 0.15625],
                    [1192.7291, 558.4929, 1.566198, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1205.1823, 509.4348, 1.486649, 0.0, 0.0],
                    [878.2725, 509.4348, 1.486649, 0.5, 0.0],
                    [874.9406, 590.3497, 1.428414, 0.5, 0.15625],
                    [1215.1780, 590.3497, 1.428414, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1192.7291, 558.4929, 1.566198, 0.0, 0.0],
                    [882.4236, 558.4929, 1.566198, 0.5, 0.0],
                    [869.9242, 538.1592, 1.348865, 0.5, 0.15625],
                    [1230.2271, 538.1592, 1.348865, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1215.1780, 590.3497, 1.428414, 0.0, 0.0],
                    [874.9406, 590.3497, 1.428414, 0.5, 0.0],
                    [878.2725, 509.4348, 1.486649, 0.5, 0.15625],
                    [1205.1823, 509.4348, 1.486649, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "oblique_flight",
        camera: [20.0, -10.0],
        rel: [0.375, 0.25, 1.625],
        rotation: [60.0, 60.0, 25.0, 25.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 11347, min_x: 544, max_x: 855, min_y: 494, max_y: 684, cx: 725.2051, cy: 598.5971 },
        mean: [85.9507, 81.3082, 73.4274],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [2325.0, 77.343, 75.602, 72.290], [59.0, 75.458, 60.881, 37.729], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [130.0, 72.185, 63.231, 48.831], [1282.0, 74.225, 60.202, 36.894], [244.0, 77.996, 63.586, 41.590], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [4.0, 82.000, 66.000, 43.000], [2101.0, 85.040, 77.787, 66.208], [2388.0, 91.732, 89.938, 86.562],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [894.0, 104.515, 104.515, 104.515], [1920.0, 91.639, 88.760, 83.160],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [775.9510, 640.8178, 1.125296, 0.0, 0.15625],
                    [779.9545, 702.6450, 1.266058, 0.15625, 0.15625],
                    [836.3706, 639.6123, 1.433900, 0.15625, 0.3125],
                    [839.0277, 578.9487, 1.293138, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [839.0277, 578.9487, 1.293138, 0.0, 0.15625],
                    [836.3706, 639.6123, 1.433900, 0.15625, 0.15625],
                    [779.9545, 702.6450, 1.266058, 0.15625, 0.3125],
                    [775.9510, 640.8178, 1.125296, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [830.7250, 618.3368, 1.196371, 0.0, 0.0],
                    [540.2616, 472.0423, 1.401909, 0.5, 0.0],
                    [565.0528, 538.1833, 1.542670, 0.5, 0.15625],
                    [828.7496, 679.2442, 1.337133, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [856.5388, 620.2652, 1.350673, 0.0, 0.0],
                    [591.4662, 488.2214, 1.556210, 0.5, 0.0],
                    [510.4133, 527.3992, 1.388368, 0.5, 0.15625],
                    [799.0151, 684.9876, 1.182831, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [828.7496, 679.2442, 1.337132, 0.0, 0.0],
                    [565.0528, 538.1833, 1.542670, 0.5, 0.0],
                    [540.2617, 472.0423, 1.401909, 0.5, 0.15625],
                    [830.7250, 618.3368, 1.196371, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [799.0151, 684.9876, 1.182831, 0.0, 0.0],
                    [510.4133, 527.3992, 1.388368, 0.5, 0.0],
                    [591.4662, 488.2214, 1.556210, 0.5, 0.15625],
                    [856.5388, 620.2652, 1.350673, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "toward_camera",
        camera: [0.0, 0.0],
        rel: [0.125, 0.125, 1.25],
        rotation: [180.0, 180.0, 10.0, 10.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 3729, min_x: 873, max_x: 950, min_y: 441, max_y: 561, cx: 912.1661, cy: 502.8138 },
        mean: [82.1432, 79.0585, 73.9289],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [207.0, 75.937, 75.937, 75.937], [395.0, 83.377, 83.377, 83.377], [248.0, 83.290, 83.290, 83.290], [0.0, 0.000, 0.000, 0.000],
            [326.0, 63.387, 63.387, 63.387], [470.0, 69.274, 62.264, 50.234], [189.0, 78.921, 73.513, 65.016], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [276.0, 87.830, 83.145, 75.493], [450.0, 75.396, 65.138, 48.264], [292.0, 97.411, 97.411, 97.411],
            [0.0, 0.000, 0.000, 0.000], [248.0, 95.000, 95.000, 95.000], [342.0, 90.310, 86.620, 80.363], [286.0, 97.220, 97.168, 97.087],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [969.9031, 534.9016, 1.859350, 0.0, 0.15625],
                    [923.1494, 581.0152, 1.831722, 0.15625, 0.15625],
                    [877.4907, 534.9016, 1.859349, 0.15625, 0.3125],
                    [924.2284, 490.1382, 1.886977, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [924.2284, 490.1382, 1.886977, 0.0, 0.15625],
                    [877.4907, 534.9016, 1.859349, 0.15625, 0.15625],
                    [923.1494, 581.0152, 1.831722, 0.15625, 0.3125],
                    [969.9031, 534.9016, 1.859350, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [947.2737, 515.8839, 1.928559, 0.0, 0.0],
                    [936.4512, 414.4019, 1.042232, 0.5, 0.0],
                    [851.1332, 494.3722, 1.014605, 0.5, 0.15625],
                    [901.8933, 560.0422, 1.900931, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [902.7257, 515.8839, 1.928559, 0.0, 0.0],
                    [854.0190, 414.4019, 1.042232, 0.5, 0.0],
                    [935.8100, 494.3722, 1.014605, 0.5, 0.15625],
                    [947.0888, 560.0422, 1.900931, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [901.8933, 560.0422, 1.900931, 0.0, 0.0],
                    [851.1332, 494.3722, 1.014605, 0.5, 0.0],
                    [936.4512, 414.4019, 1.042232, 0.5, 0.15625],
                    [947.2737, 515.8839, 1.928559, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [947.0888, 560.0422, 1.900931, 0.0, 0.0],
                    [935.8100, 494.3722, 1.014605, 0.5, 0.0],
                    [854.0190, 414.4019, 1.042232, 0.5, 0.15625],
                    [902.7257, 515.8839, 1.928559, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "away_from_camera",
        camera: [0.0, 0.0],
        rel: [-0.125, -0.125, 1.25],
        rotation: [-10.0, -10.0, -8.0, -8.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 12818, min_x: 895, max_x: 1055, min_y: 494, max_y: 655, cx: 977.8819, cy: 574.3254 },
        mean: [96.4057, 94.7967, 92.1634],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [954.0, 94.461, 94.461, 94.461], [435.0, 99.986, 99.986, 99.986], [406.0, 100.404, 100.404, 100.404], [1133.0, 92.920, 92.920, 92.920],
            [248.0, 100.387, 100.387, 100.387], [1416.0, 95.802, 91.649, 84.756], [1416.0, 96.167, 92.211, 85.703], [300.0, 100.760, 100.760, 100.760],
            [477.0, 97.925, 97.925, 97.925], [1424.0, 98.294, 95.249, 90.214], [1487.0, 99.348, 96.334, 91.572], [505.0, 92.570, 91.935, 90.889],
            [979.0, 92.962, 92.962, 92.962], [231.0, 98.519, 98.519, 98.519], [334.0, 98.392, 98.392, 98.392], [1073.0, 94.438, 94.438, 94.438],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [849.4052, 571.1459, 0.674208, 0.0, 0.15625],
                    [972.7536, 709.7832, 0.624774, 0.15625, 0.15625],
                    [1112.9238, 573.9263, 0.618953, 0.15625, 0.3125],
                    [978.1343, 444.1294, 0.668386, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [978.1343, 444.1294, 0.668386, 0.0, 0.15625],
                    [1112.9238, 573.9263, 0.618953, 0.15625, 0.15625],
                    [972.7536, 709.7832, 0.624774, 0.15625, 0.3125],
                    [849.4052, 571.1459, 0.674208, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [900.8786, 498.1999, 0.616441, 0.0, 0.0],
                    [991.5411, 568.0233, 1.494142, 0.5, 0.0],
                    [1049.7474, 627.8712, 1.444709, 0.5, 0.15625],
                    [1041.2814, 644.6020, 0.567007, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1041.7975, 496.2387, 0.588813, 0.0, 0.0],
                    [1049.8287, 568.5513, 1.466514, 0.5, 0.0],
                    [990.5980, 626.2223, 1.472336, 0.5, 0.15625],
                    [895.2189, 639.7420, 0.594635, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1041.2814, 644.6020, 0.567007, 0.0, 0.0],
                    [1049.7474, 627.8712, 1.444709, 0.5, 0.0],
                    [991.5411, 568.0233, 1.494142, 0.5, 0.15625],
                    [900.8787, 498.1999, 0.616441, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [895.2189, 639.7420, 0.594635, 0.0, 0.0],
                    [990.5980, 626.2223, 1.472336, 0.5, 0.0],
                    [1049.8287, 568.5513, 1.466514, 0.5, 0.15625],
                    [1041.7975, 496.2387, 0.588813, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "stuck_ground",
        camera: [15.0, 45.0],
        rel: [0.25, -1.125, 1.25],
        rotation: [200.0, 200.0, -55.0, -55.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 9505, min_x: 630, max_x: 810, min_y: 288, max_y: 605, cx: 714.1779, cy: 413.5146 },
        mean: [86.4054, 81.5220, 73.1806],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [1953.0, 94.420, 92.388, 88.037], [2548.0, 92.780, 90.704, 86.588], [15.0, 97.000, 97.000, 97.000], [0.0, 0.000, 0.000, 0.000],
            [167.0, 98.437, 98.437, 98.437], [1450.0, 84.680, 75.789, 62.179], [263.0, 78.719, 63.578, 40.479], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1073.0, 74.555, 60.457, 37.231], [48.0, 77.000, 62.000, 39.000],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [102.0, 78.922, 71.029, 58.578], [1886.0, 78.128, 76.204, 72.388],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [732.5389, 311.5521, 1.515206, 0.0, 0.15625],
                    [673.7145, 340.5777, 1.590695, 0.15625, 0.15625],
                    [609.8197, 340.5585, 1.386151, 0.15625, 0.3125],
                    [674.1442, 307.0020, 1.310663, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [674.1442, 307.0020, 1.310663, 0.0, 0.15625],
                    [609.8197, 340.5585, 1.386151, 0.15625, 0.15625],
                    [673.7145, 340.5777, 1.590695, 0.15625, 0.3125],
                    [732.5389, 311.5521, 1.515206, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [695.7843, 287.3632, 1.399041, 0.0, 0.0],
                    [830.6257, 595.2137, 1.621338, 0.5, 0.0],
                    [771.1169, 609.8042, 1.696826, 0.5, 0.15625],
                    [634.2073, 319.9138, 1.474530, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [630.2733, 301.2579, 1.334514, 0.0, 0.0],
                    [780.0580, 619.8843, 1.556810, 0.5, 0.0],
                    [817.9924, 587.4640, 1.761354, 0.5, 0.15625],
                    [693.5935, 306.5009, 1.539057, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [634.2073, 319.9138, 1.474530, 0.0, 0.0],
                    [771.1169, 609.8042, 1.696826, 0.5, 0.0],
                    [830.6257, 595.2137, 1.621338, 0.5, 0.15625],
                    [695.7843, 287.3632, 1.399041, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [693.5935, 306.5009, 1.539057, 0.0, 0.0],
                    [817.9924, 587.4640, 1.761354, 0.5, 0.0],
                    [780.0580, 619.8843, 1.556810, 0.5, 0.15625],
                    [630.2733, 301.2579, 1.334514, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "stuck_wall",
        camera: [-30.0, 5.0],
        rel: [-0.5, 0.0, 1.5],
        rotation: [130.0, 130.0, -5.0, -5.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 17516, min_x: 1467, max_x: 1893, min_y: 418, max_y: 526, cx: 1700.1165, cy: 478.8201 },
        mean: [87.7605, 84.0473, 77.9897],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [899.0, 101.930, 101.930, 101.930], [1312.0, 95.360, 95.360, 95.360],
            [60.0, 80.517, 80.517, 80.517], [21.0, 78.429, 74.143, 67.286], [1916.0, 89.317, 84.785, 77.490], [1500.0, 91.635, 87.884, 81.435],
            [2466.0, 83.076, 81.512, 78.480], [1940.0, 74.893, 60.289, 37.101], [2377.0, 86.562, 79.581, 68.540], [1955.0, 101.681, 101.178, 100.208],
            [2119.0, 74.181, 73.875, 73.263], [31.0, 71.290, 63.935, 50.774], [357.0, 94.958, 94.958, 94.958], [563.0, 101.959, 101.959, 101.959],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1950.5675, 463.2552, 0.990889, 0.0, 0.15625],
                    [1797.3527, 541.4855, 1.163187, 0.15625, 0.15625],
                    [1690.3614, 470.3227, 1.303060, 0.15625, 0.3125],
                    [1808.3212, 390.9260, 1.130762, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1808.3212, 390.9260, 1.130762, 0.0, 0.15625],
                    [1690.3614, 470.3227, 1.303060, 0.15625, 0.15625],
                    [1797.3527, 541.4855, 1.163187, 0.15625, 0.3125],
                    [1950.5675, 463.2552, 0.990889, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1894.6820, 422.8123, 1.070092, 0.0, 0.0],
                    [1527.7827, 457.6881, 0.921832, 0.5, 0.0],
                    [1431.4756, 541.7326, 1.094130, 0.5, 0.15625],
                    [1758.9852, 501.6641, 1.242390, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1763.5353, 431.7158, 1.226178, 0.0, 0.0],
                    [1431.7257, 462.7661, 1.077918, 0.5, 0.0],
                    [1525.8307, 549.8820, 0.938045, 0.5, 0.15625],
                    [1887.5209, 502.9440, 1.086304, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1758.9852, 501.6641, 1.242390, 0.0, 0.0],
                    [1431.4756, 541.7326, 1.094130, 0.5, 0.0],
                    [1527.7826, 457.6881, 0.921832, 0.5, 0.15625],
                    [1894.6820, 422.8123, 1.070092, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1887.5209, 502.9440, 1.086304, 0.0, 0.0],
                    [1525.8307, 549.8820, 0.938045, 0.5, 0.0],
                    [1431.7257, 462.7661, 1.077918, 0.5, 0.15625],
                    [1763.5353, 431.7158, 1.226178, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "from_below",
        camera: [0.0, -50.0],
        rel: [0.0, 1.0, 0.875],
        rotation: [45.0, 45.0, 12.0, 12.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 18960, min_x: 897, max_x: 1264, min_y: 343, max_y: 598, cx: 1121.9995, cy: 462.4163 },
        mean: [83.6393, 79.5284, 72.5203],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1459.0, 95.502, 95.502, 95.502], [1938.0, 87.892, 86.111, 82.567],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [3994.0, 85.103, 81.265, 74.943], [4183.0, 87.580, 85.104, 80.241],
            [100.0, 72.000, 58.000, 36.000], [2017.0, 70.891, 57.593, 35.593], [1413.0, 83.282, 77.377, 68.098], [972.0, 94.753, 94.753, 94.753],
            [2774.0, 72.817, 68.960, 62.362], [110.0, 72.000, 58.000, 36.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1153.0228, 354.0975, 0.882534, 0.0, 0.15625],
                    [1216.8197, 476.7144, 0.850669, 0.15625, 0.15625],
                    [1244.1318, 470.8863, 1.027161, 0.15625, 0.3125],
                    [1190.1452, 368.8795, 1.059026, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1190.1452, 368.8795, 1.059026, 0.0, 0.15625],
                    [1244.1318, 470.8863, 1.027161, 0.15625, 0.15625],
                    [1216.8197, 476.7144, 0.850669, 0.15625, 0.3125],
                    [1153.0228, 354.0975, 0.882534, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1203.4301, 342.8662, 0.936813, 0.0, 0.0],
                    [886.9763, 545.3187, 1.480284, 0.5, 0.0],
                    [918.5918, 621.5395, 1.448419, 0.5, 0.15625],
                    [1265.1754, 457.7329, 0.904948, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1246.1865, 403.1090, 1.009127, 0.0, 0.0],
                    [929.5054, 575.0448, 1.552598, 0.5, 0.0],
                    [872.2695, 592.0063, 1.376105, 0.5, 0.15625],
                    [1218.7184, 394.6966, 0.832634, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1265.1754, 457.7329, 0.904948, 0.0, 0.0],
                    [918.5918, 621.5395, 1.448419, 0.5, 0.0],
                    [886.9763, 545.3188, 1.480284, 0.5, 0.15625],
                    [1203.4301, 342.8662, 0.936813, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1218.7184, 394.6966, 0.832634, 0.0, 0.0],
                    [872.2695, 592.0063, 1.376105, 0.5, 0.0],
                    [929.5054, 575.0448, 1.552598, 0.5, 0.15625],
                    [1246.1865, 403.1090, 1.009127, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "interpolated",
        camera: [0.0, 0.0],
        rel: [0.0, 0.0, 1.5],
        rotation: [30.0, 80.0, -10.0, 15.0],
        shake: 0, partial: 0.35, burning: false,
        player: None,
        mask: Mask { count: 12017, min_x: 902, max_x: 1230, min_y: 484, max_y: 582, cx: 1112.5939, cy: 535.0645 },
        mean: [86.6928, 82.4788, 75.3340],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [446.0, 91.318, 91.318, 91.318], [1356.0, 95.599, 95.599, 95.599],
            [432.0, 84.199, 83.843, 83.282], [208.0, 70.913, 57.721, 35.721], [1269.0, 84.205, 79.538, 71.977], [1450.0, 82.717, 78.103, 69.132],
            [1447.0, 73.272, 68.702, 61.071], [859.0, 75.070, 60.809, 37.589], [1460.0, 88.993, 81.957, 71.087], [1446.0, 88.944, 84.799, 76.790],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [424.0, 99.767, 99.767, 99.767], [1220.0, 100.127, 100.127, 100.127],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1155.1141, 532.4450, 0.964778, 0.0, 0.15625],
                    [1189.3769, 612.7996, 1.079733, 0.15625, 0.15625],
                    [1213.7368, 533.9228, 1.199378, 0.15625, 0.3125],
                    [1185.8365, 454.0723, 1.084423, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1185.8365, 454.0723, 1.084423, 0.0, 0.15625],
                    [1213.7368, 533.9228, 1.199378, 0.15625, 0.15625],
                    [1189.3769, 612.7996, 1.079733, 0.15625, 0.3125],
                    [1155.1141, 532.4450, 0.964778, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1202.2052, 488.4111, 0.986607, 0.0, 0.0],
                    [885.1988, 514.7280, 1.594494, 0.5, 0.0],
                    [924.9911, 566.6735, 1.709449, 0.5, 0.15625],
                    [1230.8748, 571.7685, 1.101563, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1229.0477, 493.8929, 1.103908, 0.0, 0.0],
                    [924.2318, 516.4598, 1.711794, 0.5, 0.0],
                    [885.9566, 568.6387, 1.592149, 0.5, 0.15625],
                    [1204.1861, 575.5545, 0.984263, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1230.8748, 571.7685, 1.101563, 0.0, 0.0],
                    [924.9911, 566.6735, 1.709449, 0.5, 0.0],
                    [885.1989, 514.7280, 1.594494, 0.5, 0.15625],
                    [1202.2053, 488.4111, 0.986607, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1204.1861, 575.5545, 0.984263, 0.0, 0.0],
                    [885.9566, 568.6386, 1.592149, 0.5, 0.0],
                    [924.2318, 516.4598, 1.711794, 0.5, 0.15625],
                    [1229.0477, 493.8929, 1.103908, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "shake_7",
        camera: [10.0, 20.0],
        rel: [0.125, -0.5, 1.375],
        rotation: [100.0, 100.0, -30.0, -30.0],
        shake: 7, partial: 0.3, burning: false,
        player: None,
        mask: Mask { count: 10844, min_x: 752, max_x: 1042, min_y: 358, max_y: 607, cx: 912.1328, cy: 470.0114 },
        mean: [85.6094, 80.5717, 72.4088],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1011.0, 99.209, 99.024, 98.690], [2796.0, 90.460, 87.870, 82.862],
            [0.0, 0.000, 0.000, 0.000], [76.0, 67.487, 55.789, 32.395], [1622.0, 80.352, 68.415, 50.192], [1089.0, 101.758, 101.322, 100.579],
            [120.0, 74.425, 63.592, 45.375], [1365.0, 75.692, 60.537, 37.721], [46.0, 69.000, 57.000, 33.000], [0.0, 0.000, 0.000, 0.000],
            [2685.0, 78.519, 77.247, 74.710], [34.0, 76.412, 61.118, 37.941], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1025.9136, 401.5553, 1.317260, 0.0, 0.15625],
                    [1028.7407, 441.4791, 1.531883, 0.15625, 0.15625],
                    [977.5550, 391.3413, 1.598236, 0.15625, 0.3125],
                    [966.9237, 345.5548, 1.383612, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [966.9237, 345.5548, 1.383612, 0.0, 0.15625],
                    [977.5550, 391.3413, 1.598236, 0.15625, 0.15625],
                    [1028.7407, 441.4791, 1.531883, 0.15625, 0.3125],
                    [1025.9136, 401.5553, 1.317260, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1012.5926, 358.6413, 1.353590, 0.0, 0.0],
                    [732.4010, 594.5176, 1.303136, 0.5, 0.0],
                    [776.7595, 607.5264, 1.517759, 0.5, 0.15625],
                    [1017.1773, 403.5133, 1.568213, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [987.9803, 357.2134, 1.494078, 0.0, 0.0],
                    [734.1957, 570.0852, 1.443624, 0.5, 0.0],
                    [779.4032, 634.4628, 1.377271, 0.5, 0.15625],
                    [1043.3846, 409.4229, 1.427725, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1017.1773, 403.5133, 1.568213, 0.0, 0.0],
                    [776.7595, 607.5264, 1.517759, 0.5, 0.0],
                    [732.4010, 594.5176, 1.303136, 0.5, 0.15625],
                    [1012.5926, 358.6413, 1.353590, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1043.3846, 409.4229, 1.427725, 0.0, 0.0],
                    [779.4032, 634.4628, 1.377271, 0.5, 0.0],
                    [734.1957, 570.0852, 1.443624, 0.5, 0.15625],
                    [987.9803, 357.2135, 1.494078, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "shake_2",
        camera: [10.0, 20.0],
        rel: [0.125, -0.5, 1.375],
        rotation: [100.0, 100.0, -30.0, -30.0],
        shake: 2, partial: 0.6, burning: false,
        player: None,
        mask: Mask { count: 10551, min_x: 745, max_x: 1053, min_y: 383, max_y: 601, cx: 916.0053, cy: 484.8199 },
        mean: [85.7010, 80.6352, 72.4817],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [555.0, 100.083, 100.083, 100.083], [2474.0, 91.645, 89.213, 84.487],
            [0.0, 0.000, 0.000, 0.000], [39.0, 68.487, 56.590, 32.795], [1496.0, 79.783, 67.401, 48.563], [1690.0, 98.371, 97.159, 95.173],
            [231.0, 76.814, 70.719, 59.333], [1345.0, 76.084, 60.759, 37.978], [96.0, 68.844, 56.875, 32.938], [0.0, 0.000, 0.000, 0.000],
            [2592.0, 78.922, 77.856, 75.724], [33.0, 73.636, 59.576, 36.091], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
        // 6 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1040.5671, 429.1702, 1.354985, 0.0, 0.15625],
                    [1035.6808, 468.6487, 1.567785, 0.15625, 0.15625],
                    [990.8069, 414.4488, 1.635961, 0.15625, 0.3125],
                    [988.7493, 368.7572, 1.423162, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [988.7493, 368.7572, 1.423162, 0.0, 0.15625],
                    [990.8069, 414.4488, 1.635961, 0.15625, 0.15625],
                    [1035.6808, 468.6487, 1.567785, 0.15625, 0.3125],
                    [1040.5671, 429.1702, 1.354985, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1031.6927, 386.8135, 1.395656, 0.0, 0.0],
                    [727.3308, 583.3257, 1.290329, 0.5, 0.0],
                    [766.5795, 602.6784, 1.503129, 0.5, 0.15625],
                    [1028.1041, 430.8976, 1.608456, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1006.0076, 382.8482, 1.536144, 0.0, 0.0],
                    [729.6393, 559.7736, 1.430817, 0.5, 0.0],
                    [768.2020, 629.4042, 1.362641, 0.5, 0.15625],
                    [1054.6387, 439.2659, 1.467968, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1028.1041, 430.8976, 1.608456, 0.0, 0.0],
                    [766.5795, 602.6784, 1.503129, 0.5, 0.0],
                    [727.3308, 583.3257, 1.290329, 0.5, 0.15625],
                    [1031.6927, 386.8135, 1.395656, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1054.6387, 439.2659, 1.467968, 0.0, 0.0],
                    [768.2020, 629.4042, 1.362641, 0.5, 0.0],
                    [729.6393, 559.7736, 1.430817, 0.5, 0.15625],
                    [1006.0076, 382.8482, 1.536144, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "burning",
        camera: [0.0, 0.0],
        rel: [0.0, -0.125, 1.625],
        rotation: [90.0, 90.0, 0.0, 0.0],
        shake: 0, partial: 0.0, burning: true,
        player: None,
        mask: Mask { count: 86532, min_x: 842, max_x: 1195, min_y: 169, max_y: 614, cx: 968.6774, cy: 419.1743 },
        mean: [203.4221, 132.1698, 43.9518],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [2535.0, 191.103, 89.516, 1.990], [6521.0, 198.192, 105.875, 7.779], [2670.0, 185.298, 78.017, 0.483], [0.0, 0.000, 0.000, 0.000],
            [9515.0, 204.260, 119.842, 21.457], [9768.0, 198.579, 106.452, 8.159], [5194.0, 198.110, 104.803, 4.761], [0.0, 0.000, 0.000, 0.000],
            [9933.0, 205.803, 121.158, 10.675], [9856.0, 230.554, 184.910, 86.731], [6139.0, 215.055, 144.925, 35.667], [0.0, 0.000, 0.000, 0.000],
            [7703.0, 206.481, 139.194, 68.378], [7510.0, 233.437, 200.074, 137.957], [5258.0, 204.663, 141.381, 68.122], [3930.0, 92.141, 89.829, 85.704],
        ],
        // 7 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1187.9314, 586.0468, 1.465901, 0.0, 0.15625],
                    [1165.6153, 634.4083, 1.625000, 0.15625, 0.15625],
                    [1147.2794, 577.8342, 1.784099, 0.15625, 0.3125],
                    [1165.6153, 528.6686, 1.625000, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1165.6153, 528.6686, 1.625000, 0.0, 0.15625],
                    [1147.2794, 577.8342, 1.784099, 0.15625, 0.15625],
                    [1165.6153, 634.4083, 1.625000, 0.15625, 0.3125],
                    [1187.9314, 586.0468, 1.465901, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1195.8535, 555.8810, 1.545450, 0.0, 0.0],
                    [881.3821, 555.8810, 1.545450, 0.5, 0.0],
                    [888.7201, 604.8011, 1.704549, 0.5, 0.15625],
                    [1173.8394, 604.8011, 1.704549, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1173.8394, 554.3987, 1.704549, 0.0, 0.0],
                    [888.7201, 554.3987, 1.704549, 0.5, 0.0],
                    [881.3821, 611.4722, 1.545450, 0.5, 0.15625],
                    [1195.8535, 611.4722, 1.545450, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1173.8394, 604.8011, 1.704549, 0.0, 0.0],
                    [888.7201, 604.8011, 1.704549, 0.5, 0.0],
                    [881.3821, 555.8810, 1.545450, 0.5, 0.15625],
                    [1195.8535, 555.8810, 1.545450, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1195.8535, 611.4722, 1.545450, 0.0, 0.0],
                    [881.3821, 611.4722, 1.545450, 0.5, 0.0],
                    [888.7201, 554.3987, 1.704549, 0.5, 0.15625],
                    [1173.8394, 554.3987, 1.704549, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: false, light0: true, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [826.4310, 587.7032, 1.415000, 0.43751952, 0.21873046],
                    [1093.5689, 587.7032, 1.415000, 0.46873048, 0.21873046],
                    [1093.5689, 213.7102, 1.415000, 0.46873048, 0.18751954],
                    [826.4310, 213.7102, 1.415000, 0.43751952, 0.18751954],
                    [841.5459, 468.5515, 1.436000, 0.43751952, 0.24998046],
                    [1078.4540, 468.5515, 1.436000, 0.46873048, 0.24998046],
                    [1078.4540, 100.0278, 1.436000, 0.46873048, 0.21876954],
                    [841.5459, 100.0278, 1.436000, 0.43751952, 0.21876954],
                ],
            },
        ],
    },
];

#[rustfmt::skip]
const PLAYERS: &[Ref] = &[
    Ref {
        name: "player_front",
        camera: [0.0, 0.0],
        rel: [0.0, -1.625, 2.5],
        rotation: [180.0, 180.0, 0.0, 0.0],
        shake: 0, partial: 0.25, burning: false,
        player: Some(PlayerArgs { entity_id: 1234, ticks: 37, arrows: 4, parts: 0x7f, hitboxes: false, debug_draws: 0 }),
        mask: Mask { count: 68950, min_x: 783, max_x: 1141, min_y: 327, max_y: 905, cx: 958.8453, cy: 665.1850 },
        mean: [74.5763, 96.0644, 104.9108],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [3173.0, 209.830, 197.235, 176.564], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [1160.0, 106.964, 175.495, 173.228], [9260.0, 71.641, 79.247, 68.419], [8535.0, 66.239, 79.972, 69.296], [1831.0, 163.340, 181.922, 165.521],
            [3252.0, 122.649, 116.952, 100.171], [12491.0, 42.492, 96.992, 106.617], [12474.0, 48.080, 101.090, 104.407], [1798.0, 104.362, 97.456, 82.753],
            [0.0, 0.000, 0.000, 0.000], [7920.0, 82.966, 76.952, 127.537], [7056.0, 63.809, 57.712, 117.005], [0.0, 0.000, 0.000, 0.000],
        ],
        // 24 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1098.5134, 567.2622, 3.033983, 0.0, 0.15625],
                    [1114.0130, 542.3971, 2.882799, 0.15625, 0.15625],
                    [1145.7952, 568.1412, 2.800266, 0.15625, 0.3125],
                    [1128.2343, 592.3830, 2.951450, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1128.2343, 592.3830, 2.951450, 0.0, 0.15625],
                    [1145.7952, 568.1412, 2.800266, 0.15625, 0.15625],
                    [1114.0130, 542.3971, 2.882799, 0.15625, 0.3125],
                    [1098.5134, 567.2622, 3.033983, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1118.3403, 575.4968, 3.023157, 0.0, 0.0],
                    [1020.7006, 653.9007, 2.536102, 0.5, 0.0],
                    [1034.5032, 629.3368, 2.384919, 0.5, 0.15625],
                    [1134.9420, 550.9715, 2.871974, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1141.9159, 576.2514, 2.906299, 0.0, 0.0],
                    [1044.3062, 658.5944, 2.419244, 0.5, 0.0],
                    [1011.0316, 625.9454, 2.501777, 0.5, 0.15625],
                    [1111.3683, 551.1966, 2.988832, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1134.9420, 550.9715, 2.871974, 0.0, 0.0],
                    [1034.5032, 629.3368, 2.384919, 0.5, 0.0],
                    [1020.7007, 653.9007, 2.536102, 0.5, 0.15625],
                    [1118.3403, 575.4968, 3.023157, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1111.3683, 551.1966, 2.988832, 0.0, 0.0],
                    [1011.0316, 625.9454, 2.501777, 0.5, 0.0],
                    [1044.3062, 658.5944, 2.419244, 0.5, 0.15625],
                    [1141.9159, 576.2514, 2.906299, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [992.4043, 806.7366, 1.893789, 0.0, 0.15625],
                    [948.4984, 762.9031, 1.930674, 0.15625, 0.15625],
                    [906.9575, 811.9263, 1.857646, 0.15625, 0.3125],
                    [951.7829, 858.5111, 1.820760, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [951.7829, 858.5111, 1.820760, 0.0, 0.15625],
                    [906.9575, 811.9263, 1.857646, 0.15625, 0.15625],
                    [948.4984, 762.9031, 1.930674, 0.15625, 0.3125],
                    [992.4043, 806.7366, 1.893789, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [974.6000, 834.1211, 1.808668, 0.0, 0.0],
                    [950.3908, 811.0712, 2.586367, 0.5, 0.0],
                    [918.6675, 778.7494, 2.623253, 0.5, 0.15625],
                    [929.0249, 787.7185, 1.845554, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [930.0971, 837.0895, 1.790597, 0.0, 0.0],
                    [919.1934, 812.9786, 2.568296, 0.5, 0.0],
                    [949.2194, 777.1160, 2.641324, 0.5, 0.15625],
                    [972.2257, 785.3164, 1.863625, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [929.0249, 787.7185, 1.845554, 0.0, 0.0],
                    [918.6675, 778.7494, 2.623253, 0.5, 0.0],
                    [950.3908, 811.0712, 2.586367, 0.5, 0.15625],
                    [974.5999, 834.1211, 1.808668, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [972.2257, 785.3164, 1.863625, 0.0, 0.0],
                    [949.2194, 777.1160, 2.641324, 0.5, 0.0],
                    [919.1934, 812.9785, 2.568296, 0.5, 0.15625],
                    [930.0971, 837.0894, 1.790597, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [778.2957, 617.2744, 2.755162, 0.0, 0.15625],
                    [806.5060, 586.6489, 2.848904, 0.15625, 0.15625],
                    [825.4206, 609.9117, 2.991838, 0.15625, 0.3125],
                    [799.2135, 639.7792, 2.898096, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [799.2135, 639.7792, 2.898096, 0.0, 0.15625],
                    [825.4206, 609.9117, 2.991838, 0.15625, 0.15625],
                    [806.5060, 586.6489, 2.848904, 0.15625, 0.3125],
                    [778.2957, 617.2744, 2.755162, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [783.1597, 625.4435, 2.857529, 0.0, 0.0],
                    [895.2968, 689.8985, 2.363132, 0.5, 0.0],
                    [923.5443, 651.6152, 2.456874, 0.5, 0.15625],
                    [810.2370, 595.6207, 2.951271, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [806.6552, 621.4175, 2.975867, 0.0, 0.0],
                    [918.1258, 681.9966, 2.481470, 0.5, 0.0],
                    [900.7494, 658.0628, 2.338536, 0.5, 0.15625],
                    [786.6871, 598.6041, 2.832932, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [810.2370, 595.6207, 2.951271, 0.0, 0.0],
                    [923.5443, 651.6152, 2.456874, 0.5, 0.0],
                    [895.2968, 689.8985, 2.363132, 0.5, 0.15625],
                    [783.1597, 625.4435, 2.857529, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [786.6871, 598.6041, 2.832932, 0.0, 0.0],
                    [900.7494, 658.0629, 2.338536, 0.5, 0.0],
                    [918.1258, 681.9966, 2.481470, 0.5, 0.15625],
                    [806.6552, 621.4175, 2.975867, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [954.8559, 342.2058, 2.055154, 0.0, 0.15625],
                    [933.5811, 339.8090, 2.231066, 0.15625, 0.15625],
                    [886.2534, 352.4441, 2.167341, 0.15625, 0.3125],
                    [904.0284, 356.0337, 1.991429, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [904.0284, 356.0337, 1.991429, 0.0, 0.15625],
                    [886.2534, 352.4441, 2.167341, 0.15625, 0.15625],
                    [933.5811, 339.8090, 2.231066, 0.15625, 0.3125],
                    [954.8559, 342.2058, 2.055154, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [926.8048, 334.3336, 1.998939, 0.0, 0.0],
                    [967.9724, 533.2563, 2.388581, 0.5, 0.0],
                    [948.5639, 518.0660, 2.564492, 0.5, 0.15625],
                    [907.2490, 332.5116, 2.174851, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [891.3947, 339.9474, 2.055033, 0.0, 0.0],
                    [937.2614, 533.4111, 2.444674, 0.5, 0.0],
                    [978.0607, 517.5755, 2.508399, 0.5, 0.15625],
                    [941.0764, 327.0183, 2.118757, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [907.2490, 332.5116, 2.174851, 0.0, 0.0],
                    [948.5639, 518.0660, 2.564492, 0.5, 0.0],
                    [967.9724, 533.2563, 2.388581, 0.5, 0.15625],
                    [926.8048, 334.3336, 1.998939, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [941.0764, 327.0184, 2.118757, 0.0, 0.0],
                    [978.0607, 517.5755, 2.508399, 0.5, 0.0],
                    [937.2614, 533.4111, 2.444674, 0.5, 0.15625],
                    [891.3947, 339.9474, 2.055033, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "player_back",
        camera: [0.0, 10.0],
        rel: [0.25, -1.625, 2.375],
        rotation: [10.0, 10.0, 0.0, 0.0],
        shake: 0, partial: 0.0, burning: false,
        player: Some(PlayerArgs { entity_id: 77, ticks: 5, arrows: 6, parts: 0x00, hitboxes: false, debug_draws: 0 }),
        mask: Mask { count: 74976, min_x: 731, max_x: 1044, min_y: 375, max_y: 804, cx: 897.5619, cy: 566.5987 },
        mean: [65.6788, 88.6047, 95.3214],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [3368.0, 215.994, 206.978, 192.090], [6296.0, 30.892, 21.121, 9.121], [8162.0, 30.502, 20.908, 9.641], [2447.0, 214.725, 205.527, 190.215],
            [1950.0, 68.617, 118.235, 108.529], [8275.0, 61.757, 125.741, 119.877], [8422.0, 29.020, 117.850, 114.346], [4579.0, 49.714, 110.652, 103.594],
            [2513.0, 176.292, 159.251, 146.514], [6433.0, 57.450, 74.825, 102.461], [8162.0, 31.640, 75.813, 111.418], [3660.0, 119.242, 87.180, 70.540],
            [0.0, 0.000, 0.000, 0.000], [3465.0, 59.206, 54.010, 103.802], [7244.0, 55.036, 50.075, 97.010], [0.0, 0.000, 0.000, 0.000],
        ],
        // 36 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1029.1415, 418.0204, 2.260841, 0.0, 0.15625],
                    [993.4175, 389.6588, 2.149916, 0.15625, 0.15625],
                    [1005.8107, 414.4098, 1.992088, 0.15625, 0.3125],
                    [1043.5621, 443.5945, 2.103012, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1043.5621, 443.5945, 2.103012, 0.0, 0.15625],
                    [1005.8107, 414.4098, 1.992088, 0.15625, 0.15625],
                    [993.4175, 389.6588, 2.149916, 0.15625, 0.3125],
                    [1029.1415, 418.0204, 2.260841, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1045.3938, 420.7677, 2.160595, 0.0, 0.0],
                    [915.5855, 554.4064, 2.501899, 0.5, 0.0],
                    [878.1950, 535.2315, 2.390975, 0.5, 0.15625],
                    [1008.8021, 391.1676, 2.049670, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1035.0027, 419.1750, 2.026218, 0.0, 0.0],
                    [899.3247, 560.6284, 2.367522, 0.5, 0.0],
                    [895.4291, 530.4187, 2.525351, 0.5, 0.15625],
                    [1020.6936, 394.4664, 2.184047, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1008.8021, 391.1676, 2.049670, 0.0, 0.0],
                    [878.1950, 535.2315, 2.390975, 0.5, 0.0],
                    [915.5855, 554.4064, 2.501899, 0.5, 0.15625],
                    [1045.3938, 420.7677, 2.160595, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1020.6936, 394.4664, 2.184047, 0.0, 0.0],
                    [895.4291, 530.4187, 2.525351, 0.5, 0.0],
                    [899.3247, 560.6284, 2.367522, 0.5, 0.15625],
                    [1035.0027, 419.1750, 2.026218, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [730.9812, 415.8807, 2.545172, 0.0, 0.15625],
                    [770.9158, 394.3663, 2.582264, 0.15625, 0.15625],
                    [784.8795, 418.3347, 2.781317, 0.15625, 0.3125],
                    [748.0303, 438.6125, 2.744225, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [748.0303, 438.6125, 2.744225, 0.0, 0.15625],
                    [784.8795, 418.3347, 2.781317, 0.15625, 0.15625],
                    [770.9158, 394.3663, 2.582264, 0.15625, 0.3125],
                    [730.9812, 415.8807, 2.545172, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [735.0911, 419.9245, 2.659482, 0.0, 0.0],
                    [817.7930, 555.2704, 2.422951, 0.5, 0.0],
                    [858.4027, 530.5855, 2.460043, 0.5, 0.15625],
                    [773.2763, 399.2665, 2.696574, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [761.9020, 420.9812, 2.777554, 0.0, 0.0],
                    [843.2567, 550.1365, 2.541024, 0.5, 0.0],
                    [832.8221, 534.9113, 2.341971, 0.5, 0.15625],
                    [746.1442, 397.1822, 2.578501, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [773.2763, 399.2665, 2.696574, 0.0, 0.0],
                    [858.4027, 530.5855, 2.460043, 0.5, 0.0],
                    [817.7930, 555.2704, 2.422951, 0.5, 0.15625],
                    [735.0911, 419.9245, 2.659482, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [746.1442, 397.1822, 2.578501, 0.0, 0.0],
                    [832.8221, 534.9113, 2.341971, 0.5, 0.0],
                    [843.2567, 550.1365, 2.541024, 0.5, 0.15625],
                    [761.9020, 420.9812, 2.777554, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [741.4298, 655.5633, 2.450787, 0.0, 0.15625],
                    [743.4348, 616.1300, 2.580534, 0.15625, 0.15625],
                    [771.9451, 633.1569, 2.742301, 0.15625, 0.3125],
                    [771.4802, 670.9942, 2.612553, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [771.4802, 670.9942, 2.612553, 0.0, 0.15625],
                    [771.9451, 633.1569, 2.742301, 0.15625, 0.15625],
                    [743.4348, 616.1300, 2.580534, 0.15625, 0.3125],
                    [741.4298, 655.5633, 2.450787, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [747.5904, 667.4160, 2.541327, 0.0, 0.0],
                    [906.1789, 601.3856, 2.386821, 0.5, 0.0],
                    [899.7408, 563.7431, 2.516568, 0.5, 0.15625],
                    [749.2282, 628.7436, 2.671074, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [762.8274, 655.3397, 2.687083, 0.0, 0.0],
                    [913.2182, 592.3727, 2.532577, 0.5, 0.0],
                    [891.8253, 571.0567, 2.370811, 0.5, 0.15625],
                    [733.1096, 639.3614, 2.525318, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [749.2282, 628.7436, 2.671074, 0.0, 0.0],
                    [899.7408, 563.7431, 2.516568, 0.5, 0.0],
                    [906.1789, 601.3856, 2.386821, 0.5, 0.15625],
                    [747.5904, 667.4160, 2.541327, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [733.1096, 639.3614, 2.525318, 0.0, 0.0],
                    [891.8253, 571.0568, 2.370811, 0.5, 0.0],
                    [913.2182, 592.3727, 2.532577, 0.5, 0.15625],
                    [762.8274, 655.3397, 2.687083, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [948.2632, 589.5363, 3.109260, 0.0, 0.15625],
                    [975.1263, 566.4556, 3.057537, 0.15625, 0.15625],
                    [996.0546, 595.9698, 2.976949, 0.15625, 0.3125],
                    [968.1193, 619.1605, 3.028672, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [968.1193, 619.1605, 3.028672, 0.0, 0.15625],
                    [996.0546, 595.9698, 2.976949, 0.15625, 0.15625],
                    [975.1263, 566.4556, 3.057537, 0.15625, 0.3125],
                    [948.2632, 589.5363, 3.109260, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [962.0184, 604.5211, 3.115953, 0.0, 0.0],
                    [879.8212, 596.8958, 2.364155, 0.5, 0.0],
                    [913.8092, 566.5428, 2.312433, 0.5, 0.15625],
                    [989.0550, 581.7438, 3.064230, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [985.6417, 607.9860, 3.049798, 0.0, 0.0],
                    [908.8066, 601.2748, 2.298000, 0.5, 0.0],
                    [884.8605, 563.1564, 2.378588, 0.5, 0.15625],
                    [965.4685, 578.8494, 3.130386, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [989.0550, 581.7438, 3.064230, 0.0, 0.0],
                    [913.8092, 566.5428, 2.312433, 0.5, 0.0],
                    [879.8212, 596.8958, 2.364155, 0.5, 0.15625],
                    [962.0184, 604.5211, 3.115953, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [965.4685, 578.8494, 3.130386, 0.0, 0.0],
                    [884.8605, 563.1564, 2.378588, 0.5, 0.0],
                    [908.8066, 601.2748, 2.298000, 0.5, 0.15625],
                    [985.6417, 607.9860, 3.049798, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [735.8474, 463.6412, 2.327060, 0.0, 0.15625],
                    [764.4871, 431.9180, 2.438013, 0.15625, 0.15625],
                    [771.2432, 461.5555, 2.616449, 0.15625, 0.3125],
                    [744.9423, 492.3319, 2.505496, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [744.9423, 492.3319, 2.505496, 0.0, 0.15625],
                    [771.2432, 461.5555, 2.616449, 0.15625, 0.15625],
                    [764.4871, 431.9180, 2.438013, 0.15625, 0.3125],
                    [735.8474, 463.6412, 2.327060, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [730.3227, 473.7724, 2.420920, 0.0, 0.0],
                    [899.0248, 551.9257, 2.346645, 0.5, 0.0],
                    [920.0693, 516.4695, 2.457598, 0.5, 0.15625],
                    [758.1429, 442.7812, 2.531873, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [748.6829, 472.1375, 2.565614, 0.0, 0.0],
                    [908.1343, 545.7030, 2.491339, 0.5, 0.0],
                    [911.5736, 520.9542, 2.312904, 0.5, 0.15625],
                    [740.0966, 442.6599, 2.387179, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [758.1429, 442.7812, 2.531873, 0.0, 0.0],
                    [920.0693, 516.4695, 2.457598, 0.5, 0.0],
                    [899.0248, 551.9257, 2.346645, 0.5, 0.15625],
                    [730.3227, 473.7724, 2.420920, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [740.0966, 442.6599, 2.387179, 0.0, 0.0],
                    [911.5736, 520.9542, 2.312904, 0.5, 0.0],
                    [908.1343, 545.7030, 2.491339, 0.5, 0.15625],
                    [748.6829, 472.1375, 2.565614, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [920.6003, 528.8524, 1.770745, 0.0, 0.15625],
                    [877.4966, 484.8727, 1.805118, 0.15625, 0.15625],
                    [830.7008, 528.2214, 1.784135, 0.15625, 0.3125],
                    [873.4021, 573.5801, 1.749761, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [873.4021, 573.5801, 1.749761, 0.0, 0.15625],
                    [830.7008, 528.2214, 1.784135, 0.15625, 0.15625],
                    [877.4966, 484.8727, 1.805118, 0.15625, 0.3125],
                    [920.6003, 528.8524, 1.770745, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [894.5267, 548.3169, 1.708489, 0.0, 0.0],
                    [923.5609, 579.0271, 2.536718, 0.5, 0.0],
                    [893.2590, 547.4789, 2.571092, 0.5, 0.15625],
                    [850.3976, 502.3823, 1.742862, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [847.8718, 547.9128, 1.715184, 0.0, 0.0],
                    [892.0221, 578.6737, 2.543413, 0.5, 0.0],
                    [924.4606, 547.7471, 2.564397, 0.5, 0.15625],
                    [896.3185, 502.6045, 1.736167, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [850.3976, 502.3823, 1.742862, 0.0, 0.0],
                    [893.2590, 547.4789, 2.571092, 0.5, 0.0],
                    [923.5609, 579.0271, 2.536718, 0.5, 0.15625],
                    [894.5267, 548.3169, 1.708489, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [896.3185, 502.6045, 1.736167, 0.0, 0.0],
                    [924.4606, 547.7471, 2.564397, 0.5, 0.0],
                    [892.0221, 578.6737, 2.543413, 0.5, 0.15625],
                    [847.8718, 547.9128, 1.715184, 0.0, 0.15625],
                ],
            },
        ],
    },
    Ref {
        name: "player_hitboxes",
        camera: [0.0, 0.0],
        rel: [0.0, -1.625, 2.5],
        rotation: [180.0, 180.0, 0.0, 0.0],
        shake: 0, partial: 0.25, burning: false,
        player: Some(PlayerArgs { entity_id: 4321, ticks: 37, arrows: 3, parts: 0x7f, hitboxes: true, debug_draws: 19 }),
        mask: Mask { count: 67836, min_x: 846, max_x: 1337, min_y: 236, max_y: 1079, cx: 966.9273, cy: 678.5797 },
        mean: [70.1207, 91.8275, 106.9127],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [58.0, 0.000, 0.000, 255.000], [153.0, 0.000, 0.000, 255.000],
            [15641.0, 67.889, 102.388, 95.191], [13055.0, 73.530, 111.346, 104.192], [1807.0, 93.180, 90.529, 99.416], [0.0, 0.000, 0.000, 0.000],
            [18162.0, 63.974, 78.165, 113.047], [14384.0, 69.378, 82.906, 110.088], [99.0, 255.000, 255.000, 255.000], [0.0, 0.000, 0.000, 0.000],
            [2475.0, 87.601, 85.629, 104.854], [1837.0, 89.629, 87.757, 112.400], [148.0, 0.000, 0.000, 255.000], [17.0, 0.000, 0.000, 255.000],
        ],
        // 18 draw(s)
        draws: &[
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1023.4529, 526.9089, 1.705799, 0.0, 0.15625],
                    [976.3220, 479.4853, 1.699645, 0.15625, 0.15625],
                    [928.8627, 526.7606, 1.686691, 0.15625, 0.3125],
                    [976.5268, 574.3753, 1.692845, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [976.5268, 574.3753, 1.692845, 0.0, 0.15625],
                    [928.8627, 526.7606, 1.686691, 0.15625, 0.15625],
                    [976.3220, 479.4853, 1.699645, 0.15625, 0.3125],
                    [1023.4529, 526.9089, 1.705799, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1002.4672, 550.4937, 1.646710, 0.0, 0.0],
                    [976.3765, 551.1266, 2.488507, 0.5, 0.0],
                    [943.9897, 518.7161, 2.482353, 0.5, 0.15625],
                    [953.5600, 501.4504, 1.640556, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [953.6187, 550.5550, 1.637155, 0.0, 0.0],
                    [944.0153, 551.1695, 2.478953, 0.5, 0.0],
                    [976.3069, 518.7977, 2.491907, 0.5, 0.15625],
                    [1002.3083, 501.6737, 1.650110, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [953.5600, 501.4504, 1.640556, 0.0, 0.0],
                    [943.9897, 518.7161, 2.482353, 0.5, 0.0],
                    [976.3765, 551.1266, 2.488507, 0.5, 0.15625],
                    [1002.4672, 550.4937, 1.646710, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: false, lighting: false, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1002.3083, 501.6737, 1.650110, 0.0, 0.0],
                    [976.3069, 518.7977, 2.491907, 0.5, 0.0],
                    [944.0153, 551.1695, 2.478953, 0.5, 0.15625],
                    [953.6187, 550.5550, 1.637155, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [950.9705, 700.6312, 3.137506, 0.0, 0.15625],
                    [977.4964, 679.5063, 3.085443, 0.15625, 0.15625],
                    [993.1838, 711.7661, 2.952444, 0.15625, 0.3125],
                    [965.2118, 733.2672, 3.004508, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [965.2118, 733.2672, 3.004508, 0.0, 0.15625],
                    [993.1838, 711.7661, 2.952444, 0.15625, 0.15625],
                    [977.4964, 679.5063, 3.085443, 0.15625, 0.3125],
                    [950.9705, 700.6312, 3.137506, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [963.2121, 717.6917, 3.109813, 0.0, 0.0],
                    [859.0699, 696.0590, 2.488907, 0.5, 0.0],
                    [890.6926, 669.2136, 2.436844, 0.5, 0.15625],
                    [990.1867, 696.6658, 3.057750, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [984.2406, 723.6626, 3.017282, 0.0, 0.0],
                    [881.5257, 702.7418, 2.396376, 0.5, 0.0],
                    [868.2607, 663.8643, 2.529375, 0.5, 0.15625],
                    [969.2537, 691.5645, 3.150281, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [990.1867, 696.6658, 3.057750, 0.0, 0.0],
                    [890.6926, 669.2136, 2.436844, 0.5, 0.0],
                    [859.0699, 696.0590, 2.488907, 0.5, 0.15625],
                    [963.2121, 717.6917, 3.109813, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [969.2537, 691.5645, 3.150281, 0.0, 0.0],
                    [868.2607, 663.8643, 2.529375, 0.5, 0.0],
                    [881.5257, 702.7418, 2.396376, 0.5, 0.15625],
                    [984.2406, 723.6626, 3.017282, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1103.6788, 496.6273, 2.766230, 0.0, 0.15625],
                    [1098.5337, 474.5174, 2.590350, 0.15625, 0.15625],
                    [1138.3042, 491.6140, 2.479623, 0.15625, 0.3125],
                    [1141.0297, 513.5134, 2.655504, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [-0.05625, 0.0, 0.0],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1141.0297, 513.5134, 2.655504, 0.0, 0.15625],
                    [1138.3042, 491.6140, 2.479623, 0.15625, 0.15625],
                    [1098.5337, 474.5174, 2.590350, 0.15625, 0.3125],
                    [1103.6788, 496.6273, 2.766230, 0.0, 0.3125],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1127.6359, 496.7697, 2.719882, 0.0, 0.0],
                    [1032.2692, 633.6496, 2.575643, 0.5, 0.0],
                    [1021.4818, 619.8262, 2.399762, 0.5, 0.15625],
                    [1124.0534, 474.2668, 2.544002, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1145.6296, 494.3653, 2.576578, 0.0, 0.0],
                    [1045.7113, 639.1670, 2.432339, 0.5, 0.0],
                    [1009.2329, 615.3280, 2.543066, 0.5, 0.15625],
                    [1106.9922, 477.7721, 2.687305, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1124.0534, 474.2668, 2.544002, 0.0, 0.0],
                    [1021.4818, 619.8262, 2.399762, 0.5, 0.0],
                    [1032.2692, 633.6496, 2.575643, 0.5, 0.15625],
                    [1127.6359, 496.7697, 2.719882, 0.0, 0.15625],
                ],
            },
            Draw {
                format: Format::PositionTex,
                cull: true, lighting: true, light0: false, normalize: false,
                normal: [0.0, 0.0, 0.05625],
                // window x, window y (down), eye depth, u, v
                verts: &[
                    [1106.9922, 477.7721, 2.687305, 0.0, 0.0],
                    [1009.2329, 615.3280, 2.543066, 0.5, 0.0],
                    [1045.7113, 639.1670, 2.432339, 0.5, 0.15625],
                    [1145.6296, 494.3653, 2.576578, 0.0, 0.15625],
                ],
            },
        ],
    },
];

#[rustfmt::skip]
const PACK_ARROWS: &[Ref] = &[
    Ref {
        name: "pack_side_level",
        camera: [0.0, 0.0],
        rel: [0.0, 0.0, 1.5],
        rotation: [90.0, 90.0, 0.0, 0.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 8512, min_x: 878, max_x: 1215, min_y: 504, max_y: 575, cx: 1079.6438, cy: 540.0000 },
        mean: [51.1527, 51.1527, 52.7707],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [916.0, 74.605, 74.605, 76.773],
            [1011.0, 65.829, 65.829, 67.455], [504.0, 10.464, 10.464, 11.464], [663.0, 24.612, 24.612, 26.074], [1162.0, 52.691, 52.691, 54.172],
            [1011.0, 54.793, 54.793, 56.542], [504.0, 11.464, 11.464, 12.464], [663.0, 25.392, 25.392, 26.854], [1162.0, 57.494, 57.494, 58.975],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [916.0, 79.569, 79.569, 81.737],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_side_from_above",
        camera: [0.0, 30.0],
        rel: [0.0, -0.75, 1.25],
        rotation: [90.0, 90.0, 0.0, 0.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 11104, min_x: 876, max_x: 1229, min_y: 507, max_y: 591, cx: 1078.4271, cy: 549.3049 },
        mean: [55.2081, 55.2081, 56.7653],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [164.0, 87.854, 87.854, 89.317], [0.0, 0.000, 0.000, 0.000], [68.0, 73.000, 73.000, 76.000], [1181.0, 83.046, 83.046, 84.964],
            [1286.0, 69.628, 69.628, 71.009], [792.0, 11.187, 11.187, 12.187], [998.0, 26.247, 26.247, 27.724], [1403.0, 70.225, 70.225, 71.855],
            [1208.0, 54.188, 54.188, 55.921], [616.0, 11.432, 11.432, 12.432], [821.0, 26.798, 26.798, 28.280], [1218.0, 58.093, 58.093, 59.430],
            [126.0, 52.000, 52.000, 54.000], [0.0, 0.000, 0.000, 0.000], [39.0, 73.000, 73.000, 76.000], [1184.0, 82.603, 82.603, 84.569],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_oblique_flight",
        camera: [20.0, -10.0],
        rel: [0.375, 0.25, 1.625],
        rotation: [60.0, 60.0, 25.0, 25.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 11347, min_x: 544, max_x: 855, min_y: 494, max_y: 684, cx: 725.2051, cy: 598.5971 },
        mean: [53.4580, 53.4580, 55.1551],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [2325.0, 60.002, 60.002, 62.052], [59.0, 9.678, 9.678, 10.678], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
            [130.0, 25.015, 25.015, 26.815], [1282.0, 11.124, 11.124, 12.124], [244.0, 13.947, 13.947, 15.004], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [4.0, 13.000, 13.000, 14.000], [2101.0, 43.233, 43.233, 44.790], [2388.0, 67.021, 67.021, 68.802],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [894.0, 86.532, 86.532, 88.029], [1920.0, 61.097, 61.097, 63.072],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_toward_camera",
        camera: [0.0, 0.0],
        rel: [0.125, 0.125, 1.25],
        rotation: [180.0, 180.0, 10.0, 10.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 3729, min_x: 873, max_x: 950, min_y: 441, max_y: 561, cx: 912.1661, cy: 502.8138 },
        mean: [58.4277, 58.4277, 60.0695],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [207.0, 68.807, 68.807, 70.444], [395.0, 79.985, 79.985, 81.276], [248.0, 79.065, 79.065, 80.613], [0.0, 0.000, 0.000, 0.000],
            [326.0, 47.525, 47.525, 50.525], [470.0, 31.349, 31.349, 33.089], [189.0, 52.873, 52.873, 54.212], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [276.0, 53.957, 53.957, 55.609], [450.0, 25.593, 25.593, 26.722], [292.0, 79.634, 79.634, 81.288],
            [0.0, 0.000, 0.000, 0.000], [248.0, 74.976, 74.976, 77.105], [342.0, 63.570, 63.570, 64.491], [286.0, 77.675, 77.675, 79.598],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_away_from_camera",
        camera: [0.0, 0.0],
        rel: [-0.125, -0.125, 1.25],
        rotation: [-10.0, -10.0, -8.0, -8.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 12818, min_x: 895, max_x: 1055, min_y: 494, max_y: 655, cx: 977.8819, cy: 574.3254 },
        mean: [73.6034, 73.6034, 75.0707],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [954.0, 69.369, 69.369, 72.369], [435.0, 82.469, 82.469, 83.972], [406.0, 83.734, 83.734, 85.079], [1133.0, 68.206, 68.206, 71.203],
            [248.0, 83.371, 83.371, 84.774], [1416.0, 69.862, 69.862, 70.163], [1416.0, 71.069, 71.069, 71.356], [300.0, 84.210, 84.210, 85.520],
            [477.0, 78.509, 78.509, 80.415], [1424.0, 76.581, 76.581, 76.815], [1487.0, 77.880, 77.880, 78.108], [505.0, 72.281, 72.281, 74.200],
            [979.0, 68.222, 68.222, 71.222], [231.0, 79.169, 79.169, 81.039], [334.0, 78.817, 78.817, 80.731], [1073.0, 69.350, 69.350, 72.350],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_stuck_ground",
        camera: [15.0, 45.0],
        rel: [0.25, -1.125, 1.25],
        rotation: [200.0, 200.0, -55.0, -55.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 9505, min_x: 630, max_x: 810, min_y: 288, max_y: 605, cx: 714.1779, cy: 413.5146 },
        mean: [52.5989, 52.5989, 54.5632],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [1953.0, 67.554, 67.554, 69.486], [2548.0, 67.189, 67.189, 69.003], [15.0, 71.000, 71.000, 74.000], [0.0, 0.000, 0.000, 0.000],
            [167.0, 74.485, 74.485, 77.108], [1450.0, 35.419, 35.419, 37.823], [263.0, 11.703, 11.703, 13.703], [0.0, 0.000, 0.000, 0.000],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1073.0, 10.392, 10.392, 12.392], [48.0, 9.313, 9.313, 11.313],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [102.0, 37.549, 37.549, 40.010], [1886.0, 60.156, 60.156, 61.898],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_stuck_wall",
        camera: [-30.0, 5.0],
        rel: [-0.5, 0.0, 1.5],
        rotation: [130.0, 130.0, -5.0, -5.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 17516, min_x: 1467, max_x: 1893, min_y: 418, max_y: 526, cx: 1700.1165, cy: 478.8201 },
        mean: [59.6821, 59.6821, 61.2594],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [899.0, 81.981, 81.981, 83.846], [1312.0, 74.307, 74.307, 76.539],
            [60.0, 70.900, 70.900, 73.550], [21.0, 50.857, 50.857, 53.286], [1916.0, 56.459, 56.459, 58.072], [1500.0, 64.078, 64.078, 65.023],
            [2466.0, 71.081, 71.081, 72.504], [1940.0, 10.985, 10.985, 11.988], [2377.0, 45.696, 45.696, 47.234], [1955.0, 81.605, 81.605, 83.162],
            [2119.0, 60.125, 60.125, 62.045], [31.0, 28.968, 28.968, 30.419], [357.0, 73.134, 73.134, 75.504], [563.0, 80.677, 80.677, 82.792],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_from_below",
        camera: [0.0, -50.0],
        rel: [0.0, 1.0, 0.875],
        rotation: [45.0, 45.0, 12.0, 12.0],
        shake: 0, partial: 0.0, burning: false,
        player: None,
        mask: Mask { count: 18960, min_x: 897, max_x: 1264, min_y: 343, max_y: 598, cx: 1121.9995, cy: 462.4163 },
        mean: [53.1060, 53.1060, 54.7782],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1459.0, 78.343, 78.343, 79.974], [1938.0, 60.720, 60.720, 63.005],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [3994.0, 56.593, 56.593, 57.970], [4183.0, 60.252, 60.252, 62.069],
            [100.0, 11.940, 11.940, 12.940], [2017.0, 10.654, 10.654, 11.654], [1413.0, 45.180, 45.180, 47.118], [972.0, 76.639, 76.639, 78.460],
            [2774.0, 48.490, 48.490, 50.314], [110.0, 12.000, 12.000, 13.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_interpolated",
        camera: [0.0, 0.0],
        rel: [0.0, 0.0, 1.5],
        rotation: [30.0, 80.0, -10.0, 15.0],
        shake: 0, partial: 0.35, burning: false,
        player: None,
        mask: Mask { count: 12017, min_x: 902, max_x: 1230, min_y: 484, max_y: 582, cx: 1112.5939, cy: 535.0645 },
        mean: [55.3061, 55.3061, 56.9755],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [446.0, 68.966, 68.966, 71.637], [1356.0, 75.299, 75.299, 77.468],
            [432.0, 76.632, 76.632, 78.519], [208.0, 10.659, 10.659, 11.659], [1269.0, 52.761, 52.761, 54.173], [1450.0, 50.078, 50.078, 51.429],
            [1447.0, 44.311, 44.311, 46.040], [859.0, 11.255, 11.255, 12.255], [1460.0, 49.581, 49.581, 50.897], [1446.0, 56.888, 56.888, 58.313],
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [424.0, 74.710, 74.710, 77.533], [1220.0, 79.302, 79.302, 81.447],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_shake_7",
        camera: [10.0, 20.0],
        rel: [0.125, -0.5, 1.375],
        rotation: [100.0, 100.0, -30.0, -30.0],
        shake: 7, partial: 0.3, burning: false,
        player: None,
        mask: Mask { count: 10844, min_x: 752, max_x: 1042, min_y: 358, max_y: 607, cx: 912.1328, cy: 470.0114 },
        mean: [52.7510, 52.7510, 54.5566],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [1011.0, 78.606, 78.606, 80.569], [2796.0, 62.093, 62.093, 64.084],
            [0.0, 0.000, 0.000, 0.000], [76.0, 9.000, 9.000, 10.697], [1622.0, 23.752, 23.752, 25.299], [1089.0, 82.567, 82.567, 84.094],
            [120.0, 22.125, 22.125, 24.075], [1365.0, 10.815, 10.815, 12.199], [46.0, 9.000, 9.000, 11.000], [0.0, 0.000, 0.000, 0.000],
            [2685.0, 63.943, 63.943, 65.970], [34.0, 9.000, 9.000, 11.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_shake_2",
        camera: [10.0, 20.0],
        rel: [0.125, -0.5, 1.375],
        rotation: [100.0, 100.0, -30.0, -30.0],
        shake: 2, partial: 0.6, burning: false,
        player: None,
        mask: Mask { count: 10551, min_x: 745, max_x: 1053, min_y: 383, max_y: 601, cx: 916.0053, cy: 484.8199 },
        mean: [53.1678, 53.1678, 54.8100],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000], [555.0, 77.308, 77.308, 79.724], [2474.0, 64.816, 64.816, 66.500],
            [0.0, 0.000, 0.000, 0.000], [39.0, 9.897, 9.897, 10.897], [1496.0, 22.388, 22.388, 23.749], [1690.0, 75.808, 75.808, 77.486],
            [231.0, 41.528, 41.528, 43.550], [1345.0, 11.220, 11.220, 12.220], [96.0, 9.969, 9.969, 10.969], [0.0, 0.000, 0.000, 0.000],
            [2592.0, 65.490, 65.490, 67.405], [33.0, 10.000, 10.000, 11.000], [0.0, 0.000, 0.000, 0.000], [0.0, 0.000, 0.000, 0.000],
        ],
        draws: &[],
    },
    Ref {
        name: "pack_burning",
        camera: [0.0, 0.0],
        rel: [0.0, -0.125, 1.625],
        rotation: [90.0, 90.0, 0.0, 0.0],
        shake: 0, partial: 0.0, burning: true,
        player: None,
        mask: Mask { count: 86532, min_x: 842, max_x: 1195, min_y: 169, max_y: 614, cx: 968.6774, cy: 419.1743 },
        mean: [201.8525, 130.7647, 42.9238],
        // 4x4 cells over the mask bounds, row-major: [pixels, mean r, g, b]
        grid: [
            [2535.0, 191.103, 89.516, 1.990], [6521.0, 198.192, 105.875, 7.779], [2670.0, 185.298, 78.017, 0.483], [0.0, 0.000, 0.000, 0.000],
            [9515.0, 204.260, 119.842, 21.457], [9768.0, 198.579, 106.452, 8.159], [5194.0, 198.110, 104.803, 4.761], [0.0, 0.000, 0.000, 0.000],
            [9933.0, 205.803, 121.158, 10.675], [9856.0, 230.554, 184.910, 86.731], [6139.0, 215.055, 144.925, 35.667], [0.0, 0.000, 0.000, 0.000],
            [7703.0, 205.794, 138.508, 67.772], [7510.0, 233.171, 199.808, 137.723], [5258.0, 200.426, 138.123, 66.442], [3930.0, 65.105, 65.105, 66.951],
        ],
        draws: &[],
    },
];
