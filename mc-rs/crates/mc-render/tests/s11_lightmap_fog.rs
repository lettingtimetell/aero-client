//! S20 -> S11: the lightmap and the per-draw fog term inside `terrain.wgsl`.
//!
//! Two §13 rows meet here. The first is the open S20 -> S11 request —
//! `sky::lightmap::Lightmap`/`LightmapParams` and `Environment::sun_brightness` were built and
//! tested but nothing consumed them, because `terrain.wgsl` and `TerrainRenderer` are S11's.
//! The second is `sky::fog::FogPhase`'s own documented limitation: it is a screen-space pass
//! over the shared depth buffer, so it can only fog the nearest *depth-writing* surface and gets
//! translucent terrain wrong. Both are now inside the terrain shader.
//!
//! Every expectation below is computed on the CPU from the atlas bytes and S20's own public
//! tables — plain GL_RGBA byte sample, vertex colour, lightmap texel, then the fog blend — rather
//! than compared against a stored image, so a wrong texture-format choice or a wrong fog falloff
//! fails with a number rather than a diff.
//!
//! # Why the camera looks straight down
//!
//! A horizontal surface seen from directly above is a plane of constant eye-space depth. Every
//! point on it shares one clip `w`, so the projection restricted to that plane is affine and a
//! chosen world point maps to a known pixel with a known UV — and, for the fog half, to one
//! exactly known fog distance for the whole surface.

use glam::Vec4;
use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::sky::fog::{FogMode, FogState};
use mc_render::sky::lightmap::{Lightmap, LightmapParams};
use mc_render::terrain::{fixture, mesh_section, mesh_section_with, MeshSettings, TerrainPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 192;
/// World y of the stone slab's top surface, and the camera's height above it.
const SURFACE_Y: f64 = 21.0;
const EYE_Y: f64 = 41.0;
/// Therefore every point of the surface is exactly this far from the eye, in eye-space z.
const EYE_DISTANCE: f32 = 20.0;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

struct Fixture {
    atlas: mc_assets::Atlas,
    gpu: Gpu,
}

fn fixture_or_skip() -> Option<Fixture> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return None;
    }
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let (assets, report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    assert!(report.is_clean());
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    let _ = mc_render::terrain::tint::load_colormaps(&rm);

    let gpu = match Gpu::headless(
        WIDTH,
        HEIGHT,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => {
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            g
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available");
            return None;
        }
        Err(e) => panic!("headless context: {e}"),
    };
    Some(Fixture { atlas, gpu })
}

fn stone_slab() -> mc_world::World {
    let mut w = fixture::blank_world(1);
    fixture::fill(
        &mut w,
        BlockPos::new(0, SURFACE_Y as i32 - 1, 0),
        BlockPos::new(15, SURFACE_Y as i32 - 1, 15),
        fixture::state_of("minecraft:stone"),
    );
    w
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(8.0, EYE_Y, 8.0),
        yaw: 0.0,
        pitch: 90.0, // straight down
        // Keep the 16x16 stone texel footprint below one screen pixel. The production solid
        // sampler is mipmapped, so FOV 70 at this eye distance would intentionally select a
        // lower atlas mip instead of the base texel this scalar oracle names.
        fov: 20.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

fn atlas_byte(c: u8) -> f32 {
    f32::from(c) / 255.0
}

/// The stone slab's up-facing quad over block `(7, *, 7)`, as it comes out of the mesher.
struct TopFace {
    /// Section-local corner UVs, in `EnumFaceDirection`'s UP order.
    uv: [[f32; 2]; 4],
    /// Section-local corner positions, in the same order as `uv`.
    position: [[f32; 3]; 4],
    color: [u8; 4],
    light: [u8; 4],
}

fn top_face(mesh: &mc_render::terrain::ChunkMesh) -> TopFace {
    let local_y = (SURFACE_Y - 16.0) as f32;
    for i in 0..mesh.solid.quad_count() {
        let q = mesh.solid.quad(i);
        if !q.iter().all(|v| (v.position()[1] - local_y).abs() < 1e-4) {
            continue;
        }
        if q[0].position()[0].floor() as i32 != 7 || q[0].position()[2].floor() as i32 != 7 {
            continue;
        }
        // Vanilla's UP vertex table is (W,U,N) (W,U,S) (E,U,S) (E,U,N), so u tracks x and v
        // tracks z. Asserted rather than assumed, because the sample point below relies on it.
        assert!(q[0].position()[0] < q[2].position()[0] && q[0].position()[2] < q[1].position()[2]);
        assert!(q[0].texcoord()[0] < q[2].texcoord()[0] && q[0].texcoord()[1] < q[1].texcoord()[1]);
        return TopFace {
            uv: [
                q[0].texcoord(),
                q[1].texcoord(),
                q[2].texcoord(),
                q[3].texcoord(),
            ],
            position: [
                q[0].position(),
                q[1].position(),
                q[2].position(),
                q[3].position(),
            ],
            color: q[0].color,
            light: q[0].light,
        };
    }
    panic!("no up-facing stone quad over block (7,*,7)");
}

/// Continuous framebuffer coordinates for `world_pt`.
fn screen_of(cam: &Camera, world_pt: DVec3) -> [f32; 2] {
    let rel = cam.relative(world_pt);
    let clip = cam.view_proj() * Vec4::new(rel.x, rel.y, rel.z, 1.0);
    let ndc = clip.truncate() / clip.w;
    [
        (ndc.x * 0.5 + 0.5) * WIDTH as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * HEIGHT as f32,
    ]
}

/// The framebuffer pixel that `world_pt` projects to. Exact because the surface is a plane of
/// constant depth (see the module docs).
fn pixel_of(cam: &Camera, world_pt: DVec3) -> (u32, u32) {
    let screen = screen_of(cam, world_pt);
    let x = screen[0].floor();
    let y = screen[1].floor();
    assert!(
        (0.0..WIDTH as f32).contains(&x) && (0.0..HEIGHT as f32).contains(&y),
        "sample point projects off screen"
    );
    (x as u32, y as u32)
}

/// Interpolate the face UV at the centre of the selected raster pixel.
///
/// The world point used to choose a pixel is the centre of an atlas texel in ideal continuous
/// space, but a rasterizer evaluates the fragment at `(pixel + 0.5)`. Since the face is planar,
/// its screen mapping is affine here; solving the two screen axes gives the actual interpolated
/// UV. This keeps the CPU oracle tied to the texel the shader samples instead of assuming that
/// the pixel centre and the requested world point coincide.
fn uv_at_pixel_center(cam: &Camera, face: &TopFace, pixel: (u32, u32)) -> [f32; 2] {
    let screen_position = |p: [f32; 3]| {
        screen_of(
            cam,
            DVec3::new(f64::from(p[0]), f64::from(p[1] + 16.0), f64::from(p[2])),
        )
    };
    // UP's order is (W,U,N), (W,U,S), (E,U,S), (E,U,N): q0 + x*(q3-q0) + z*(q1-q0).
    let origin = screen_position(face.position[0]);
    let x_axis = {
        let p = screen_position(face.position[3]);
        [p[0] - origin[0], p[1] - origin[1]]
    };
    let z_axis = {
        let p = screen_position(face.position[1]);
        [p[0] - origin[0], p[1] - origin[1]]
    };
    let target = [
        pixel.0 as f32 + 0.5 - origin[0],
        pixel.1 as f32 + 0.5 - origin[1],
    ];
    let determinant = x_axis[0] * z_axis[1] - x_axis[1] * z_axis[0];
    assert!(
        determinant.abs() > f32::EPSILON,
        "top face projected to a line"
    );
    let x = (target[0] * z_axis[1] - target[1] * z_axis[0]) / determinant;
    let z = (x_axis[0] * target[1] - x_axis[1] * target[0]) / determinant;
    assert!(
        (-0.01..=1.01).contains(&x) && (-0.01..=1.01).contains(&z),
        "sample pixel left the selected face: ({x}, {z})"
    );
    [
        face.uv[0][0] + (face.uv[3][0] - face.uv[0][0]) * x + (face.uv[1][0] - face.uv[0][0]) * z,
        face.uv[0][1] + (face.uv[3][1] - face.uv[0][1]) * x + (face.uv[1][1] - face.uv[0][1]) * z,
    ]
}

/// Render the slab and read back the pixel over the centre of texel `(5,5)` of the top face,
/// plus the atlas texel that pixel samples.
fn shoot(
    f: &mut Fixture,
    lightmap: Option<LightmapParams>,
    fog: Option<FogState>,
) -> ([u8; 4], [u8; 4], TopFace) {
    let world = stone_slab();
    // The expected value below follows one face's colour/light pair.  Keep the fixture on
    // vanilla's flat `renderModelStandard` path so smooth AO does not interpolate four distinct
    // vertex values into the sampled pixel and make this lightmap/fog oracle test an AO test too.
    let mesh = mesh_section_with(
        &world,
        ChunkSectionPos::new(0, 1, 0),
        MeshSettings {
            ambient_occlusion: false,
            ..MeshSettings::default()
        },
    );
    let face = top_face(&mesh);
    let cam = camera();

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    let shared = TerrainPhase::install(&f.gpu, &f.atlas, &mut phases);
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        if let Some(p) = lightmap {
            r.set_lightmap_params(&f.gpu, p);
        }
        r.set_fog(fog);
        r.upload(&f.gpu, mesh, cam.pos);
    }
    let frame = f.gpu.acquire().expect("acquire");
    phases.render(&f.gpu, frame.view(), &cam, 0.0);
    let px = f.gpu.read_pixels().expect("readback");

    // 5.5/16 into the quad, i.e. the centre of texel (5, 5) of stone.png.
    let frac = 5.5 / 16.0;
    let world_pt = DVec3::new(7.0 + frac, SURFACE_Y, 7.0 + frac);
    let (sx, sy) = pixel_of(&cam, world_pt);
    let o = ((sy * WIDTH + sx) * 4) as usize;
    let got = [px[o], px[o + 1], px[o + 2], px[o + 3]];

    // The shader interpolates at the pixel centre. On this planar face that is an affine UV
    // interpolation, and the level-zero nearest sampler then picks this atlas texel.
    let [u, v] = uv_at_pixel_center(&cam, &face, (sx, sy));
    let ax = (u * f.atlas.w as f32).floor() as u32;
    let ay = (v * f.atlas.h as f32).floor() as u32;
    let ao = ((ay * f.atlas.w + ax) * 4) as usize;
    let texel = [
        f.atlas.texture[ao],
        f.atlas.texture[ao + 1],
        f.atlas.texture[ao + 2],
        f.atlas.texture[ao + 3],
    ];
    (got, texel, face)
}

/// The pre-fog shader value: normalized GL_RGBA atlas byte × vertex colour × lightmap.
fn shaded(texel: [u8; 4], face: &TopFace, lm: [u8; 4]) -> [f32; 3] {
    std::array::from_fn(|c| {
        atlas_byte(texel[c]) * (f32::from(face.color[c]) / 255.0) * (f32::from(lm[c]) / 255.0)
    })
}

fn to_bytes(v: [f32; 3]) -> [u8; 3] {
    std::array::from_fn(|c| (v[c] * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// The lightmap is genuinely sampled: full daylight and dusk produce the exact colours
/// `Lightmap::generate` predicts, and they differ.
#[test]
fn terrain_samples_the_real_lightmap_texture() {
    let Some(mut f) = fixture_or_skip() else {
        return;
    };

    let noon = LightmapParams::default();
    let (got_noon, texel, face) = shoot(&mut f, Some(noon), None);
    let expect_noon = to_bytes(shaded(
        texel,
        &face,
        Lightmap::generate(noon).sample(face.light[0] / 16, face.light[1] / 16),
    ));
    eprintln!("noon: got {got_noon:?} expected {expect_noon:?} from texel {texel:?}");
    for c in 0..3 {
        assert!(
            got_noon[c].abs_diff(expect_noon[c]) <= 1,
            "noon channel {c}: {got_noon:?} vs {expect_noon:?}"
        );
    }

    // java: World.getSunBrightness well after sunset. Same geometry, same texel, only the
    // lightmap table changes.
    let dusk = LightmapParams {
        sun_brightness: 0.25,
        ..LightmapParams::default()
    };
    let (got_dusk, texel_dusk, face_dusk) = shoot(&mut f, Some(dusk), None);
    assert_eq!(texel_dusk, texel, "the sample point moved between runs");
    let expect_dusk = to_bytes(shaded(
        texel,
        &face_dusk,
        Lightmap::generate(dusk).sample(face.light[0] / 16, face.light[1] / 16),
    ));
    eprintln!("dusk: got {got_dusk:?} expected {expect_dusk:?}");
    for c in 0..3 {
        assert!(
            got_dusk[c].abs_diff(expect_dusk[c]) <= 1,
            "dusk channel {c}: {got_dusk:?} vs {expect_dusk:?}"
        );
    }
    assert!(
        i32::from(got_dusk[0]) < i32::from(got_noon[0]) - 20,
        "a quarter-brightness sun must visibly darken sky-lit terrain ({got_dusk:?} vs \
         {got_noon:?}) — an unwired lightmap would leave them identical"
    );
}

/// `TerrainRenderer::sky_brightness` is the shorthand path into the same table, so setting it
/// must land on exactly the same pixels as the full `LightmapParams`.
#[test]
fn sky_brightness_drives_the_lightmap() {
    let Some(mut f) = fixture_or_skip() else {
        return;
    };

    let params = LightmapParams {
        sun_brightness: 0.4,
        ..LightmapParams::default()
    };
    let (via_params, _, _) = shoot(&mut f, Some(params), None);

    // The same thing via the plain field, which `begin_frame` notices and regenerates from.
    let world = stone_slab();
    let mesh = mesh_section(&world, ChunkSectionPos::new(0, 1, 0));
    let cam = camera();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    let shared = TerrainPhase::install(&f.gpu, &f.atlas, &mut phases);
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        r.sky_brightness = 0.4;
        r.upload(&f.gpu, mesh, cam.pos);
    }
    let frame = f.gpu.acquire().expect("acquire");
    phases.render(&f.gpu, frame.view(), &cam, 0.0);
    let px = f.gpu.read_pixels().expect("readback");
    let (sx, sy) = pixel_of(
        &cam,
        DVec3::new(7.0 + 5.5 / 16.0, SURFACE_Y, 7.0 + 5.5 / 16.0),
    );
    let o = ((sy * WIDTH + sx) * 4) as usize;
    let via_field = [px[o], px[o + 1], px[o + 2], px[o + 3]];
    assert_eq!(via_field, via_params);
}

/// Linear fog, at a distance chosen so the factor is exactly `0.5`.
///
/// java: `GL_LINEAR` fog is `(end - z) / (end - start)`, and the blend is
/// `fogColor * (1 - factor) + fragment * factor`. With `start = 10`, `end = 30` and a surface
/// at exactly 20 blocks, the factor is a hand-derived one half — no tolerance hiding a
/// half-right falloff.
#[test]
fn linear_fog_blends_the_terrain_fragment_by_the_hand_derived_factor() {
    let Some(mut f) = fixture_or_skip() else {
        return;
    };

    let (unfogged, texel, face) = shoot(&mut f, None, None);
    let lm = Lightmap::generate(LightmapParams::default())
        .sample(face.light[0] / 16, face.light[1] / 16);
    let base = shaded(texel, &face, lm);
    assert_eq!(to_bytes(base), [unfogged[0], unfogged[1], unfogged[2]]);

    let fog = FogState {
        color: glam::Vec3::new(0.8, 0.2, 0.1),
        mode: FogMode::Linear {
            start: 10.0,
            end: 30.0,
        },
    };
    assert_eq!(fog.mode.factor(EYE_DISTANCE), 0.5, "hand-derived factor");

    let (got, texel2, _) = shoot(&mut f, None, Some(fog));
    assert_eq!(texel2, texel);
    let expected = to_bytes(std::array::from_fn(|c| {
        let fog_c = [fog.color.x, fog.color.y, fog.color.z][c];
        fog_c * 0.5 + base[c] * 0.5
    }));
    eprintln!("fogged: got {got:?} expected {expected:?} (unfogged {unfogged:?})");
    for c in 0..3 {
        assert!(
            got[c].abs_diff(expected[c]) <= 1,
            "channel {c}: {got:?} vs {expected:?}"
        );
    }
}

/// Exponential fog, and the two ends of the ramp: past the linear `end` the surface *is* the
/// fog colour, and with fog off the frame is unchanged.
#[test]
fn exponential_fog_and_the_two_degenerate_ends_are_exact() {
    let Some(mut f) = fixture_or_skip() else {
        return;
    };

    let (unfogged, texel, face) = shoot(&mut f, None, None);
    let lm = Lightmap::generate(LightmapParams::default())
        .sample(face.light[0] / 16, face.light[1] / 16);
    let base = shaded(texel, &face, lm);

    // java: GL_EXP, factor = exp(-density * z). density 0.05 at 20 blocks -> exp(-1).
    let fog = FogState {
        color: glam::Vec3::new(0.1, 0.7, 0.3),
        mode: FogMode::Exp { density: 0.05 },
    };
    let factor = (-1.0f32).exp();
    assert!((fog.mode.factor(EYE_DISTANCE) - factor).abs() < 1e-6);
    let (got, _, _) = shoot(&mut f, None, Some(fog));
    let expected = to_bytes(std::array::from_fn(|c| {
        let fog_c = [fog.color.x, fog.color.y, fog.color.z][c];
        fog_c * (1.0 - factor) + base[c] * factor
    }));
    eprintln!("exp fog: got {got:?} expected {expected:?}");
    for c in 0..3 {
        assert!(
            got[c].abs_diff(expected[c]) <= 1,
            "channel {c}: {got:?} vs {expected:?}"
        );
    }

    // Fully fogged: the surface must *equal* the fog colour.
    let opaque = FogState {
        color: glam::Vec3::new(0.25, 0.5, 0.75),
        mode: FogMode::Linear {
            start: 1.0,
            end: 5.0,
        },
    };
    let (fully, _, _) = shoot(&mut f, None, Some(opaque));
    let fog_bytes = to_bytes([0.25, 0.5, 0.75]);
    for c in 0..3 {
        // +-1 only because 0.5 * 255 = 127.5 lands exactly on a rounding tie, which the unorm
        // conversion resolves downward; every other channel is exact.
        assert!(
            fully[c].abs_diff(fog_bytes[c]) <= 1,
            "past the fog end distance the fragment must be the fog colour: {fully:?} vs              {fog_bytes:?}"
        );
    }

    // And `None` is genuinely off, not "fog with a zero colour".
    let (again, _, _) = shoot(&mut f, None, None);
    assert_eq!(again, unfogged);
}

/// The seam the request actually names: S20's own `FogState::compute` reaches terrain.
#[test]
fn a_fog_state_computed_by_s20_reaches_the_terrain_shader() {
    let Some(mut f) = fixture_or_skip() else {
        return;
    };
    // java: `setupFog`'s underwater branch, `GL_EXP` at density 0.1 — the one `FogState::compute`
    // configuration whose ramp covers 20 blocks without inventing a render distance. In air its
    // linear ramp is `farPlane * 0.75 .. farPlane`, which at any sane render distance puts a
    // 20-block surface fully clear.
    let env = mc_render::sky::environment::Environment::default();
    let fog = FogState::compute(
        &env,
        mc_render::sky::fog::FogInputs {
            medium: mc_render::sky::fog::CameraMedium::Water,
            ..mc_render::sky::fog::FogInputs::default()
        },
        1.0,
    );
    // exp(-0.1 * 20) = exp(-2).
    assert!((fog.mode.factor(EYE_DISTANCE) - (-2.0f32).exp()).abs() < 1e-6);
    let factor = fog.mode.factor(EYE_DISTANCE);
    assert!(
        (0.01..0.99).contains(&factor),
        "pick a render distance that puts the surface inside the ramp; factor was {factor}"
    );

    let (unfogged, texel, face) = shoot(&mut f, None, None);
    let lm = Lightmap::generate(LightmapParams::default())
        .sample(face.light[0] / 16, face.light[1] / 16);
    let base = shaded(texel, &face, lm);
    let (got, _, _) = shoot(&mut f, None, Some(fog));
    let expected = to_bytes(std::array::from_fn(|c| {
        let fog_c = [fog.color.x, fog.color.y, fog.color.z][c];
        fog_c * (1.0 - factor) + base[c] * factor
    }));
    eprintln!("computed fog {fog:?} factor {factor}: got {got:?} expected {expected:?}");
    assert_ne!(got, unfogged);
    for c in 0..3 {
        assert!(
            got[c].abs_diff(expected[c]) <= 1,
            "channel {c}: {got:?} vs {expected:?}"
        );
    }
}
