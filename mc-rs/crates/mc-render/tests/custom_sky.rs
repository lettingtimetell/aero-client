//! Custom Sky: resource-pack custom sky layers on the GPU.
//!
//! The first test pins the **vanilla** sky: golden FNV-1a hashes of `SkyPhase` frames captured
//! from the tree *before* any custom-sky code existed (HEAD `fbbe730`). With no custom sky
//! attached the phase must keep producing exactly those bytes, which is the "off costs nothing
//! and changes nothing" half of the mod.
//!
//! Everything skips (printing `SKIP:`) without the vanilla jar or a GPU adapter. Point
//! `MC_JARS_DIR` at a `jars/` directory to override the asset location.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use glam::DVec3;
use mc_assets::custom_sky::parse_time;
use mc_assets::{FolderResourcePack, ResourceManager, ZipResourcePack};
use mc_render::sky::custom::{CustomSkyAssets, CustomSkyRenderer};
use mc_render::sky::dome::SkyPhase;
use mc_render::sky::{Dimension, Environment, Weather};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

fn headless(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => {
            gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(gpu)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn camera(yaw: f32, pitch: f32, eye_y: f64) -> Camera {
    let mut cam = Camera {
        pos: DVec3::new(0.0, eye_y, 0.0),
        yaw,
        pitch,
        ..Camera::default()
    };
    cam.update();
    cam
}

/// One vanilla sky frame: the jar's sun and moon, the given time, weather and camera.
fn render_vanilla_sky(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    world_time: i64,
    weather: Weather,
    cam: &Camera,
) -> Vec<u8> {
    let mut sky = SkyPhase::new(gpu);
    sky.load_textures(gpu, rm);
    sky.environment = Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather,
    };
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    gpu.read_pixels().expect("readback")
}

/// `(label, world time, rain, thunder, yaw, pitch, eye y)`.
type GoldenCase = (&'static str, i64, f32, f32, f32, f32, f64);

const GOLDEN_CASES: [GoldenCase; 8] = [
    ("noon, level", 6000, 0.0, 0.0, 0.0, 0.0, 80.0),
    ("sunset, west and up", 12_000, 0.0, 0.0, 90.0, -30.0, 80.0),
    ("dusk, fan and stars", 12_800, 0.0, 0.0, 90.0, -10.0, 80.0),
    ("midnight, zenith", 18_000, 0.0, 0.0, 0.0, -89.0, 80.0),
    (
        "midnight, rain and thunder",
        18_000,
        0.6,
        0.3,
        45.0,
        -45.0,
        80.0,
    ),
    ("dawn, east", 23_200, 0.0, 0.0, -90.0, -15.0, 80.0),
    (
        "night, below the horizon",
        15_000,
        0.0,
        0.0,
        180.0,
        20.0,
        40.0,
    ),
    ("morning, south and down", 1_000, 0.0, 0.0, 0.0, 45.0, 70.0),
];

/// Hashes of [`GOLDEN_CASES`] at 160x90. First captured from HEAD `fbbe730` before the custom
/// sky existed; re-captured on 2026-11-11 after
/// the earlier H7 F3/F4 fixes made the vanilla sky vanilla-exact (stars rotate with the
/// celestial angle; the horizon plane is back-face culled from below). Negative control: with
/// `SkyPhase::rotate_stars` and `cull_planes` both false, the old hashes still match exactly,
/// so those two fixes are the only change.
const GOLDEN_HASHES: [u64; 8] = [
    0x6923_a0f3_033d_bd2d,
    0x854a_e527_6b88_9bab,
    0xdab7_ea5e_f443_b212,
    0x1a3a_a0cc_132a_70e2,
    0xeff6_ec19_71ef_4935,
    0x2ad9_0b37_257d_cfbb,
    0x794a_0753_a7e1_63e0,
    0xc130_b79e_e351_e1e1,
];

/// The vanilla sky is byte-identical to the tree before the custom sky existed. A phase with no
/// custom sky attached (the mod off) takes exactly the old path.
#[test]
fn vanilla_sky_is_byte_identical_to_before_custom_sky() {
    let Some(rm) = vanilla_resources() else {
        eprintln!("SKIP: the vanilla jar is not available");
        return;
    };
    let Some(mut gpu) = headless(160, 90) else {
        return;
    };
    let mut hashes = Vec::new();
    for (label, time, rain, thunder, yaw, pitch, eye_y) in GOLDEN_CASES {
        let weather = Weather {
            rain_strength: rain,
            thunder_strength: thunder,
            last_lightning_bolt: 0,
        };
        let pixels = render_vanilla_sky(&mut gpu, &rm, time, weather, &camera(yaw, pitch, eye_y));
        let hash = fnv1a(&pixels);
        eprintln!("golden {label:32} {hash:#018x}");
        hashes.push(hash);
    }
    if GOLDEN_HASHES.iter().all(|&h| h == 0) {
        eprintln!("CAPTURE MODE: no golden hashes recorded yet");
        return;
    }
    assert_eq!(hashes, GOLDEN_HASHES, "the vanilla sky changed");
}

// ------------------------------------------------------------ synthetic sky packs

fn png(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            rgba.extend_from_slice(&pixel(x, y));
        }
    }
    let mut out = std::io::Cursor::new(Vec::new());
    image::ImageEncoder::write_image(
        image::codecs::png::PngEncoder::new(&mut out),
        &rgba,
        width,
        height,
        image::ExtendedColorType::Rgba8,
    )
    .expect("encode png");
    out.into_inner()
}

/// A folder pack with `files` under `assets/minecraft/`, alone in a resource manager.
fn sky_pack(files: &[(&str, Vec<u8>)]) -> (tempfile::TempDir, ResourceManager) {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("pack.mcmeta"),
        r#"{"pack":{"pack_format":1,"description":"sky test"}}"#,
    )
    .unwrap();
    for (path, bytes) in files {
        let path = dir.path().join("assets/minecraft").join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        FolderResourcePack::open(dir.path()).expect("pack"),
    ));
    (dir, rm)
}

/// One layer `sky1` in `world0` with this properties text and texture.
fn one_layer(properties: &str, texture: Vec<u8>) -> (tempfile::TempDir, ResourceManager) {
    sky_pack(&[
        (
            "mcpatcher/sky/world0/sky1.properties",
            properties.as_bytes().to_vec(),
        ),
        ("mcpatcher/sky/world0/sky1.png", texture),
    ])
}

/// The format's labelled template, one colour per face.
const FACE_COLORS: [(&str, [u8; 3]); 6] = [
    ("bottom", [0, 220, 220]),
    ("top", [255, 0, 0]),
    ("south", [255, 0, 255]),
    ("west", [255, 255, 0]),
    ("north", [0, 0, 255]),
    ("east", [0, 255, 0]),
];
const MARKER: [u8; 3] = [255, 255, 255];

fn face_color(face: &str) -> [u8; 3] {
    FACE_COLORS
        .iter()
        .find(|(name, _)| *name == face)
        .unwrap()
        .1
}

/// The template as a 96x64 PNG (32 px cells). With `markers`, each cell's image top-left
/// quadrant is white, so orientation and mirroring show.
fn template_png(markers: bool) -> Vec<u8> {
    png(96, 64, |x, y| {
        let face = ((y / 32) * 3 + x / 32) as usize;
        let rgb = if markers && x % 32 < 16 && y % 32 < 16 {
            MARKER
        } else {
            FACE_COLORS[face].1
        };
        [rgb[0], rgb[1], rgb[2], 255]
    })
}

/// Only the custom layers, over the frame's clear colour.
struct LayersOnly {
    renderer: CustomSkyRenderer,
    env: Environment,
}

impl RenderPhase for LayersOnly {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.renderer.record(gpu, enc, view, cam, &self.env, 0.0);
    }
}

fn env_at(world_time: i64, rain: f32, thunder: f32) -> Environment {
    Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather: Weather {
            rain_strength: rain,
            thunder_strength: thunder,
            last_lightning_bolt: 0,
        },
    }
}

/// `hh:mm` as a world time (ticks after 06:00 on day 0).
fn at(hhmm: &str) -> i64 {
    i64::from(parse_time(hhmm).expect("time"))
}

/// A square view with a 90 degree FOV: a cube face seen head-on fills it exactly.
fn square_camera(yaw: f32, pitch: f32) -> Camera {
    let mut cam = Camera {
        yaw,
        pitch,
        fov: 90.0,
        aspect: 1.0,
        ..Camera::default()
    };
    cam.update();
    cam
}

fn render_layers(
    gpu: &mut Gpu,
    assets: &Arc<CustomSkyAssets>,
    env: Environment,
    cam: &Camera,
    clear: [f64; 3],
) -> Vec<u8> {
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: clear[0],
        g: clear[1],
        b: clear[2],
        a: 1.0,
    };
    let renderer = CustomSkyRenderer::new(gpu, Arc::clone(assets));
    phases.insert(PhaseOrder::SKY, Box::new(LayersOnly { renderer, env }));
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    gpu.read_pixels().expect("readback")
}

fn load(gpu: &Gpu, rm: &ResourceManager) -> Arc<CustomSkyAssets> {
    CustomSkyAssets::load(gpu, rm, false).expect("the test pack has a drawable layer")
}

fn rgb(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn save_png(name: &str, pixels: &[u8], width: u32, height: u32) {
    if let Some(dir) = std::env::var_os("W3_SHOTS_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("shots dir");
        image::save_buffer(
            dir.join(name),
            pixels,
            width,
            height,
            image::ExtendedColorType::Rgba8,
        )
        .expect("save png");
    }
}

/// Face order and orientation: looking at each side head-on shows that face's colour, with the
/// image's top-left quadrant in the screen's top-left (upright, not mirrored). The top face has
/// south at its image top and the bottom face north, so both are checked facing north.
#[test]
fn every_face_lands_on_its_side_upright_and_unmirrored() {
    let Some(mut gpu) = headless(64, 64) else {
        return;
    };
    let (_dir, rm) = one_layer("blend=replace\nrotate=false\n", template_png(true));
    let assets = load(&gpu, &rm);
    for (face, yaw, pitch) in [
        ("north", 180.0, 0.0),
        ("east", -90.0, 0.0),
        ("south", 0.0, 0.0),
        ("west", 90.0, 0.0),
        ("top", 180.0, -90.0),
        ("bottom", 180.0, 90.0),
    ] {
        let pixels = render_layers(
            &mut gpu,
            &assets,
            env_at(6000, 0.0, 0.0),
            &square_camera(yaw, pitch),
            [0.0; 3],
        );
        save_png(&format!("synthetic_face_{face}.png"), &pixels, 64, 64);
        assert_eq!(
            rgb(&pixels, 64, 16, 16),
            MARKER,
            "{face}: the image's top-left quadrant is the screen's top-left"
        );
        for (x, y) in [(48, 16), (16, 48), (48, 48), (40, 40), (60, 3), (3, 60)] {
            assert_eq!(
                rgb(&pixels, 64, x, y),
                face_color(face),
                "{face} at ({x}, {y})"
            );
        }
    }
}

/// `startFadeIn=18:00 endFadeIn=19:00 endFadeOut=6:00` (fade-out derived from 05:00): 0, a
/// linear rise, 1 through the night, a linear fall, 0. `add` of a grey 100 over black shows the
/// brightness directly. Default weather (clear): rain fades it by `1 - rain`.
#[test]
fn a_layer_fades_at_its_times_and_with_the_rain() {
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let (_dir, rm) = one_layer(
        "startFadeIn=18:00\nendFadeIn=19:00\nendFadeOut=6:00\nrotate=false\n",
        png(3, 2, |_, _| [100, 100, 100, 255]),
    );
    let assets = load(&gpu, &rm);
    let cam = square_camera(180.0, 0.0);
    let mut check = |time: i64, rain: f32, thunder: f32, expected: i32, what: &str| {
        let pixels = render_layers(
            &mut gpu,
            &assets,
            env_at(time, rain, thunder),
            &cam,
            [0.0; 3],
        );
        let got = i32::from(rgb(&pixels, 16, 8, 8)[0]);
        assert!(
            (got - expected).abs() <= 1,
            "{what}: {got}, expected {expected}"
        );
    };
    for (hhmm, expected) in [
        ("12:00", 0),
        ("17:59", 0),
        ("18:00", 0),
        ("18:30", 50),
        ("18:45", 75),
        ("19:00", 100),
        ("0:00", 100),
        ("5:00", 100),
        ("5:30", 50),
        ("6:00", 0),
    ] {
        check(at(hhmm), 0.0, 0.0, expected, hhmm);
    }
    // The same times on a later day.
    check(at("18:30") + 24_000 * 9, 0.0, 0.0, 50, "day 9, 18:30");
    // Rain: clear-weather layers fade with it.
    check(at("0:00"), 0.5, 0.0, 50, "midnight, half rain");
    check(at("0:00"), 1.0, 0.6, 0, "midnight, a storm");
}

/// Every `blend=` mode against its GL blend function, over a known frame colour C with a texel
/// T of alpha A, at full brightness and at half (18:30 in an 18:00-19:00 fade-in).
#[test]
fn every_blend_mode_matches_its_gl_blend_function() {
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    // Exact in 8 bits: C = (102, 51, 153), T = (204, 102, 51), A = 153.
    let c = [0.4f32, 0.2, 0.6];
    let t = [0.8f32, 0.4, 0.2];
    let a = 0.6f32;
    let expected = |mode: &str, c: f32, t: f32, b: f32| -> f32 {
        let v = match mode {
            "alpha" => t * a * b + c * (1.0 - a * b),
            "add" => c + t * a * b,
            "subtract" => t * b * (1.0 - c),
            "multiply" => t * b * c + c * (1.0 - a * b),
            "dodge" => t * b + c,
            "burn" => c * (1.0 - t * b),
            "screen" => t * b + c * (1.0 - t * b),
            "overlay" => 2.0 * t * b * c,
            "replace" => {
                if a * b > 0.1 {
                    t
                } else {
                    c
                }
            }
            _ => unreachable!(),
        };
        v.clamp(0.0, 1.0)
    };
    let cam = square_camera(180.0, 0.0);
    for mode in [
        "alpha", "add", "subtract", "multiply", "dodge", "burn", "screen", "overlay", "replace",
    ] {
        let (_dir, rm) = one_layer(
            &format!(
                "blend={mode}\nrotate=false\nstartFadeIn=18:00\nendFadeIn=19:00\nendFadeOut=6:00"
            ),
            png(3, 2, |_, _| [204, 102, 51, 153]),
        );
        let assets = load(&gpu, &rm);
        // 0:00 is in the hold (b = 1), 18:30 half way up, 18:06 one tenth (replace's cutoff).
        for (hhmm, b) in [("0:00", 1.0f32), ("18:30", 0.5), ("18:06", 0.1)] {
            let pixels = render_layers(
                &mut gpu,
                &assets,
                env_at(at(hhmm), 0.0, 0.0),
                &cam,
                [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])],
            );
            let got = rgb(&pixels, 16, 8, 8);
            for ch in 0..3 {
                let want = expected(mode, c[ch], t[ch], b) * 255.0;
                assert!(
                    (f32::from(got[ch]) - want).abs() <= 1.5,
                    "{mode} at b = {b}: channel {ch} is {}, expected {want:.1}",
                    got[ch]
                );
            }
        }
    }
}

/// A rotating layer turns like the sun about the default (south) axis: at noon looking west
/// shows the west face, at 18:00 (celestial angle 0.2155, 78 degrees) the top face has swung
/// down into the west. About `axis=0 1 0` the same turn brings the north face round to the west.
#[test]
fn a_rotating_layer_turns_like_the_sun_about_its_axis() {
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let west = square_camera(90.0, 0.0);
    let (_dir, rm) = one_layer("blend=replace\n", template_png(false));
    let assets = load(&gpu, &rm);
    let px = |gpu: &mut Gpu, assets: &Arc<CustomSkyAssets>, time: i64| {
        rgb(
            &render_layers(gpu, assets, env_at(time, 0.0, 0.0), &west, [0.0; 3]),
            16,
            8,
            8,
        )
    };
    assert_eq!(px(&mut gpu, &assets, at("12:00")), face_color("west"));
    assert_eq!(px(&mut gpu, &assets, at("18:00")), face_color("top"));
    // Midnight: half a turn, the bottom face is overhead and the east face in the west.
    assert_eq!(px(&mut gpu, &assets, at("0:00")), face_color("east"));

    let (_dir, rm) = one_layer("blend=replace\naxis=0 1 0\n", template_png(false));
    let assets = load(&gpu, &rm);
    assert_eq!(px(&mut gpu, &assets, at("18:00")), face_color("north"));
    // `rotate=false` never turns.
    let (_dir, rm) = one_layer("blend=replace\nrotate=false\n", template_png(false));
    let assets = load(&gpu, &rm);
    assert_eq!(px(&mut gpu, &assets, at("18:00")), face_color("west"));
}

/// Count pixels clearly above black: at midnight with no sun or moon textures, the stars.
fn lit_pixels(pixels: &[u8]) -> usize {
    pixels
        .chunks(4)
        .filter(|p| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]) > 60)
        .count()
}

/// While the world has custom layers the vanilla stars are not drawn (the format's "disable
/// default stars"), even when no layer is visible right now; an End-only sky keeps them.
#[test]
fn a_custom_sky_replaces_the_vanilla_stars() {
    // 256 px, as `s20_sky`'s star test: smaller frames make the stars sub-pixel.
    let Some(mut gpu) = headless(256, 256) else {
        return;
    };
    let up = camera(0.0, -90.0, 80.0);
    let render = |gpu: &mut Gpu, custom: Option<&Arc<CustomSkyAssets>>| {
        let mut sky = SkyPhase::new(gpu);
        sky.environment = env_at(at("0:00"), 0.0, 0.0);
        sky.custom_sky = custom.map(|assets| CustomSkyRenderer::new(gpu, Arc::clone(assets)));
        let mut phases = PhaseList::new();
        phases.insert(PhaseOrder::SKY, Box::new(sky));
        let frame = gpu.acquire().expect("acquire");
        phases.render(gpu, frame.view(), &up, 0.0);
        gpu.read_pixels().expect("readback")
    };
    let vanilla = render(&mut gpu, None);
    assert!(lit_pixels(&vanilla) > 50, "vanilla stars at midnight");

    // A storm-only layer: present, invisible in clear weather.
    let (_dir, rm) = one_layer("weather=thunder\n", png(3, 2, |_, _| [255, 255, 255, 255]));
    let storm = load(&gpu, &rm);
    let custom = render(&mut gpu, Some(&storm));
    assert_eq!(
        lit_pixels(&custom),
        0,
        "no vanilla stars under a custom sky"
    );

    // A sky for the End only leaves the overworld's stars alone.
    let (_dir, rm) = sky_pack(&[
        ("optifine/sky/world1/sky1.properties", b"blend=add".to_vec()),
        (
            "optifine/sky/world1/sky1.png",
            png(3, 2, |_, _| [255, 255, 255, 255]),
        ),
    ]);
    let end_only = load(&gpu, &rm);
    assert_eq!(render(&mut gpu, Some(&end_only)), vanilla);
}

/// `world1` layers draw in the End only, `world0` layers in the overworld only; nothing in the
/// Nether.
#[test]
fn layers_draw_only_in_their_own_world() {
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let (_dir, rm) = sky_pack(&[
        (
            "mcpatcher/sky/world0/sky1.properties",
            b"rotate=false".to_vec(),
        ),
        (
            "mcpatcher/sky/world0/sky1.png",
            png(3, 2, |_, _| [0, 90, 0, 255]),
        ),
        (
            "mcpatcher/sky/world1/sky1.properties",
            b"rotate=false".to_vec(),
        ),
        (
            "mcpatcher/sky/world1/sky1.png",
            png(3, 2, |_, _| [90, 0, 0, 255]),
        ),
    ]);
    let assets = load(&gpu, &rm);
    assert_eq!(assets.stats().layers, 2);
    let cam = square_camera(180.0, 0.0);
    for (dimension, expected) in [
        (Dimension::Overworld, [0, 90, 0]),
        (Dimension::End, [90, 0, 0]),
        (Dimension::Nether, [0, 0, 0]),
    ] {
        let env = Environment {
            dimension,
            ..env_at(6000, 0.0, 0.0)
        };
        let pixels = render_layers(&mut gpu, &assets, env, &cam, [0.0; 3]);
        assert_eq!(rgb(&pixels, 16, 8, 8), expected, "{dimension:?}");
    }
}

// ------------------------------------------------------------ the reference pack

/// The reference pack (never checked in): `MC_REFERENCE_PACK`, or else the only folder in the
/// workspace's `resourcepacks/`.
fn users_pack() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_REFERENCE_PACK") {
        return Some(dir.into());
    }
    let packs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resourcepacks");
    let mut dirs = std::fs::read_dir(packs)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir());
    let only = dirs.next()?;
    dirs.next().is_none().then_some(only)
}

/// The jar under the reference pack (read-only), as the client stacks them.
fn users_resources() -> Option<ResourceManager> {
    let mut rm = vanilla_resources()?;
    rm.push_pack(Box::new(FolderResourcePack::open(users_pack()?).ok()?));
    Some(rm)
}

/// The whole sky phase (vanilla sun and moon from the jar) with the user's custom sky.
fn render_users_sky(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    assets: Option<&Arc<CustomSkyAssets>>,
    world_time: i64,
    cam: &Camera,
    smooth: bool,
) -> (Vec<u8>, usize) {
    let mut sky = SkyPhase::new(gpu);
    sky.load_textures(gpu, rm);
    sky.environment = env_at(world_time, 0.0, 0.0);
    sky.custom_sky = assets.map(|assets| {
        let mut renderer = CustomSkyRenderer::new(gpu, Arc::clone(assets));
        renderer.smooth = smooth;
        renderer
    });
    let sky = std::rc::Rc::new(std::cell::RefCell::new(sky));
    struct Shared(std::rc::Rc<std::cell::RefCell<SkyPhase>>);
    impl RenderPhase for Shared {
        fn record(
            &mut self,
            gpu: &Gpu,
            enc: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            cam: &Camera,
        ) {
            self.0.borrow_mut().record(gpu, enc, view, cam);
        }
    }
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::SKY, Box::new(Shared(std::rc::Rc::clone(&sky))));
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    let drawn = sky
        .borrow()
        .custom_sky
        .as_ref()
        .map_or(0, CustomSkyRenderer::last_draw_count);
    (gpu.read_pixels().expect("readback"), drawn)
}

/// The reference pack: 21 drawable layers from 15 textures load and draw. Renders it at
/// the times of day that matter (night stars, the shooting star after midnight, sunrise, the
/// day box, the sunset boxes) and, with `W3_SHOTS_DIR` set, saves each frame and the vanilla
/// sky beside it.
#[test]
fn the_users_pack_draws_its_sky_through_the_day() {
    let Some(rm) = users_resources() else {
        eprintln!("SKIP: the vanilla jar or the reference pack is not available");
        return;
    };
    let (width, height) = (960, 540);
    let Some(mut gpu) = headless(width, height) else {
        return;
    };
    let started = Instant::now();
    let assets = CustomSkyAssets::load(&gpu, &rm, false).expect("the reference pack has a sky");
    eprintln!(
        "loaded in {:.0} ms (test profile): {:?}",
        started.elapsed().as_secs_f32() * 1000.0,
        assets.stats()
    );
    assert_eq!(assets.stats().layers, 21);
    assert_eq!(assets.stats().textures, 15);
    assert_eq!(assets.stats().skipped, 26);

    let view = |yaw: f32, pitch: f32| {
        let mut cam = Camera {
            yaw,
            pitch,
            fov: 90.0,
            aspect: width as f32 / height as f32,
            pos: DVec3::new(0.0, 80.0, 0.0),
            ..Camera::default()
        };
        cam.update();
        cam
    };
    let shots = [
        ("0000_midnight_up", "0:00", 180.0, -55.0),
        ("0003_shooting_star_up", "0:03", 180.0, -55.0),
        ("0300_night_north", "3:00", 180.0, -15.0),
        ("0450_dawn_east", "4:50", -90.0, -10.0),
        ("0530_sunrise_east", "5:30", -90.0, -10.0),
        ("0600_sunrise_east", "6:00", -90.0, -10.0),
        ("0700_morning_east", "7:00", -90.0, -10.0),
        ("1200_noon_north", "12:00", 180.0, -15.0),
        ("1730_evening_west", "17:30", 90.0, -10.0),
        ("1800_sunset_west", "18:00", 90.0, -10.0),
        ("1820_sunset_west", "18:20", 90.0, -10.0),
        ("1840_dusk_west", "18:40", 90.0, -10.0),
        ("1900_dusk_west", "19:00", 90.0, -10.0),
        ("2000_night_up", "20:00", 0.0, -55.0),
        ("2200_night_south", "22:00", 0.0, -20.0),
    ];
    for (name, hhmm, yaw, pitch) in shots {
        let cam = view(yaw, pitch);
        let (custom, drawn) = render_users_sky(&mut gpu, &rm, Some(&assets), at(hhmm), &cam, false);
        let (vanilla, _) = render_users_sky(&mut gpu, &rm, None, at(hhmm), &cam, false);
        eprintln!("{hhmm:>5}: {drawn:2} layers drawn ({name})");
        save_png(&format!("pack_{name}.png"), &custom, width, height);
        save_png(&format!("vanilla_{name}.png"), &vanilla, width, height);
        assert!(drawn > 0, "{hhmm}: the pack has layers at every hour");
        assert_ne!(custom, vanilla, "{hhmm}: the custom sky shows");
    }
    // Nearest (the default) and smooth sampling, for the look.
    let cam = view(90.0, -10.0);
    let (smooth, _) = render_users_sky(&mut gpu, &rm, Some(&assets), at("18:20"), &cam, true);
    save_png("pack_1820_sunset_west_smooth.png", &smooth, width, height);
    let cam = view(180.0, -55.0);
    let (smooth, _) = render_users_sky(&mut gpu, &rm, Some(&assets), at("0:00"), &cam, true);
    save_png("pack_0000_midnight_up_smooth.png", &smooth, width, height);
}

/// How many of the user's layers draw at each hour (printed for the report).
#[test]
fn the_users_pack_active_layers_by_hour() {
    let Some(rm) = users_resources() else {
        eprintln!("SKIP: the vanilla jar or the reference pack is not available");
        return;
    };
    let Some(mut gpu) = headless(32, 32) else {
        return;
    };
    let assets = CustomSkyAssets::load(&gpu, &rm, false).expect("the reference pack has a sky");
    let cam = camera(0.0, -30.0, 80.0);
    let mut most = 0;
    let mut line = String::new();
    for hour in 0..24 {
        let time = at(&format!("{hour}:00"));
        let (_, drawn) = render_users_sky(&mut gpu, &rm, Some(&assets), time, &cam, false);
        line.push_str(&format!("{hour:02}h:{drawn} "));
        most = most.max(drawn);
    }
    for hhmm in ["18:20", "18:30", "5:05", "0:03"] {
        let (_, drawn) = render_users_sky(&mut gpu, &rm, Some(&assets), at(hhmm), &cam, false);
        line.push_str(&format!("{hhmm}:{drawn} "));
        most = most.max(drawn);
    }
    eprintln!("active layers: {line}");
    assert!(most <= 21);
}

/// Frame cost at 1920x1080, vanilla sky vs the user's custom sky (provisional when other builds
/// run). `cargo test -p mc-render --test custom_sky -- --ignored --nocapture cost`
#[test]
#[ignore = "timing; run by hand"]
fn cost_of_the_users_sky_at_1080p() {
    let Some(rm) = users_resources() else {
        eprintln!("SKIP: the vanilla jar or the reference pack is not available");
        return;
    };
    let (width, height) = (1920, 1080);
    let Some(mut gpu) = headless(width, height) else {
        return;
    };
    let assets = CustomSkyAssets::load(&gpu, &rm, false).expect("the reference pack has a sky");
    let mut cam = Camera {
        yaw: 90.0,
        pitch: -20.0,
        fov: 90.0,
        aspect: width as f32 / height as f32,
        pos: DVec3::new(0.0, 80.0, 0.0),
        ..Camera::default()
    };
    cam.update();
    for (label, time) in [
        ("midnight", at("0:00")),
        ("18:20", at("18:20")),
        ("noon", at("12:00")),
    ] {
        for custom in [false, true] {
            let mut sky = SkyPhase::new(&gpu);
            sky.load_textures(&gpu, &rm);
            sky.environment = env_at(time, 0.0, 0.0);
            sky.custom_sky = custom.then(|| CustomSkyRenderer::new(&gpu, Arc::clone(&assets)));
            let mut phases = PhaseList::new();
            phases.insert(PhaseOrder::SKY, Box::new(sky));
            let frames = 300;
            let mut run = |n: u32| {
                let started = Instant::now();
                for _ in 0..n {
                    let frame = gpu.acquire().expect("acquire");
                    phases.render(&gpu, frame.view(), &cam, 0.0);
                    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                }
                started.elapsed().as_secs_f64() * 1000.0 / f64::from(n)
            };
            run(30);
            let ms = run(frames);
            eprintln!(
                "{label:>8} custom={custom:5}: {ms:.3} ms/frame (sky phase only, {frames} frames)"
            );
        }
    }
}
