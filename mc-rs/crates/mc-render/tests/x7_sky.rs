//! The Sky mod ("toggle sky, to remove fighting of
//! custom sky with normal sky"), renderer half: `SkyPhase::{vanilla_sky, vanilla_sun_and_moon,
//! vanilla_stars}`.
//!
//! The default (all `true`) is pinned byte for byte by `custom_sky.rs`'s golden test, which
//! renders a default `SkyPhase`. These tests cover the switch turned off.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::DVec3;
use mc_assets::{FolderResourcePack, ResourceManager, ZipResourcePack};
use mc_render::sky::custom::{CustomSkyAssets, CustomSkyRenderer};
use mc_render::sky::dome::SkyPhase;
use mc_render::sky::{Dimension, Environment, Weather};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const SIDE: u32 = 64;

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

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIDE,
        SIDE,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// A square 90-degree view from 80 blocks up (above the overworld horizon at 63).
fn camera(yaw: f32, pitch: f32) -> Camera {
    let mut cam = Camera {
        pos: DVec3::new(0.0, 80.0, 0.0),
        yaw,
        pitch,
        fov: 90.0,
        aspect: 1.0,
        ..Camera::default()
    };
    cam.update();
    cam
}

fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    world_time: i64,
    dimension: Dimension,
    cam: &Camera,
    configure: impl FnOnce(&Gpu, &mut SkyPhase),
) -> Vec<u8> {
    let mut sky = SkyPhase::new(gpu);
    sky.load_textures(gpu, rm);
    sky.environment = Environment {
        world_time,
        dimension,
        weather: Weather::default(),
    };
    configure(gpu, &mut sky);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.1,
        g: 0.2,
        b: 0.3,
        a: 1.0,
    };
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    gpu.read_pixels().expect("readback")
}

fn rgb(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * SIDE + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn uniform(pixels: &[u8]) -> bool {
    let first = rgb(pixels, 0, 0);
    (0..SIDE).all(|y| (0..SIDE).all(|x| rgb(pixels, x, y) == first))
}

/// Pixels that differ between two frames, as `(x, y)`.
fn differences(a: &[u8], b: &[u8]) -> Vec<(u32, u32)> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .filter(|&(x, y)| rgb(a, x, y) != rgb(b, x, y))
        .collect()
}

/// `(label, world time, dimension, yaw, pitch)`.
const VIEWS: [(&str, i64, Dimension, f32, f32); 6] = [
    ("noon, level", 6_000, Dimension::Overworld, 0.0, 0.0),
    ("noon, zenith", 6_000, Dimension::Overworld, 0.0, -89.0),
    (
        "dusk, glow in the west",
        12_800,
        Dimension::Overworld,
        90.0,
        -10.0,
    ),
    ("midnight, zenith", 18_000, Dimension::Overworld, 0.0, -89.0),
    ("noon, looking down", 6_000, Dimension::Overworld, 0.0, 89.0),
    ("the End", 6_000, Dimension::End, 0.0, 0.0),
];

/// Off, nothing vanilla is left: the overworld frame is the fog-colour backdrop alone and the End
/// frame is the clear colour, in every view; the vanilla frames are not flat.
#[test]
fn sky_off_leaves_only_the_fog_colour() {
    let Some(rm) = vanilla_resources() else {
        eprintln!("SKIP: the vanilla jar is not available");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    for (label, time, dimension, yaw, pitch) in VIEWS {
        let cam = camera(yaw, pitch);
        let vanilla = render(&mut gpu, &rm, time, dimension, &cam, |_, _| {});
        let off = render(&mut gpu, &rm, time, dimension, &cam, |_, sky| {
            sky.vanilla_sky = false;
        });
        assert!(
            !uniform(&vanilla),
            "{label}: the vanilla sky draws something"
        );
        assert!(uniform(&off), "{label}: only the fog colour is left");
    }
}

/// The two options take away their own part and nothing else.
#[test]
fn sun_and_moon_and_stars_options_remove_only_their_part() {
    let Some(rm) = vanilla_resources() else {
        eprintln!("SKIP: the vanilla jar is not available");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    let centre = |&(x, y): &(u32, u32)| (16..48).contains(&x) && (16..48).contains(&y);

    // Noon, straight up: the sun sits in the middle of the view.
    let cam = camera(0.0, -89.0);
    let vanilla = render(&mut gpu, &rm, 6_000, Dimension::Overworld, &cam, |_, _| {});
    let no_sun = render(
        &mut gpu,
        &rm,
        6_000,
        Dimension::Overworld,
        &cam,
        |_, sky| {
            sky.vanilla_sun_and_moon = false;
        },
    );
    let gone = differences(&vanilla, &no_sun);
    assert!(gone.len() > 50, "the sun is removed ({} px)", gone.len());
    assert!(gone.iter().all(centre), "only the sun, in the middle");

    // Midnight, straight up: the moon in the middle, stars everywhere.
    let cam = camera(0.0, -89.0);
    let vanilla = render(&mut gpu, &rm, 18_000, Dimension::Overworld, &cam, |_, _| {});
    let no_stars = render(
        &mut gpu,
        &rm,
        18_000,
        Dimension::Overworld,
        &cam,
        |_, sky| {
            sky.vanilla_stars = false;
        },
    );
    let no_moon = render(
        &mut gpu,
        &rm,
        18_000,
        Dimension::Overworld,
        &cam,
        |_, sky| {
            sky.vanilla_sun_and_moon = false;
        },
    );
    let stars = differences(&vanilla, &no_stars);
    assert!(
        stars.iter().any(|p| !centre(p)),
        "stars disappear all over the sky ({} px)",
        stars.len()
    );
    let moon = differences(&vanilla, &no_moon);
    assert!(moon.len() > 20, "the moon is removed ({} px)", moon.len());
    assert!(moon.iter().all(centre), "only the moon, in the middle");
}

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

/// The custom-sky "fighting" itself: a pack's opaque sky box (`blend=replace`), with the vanilla
/// sky on, still has the vanilla sun added on top of it overhead and the dark horizon plane
/// painted over its lower half. With the Sky mod off the frame is the pack's box alone.
#[test]
fn a_custom_sky_stands_alone_with_the_sky_off() {
    let Some(rm) = vanilla_resources() else {
        eprintln!("SKIP: the vanilla jar is not available");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    // The format's 3x2 template: bottom, top, south / west, north, east, one colour each.
    const FACES: [[u8; 3]; 6] = [
        [0, 220, 220],
        [255, 0, 0],
        [255, 0, 255],
        [255, 255, 0],
        [0, 0, 255],
        [0, 255, 0],
    ];
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("pack.mcmeta"),
        r#"{"pack":{"pack_format":1,"description":"x7 sky test"}}"#,
    )
    .unwrap();
    let sky_dir = dir.path().join("assets/minecraft/mcpatcher/sky/world0");
    std::fs::create_dir_all(&sky_dir).unwrap();
    std::fs::write(
        sky_dir.join("sky1.properties"),
        "blend=replace\nrotate=false\n",
    )
    .unwrap();
    let template = png(96, 64, |x, y| {
        let c = FACES[((y / 32) * 3 + x / 32) as usize];
        [c[0], c[1], c[2], 255]
    });
    std::fs::write(sky_dir.join("sky1.png"), template).unwrap();
    let mut pack = ResourceManager::new();
    pack.push_pack(Box::new(
        FolderResourcePack::open(dir.path()).expect("pack"),
    ));
    let assets: Arc<CustomSkyAssets> =
        CustomSkyAssets::load(&gpu, &pack, false).expect("a drawable layer");

    for (label, pitch, face) in [("overhead", -90.0_f32, FACES[1]), ("below", 90.0, FACES[0])] {
        let cam = camera(180.0, pitch);
        let with = |vanilla_sky: bool| {
            let assets = Arc::clone(&assets);
            move |gpu: &Gpu, sky: &mut SkyPhase| {
                sky.custom_sky = Some(CustomSkyRenderer::new(gpu, assets));
                sky.vanilla_sky = vanilla_sky;
            }
        };
        let on = render(&mut gpu, &rm, 6_000, Dimension::Overworld, &cam, with(true));
        let off = render(
            &mut gpu,
            &rm,
            6_000,
            Dimension::Overworld,
            &cam,
            with(false),
        );
        assert_ne!(
            rgb(&on, 32, 32),
            face,
            "{label}: with the vanilla sky on, vanilla draws over the pack's face"
        );
        assert!(
            (0..SIDE).all(|y| (0..SIDE).all(|x| rgb(&off, x, y) == face)),
            "{label}: with the Sky mod off the pack's face is all there is"
        );
    }
}
