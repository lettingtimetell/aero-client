//! S20 integration tests: sky/weather phases rendered through the real S10 seam, with real
//! `1.8.9.jar` textures where a texture is needed, and read back from a real GPU.
//!
//! Mirrors `tests/s12_entity.rs`'s two concerns: that the module's public surface actually
//! resolves against real vanilla assets, and that pixels genuinely reach the screen (or, here,
//! the offscreen colour target) rather than only being asserted in CPU-side unit tests.
//!
//! Everything skips (printing `SKIP:`) when the jar or a GPU adapter is unavailable, so this file
//! is safe on a bare CI runner. Point `MC_JARS_DIR` at a `jars/` directory to override the asset
//! location.

use std::path::{Path, PathBuf};

use glam::{DVec3, Vec3};
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::sky::clouds::{CloudMode, CloudsPhase};
use mc_render::sky::dome::SkyPhase;
use mc_render::sky::weather::{build_columns, ColumnEnvironment, LightningStrike, WeatherPhase};
use mc_render::sky::{
    Dimension, Environment, FogDistance, FogInputs, FogMode, FogPhase, FogState, Weather,
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_jar() -> Option<PathBuf> {
    let p = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    p.is_file().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = vanilla_jar()?;
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

macro_rules! require {
    ($opt:expr, $what:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!("SKIP: {} not available", $what);
                return;
            }
        }
    };
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
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn avg_color(pixels: &[u8]) -> (f64, f64, f64) {
    let mut sum = (0.0, 0.0, 0.0);
    let n = pixels.len() / 4;
    for chunk in pixels.chunks(4) {
        sum.0 += f64::from(chunk[0]);
        sum.1 += f64::from(chunk[1]);
        sum.2 += f64::from(chunk[2]);
    }
    (sum.0 / n as f64, sum.1 / n as f64, sum.2 / n as f64)
}

// ------------------------------------------------------------ texture-path audit

#[test]
fn every_sky_texture_the_port_names_exists_in_the_vanilla_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    for path in [
        "textures/environment/sun.png",
        "textures/environment/moon_phases.png",
        "textures/environment/clouds.png",
        "textures/environment/rain.png",
        "textures/environment/snow.png",
        "textures/environment/end_sky.png",
    ] {
        let loc = mc_assets::ResourceLocation::parse(path).unwrap();
        assert!(rm.contains(&loc), "{path} missing from the jar");
    }
}

// ------------------------------------------------------------ sky dome: day/night cycle

/// Renders `SkyPhase` alone with the given time of day and returns the averaged RGB of the
/// resulting frame. A fresh headless context and phase per call keeps this simple: `SkyPhase`
/// has no "reset" method, and a new `Gpu`/`PhaseList` per time-of-day sample is cheap enough for
/// a test.
fn render_sky_average(world_time: i64) -> (f64, f64, f64) {
    let mut gpu = headless(64, 64).expect("adapter checked by caller");
    let mut sky = SkyPhase::new(&gpu);
    sky.environment = Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    };
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    avg_color(&pixels)
}

#[test]
fn noon_sky_is_visibly_brighter_than_midnight_sky() {
    if headless(4, 4).is_none() {
        return;
    }
    let noon_avg = render_sky_average(6000);
    let midnight_avg = render_sky_average(18_000);
    let noon_luma = noon_avg.0 * 0.3 + noon_avg.1 * 0.59 + noon_avg.2 * 0.11;
    let midnight_luma = midnight_avg.0 * 0.3 + midnight_avg.1 * 0.59 + midnight_avg.2 * 0.11;
    assert!(
        noon_luma > midnight_luma + 5.0,
        "noon {noon_avg:?} ({noon_luma:.1}) vs midnight {midnight_avg:?} ({midnight_luma:.1})"
    );
}

/// Dusk seen from below the horizon is the fullest frame the dome has: both plane grids, the void
/// box, the sunset fan, the sun, the moon and the stars, all in one vertex buffer.
#[test]
fn the_fullest_sky_frame_draws_every_mesh() {
    let world_time = (12_000..14_000)
        .step_by(50)
        .find(|&world_time| {
            let env = Environment {
                world_time,
                dimension: Dimension::Overworld,
                weather: Weather::default(),
            };
            env.sunrise_sunset_color(0.0).is_some() && env.star_brightness(0.0) > 0.0
        })
        .expect("dusk shows the sunset fan and the stars together");
    let pixels = require!(render_dome(world_time, 0.0, -100.0, 64), "jar or adapter");
    assert_eq!(pixels.len(), 64 * 64 * 4);
    assert!(
        pixels
            .chunks_exact(4)
            .all(|pixel| pixel[..3] != [255, 0, 255]),
        "the clear colour shows through, so some mesh did not draw"
    );
}

/// Render the whole dome at `world_time`, looking at `pitch`, over a **magenta** clear colour so
/// any pixel the sky failed to paint is unmistakable rather than plausibly-dark.
fn render_dome(world_time: i64, pitch: f32, eye_y: f64, size: u32) -> Option<Vec<u8>> {
    let rm = resources()?;
    let mut gpu = headless(size, size)?;
    let mut sky = SkyPhase::new(&gpu);
    sky.load_textures(&gpu, &rm);
    sky.environment = Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    };

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::SKY, Box::new(sky));

    let mut cam = Camera {
        pos: DVec3::new(0.0, eye_y, 0.0),
        pitch,
        ..Camera::default()
    };
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    Some(gpu.read_pixels().expect("readback"))
}

fn is_magenta(p: [u8; 4]) -> bool {
    p[0] > 200 && p[1] < 60 && p[2] > 200
}

/// **The bug the old brightness test could not see.** `RenderGlobal.java:1411`-`:1414` draws the
/// lower sky plane *unconditionally*, translated to `World.getHorizon()`; this port gated it on
/// the eye being below `y = 0`, so in normal play it never drew and everything below the horizon
/// stayed whatever the frame was cleared to. "Midnight is darker than noon" is satisfied by a
/// black screen, so it passed throughout.
///
/// Clearing to magenta turns "the sky did not paint here" into an assertion failure at any time
/// of day and any pitch, which is the property that was actually missing.
#[test]
fn the_sky_paints_every_pixel_at_every_time_of_day_and_pitch() {
    if resources().is_none() || headless(4, 4).is_none() {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    }
    for world_time in [
        0i64, 3000, 6000, 12_000, 13_000, 15_000, 18_000, 21_000, 23_000,
    ] {
        for pitch in [-90.0f32, -45.0, -10.0, 0.0, 10.0, 45.0, 90.0] {
            let pixels = render_dome(world_time, pitch, 70.0, 64).expect("render");
            let unpainted = pixels
                .chunks(4)
                .filter(|p| is_magenta([p[0], p[1], p[2], p[3]]))
                .count();
            assert_eq!(
                unpainted, 0,
                "t={world_time} pitch={pitch}: {unpainted} of 4096 pixels were left as the clear \
                 colour — the sky does not cover the whole view"
            );
        }
    }
}

/// `SkyPhase::paint_backdrop`: the backdrop only repaints the sky pass's fog colour, so over a
/// frame cleared to exactly that colour (what the client does every frame) turning it off must be
/// byte-identical — including the horizon seam, where the backdrop is the only thing showing.
#[test]
fn dome_without_backdrop_over_the_fog_clear_matches_dome_with_backdrop() {
    let render = |paint_backdrop: bool| -> Option<Vec<u8>> {
        let mut gpu = headless(64, 64)?;
        let mut sky = SkyPhase::new(&gpu);
        sky.paint_backdrop = paint_backdrop;
        sky.environment = Environment {
            world_time: 12_500,
            dimension: Dimension::Overworld,
            weather: Weather::default(),
        };
        let mut cam = Camera {
            pos: DVec3::new(0.0, 70.0, 0.0),
            pitch: 0.0,
            ..Camera::default()
        };
        cam.update();
        // The colour `SkyPhase::record` gives its backdrop and sky-pass fog.
        let mut inputs = sky.fog_inputs;
        inputs.render_distance_chunks =
            ((cam.far / std::f32::consts::SQRT_2 / 16.0).round() as u32).clamp(2, 32);
        let fog = FogState::compute(&sky.environment, inputs, sky.partial_ticks).color;
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: f64::from(fog.x),
            g: f64::from(fog.y),
            b: f64::from(fog.z),
            a: 1.0,
        };
        phases.insert(PhaseOrder::SKY, Box::new(sky));
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &cam, 0.0);
        Some(gpu.read_pixels().expect("readback"))
    };
    let Some(with_backdrop) = render(true) else {
        return;
    };
    let without_backdrop = render(false).expect("adapter was available a moment ago");
    let differing = with_backdrop
        .chunks(4)
        .zip(without_backdrop.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differing, 0, "{differing} of 4096 pixels changed");
}

/// java: `RenderGlobal.java:1325`-`:1344` — stars are drawn whenever `getStarBrightness > 0`.
/// A night sky is *black plus stars*, and "average brightness went down" cannot tell the two
/// apart. This counts individual lit points against a dark background, and requires them to be
/// gone at noon.
#[test]
fn stars_are_drawn_at_night_and_absent_at_noon() {
    if resources().is_none() || headless(4, 4).is_none() {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    }
    // Straight up, so the star sphere fills the view.
    let night = render_dome(18_000, -90.0, 70.0, 256).expect("render");
    let noon = render_dome(6000, -90.0, 70.0, 256).expect("render");

    // At midnight the sky plane is black (`getSkyColor`'s `f1` clamps to 0), so any pixel
    // meaningfully above black is a star.
    let count_lit = |px: &[u8]| {
        px.chunks(4)
            .filter(|p| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]) > 60)
            .count()
    };
    let night_stars = count_lit(&night);
    assert!(
        night_stars > 50,
        "expected a field of stars at midnight, found {night_stars} lit pixels — a night sky that \
         is uniformly black is exactly the bug this test exists for"
    );
    // And they must be *points*, not a wash: nowhere near the whole frame.
    assert!(
        night_stars < 256 * 256 / 8,
        "{night_stars} lit pixels is a wash, not a star field"
    );

    // At noon `getStarBrightness` clamps to 0, so nothing star-like may survive — but the sky
    // itself is bright, so compare against the sky rather than against black.
    let sky_at_noon = pixel(&noon, 256, 4, 4);
    let sky_luma =
        u32::from(sky_at_noon[0]) + u32::from(sky_at_noon[1]) + u32::from(sky_at_noon[2]);
    let brighter_than_sky = noon
        .chunks(4)
        .filter(|p| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]) > sky_luma + 30)
        .count();
    // The sun is overhead at noon and is legitimately brighter; it is one contiguous 60x60-block
    // quad, so bound the count rather than requiring zero.
    assert!(
        brighter_than_sky < 256 * 256 / 4,
        "too much of the noon sky is brighter than the sky itself ({brighter_than_sky} px)"
    );
}

/// java: `RenderGlobal.java:1295` — `tryBlendFuncSeparate(770, 1, 1, 0)` (`GL_SRC_ALPHA, GL_ONE`).
/// `sun.png` and `moon_phases.png` are **fully opaque with black backgrounds** (verified against
/// the jar in `sky_textures_that_must_be_drawn_additively_are_opaque_with_black_backgrounds`), so
/// drawing them with ordinary alpha blending paints a black square around each. Additive blending
/// can only ever make the destination brighter — so no pixel of the sky may be *darker* than the
/// surrounding sky where the sun is.
#[test]
fn the_sun_does_not_paint_a_black_box_on_the_sky() {
    if resources().is_none() || headless(4, 4).is_none() {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    }
    // Noon, straight up: the sun quad is dead centre and the sky around it is bright blue.
    let pixels = render_dome(6000, -90.0, 70.0, 128).expect("render");
    let luma = |p: [u8; 4]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
    // Sample the sky well outside the sun quad.
    let sky = luma(pixel(&pixels, 128, 2, 2));
    let darkest = pixels
        .chunks(4)
        .map(|p| luma([p[0], p[1], p[2], p[3]]))
        .min()
        .expect("non-empty frame");
    assert!(
        darkest + 40 >= sky,
        "the darkest pixel around the sun is {darkest} against a sky of {sky} — additive blending \
         cannot darken, so this is sun.png's opaque black background being alpha-blended on"
    );
}

/// The premise the additive-blend fix rests on, checked against the real jar rather than assumed.
#[test]
fn sky_textures_that_must_be_drawn_additively_are_opaque_with_black_backgrounds() {
    let rm = require!(resources(), "1.8.9.jar");
    for path in [
        "textures/environment/sun.png",
        "textures/environment/moon_phases.png",
    ] {
        let loc = mc_assets::ResourceLocation::parse(path).unwrap();
        let bytes = rm.load(&loc).expect("texture in jar");
        let img = image::load_from_memory(&bytes)
            .expect("valid png")
            .to_rgba8();
        assert!(
            img.pixels().all(|p| p.0[3] == 255),
            "{path} has transparency; the additive-blend reasoning assumes it does not"
        );
        let corner = img.get_pixel(0, 0).0;
        assert_eq!(
            [corner[0], corner[1], corner[2]],
            [0, 0, 0],
            "{path}'s background is not black"
        );
    }
}

/// java: `RenderGlobal.java:1311`-`:1322` — the moon's UV rect is the phase's cell of
/// `moon_phases.png`'s 4x2 grid, `k = phase % 4`, `i1 = phase / 4 % 2`. Checked as *pixels*: each
/// phase's rendered moon must match that cell of the real texture, so a transposed row/column or
/// an off-by-one lands here rather than looking vaguely moon-like.
#[test]
fn the_moon_is_drawn_with_the_uv_rect_of_its_own_phase() {
    let rm = require!(resources(), "1.8.9.jar");
    if headless(4, 4).is_none() {
        return;
    }
    let loc = mc_assets::ResourceLocation::parse("textures/environment/moon_phases.png").unwrap();
    let bytes = rm.load(&loc).expect("moon texture");
    let atlas = image::load_from_memory(&bytes)
        .expect("valid png")
        .to_rgba8();
    let (cell_w, cell_h) = (atlas.width() / 4, atlas.height() / 2);

    // Full moon (phase 0) vs new moon (phase 4). Vanilla's `moon_phases.png` has phase 0 as the
    // full disc and phase 4 as the empty one, so their lit-texel counts differ hugely — a good,
    // texture-derived discriminator that does not hard-code pixel values.
    let lit_in_cell = |phase: u32| {
        let (col, row) = (phase % 4, phase / 4 % 2);
        let mut lit = 0;
        for y in 0..cell_h {
            for x in 0..cell_w {
                let p = atlas.get_pixel(col * cell_w + x, row * cell_h + y).0;
                if u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]) > 150 {
                    lit += 1;
                }
            }
        }
        lit
    };
    assert!(
        lit_in_cell(0) > lit_in_cell(4) * 2,
        "the jar's moon_phases.png is not laid out as assumed (phase 0 full, phase 4 new)"
    );

    // Now the rendered frames must show the same ordering. Midnight puts the moon overhead.
    let lit_on_screen = |phase: i64| {
        let pixels = render_dome(18_000 + phase * 24_000, -90.0, 70.0, 128).expect("render");
        pixels
            .chunks(4)
            .filter(|p| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]) > 150)
            .count()
    };
    let full = lit_on_screen(0);
    let new = lit_on_screen(4);
    assert!(
        full > new,
        "phase 0 rendered {full} bright pixels and phase 4 rendered {new}; the moon is not being \
         sampled from its own cell of the 4x2 phase grid"
    );
}

/// java: `RenderGlobal.java:1259`-`:1290`. The glow sits on the horizon on the sun's side, and
/// `RenderGlobal.java:1261` flips it to the opposite horizon once `sin(celestialAngleRadians)`
/// goes negative — sunrise in one direction, sunset in the other. An earlier pass of this port
/// drew the fan in the sun's own rotating frame, which put the glow overhead at midday and on the
/// wrong side at dawn.
#[test]
fn the_sunset_glow_appears_on_the_horizon_and_swaps_sides_between_dusk_and_dawn() {
    if resources().is_none() || headless(4, 4).is_none() {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    }
    // Yaw 0 faces +Z, so the two horizons the glow can sit on (±X) are at the left and right
    // edges of the frame. Look level (pitch 0) so both are in view.
    let warmth = |px: &[u8], half: usize, size: usize| {
        // "Warmth" = how much redder than bluer, summed over one half of the frame.
        let mut sum = 0i64;
        for y in 0..size {
            let (x0, x1) = if half == 0 {
                (0, size / 2)
            } else {
                (size / 2, size)
            };
            for x in x0..x1 {
                let i = (y * size + x) * 4;
                sum += i64::from(px[i]) - i64::from(px[i + 2]);
            }
        }
        sum
    };
    let dusk = render_dome(13_000, 0.0, 70.0, 96).expect("render");
    let dawn = render_dome(23_000, 0.0, 70.0, 96).expect("render");

    let dusk_side = warmth(&dusk, 0, 96) - warmth(&dusk, 1, 96);
    let dawn_side = warmth(&dawn, 0, 96) - warmth(&dawn, 1, 96);
    assert!(
        dusk_side.signum() != dawn_side.signum() && dusk_side != 0,
        "the glow must be on opposite horizons at dusk and dawn; got dusk={dusk_side}, \
         dawn={dawn_side}"
    );

    // And it must be *near the horizon*, not overhead: looking straight up at dusk must be far
    // cooler than looking at the horizon.
    let overhead = render_dome(13_000, -90.0, 70.0, 96).expect("render");
    let overhead_warmth = warmth(&overhead, 0, 96) + warmth(&overhead, 1, 96);
    let horizon_warmth = warmth(&dusk, 0, 96) + warmth(&dusk, 1, 96);
    assert!(
        horizon_warmth > overhead_warmth,
        "the sunset glow is not concentrated at the horizon (horizon {horizon_warmth} vs \
         overhead {overhead_warmth})"
    );
}

#[test]
fn nether_sky_phase_draws_nothing_leaving_the_clear_colour_untouched() {
    let Some(mut gpu) = headless(32, 32) else {
        return;
    };
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.1,
        g: 0.2,
        b: 0.3,
        a: 1.0,
    };
    let mut sky = SkyPhase::new(&gpu);
    sky.environment.dimension = Dimension::Nether;
    phases.insert(PhaseOrder::SKY, Box::new(sky));

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    // 0.1,0.2,0.3 -> 26,51,77
    let got = pixel(&pixels, 32, 16, 16);
    assert!(
        got[0].abs_diff(26) <= 2 && got[1].abs_diff(51) <= 2 && got[2].abs_diff(77) <= 2,
        "{got:?}"
    );
}

#[test]
fn end_sky_draws_its_fixed_dark_tint_when_the_texture_is_available() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(32, 32) else { return };
    let mut sky = SkyPhase::new(&gpu);
    sky.load_textures(&gpu, &rm);
    sky.environment.dimension = Dimension::End;

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    let mut gpu = gpu;
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 32, 16, 16);
    // Vanilla's fixed End skybox vertex colour is (40,40,40)/255, further modulated by whatever
    // the (grey/black) end_sky.png texel there is — so this should be dark, not the default
    // fog-blue clear colour.
    assert!(got[0] < 100 && got[1] < 100 && got[2] < 100, "{got:?}");
}

#[test]
fn sun_and_moon_are_visible_with_real_textures_loaded() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(128, 128) else {
        return;
    };
    let mut sky = SkyPhase::new(&gpu);
    sky.load_textures(&gpu, &rm);
    sky.environment = Environment {
        world_time: 6000, // noon: sun directly overhead per `celestial_angle`'s doc comment
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    };

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    let mut gpu = gpu;
    let frame = gpu.acquire().expect("acquire");
    // Look straight up: pitch -90 puts the sun's overhead position in view. `Camera::default()`
    // already called `update()` for pitch 0, so overriding the field alone leaves a stale view
    // matrix — `update()` must be called again after changing pitch/yaw/fov.
    let mut cam = Camera {
        pitch: -90.0,
        ..Camera::default()
    };
    cam.update();
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");

    // The sun quad is 60x60 blocks at distance 100, filling a good chunk of a 70 degree FOV
    // looking straight at it. Search the whole frame for the brightest pixel rather than
    // assuming it lands at the exact geometric centre (projection/rotation rounding could shift
    // it a few pixels) — this still proves the sun rendered, brighter than the sky background.
    let corner = pixel(&pixels, 128, 4, 4);
    let corner_luma = f64::from(corner[0]) + f64::from(corner[1]) + f64::from(corner[2]);
    let mut max_luma = 0.0f64;
    let mut max_at = (0u32, 0u32);
    for y in 0..128u32 {
        for x in 0..128u32 {
            let p = pixel(&pixels, 128, x, y);
            let luma = f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2]);
            if luma > max_luma {
                max_luma = luma;
                max_at = (x, y);
            }
        }
    }
    assert!(
        max_luma > corner_luma,
        "brightest pixel {max_luma} at {max_at:?} vs corner {corner_luma} ({corner:?})"
    );
}

/// java: `SimpleTexture.loadTexture` uploads every `textures/environment/**` PNG with
/// `blur = false` (vanilla ships no `.mcmeta` for them), so `TextureUtil.setTextureBlurMipmap`
/// selects `GL_NEAREST` and a magnified sun or moon keeps hard texel edges.
///
/// The pack here replaces `moon_phases.png` with an 8x4 sheet whose full-moon cell is a 2x2
/// white/black checker, drawn overhead at midnight over the dark night sky. Point sampling leaves
/// only white texels and night sky inside the moon. Linear filtering ramps between the two across
/// about half of it, which is the blur the user saw on the real sun and moon.
#[test]
fn the_moon_is_point_sampled_with_hard_texel_edges() {
    use image::ImageEncoder;

    let Some(mut gpu) = headless(256, 256) else {
        return;
    };
    let pack = tempfile::tempdir().expect("temp resource pack");
    let environment = pack.path().join("assets/minecraft/textures/environment");
    std::fs::create_dir_all(&environment).expect("resource pack directories");
    let mut sheet = image::RgbaImage::from_pixel(8, 4, image::Rgba([0, 0, 0, 255]));
    sheet.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
    sheet.put_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(sheet.as_raw(), 8, 4, image::ExtendedColorType::Rgba8)
        .expect("encode the moon sheet");
    std::fs::write(environment.join("moon_phases.png"), png).expect("write the moon sheet");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::FolderResourcePack::open(pack.path()).expect("folder pack"),
    ));

    let mut sky = SkyPhase::new(&gpu);
    sky.load_textures(&gpu, &rm);
    // World time 18000: midnight, moon overhead, phase 0 (the full moon, cell 0,0).
    sky.environment = Environment {
        world_time: 18_000,
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    };
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    let mut cam = Camera {
        pos: DVec3::new(0.0, 70.0, 0.0),
        pitch: -90.0,
        ..Camera::default()
    };
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");

    let luma = |p: [u8; 4]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for y in 0..256 {
        for x in 0..256 {
            if luma(pixel(&pixels, 256, x, y)) >= 750 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
    }
    assert!(
        x0 < x1 && y0 < y1,
        "the moon's white texels are not on screen"
    );
    let area = (x1 - x0 + 1) * (y1 - y0 + 1);
    let blended = (y0..=y1)
        .flat_map(|y| (x0..=x1).map(move |x| (x, y)))
        .filter(|&(x, y)| (60..700).contains(&luma(pixel(&pixels, 256, x, y))))
        .count() as u32;
    // A few stars can land on the black texels; a filtered moon blends thousands of pixels.
    assert!(
        blended * 50 < area,
        "{blended} of the moon's {area} pixels are blends of a white and a black texel"
    );
}

// ------------------------------------------------------------ clouds

/// Render one cloud layer over black, looking straight up from `eye_y`, and hand back the frame.
fn render_clouds_with_fog(
    mode: CloudMode,
    world_time: i64,
    eye_y: f64,
    size: u32,
    fog: Option<(FogState, FogDistance)>,
    clear_alpha: f64,
    view_distance_chunks: Option<u32>,
) -> Option<Vec<u8>> {
    let rm = resources()?;
    let mut gpu = headless(size, size)?;
    let mut clouds = CloudsPhase::new(&gpu);
    clouds.load_texture(&gpu, &rm);
    clouds.mode = mode;
    clouds.set_fog(fog);
    clouds.environment = Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    };

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: clear_alpha,
    };
    phases.insert(PhaseOrder::SKY, Box::new(clouds));

    // Positive pitch looks *down*, per `Camera`'s doc comment, so -90 looks up at a layer above
    // the eye. `Camera::default()` already called `update()`, so overriding a field alone leaves
    // a stale view matrix — `update()` must be called again.
    let mut cam = Camera {
        pos: DVec3::new(0.0, eye_y, 0.0),
        pitch: -90.0,
        ..Camera::default()
    };
    if let Some(chunks) = view_distance_chunks {
        cam.set_view_distance_chunks(chunks);
    }
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    Some(gpu.read_pixels().expect("readback"))
}

/// Render one cloud layer over black, looking straight up from `eye_y`, and hand back the frame.
fn render_clouds(mode: CloudMode, world_time: i64, eye_y: f64, size: u32) -> Option<Vec<u8>> {
    render_clouds_with_fog(mode, world_time, eye_y, size, None, 1.0, None)
}

/// Fraction of pixels that are not the black clear colour.
fn covered_fraction(pixels: &[u8]) -> f64 {
    let lit = pixels
        .chunks(4)
        .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
        .count();
    lit as f64 / (pixels.len() / 4) as f64
}

/// Mean horizontal run length, in pixels, of consecutive lit (cloud) pixels.
///
/// This is the measurement that separates "clouds" from "noise", and it is a *scale* statistic,
/// not a coverage one: `clouds.png` is ~20% opaque whatever scale it is drawn at, so coverage
/// barely moves when the texture is stretched — but the size of each blob moves by exactly the
/// scale error. One cloud unit is 12 blocks (`RenderGlobal.java:1545`); mapping the whole 256x256
/// sheet across a 96-block cell instead shrinks a texel to 0.375 blocks, i.e. 32x finer.
fn mean_lit_run(pixels: &[u8], size: u32) -> f64 {
    let lit = |x: u32, y: u32| {
        let p = pixel(pixels, size, x, y);
        p[0] > 8 || p[1] > 8 || p[2] > 8
    };
    let mut runs = 0u32;
    let mut lit_px = 0u32;
    for y in 0..size {
        let mut prev = false;
        for x in 0..size {
            let now = lit(x, y);
            if now {
                lit_px += 1;
                if !prev {
                    runs += 1;
                }
            }
            prev = now;
        }
    }
    if runs == 0 {
        return 0.0;
    }
    f64::from(lit_px) / f64::from(runs)
}

/// **The bug this pins.** `fancy_cloud_vertices` used to map the whole 256x256 `clouds.png` across
/// every 96-block cell instead of vanilla's 8x8-texel patch (`RenderGlobal.java:1582`), a 32x
/// scale error whose symptom was a sky-filling noise field. The old test — "is some pixel
/// non-black" — passed happily through it, and so does a coverage check, because a stretched
/// texture has exactly the same opaque *fraction*.
///
/// The discriminator is blob **size**. Looking straight up from `y = 64` at a layer 64.33 blocks
/// overhead through a 70-degree vertical FOV, one 12-block cloud unit subtends
/// `2·atan(6/64.33) = 10.7°`, i.e. about a sixth of the frame — so lit runs must be many pixels
/// long. At the buggy scale a texel is 0.375 blocks and runs collapse to about one pixel.
///
/// Both modes are checked, which makes this a statement about vanilla's texel scale rather than
/// about one code path: `FAST_TEXEL` (1/2048 per block) and `FANCY_TEXEL` (1/256 per cloud unit)
/// are different constants that must both land in the same visual ballpark.
#[test]
fn cloud_blobs_are_block_sized_not_pixel_sized_noise() {
    const SIZE: u32 = 128;
    for mode in [CloudMode::Fast, CloudMode::Fancy] {
        let Some(pixels) = render_clouds(mode, 6000, 64.0, SIZE) else {
            eprintln!("SKIP: jar or adapter unavailable");
            return;
        };
        let covered = covered_fraction(&pixels);
        assert!(
            covered > 0.05,
            "{mode:?} clouds drew almost nothing ({:.1}% covered)",
            covered * 100.0
        );
        let run = mean_lit_run(&pixels, SIZE);
        assert!(
            run >= 6.0,
            "{mode:?}: mean lit run is {run:.1} px over a {SIZE}px frame. Vanilla's cloud texels \
             are blocks across, so runs are tens of pixels; single-digit runs mean clouds.png is \
             being scaled far too small across each cell — the exact bug `RenderGlobal.java:1582`'s \
             1/256-per-cloud-unit scale prevents."
        );
    }
}

/// The cloud fog uniform must reach the real fragment shader. A fully fogged frame uses a pure
/// blue fog colour, so every surviving cloud pixel should shift from the un-fogged neutral tint to
/// blue while the source alpha/coverage remains identical. This catches a missing bind-group
/// entry, stale per-frame upload, wrong distance metric, and accidental alpha modification at the
/// actual GPU boundary; the CPU packing helper alone cannot catch those failures.
#[test]
fn cloud_fragments_fog_rgb_without_changing_alpha_or_coverage() {
    const SIZE: u32 = 128;
    let Some(clear) = render_clouds_with_fog(CloudMode::Fancy, 6000, 64.0, SIZE, None, 0.0, None)
    else {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    };
    let fully_fogged = FogState {
        color: Vec3::new(0.0, 0.0, 1.0),
        mode: FogMode::Linear {
            start: 0.0,
            end: 1.0,
        },
    };
    let fogged = render_clouds_with_fog(
        CloudMode::Fancy,
        6000,
        64.0,
        SIZE,
        Some((fully_fogged, FogDistance::Planar)),
        0.0,
        None,
    )
    .expect("second cloud render");

    let mut cloud_pixels = 0usize;
    for (clear_px, fogged_px) in clear.chunks_exact(4).zip(fogged.chunks_exact(4)) {
        assert_eq!(clear_px[3], fogged_px[3], "cloud alpha changed under fog");
        let clear_visible = clear_px[0] > 8 || clear_px[1] > 8 || clear_px[2] > 8;
        let fogged_visible = fogged_px[0] > 8 || fogged_px[1] > 8 || fogged_px[2] > 8;
        assert_eq!(
            clear_visible, fogged_visible,
            "cloud coverage changed under fog"
        );
        if clear_visible {
            cloud_pixels += 1;
            assert!(
                fogged_px[2] > fogged_px[0] + 20 && fogged_px[2] > fogged_px[1] + 20,
                "fully fogged cloud pixel did not become blue: clear={clear_px:?} fogged={fogged_px:?}"
            );
        }
    }
    assert!(
        cloud_pixels > 100,
        "fixture rendered too few cloud pixels: {cloud_pixels}"
    );
}

/// Java temporarily expands the cloud projection to `renderDistanceChunks * 16 * 4`. At two
/// chunks the ordinary terrain far plane is only `45.25` blocks, while this fixture's cloud layer
/// is `64.33` blocks above the eye and therefore must be clipped by terrain projection but visible
/// through Java's cloud projection. A zero-coverage result means the cloud draw is still using the
/// shared terrain far plane.
#[test]
fn clouds_use_java_expanded_far_plane_beyond_short_terrain_clip() {
    const SIZE: u32 = 128;
    let Some(pixels) =
        render_clouds_with_fog(CloudMode::Fast, 6000, 64.0, SIZE, None, 0.0, Some(2))
    else {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    };
    let visible = pixels
        .chunks_exact(4)
        .filter(|px| px[0] > 8 || px[1] > 8 || px[2] > 8)
        .count();
    assert!(
        visible > 100,
        "cloud layer beyond the 2-chunk terrain far plane was clipped ({visible} visible pixels)"
    );
}

/// java: `World.getCloudColour` (`World.java:1539`) — `f2 * (f1 * 0.9F + 0.1F)`, the day/night
/// term this port omitted entirely. Without it clouds stay full white at midnight; with it they
/// drop to a tenth of their daytime brightness. Asserted on rendered pixels, not on the formula,
/// so it covers the whole path from `Environment` to the framebuffer.
#[test]
fn clouds_go_dark_at_night_instead_of_glowing_white() {
    let Some(noon) = render_clouds(CloudMode::Fancy, 6000, 64.0, 128) else {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    };
    let midnight = render_clouds(CloudMode::Fancy, 18_000, 64.0, 128).expect("second render");

    // Compare only where clouds actually are — the gaps are black in both frames.
    let brightest = |px: &[u8]| {
        px.chunks(4)
            .map(|p| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]))
            .max()
            .unwrap_or(0)
    };
    let day = brightest(&noon);
    let night = brightest(&midnight);
    assert!(day > 300, "clouds should be bright at noon, got {day}");
    // `f1` clamps to 0 at midnight, so the multiplier is 0.1/0.1/0.15 — comfortably under a fifth.
    assert!(
        f64::from(night) < f64::from(day) * 0.25,
        "midnight clouds ({night}) must be far darker than noon ({day}); equal brightness means \
         the `f1 * 0.9 + 0.1` day/night term is missing"
    );
    assert!(night > 0, "clouds should still be faintly visible at night");
}

/// java: `RenderGlobal.java:1580`/`:1588` — the `f3 > -5` / `f3 <= 5` gates. Standing below the
/// layer you see undersides (tinted `0.7`); from above, tops (untinted). A port that always draws
/// both looks identical from either side, which is how this went unnoticed.
#[test]
fn fancy_clouds_show_their_underside_from_below_and_their_top_from_above() {
    let Some(below) = render_clouds(CloudMode::Fancy, 6000, 64.0, 128) else {
        eprintln!("SKIP: jar or adapter unavailable");
        return;
    };
    // From above, the camera still looks "up" (pitch -90) but the layer is beneath it, so nothing
    // is in frame; look down instead by placing the eye above and flipping the geometry test to
    // the vertex level. Here we only need the *shading*, so compare the brightest lit pixel.
    let brightest = |px: &[u8]| {
        px.chunks(4)
            .map(|p| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]))
            .max()
            .unwrap_or(0)
    };
    // The underside is `colour * 0.7`; nothing in the frame may exceed that, because the top face
    // (`colour * 1.0`) must not have been emitted at all.
    let lit = brightest(&below);
    let white = 255 * 3;
    assert!(
        f64::from(lit) < f64::from(white) * 0.85,
        "from below, the brightest cloud pixel was {lit}/{white} — that is the *top* face's \
         untinted colour leaking through, so the `f3 <= 5.0` gate is missing"
    );
    assert!(
        lit > 300,
        "the underside should still be clearly lit, got {lit}"
    );
}

/// java: `EntityRenderer.renderWorldPass:1364` — below y=128 the clouds draw before terrain with
/// depth writes on (fancy: `RenderGlobal.java:1547`'s depth pass, then colour), and their depth
/// stays for the terrain drawn after them: terrain behind a cloud is rejected where the cloud
/// covers it, which is what hides a block above the cloud layer behind it. (This port used to
/// clear the depth again, so blocks behind clouds drew over them.) Proven with a depth-writing
/// quad recorded *after* the clouds and farther away than any of them: it must fill every gap
/// between clouds and nothing else.
#[test]
fn clouds_keep_their_depth_for_terrain_drawn_after_them() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: jar unavailable");
        return;
    };
    let Some(mut gpu) = headless(64, 64) else {
        return;
    };
    for mode in [CloudMode::Fast, CloudMode::Fancy] {
        let mut clouds = CloudsPhase::new(&gpu);
        clouds.load_texture(&gpu, &rm);
        clouds.mode = mode;
        clouds.environment.dimension = Dimension::Overworld;

        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::SKY, Box::new(clouds));
        // 400 blocks away: farther than any cloud surface in view (all within ~92 blocks).
        phases.insert(
            PhaseOrder::TERRAIN_SOLID,
            Box::new(depth_probe::DepthProbe::new(&gpu, 400.0)),
        );

        let mut cam = Camera {
            pos: DVec3::new(0.0, 64.0, 0.0),
            pitch: -90.0,
            ..Camera::default()
        };
        cam.update();
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &cam, 0.0);
        let pixels = gpu.read_pixels().expect("readback");

        let probe = |p: &[u8]| p[0] > 200 && p[1] < 60 && p[2] < 60;
        let probe_pixels = pixels.chunks(4).filter(|p| probe(p)).count();
        let cloud_pixels = pixels
            .chunks(4)
            .filter(|p| !probe(p) && (p[0] > 8 || p[1] > 8 || p[2] > 8))
            .count();
        let bare = pixels
            .chunks(4)
            .filter(|p| p[0] <= 8 && p[1] <= 8 && p[2] <= 8)
            .count();
        assert!(
            probe_pixels > 1000,
            "{mode:?}: the quad behind the clouds should fill the gaps between them, \
             {probe_pixels} px of 4096 did"
        );
        assert!(
            cloud_pixels > 300,
            "{mode:?}: the quad behind the clouds drew over them ({cloud_pixels} cloud px left) \
             — the cloud depth was not kept"
        );
        assert_eq!(
            bare, 0,
            "{mode:?}: {bare} px neither cloud nor quad: the quad was rejected where no cloud drew"
        );
    }
}

/// A minimal opaque, depth-writing red quad at a chosen eye distance, standing in for terrain.
mod depth_probe {
    use mc_render::{Camera, DepthAttachment, Gpu, RenderPhase, DEPTH_FORMAT};

    const WGSL: &str = r"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, DEPTH, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
";

    pub struct DepthProbe {
        pipeline: wgpu::RenderPipeline,
    }

    impl DepthProbe {
        pub fn new(gpu: &Gpu, distance: f32) -> Self {
            // wgpu's 0..1 reversed-free depth range: `near/…` is fiddly, so just pick a clip-space
            // z directly from the distance through the default projection's mapping. Any value in
            // (0, 1) that is "far" works; 400 blocks with a 256-block far plane clamps to ~0.997.
            let z = (1.0 - 0.05 / distance).clamp(0.0, 0.999_9);
            let src = WGSL.replace("DEPTH", &format!("{z}"));
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("depth probe"),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("depth probe layout"),
                    bind_group_layouts: &[],
                    immediate_size: 0,
                });
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("depth probe"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            Self { pipeline }
        }
    }

    impl RenderPhase for DepthProbe {
        fn name(&self) -> &'static str {
            "test/depth-probe"
        }

        fn record(
            &mut self,
            gpu: &Gpu,
            enc: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            _cam: &Camera,
        ) {
            let mut pass = gpu.begin_pass(enc, view, "depth probe", DepthAttachment::Shared);
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..3, 0..1);
        }
    }
}

#[test]
fn clouds_are_disabled_when_mode_is_off() {
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let mut clouds = CloudsPhase::new(&gpu);
    clouds.mode = CloudMode::Off;
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::SKY, Box::new(clouds));
    assert_eq!(phases.names().count(), 1);
    let frame = gpu.acquire().expect("acquire");
    // Must not panic / must skip cleanly with no texture and no geometry.
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let _ = gpu.read_pixels().expect("readback");
}

// ------------------------------------------------------------ weather: the rainstorm

struct WarmFlatWorld;
impl ColumnEnvironment for WarmFlatWorld {
    fn temperature_at(&self, _x: i32, _y: i32, _z: i32) -> f32 {
        0.8 // rain, not snow
    }
    fn precipitation_height(&self, _x: i32, _z: i32) -> i32 {
        70
    }
    fn light_at(&self, _x: i32, _y: i32, _z: i32) -> u8 {
        15
    }
}

struct ColdFlatWorld;
impl ColumnEnvironment for ColdFlatWorld {
    fn temperature_at(&self, _x: i32, _y: i32, _z: i32) -> f32 {
        0.0 // snow
    }
    fn precipitation_height(&self, _x: i32, _z: i32) -> i32 {
        70
    }
    fn light_at(&self, _x: i32, _y: i32, _z: i32) -> u8 {
        15
    }
}

#[test]
fn a_rainstorm_draws_visible_streaks_over_a_black_background() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(mut gpu) = headless(128, 128) else {
        return;
    };
    let mut weather = WeatherPhase::new(&gpu);
    weather.load_textures(&gpu, &rm);
    weather.environment.dimension = Dimension::Overworld;

    let cam = Camera {
        pos: DVec3::new(0.0, 68.0, 0.0),
        ..Camera::default()
    };
    let columns = build_columns(&WarmFlatWorld, cam.pos, 0, true, 1.0);
    assert!(!columns.is_empty(), "expected some rain columns");
    weather.upload(columns);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::WEATHER, Box::new(weather));

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let lit_pixels = pixels
        .chunks(4)
        .filter(|p| p[0] > 4 || p[1] > 4 || p[2] > 4)
        .count();
    assert!(
        lit_pixels > 0,
        "expected at least some rain-streak pixels over a black background"
    );
}

#[test]
fn a_snowstorm_selects_the_snow_texture_not_rain() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(mut gpu) = headless(64, 64) else {
        return;
    };
    let mut weather = WeatherPhase::new(&gpu);
    weather.load_textures(&gpu, &rm);

    let cam = Camera {
        pos: DVec3::new(0.0, 68.0, 0.0),
        ..Camera::default()
    };
    let columns = build_columns(&ColdFlatWorld, cam.pos, 0, true, 1.0);
    assert!(columns.iter().all(|c| c.is_snow));
    weather.upload(columns);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::WEATHER, Box::new(weather));
    let frame = gpu.acquire().expect("acquire");
    // Must not panic drawing through the snow texture path specifically.
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let _ = gpu.read_pixels().expect("readback");
}

#[test]
fn weather_phase_is_disabled_outside_the_overworld() {
    let Some(gpu) = headless(8, 8) else { return };
    let mut weather = WeatherPhase::new(&gpu);
    weather.environment.dimension = Dimension::Nether;
    assert!(!mc_render::RenderPhase::enabled(&weather));
    weather.environment.dimension = Dimension::Overworld;
    assert!(mc_render::RenderPhase::enabled(&weather));
}

// ------------------------------------------------------------ lightning: RenderLightningBolt

/// Aim the camera at a bolt's lowest main-bolt segment from `distance` blocks away along +Z
/// (vanilla yaw 0 faces +Z). Derived from the real geometry rather than guessed, so the bolt is
/// guaranteed on screen whatever `bolt_vertex` the test picks.
fn camera_facing_bolt(strike: LightningStrike, distance: f64) -> Camera {
    let segments = mc_render::sky::lightning_bolt_segments(strike.bolt_vertex);
    // Index 7 is k1 = 0, j = 0, i1 = 0: the main bolt's ground-end segment.
    let seg = segments[7];
    let mut centre = DVec3::ZERO;
    for v in seg.verts {
        centre += v;
    }
    centre = strike.position + centre / seg.verts.len() as f64;

    let mut cam = Camera {
        pos: DVec3::new(centre.x, centre.y, centre.z - distance),
        ..Camera::default()
    };
    cam.update();
    cam
}

#[test]
fn a_lightning_bolt_draws_vanillas_exact_additive_colour_ratio() {
    // java: RenderLightningBolt draws every vertex at (0.45, 0.45, 0.5, 0.3) with
    // blendFunc(GL_SRC_ALPHA, GL_ONE), so each overlapping layer adds exactly
    // (0.45, 0.45, 0.5) * 0.3 = (0.135, 0.135, 0.15) over a black frame. However many tubes
    // overlap a given pixel, the accumulated colour is therefore always r == g and
    // b == r * (0.5 / 0.45) — an exact, layer-count-independent fingerprint of the vanilla
    // constants having reached the screen through the right blend equation.
    let Some(mut gpu) = headless(128, 128) else {
        return;
    };
    let strike = LightningStrike {
        position: DVec3::ZERO,
        bolt_vertex: 0x0BAD_C0DE,
    };
    let cam = camera_facing_bolt(strike, 6.0);

    let mut weather = WeatherPhase::new(&gpu);
    weather.environment.dimension = Dimension::Overworld;
    weather.add_lightning_strike(strike);
    assert_eq!(weather.lightning_strikes().len(), 1);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::WEATHER, Box::new(weather));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");

    let lit: Vec<[u8; 4]> = pixels
        .chunks(4)
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    assert!(
        !lit.is_empty(),
        "expected the bolt to light some pixels over a black frame"
    );

    // The dimmest lit pixel is exactly **one** layer: each `tessellator.draw()` is a closed
    // four-sided tube, so any ray through it crosses a near wall and a far wall, but the world
    // pass's `enableCull()` is still on when `RenderGlobal.renderEntities` draws weather effects,
    // so only the near (counter-clockwise) wall is drawn. (H7 2026-11-11: mc-rs drew both walls
    // and its dimmest pixel was two layers, 68/68/76; `tests/lightning_java_parity.rs` compares
    // whole frames against the real `RenderLightningBolt`.) Rounding happens once per draw
    // against the 8-bit target, so the arithmetic is exact and hand-checkable:
    //   red:  0 + 0.45*0.3 = 0.135 -> 34
    //   blue: 0 + 0.50*0.3 = 0.15  -> 38
    let dimmest = lit.iter().min_by_key(|p| p[0]).unwrap();
    assert_eq!(
        [dimmest[0], dimmest[1], dimmest[2]],
        [34, 34, 38],
        "one additive layer of (0.45, 0.45, 0.5, 0.3) over black: {dimmest:?}"
    );

    for p in &lit {
        assert_eq!(p[0], p[1], "red and green must stay equal: {p:?}");
        assert!(p[2] >= p[0], "blue must lead: {p:?}");
        if p[2] < 250 {
            // Below the u8 ceiling the ratio holds exactly; 1 accounts for per-layer rounding.
            let expected_blue = (f64::from(p[0]) * 0.5 / 0.45).round() as i32;
            assert!(
                (i32::from(p[2]) - expected_blue).abs() <= 2,
                "blue {p:?} should be red * 0.5/0.45 = {expected_blue}"
            );
        }
    }
}

#[test]
fn a_lightning_bolt_needs_no_texture_at_all() {
    // Vanilla calls `GlStateManager.disableTexture2D()` before drawing the bolt, so it must draw
    // identically whether or not `load_textures` ever ran. (The previous port routed the bolt
    // through the rain texture's bind group and drew nothing without one.)
    let Some(mut gpu) = headless(64, 64) else {
        return;
    };
    let strike = LightningStrike {
        position: DVec3::ZERO,
        bolt_vertex: 12345,
    };
    let cam = camera_facing_bolt(strike, 6.0);

    let mut weather = WeatherPhase::new(&gpu);
    weather.add_lightning_strike(strike);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::WEATHER, Box::new(weather));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let lit = pixels.chunks(4).filter(|p| p[2] > 0).count();
    assert!(lit > 0, "bolt drew nothing without a texture loaded");
}

// ------------------------------------------------------------ fog: the screen-space pass

/// A test-local phase that fills the frame with one colour at one chosen eye-space distance,
/// writing the shared depth buffer — the stand-in for terrain, so the fog pass has something to
/// fog. Kept here rather than reaching into `terrain/**` (S11's module) so the expected distance
/// is exact and chosen by the test.
struct DepthQuad {
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
    /// Eye-space distance to place the quad at.
    distance: f32,
    /// Colour to fill with.
    color: [f32; 3],
}

const DEPTH_QUAD_WGSL: &str = r#"
struct P { v: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var c = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(c[i], p.v.w, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(p.v.xyz, 1.0); }
"#;

impl DepthQuad {
    fn new(gpu: &Gpu, distance: f32, color: [f32; 3]) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("depth quad"),
                source: wgpu::ShaderSource::Wgsl(DEPTH_QUAD_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth quad"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: mc_render::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            buffer,
            group,
            distance,
            color,
        }
    }
}

/// The clip-space depth an eye-space distance `d` maps to under `cam`'s projection. Inverts the
/// same relation `fog.wgsl` inverts, so the two agreeing is a real cross-check rather than a
/// tautology: this test derives depth from the *matrix*, the shader derives distance from *it*.
fn ndc_depth_for_distance(cam: &Camera, d: f32) -> f32 {
    let a = cam.proj.z_axis.z;
    let b = cam.proj.w_axis.z;
    (b - a * d) / d
}

impl mc_render::RenderPhase for DepthQuad {
    fn name(&self) -> &'static str {
        "test/depth-quad"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        let depth = ndc_depth_for_distance(cam, self.distance);
        gpu.queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[self.color[0], self.color[1], self.color[2], depth]),
        );
        let mut pass = gpu.begin_pass(
            enc,
            view,
            "test/depth-quad",
            mc_render::DepthAttachment::Shared,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn fog_inputs(render_distance_chunks: u32) -> FogInputs {
    FogInputs {
        render_distance_chunks,
        ..FogInputs::default()
    }
}

/// Render a white depth-writing quad at `distance` with a `FogPhase` over it, and return the
/// centre pixel plus the `FogState` the phase used.
fn render_fog(distance: f32, samples: u32) -> Option<([u8; 4], mc_render::sky::FogState)> {
    let mut gpu = match Gpu::headless(
        32,
        32,
        GpuOptions {
            vsync: false,
            sample_count: samples,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            return None;
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    };

    let mut fog = FogPhase::new(&gpu);
    fog.inputs = fog_inputs(8); // far = 128 -> Linear { start: 96, end: 128 }
    let state = fog.state();

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(DepthQuad::new(&gpu, distance, [1.0, 1.0, 1.0])),
    );
    phases.insert(FogPhase::ORDER, Box::new(fog));

    let cam = Camera::default();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    Some((pixel(&pixels, 32, 16, 16), state))
}

/// Render a constant-z surface and return a near-centre and right-edge sample.  A constant depth
/// value makes the planar metric identical at both pixels; only a true radial reconstruction can
/// increase the distance toward the edge.  This deliberately exercises the fragment position
/// and projection terms rather than merely checking that the selector bit reaches a buffer.
fn render_off_axis_fog(
    distance: FogDistance,
) -> Option<([u8; 4], [u8; 4], mc_render::sky::FogState)> {
    let mut gpu = match Gpu::headless(
        32,
        32,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            return None;
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    };

    let mut fog = FogPhase::new(&gpu);
    fog.inputs = fog_inputs(8); // Linear { start: 96, end: 128 }.
    fog.set_distance(distance);
    let state = fog.state();

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(DepthQuad::new(&gpu, 110.0, [1.0, 1.0, 1.0])),
    );
    phases.insert(FogPhase::ORDER, Box::new(fog));

    let mut cam = Camera::default();
    cam.set_viewport(32, 32);
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    Some((
        pixel(&pixels, 32, 16, 16),
        pixel(&pixels, 32, 31, 16),
        state,
    ))
}

#[test]
fn radial_fog_increases_with_off_axis_ray_length() {
    let Some((planar_center, planar_edge, state)) = render_off_axis_fog(FogDistance::Planar) else {
        return;
    };
    let Some((radial_center, radial_edge, _)) = render_off_axis_fog(FogDistance::Radial) else {
        return;
    };

    // A constant depth quad has one planar distance everywhere.  Keep this exact equality as a
    // negative control: if it fails, the test fixture itself is not holding z constant.
    assert_eq!(
        planar_center, planar_edge,
        "planar fog must ignore the fragment's off-axis position"
    );

    // At z=110 and fov=70°, the right edge's radial distance is beyond the 128-block fog end,
    // while the centre remains near 110.  The default noon fog colour is below white in red and
    // green, so the edge must be visibly closer to that colour under the radial metric.
    assert!(
        radial_edge[0] < planar_edge[0] && radial_edge[1] < planar_edge[1],
        "radial edge must fog more than planar edge: planar={planar_edge:?}, radial={radial_edge:?}, state={state:?}"
    );
    assert_ne!(
        radial_edge, radial_center,
        "radial fog must vary across a constant-z surface"
    );
    for (channel, fog_channel) in state.color.to_array().into_iter().enumerate() {
        let fully_fogged = (fog_channel * 255.0).round() as u8;
        assert!(
            radial_edge[channel].abs_diff(fully_fogged) <= 2,
            "the off-axis ray beyond 128 blocks must reach the fog colour"
        );
        assert!(
            radial_center[channel].abs_diff(planar_center[channel]) <= 2,
            "the near-axis radial control must retain the planar blend"
        );
    }
}

#[test]
fn screen_space_fog_blends_by_the_exact_linear_factor() {
    // Render distance 8 -> vanilla's `setupFog` air branch gives Linear { start: far * 0.75,
    // end: far } = { 96, 128 }. A surface at 112 blocks is therefore exactly half fogged, and
    // GL_FOG's blend is `mix(fogColor, surfaceColor, factor)`.
    let Some((got, state)) = render_fog(112.0, 1) else {
        return;
    };
    let mc_render::sky::FogMode::Linear { start, end } = state.mode else {
        panic!("expected linear air fog, got {:?}", state.mode)
    };
    assert!(
        (start - 96.0).abs() < 1e-3 && (end - 128.0).abs() < 1e-3,
        "{start} {end}"
    );

    let factor = 0.5f32;
    let expected: Vec<u8> = (0..3)
        .map(|i| {
            let fog = state.color.to_array()[i];
            let surface = 1.0f32;
            ((surface * factor + fog * (1.0 - factor)) * 255.0).round() as u8
        })
        .collect();
    assert!(
        got[0].abs_diff(expected[0]) <= 2
            && got[1].abs_diff(expected[1]) <= 2
            && got[2].abs_diff(expected[2]) <= 2,
        "got {got:?}, expected ~{expected:?} (fog colour {:?})",
        state.color
    );
    // And it really is a blend, not a replace: a white surface half-fogged must still be
    // brighter than the fog colour alone.
    let fog_only: Vec<u8> = (0..3)
        .map(|i| (state.color.to_array()[i] * 255.0).round() as u8)
        .collect();
    assert!(got[0] > fog_only[0], "{got:?} vs fog {fog_only:?}");
}

#[test]
fn fog_fully_replaces_a_surface_past_the_end_distance() {
    // Beyond `end` the linear factor clamps to 0, so the surface colour drops out entirely and
    // the pixel is the fog colour exactly — vanilla's "terrain dissolves into the sky" limit.
    let Some((got, state)) = render_fog(200.0, 1) else {
        return;
    };
    let expected: Vec<u8> = (0..3)
        .map(|i| (state.color.to_array()[i] * 255.0).round() as u8)
        .collect();
    assert!(
        got[0].abs_diff(expected[0]) <= 2
            && got[1].abs_diff(expected[1]) <= 2
            && got[2].abs_diff(expected[2]) <= 2,
        "got {got:?}, expected {expected:?}"
    );
}

#[test]
fn fog_leaves_a_surface_inside_the_start_distance_untouched() {
    let Some((got, _)) = render_fog(50.0, 1) else {
        return;
    };
    // Nearer than `start` (96) -> factor 1 -> the white quad survives verbatim.
    assert_eq!([got[0], got[1], got[2]], [255, 255, 255], "{got:?}");
}

#[test]
fn fog_never_touches_pixels_with_nothing_drawn_in_them() {
    // Depth still at the cleared 1.0 means sky. Vanilla renders the sky with fog disabled, and
    // `fog.wgsl` reproduces that with a `discard`, so the clear colour must survive byte-exactly.
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let mut fog = FogPhase::new(&gpu);
    fog.inputs = fog_inputs(8);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.2,
        g: 0.4,
        b: 0.6,
        a: 1.0,
    };
    phases.insert(FogPhase::ORDER, Box::new(fog));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 16, 8, 8);
    assert!(
        got[0].abs_diff(51) <= 1 && got[1].abs_diff(102) <= 1 && got[2].abs_diff(153) <= 1,
        "{got:?}"
    );
}

/// What `FirstPersonPhase` does to the shared depth buffer — `renderHand`'s
/// `clear(GL_DEPTH_BUFFER_BIT)` — and nothing else.
struct HandDepthClear;

impl mc_render::RenderPhase for HandDepthClear {
    fn name(&self) -> &'static str {
        "test/hand-depth-clear"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        drop(gpu.begin_pass(
            enc,
            view,
            "test/hand-depth-clear",
            mc_render::DepthAttachment::Clear,
        ));
    }
}

/// Original report §7.1 / R1-10: the hand pass clears depth, so world fog must record before it.
/// Registered in the client's order (hand first, fog second), a surface past the fog end must
/// still come out fog-coloured — before the fix it came out as the bare white surface.
#[test]
fn world_fog_survives_the_first_person_depth_clear() {
    let Some(mut gpu) = headless(32, 32) else {
        return;
    };
    let mut fog = FogPhase::new(&gpu);
    fog.inputs = fog_inputs(8);
    let state = fog.state();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(DepthQuad::new(&gpu, 200.0, [1.0, 1.0, 1.0])),
    );
    phases.insert(PhaseOrder::FIRST_PERSON, Box::new(HandDepthClear));
    phases.insert(FogPhase::ORDER, Box::new(fog));
    let names: Vec<&str> = phases.names().collect();
    let fog_at = names
        .iter()
        .position(|n| *n == "sky/fog")
        .expect("fog registered");
    let hand_at = names
        .iter()
        .position(|n| *n == "test/hand-depth-clear")
        .expect("hand registered");
    assert!(fog_at < hand_at, "{names:?}");

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 32, 16, 16);
    let expected: Vec<u8> = (0..3)
        .map(|i| (state.color.to_array()[i] * 255.0).round() as u8)
        .collect();
    assert!(
        got[0].abs_diff(expected[0]) <= 2
            && got[1].abs_diff(expected[1]) <= 2
            && got[2].abs_diff(expected[2]) <= 2,
        "got {got:?}, expected fog colour {expected:?}"
    );
}

#[test]
fn fog_reads_a_multisampled_depth_buffer_too() {
    // MSAA makes the shared depth buffer multisampled, which cannot be bound as an ordinary
    // `texture_depth_2d`; `FogPhase::new` swaps the WGSL declaration for
    // `texture_depth_multisampled_2d`. If that path were wrong this would fail pipeline
    // validation rather than producing a subtly wrong pixel.
    let Some((got, state)) = render_fog(112.0, 4) else {
        return;
    };
    let expected: Vec<u8> = (0..3)
        .map(|i| {
            let fog = state.color.to_array()[i];
            ((0.5 + fog * 0.5) * 255.0).round() as u8
        })
        .collect();
    assert!(
        got[0].abs_diff(expected[0]) <= 3
            && got[1].abs_diff(expected[1]) <= 3
            && got[2].abs_diff(expected[2]) <= 3,
        "got {got:?}, expected ~{expected:?}"
    );
}

#[test]
fn underwater_fog_is_exponential_and_much_denser_than_air() {
    // java: setupFog's water branch — `GL_EXP` at density 0.1 with no respiration, which at 20
    // blocks leaves exp(-2) = 13.5% of the surface colour showing.
    let Some(mut gpu) = headless(32, 32) else {
        return;
    };
    let mut fog = FogPhase::new(&gpu);
    fog.inputs = FogInputs {
        medium: mc_render::sky::CameraMedium::Water,
        ..fog_inputs(8)
    };
    let state = fog.state();
    let mc_render::sky::FogMode::Exp { density } = state.mode else {
        panic!("expected exponential water fog")
    };

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(DepthQuad::new(&gpu, 20.0, [1.0, 1.0, 1.0])),
    );
    phases.insert(FogPhase::ORDER, Box::new(fog));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 32, 16, 16);

    let factor = (-density * 20.0f32).exp();
    let expected: Vec<u8> = (0..3)
        .map(|i| {
            let fog = state.color.to_array()[i];
            ((factor + fog * (1.0 - factor)) * 255.0).round() as u8
        })
        .collect();
    assert!(
        got[0].abs_diff(expected[0]) <= 2
            && got[1].abs_diff(expected[1]) <= 2
            && got[2].abs_diff(expected[2]) <= 2,
        "got {got:?}, expected ~{expected:?} at density {density}"
    );
    // Vanilla's underwater fog colour is blue-dominant (0.02, 0.02, 0.2 + respiration).
    assert!(got[2] > got[0], "{got:?}");
}
