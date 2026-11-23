//! The F3+Shift profiler pie on the GPU (java: `Minecraft.displayDebugInfo`).
//!
//! Each slice's side is a `GL_TRIANGLE_STRIP` of `[point, point + 10]` pairs, drawn while the
//! back-face cull the world pass left on is still on: a filled 10-pixel band in the half-bright
//! colour under the pie's near edge, and nothing of it over the pie's top. These render the real
//! chart through the UI phase and read the pixels back. They skip (loudly) without a wgpu adapter
//! or the vanilla jar (the legend's font). `MC_PIE_DUMP=<folder>` also writes the frame as a PNG.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::debug::{profiler_color, DebugOverlay, ProfilerResult};
use mc_ui::widget::keys::Modifiers;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase};

const W: u32 = 854;
const H: u32 = 480;

fn gpu() -> Option<Gpu> {
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

fn assets() -> Option<ResourceManager> {
    let jars = std::env::var("MC_JARS_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
            Some(root.join("MCP-919-main").join("jars"))
        })?;
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(resources)
}

/// `guiScale = 1`: vanilla draws this chart in window pixels.
fn resolution() -> ScaledResolution {
    ScaledResolution::new(W, H, 1, false)
}

fn rgb(px: &[u8], x: f32, y: f32) -> [u8; 3] {
    let i = ((y.floor() as u32 * W + x.floor() as u32) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

fn colour(rgb: u32) -> [u8; 3] {
    [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8]
}

fn half(rgb: u32) -> [u8; 3] {
    colour(rgb).map(|channel| channel >> 1)
}

#[test]
fn the_pie_rim_is_a_filled_band_under_the_near_edge_only() {
    let Some(mut gpu) = gpu() else { return };
    let Some(resources) = assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    let mut font = FontRenderer::load(&resources, false, false).expect("load font");
    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    phase.renderer.upload_ascii_font(&gpu, font.data());
    phase.set_resolution(resolution());

    let mut overlay = DebugOverlay::new();
    overlay.on_f3(Modifiers::SHIFT);
    let results = [
        ProfilerResult::root(),
        ProfilerResult::new("tick", 50.0, 50.0),
        ProfilerResult::new("render", 30.0, 30.0),
        ProfilerResult::new("sound", 20.0, 20.0),
    ];
    let ctx = ScreenCtx::new(resolution(), 0.0);
    let mut list = DrawList::new();
    overlay.draw_profiler_chart(&mut list, &mut font, &ctx, &results);

    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let px = gpu.read_pixels().expect("readback");
    if let Some(folder) = std::env::var_os("MC_PIE_DUMP") {
        let path = PathBuf::from(folder).join("profiler_pie.png");
        image::save_buffer(&path, &px, W, H, image::ColorType::Rgba8).expect("write the PNG");
        eprintln!("wrote {}", path.display());
    }

    // The pie: 160 pixels wide each way, 80 high, centred 10 + 160 from the right and 320 from
    // the bottom. "tick" is its first half, clockwise from the top (0 to 180 degrees).
    let (centre_x, centre_y) = (W as f32 - 170.0, H as f32 - 320.0);
    let edge = |degrees: f32| {
        let angle = degrees.to_radians();
        (
            centre_x + angle.sin() * 160.0,
            centre_y - angle.cos() * 80.0,
        )
    };
    let tick = profiler_color("tick");

    // Near side (lower right, 135 degrees): the band runs 10 pixels down from the edge.
    let (x, y) = edge(135.0);
    for down in [2.0, 5.0, 8.0] {
        assert_eq!(
            rgb(&px, x, y + down),
            half(tick),
            "the near rim, {down} pixels under the edge"
        );
    }
    assert_ne!(
        rgb(&px, x, y + 13.0),
        half(tick),
        "the band is 10 pixels deep"
    );

    // Far side (upper right, 45 degrees): under the edge is the pie's top, not the rim. With
    // the strip's far half drawn (no cull), these pixels would be the half colour.
    let (x, y) = edge(45.0);
    for down in [2.0, 5.0, 8.0] {
        assert_eq!(
            rgb(&px, x, y + down),
            colour(tick),
            "the pie's top, {down} pixels under the far edge"
        );
    }
}
