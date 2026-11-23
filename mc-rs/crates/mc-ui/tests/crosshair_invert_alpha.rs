//! "Crosshair on my texture pack looks weird, why?"
//!
//! The pack's crosshair cell (`icons.png`, 16x16 at the origin) is a plain white 7x7 plus,
//! fully opaque, with no partial alpha. But three of its clear texels, (6,6), (8,6) and (6,8),
//! diagonal to the centre, store white RGB under alpha 0; every other clear texel is black, as
//! all of vanilla's are.
//!
//! Vanilla draws the crosshair with `tryBlendFuncSeparate(775, 769, 1, 0)` (colour
//! `src * (1 - dst) + dst * (1 - src)`, which ignores source alpha) and `enableAlpha()` under the
//! `alphaFunc(GL_GREATER, 0.1F)` `EntityRenderer` set before `renderGameOverlay`
//! (`GuiIngame.java:175-180`, `EntityRenderer.java:1168`). The alpha test discards those three
//! texels. mc-rs drew the batch without it, so they turned the scene under them to `1 - dst`:
//! three stray inverted dots at the plus's centre, at GUI scale. A black clear texel adds
//! nothing under that blend, which is why the vanilla sprite always looked right.
//!
//! These tests render the real `Hud::draw_crosshair` through `UiPhase` on the GPU and compare
//! every pixel against a CPU model of vanilla's GL state, over several backgrounds. Skips
//! (loudly) without a wgpu adapter.

use std::path::Path;

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::hud::Hud;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

/// 640x480 is the smallest window that `ScaledResolution` gives GUI scale 2, so every texel
/// covers 2x2 physical pixels, as on the user's screen.
const W: u32 = 640;
const H: u32 = 480;
const SCALE: i32 = 2;

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

fn resolution() -> ScaledResolution {
    let resolution = ScaledResolution::new(W, H, SCALE, false);
    assert_eq!(resolution.scale_factor(), SCALE);
    resolution
}

/// A 256x256 RGBA `icons.png` whose crosshair cell is `cell` and the rest is clear black.
fn icons_with_cell(cell: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut rgba = vec![0u8; 256 * 256 * 4];
    for y in 0..16 {
        for x in 0..16 {
            let i = ((y * 256 + x) * 4) as usize;
            rgba[i..i + 4].copy_from_slice(&cell(x, y));
        }
    }
    rgba
}

/// The reference pack's crosshair cell, byte for byte (dumped from its
/// `assets/minecraft/textures/gui/icons.png`).
fn pack_cell(x: u32, y: u32) -> [u8; 4] {
    let plus = (x == 7 && (4..=10).contains(&y)) || (y == 7 && (4..=10).contains(&x));
    if plus {
        [255, 255, 255, 255]
    } else if matches!((x, y), (6, 6) | (8, 6) | (6, 8)) {
        [255, 255, 255, 0]
    } else {
        [0, 0, 0, 0]
    }
}

/// The pack cell plus texels at the alpha-test boundary and partial-alpha colours: `0x19`
/// (0.098) must be discarded, `0x1A` (0.102) kept, and a kept texel inverts by its RGB whatever
/// its alpha.
fn threshold_cell(x: u32, y: u32) -> [u8; 4] {
    match (x, y) {
        (1, 1) => [255, 255, 255, 0x19],
        (2, 1) => [255, 255, 255, 0x1A],
        (3, 1) => [255, 255, 255, 0x80],
        (4, 1) => [255, 0, 0, 255],
        (5, 1) => [128, 64, 200, 0x80],
        (14, 14) => [255, 255, 255, 0x0A],
        _ => pack_cell(x, y),
    }
}

/// One full-screen background, drawn opaque before the crosshair.
#[derive(Clone, Copy)]
enum Background {
    Solid([u8; 3]),
    /// Four quadrants meeting at the screen centre, so each arm of the plus sees a different
    /// colour.
    Quadrants([[u8; 3]; 4]),
}

impl Background {
    fn at(self, x: u32, y: u32) -> [u8; 3] {
        match self {
            Self::Solid(rgb) => rgb,
            Self::Quadrants(q) => q[usize::from(x >= W / 2) + 2 * usize::from(y >= H / 2)],
        }
    }

    fn draw(self, d: &mut DrawList) {
        let argb = |[r, g, b]: [u8; 3]| {
            0xFF00_0000 | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
        };
        // GUI pixels: the 2x scale puts the physical centre on a GUI pixel boundary.
        let (gw, gh) = ((W / 2) as i32, (H / 2) as i32);
        match self {
            Self::Solid(rgb) => d.rect(0, 0, gw, gh, argb(rgb)),
            Self::Quadrants(q) => {
                d.rect(0, 0, gw / 2, gh / 2, argb(q[0]));
                d.rect(gw / 2, 0, gw, gh / 2, argb(q[1]));
                d.rect(0, gh / 2, gw / 2, gh, argb(q[2]));
                d.rect(gw / 2, gh / 2, gw, gh, argb(q[3]));
            }
        }
    }
}

const BACKGROUNDS: [Background; 5] = [
    Background::Solid([0, 0, 0]),
    Background::Solid([255, 255, 255]),
    Background::Solid([128, 128, 128]),
    Background::Solid([51, 102, 153]),
    Background::Quadrants([[200, 30, 30], [30, 200, 30], [30, 30, 200], [230, 210, 40]]),
];

/// Render `background` then the real vanilla crosshair, with `icons` bound as `icons.png`.
fn render(gpu: &mut Gpu, icons: &[u8], background: Background) -> Vec<u8> {
    let ctx = ScreenCtx::new(resolution(), 0.0);
    let mut d = DrawList::new();
    background.draw(&mut d);
    Hud::new().draw_crosshair(&mut d, &ctx);

    let mut phase = UiPhase::new(gpu, W, H, SCALE, false);
    phase
        .renderer
        .upload(gpu, UiTexture::icons().clone(), icons, 256, 256);
    phase.set_resolution(resolution());
    phase.submit(d);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

/// Vanilla's GL for one physical pixel: the crosshair texel under it (if any) passes
/// `alphaFunc(GL_GREATER, 0.1F)` and blends `src * (1 - dst) + dst * (1 - src)` per channel.
/// `alpha_test: false` is the pre-fix mc-rs behaviour, for the negative control.
fn reference(icons: &[u8], background: Background, x: u32, y: u32, alpha_test: bool) -> [u8; 3] {
    let dst = background.at(x, y);
    // GuiIngame: drawTexturedModalRect(i / 2 - 7, j / 2 - 7, 0, 0, 16, 16), in GUI pixels.
    let (gw, gh) = (W as i32 / SCALE, H as i32 / SCALE);
    let tx = x as i32 / SCALE - (gw / 2 - 7);
    let ty = y as i32 / SCALE - (gh / 2 - 7);
    if !(0..16).contains(&tx) || !(0..16).contains(&ty) {
        return dst;
    }
    let i = ((ty * 256 + tx) * 4) as usize;
    let src = &icons[i..i + 4];
    if alpha_test && f32::from(src[3]) / 255.0 <= 0.1 {
        return dst;
    }
    std::array::from_fn(|c| {
        let (s, d) = (f32::from(src[c]) / 255.0, f32::from(dst[c]) / 255.0);
        ((s * (1.0 - d) + d * (1.0 - s)) * 255.0).round() as u8
    })
}

fn rgb(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

/// Compare a whole frame with vanilla, allowing one step of blend rounding. Returns how many
/// pixels the pre-fix (no alpha test) model would have got wrong.
fn assert_matches_vanilla(gpu: &mut Gpu, label: &str, icons: &[u8]) -> usize {
    let mut old_wrong = 0;
    for background in BACKGROUNDS {
        let px = render(gpu, icons, background);
        for y in 0..H {
            for x in 0..W {
                let got = rgb(&px, x, y);
                let want = reference(icons, background, x, y, true);
                let close = got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 1);
                assert!(
                    close,
                    "{label}: pixel ({x}, {y}) is {got:?}, vanilla draws {want:?} (background \
                     {:?})",
                    background.at(x, y)
                );
                let old = reference(icons, background, x, y, false);
                if old.iter().zip(want).any(|(&o, w)| o.abs_diff(w) > 1) {
                    old_wrong += 1;
                }
            }
        }
    }
    old_wrong
}

#[test]
fn the_pack_crosshair_draws_exactly_as_vanillas_gl_does() {
    let Some(mut gpu) = gpu() else { return };
    let icons = icons_with_cell(pack_cell);
    let old_wrong = assert_matches_vanilla(&mut gpu, "pack cell", &icons);
    // Negative control: without the alpha test, the three clear-white texels (2x2 pixels each)
    // invert every background except where 1 - dst rounds to dst (mid grey 128 -> 127 is within
    // the rounding allowance), i.e. on four of the five backgrounds.
    assert_eq!(
        old_wrong,
        3 * 4 * 4,
        "the pre-fix model must differ at exactly the three diagonal texels"
    );
}

#[test]
fn the_alpha_test_boundary_and_partial_alpha_match_vanilla() {
    let Some(mut gpu) = gpu() else { return };
    let icons = icons_with_cell(threshold_cell);
    let old_wrong = assert_matches_vanilla(&mut gpu, "threshold cell", &icons);
    assert!(
        old_wrong > 0,
        "the threshold cell must exercise discarded texels"
    );
}

/// The resource pack the cell above was dumped from (never checked in): `MC_REFERENCE_PACK`, or
/// else the only folder in the workspace's `resourcepacks/`.
fn reference_pack(root: &Path) -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("MC_REFERENCE_PACK") {
        return Some(dir.into());
    }
    let mut dirs = std::fs::read_dir(root.join("resourcepacks"))
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir());
    let only = dirs.next()?;
    dirs.next().is_none().then_some(only)
}

/// The reference pack's actual file, when this machine has it, and the vanilla jar's crosshair
/// (whose clear texels are all black, so the alpha test changes none of its pixels).
#[test]
fn the_real_pack_and_vanilla_crosshairs_match_vanilla() {
    let Some(mut gpu) = gpu() else { return };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let pack = reference_pack(root);
    match pack {
        Some(pack) => {
            let png = std::fs::read(pack.join("assets/minecraft/textures/gui/icons.png"))
                .expect("the reference pack has an icons.png");
            let icons = image::load_from_memory(&png).expect("decode").to_rgba8();
            assert_eq!(
                icons.dimensions(),
                (256, 256),
                "a 16x pack: no rescale involved"
            );
            let icons = icons.into_raw();
            for y in 0..16 {
                for x in 0..16 {
                    let i = ((y * 256 + x) * 4) as usize;
                    assert_eq!(
                        icons[i..i + 4],
                        pack_cell(x, y),
                        "the synthetic cell must stay byte-identical to the pack at ({x}, {y})"
                    );
                }
            }
            assert_matches_vanilla(&mut gpu, "reference pack file", &icons);
        }
        None => eprintln!(
            "SKIP: no reference pack (set MC_REFERENCE_PACK, or keep one folder in resourcepacks/)"
        ),
    }

    let jar = std::env::var("MC_JARS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            root.parent()
                .expect("project root")
                .join("MCP-919-main/jars")
        })
        .join("versions/1.8.9/1.8.9.jar");
    let Ok(pack) = ZipResourcePack::open(&jar) else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    let png = rm
        .resource(&ResourceLocation::minecraft("textures/gui/icons.png"))
        .expect("vanilla icons.png")
        .data;
    let icons = image::load_from_memory(&png)
        .expect("decode")
        .to_rgba8()
        .into_raw();
    let old_wrong = assert_matches_vanilla(&mut gpu, "vanilla icons.png", &icons);
    assert_eq!(
        old_wrong, 0,
        "vanilla's clear texels are black: no pixel changes"
    );
}
