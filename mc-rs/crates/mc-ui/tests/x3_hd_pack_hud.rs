//! "HUD element size changes with texture pack".
//!
//! Vanilla addresses every GUI sprite sheet in 1/256 units whatever the PNG's real size
//! (`Gui.drawTexturedModalRect`), measures the default font's widths in 8-pixel cell units
//! (`FontRenderer.readFontTexture`: `8 / cell_width`) and samples glyphs in 1/128 units, so an HD
//! pack changes how sharp the HUD is, never how big it is.
//!
//! This test builds two fake HD packs from the jar's own art, every texel blown up to a 2x2 and a
//! 4x4 block (nearest neighbour, so the art is unchanged and only the resolution differs):
//! `gui/widgets.png`, `gui/icons.png`, `gui/container/inventory.png` and `font/ascii.png`. It
//! then draws the whole vanilla HUD (hotbar and selection, hearts with absorption, armour,
//! food, air, the XP bar and level, the crosshair, the boss bar, the held-item name, the action
//! bar, a title, the potion list), the tab list's ping bars and the mod HUD's potion-status
//! icons through each pack, and proves:
//!
//! 1. the draw lists are **identical** (every vertex position, UV and colour): nothing in the
//!    HUD is sized from a texture's pixel size, text included;
//! 2. the GPU frames are **pixel-identical** (real `UI 800` phase, headless readback): the UVs
//!    land on the same art at every resolution.
//!
//! Set `X3_HD_PACK_OUT=<dir>` to also write the two packs there (to try them in game).
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{FolderResourcePack, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::modhud::{HudCtx, HudData, HudEffect, ModHud};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::hud::{ActiveEffect, Hud, PlayerHud};
use mc_ui::screens::tab_list::{TabListOverlay, TabRow};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase};

const W: u32 = 480;
const H: u32 = 320;

/// The GUI sheets and the font, each upscaled by the pack's factor.
const UPSCALED: &[&str] = &[
    "textures/gui/widgets.png",
    "textures/gui/icons.png",
    "textures/gui/container/inventory.png",
    "textures/font/ascii.png",
];

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
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn jar() -> Option<PathBuf> {
    let dir = match std::env::var("MC_JARS_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .join("MCP-919-main")
            .join("jars"),
    };
    let jar = dir.join("versions").join("1.8.9").join("1.8.9.jar");
    jar.is_file().then_some(jar)
}

fn vanilla(jar: &Path) -> ResourceManager {
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(jar).expect("open the jar")));
    rm
}

/// Write a pack whose `UPSCALED` textures are the jar's, `factor` times larger.
fn write_hd_pack(jar_rm: &ResourceManager, root: &Path, factor: u32) {
    let textures = root.join("assets").join("minecraft");
    for path in UPSCALED {
        let bytes = jar_rm
            .load(&ResourceLocation::minecraft(*path))
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .to_rgba8();
        let big = image::imageops::resize(
            &img,
            img.width() * factor,
            img.height() * factor,
            image::imageops::FilterType::Nearest,
        );
        let out = textures.join(path);
        std::fs::create_dir_all(out.parent().unwrap()).unwrap();
        big.save(&out)
            .unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    }
    std::fs::write(
        root.join("pack.mcmeta"),
        format!(
            "{{\"pack\":{{\"pack_format\":1,\"description\":\"X3 fake HD pack: the vanilla \
             HUD art at {factor}x\"}}}}"
        ),
    )
    .unwrap();
}

fn with_pack(jar: &Path, pack: &Path) -> ResourceManager {
    let mut rm = vanilla(jar);
    rm.push_pack(Box::new(
        FolderResourcePack::open(pack).expect("open the fake pack"),
    ));
    rm
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(W, H, 1, false), 0.5)
}

/// Everything the HUD draws from the GUI sheets and the font, in one list.
fn draw_hud(font: &mut FontRenderer) -> DrawList {
    let c = ctx();
    let mut hud = Hud::new();
    hud.set_boss_status(150.0, 300.0, "\u{a7}dEnder Dragon", false);
    hud.set_record_playing("\u{a7}aAction bar: +12 coins", false);
    hud.display_title("\u{a7}6VICTORY", "You were the last one standing");
    hud.held_item_name = "\u{a7}bDiamond Sword".to_owned();
    hud.highlight_ticks = 30;
    let state = PlayerHud {
        health: 13.0,
        absorption: 4.0,
        armor: 14,
        food: 15,
        air: 120,
        underwater: true,
        experience: 0.6,
        experience_level: 27,
        selected_slot: 3,
        show_effect_overlay: true,
        effects: vec![
            ActiveEffect {
                icon_index: 0,
                name: "Speed II".to_owned(),
                duration: "1:30".to_owned(),
                beneficial: true,
            },
            ActiveEffect {
                icon_index: 3,
                name: "Strength".to_owned(),
                duration: "0:12".to_owned(),
                beneficial: true,
            },
        ],
        ..PlayerHud::default()
    };
    let mut d = DrawList::new();
    hud.draw(&mut d, font, None, &c, &state);

    // The tab list's five-bar ping sprites come out of icons.png too.
    let rows: Vec<TabRow> = [12, 180, 420, 800, 1500, -1]
        .iter()
        .enumerate()
        .map(|(i, &ping)| TabRow::new(format!("Player{i}"), ping))
        .collect();
    TabListOverlay::new().draw(
        &mut d,
        font,
        c.width,
        &rows,
        Some("Header"),
        None,
        None,
        false,
    );

    // The mod HUD's potion-status icons come out of inventory.png.
    let mut settings = mc_mods::ModSettings::new();
    settings.set_enabled("potion_status", true);
    let mut mod_hud = ModHud::new();
    mod_hud.sync(&settings);
    let data = HudData {
        effects: vec![
            HudEffect {
                id: 1,
                amplifier: 1,
                duration_ticks: 1800,
                duration_max: false,
            },
            HudEffect {
                id: 5,
                amplifier: 0,
                duration_ticks: 240,
                duration_max: false,
            },
        ],
        ..HudData::default()
    };
    let hud_ctx = HudCtx {
        data: &data,
        settings: &settings,
        screen: (W as f32, H as f32),
        pixel_scale: 1.0,
        editing: false,
    };
    mod_hud.draw(&hud_ctx, font, &mut d);
    d
}

fn render(gpu: &mut Gpu, rm: &ResourceManager, list: DrawList) -> Vec<u8> {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(W, H, 1, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

#[test]
fn the_hud_is_the_same_size_and_the_same_pixels_with_a_2x_or_4x_pack() {
    let Some(jar) = jar() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let jar_rm = vanilla(&jar);
    let dir = tempfile::tempdir().expect("tempdir");
    let packs: Vec<(u32, PathBuf)> = [2, 4]
        .into_iter()
        .map(|factor| {
            let root = dir.path().join(format!("x3-hd-{factor}x"));
            write_hd_pack(&jar_rm, &root, factor);
            if let Ok(out) = std::env::var("X3_HD_PACK_OUT") {
                write_hd_pack(
                    &jar_rm,
                    &Path::new(&out).join(format!("X3 HD {factor}x")),
                    factor,
                );
            }
            (factor, root)
        })
        .collect();

    // The packs really are HD: the resource manager hands out the big sheets.
    for (factor, root) in &packs {
        let rm = with_pack(&jar, root);
        let bytes = rm
            .load(&ResourceLocation::minecraft("textures/gui/widgets.png"))
            .unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!(
            img.width(),
            256 * factor,
            "the {factor}x pack's widgets.png"
        );
    }

    // 1. Geometry: the draw lists are identical, vertex for vertex.
    let mut vanilla_font = FontRenderer::load(&jar_rm, false, false).expect("font");
    let reference = draw_hud(&mut vanilla_font);
    assert!(reference.vertices().len() > 500, "the HUD drew too little");
    for (factor, root) in &packs {
        let rm = with_pack(&jar, root);
        let mut font = FontRenderer::load(&rm, false, false).expect("HD font");
        let list = draw_hud(&mut font);
        assert_eq!(
            list.vertices().len(),
            reference.vertices().len(),
            "{factor}x: a different number of vertices"
        );
        for (i, (a, b)) in list.vertices().iter().zip(reference.vertices()).enumerate() {
            assert_eq!(a, b, "{factor}x: vertex {i} moved (position, UV or colour)");
        }
        assert_eq!(list, reference, "{factor}x: batches differ");
    }

    // 2. Pixels: the real UI phase draws the same frame from each pack.
    let Some(mut gpu) = gpu() else { return };
    let base = render(&mut gpu, &jar_rm, draw_hud(&mut vanilla_font));
    let lit = base.chunks_exact(4).filter(|p| p[..3] != [0, 0, 0]).count();
    assert!(lit > 5_000, "only {lit} lit pixels: the HUD did not render");
    for (factor, root) in &packs {
        let rm = with_pack(&jar, root);
        let mut font = FontRenderer::load(&rm, false, false).expect("HD font");
        let frame = render(&mut gpu, &rm, draw_hud(&mut font));
        let differing = base
            .chunks_exact(4)
            .zip(frame.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differing, 0,
            "{factor}x: {differing} pixels differ from vanilla"
        );
    }
}
