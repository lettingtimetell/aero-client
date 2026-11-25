//! The 1.7.10 player list and sidebar, rendered through `UiPhase` with the jar's font and read
//! back: the grid java17 `GuiIngame.renderGameOverlay` sizes by the server's maximum player
//! count, and the sidebar java17 `GuiIngame.renderScoreboard` hides above 15 scores, next to the
//! 1.8.9 list and sidebar for the same players and scores.
//!
//! `MC_V5_HUD_PNG_DIR=<dir>` also writes the frames there (`tab-*.png`, `sidebar-*.png`) for a
//! look.
//! Skips (loudly) without a wgpu adapter or the jar; set `MC_JARS_DIR`.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::hud::{Hud, Sidebar};
use mc_ui::screens::tab_list::{TabCellV5, TabListOverlay, TabObjective, TabRow};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase};

const W: u32 = 854;
const H: u32 = 480;
const SCALE: i32 = 2;
/// The scaled screen: 427 x 240.
const GUI_W: i32 = 427;

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

fn resources() -> Option<ResourceManager> {
    let dir = match std::env::var("MC_JARS_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .join("MCP-919-main")
            .join("jars"),
    };
    let jar = dir.join("versions").join("1.8.9").join("1.8.9.jar");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// A mid-grey backdrop, then whatever `paint` draws with the jar's font.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    paint: impl FnOnce(&mut DrawList, &mut FontRenderer, &ScreenCtx),
) -> Vec<u8> {
    let resolution = ScaledResolution::new(W, H, SCALE, false);
    let ctx = ScreenCtx::new(resolution, 0.0);
    let mut font = FontRenderer::load(rm, false, false).expect("the jar's font");
    let mut d = DrawList::new();
    d.rect(0, 0, ctx.width, ctx.height, 0xFF64_7896);
    paint(&mut d, &mut font, &ctx);
    let mut phase = UiPhase::new(gpu, W, H, SCALE, false);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &d);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(resolution);
    phase.submit(d);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

fn save(px: &[u8], name: &str) {
    let Some(dir) = std::env::var_os("MC_V5_HUD_PNG_DIR") else {
        return;
    };
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).expect("the PNG folder");
    let image = image::RgbaImage::from_raw(W, H, px.to_vec()).expect("frame size");
    let path = dir.join(format!("{name}.png"));
    image.save(&path).expect("write the PNG");
    eprintln!("saved {}", path.display());
}

fn rgb(px: &[u8], gui_x: i32, gui_y: i32) -> [u8; 3] {
    let (x, y) = ((gui_x * SCALE) as u32, (gui_y * SCALE) as u32);
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

const BACKDROP: [u8; 3] = [0x64, 0x78, 0x96];

/// Players in the order the server listed them, with a list objective's yellow score for some.
fn players() -> Vec<(String, i32, Option<i32>)> {
    [
        ("Zed", 40, Some(3)),
        ("Nora", 120, None),
        ("\u{a7}cRedTeam", 260, Some(12)),
        ("Alex", 480, Some(0)),
        ("Steve", 900, None),
        ("Lag", 1500, None),
        ("Fresh", -1, None),
    ]
    .into_iter()
    .map(|(name, ping, score)| (name.to_owned(), ping, score))
    .collect()
}

#[test]
fn the_1_7_10_player_list_is_a_grid_sized_by_the_maximum_players() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return;
    };
    for max_players in [20u8, 60] {
        let mut overlay = TabListOverlay::default();
        overlay.refresh_v5((1, 1, max_players), || {
            players()
                .into_iter()
                .map(|(name, ping_ms, score)| TabCellV5 {
                    name,
                    score: score.map(|points| format!("\u{a7}e{points}")),
                    ping_ms,
                })
                .collect()
        });
        let px = render(&mut gpu, &rm, |d, font, ctx| {
            overlay.draw_v5(d, font, ctx.width, max_players);
        });
        save(&px, &format!("tab-1.7.10-max{max_players}"));
        // The grid: 1 column of 150 for 20 slots, 3 of 100 for 60; every slot gets a plate,
        // the empty ones too, and the backdrop spans the grid only.
        let (columns, column_width, rows) = if max_players == 20 {
            (1, 150, 20)
        } else {
            (3, 100, 20)
        };
        let left = (GUI_W - columns * column_width) / 2;
        let last_plate_y = 10 + (rows - 1) * 9 + 4;
        let plate = rgb(&px, left + column_width - 30, last_plate_y);
        assert_ne!(
            plate, BACKDROP,
            "the last slot's plate is drawn ({max_players})"
        );
        assert_eq!(rgb(&px, left - 6, 14), BACKDROP, "nothing left of the grid");
        assert_eq!(
            rgb(&px, left + columns * column_width + 6, 14),
            BACKDROP,
            "nothing right of the grid"
        );
    }

    // 1.8.9 (negative control): the same players, sorted, in a grid sized by their number.
    let rows: Vec<TabRow> = players()
        .into_iter()
        .map(|(name, ping, _)| TabRow::new(name, ping))
        .collect();
    let mut overlay = TabListOverlay::default();
    let px = render(&mut gpu, &rm, |d, font, ctx| {
        overlay.draw(
            d,
            font,
            ctx.width,
            &rows,
            None,
            None,
            None::<&TabObjective>,
            false,
        );
    });
    save(&px, "tab-1.8.9");
}

fn scores(n: usize) -> Vec<(String, i32)> {
    let mut scores: Vec<(String, i32)> = (0..n)
        .map(|i| (format!("player{i}"), i as i32 * 3))
        .collect();
    scores.insert(0, ("#hidden".to_owned(), -1));
    scores
}

#[test]
fn the_1_7_10_sidebar_hides_above_fifteen_scores_and_shows_hash_names() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return;
    };
    let hud = Hud::new();
    // 1 `#hidden` + 14 others = 15: both draw; 1.7.10 shows the `#` row too.
    let fifteen = scores(14);
    let v5 = Sidebar::from_scores_v5("Board", &fifteen, str::to_owned).expect("15 rows");
    let v47 = Sidebar::from_scores("Board", &fifteen, str::to_owned);
    assert_eq!(v5.visible_rows().len(), 15);
    assert_eq!(v47.visible_rows().len(), 14);
    for (sidebar, name) in [(&v5, "sidebar-1.7.10-15"), (&v47, "sidebar-1.8.9-15")] {
        let px = render(&mut gpu, &rm, |d, font, ctx| {
            hud.draw_scoreboard(d, font, ctx, sidebar);
        });
        save(&px, name);
        // The board is drawn at the right edge, around the vertical middle.
        assert_ne!(rgb(&px, GUI_W - 3, 120), BACKDROP, "{name}");
    }
    // 16 + the `#` row = 17: 1.7.10 draws nothing; 1.8.9 its last 15.
    let seventeen = scores(16);
    assert!(Sidebar::from_scores_v5("Board", &seventeen, str::to_owned).is_none());
    let v47 = Sidebar::from_scores("Board", &seventeen, str::to_owned);
    let px = render(&mut gpu, &rm, |d, font, ctx| {
        hud.draw_scoreboard(d, font, ctx, &v47);
    });
    save(&px, "sidebar-1.8.9-17");
    assert_ne!(rgb(&px, GUI_W - 3, 120), BACKDROP);
    let px = render(&mut gpu, &rm, |_, _, _| {});
    save(&px, "sidebar-1.7.10-17");
    assert_eq!(rgb(&px, GUI_W - 3, 120), BACKDROP);
}
