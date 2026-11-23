//! Headless GPU tests for S16's screens.
//!
//! Same technique as `headless_ui.rs`: render through `mc_render::Gpu::headless` and read the
//! framebuffer back, then compare against the *source texels* of the real 1.8.9 textures. A
//! screen that "renders without panicking" proves nothing; these assert that the hotbar's art,
//! the heart sprites, the wordmark and the F3 plates land on the exact pixels vanilla puts them
//! on, sampled from the exact texels vanilla samples.
//!
//! `guiScale` is pinned to 1 throughout, so one GUI pixel is one framebuffer pixel.

use std::path::{Path, PathBuf};

use mc_assets::{Language, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::common::TITLE_TEXTURE;
use mc_ui::screens::debug::{DebugInfo, DebugOverlay};
use mc_ui::screens::hud::{Hud, PlayerHud};
use mc_ui::screens::menu::MainMenuScreen;
use mc_ui::screens::multiplayer::ServerEntry;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, ICONS_TEXTURE, WIDGETS_TEXTURE};

const W: u32 = 480;
const H: u32 = 320;

fn gpu(w: u32, h: u32) -> Option<Gpu> {
    match Gpu::headless(
        w,
        h,
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

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn assets() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// Decode one of the jar's PNGs into RGBA plus its size.
fn texture(rm: &ResourceManager, path: &str) -> (Vec<u8>, u32, u32) {
    let bytes = rm
        .load(&ResourceLocation::minecraft(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgba8();
    let (w, h) = (img.width(), img.height());
    (img.into_raw(), w, h)
}

fn texel(src: &(Vec<u8>, u32, u32), x: u32, y: u32) -> [u8; 4] {
    let i = ((y * src.1 + x) * 4) as usize;
    [src.0[i], src.0[i + 1], src.0[i + 2], src.0[i + 3]]
}

struct Frame {
    pixels: Vec<u8>,
    width: u32,
}

impl Frame {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }
}

/// Render one draw list against a black background and hand back the framebuffer.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    w: u32,
    h: u32,
    list: DrawList,
    font: Option<&mc_ui::font::FontData>,
) -> Frame {
    let mut phase = UiPhase::new(gpu, w, h, 1, false);
    if let Some(data) = font {
        phase.renderer.upload_ascii_font(gpu, data);
    }
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    Frame {
        pixels: gpu.read_pixels().expect("readback"),
        width: w,
    }
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(W, H, 1, false), 0.0)
}

/// The hotbar frame and its selection cursor, texel for texel out of `widgets.png`.
#[test]
fn the_hotbar_blits_widgets_png_onto_vanillas_pixels() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let widgets = texture(&rm, WIDGETS_TEXTURE);
    assert_eq!((widgets.1, widgets.2), (256, 256));

    let hud = Hud::new();
    let c = ctx();
    let state = PlayerHud {
        selected_slot: 4,
        ..PlayerHud::default()
    };
    let mut list = DrawList::new();
    hud.draw_hotbar(&mut list, &c, &state);
    let f = render(&mut gpu, &rm, W, H, list, None);

    // The bar is 182x22 at (w/2 - 91, h - 22), sampling texels (0, 0)..(182, 22).
    let bx = (W / 2) - 91;
    let by = H - 22;
    let cursor_x0 = (W / 2) - 92 + 4 * 20;
    let mut compared = 0;
    for dy in 0..22u32 {
        for dx in 0..182u32 {
            // The selection cursor overlaps the bar, so skip its 24-pixel span, and only
            // compare fully opaque texels: `widgets.png`'s bar has anti-aliased edges whose
            // alpha blends against the clear colour.
            if (cursor_x0..cursor_x0 + 24).contains(&(bx + dx)) {
                continue;
            }
            let src = texel(&widgets, dx, dy);
            if src[3] != 255 {
                continue;
            }
            compared += 1;
            assert_eq!(f.at(bx + dx, by + dy), src, "hotbar texel ({dx}, {dy})");
        }
    }
    assert!(
        compared > 1200,
        "only {compared} opaque hotbar texels compared; the blit is not landing"
    );

    // The cursor is 24x22 at (w/2 - 92 + slot*20, h - 23), sampling texels (0, 22)..(24, 44).
    let cx = (W / 2) - 92 + 4 * 20;
    let cy = H - 23;
    let mut compared = 0;
    for dy in 0..22u32 {
        for dx in 0..24u32 {
            let src = texel(&widgets, dx, 22 + dy);
            if src[3] != 255 {
                continue;
            }
            compared += 1;
            assert_eq!(
                f.at(cx + dx, cy + dy),
                src,
                "cursor texel ({dx}, {})",
                22 + dy
            );
        }
    }
    assert!(
        compared > 50,
        "only {compared} opaque cursor texels compared"
    );
}

/// The health, armour and hunger rows out of `icons.png`.
#[test]
fn the_health_and_hunger_rows_blit_icons_png_onto_vanillas_pixels() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let icons = texture(&rm, ICONS_TEXTURE);

    let hud = Hud::new();
    let c = ctx();
    // Full health, no armour: ten containers with ten full hearts drawn straight on top.
    let mut list = DrawList::new();
    hud.draw_player_stats(&mut list, &c, &PlayerHud::default());
    let f = render(&mut gpu, &rm, W, H, list, None);

    let left = (W / 2) - 91;
    let base = H - 39;
    // The filled heart is texels (52, 0)..(61, 9). Compare every one of its 81 texels for the
    // leftmost heart, skipping the ones the transparent container shows through.
    let mut opaque = 0;
    for dy in 0..9u32 {
        for dx in 0..9u32 {
            let src = texel(&icons, 52 + dx, dy);
            if src[3] != 255 {
                continue;
            }
            opaque += 1;
            assert_eq!(f.at(left + dx, base + dy), src, "heart texel ({dx}, {dy})");
        }
    }
    assert!(
        opaque > 30,
        "the heart sprite should be mostly opaque, got {opaque}"
    );

    // The drumstick is texels (52, 27)..(61, 36), drawn at (w/2 + 91 - 9, h - 39) for i = 0.
    let food_x = (W / 2) + 91 - 9;
    let mut lit = 0;
    for dy in 0..9u32 {
        for dx in 0..9u32 {
            let src = texel(&icons, 52 + dx, 27 + dy);
            if src[3] != 255 {
                continue;
            }
            lit += 1;
            assert_eq!(
                f.at(food_x + dx, base + dy),
                src,
                "drumstick texel ({dx}, {dy})"
            );
        }
    }
    assert!(
        lit >= 15,
        "the drumstick sprite should have a solid core, got {lit} opaque texels"
    );
}

/// The armour bar is pushed up by the heart stack, and lands on `icons.png`'s row-9 sprites.
#[test]
fn the_armour_row_sits_above_a_two_row_heart_stack() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let icons = texture(&rm, ICONS_TEXTURE);
    let hud = Hud::new();
    let c = ctx();
    // 40 max health => 2 rows, pitch 10, so the armour row is at (h - 39) - 10 - 10.
    let state = PlayerHud {
        health: 40.0,
        max_health: 40.0,
        armor: 20,
        ..PlayerHud::default()
    };
    let mut list = DrawList::new();
    hud.draw_player_stats(&mut list, &c, &state);
    let f = render(&mut gpu, &rm, W, H, list, None);

    let left = (W / 2) - 91;
    let armour_y = (H - 39) - 10 - 10;
    // The full chestplate icon is texels (34, 9)..(43, 18).
    let mut checked = 0;
    for dy in 0..9u32 {
        for dx in 0..9u32 {
            let src = texel(&icons, 34 + dx, 9 + dy);
            if src[3] != 255 {
                continue;
            }
            checked += 1;
            assert_eq!(
                f.at(left + dx, armour_y + dy),
                src,
                "armour texel ({dx}, {dy})"
            );
        }
    }
    assert!(checked > 10, "the armour sprite should have opaque texels");
}

/// The XP bar's fill really is 183 pixels at full, sampled from `icons.png` row 69.
#[test]
fn the_experience_bar_fills_to_one_hundred_and_eighty_three_pixels() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let icons = texture(&rm, ICONS_TEXTURE);
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let data = mc_ui::font::FontData::load(&rm).expect("font data");
    let hud = Hud::new();
    let c = ctx();

    let mut list = DrawList::new();
    hud.draw_experience(
        &mut list,
        &mut font,
        &c,
        &PlayerHud {
            experience: 1.0,
            ..PlayerHud::default()
        },
    );
    let f = render(&mut gpu, &rm, W, H, list, Some(&data));

    let x = (W / 2) - 91;
    let y = H - 32 + 3;
    // Row 2 of the fill sprite is solid, so it is a clean comparison.
    for dx in [0u32, 1, 90, 181] {
        assert_eq!(
            f.at(x + dx, y + 2),
            texel(&icons, dx, 71),
            "xp fill column {dx}"
        );
    }
    // Vanilla's fill width is `(int)(experience * 183)`, one wider than the 182-texel sprite,
    // so a full bar samples one column past the artwork. That column is empty in `icons.png`,
    // which is why the bar's right end looks a pixel short at 100%. Reproduced, not fixed.
    assert_eq!(
        texel(&icons, 182, 71)[3],
        0,
        "the 183rd column is transparent"
    );
    assert_eq!(
        f.at(x + 182, y + 2),
        [0, 0, 0, 255],
        "so nothing is drawn there"
    );
}

/// The wordmark: two 155×44 halves of `title/minecraft.png`, at `(width/2 - 137, 30)`.
#[test]
fn the_main_menu_wordmark_blits_the_real_title_texture() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let title = texture(&rm, TITLE_TEXTURE);
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let data = mc_ui::font::FontData::load(&rm).expect("font data");

    let mut menu = MainMenuScreen::new("Hello!");
    let c = ctx();
    menu.init(&c, &Language::new("en_US"));
    let mut list = DrawList::new();
    // `panorama_steps = 0` keeps the six 256×256 skybox textures out of this test; the
    // panorama has its own.
    menu.draw(&mut list, &mut font, &c, (0, 0), 0, 0);
    let f = render(&mut gpu, &rm, W, H, list, Some(&data));

    let x0 = (W / 2) - 137;
    // Sample opaque texels of the left half: the "MINECR" glyphs are solid white.
    let mut compared = 0;
    for (dx, dy) in [(10u32, 10u32), (40, 20), (80, 30), (120, 15), (154, 43)] {
        let src = texel(&title, dx, dy);
        if src[3] != 255 {
            continue;
        }
        compared += 1;
        assert_eq!(f.at(x0 + dx, 30 + dy), src, "title texel ({dx}, {dy})");
    }
    assert!(compared >= 2, "the sample points should hit opaque art");

    // The right half starts at x0 + 155 and samples from v = 45.
    for (dx, dy) in [(10u32, 10u32), (60, 20), (100, 30)] {
        let src = texel(&title, dx, 45 + dy);
        if src[3] != 255 {
            continue;
        }
        assert_eq!(
            f.at(x0 + 155 + dx, 30 + dy),
            src,
            "title right-half texel ({dx}, {})",
            45 + dy
        );
    }
}

/// The panorama really samples the six skybox textures and covers the whole screen.
#[test]
fn the_panorama_covers_every_pixel_with_skybox_texels() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let data = mc_ui::font::FontData::load(&rm).expect("font data");

    let mut menu = MainMenuScreen::new("");
    let c = ctx();
    menu.init(&c, &Language::new("en_US"));
    let mut list = DrawList::new();
    // 16 subdivisions per face, and no other geometry: just the cube.
    menu.panorama.draw(&mut list, c.width, c.height, 0.0, 16);
    let f = render(&mut gpu, &rm, W, H, list, Some(&data));

    // Not one pixel of the black clear colour may survive: the cube surrounds the camera.
    let mut black = 0;
    for y in 0..H {
        for x in 0..W {
            if f.at(x, y) == [0, 0, 0, 255] {
                black += 1;
            }
        }
    }
    // The panorama textures do contain near-black pixels (night sky, cave mouths), so allow a
    // small residue but not a hole.
    assert!(
        black < (W * H) / 50,
        "{black} of {} pixels are pure black; the panorama has a hole",
        W * H
    );
}

/// The F3 overlay's plates and text actually reach the framebuffer, flush to both edges.
#[test]
fn the_debug_overlay_plates_both_columns_against_the_screen_edges() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let data = mc_ui::font::FontData::load(&rm).expect("font data");

    let mut overlay = DebugOverlay::new();
    overlay.visible = true;
    let info = DebugInfo {
        fps: "120 fps".into(),
        position: (1.5, 64.0, -2.5),
        memory: (256 * 1024 * 1024, 512 * 1024 * 1024, 1024 * 1024 * 1024),
        ..DebugInfo::default()
    };
    let c = ctx();
    let mut list = DrawList::new();
    overlay.draw(&mut list, &mut font, &c, &info, None);
    let f = render(&mut gpu, &rm, W, H, list, Some(&data));

    // Java GuiOverlayDebug uses -1873784752 = 0x90505050: 0x50 * 0x90/255 rounds to 45.
    let plate = f.at(1, 1);
    assert_eq!(plate[3], 255);
    assert!(
        (44..=46).contains(&plate[0]) && plate[0] == plate[1] && plate[1] == plate[2],
        "the left plate should be a dark grey, got {plate:?}"
    );
    // The left column's first line has ink somewhere in its 9-pixel band.
    let inked = (2..200).any(|x| {
        let p = f.at(x, 4);
        p[0] > 100 && p[1] > 100
    });
    assert!(inked, "no text in the left column");

    // The right column ends one pixel in from the right edge.
    let right_plate = f.at(W - 2, 1);
    assert!(
        (44..=46).contains(&right_plate[0]),
        "the right plate should reach the edge, got {right_plate:?}"
    );
    assert_eq!(f.at(W - 1, 1), [0, 0, 0, 255], "and stop one pixel short");
}

/// A server-list row draws the placeholder icon out of `unknown_server.png` at 1:1.
///
/// The 32×32 favicon is the one blit in the whole GUI whose source texture is *not* 256×256;
/// getting the divisor wrong shrinks it to a quarter of a pixel, which this catches.
#[test]
fn a_server_row_draws_the_placeholder_icon_at_full_size() {
    let Some(mut gpu) = gpu(W, H) else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let unknown = texture(&rm, mc_ui::screens::common::UNKNOWN_SERVER_TEXTURE);
    // The shipped art is 128x128, but vanilla blits it with `texWidth = texHeight = 32.0F`, so
    // the 32-texel window is the *whole* texture and it lands 4:1 minified. Passing 256 here —
    // the size every other GUI blit uses — would shrink it to an eighth of a pixel.
    assert_eq!((unknown.1, unknown.2), (128, 128));
    let scale = unknown.1 / 32;
    let mut font = FontRenderer::load(&rm, false, false).expect("font");

    let data = mc_ui::font::FontData::load(&rm).expect("font data");

    // Draw the row on its own: `MultiplayerScreen::draw` paints the list's top shadow band
    // over the first few rows, and the claim under test is the row's own blit.
    let entry = ServerEntry::new("Test", "127.0.0.1");
    let mut list = DrawList::new();
    let icon_x = 40u32;
    let icon_y = 60u32;
    mc_ui::screens::multiplayer::row::draw_row(
        &mut list,
        &mut font,
        &entry,
        0,
        icon_x as i32,
        icon_y as i32,
        mc_ui::screens::multiplayer::row::LIST_WIDTH,
        (0, 0),
        false,
        false,
        false,
        0,
        false,
    );
    let f = render(&mut gpu, &rm, W, H, list, Some(&data));

    // Nearest sampling of a 4:1 minification picks texel `4k + 2` under output pixel `k`.
    let mut compared = 0;
    for dy in 0..32u32 {
        for dx in 0..32u32 {
            let src = texel(&unknown, dx * scale + scale / 2, dy * scale + scale / 2);
            if src[3] != 255 {
                continue;
            }
            compared += 1;
            assert_eq!(
                f.at(icon_x + dx, icon_y + dy),
                src,
                "placeholder pixel ({dx}, {dy}) should sample texel ({}, {})",
                dx * scale + scale / 2,
                dy * scale + scale / 2
            );
        }
    }
    assert!(
        compared > 500,
        "only {compared} of 1024 placeholder pixels were opaque; the blit is wrong"
    );
    // And it really is 32 pixels wide: the column past its right edge is background.
    assert_eq!(
        f.at(icon_x + 32, icon_y + 16),
        [0, 0, 0, 255],
        "the icon spilled past 32 pixels"
    );
}
