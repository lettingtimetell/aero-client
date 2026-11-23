//! Headless GPU tests for the achievements map (S16).
//!
//! Same technique as `headless_screens.rs`: render through the real `UI 800` phase with
//! `mc_render::Gpu::headless`, read the framebuffer back, and compare against the *source
//! texels* of the real `1.8.9.jar`. The map is the one screen in this section that is mostly
//! made of art, so "it draws" is not a claim worth making without pixels behind it.
//!
//! What is checked here:
//!
//! * the tiled parchment backdrop, tile by tile, against `assets/minecraft/textures/blocks/*.png`
//!   with the per-row shading applied — including that the tile at a given map coordinate is the
//!   *particular* block `GuiAchievements`' seeded cascade picks;
//! * a connecting line's exact pixel span, which is where vanilla's
//!   `drawHorizontalLine`/`drawVerticalLine` asymmetry becomes visible;
//! * the link's arrow head, the 26×26 cell plate and the 256×202 window frame, texel for texel
//!   out of `achievement_background.png`.
//!
//! `guiScale` is 1 and the zoom is 1×, so one GUI pixel is one framebuffer pixel and the map's
//! transform is a pure translation. The UI sampler is `Nearest`, so the blits are texel-exact.
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{bake_all, install, Language, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::common::ACHIEVEMENT_BACKGROUND_TEXTURE;
use mc_ui::screens::stats::{
    backdrop_tile, AchievementGrid, AchievementNode, AchievementsScreen, MapView,
    BACKDROP_TEXTURES, FRAME_HEIGHT, FRAME_WIDTH, MAP_ORIGIN_X, MAP_ORIGIN_Y,
};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 320;
const H: u32 = 256;
/// An arbitrary but fixed `Session.getPlayerID().hashCode()`, so the scatter is reproducible.
const PLAYER_HASH: i32 = -991_726_233;

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

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn baked(rm: &ResourceManager) -> mc_assets::Atlas {
    let (baked, _report) = bake_all(rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    install(baked);
    atlas
}

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
}

impl Frame {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * W + x) * 4) as usize;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }
}

fn atlas_key() -> UiTexture {
    UiTexture::Resource(ResourceLocation::minecraft("mc-ui/achievements-atlas"))
}

fn render(gpu: &mut Gpu, rm: &ResourceManager, atlas: &mc_assets::Atlas, list: DrawList) -> Frame {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    phase
        .renderer
        .upload(gpu, atlas_key(), &atlas.texture, atlas.w, atlas.h);
    let font = mc_ui::font::FontData::load(rm).expect("font data");
    phase.renderer.upload_ascii_font(gpu, &font);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(W, H, 1, false));
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    Frame {
        pixels: gpu.read_pixels().expect("readback"),
    }
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(W, H, 1, false), 0.0)
}

fn node(key: &str, column: i32, row: i32, parent: Option<&str>, unlocked: bool) -> AchievementNode {
    AchievementNode {
        key: key.into(),
        column,
        row,
        parent: parent.map(ToOwned::to_owned),
        unlocked,
        special: false,
        icon: None,
    }
}

/// The whole parchment: every tile of the backdrop is the block vanilla's seeded cascade picks,
/// drawn from that block's own texture, shaded by its map row.
#[test]
fn the_parchment_backdrop_tiles_real_block_textures_with_vanillas_row_shading() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let atlas = baked(&rm);

    // A view whose origin is well inside the framebuffer, at 1x and scrolled to the origin, so
    // tile `(col, row)` lands at exactly `origin + (col * 16, row * 16)`.
    let view = MapView {
        origin: (32, 27),
        scroll: (0, 0),
        zoom: 1.0,
    };
    let mut screen = AchievementsScreen::new(AchievementGrid::default());
    screen.player_id_hash = PLAYER_HASH;

    let mut list = DrawList::new();
    screen.draw_backdrop(&mut list, &atlas_key(), &view);
    assert!(!list.is_empty(), "the backdrop drew nothing");
    let f = render(&mut gpu, &rm, &atlas, list);

    // java: `k1 = i + 288 >> 4`, `l1 = j + 288 >> 4` with i = j = 0.
    let (base_col, base_row) = (288 >> 4, 288 >> 4);
    let mut compared = 0;
    let mut blocks_seen = std::collections::BTreeSet::new();
    // Only the tiles fully inside the framebuffer; the loop runs past 224x155 by design.
    for tile_row in 0..9u32 {
        let shade = 0.6 - (base_row + tile_row as i32) as f32 / 25.0 * 0.3;
        for tile_col in 0..12u32 {
            let which = backdrop_tile(
                PLAYER_HASH,
                base_col + tile_col as i32,
                base_row + tile_row as i32,
            );
            blocks_seen.insert(which);
            let src = texture(&rm, &format!("{}.png", BACKDROP_TEXTURES[which]));
            assert_eq!((src.1, src.2), (16, 16), "{}", BACKDROP_TEXTURES[which]);
            let (x0, y0) = (
                view.origin.0 as u32 + tile_col * 16,
                view.origin.1 as u32 + tile_row * 16,
            );
            for dy in 0..16u32 {
                for dx in 0..16u32 {
                    let t = texel(&src, dx, dy);
                    if t[3] != 255 {
                        continue;
                    }
                    let got = f.at(x0 + dx, y0 + dy);
                    for ch in 0..3 {
                        let want = (f32::from(t[ch]) * shade).round() as i32;
                        assert!(
                            (i32::from(got[ch]) - want).abs() <= 1,
                            "tile ({tile_col}, {tile_row}) = {}, texel ({dx}, {dy}) ch {ch}: \
                             got {got:?} want ~{want} (shade {shade})",
                            BACKDROP_TEXTURES[which]
                        );
                    }
                    compared += 1;
                }
            }
        }
    }
    assert!(
        compared > 20_000,
        "only {compared} backdrop texels compared"
    );
    assert!(
        blocks_seen.len() >= 3,
        "the scatter should show several blocks, saw {blocks_seen:?}"
    );
}

/// A connecting line on real pixels, including the reason its two halves disagree: the
/// horizontal segment covers both endpoints, the vertical one covers neither.
#[test]
fn a_connecting_line_covers_exactly_vanillas_pixels() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let atlas = baked(&rm);

    // Child at column 3 / row 3, parent (unlocked) at column 1 / row 1: the link runs left and
    // then up, so both halves are non-degenerate and the arrow points left.
    let grid = AchievementGrid::new(vec![
        node("parent", 1, 1, None, true),
        node("child", 3, 3, Some("parent"), false),
    ]);
    let screen = AchievementsScreen::new(grid);
    let view = MapView {
        origin: (32, 27),
        scroll: (0, 0),
        zoom: 1.0,
    };
    let mut list = DrawList::new();
    screen.draw_connections(&mut list, &view);
    let f = render(&mut gpu, &rm, &atlas, list);

    // java: k5 = 3*24 + 11 = 83, l5 = 83, j6 = 1*24 + 11 = 35, k6 = 35.
    let (ox, oy) = (view.origin.0 as u32, view.origin.1 as u32);
    // The child is reachable (its parent is unlocked), so the link is `-16711936`: pure green.
    let green = [0, 255, 0, 255];

    // drawHorizontalLine(83, 35, 83) -> drawRect(35, 83, 84, 84): columns 35..=83 inclusive.
    // Columns 65..=71 are skipped: the arrow head (`k5 - 11 - 7 = 65`, 7 wide) is drawn *over*
    // the line there, and its own texels are partly transparent.
    for x in (35..=83u32).filter(|x| !(65..=71).contains(x)) {
        assert_eq!(f.at(ox + x, oy + 83), green, "horizontal link at x = {x}");
    }
    assert_eq!(
        f.at(ox + 34, oy + 83),
        [0, 0, 0, 255],
        "one past the left end"
    );
    assert_eq!(
        f.at(ox + 84, oy + 83),
        [0, 0, 0, 255],
        "one past the right end"
    );
    assert_eq!(
        f.at(ox + 83, oy + 84),
        [0, 0, 0, 255],
        "the line is one pixel tall"
    );

    // drawVerticalLine(35, 83, 35) -> drawRect(35, 36, 36, 83): rows 36..=82, so the row the
    // horizontal line sits on (83) and the row at the parent end (35) are both *missed*.
    for y in 36..=82u32 {
        assert_eq!(f.at(ox + 35, oy + y), green, "vertical link at y = {y}");
    }
    assert_eq!(
        f.at(ox + 35, oy + 35),
        [0, 0, 0, 255],
        "the vertical line misses its start row — vanilla's `startY + 1`"
    );
    assert_eq!(
        f.at(ox + 36, oy + 60),
        [0, 0, 0, 255],
        "and it is one pixel wide"
    );
    // The corner where the two halves meet is covered by the horizontal one only.
    assert_eq!(f.at(ox + 35, oy + 83), green);
}

/// The link's arrow head, the cell plate and the window frame all come out of
/// `achievement_background.png`, texel for texel.
#[test]
fn the_arrow_head_plate_and_frame_are_texel_exact_against_the_real_texture() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let atlas = baked(&rm);
    let art = texture(&rm, ACHIEVEMENT_BACKGROUND_TEXTURE);
    assert_eq!((art.1, art.2), (256, 256), "the sheet is 256x256");

    let grid = AchievementGrid::new(vec![
        node("parent", 1, 3, None, true),
        node("child", 3, 3, Some("parent"), true),
    ]);
    let screen = AchievementsScreen::new(grid);
    let view = MapView {
        origin: (32, 27),
        scroll: (0, 0),
        zoom: 1.0,
    };
    let (ox, oy) = (view.origin.0 as u32, view.origin.1 as u32);

    // ---- the arrow head: k5 = 83 > j6 = 35, so `drawTexturedModalRect(83 - 18, 78, 114, 234, 7, 11)`.
    let mut list = DrawList::new();
    screen.draw_connections(&mut list, &view);
    let f = render(&mut gpu, &rm, &atlas, list);
    let mut compared = 0;
    for dy in 0..11u32 {
        for dx in 0..7u32 {
            let t = texel(&art, 114 + dx, 234 + dy);
            if t[3] != 255 {
                continue;
            }
            compared += 1;
            assert_eq!(
                f.at(ox + 65 + dx, oy + 78 + dy),
                t,
                "arrow texel ({dx}, {dy})"
            );
        }
    }
    assert!(
        compared > 20,
        "only {compared} opaque arrow texels compared"
    );

    // ---- an unlocked cell plate: `drawTexturedModalRect(l6 - 2, j7 - 2, 0, 202, 26, 26)` under
    // a 0.75 tint.
    let mut list = DrawList::new();
    screen.draw_cells(&mut list, None, &view);
    let f = render(&mut gpu, &rm, &atlas, list);
    // The parent is at column 1 / row 3 -> local (24, 72), plate at (22, 70).
    let mut compared = 0;
    for dy in 0..26u32 {
        for dx in 0..26u32 {
            let t = texel(&art, dx, 202 + dy);
            if t[3] != 255 {
                continue;
            }
            compared += 1;
            let got = f.at(ox + 22 + dx, oy + 70 + dy);
            for ch in 0..3 {
                let want = (f32::from(t[ch]) * 0.75).round() as i32;
                assert!(
                    (i32::from(got[ch]) - want).abs() <= 1,
                    "plate texel ({dx}, {dy}) ch {ch}: got {got:?} want ~{want}"
                );
            }
        }
    }
    assert!(
        compared > 300,
        "only {compared} opaque plate texels compared"
    );

    // ---- the window frame, drawn last and untinted, at the screen's centred frame position.
    let mut lang = Language::new("en_US");
    lang.insert("gui.done", "Done");
    lang.insert("gui.achievements", "Achievements");
    let mut screen = AchievementsScreen::new(AchievementGrid::default());
    screen.done_loading();
    let c = ctx();
    screen.init(&c, &lang);
    let mut font = mc_ui::font::FontRenderer::load(&rm, false, false).expect("font");
    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, None, &c, (-99, -99), &lang);
    let f = render(&mut gpu, &rm, &atlas, list);
    let fr = screen.frame();
    assert_eq!(
        (fr.x, fr.y),
        ((W as i32 - FRAME_WIDTH) / 2, (H as i32 - FRAME_HEIGHT) / 2)
    );
    // Sample the frame's own border, which is outside both the map window and the Done button.
    let mut compared = 0;
    for dy in 0..FRAME_HEIGHT as u32 {
        for dx in 0..8u32 {
            let t = texel(&art, dx, dy);
            if t[3] != 255 {
                continue;
            }
            compared += 1;
            assert_eq!(
                f.at(fr.x as u32 + dx, fr.y as u32 + dy),
                t,
                "frame texel ({dx}, {dy})"
            );
        }
    }
    assert!(
        compared > 800,
        "only {compared} opaque frame texels compared"
    );
}

/// The map's origin really is `(frame.x + 16, frame.y + 17)`, so the whole layer is where
/// `translate(i1, j1, -200)` puts it.
#[test]
fn the_map_layer_sits_at_the_frames_sixteen_seventeen_inset() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let atlas = baked(&rm);

    let mut lang = Language::new("en_US");
    lang.insert("gui.done", "Done");
    // Column 2 / row 2, so the 26×26 plate lands well inside the frame's window rather than
    // under its opaque border — which is exactly what vanilla's own frame-over-map order does
    // to an achievement sitting at the map's top-left corner.
    let grid = AchievementGrid::new(vec![node("only", 2, 2, None, true)]);
    let mut screen = AchievementsScreen::new(grid);
    screen.done_loading();
    let c = ctx();
    screen.init(&c, &lang);
    screen.set_scroll(0.0, 0.0);
    let fr = screen.frame();

    let mut list = DrawList::new();
    let mut font = mc_ui::font::FontRenderer::load(&rm, false, false).expect("font");
    // Mouse parked off-screen so no tooltip covers the map.
    screen.draw(
        &mut list,
        &mut font,
        Some(&atlas_key()),
        &c,
        (-99, -99),
        &lang,
    );
    let f = render(&mut gpu, &rm, &atlas, list);

    let art = texture(&rm, ACHIEVEMENT_BACKGROUND_TEXTURE);
    // `draw` clamps the scroll first — a one-node tree's bounds are tight, so `set_scroll(0, 0)`
    // does *not* survive to the map view. Read the view back and derive the plate's position
    // from it, which is what makes the `+16 / +17` inset the thing actually under test.
    let v = screen.map_view(0.0);
    assert_eq!(
        v.origin,
        (fr.x + MAP_ORIGIN_X, fr.y + MAP_ORIGIN_Y),
        "the map layer's origin"
    );
    assert_eq!(v.zoom, 1.0);
    let (px, py) = (
        (v.origin.0 + 2 * 24 - v.scroll.0 - 2) as u32,
        (v.origin.1 + 2 * 24 - v.scroll.1 - 2) as u32,
    );
    let mut compared = 0;
    for dy in 0..26u32 {
        for dx in 0..26u32 {
            let t = texel(&art, dx, 202 + dy);
            if t[3] != 255 {
                continue;
            }
            let got = f.at(px + dx, py + dy);
            for ch in 0..3 {
                let want = (f32::from(t[ch]) * 0.75).round() as i32;
                assert!(
                    (i32::from(got[ch]) - want).abs() <= 1,
                    "plate texel ({dx}, {dy}) ch {ch}: got {got:?} want ~{want}"
                );
            }
            compared += 1;
        }
    }
    assert!(compared > 300, "only {compared} plate texels compared");
}

/// The map is drawn through `scale(1 / zoom)`, and the item icon in each cell reaches that scale
/// through S15's matrix stack rather than through [`MapView`]'s own arithmetic — because
/// `draw_item_icon_tinted` takes integer positions. This pins that the two agree: at 2× the
/// plate and the icon both come out half size, in the same place.
#[test]
fn zooming_out_halves_the_cell_plate_and_its_item_icon_together() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let _atlas = baked(&rm);

    let mut only = node("only", 0, 0, None, true);
    only.icon = Some(mc_ui::screens::container::stack_ext::ItemStack {
        item_id: mc_data::item_by_name("stick").expect("stick").0 as i16,
        count: 1,
        damage: 0,
        nbt: None,
    });
    let screen = AchievementsScreen::new(AchievementGrid::new(vec![only]));
    let key = atlas_key();

    let measure = |zoom: f32| {
        let view = MapView {
            origin: (0, 0),
            scroll: (0, 0),
            zoom,
        };
        let mut d = DrawList::new();
        screen.draw_cells(&mut d, Some(&key), &view);
        let plate = &d.vertices()[0..6];
        let plate_w = plate.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max)
            - plate.iter().map(|v| v.pos[0]).fold(f32::MAX, f32::min);
        let batch = d
            .batches()
            .iter()
            .find(|b| b.texture == key)
            .expect("the icon is drawn");
        let verts: Vec<[f32; 2]> = d.vertices()[batch.range.clone()]
            .iter()
            .map(|v| v.pos)
            .collect();
        let icon_x0 = verts.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        let icon_x1 = verts.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        (plate_w, icon_x0, icon_x1 - icon_x0)
    };

    let (plate1, x1, icon1) = measure(1.0);
    let (plate2, x2, icon2) = measure(2.0);
    assert_eq!(plate1, 26.0);
    assert_eq!(plate2, 13.0, "the plate halves");
    // The cell is at local (0, 0), so its icon starts at local (3, 3) — 3 at 1x, 1.5 at 2x.
    assert_eq!(x1, 3.0);
    assert_eq!(x2, 1.5, "the icon moved with the map, not against it");
    assert!(
        (icon1 - icon2 * 2.0).abs() < 1e-4,
        "the icon must halve too: {icon1} vs {icon2}"
    );
}
