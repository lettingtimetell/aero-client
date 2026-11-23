//! The achievements map, checked against values transcribed **from the Java**.
//!
//! `tests/javaref/gen_achievements_golden.py` is a line-by-line Python transcription of
//! `GuiAchievements.drawAchievementScreen`, `updateScreen`, `drawScreen`'s four clamps,
//! `Gui.drawHorizontalLine`/`drawVerticalLine`/`drawRect` and `StatFileWriter.func_150874_c`,
//! including `java.util.Random`'s LCG for the backdrop's ore scatter. It has never been run
//! against this crate; its output is committed as `tests/fixtures/achievements_golden.json` and
//! replayed here.
//!
//! Same reasoning as `vanilla_hud_debug.rs`: a port asserted against itself proves only
//! self-consistency. Every number below has to survive a second, independent reading of the
//! Java — which is what makes the vertical-line asymmetry, the `+ 288 >> 4` tile origin and the
//! `(l1 + l3) / 2` truncation checkable rather than merely plausible.
//!
//! No GPU and no jar are needed: with nothing installed, `mc_assets::sprite` hands back its
//! placeholder, so the tile *positions* and *tints* are still exact even though every tile
//! samples the same UV rect. Which block each tile shows is checked directly against
//! [`mc_ui::screens::stats::backdrop_tile`], and the real texels are `tests/achievements_map.rs`.

use std::path::{Path, PathBuf};

use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::stats::{
    backdrop_tile, AchievementGrid, AchievementNode, AchievementsScreen, MapView, BACKDROP_TEXTURES,
};
use mc_ui::widget::{unpack_argb, DrawList, ScaledResolution, UiTexture};
use serde_json::Value;

fn fixture() -> Value {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("achievements_golden.json");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("missing golden fixture {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("the fixture is valid JSON")
}

fn i32_at(v: &Value, i: usize) -> i32 {
    v[i].as_i64().expect("integer") as i32
}

fn tree(v: &Value) -> AchievementGrid {
    let nodes = v["tree"]
        .as_array()
        .expect("tree")
        .iter()
        .map(|n| AchievementNode {
            key: n["key"].as_str().expect("key").to_owned(),
            column: n["column"].as_i64().expect("column") as i32,
            row: n["row"].as_i64().expect("row") as i32,
            parent: n["parent"].as_str().map(ToOwned::to_owned),
            unlocked: n["unlocked"].as_bool().expect("unlocked"),
            special: n["special"].as_bool().expect("special"),
            icon: None,
        })
        .collect();
    AchievementGrid::new(nodes)
}

/// A view at the origin, so grid-local pixels *are* the numbers the Java prints.
fn view(scroll: (i32, i32)) -> MapView {
    MapView {
        origin: (0, 0),
        scroll,
        zoom: 1.0,
    }
}

/// Every quad in `d` as `[x0, y0, x1, y1]`, in submission order.
fn quad_rects(d: &DrawList) -> Vec<[i32; 4]> {
    d.vertices()
        .chunks(6)
        .map(|q| {
            let xs = q.iter().map(|v| v.pos[0]);
            let ys = q.iter().map(|v| v.pos[1]);
            [
                xs.clone().fold(f32::MAX, f32::min) as i32,
                ys.clone().fold(f32::MAX, f32::min) as i32,
                xs.fold(f32::MIN, f32::max) as i32,
                ys.fold(f32::MIN, f32::max) as i32,
            ]
        })
        .collect()
}

/// Every quad's first vertex colour, packed back to `0xAARRGGBB`.
fn quad_colors(d: &DrawList) -> Vec<u32> {
    d.vertices()
        .chunks(6)
        .map(|q| {
            let c = q[0].color;
            let b = |f: f32| (f * 255.0).round() as u32;
            (b(c[3]) << 24) | (b(c[0]) << 16) | (b(c[1]) << 8) | b(c[2])
        })
        .collect()
}

#[test]
fn every_backdrop_tile_matches_the_java_transcription() {
    let g = fixture();
    let samples = g["backdrop_tiles"].as_array().expect("backdrop_tiles");
    assert!(samples.len() > 1000, "the sweep must be wide");
    for s in samples {
        let hash = s["hash"].as_i64().expect("hash") as i32;
        let col = s["col"].as_i64().expect("col") as i32;
        let row = s["row"].as_i64().expect("row") as i32;
        let want = s["block"].as_str().expect("block");
        let got = BACKDROP_TEXTURES[backdrop_tile(hash, col, row)];
        assert_eq!(
            got,
            format!("textures/blocks/{want}"),
            "tile (hash {hash}, col {col}, row {row})"
        );
    }
}

#[test]
fn the_backdrop_tile_grid_lands_on_the_java_positions_and_shades() {
    let g = fixture();
    let atlas = UiTexture::gui("mc-ui/achievements-test-atlas");
    for (key, entry) in g["backdrop_layout"].as_object().expect("layout") {
        let scroll = (i32_at(&entry["scroll"], 0), i32_at(&entry["scroll"], 1));
        let hash = entry["hash"].as_i64().expect("hash") as i32;
        let mut screen = AchievementsScreen::new(tree(&g));
        screen.player_id_hash = hash;
        let v = view(scroll);
        let mut d = DrawList::new();
        screen.draw_backdrop(&mut d, &atlas, &v);

        let tiles = entry["tiles"].as_array().expect("tiles");
        let rects = quad_rects(&d);
        let colors = quad_colors(&d);
        assert_eq!(
            rects.len(),
            tiles.len(),
            "scroll {key}: tile count differs from the Java loop"
        );
        for (i, t) in tiles.iter().enumerate() {
            let (x, y) = (
                t["x"].as_i64().expect("x") as i32,
                t["y"].as_i64().expect("y") as i32,
            );
            assert_eq!(rects[i], [x, y, x + 16, y + 16], "scroll {key}, tile {i}");
            let want = (t["shade"].as_f64().expect("shade").max(0.0) * 255.0).round() as u32;
            let got = colors[i] & 0xFF;
            assert!(
                got.abs_diff(want) <= 1,
                "scroll {key}, tile {i}: shade {got} vs {want}"
            );
        }
    }
}

#[test]
fn every_connecting_line_and_arrow_matches_the_java_transcription() {
    let g = fixture();
    for (key, entries) in g["connections"].as_object().expect("connections") {
        let scroll = {
            let mut it = key.split(',');
            (
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
            )
        };
        let screen = AchievementsScreen::new(tree(&g));
        let mut d = DrawList::new();
        screen.draw_connections(&mut d, &view(scroll));
        let rects = quad_rects(&d);
        let colors = quad_colors(&d);

        let entries = entries.as_array().expect("entries");
        let mut cursor = 0;
        for e in entries {
            let name = e["key"].as_str().expect("key");
            let color = e["color"].as_u64().expect("color") as u32;
            let h = &e["horizontal"];
            assert_eq!(
                rects[cursor],
                [i32_at(h, 0), i32_at(h, 1), i32_at(h, 2), i32_at(h, 3)],
                "{key} / {name}: horizontal half of the link"
            );
            assert_eq!(colors[cursor], color, "{key} / {name}: link colour");
            cursor += 1;

            let v = &e["vertical"];
            assert_eq!(
                rects[cursor],
                [i32_at(v, 0), i32_at(v, 1), i32_at(v, 2), i32_at(v, 3)],
                "{key} / {name}: vertical half of the link"
            );
            assert_eq!(colors[cursor], color);
            cursor += 1;

            if let Some(a) = e.get("arrow") {
                let (x, y, w, h2) = (i32_at(a, 0), i32_at(a, 1), i32_at(a, 4), i32_at(a, 5));
                assert_eq!(
                    rects[cursor],
                    [x, y, x + w, y + h2],
                    "{key} / {name}: arrow head"
                );
                // The arrow sprite is untinted white, whatever the link's colour is.
                assert_eq!(colors[cursor], 0xFFFF_FFFF);
                let uv = d.vertices()[cursor * 6].uv;
                assert!(
                    (uv[0] - i32_at(a, 2) as f32 / 256.0).abs() < 1e-6
                        && (uv[1] - i32_at(a, 3) as f32 / 256.0).abs() < 1e-6,
                    "{key} / {name}: arrow sprite uv {uv:?}"
                );
                cursor += 1;
            }
        }
        assert_eq!(
            cursor,
            rects.len(),
            "{key}: extra geometry beyond the Java's"
        );
    }
}

#[test]
fn every_cell_plate_matches_the_java_transcription() {
    let g = fixture();
    for (key, entries) in g["cells"].as_object().expect("cells") {
        let scroll = {
            let mut it = key.split(',');
            (
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
            )
        };
        let screen = AchievementsScreen::new(tree(&g));
        let mut d = DrawList::new();
        // No atlas: the plates draw, the item icons do not, so quad *n* is cell *n*.
        screen.draw_cells(&mut d, None, &view(scroll));
        let rects = quad_rects(&d);
        let entries = entries.as_array().expect("entries");
        assert_eq!(rects.len(), entries.len(), "{key}: visible cell count");
        for (i, e) in entries.iter().enumerate() {
            let p = &e["plate"];
            let (x, y, u, v, w, h) = (
                i32_at(p, 0),
                i32_at(p, 1),
                i32_at(p, 2),
                i32_at(p, 3),
                i32_at(p, 4),
                i32_at(p, 5),
            );
            assert_eq!(rects[i], [x, y, x + w, y + h], "{key} cell {i}");
            let uv = d.vertices()[i * 6].uv;
            assert!(
                (uv[0] - u as f32 / 256.0).abs() < 1e-6 && (uv[1] - v as f32 / 256.0).abs() < 1e-6,
                "{key} cell {i}: plate uv {uv:?}"
            );
            let shade = e["shade"].as_f64().expect("shade") as f32;
            let c = unpack_argb(quad_colors(&d)[i]);
            assert!((c[0] - shade).abs() < 0.005, "{key} cell {i}: {c:?}");
        }
    }
}

#[test]
fn the_scroll_bounds_and_clamps_match_the_java() {
    let g = fixture();
    let want = (
        i32_at(&g["bounds"], 0),
        i32_at(&g["bounds"], 1),
        i32_at(&g["bounds"], 2),
        i32_at(&g["bounds"], 3),
    );
    let grid = tree(&g);
    assert_eq!(grid.scroll_bounds(), want);

    let mut screen = AchievementsScreen::new(tree(&g));
    for c in g["clamps"].as_array().expect("clamps") {
        let input = c["in"].as_f64().expect("in");
        screen.set_scroll(input, input);
        screen.clamp_target();
        // `set_scroll` leaves the live position unclamped; the target is what moves.
        let v = screen.map_view(1.0);
        assert_eq!(
            (f64::from(v.scroll.0), f64::from(v.scroll.1)),
            (c["x"].as_f64().expect("x"), c["y"].as_f64().expect("y")),
            "clamping {input}"
        );
    }
}

#[test]
fn the_drag_easing_curve_matches_the_java_tick_for_tick() {
    let g = fixture();
    // A tree wide enough that nothing here is clamped, so the easing is what is under test.
    let grid = AchievementGrid::new(vec![AchievementNode {
        key: "root".into(),
        column: 40,
        row: 40,
        parent: None,
        unlocked: true,
        special: false,
        icon: None,
    }]);
    let mut screen = AchievementsScreen::new(grid);
    screen.done_loading();
    screen.set_scroll(0.0, 0.0);
    screen.set_target(100.0, -40.0);
    for (i, step) in g["easing"].as_array().expect("easing").iter().enumerate() {
        screen.tick();
        let (x, y) = screen.scroll();
        assert!(
            (x - step[0].as_f64().expect("x")).abs() < 1e-6
                && (y - step[1].as_f64().expect("y")).abs() < 1e-6,
            "tick {i}: ({x}, {y}) vs {step}"
        );
    }
}

/// The map layer is drawn, then the frame over it, then the buttons, then the tooltip, and the
/// title last of all — `drawScreen` calls `drawTitle` *after* `drawAchievementScreen` returns.
#[test]
fn the_screen_draws_in_vanillas_order() {
    let g = fixture();
    let mut lang = mc_assets::Language::new("en_US");
    lang.insert("gui.achievements", "Achievements");
    lang.insert("gui.done", "Done");
    let mut screen = AchievementsScreen::new(tree(&g));
    screen.done_loading();
    let res = ScaledResolution::new(854, 480, 1, false);
    let ctx = ScreenCtx::new(res, 0.0);
    screen.init(&ctx, &lang);
    screen.set_scroll(0.0, 0.0);

    let mut font = mc_ui::font::FontRenderer::new(
        mc_ui::font::FontData::from_parts(&blank_ascii_png(), &[0u8; 0x1_0000]).expect("font"),
        false,
        false,
    );
    let mut d = DrawList::new();
    let atlas = UiTexture::gui("mc-ui/achievements-test-atlas");
    screen.draw(&mut d, &mut font, Some(&atlas), &ctx, (0, 0), &lang);

    let first_atlas = d
        .batches()
        .iter()
        .position(|b| b.texture == atlas)
        .expect("the backdrop samples the atlas");
    let first_frame = d
        .batches()
        .iter()
        .position(|b| {
            matches!(&b.texture, UiTexture::Resource(r)
            if r.path == "textures/gui/achievement/achievement_background.png")
        })
        .expect("the frame is drawn");
    assert!(
        first_atlas < first_frame,
        "the backdrop must go under the frame, not over it"
    );
    assert!(
        matches!(
            d.batches().last().map(|b| &b.texture),
            Some(UiTexture::Font(_))
        ),
        "the title is the very last thing drawn"
    );
}

/// A 128×128 sheet with a single inked column per cell, enough for `FontData` to load.
fn blank_ascii_png() -> Vec<u8> {
    let mut img = image::RgbaImage::new(128, 128);
    for cell in 0..256u32 {
        let cx = (cell % 16) * 8;
        let cy = (cell / 16) * 8;
        for col in 0..4 {
            img.put_pixel(cx + col, cy + 2, image::Rgba([255, 255, 255, 255]));
        }
    }
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .expect("encode");
    buf.into_inner()
}
