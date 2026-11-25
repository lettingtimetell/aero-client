//! 1.7.10's player-list grid against java17 `GuiIngame.renderGameOverlay:388-483`.

use std::cell::Cell;

use super::*;
use crate::screens::tab_list::TabListStyle;
use crate::widget::test_support::uniform_font;

fn cell(name: &str, ping_ms: i32) -> TabCellV5 {
    TabCellV5 {
        name: name.to_owned(),
        score: None,
        ping_ms,
    }
}

/// The top-left corner of every `icons.png` quad, in draw order.
fn icon_corners(d: &DrawList) -> Vec<(i32, i32)> {
    let icons = UiTexture::icons();
    let mut corners = Vec::new();
    for batch in d.batches().iter().filter(|b| &b.texture == icons) {
        for quad in d.vertices()[batch.range.clone()].chunks(6) {
            let x = quad.iter().map(|v| v.pos[0]).fold(f32::MAX, f32::min);
            let y = quad.iter().map(|v| v.pos[1]).fold(f32::MAX, f32::min);
            corners.push((x as i32, y as i32));
        }
    }
    corners
}

#[test]
fn the_grid_grows_columns_until_none_holds_more_than_twenty() {
    let grid = |max| {
        let g = TabGridV5::for_max_players(max);
        (g.columns, g.rows, g.column_width)
    };
    assert_eq!(grid(20), (1, 20, 150));
    assert_eq!(grid(21), (2, 11, 150));
    assert_eq!(grid(60), (3, 20, 100));
    assert_eq!(grid(100), (5, 20, 60));
    assert_eq!(grid(255), (13, 20, 23));
    assert_eq!(grid(0), (1, 0, 150));
}

fn overlay_with(cells: Vec<TabCellV5>) -> TabListOverlay {
    let mut overlay = TabListOverlay::default();
    overlay.refresh_v5((1, 1, 0), || cells);
    overlay
}

#[test]
fn players_fill_the_cells_row_by_row_in_arrival_order() {
    let mut font = uniform_font();
    let overlay = overlay_with((0..4).map(|i| cell(&format!("p{i}"), 10)).collect());
    let mut d = DrawList::new();
    // 60 slots: 3 columns of 100, centred in 854: left = (854 - 300) / 2 = 277.
    overlay.draw_v5(&mut d, &mut font, 854, 60);
    assert_eq!(
        icon_corners(&d),
        [
            (277 + 88, 10),
            (377 + 88, 10),
            (477 + 88, 10),
            (277 + 88, 19)
        ]
    );
}

#[test]
fn players_past_the_maximum_are_not_drawn() {
    let mut font = uniform_font();
    let overlay = overlay_with((0..5).map(|i| cell(&format!("p{i}"), 10)).collect());
    let mut d = DrawList::new();
    overlay.draw_v5(&mut d, &mut font, 854, 3);
    assert_eq!(icon_corners(&d).len(), 3);
}

#[test]
fn the_latency_follows_the_tab_list_mods_style() {
    let mut font = uniform_font();
    for (ping, icons) in [
        (TabPing::Bars, 2),
        (TabPing::Numbers, 0),
        (TabPing::Hidden, 0),
    ] {
        let mut overlay = overlay_with(vec![cell("a", 50), cell("b", -1)]);
        overlay.style = TabListStyle {
            ping,
            ..TabListStyle::VANILLA
        };
        let mut d = DrawList::new();
        overlay.draw_v5(&mut d, &mut font, 854, 20);
        assert_eq!(icon_corners(&d).len(), icons, "{ping:?}");
    }
}

#[test]
fn a_score_needs_more_than_five_pixels_of_room() {
    let grid = TabGridV5::for_max_players(20);
    // (150 - 17) - (w + 5) > 5  <=>  w < 123.
    assert!(grid.has_score_room(122));
    assert!(!grid.has_score_room(123));
}

#[test]
fn cells_are_rebuilt_only_when_their_inputs_change() {
    let mut overlay = TabListOverlay::default();
    let builds = Cell::new(0);
    let build = || {
        builds.set(builds.get() + 1);
        vec![cell("a", 1)]
    };
    overlay.refresh_v5((1, 1, 20), build);
    assert_eq!(overlay.cells_v5().len(), 1);
    overlay.refresh_v5((1, 1, 20), build);
    assert_eq!(builds.get(), 1);
    overlay.refresh_v5((2, 1, 20), build);
    overlay.refresh_v5((2, 1, 21), build);
    assert_eq!(builds.get(), 3);
    overlay.clear_v5();
    assert!(overlay.cells_v5().is_empty());
    overlay.refresh_v5((2, 1, 21), build);
    assert_eq!(builds.get(), 4);
}
