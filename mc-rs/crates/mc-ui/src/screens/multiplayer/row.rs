//! One row of the server list: layout, hit-testing, and drawing.
//!
//! java: `client/gui/ServerListEntryNormal.java`
//!
//! A row is 36 pixels of pitch and 32 of content: a 32×32 favicon on the left, the server name
//! and two wrapped MOTD lines beside it, and the population string plus a five-bar ping icon
//! flush right. When the row is hovered (or the touchscreen option is on) three 32×32 overlay
//! buttons appear on top of the favicon — join, move up, move down — and *those* take the click
//! instead.

use crate::font::FontRenderer;
use crate::screens::common::{SERVER_SELECTION_TEXTURE, UNKNOWN_SERVER_TEXTURE};
use crate::screens::multiplayer::server_data::{PingState, ServerEntry};
use crate::widget::draw::{DrawList, UiTexture, ICONS_TEXTURE};
use crate::widget::draw_centered_string;

/// Vanilla's list pitch: 32 pixels of row plus a 4-pixel gap.
// java: GuiMultiplayer.java:72 — `new ServerSelectionList(..., 36)`
pub const SLOT_HEIGHT: i32 = 36;

/// The favicon square, and the side of the three overlay buttons.
pub const ICON_SIZE: i32 = 32;

/// `ServerSelectionList.getListWidth()` = `GuiSlot`'s 220 + 85.
// java: ServerSelectionList.java:99
pub const LIST_WIDTH: i32 = 305;

/// `ServerSelectionList.getScrollBarX()` = `GuiSlot`'s `width / 2 + 124`, plus 30.
// java: ServerSelectionList.java:91
pub const SCROLLBAR_OFFSET: i32 = 124 + 30;

/// Draw vanilla's permanent, unclickable LAN-discovery row.
///
/// The detector may not have found anything yet, but `ServerSelectionList` always appends this
/// row after the saved internet servers while discovery remains active.
// java: ServerListEntryLanScan.java:11
pub fn draw_lan_scan_row(
    d: &mut DrawList,
    font: &mut FontRenderer,
    label: &str,
    screen_width: i32,
    y: i32,
    system_time_ms: u64,
) {
    // GuiSlot passes `slotHeight - 4` (32), not the 36-pixel row pitch, as drawEntry's
    // `slotHeight` argument.
    let text_y = y + ICON_SIZE / 2 - font.font_height() / 2;
    draw_centered_string(d, font, label, screen_width / 2, text_y, 0x00FF_FFFF);
    let spinner = match (system_time_ms / 300) % 4 {
        0 => "O o o",
        1 | 3 => "o O o",
        2 => "o o O",
        _ => unreachable!(),
    };
    draw_centered_string(
        d,
        font,
        spinner,
        screen_width / 2,
        text_y + font.font_height(),
        0x0080_8080,
    );
}

/// Draw one detected LAN server row.
///
/// Unlike a saved server row, vanilla draws no favicon, ping icon, population, or overlay
/// controls here. The blank 32-pixel icon column is still reserved, followed by the translated
/// LAN title, the announced MOTD, and either the address or the hidden-address label.
// java: ServerListEntryLanDetected.drawEntry
#[allow(clippy::too_many_arguments)]
pub fn draw_lan_server_row(
    d: &mut DrawList,
    font: &mut FontRenderer,
    title: &str,
    motd: &str,
    address: &str,
    hidden_address: &str,
    hide_address: bool,
    x: i32,
    y: i32,
) {
    let text_x = (x + ICON_SIZE + 3) as f32;
    font.draw_string(d, title, text_x, (y + 1) as f32, 0x00FF_FFFF, false);
    font.draw_string(d, motd, text_x, (y + 12) as f32, 0x0080_8080, false);
    font.draw_string(
        d,
        if hide_address {
            hidden_address
        } else {
            address
        },
        text_x,
        (y + 23) as f32,
        0x0030_3030,
        false,
    );
}

/// Which of a row's overlay buttons the cursor is over.
///
/// The regions are exactly vanilla's: the right half of the icon joins, the top-left quarter
/// moves up, and the bottom-left quarter moves down. They only exist while the row is selected.
// java: ServerListEntryNormal.java:293 (mousePressed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowHit {
    /// Nothing special — the click just selects (or, if it is a double click, joins).
    Body,
    /// The join arrow.
    Join,
    /// The move-up arrow.
    MoveUp,
    /// The move-down arrow.
    MoveDown,
}

/// Hit-test a click at `(rel_x, rel_y)` relative to the row's top-left corner.
///
/// `can_move_up`/`can_move_down` are vanilla's `func_175392_a`/`func_175394_b`: the first and
/// last rows have no arrow, so a click there falls through to the body.
// java: ServerListEntryNormal.java:293
#[must_use]
pub fn hit_test(rel_x: i32, rel_y: i32, can_move_up: bool, can_move_down: bool) -> RowHit {
    // java: `if (p_148278_5_ <= 32)` gates all three, and each arm then re-tests the halves.
    // Note the asymmetry vanilla ships: the join region is `x < 32 && x > 16`, so the single
    // column at x == 16 belongs to no button and falls through to the body.
    if rel_x <= ICON_SIZE {
        if rel_x < 32 && rel_x > 16 {
            return RowHit::Join;
        }
        if rel_x < 16 && rel_y < 16 && can_move_up {
            return RowHit::MoveUp;
        }
        if rel_x < 16 && rel_y > 16 && can_move_down {
            return RowHit::MoveDown;
        }
    }
    RowHit::Body
}

/// The label vanilla shows when the cursor rests on the ping icon.
// java: ServerListEntryNormal.java:97
#[must_use]
pub fn ping_tooltip(entry: &ServerEntry) -> String {
    if entry.version_mismatch() {
        return if entry.protocol > crate::screens::multiplayer::CLIENT_PROTOCOL {
            "Client out of date!".to_owned()
        } else {
            "Server out of date!".to_owned()
        };
    }
    match entry.state {
        PingState::Unqueried | PingState::Pinging => "Pinging...".to_owned(),
        // The literal `Pinging...` is the MOTD in this phase; Java still exposes the
        // `pingToServer == -1` tooltip beside its red-X icon until S00/S01 completes.
        PingState::AwaitingStatus | PingState::Failed => "(no connection)".to_owned(),
        PingState::Ok if entry.ping_ms < 0 => "(no connection)".to_owned(),
        PingState::Ok => format!("{}ms", entry.ping_ms),
    }
}

/// Which sprite column and row of `icons.png` the ping bars come from.
///
/// Returns `(u, v)` in texels. The animated "pinging" state walks a 0→4→0 triangle keyed on
/// wall-clock time and the row index, so adjacent rows are out of phase — vanilla's little
/// barber-pole effect.
// java: ServerListEntryNormal.java:142
#[must_use]
pub fn ping_icon_uv(entry: &ServerEntry, row: usize, system_time_ms: u64) -> (f32, f32) {
    let (col, mut bar) = match entry.state {
        _ if entry.version_mismatch() => (0, 5),
        PingState::Unqueried | PingState::Pinging => {
            let l = (system_time_ms / 100 + (row as u64 * 2)) & 7;
            (1, l as i32)
        }
        _ => (0, entry.ping_bars()),
    };
    if col == 1 && bar > 4 {
        bar = 8 - bar;
    }
    ((col * 10) as f32, (176 + bar * 8) as f32)
}

/// Draw one row.
///
/// `x`/`y` are the row's top-left corner and `width` its width (vanilla's `listWidth`), which
/// together with the mouse position is everything the row needs. Returns the hover tooltip the
/// screen should show, if any — vanilla stores it on the parent screen through
/// `setHoveringText`.
#[allow(clippy::too_many_arguments)]
pub fn draw_row(
    d: &mut DrawList,
    font: &mut FontRenderer,
    entry: &ServerEntry,
    row: usize,
    x: i32,
    y: i32,
    width: i32,
    mouse: (i32, i32),
    show_overlay: bool,
    can_move_up: bool,
    can_move_down: bool,
    system_time_ms: u64,
    have_icon: bool,
) -> Option<String> {
    font.draw_string(
        d,
        &entry.name,
        (x + 32 + 3) as f32,
        (y + 1) as f32,
        0x00FF_FFFF,
        false,
    );
    // java: listFormattedStringToWidth(motd, listWidth - 32 - 2), first two lines only.
    for (i, line) in font
        .wrap_to_lines(&entry.motd, width - 32 - 2)
        .into_iter()
        .take(2)
        .enumerate()
    {
        font.draw_string(
            d,
            &line,
            (x + 32 + 3) as f32,
            (y + 12 + font.font_height() * i as i32) as f32,
            0x0080_8080,
            false,
        );
    }

    let right_label = if entry.version_mismatch() {
        format!("\u{a7}4{}", entry.game_version)
    } else {
        entry.population.clone()
    };
    let label_width = font.string_width(&right_label);
    font.draw_string(
        d,
        &right_label,
        (x + width - label_width - 15 - 2) as f32,
        (y + 1) as f32,
        0x0080_8080,
        false,
    );

    let (u, v) = ping_icon_uv(entry, row, system_time_ms);
    d.blit_sized(
        &UiTexture::gui(ICONS_TEXTURE),
        x + width - 15,
        y,
        u,
        v,
        10,
        8,
        256.0,
        256.0,
    );

    // The favicon is a 32×32 texture drawn at 1:1, so its own size is the texture size — not
    // 256. Getting this wrong is the classic "server icon is a quarter of a pixel" bug.
    let icon = if have_icon {
        UiTexture::Resource(favicon_location(&entry.address))
    } else {
        UiTexture::gui(UNKNOWN_SERVER_TEXTURE)
    };
    d.blit_sized(&icon, x, y, 0.0, 0.0, ICON_SIZE, ICON_SIZE, 32.0, 32.0);

    let (mx, my) = (mouse.0 - x, mouse.1 - y);
    let tooltip = if mx >= width - 15 && mx <= width - 5 && (0..=8).contains(&my) {
        Some(ping_tooltip(entry))
    } else if mx >= width - label_width - 15 - 2
        && mx <= width - 15 - 2
        && (0..=8).contains(&my)
        && !entry.player_list.is_empty()
    {
        Some(entry.player_list.clone())
    } else {
        None
    };

    if show_overlay {
        // java: drawRect(x, y, x + 32, y + 32, -1601138544)
        d.rect(x, y, x + 32, y + 32, 0xA090_9090);
        let sel = UiTexture::gui(SERVER_SELECTION_TEXTURE);
        let hot = |cond: bool| if cond { 32.0 } else { 0.0 };
        d.blit_sized(
            &sel,
            x,
            y,
            0.0,
            hot(mx < 32 && mx > 16),
            ICON_SIZE,
            ICON_SIZE,
            256.0,
            256.0,
        );
        if can_move_up {
            d.blit_sized(
                &sel,
                x,
                y,
                96.0,
                hot(mx < 16 && my < 16),
                ICON_SIZE,
                ICON_SIZE,
                256.0,
                256.0,
            );
        }
        if can_move_down {
            d.blit_sized(
                &sel,
                x,
                y,
                64.0,
                hot(mx < 16 && my > 16),
                ICON_SIZE,
                ICON_SIZE,
                256.0,
                256.0,
            );
        }
    }

    tooltip
}

/// The synthetic resource name a decoded favicon is uploaded under.
// java: ServerListEntryNormal.java:44 — `new ResourceLocation("servers/" + ip + "/icon")`
#[must_use]
pub fn favicon_location(address: &str) -> mc_assets::ResourceLocation {
    mc_assets::ResourceLocation::minecraft(format!("servers/{address}/icon"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;

    #[test]
    fn the_right_half_of_the_icon_joins_and_the_left_quarters_reorder() {
        assert_eq!(hit_test(20, 5, true, true), RowHit::Join);
        assert_eq!(hit_test(31, 31, true, true), RowHit::Join);
        assert_eq!(hit_test(5, 5, true, true), RowHit::MoveUp);
        assert_eq!(hit_test(5, 25, true, true), RowHit::MoveDown);
        assert_eq!(hit_test(40, 5, true, true), RowHit::Body, "past the icon");
    }

    /// Two of vanilla's own edge cases, reproduced rather than tidied: the single column at
    /// `x == 16` belongs to no button, and the single row at `y == 16` belongs to neither
    /// arrow (`< 16` and `> 16` leave the middle out).
    #[test]
    fn vanillas_off_by_one_dead_columns_are_reproduced() {
        assert_eq!(hit_test(16, 5, true, true), RowHit::Body, "x == 16 is dead");
        assert_eq!(hit_test(32, 5, true, true), RowHit::Body, "x == 32 is dead");
        assert_eq!(hit_test(5, 16, true, true), RowHit::Body, "y == 16 is dead");
    }

    #[test]
    fn the_first_and_last_rows_have_no_reorder_arrow() {
        assert_eq!(hit_test(5, 5, false, true), RowHit::Body);
        assert_eq!(hit_test(5, 25, true, false), RowHit::Body);
    }

    #[test]
    fn the_ping_tooltip_names_the_state() {
        let mut e = ServerEntry::new("n", "a");
        assert_eq!(ping_tooltip(&e), "Pinging...");
        e.state = PingState::Failed;
        e.ping_ms = -1;
        assert_eq!(ping_tooltip(&e), "(no connection)");
        e.state = PingState::AwaitingStatus;
        assert_eq!(ping_tooltip(&e), "(no connection)");
        e.state = PingState::Ok;
        e.ping_ms = 42;
        assert_eq!(ping_tooltip(&e), "42ms");
        e.protocol = 5;
        assert_eq!(ping_tooltip(&e), "Server out of date!");
        e.protocol = 500;
        assert_eq!(ping_tooltip(&e), "Client out of date!");
    }

    #[test]
    fn the_ping_icon_walks_the_second_column_while_pinging() {
        let mut e = ServerEntry::new("n", "a");
        // Column 1 (u = 10) is the animated set; the bar index folds back at 4.
        let (u, _) = ping_icon_uv(&e, 0, 0);
        assert_eq!(u, 10.0);
        let bars: Vec<f32> = (0..8).map(|t| ping_icon_uv(&e, 0, t * 100).1).collect();
        assert_eq!(
            bars,
            [176.0, 184.0, 192.0, 200.0, 208.0, 200.0, 192.0, 184.0],
            "0 1 2 3 4 3 2 1 — the triangle, not a sawtooth"
        );
        // Adjacent rows are two steps out of phase: row 1 starts at bar 2.
        assert_eq!(ping_icon_uv(&e, 1, 0).1, 192.0);

        e.state = PingState::Ok;
        e.ping_ms = 700;
        assert_eq!(ping_icon_uv(&e, 0, 0), (0.0, (176 + 3 * 8) as f32));

        // `OldServerPinger` switches from the animated -2 sentinel to -1 after connecting,
        // before it receives S00. Its sprite is therefore the static red-X bar (row 5).
        e.state = PingState::AwaitingStatus;
        e.ping_ms = -1;
        assert_eq!(ping_icon_uv(&e, 0, 999), (0.0, 216.0));
    }

    #[test]
    fn a_version_mismatch_freezes_the_icon_on_the_no_connection_bar() {
        let mut e = ServerEntry::new("n", "a");
        e.protocol = 5;
        assert_eq!(ping_icon_uv(&e, 0, 12345), (0.0, (176 + 5 * 8) as f32));
    }

    #[test]
    fn the_row_puts_the_name_and_ping_icon_where_vanilla_does() {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let mut e = ServerEntry::new("Server", "host");
        e.state = PingState::Ok;
        e.ping_ms = 10;
        e.population = "1/20".into();
        draw_row(
            &mut d,
            &mut font,
            &e,
            0,
            100,
            50,
            LIST_WIDTH,
            (0, 0),
            false,
            false,
            false,
            0,
            false,
        );
        // The name starts 35 pixels in and one pixel down.
        let name = d
            .vertices()
            .iter()
            .find(|v| v.pos == [135.0, 51.0])
            .is_some();
        assert!(name, "server name at x + 35, y + 1");
        // The ping icon is 10 wide at listWidth - 15.
        let icon_x = 100 + LIST_WIDTH - 15;
        assert!(d.vertices().iter().any(|v| v.pos == [icon_x as f32, 50.0]));
        assert!(d
            .vertices()
            .iter()
            .any(|v| v.pos == [(icon_x + 10) as f32, 58.0]));
    }

    #[test]
    fn only_two_motd_lines_are_drawn_however_long_the_motd_is() {
        let mut font = uniform_font();
        let long = "aaaa ".repeat(60);
        let all = font.wrap_to_lines(&long, LIST_WIDTH - 34);
        assert!(
            all.len() > 2,
            "the fixture must actually wrap: {}",
            all.len()
        );

        let mut d = DrawList::new();
        let mut e = ServerEntry::new("n", "a");
        e.motd = long;
        draw_row(
            &mut d,
            &mut font,
            &e,
            0,
            0,
            0,
            LIST_WIDTH,
            (0, 0),
            false,
            false,
            false,
            0,
            false,
        );
        // MOTD lines start at y + 12 and step by the font height; nothing may reach line 3.
        let third = (12 + font.font_height() * 2) as f32;
        assert!(
            !d.vertices().iter().any(|v| v.pos[1] == third),
            "a third MOTD line was drawn"
        );
    }

    #[test]
    fn hovering_the_ping_icon_yields_its_tooltip_and_elsewhere_yields_none() {
        let mut font = uniform_font();
        let mut e = ServerEntry::new("n", "a");
        e.state = PingState::Ok;
        e.ping_ms = 7;
        let mut d = DrawList::new();
        let t = draw_row(
            &mut d,
            &mut font,
            &e,
            0,
            0,
            0,
            LIST_WIDTH,
            (LIST_WIDTH - 10, 4),
            false,
            false,
            false,
            0,
            false,
        );
        assert_eq!(t.as_deref(), Some("7ms"));

        let mut d2 = DrawList::new();
        let t2 = draw_row(
            &mut d2,
            &mut font,
            &e,
            0,
            0,
            0,
            LIST_WIDTH,
            (50, 20),
            false,
            false,
            false,
            0,
            false,
        );
        assert_eq!(t2, None);
    }

    #[test]
    fn a_row_asked_to_show_overlays_draws_the_buttons_it_is_allowed() {
        let mut font = uniform_font();
        let e = ServerEntry::new("n", "a");
        let mut count = |up: bool, down: bool| {
            let mut d = DrawList::new();
            draw_row(
                &mut d,
                &mut font,
                &e,
                0,
                0,
                0,
                LIST_WIDTH,
                (0, 0),
                true,
                up,
                down,
                0,
                false,
            );
            d.batches()
                .iter()
                .filter(|b| {
                    matches!(&b.texture, UiTexture::Resource(r)
                        if r.path == SERVER_SELECTION_TEXTURE)
                })
                .map(|b| b.range.len() / 6)
                .sum::<usize>()
        };
        assert_eq!(count(false, false), 1, "join only");
        assert_eq!(count(true, false), 2);
        assert_eq!(count(true, true), 3);
    }

    #[test]
    fn the_favicon_resource_name_is_keyed_on_the_address() {
        assert_eq!(
            favicon_location("mc.example:25566").path,
            "servers/mc.example:25566/icon"
        );
    }

    #[test]
    fn the_lan_scan_spinner_repeats_vanillas_four_frame_cycle() {
        let render = |now_ms| {
            let mut font = uniform_font();
            let mut d = DrawList::new();
            draw_lan_scan_row(&mut d, &mut font, "Scanning", 400, 36, now_ms);
            d
        };
        assert_eq!(render(0), render(1_200), "frame 0 repeats after four ticks");
        assert_eq!(render(300), render(900), "frames 1 and 3 share o O o");
        assert_ne!(render(0), render(300));
        assert_ne!(render(300), render(600));
        let frame = render(0);
        let min_y = frame
            .vertices()
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::INFINITY, f32::min);
        assert_eq!(min_y, 48.0, "y + 32/2 - FONT_HEIGHT/2 for a 9px font");
    }

    #[test]
    fn selected_icon_plate_uses_vanillas_exact_argb() {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        draw_row(
            &mut d,
            &mut font,
            &ServerEntry::new("n", "a"),
            0,
            10,
            20,
            LIST_WIDTH,
            (0, 0),
            true,
            false,
            false,
            0,
            false,
        );
        let expected = crate::widget::draw::unpack_argb(0xA090_9090);
        assert!(d.vertices().iter().any(|v| v.color == expected));
        let old = crate::widget::draw::unpack_argb(0xA08F_8F90);
        assert!(!d.vertices().iter().any(|v| v.color == old));
    }
}
